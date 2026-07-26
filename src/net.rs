//! Peer-to-peer transport over iroh (QUIC with NAT traversal).
//!
//! One player clicks **Host** and gets a one-string *invite code*; the other
//! pastes it and clicks **Join**. iroh rendezvouses the two through a relay
//! server, attempts UDP hole punching in the background, and hands us a QUIC
//! connection — direct when punching succeeds, relayed when it can't (VPNs,
//! CGNAT). Neither side configures ports or IPs, and the connection is
//! authenticated by the host's key, so strangers can't inject notes.
//!
//! The invite code is normally the host's bare [`EndpointId`] — 64 hex chars —
//! and the joiner resolves the actual dial info (relay + addresses) through
//! n0's discovery service, which the `N0` preset both publishes to and reads
//! from. Both forms are always useful, not "short with a fallback": the full
//! [`EndpointTicket`] (~4× longer) is available the instant the endpoint
//! binds and is *strictly more dialable* (relay + direct addresses inline) —
//! it's just ugly — while the short code is a UX nicety that needs discovery
//! to have caught up. So the host publishes both, promotes the short one to
//! primary once it's *verified* dialable (a self-resolve through the same
//! discovery path a joiner would use), and keeps re-publishing on a heartbeat
//! for as long as hosting continues — an idempotent whole-state snapshot, not
//! a one-shot decision, per the wire-protocol convention in CLAUDE.md. A host
//! that only reaches a relay at t = 30s therefore upgrades its own invite
//! code in place instead of being stuck with the long one forever. Joining
//! accepts either form, which also keeps codes from older versions working.
//!
//! Latency model is unchanged from the old raw-UDP transport: every [`Packet`]
//! rides an *unreliable QUIC datagram*, sent immediately, no batching, no
//! retransmit — for a live duet a dropped note-on is corrected by the next
//! event within milliseconds, whereas reliable-stream head-of-line blocking
//! would add lag. The wire format (`note.rs`) is byte-identical to before.
//!
//! Threading: iroh needs an async runtime, so each session spawns one "net"
//! thread running a current-thread tokio runtime. The GUI stays sync and talks
//! to it over two channels: an unbounded sender for outgoing packets (so
//! `send` never blocks the frame) and an mpsc receiver of [`NetEvent`]s
//! (status, invite code, connect/disconnect, incoming packets) that the UI
//! drains once per frame. Dropping the [`Peer`] closes both channels, which
//! the net thread notices and shuts down on.

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use bytes::Bytes;
use iroh::endpoint::presets;
use iroh::endpoint::Connection;
use iroh::{Endpoint, EndpointAddr, EndpointId};
use iroh_tickets::endpoint::EndpointTicket;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::diag::{self, Area};
use crate::note::Packet;

/// Application-level protocol id required to match on both ends of a
/// connection; bump the suffix if the wire format ever changes incompatibly.
const ALPN: &[u8] = b"open-piano/0";

/// How long the host waits for first relay contact before showing a
/// provisional (long-ticket) invite code instead of a bare spinner. No longer
/// a hard deadline: once online, the code upgrades in place (see
/// `publish_invite_code`). A healthy connection handshakes in well under a
/// second, so this state is normally never seen.
const LAN_FALLBACK_DELAY: Duration = Duration::from_secs(5);
/// Grace period after `online()` before promoting the short code when the
/// host's own self-resolve (`confirm_published`) is inconclusive rather than
/// a confirmed failure.
const PUBLISH_SETTLE: Duration = Duration::from_millis(1500);
/// Bound on how long the host's self-resolve check is allowed to take.
const PUBLISH_CONFIRM_TIMEOUT: Duration = Duration::from_secs(8);
/// How often the host re-publishes its invite code once settled — a
/// heartbeat, not a one-shot, per CLAUDE.md's wire-protocol convention: it
/// self-heals a dropped first send, and (just as importantly) is what keeps
/// `publish_invite_code` alive so it can never "win" `run_host`'s `select!`
/// on its own — only `accept_loop` noticing real shutdown ends the session.
const TICKET_HEARTBEAT: Duration = Duration::from_secs(5);

/// Joiner reconnect/retry schedule: pause *before* each dial, so the first
/// attempt is immediate. The dense head (0, 1, 2 s) covers a discovery record
/// that hasn't propagated yet; the longer tail covers a host that's slow to
/// start — and, once past the initial connect, a host that's mid-restart or
/// a Wi-Fi blip. `run_join` cycles this schedule *forever* (capping at the
/// last, 30 s, entry) rather than giving up after one pass — a cap strands a
/// joiner whose host took a coffee break, and its failure mode ("nothing
/// happens, press Join again") is exactly the bug this loop exists to remove.
/// Cancellable at every step: dropping the UI's `Peer` (a fresh Join click,
/// or closing the app) closes `outgoing`, which every `select!` below races.
const JOIN_BACKOFF: [Duration; 7] = [
    Duration::from_secs(0),
    Duration::from_secs(1),
    Duration::from_secs(2),
    Duration::from_secs(4),
    Duration::from_secs(8),
    Duration::from_secs(15),
    Duration::from_secs(30),
];
/// A session up at least this long counts as "real": the schedule resets to
/// its dense head afterwards, so a peer who plays for an hour then drops gets
/// an immediate 1 s retry rather than inheriting a 30 s pause from whatever
/// the schedule was mid-way through before this session started.
const RECONNECT_RESET_AFTER: Duration = Duration::from_secs(20);

/// Everything the net thread reports back to the UI, drained once per frame.
#[derive(Debug, Clone)]
pub enum NetEvent {
    /// Host only: the current invite code(s), re-sent whenever they improve
    /// and periodically thereafter (see `TICKET_HEARTBEAT`) — a whole-state
    /// snapshot, not a one-shot. `full` is available the moment we bind;
    /// `short` arrives once we're online AND a self-resolve confirms
    /// discovery can actually serve us.
    Ticket { short: Option<String>, full: String },
    /// Human-readable progress / error line for the status bar.
    Status(String),
    /// A peer connection is live. The UI clears remote key state (unknown
    /// across a reconnect) and re-announces its color.
    Connected,
    /// The peer connection dropped. A host keeps listening for a rejoin; a
    /// joiner keeps redialing on its own via `run_join`'s backoff loop —
    /// pressing Join again is a hard reset (drops and rebuilds the session),
    /// not the only way to reconnect.
    Disconnected,
    /// A decoded packet from the peer (note event, color, or metronome control).
    Packet(Packet),
    /// A metronome beat marker from the host, split out from `Packet` so we can
    /// stamp `one_way` — half the current QUIC RTT, measured on the net thread
    /// where it's freshest — which the follower uses to phase-align its local
    /// click schedule with the host's (see `main.rs`).
    MetroBeat {
        bpm: u16,
        beat_in_bar: u8,
        beats_per_bar: u8,
        on: bool,
        one_way: Duration,
    },
}

/// A live networking session (hosting or joining). Dropping it disconnects
/// and shuts the net thread down.
pub struct Peer {
    outgoing: UnboundedSender<Packet>,
    pub events: Receiver<NetEvent>,
}

impl Peer {
    /// Queue a packet for the remote peer. Never blocks; while no connection
    /// is up the net thread discards traffic (all packets are refreshed by
    /// later events, so nothing needs replaying on connect).
    pub fn send(&self, packet: Packet) {
        let _ = self.outgoing.send(packet);
    }
}

/// Start hosting: binds an endpoint, then emits `Ticket` with the invite code
/// and waits for a peer. Keeps accepting across peer disconnects. `secret`
/// pins the endpoint's identity (so its `EndpointId` — and therefore the
/// invite code — survives a restart); `None` binds an ephemeral one-off
/// identity instead (used by the test at the bottom of this file).
pub fn host(secret: Option<[u8; 32]>) -> Peer {
    start(Role::Host, secret)
}

/// Join a host from a pasted invite code (parsed and validated on the net
/// thread; a bad code comes back as a `Status` event). See [`host`] for
/// `secret`.
pub fn join(ticket: String, secret: Option<[u8; 32]>) -> Peer {
    start(Role::Join(ticket), secret)
}

/// A fresh, random identity for [`host`]/[`join`]'s `secret` — call once and
/// persist the result (`prefs::Prefs::set_endpoint_secret_bytes`) rather than
/// generating on every launch, or the `EndpointId` (and the invite code built
/// from it) changes every time. Deliberately `iroh::SecretKey::generate()`,
/// not a hand-rolled RNG: it's zero extra dependencies (already transitive
/// via iroh) and its trait-version compatibility with the rest of iroh's
/// `rand`/`rand_core` graph is iroh's problem to keep solved, not ours.
pub fn generate_secret() -> [u8; 32] {
    iroh::SecretKey::generate().to_bytes()
}

enum Role {
    Host,
    Join(String),
}

fn start(role: Role, secret: Option<[u8; 32]>) -> Peer {
    let (event_tx, event_rx) = mpsc::channel::<NetEvent>();
    let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<Packet>();

    // Cloned before the closure captures `event_tx`: a failed `spawn()` drops
    // the closure (and everything it captured) without ever running it, so
    // this is the only sender left to report through (d3) — the GUI thread
    // must not `.expect()` its way into a crash over a transient OS resource
    // failure the same way the tokio-runtime-build failure two lines below
    // is already handled as a `Status` event, not a panic.
    let spawn_failed_tx = event_tx.clone();
    let spawned = thread::Builder::new().name("net".into()).spawn(move || {
        // First statement in the closure: reports a panic anywhere below
        // instead of the net thread just vanishing with the UI never hearing
        // from it again (d2). `Drop` runs during unwind.
        let _guard = NetThreadGuard { events: event_tx.clone() };
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                let _ = event_tx.send(NetEvent::Status(format!("Net init failed: {e}")));
                return;
            }
        };
        rt.block_on(run(role, secret, out_rx, event_tx));
    });

    if let Err(e) = spawned {
        diag::log(Area::Net, format!("failed to spawn net thread: {e}"));
        let _ = spawn_failed_tx.send(NetEvent::Status(format!("Failed to start networking: {e}")));
        // No thread will ever emit `Disconnected` for this session — say so
        // ourselves so `pump_network`'s d1 teardown still runs and the UI
        // doesn't hold a `Peer` that will never do anything (d4's invariant).
        let _ = spawn_failed_tx.send(NetEvent::Disconnected);
    }

    Peer {
        outgoing: out_tx,
        events: event_rx,
    }
}

/// Reports a panicking net thread instead of it vanishing silently — without
/// this, a panic mid-session looks identical to the thread just going quiet:
/// no `Disconnected`, `peer_connected` stays `true`, and every remote key and
/// peer synth voice would be stuck lit/sounding forever (the pre-0.2.0 bug).
/// Held as the closure's first local so `Drop` (which runs during unwind)
/// fires before the closure's own `Sender` capture is dropped, ensuring
/// `pump_network`'s d1 teardown sees these messages ahead of the channel
/// closing.
struct NetThreadGuard {
    events: Sender<NetEvent>,
}

impl Drop for NetThreadGuard {
    fn drop(&mut self) {
        if thread::panicking() {
            diag::log(Area::Net, "net thread panicked");
            let _ = self.events.send(NetEvent::Disconnected);
            let _ = self
                .events
                .send(NetEvent::Status("Networking crashed unexpectedly".into()));
        }
    }
}

/// Net-thread main. Any `Err` is a fatal setup problem already reported as a
/// `Status` event; connection-level errors are handled inside and don't end
/// the session (the host goes back to listening).
async fn run(
    role: Role,
    secret: Option<[u8; 32]>,
    mut outgoing: UnboundedReceiver<Packet>,
    events: Sender<NetEvent>,
) {
    let status = |s: String| {
        let _ = events.send(NetEvent::Status(s));
    };

    // Parse the invite code first (join only) so a typo fails fast, before
    // any network work. Short form (bare endpoint id) is tried first; the
    // long form (full ticket: LAN-only hosts, older versions) second. A
    // joiner who pasted a full ticket is already immune to every discovery
    // issue below — `addr` carries relay + direct addresses inline, so
    // `connect()` never touches discovery at all. `short_code` is threaded
    // through to `run_join` purely so its terminal failure message can
    // suggest the "same-network code" only when that's actually a next step.
    let (target, short_code): (Option<EndpointAddr>, bool) = match &role {
        Role::Host => (None, false),
        Role::Join(code) => match code.parse::<EndpointId>() {
            Ok(id) => (Some(id.into()), true),
            Err(_) => match code.parse::<EndpointTicket>() {
                Ok(t) => (Some(t.endpoint_addr().clone()), false),
                Err(e) => {
                    status(format!("Invalid invite code: {e}"));
                    return;
                }
            },
        },
    };

    status("Setting up p2p endpoint…".into());
    // The N0 preset = n0's public relay servers + endpoint discovery. This is
    // what makes the whole thing zero-config across NATs. Pinning `secret_key`
    // (when the caller has a persisted identity) is what keeps the resulting
    // `EndpointId` — and therefore the invite code handed out for it — stable
    // across restarts, crashes, and auto-updates; omitting it (the test at
    // the bottom of this file does) binds a fresh one-off identity instead.
    let mut builder = Endpoint::builder(presets::N0).alpns(vec![ALPN.to_vec()]);
    if let Some(bytes) = secret {
        builder = builder.secret_key(iroh::SecretKey::from_bytes(&bytes));
    }
    let endpoint = match builder.bind().await {
        Ok(ep) => ep,
        Err(e) => {
            diag::log(Area::Net, format!("endpoint bind failed: {e}"));
            status(format!("Failed to start networking: {e}"));
            return;
        }
    };
    diag::log(Area::Net, format!("endpoint bound: {}", endpoint.id().fmt_short()));

    match target {
        None => run_host(&endpoint, &mut outgoing, &events).await,
        Some(addr) => run_join(&endpoint, addr, &mut outgoing, &events, short_code).await,
    }

    // Graceful close tells the peer immediately instead of leaving it to the
    // QUIC idle timeout.
    diag::log(Area::Net, "closing endpoint");
    endpoint.close().await;
}

/// Run the invite-code publisher and the peer-accept loop concurrently, over
/// borrowed references (no `tokio::spawn`, no `'static` bounds — matches the
/// file's existing idiom). Whichever finishes first ends the session: in
/// practice that's always `accept_loop` (UI dropped the `Peer` / endpoint
/// closed) — `publish_invite_code` is built to never finish on its own (see
/// its doc comment), so this is really "run `accept_loop`, with the
/// publisher tagging along and getting dropped when it does."
async fn run_host(
    endpoint: &Endpoint,
    outgoing: &mut UnboundedReceiver<Packet>,
    events: &Sender<NetEvent>,
) {
    tokio::select! {
        _ = publish_invite_code(endpoint, events) => {}
        _ = accept_loop(endpoint, outgoing, events) => {}
    }
}

/// Publish the invite code, upgrading it in place as reachability improves,
/// then keep re-emitting it on `TICKET_HEARTBEAT` for as long as hosting
/// continues. That heartbeat is doing double duty: it self-heals a dropped
/// send (the wire-protocol convention in CLAUDE.md), and — just as
/// importantly — it's an unbounded loop with no normal exit, which is what
/// lets `run_host`'s `select!` above be a plain two-armed race: this task
/// structurally cannot "win" it under normal operation, only get dropped when
/// `accept_loop` notices the real shutdown signal.
async fn publish_invite_code(endpoint: &Endpoint, events: &Sender<NetEvent>) {
    let status = |s: String| {
        let _ = events.send(NetEvent::Status(s));
    };
    status("Contacting relay…".into());

    // Not a hard deadline (see `LAN_FALLBACK_DELAY`'s doc) — just when we
    // stop showing a spinner and publish a *provisional* code so same-network
    // play keeps working fully offline. A healthy connection handshakes in
    // well under a second, so this branch is normally never taken.
    let online = tokio::time::timeout(LAN_FALLBACK_DELAY, endpoint.online()).await.is_ok();
    if !online {
        diag::log(Area::Net, "no relay yet; publishing a provisional full ticket");
        let full = EndpointTicket::from(endpoint.addr()).to_string();
        if events.send(NetEvent::Ticket { short: None, full }).is_err() {
            return; // UI dropped the session.
        }
        status("No relay yet — the code below only works on this network. Still trying…".into());
        // Unbounded: it pends forever with no WAN, which is fine — see this
        // function's own doc comment for why that's not a leak.
        endpoint.online().await;
    }

    diag::log(Area::Net, "relay online; confirming discovery can serve us");
    if confirm_published(endpoint).await {
        diag::log(Area::Net, "self-resolve confirmed; promoting the short invite code");
    } else {
        // Our own DNS resolution failing says nothing about the joiner's —
        // this never blocks promotion, it just falls through to a settle
        // timer instead of an immediate promotion.
        tokio::time::sleep(PUBLISH_SETTLE).await;
    }

    let short = endpoint.id().to_string();
    loop {
        // Recomputed every heartbeat, not hoisted out of the loop: once
        // online it carries the relay URL too (strictly better than the
        // pre-online provisional ticket), and stays current if our direct
        // addresses change later in the session.
        let full = EndpointTicket::from(endpoint.addr()).to_string();
        if events.send(NetEvent::Ticket { short: Some(short.clone()), full }).is_err() {
            return; // UI dropped the session.
        }
        tokio::time::sleep(TICKET_HEARTBEAT).await;
    }
}

/// The strongest available proof that our short code is dialable right now:
/// resolve OUR OWN record through the exact discovery path a joiner would
/// use. Never blocks promotion on failure (see caller) — this only decides
/// whether promotion happens immediately or after `PUBLISH_SETTLE`.
async fn confirm_published(endpoint: &Endpoint) -> bool {
    let Ok(dns) = endpoint.dns_resolver() else {
        return false;
    };
    // `presets::N0` picks the *staging* DNS origin only when iroh is built
    // with the `test-utils` feature; this repo doesn't enable it, so PROD is
    // correct here. A wrong origin would silently degrade to the settle timer
    // above rather than break anything, so this is a safe default even if
    // that ever changes.
    let origin = iroh::dns::N0_DNS_ENDPOINT_ORIGIN_PROD;
    let delays_ms = [250, 500, 1000, 2000];
    let id = endpoint.id();
    tokio::time::timeout(PUBLISH_CONFIRM_TIMEOUT, dns.lookup_endpoint_by_id_staggered(&id, origin, &delays_ms))
        .await
        .is_ok_and(|r| r.is_ok())
}

/// Accept loop: one peer at a time, but keep listening across disconnects so
/// the same invite code lets the peer rejoin after a network blip. This is
/// the sub-task whose completion actually ends `run_host` — see its doc
/// comment.
async fn accept_loop(endpoint: &Endpoint, outgoing: &mut UnboundedReceiver<Packet>, events: &Sender<NetEvent>) {
    let status = |s: String| {
        let _ = events.send(NetEvent::Status(s));
    };
    loop {
        status("Waiting for peer to join… (send them the invite code)".into());
        let conn = tokio::select! {
            incoming = endpoint.accept() => {
                let Some(incoming) = incoming else {
                    diag::log(Area::Net, "endpoint closed while accepting");
                    return;
                };
                // Race the QUIC handshake itself against UI shutdown too: a slow,
                // stalled, or malicious handshake would otherwise pin the net
                // thread here — never noticing the UI dropped the `Peer` nor
                // draining `outgoing` — until it times out seconds later (R26).
                tokio::select! {
                    res = incoming => match res {
                        Ok(conn) => conn,
                        Err(e) => {
                            diag::log(Area::Net, format!("peer failed to connect: {e}"));
                            status(format!("Peer failed to connect: {e}"));
                            continue;
                        }
                    },
                    _ = discard_until_closed(outgoing) => return,
                }
            }
            // Drain (and drop) locally-played packets while nobody is
            // listening, and notice the UI dropping the session.
            _ = discard_until_closed(outgoing) => return,
        };

        if relay_session(&conn, outgoing, events).await == SessionEnd::UiGone {
            return;
        }
        // Peer went away; loop back to accepting a rejoin.
    }
}

/// `short_code`: whether the pasted code was a bare endpoint id (vs. a full
/// ticket) — threaded through purely so the "still not reachable" hint can
/// suggest the "same-network code" only when that's actually a next step (a
/// joiner who pasted a full ticket is already immune to every discovery
/// issue here; see `run`'s parse comment).
///
/// Runs until the UI drops the `Peer` — there is no permanent give-up state.
/// Reuses the single `Endpoint` across every attempt (never rebinds): it owns
/// the UDP socket, relay actor, discovery client, and magicsock per-peer path
/// state, so rebuilding it would throw away discovered/hole-punched paths and
/// turn a 2 s blip into a full cold start.
async fn run_join(
    endpoint: &Endpoint,
    addr: EndpointAddr,
    outgoing: &mut UnboundedReceiver<Packet>,
    events: &Sender<NetEvent>,
    short_code: bool,
) {
    let status = |s: String| {
        let _ = events.send(NetEvent::Status(s));
    };

    let mut backoff_idx = 0usize;
    let mut ever_connected = false;
    loop {
        let pause = JOIN_BACKOFF[backoff_idx.min(JOIN_BACKOFF.len() - 1)];
        if !pause.is_zero() {
            tokio::select! {
                _ = tokio::time::sleep(pause) => {}
                _ = discard_until_closed(outgoing) => return,
            }
        }
        diag::log(Area::Net, format!("dialing host (backoff step {})", backoff_idx + 1));
        status(if ever_connected {
            "Reconnecting to host…".into()
        } else {
            "Connecting to host…".into()
        });
        let result = tokio::select! {
            conn = endpoint.connect(addr.clone(), ALPN) => conn,
            _ = discard_until_closed(outgoing) => return,
        };
        match result {
            Ok(conn) => {
                diag::log(Area::Net, format!("connected to peer {}", conn.remote_id().fmt_short()));
                ever_connected = true;
                let connected_at = Instant::now();
                if relay_session(&conn, outgoing, events).await == SessionEnd::UiGone {
                    return;
                }
                // A session that stood up for a while counts as "real": reset
                // to the dense head of the schedule rather than inheriting
                // whatever pause the dial-retry schedule was mid-way through.
                backoff_idx = if connected_at.elapsed() >= RECONNECT_RESET_AFTER {
                    0
                } else {
                    backoff_idx.saturating_add(1)
                };
                status("Disconnected — reconnecting…".into());
            }
            Err(e) => {
                diag::log(Area::Net, format!("dial failed ({e}); retrying after backoff"));
                // A stale POSITIVE record (the host restarted or moved
                // networks) is the only thing worth clearing here — negative
                // answers are never cached in the first place (iroh-dns sets
                // `negative_max_ttl = Duration::ZERO`), so this can't be
                // "optimized away" on the negative-caching theory.
                if let Ok(dns) = endpoint.dns_resolver() {
                    dns.clear_cache();
                }
                // Once we've cycled past the dense head into the 30 s ceiling,
                // add the actionable hint — a joiner stuck this long is the
                // exact case it exists for — without ever fully giving up.
                let hint = if short_code && backoff_idx >= JOIN_BACKOFF.len() - 1 {
                    " If you're both on the same Wi-Fi, ask them for the \"same-network code\" instead."
                } else {
                    ""
                };
                status(format!("Not reachable yet ({e}); retrying…{hint}"));
                backoff_idx = backoff_idx.saturating_add(1);
            }
        }
    }
}

#[derive(PartialEq)]
enum SessionEnd {
    /// The connection dropped (peer left / network died).
    PeerGone,
    /// The UI dropped the `Peer` handle; shut the whole session down.
    UiGone,
}

/// Pump one live connection: outgoing packets become datagrams, incoming
/// datagrams become `Packet` events. Returns why the session ended.
async fn relay_session(
    conn: &Connection,
    outgoing: &mut UnboundedReceiver<Packet>,
    events: &Sender<NetEvent>,
) -> SessionEnd {
    diag::log(Area::Net, format!("session started with peer {}", conn.remote_id().fmt_short()));
    let _ = events.send(NetEvent::Connected);
    let _ = events.send(NetEvent::Status(format!(
        "Connected to peer {}",
        conn.remote_id().fmt_short()
    )));

    // Last datagram-send error surfaced to the UI, so a *persistent* failure is
    // reported once (not every frame) and only when the failure kind changes
    // (R9). Without this the UI keeps showing "Connected" while a packet class —
    // or all traffic — silently stops flowing.
    let mut last_send_err: Option<String> = None;

    loop {
        tokio::select! {
            datagram = conn.read_datagram() => match datagram {
                Ok(bytes) => {
                    if let Some(packet) = Packet::decode(&bytes) {
                        // Metronome markers carry an RTT-derived one-way estimate,
                        // stamped here where the RTT is freshest; everything else
                        // rides the generic Packet event.
                        let event = match packet {
                            Packet::MetroBeat { bpm, beat_in_bar, beats_per_bar, on } => {
                                let rtt = conn
                                    .rtt(iroh::endpoint::PathId::ZERO)
                                    .unwrap_or_default();
                                NetEvent::MetroBeat {
                                    bpm,
                                    beat_in_bar,
                                    beats_per_bar,
                                    on,
                                    one_way: rtt / 2,
                                }
                            }
                            other => NetEvent::Packet(other),
                        };
                        if events.send(event).is_err() {
                            conn.close(0u32.into(), b"closed");
                            return SessionEnd::UiGone;
                        }
                    }
                }
                Err(e) => {
                    diag::log(Area::Net, format!("session ended: peer disconnected ({e})"));
                    let _ = events.send(NetEvent::Disconnected);
                    let _ = events.send(NetEvent::Status(format!("Peer disconnected: {e}")));
                    return SessionEnd::PeerGone;
                }
            },
            packet = outgoing.recv() => match packet {
                // Best-effort, like the old UDP path: a lost datagram is expected
                // (the heartbeat re-sends). But a *persistent* error like
                // `TooLarge` (an oversized snapshot exceeding the path MTU) would
                // silently stop that packet type from ever arriving, so surface
                // it rather than discarding it blindly (F21).
                Some(p) => {
                    match conn.send_datagram(Bytes::from(p.encode())) {
                        Ok(()) => {
                            // Recovered: let the UI clear the warning.
                            if last_send_err.take().is_some() {
                                let _ = events.send(NetEvent::Status(format!(
                                    "Connected to peer {}",
                                    conn.remote_id().fmt_short()
                                )));
                            }
                        }
                        Err(e) => {
                            // Surface it to the log and the UI (release builds
                            // have no stderr), rate-limited to error-kind
                            // changes so a sustained failure doesn't spam
                            // either (R9).
                            let kind = e.to_string();
                            if last_send_err.as_deref() != Some(kind.as_str()) {
                                diag::log(Area::Net, format!("datagram send failed: {kind}"));
                                let _ = events.send(NetEvent::Status(format!(
                                    "Connected, but some data isn't syncing: {kind}"
                                )));
                                last_send_err = Some(kind);
                            }
                        }
                    }
                }
                None => {
                    conn.close(0u32.into(), b"closed");
                    return SessionEnd::UiGone;
                }
            },
        }
    }
}

/// Resolve only when the UI has dropped its `Peer` (sender side closed),
/// discarding any packets queued meanwhile. Used while no peer is connected.
async fn discard_until_closed(outgoing: &mut UnboundedReceiver<Packet>) {
    while outgoing.recv().await.is_some() {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::note::NoteMsg;
    use std::time::Instant;

    /// Wait (with a deadline) for the next event matching `pick`, skipping
    /// `Status` noise along the way.
    fn wait_for<T>(peer: &Peer, what: &str, pick: impl Fn(&NetEvent) -> Option<T>) -> T {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .unwrap_or_else(|| panic!("timed out waiting for {what}"));
            match peer.events.recv_timeout(remaining) {
                Ok(ev) => {
                    if let Some(v) = pick(&ev) {
                        return v;
                    }
                }
                Err(e) => panic!("waiting for {what}: {e}"),
            }
        }
    }

    /// End-to-end over real iroh: host issues a ticket, a second endpoint
    /// joins with it, and note datagrams flow both ways. Needs a network
    /// stack (loopback at minimum); with no internet the host stays on the
    /// provisional full ticket (`short: None`) past `LAN_FALLBACK_DELAY`, so
    /// the test still passes on whichever code is available — just slower.
    #[test]
    fn host_join_exchange_notes() {
        let host = host(None);
        let code = wait_for(&host, "invite ticket", |ev| match ev {
            NetEvent::Ticket { short, full } => Some(short.clone().unwrap_or_else(|| full.clone())),
            _ => None,
        });

        let joiner = join(code, None);
        wait_for(&joiner, "joiner connect", |ev| matches!(ev, NetEvent::Connected).then_some(()));
        wait_for(&host, "host connect", |ev| matches!(ev, NetEvent::Connected).then_some(()));

        // Datagrams are fire-and-forget, so poll-and-resend instead of
        // asserting on a single send (matches how the app's 1 s color
        // heartbeat papers over any individual loss).
        let exchange = |from: &Peer, to: &Peer, packet: Packet, what: &str| {
            for _ in 0..100 {
                from.send(packet.clone());
                match to.events.recv_timeout(Duration::from_millis(500)) {
                    Ok(NetEvent::Packet(p)) if p == packet => return,
                    Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(e) => panic!("waiting for {what}: {e}"),
                }
            }
            panic!("never received {what}");
        };
        exchange(&joiner, &host, Packet::Note(NoteMsg::On(60, 100), 0), "note at host");
        exchange(&host, &joiner, Packet::Color([1, 2, 3]), "color at joiner");

        // Metronome beat markers surface as a distinct `NetEvent::MetroBeat`
        // (with an RTT-derived one-way stamp), not a generic `Packet` event.
        let mut got_beat = false;
        for _ in 0..100 {
            host.send(Packet::MetroBeat { bpm: 128, beat_in_bar: 2, beats_per_bar: 4, on: true });
            match joiner.events.recv_timeout(Duration::from_millis(500)) {
                Ok(NetEvent::MetroBeat { bpm, beat_in_bar, beats_per_bar, on, .. }) => {
                    assert_eq!((bpm, beat_in_bar, beats_per_bar, on), (128, 2, 4, true));
                    got_beat = true;
                    break;
                }
                Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(e) => panic!("waiting for metro beat: {e}"),
            }
        }
        assert!(got_beat, "never received a metronome beat marker");
    }
}
