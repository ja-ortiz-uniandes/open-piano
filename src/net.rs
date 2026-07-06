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
//! from. When the host can't reach a relay (offline / LAN-only), discovery
//! would have nothing to serve, so it falls back to a full [`EndpointTicket`]
//! (~4× longer) that carries the direct addresses inline. Joining accepts
//! either form, which also keeps codes from older versions working.
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
use std::time::Duration;

use bytes::Bytes;
use iroh::endpoint::presets;
use iroh::endpoint::Connection;
use iroh::{Endpoint, EndpointAddr, EndpointId};
use iroh_tickets::endpoint::EndpointTicket;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::note::Packet;

/// Application-level protocol id required to match on both ends of a
/// connection; bump the suffix if the wire format ever changes incompatibly.
const ALPN: &[u8] = b"open-piano/0";

/// How long the host waits for first relay contact before publishing the
/// invite code anyway. Without a relay the code still carries the direct
/// (LAN) addresses, so same-network play keeps working fully offline.
const ONLINE_TIMEOUT: Duration = Duration::from_secs(15);

/// How many times a joiner dials before giving up, and the pause between
/// tries. Retrying matters most for short (id-only) invite codes: the host
/// publishes its relay address to n0's discovery service asynchronously, so a
/// joiner who is very quick off the mark can look the id up before the record
/// has propagated. A couple of spaced retries papers over that window.
const JOIN_ATTEMPTS: u32 = 4;
const JOIN_RETRY_PAUSE: Duration = Duration::from_secs(3);

/// Everything the net thread reports back to the UI, drained once per frame.
#[derive(Debug, Clone)]
pub enum NetEvent {
    /// Host only: the invite code is ready to be copied and sent to the peer.
    Ticket(String),
    /// Human-readable progress / error line for the status bar.
    Status(String),
    /// A peer connection is live. The UI clears remote key state (unknown
    /// across a reconnect) and re-announces its color.
    Connected,
    /// The peer connection dropped. A host keeps listening for a rejoin;
    /// a joiner must press Join again.
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
/// and waits for a peer. Keeps accepting across peer disconnects.
pub fn host() -> Peer {
    start(Role::Host)
}

/// Join a host from a pasted invite code (parsed and validated on the net
/// thread; a bad code comes back as a `Status` event).
pub fn join(ticket: String) -> Peer {
    start(Role::Join(ticket))
}

enum Role {
    Host,
    Join(String),
}

fn start(role: Role) -> Peer {
    let (event_tx, event_rx) = mpsc::channel::<NetEvent>();
    let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<Packet>();

    thread::Builder::new()
        .name("net".into())
        .spawn(move || {
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
            rt.block_on(run(role, out_rx, event_tx));
        })
        .expect("failed to spawn net thread");

    Peer {
        outgoing: out_tx,
        events: event_rx,
    }
}

/// Net-thread main. Any `Err` is a fatal setup problem already reported as a
/// `Status` event; connection-level errors are handled inside and don't end
/// the session (the host goes back to listening).
async fn run(role: Role, mut outgoing: UnboundedReceiver<Packet>, events: Sender<NetEvent>) {
    let status = |s: String| {
        let _ = events.send(NetEvent::Status(s));
    };

    // Parse the invite code first (join only) so a typo fails fast, before
    // any network work. Short form (bare endpoint id) is tried first; the
    // long form (full ticket: LAN-only hosts, older versions) second.
    let target: Option<EndpointAddr> = match &role {
        Role::Host => None,
        Role::Join(code) => match code.parse::<EndpointId>() {
            Ok(id) => Some(id.into()),
            Err(_) => match code.parse::<EndpointTicket>() {
                Ok(t) => Some(t.endpoint_addr().clone()),
                Err(e) => {
                    status(format!("Invalid invite code: {e}"));
                    return;
                }
            },
        },
    };

    status("Setting up p2p endpoint…".into());
    // The N0 preset = n0's public relay servers + endpoint discovery. This is
    // what makes the whole thing zero-config across NATs.
    let endpoint = match Endpoint::builder(presets::N0)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await
    {
        Ok(ep) => ep,
        Err(e) => {
            status(format!("Failed to start networking: {e}"));
            return;
        }
    };

    match target {
        None => run_host(&endpoint, &mut outgoing, &events).await,
        Some(addr) => run_join(&endpoint, addr, &mut outgoing, &events).await,
    }

    // Graceful close tells the peer immediately instead of leaving it to the
    // QUIC idle timeout.
    endpoint.close().await;
}

async fn run_host(
    endpoint: &Endpoint,
    outgoing: &mut UnboundedReceiver<Packet>,
    events: &Sender<NetEvent>,
) {
    let status = |s: String| {
        let _ = events.send(NetEvent::Status(s));
    };

    // Wait for relay contact, then hand out the short code (bare endpoint
    // id): once we're online, discovery serves the joiner everything else.
    // Offline/LAN-only hosts get no relay and no discovery, so after the
    // timeout fall back to the long code with the direct addresses inline.
    status("Contacting relay…".into());
    let code = if tokio::time::timeout(ONLINE_TIMEOUT, endpoint.online())
        .await
        .is_ok()
    {
        endpoint.id().to_string()
    } else {
        status("No relay reachable — invite code will only work on this network".into());
        EndpointTicket::from(endpoint.addr()).to_string()
    };
    if events.send(NetEvent::Ticket(code)).is_err() {
        return; // UI dropped the session.
    }

    // Accept loop: one peer at a time, but keep listening across disconnects
    // so the same invite code lets the peer rejoin after a network blip.
    loop {
        status("Waiting for peer to join… (send them the invite code)".into());
        let conn = tokio::select! {
            incoming = endpoint.accept() => {
                let Some(incoming) = incoming else { return }; // endpoint closed
                match incoming.await {
                    Ok(conn) => conn,
                    Err(e) => {
                        status(format!("Peer failed to connect: {e}"));
                        continue;
                    }
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

async fn run_join(
    endpoint: &Endpoint,
    addr: EndpointAddr,
    outgoing: &mut UnboundedReceiver<Packet>,
    events: &Sender<NetEvent>,
) {
    let status = |s: String| {
        let _ = events.send(NetEvent::Status(s));
    };

    for attempt in 1..=JOIN_ATTEMPTS {
        if attempt == 1 {
            status("Connecting to host…".into());
        } else {
            status(format!(
                "Connecting to host… (attempt {attempt}/{JOIN_ATTEMPTS})"
            ));
        }
        let result = tokio::select! {
            conn = endpoint.connect(addr.clone(), ALPN) => conn,
            _ = discard_until_closed(outgoing) => return,
        };
        match result {
            Ok(conn) => {
                if relay_session(&conn, outgoing, events).await == SessionEnd::PeerGone {
                    status("Disconnected — press Join to reconnect".into());
                }
                return;
            }
            Err(e) if attempt < JOIN_ATTEMPTS => {
                // Most likely the host's discovery record hasn't propagated
                // yet (id-only code, host just started); pause and retry.
                status(format!("Not reachable yet ({e}); retrying…"));
                tokio::select! {
                    _ = tokio::time::sleep(JOIN_RETRY_PAUSE) => {}
                    _ = discard_until_closed(outgoing) => return,
                }
            }
            Err(e) => status(format!("Could not reach host: {e}")),
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
    let _ = events.send(NetEvent::Connected);
    let _ = events.send(NetEvent::Status(format!(
        "Connected to peer {}",
        conn.remote_id().fmt_short()
    )));

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
                    let _ = events.send(NetEvent::Disconnected);
                    let _ = events.send(NetEvent::Status(format!("Peer disconnected: {e}")));
                    return SessionEnd::PeerGone;
                }
            },
            packet = outgoing.recv() => match packet {
                // Best-effort, like the old UDP path: a send error just means
                // the connection is going away; read_datagram reports it.
                Some(p) => { let _ = conn.send_datagram(Bytes::from(p.encode())); }
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
    /// stack (loopback at minimum); with no internet the host falls back to
    /// a direct-addresses-only ticket after `ONLINE_TIMEOUT`, so the test
    /// still passes — just slower.
    #[test]
    fn host_join_exchange_notes() {
        let host = host();
        let code = wait_for(&host, "invite ticket", |ev| match ev {
            NetEvent::Ticket(t) => Some(t.clone()),
            _ => None,
        });

        let joiner = join(code);
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
        exchange(&joiner, &host, Packet::Note(NoteMsg::On(60)), "note at host");
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
