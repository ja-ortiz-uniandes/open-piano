//! open-piano — a real-time, networked, peer-to-peer acoustic piano visualizer.
//!
//! * Note input comes from one of two backends, chosen automatically at startup
//!   (see `input.rs`): a connected **MIDI** device (preferred — exact events),
//!   or, as a fallback, **microphone** audio captured via cpal and transcribed
//!   by an ONNX model (Spotify Basic Pitch) on a dedicated inference thread.
//!   Either way the resulting note transitions arrive on one mpsc channel.
//! * Notes played locally light up RED; notes arriving over the p2p connection
//!   from the remote peer light up BLUE (both -> purple).
//!
//! See `input.rs`, `midi.rs`, `audio.rs`, `inference.rs`, `net.rs` and
//! `note.rs` for the subsystems.

// Hide the console window on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;
mod bundle;
mod inference;
mod input;
mod midi;
mod net;
mod note;
mod playback;
mod prefs;
mod record;
mod roll;
mod score;
mod synth;
mod update;

use eframe::egui;

use std::path::PathBuf;

use std::time::{Duration, Instant};

use input::{InputEngine, Source};
use net::{NetEvent, Peer};
use note::{
    is_black_key, midi_to_key_index, pack_held, unpack_held, NoteMsg, Packet, HELD_MASK_BYTES,
    KEY_COUNT, MIDI_HIGH, MIDI_LOW,
};

// Defensive: when ONNX Runtime later probes for optional execution-provider
// DLLs (CUDA/DirectML/etc.) on the inference thread, a missing one can make
// Windows pop a *blocking* "hard error" message box that's invisible to a GUI
// app. This tells the OS to fail such loads quietly (return an error) instead.
// (SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX = 0x8001.)
#[cfg(windows)]
fn suppress_dll_error_dialogs() {
    #[link(name = "kernel32")]
    extern "system" {
        fn SetErrorMode(mode: u32) -> u32;
    }
    unsafe {
        SetErrorMode(0x0001 | 0x8000);
    }
}

fn main() -> eframe::Result<()> {
    #[cfg(windows)]
    suppress_dll_error_dialogs();

    // Extract the embedded ONNX Runtime and point ORT_DYLIB_PATH at it — file
    // I/O only, do NOT load the DLL here. Loading ONNX Runtime spins up its
    // own threads during initialisation; doing that from the main thread
    // before the event loop deadlocks against the Windows loader lock and
    // freezes the app before any window appears. ort loads the DLL lazily the
    // first time a Session is built — which happens on the dedicated
    // inference thread (see inference.rs) where a slow or failing load can't
    // block the GUI.
    bundle::prepare_ort_dylib();

    let icon = eframe::icon_data::from_png_bytes(bundle::ICON_PNG)
        .expect("assets/icon.png must be a valid PNG (embedded at compile time)");
    // Load prefs here (not inside `PianoApp::new`) so the OS window can be
    // created *already* at the remembered compact size — no post-launch
    // resize flash.
    let prefs = prefs::Prefs::load();
    let start_compact = prefs.remember_window_state && prefs.compact_mode;
    let (inner_size, min_size) = if start_compact {
        ([DEFAULT_WINDOW_SIZE[0], COMPACT_WINDOW_H], COMPACT_MIN_SIZE)
    } else {
        (DEFAULT_WINDOW_SIZE, NORMAL_MIN_SIZE)
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(inner_size)
            .with_min_inner_size(min_size)
            .with_icon(icon)
            // Custom window chrome: we draw our own title bar (File/Edit menus,
            // min/max/close) so the menus are always the topmost row, and add
            // our own edge-resize handles (see `title_bar`). `with_title` still
            // sets the taskbar/alt-tab label. NOTE: Windows 11 snap-layouts (the
            // hover menu over the maximize button) need WM_NCHITTEST hooks that
            // winit/egui don't expose, so they're unavailable with custom chrome.
            .with_decorations(false)
            .with_title(concat!(
                "open-piano v",
                env!("CARGO_PKG_VERSION"),
                " — P2P acoustic piano visualizer"
            )),
        ..Default::default()
    };

    eframe::run_native(
        "open-piano",
        options,
        Box::new(move |_cc| Ok(Box::new(PianoApp::new(prefs)))),
    )
}

/// The version compiled into this build (from Cargo.toml) — what the About
/// dialog and window title report, and what the auto-updater compares against
/// GitHub Releases.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default note colors (sRGB). The live *local* color is seeded from
/// [`prefs::Prefs`] (user-configurable and persisted); these constants remain
/// the compile-time fallbacks — `DEFAULT_LOCAL_COLOR` also colors a loaded
/// file's Local track (see `score.rs`), and `DEFAULT_REMOTE_COLOR` is what the
/// peer always renders as locally: we deliberately ignore the peer's announced
/// color (see the `Packet::Color` handler) so two un-customized peers — both
/// defaulting `local_color` to the same red — never render identically.
const DEFAULT_LOCAL_COLOR: [u8; 3] = [220, 60, 60]; // warm red
const DEFAULT_REMOTE_COLOR: [u8; 3] = [60, 110, 230]; // blue (the peer, always)

/// Placeholder name shown for the peer until it announces its own (see the
/// `Packet::Name` heartbeat). Transient — overwritten on the first announce.
const DEFAULT_REMOTE_NAME: &str = "Peer";

/// How often to re-broadcast our color to the peer. A low-rate heartbeat means
/// color syncs regardless of who connects first, and recovers from a dropped
/// announcement, at a negligible 1 datagram/sec.
const COLOR_HEARTBEAT: Duration = Duration::from_secs(1);

/// Cap on the number of manual segment breaks put in one `Packet::Separators`
/// snapshot. Each is 8 bytes; the datagram must stay under the ~1200-byte
/// path-MTU limit (past which QUIC drops it and separator sync silently stops),
/// so we bound the count well below that (F21). Far more manual breaks than any
/// real session has.
const MAX_WIRE_SEPARATORS: usize = 120;

/// Debounce for persisting preferences edited by a continuous control (sliders,
/// drag-values): egui fires `.changed()` every frame of a drag, so saving
/// immediately would serialize + temp-write + rename at ~60 Hz on the GUI
/// thread. Instead the change schedules a single save this long after the last
/// edit (see `save_prefs_soon`) — well under a second, imperceptible (M7).
const PREFS_SAVE_DEBOUNCE: Duration = Duration::from_millis(500);

// The mic echo hold-off, roll zoom (px/s), scrollback idle window, detection
// threshold and default local color are now user preferences (see prefs.rs) —
// read live from `self.prefs` at their use sites instead of compile-time consts.

/// Exponential ease-back rate (1/s) for a scrolled roll returning home —
/// matches the ~0.2 s feel the history roll's drag-release always had.
const SCROLLBACK_EASE_RATE: f64 = 12.0;

/// How the central panel splits between the keyboard (top) and the roll
/// (bottom): the keyboard takes this fraction of the height, but never less
/// than `MIN_KEYBOARD_H` so keys stay playable in a short window.
const KEYBOARD_FRACTION: f32 = 0.45;
const MIN_KEYBOARD_H: f32 = 140.0;
/// Cap on a user-dragged keyboard height override, as a fraction of the
/// central panel's height.
const MAX_KEYBOARD_FRACTION: f32 = 0.85;
/// Thickness (px) of the invisible drag-handle strip straddling the
/// keyboard's top/bottom edge.
const KB_RESIZE_HANDLE_H: f32 = 6.0;

/// Compact mode ("alternate minimize"): the window shrinks to just the title
/// bar + keyboard. Height chosen so the keys stay comfortably playable; the
/// compact minimum still leaves room for the full 88-key span.
const COMPACT_WINDOW_H: f32 = 230.0;
const COMPACT_MIN_SIZE: [f32; 2] = [640.0, 190.0];
/// Normal-mode window sizing — one source of truth shared by `main()`'s
/// `ViewportBuilder` and the compact-mode restore path.
const NORMAL_MIN_SIZE: [f32; 2] = [640.0, 420.0];
const DEFAULT_WINDOW_SIZE: [f32; 2] = [1100.0, 620.0];

/// Height of the custom title bar. The top strip is reserved for it (move-drag,
/// the File/Edit menus, and the window buttons), so the edge-resize handles keep
/// clear of it — otherwise a touch-sized corner handle sits over the ✕ button
/// and a tap there starts a resize instead of closing (F23).
const TITLEBAR_H: f32 = 30.0;

/// With a score loaded, the space not taken by the keyboard splits between
/// the falling-notes panel (above the keys) and the history roll (below).
/// Biased toward the falling notes — the forward-looking practice aid — over
/// history review.
const FALLING_FRACTION: f32 = 0.55;

/// Ctrl+S (Cmd+S on mac) quick-saves the roll — same action as File ▸ Save.
const SAVE_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S);

/// Ctrl+, (Cmd+, on mac) opens Edit ▸ Preferences — the conventional shortcut.
const PREFS_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Comma);

/// Metronome tempo bounds (BPM), shared by the UI control and the wire clamp.
const MIN_BPM: u16 = 30;
const MAX_BPM: u16 = 240;

/// A few common click pitches (Hz), offered as quick-pick stops on the preset
/// slider ahead of each beat's precise Hz field — low/mid/high/bright.
const METRO_FREQ_PRESETS: [f32; 4] = [800.0, 1200.0, 1800.0, 2400.0];

/// Shared metronome state. The **authority** (the host, or a solo player with no
/// peer) owns the canonical beat grid and, when connected, broadcasts a
/// `MetroBeat` each beat; a **follower** (joined to a host) anchors a local click
/// schedule to those markers — clicks are always generated locally, so packet
/// loss/jitter never drops or delays one. Either side sets tempo / start-stop; a
/// follower's edit is a `MetroCtl` request the host adopts (last-writer-wins,
/// one grid). `muted`/`volume` control *this* player's click only (local channel
/// gain), never the peer's — but `beat_freqs`/`beat_volumes` (the per-beat pitch
/// and level tables) ARE synced (`Packet::MetroBeatTable`, no host authority:
/// whoever edits last wins), so both sides' clicks sound identical.
struct Metronome {
    enabled: bool,
    bpm: u16,
    beats_per_bar: u8,
    muted: bool,
    /// Click output level (0..1), independent of mute.
    volume: f32,
    /// Pitch (Hz) of each beat's click, indexed by position in the bar — index 0
    /// is the accent/downbeat. Resized to match `beats_per_bar` by the
    /// Preferences UI; a beat index beyond the table falls back to a plain tone
    /// (see `drive_metronome`).
    beat_freqs: Vec<f32>,
    /// Per-beat click level (0..1), indexed the same way as `beat_freqs`.
    /// Resized alongside it; a beat index beyond the table falls back to full
    /// volume (see `drive_metronome`).
    beat_volumes: Vec<f32>,
    /// When the next local click fires. On enable, the authority anchors this to
    /// the next tick of a fixed grid derived from the *roll's* clock (see
    /// `PianoApp::metro_start_now`) — beat 0 of the grid sits at roll-time zero,
    /// so beats always land on round absolute positions (e.g. every whole
    /// minute) regardless of when the metronome was actually started, rather
    /// than free-running from an arbitrary "now". It free-runs (adds `period`)
    /// from there. A follower has it (re)anchored by each incoming marker.
    /// `None` = no click scheduled (off, or a follower awaiting its first
    /// marker).
    next_beat_at: Option<Instant>,
    /// The next click's position in the bar (accent when 0 = the downbeat).
    next_beat_in_bar: u8,
    /// When the most recent beat was processed (sounded or not), on the local
    /// clock. Lets a follower tell a genuinely new incoming marker from a
    /// re-delivery of one it already handled, so it doesn't double-click when
    /// firing the current beat on marker arrival (F3).
    last_beat_at: Option<Instant>,
}

impl Metronome {
    fn new(bpm: u16) -> Self {
        Metronome {
            enabled: false,
            bpm: bpm.clamp(MIN_BPM, MAX_BPM),
            beats_per_bar: 4,
            muted: false,
            volume: 1.0,
            beat_freqs: vec![1800.0, 1200.0, 1200.0, 1200.0],
            beat_volumes: vec![1.0, 1.0, 1.0, 1.0],
            next_beat_at: None,
            next_beat_in_bar: 0,
            last_beat_at: None,
        }
    }

    /// Seconds per beat at the current tempo.
    fn period(&self) -> f64 {
        60.0 / self.bpm.max(1) as f64
    }

    /// The pitch (Hz) for a given position in the bar, falling back to a plain
    /// tone if the table is shorter than `beats_per_bar` (e.g. mid-edit, or a
    /// peer just broadcast a larger `beats_per_bar` than our local table).
    fn freq_for_beat(&self, beat_in_bar: u8) -> f32 {
        self.beat_freqs.get(beat_in_bar as usize).copied().unwrap_or(1200.0)
    }

    /// The click level (0..1) for a given position in the bar, falling back to
    /// full volume if the table is shorter than `beats_per_bar`.
    fn volume_for_beat(&self, beat_in_bar: u8) -> f32 {
        self.beat_volumes.get(beat_in_bar as usize).copied().unwrap_or(1.0)
    }
}

/// Which page of the Preferences dialog is showing — drives the sidebar
/// navigation (see `PianoApp::preferences_window`). Pure UI-nav state,
/// deliberately not persisted.
#[derive(Clone, Copy, PartialEq)]
enum PrefsSection {
    StartupWindow,
    RollHistory,
    Appearance,
    Pedal,
    RollBehavior,
    AudioMic,
    Metronome,
    Advanced,
}

impl PrefsSection {
    /// Sidebar order.
    const ALL: [PrefsSection; 8] = [
        PrefsSection::StartupWindow,
        PrefsSection::RollHistory,
        PrefsSection::Appearance,
        PrefsSection::Pedal,
        PrefsSection::RollBehavior,
        PrefsSection::AudioMic,
        PrefsSection::Metronome,
        PrefsSection::Advanced,
    ];

    fn label(self) -> &'static str {
        match self {
            PrefsSection::StartupWindow => "Startup & window",
            PrefsSection::RollHistory => "Roll & history",
            PrefsSection::Appearance => "Appearance",
            PrefsSection::Pedal => "Pedal",
            PrefsSection::RollBehavior => "Roll behavior",
            PrefsSection::AudioMic => "Audio / mic",
            PrefsSection::Metronome => "Metronome",
            PrefsSection::Advanced => "Advanced",
        }
    }
}

struct PianoApp {
    // --- note input (MIDI or microphone fallback) ---
    input: InputEngine,
    // Built-in synth voicing the notes that have no acoustic source: the keys
    // you click on screen and the ones the peer plays (MIDI/mic make their own
    // real sound, so they're intentionally not synthesized).
    synth: synth::Synth,
    // Per-source synth controls. `*_volume` is the slider value (0..1) and
    // `*_muted` is the mute toggle; the effective gain pushed to the synth is
    // `volume` unless muted, in which case 0. "Screen" = keys you click here;
    // "peer" = notes the remote player sends.
    screen_volume: f32,
    screen_muted: bool,
    peer_volume: f32,
    peer_muted: bool,
    threshold: f32,
    // Last input-switch epoch we acted on; a change means the backend swapped
    // (e.g. MIDI unplugged) and we must force-release held notes.
    notes_epoch: u64,
    // Whether a MIDI device was connected as of the last frame. A change drives
    // the source-dependent synth default in `sync_synth_to_source` (mute the
    // on-screen synth while a real piano is connected; unmute on the mic
    // fallback). Starts `false` so the first MIDI connection triggers the mute.
    was_midi: bool,

    // --- key state ---
    local: [bool; KEY_COUNT],
    remote: [bool; KEY_COUNT],
    // Keys the user has "pinned" down with Ctrl+click — a display aid for
    // holding a chord to point at while explaining. ORed into `local`'s
    // rendering only *while Ctrl is held*; it never touches `local_note`, the
    // synth, or the roll, and toggling is gated by the same recording/eval/
    // playback lock as mouse-play. It IS shared with the peer, though (see
    // `Packet::Held` / `remote_held`): the keyboard is one surface both players
    // look at, so a pinned chord lights up on both screens (in the pinner's
    // color) just like a live press. Memoryless by design: releasing Ctrl both
    // hides *and clears* the pinned set (see `held_cmd_down`) — and broadcasts
    // the clear — so each Ctrl-hold starts fresh instead of resurrecting a
    // chord pinned in some earlier, unrelated hold.
    held: [bool; KEY_COUNT],
    // The peer's currently-pinned keys (their `held`, received via
    // `Packet::Held`). Rendered like remote presses, in the remote color —
    // cleared on connect/disconnect like every other remote state.
    remote_held: [bool; KEY_COUNT],
    // The last pinned-key mask we sent the peer, so we only broadcast on change
    // (mirrors `last_pedal_sent`); reset on (re)connect so a fresh peer resyncs.
    last_held_sent: [u8; HELD_MASK_BYTES],
    // The last live (sounding) note mask we sent the peer as a self-heal
    // snapshot (`Packet::Live`); reset on (re)connect. See `broadcast_live`.
    last_live_sent: [u8; HELD_MASK_BYTES],
    // Per-sender monotonic sequence stamped on every outbound note event and
    // `Packet::Live` snapshot, so the receiver can drop a stale snapshot that
    // reordered past a newer note transition (F6). Monotonic for the process
    // lifetime (never reset), so a reconnecting peer — which resets its own
    // high-water mark — always sees our seq climb.
    live_seq: u32,
    // Sequence stamped on every outbound `Packet::Held` snapshot (F6). Held has
    // no per-event channel, so this just orders successive whole-state snapshots.
    held_seq: u32,
    // Highest note/live seq we've applied from the peer, and highest held seq —
    // snapshots older than these are ignored. Reset on (re)connect (a fresh peer
    // starts its seq low), see `clear_remote_keys`.
    remote_live_seq: u32,
    remote_held_seq: u32,
    // Whether Ctrl/Cmd was down as of the last frame — lets us detect the
    // release edge and clear `held` exactly once (see `held` doc comment).
    held_cmd_down: bool,

    // --- sustain pedal (CC64; MIDI input only — the mic path can't produce
    // pedal events, see input.rs) ---
    // Latest level per side (0 = up). Local levels feed the roll's pedal lane
    // and are forwarded to the peer send-on-change (`last_pedal_sent` dedupes;
    // no heartbeat — CC64 re-fires constantly while half-pedaling, so a lost
    // datagram self-heals on the next change).
    local_pedal: u8,
    remote_pedal: u8,
    last_pedal_sent: u8,
    // This frame's local note onsets as (midi, velocity), cleared at the top
    // of every `update()`. The evaluation scorer needs velocities, which only
    // exist at the instant of attack — they can't be reconstructed later from
    // the bare held-key set.
    frame_onsets: Vec<(u8, u8)>,

    // Ignore mic-detected note *onsets* while set (offs still pass, so held
    // notes close instead of sticking). Useful when ambient noise keeps
    // painting phantom notes on the roll — and during Learn-mode playback if
    // the room is noisy.
    mic_muted: bool,

    // --- mic↔synth echo guard (only relevant when the mic is the source) ---
    // The synth voices the on-screen keyboard and the peer through the speakers,
    // which the microphone then hears and re-transcribes. To break that loop we
    // track, per MIDI note, whether the synth is currently voicing it (indexed by
    // `synth::Channel as usize`) and, once it stops, an instant until which a
    // mic-detected onset of that note is still ignored (`prefs.echo_holdoff_ms`).
    echo_held: [[bool; 2]; 128],
    echo_until: [Option<Instant>; 128],

    // --- mouse "play the on-screen keyboard" input ---
    // The single MIDI note the mouse is currently holding down (one pointer →
    // one note; dragging across keys releases the old and presses the new).
    mouse_note: Option<u8>,
    // Pointer-stillness tracking, used to expand the "keyboard locked" tooltip
    // shown while recording after ~1 s of the mouse sitting still.
    pointer_still_since: Instant,
    last_pointer_pos: Option<egui::Pos2>,

    // --- colors (sRGB) ---
    local_color: [u8; 3], // our notes; editable, broadcast to the peer
    // The color the peer's notes are drawn in *on this screen*. Fixed to
    // `DEFAULT_REMOTE_COLOR`: the peer's announced `Packet::Color` is
    // deliberately discarded on receive (every fresh install defaults to the
    // same red, so honoring it would render both sides identically). Not "the
    // peer's chosen color".
    remote_color: [u8; 3],
    last_color_send: Instant,

    // --- display names (ride the same heartbeat as colors) ---
    local_name: String,  // our name; editable, persisted, broadcast to the peer
    remote_name: String, // the peer's name; received from the peer

    // --- networking (see net.rs: host/join with a one-string invite code) ---
    // Our invite code, once the net thread reports it (hosting only). Shown
    // with a Copy button so it can be pasted to the other player.
    my_ticket: Option<String>,
    // The paste box for an invite code received from a host.
    join_ticket: String,
    peer: Option<Peer>,
    net_status: String,
    // Whether *we* are the host of the current session (the metronome timing
    // authority). Meaningless without a peer — then we're always the authority.
    is_host: bool,
    // Whether a peer connection is actually live right now (true between
    // `NetEvent::Connected` and `Disconnected`). A joiner whose host quit still
    // holds a `Some(peer)`, so this — not `peer.is_some()` — is what lets it
    // reclaim metronome authority and keep clicking (M9).
    peer_connected: bool,
    // The playback key range as of last frame, to detect a mid-take edit and
    // restart the evaluation take (M13). `None` when no score is loaded.
    last_key_range: Option<(u8, u8)>,
    // When a debounced preferences save is due (see `save_prefs_soon`), coalescing
    // the per-frame `.changed()` storm of a slider drag into one write (M7).
    prefs_save_due: Option<Instant>,
    // Roll-clock times of segment breaks *we* inserted by hand (Ctrl+click /
    // context menu). Broadcast to the peer on change and on the heartbeat
    // (`Packet::Separators`) so a manual break is a shared surface element, not
    // a line on one screen (M2). Auto-pause breaks stay local — the two rolls
    // run independent clocks, so syncing derived boundaries would misplace them.
    manual_separators: Vec<f64>,

    // --- synced metronome (see Metronome + drive_metronome) ---
    metro: Metronome,

    // --- in-app auto-update (checks GitHub Releases on launch) ---
    updater: update::Updater,

    // Whether the About window is open (toggled from the status bar).
    show_about: bool,

    // --- persisted user preferences (see prefs.rs + Edit ▸ Preferences) ---
    // Loaded on startup; every Preferences edit mutates this, applies live to
    // the relevant consumer, and saves. Consumers read from here at their use
    // sites (roll zoom, scrollback window, echo hold-off, mic tunables, …).
    prefs: prefs::Prefs,
    // Whether the Preferences window is open.
    show_prefs: bool,
    // Which Preferences tab the sidebar has selected.
    prefs_section: PrefsSection,

    // --- piano-roll history (see roll.rs + draw_roll) ---
    roll: roll::Roll,
    // Drag-to-review view state: `Some(t)` is the roll time rendered at the
    // strip's top edge while scrolled back (or animating home); `None` = live.
    scrollback: Option<f64>,
    // When the last drag/scroll input on the history roll stopped; the view
    // holds still until the scrollback-hold window elapses, then eases to live.
    scrollback_idle_since: Option<Instant>,
    // Falling-panel review state: a view-time offset from the playhead
    // (negative = looking at the past). Purely a rendering offset — never
    // touches `PlaybackEngine::playhead_s`, so Learn-mode gating is unaffected.
    falling_scrollback: Option<f64>,
    falling_scrollback_idle_since: Option<Instant>,
    // The history roll's time (`roll.now_s()`) at the moment the current file
    // was opened — i.e. where the falling panel's score-time zero sits on the
    // roll's absolute timeline. Purely a *label* offset (see `draw_ruler`): it
    // makes the two strips' time rulers read as one continuous timeline across
    // the keyboard, without touching the falling panel's own score-time-based
    // note positions, Learn-mode gating, or the roll's segment history.
    score_roll_origin_s: f64,
    // Result of the last save attempt, shown next to the File menu.
    roll_status: String,
    // Whether the "unsaved roll" confirmation is up (close was intercepted).
    show_close_confirm: bool,
    // Set when the unsaved-roll confirmation was raised by "Restart now" rather
    // than a window close, so the dialog restarts (instead of quitting) once the
    // roll is saved/discarded (F22).
    pending_restart: bool,
    // Set once the user confirms quitting, so the re-issued close passes the
    // interception even though the roll is still unsaved.
    allow_close: bool,

    // --- playback of a loaded score (see playback.rs; None = live mode) ---
    playback: Option<playback::PlaybackEngine>,
    // Per-channel volume/mute for the playback synth source, mirroring the
    // screen/peer pairs above.
    playback_volume: f32,
    playback_muted: bool,
    // Result of the last File > Open (load warnings/errors).
    open_status: String,
    // Whether the segment + key-range row is shown (checkbox in the playback
    // controls). The Learn side panel has its own collapse state below.
    segment_row_visible: bool,
    // Whether the top config panel is collapsed to its title strip (chevron).
    config_collapsed: bool,
    // User-dragged keyboard height as a fraction of the central panel's
    // height (None = the KEYBOARD_FRACTION default). Mirrors
    // `prefs.keyboard_height_frac`, which persists it on drag-stop.
    keyboard_height_frac: Option<f32>,
    // Compact mode ("alternate minimize" — keyboard + title bar only). This is
    // user *intent*, toggled by the title-bar button and persisted when
    // "Remember window state" is on.
    compact_mode: bool,
    // Window size to restore to when leaving compact mode, snapshotted on
    // entry (and persisted via `prefs.normal_window_size`, which also seeds it
    // at startup when the app launches already compact).
    normal_size: Option<egui::Vec2>,
    // What's actually been applied to the OS window right now. Reconciled once
    // per frame against `compact_mode` (see `sync_compact_viewport`), so
    // resizes fire only on real transitions.
    compact_applied: bool,
    // Whether the window is currently pinned always-on-top. Reconciled every
    // frame against `compact_mode && prefs.compact_always_on_top` (see
    // `sync_compact_viewport`), so a live pref toggle takes effect immediately.
    on_top_applied: bool,

    // --- custom-chrome window move/resize (manual, touch-compatible) ---
    // These drive the frameless window ourselves every frame from egui's drag
    // deltas, instead of handing off to Windows' native SC_MOVE/SC_SIZE modal
    // loop (`ViewportCommand::StartDrag`/`BeginResize`) — that native loop is
    // built around real mouse-button state and does not sustain a
    // touch-originated gesture, which is why touch move was broken.
    //
    // Grab offset for an active title-bar move drag: the pointer's position
    // *relative to the window's client origin* at the moment the drag started,
    // held constant for the whole gesture. Each frame we solve the window
    // origin absolutely (`outer_rect.min + pointer_local - grab`) so the grab
    // point stays pinned under the pointer. Absolute (not delta-accumulated),
    // so re-issuing the same command is idempotent and a one-frame lag in the
    // reported outer rect self-corrects instead of feeding the window's own
    // motion back in as jitter. `None` when not moving.
    titlebar_drag: Option<egui::Pos2>,
    // Target (outer position, inner size) accumulated across an active
    // edge/corner resize drag; `None` when not resizing.
    resize_drag: Option<(egui::Pos2, egui::Vec2)>,
    // Whether recent input came from touch (vs. mouse/trackpad). Drives the
    // enlarged, visible resize-handle affordance so it only appears in actual
    // tablet use. Updated once per frame from raw egui events; left unchanged
    // on idle frames so it tracks "most recently used," not flickers.
    touch_mode: bool,
    // Whether the Learn side panel is expanded ("‹"/"›" arrows). Preserved
    // across Learn-mode exits, so re-entering restores the last choice.
    learn_panel_expanded: bool,
    // Same, for the Evaluation/review side panel.
    evaluation_panel_expanded: bool,
    // Whether the "Evaluation results" window is up — set on the take-finished
    // transition (same pattern as show_close_confirm/show_refine_range).
    show_eval_results: bool,
    // Right-click time stash: a `context_menu` closure runs on frames after
    // the opening click, so the clicked time must be captured when
    // `secondary_clicked()` fires, not re-derived inside the menu.
    pending_break_t: Option<f64>,
    // In-progress key-range drag on the falling panel: (start x, current x).
    range_drag: Option<(f32, f32)>,
    // "Refine range" dialog state: visibility + the two editable key names.
    show_refine_range: bool,
    refine_lo: String,
    refine_hi: String,
    // Persistent buffers for the two rename-in-place fields. A TextEdit's
    // backing string must survive across frames while focused, or typed
    // characters are wiped on the next repaint; these mirror the app state
    // while idle and only commit on lost focus.
    instance_edit: String,
    segment_edit: String,
    // Which segment `segment_edit` is editing — pinned on focus so a moving
    // playhead can't silently redirect the rename to a different segment.
    segment_edit_idx: usize,
}

impl PianoApp {
    fn new(prefs: prefs::Prefs) -> Self {
        // Preferences drive the startup seeds below (and the input backend's
        // initial tunables). Loaded by `main()` — which also sized the OS
        // window from them — and passed in.
        let input = input::start(
            prefs.threshold,
            audio::InferenceTunables::new(prefs.silence_rms, prefs.norm_max_gain, prefs.frame_off),
            prefs.midi_poll_ms,
        );
        let mut roll = roll::Roll::new();
        roll.set_timing(
            Duration::from_secs_f64(prefs.section_tail_s.max(0.0)),
            Duration::from_secs_f64(prefs.section_lead_in_s.max(0.0)),
            prefs.idle_pause.as_duration(),
        );
        let mut metro = Metronome::new(prefs.metro_bpm);
        metro.beats_per_bar = prefs.metro_beats_per_bar;
        metro.beat_freqs = prefs.metro_beat_freqs.clone();
        metro.beat_volumes = prefs.metro_beat_volumes.clone();
        // Mirror `main()`'s startup decision: the OS window was already created
        // compact when this is true, so `compact_applied` MUST be seeded equal
        // to `compact_mode` — seeding `false` would make frame 1 see a spurious
        // transition and snapshot the already-compact rect as "normal".
        let compact_mode = prefs.remember_window_state && prefs.compact_mode;
        // Seed the normal-size restore target from the last session, so a
        // launch directly into compact mode still knows what "normal" is —
        // without this, leaving compact fell back to the default height while
        // keeping the live width (the disproportionate-window bug).
        let normal_size = prefs.normal_window_size.map(|[w, h]| egui::vec2(w, h));
        let reopen_path = prefs
            .reopen_last_file
            .then(|| prefs.last_file_path.clone())
            .flatten();
        let keyboard_height_frac = prefs.keyboard_height_frac;
        let mut app = Self {
            input,
            synth: synth::Synth::start(),
            screen_volume: 1.0,
            screen_muted: false,
            peer_volume: 1.0,
            peer_muted: false,
            threshold: prefs.threshold,
            mic_muted: prefs.mic_muted,
            notes_epoch: 0,
            was_midi: false,
            local: [false; KEY_COUNT],
            remote: [false; KEY_COUNT],
            held: [false; KEY_COUNT],
            remote_held: [false; KEY_COUNT],
            last_held_sent: [0; HELD_MASK_BYTES],
            last_live_sent: [0; HELD_MASK_BYTES],
            live_seq: 0,
            held_seq: 0,
            remote_live_seq: 0,
            remote_held_seq: 0,
            held_cmd_down: false,
            local_pedal: 0,
            remote_pedal: 0,
            last_pedal_sent: 0,
            frame_onsets: Vec::new(),
            echo_held: [[false; 2]; 128],
            echo_until: [None; 128],
            mouse_note: None,
            pointer_still_since: Instant::now(),
            last_pointer_pos: None,
            local_color: prefs.local_color,
            remote_color: DEFAULT_REMOTE_COLOR,
            last_color_send: Instant::now(),
            local_name: prefs.local_name.clone(),
            remote_name: DEFAULT_REMOTE_NAME.to_string(),
            my_ticket: None,
            join_ticket: String::new(),
            peer: None,
            net_status: "Not connected".to_string(),
            is_host: false,
            peer_connected: false,
            last_key_range: None,
            prefs_save_due: None,
            manual_separators: Vec::new(),
            metro,
            // Kick off the background GitHub Releases check; the UI polls its
            // state each frame (see `update_controls`).
            updater: update::start(),
            show_about: false,
            prefs,
            show_prefs: false,
            prefs_section: PrefsSection::StartupWindow,
            roll,
            scrollback: None,
            scrollback_idle_since: None,
            falling_scrollback: None,
            falling_scrollback_idle_since: None,
            score_roll_origin_s: 0.0,
            roll_status: String::new(),
            show_close_confirm: false,
            pending_restart: false,
            allow_close: false,
            playback: None,
            playback_volume: 1.0,
            playback_muted: false,
            open_status: String::new(),
            segment_row_visible: true,
            config_collapsed: false,
            keyboard_height_frac,
            compact_mode,
            normal_size,
            compact_applied: compact_mode,
            on_top_applied: false,
            titlebar_drag: None,
            resize_drag: None,
            touch_mode: false,
            learn_panel_expanded: true,
            evaluation_panel_expanded: true,
            show_eval_results: false,
            pending_break_t: None,
            range_drag: None,
            show_refine_range: false,
            refine_lo: String::new(),
            refine_hi: String::new(),
            instance_edit: String::new(),
            segment_edit: String::new(),
            segment_edit_idx: 0,
        };
        // Startup reopen (opt-in): reload the last-opened score. A moved/deleted
        // file falls through to `load_score_path`'s non-panicking "open failed"
        // status — startup never blocks on a stale path.
        if let Some(path) = reopen_path {
            app.load_score_path(path);
        }
        app
    }

    /// Schedule a debounced preferences save (see `PREFS_SAVE_DEBOUNCE`). Use
    /// from continuous controls (sliders / drag-values) whose `.changed()` fires
    /// every frame of a drag, instead of the immediate `self.prefs.save()` — that
    /// would hammer the disk on the GUI thread ~60×/s (M7).
    fn save_prefs_soon(&mut self) {
        self.prefs_save_due = Some(Instant::now() + PREFS_SAVE_DEBOUNCE);
    }

    /// Flush a due debounced prefs save. Called once per frame from `update`.
    fn flush_prefs_save(&mut self) {
        if let Some(due) = self.prefs_save_due {
            if Instant::now() >= due {
                self.prefs.save();
                self.prefs_save_due = None;
            }
        }
    }

    /// Send our chosen color to the peer (if connected) and reset the heartbeat.
    fn send_color(&mut self) {
        if let Some(peer) = &self.peer {
            peer.send(Packet::Color(self.local_color));
        }
        self.last_color_send = Instant::now();
    }

    /// Send our display name to the peer (if connected). Rides the color
    /// heartbeat, so a dropped datagram self-heals within a second.
    fn send_name(&self) {
        if let Some(peer) = &self.peer {
            peer.send(Packet::Name(self.local_name.clone()));
        }
    }

    /// Broadcast our per-beat click pitch/volume tables to the peer, so both
    /// sides' clicks sound identical (see `Packet::MetroBeatTable`). Sent
    /// immediately on every Preferences edit, on connect, and alongside the
    /// color heartbeat so a dropped datagram doesn't leave the two sides
    /// mismatched for long.
    fn send_metro_table(&self) {
        if let Some(peer) = &self.peer {
            peer.send(Packet::MetroBeatTable {
                freqs: self.metro.beat_freqs.clone(),
                volumes: self.metro.beat_volumes.clone(),
            });
        }
    }

    /// Broadcast our Ctrl+click-pinned key set to the peer as a whole-state
    /// snapshot, so the pinned chord lights up on both screens (see
    /// `Packet::Held`). `force` re-sends even if the mask is unchanged — used
    /// by the color heartbeat so a dropped snapshot self-heals while keys stay
    /// pinned. `last_held_sent` is updated only when a peer actually exists, so
    /// a set pinned while disconnected is still announced once a session comes
    /// up (mirrors the pedal send-on-change).
    fn broadcast_held(&mut self, force: bool) {
        let mask = pack_held(&self.held);
        if !force && mask == self.last_held_sent {
            return;
        }
        self.held_seq = self.held_seq.wrapping_add(1);
        let seq = self.held_seq;
        if let Some(peer) = &self.peer {
            peer.send(Packet::Held { seq, mask });
            self.last_held_sent = mask;
        }
    }

    /// Forward a local note transition to the peer, stamped with the next
    /// note sequence number (F6). The bump happens whether or not a peer exists
    /// so the counter stays a faithful running index of local note events; the
    /// [`Packet::Live`] snapshot then carries the latest value.
    fn send_note(&mut self, msg: NoteMsg) {
        self.live_seq = self.live_seq.wrapping_add(1);
        let seq = self.live_seq;
        if let Some(peer) = &self.peer {
            peer.send(Packet::Note(msg, seq));
        }
    }

    /// Insert a manual segment break locally and share it with the peer so the
    /// break shows on both screens (M2). The full manual-break list is re-sent
    /// on the heartbeat, so a dropped datagram converges.
    fn insert_manual_separator(&mut self, at: f64) {
        self.roll.insert_separator(at);
        // Track the clamped time the roll actually used, deduped, so our
        // snapshot matches what we rendered.
        let at = at.clamp(0.0, self.roll.now_s());
        if at > 0.0 && !self.manual_separators.iter().any(|&x| (x - at).abs() < 1e-9) {
            self.manual_separators.push(at);
        }
        self.broadcast_separators();
    }

    /// Broadcast our manual segment breaks to the peer (see `manual_separators`).
    /// Reconciles the broadcast list against the roll first: the idle snap-back
    /// (`Roll::tick`) trims breaks that fall into rewound dead air, and a break
    /// that's gone from our own roll must stop riding the heartbeat — otherwise
    /// the heartbeat keeps *re-creating* it on the peer, so the originator shows
    /// no break while the peer shows one forever (F1). Capped so an oversized
    /// snapshot can't silently blow the datagram MTU and stop sync (F21).
    fn broadcast_separators(&mut self) {
        self.manual_separators.retain(|&t| self.roll.has_separator_at(t));
        if self.manual_separators.len() > MAX_WIRE_SEPARATORS {
            self.manual_separators.truncate(MAX_WIRE_SEPARATORS);
        }
        if self.manual_separators.is_empty() {
            return;
        }
        if let Some(peer) = &self.peer {
            peer.send(Packet::Separators(self.manual_separators.clone()));
        }
    }

    /// Start hosting a session. Replaces any existing session (dropping the
    /// old `Peer` shuts its net thread down); the invite code arrives async
    /// as a `NetEvent::Ticket`.
    fn host(&mut self) {
        self.my_ticket = None;
        self.clear_remote_keys();
        // Fresh session: don't feed a brand-new peer the manual breaks of a
        // previous, unrelated session (F1). The breaks stay on our own roll;
        // we just stop broadcasting the stale list.
        self.manual_separators.clear();
        self.net_status = "Starting…".into();
        self.is_host = true;
        self.peer_connected = false;
        // Fresh session: re-anchor the metronome (we're the authority now).
        self.metro.next_beat_at = None;
        self.peer = Some(net::host());
    }

    /// Join a host from the pasted invite code. Progress and errors (bad
    /// code, unreachable host) come back as `NetEvent::Status`.
    fn join(&mut self) {
        let code = self.join_ticket.trim().to_string();
        if code.is_empty() {
            self.net_status = "Paste an invite code first".into();
            return;
        }
        self.my_ticket = None;
        self.clear_remote_keys();
        // Fresh session: don't leak a previous session's manual breaks (F1).
        self.manual_separators.clear();
        self.net_status = "Joining…".into();
        self.is_host = false;
        self.peer_connected = false;
        // As a follower we no longer own the grid; wait for the host's markers.
        self.metro.next_beat_at = None;
        self.peer = Some(net::join(code));
    }

    /// Unlight every remote key and stop the synth voicing it. Needed whenever
    /// remote state becomes unknown (connect, disconnect, new session): the
    /// matching note-offs will never arrive, so keys — and synth voices —
    /// would otherwise be stuck on.
    fn clear_remote_keys(&mut self) {
        for idx in 0..KEY_COUNT {
            if self.remote[idx] {
                self.remote[idx] = false;
                self.synth_note_off(MIDI_LOW + idx as u8, synth::Channel::Peer);
            }
        }
        self.roll.release_all(roll::Who::Remote);
        // The matching pedal-up will never arrive either.
        self.roll.release_pedal(roll::Who::Remote);
        self.remote_pedal = 0;
        // Drop the peer's pinned keys (display-only, no synth); a fresh session
        // starts with an empty shared overlay. Reset `last_held_sent` too so a
        // set we're still holding is re-announced to the new peer.
        self.remote_held = [false; KEY_COUNT];
        self.last_held_sent = [0; HELD_MASK_BYTES];
        // Same for the live-note snapshot: force a fresh re-announce of whatever
        // we're currently holding to the new peer (H1).
        self.last_live_sent = [0; HELD_MASK_BYTES];
        // Reset the applied-seq high-water marks: a fresh/rejoined peer starts
        // its own seq counter low, so we must accept its first packets (F6).
        self.remote_live_seq = 0;
        self.remote_held_seq = 0;
        // And the pedal: reset the send-on-change latch (parallel to
        // `last_held_sent`) so a pedal held at a constant level is re-announced
        // to a fresh/rejoined peer rather than staying silent (M1).
        self.last_pedal_sent = 0;
    }

    /// Reconcile the peer's live (sounding) notes from a whole-state snapshot
    /// (`Packet::Live`): light/extinguish only the keys that actually differ, so
    /// a dropped note-on/off datagram self-heals on the next heartbeat instead
    /// of leaving a remote key (and its synth voice) stuck (H1). Velocity isn't
    /// carried by the snapshot — a note first *seen* here uses the placeholder,
    /// which the real (velocity-bearing) note-on normally supplied already.
    fn reconcile_remote_live(&mut self, mask: [u8; HELD_MASK_BYTES]) {
        let want = unpack_held(&mask);
        for idx in 0..KEY_COUNT {
            let midi = MIDI_LOW + idx as u8;
            if want[idx] && !self.remote[idx] {
                self.remote[idx] = true;
                let msg = NoteMsg::On(midi, note::DEFAULT_VELOCITY);
                self.roll.note(roll::Who::Remote, msg, self.remote_color);
                self.play_synth(msg, synth::Channel::Peer);
            } else if !want[idx] && self.remote[idx] {
                self.remote[idx] = false;
                let msg = NoteMsg::Off(midi);
                self.roll.note(roll::Who::Remote, msg, self.remote_color);
                self.play_synth(msg, synth::Channel::Peer);
            }
        }
    }

    /// Broadcast our live (sounding) note set to the peer as an idempotent
    /// whole-state snapshot (`Packet::Live`) — the self-heal companion to the
    /// per-event note datagrams (H1). `force` re-sends even when unchanged (used
    /// by the heartbeat). `last_live_sent` is updated only when a peer exists.
    fn broadcast_live(&mut self, force: bool) {
        let mask = pack_held(&self.local);
        if !force && mask == self.last_live_sent {
            return;
        }
        let seq = self.live_seq;
        if let Some(peer) = &self.peer {
            peer.send(Packet::Live { seq, mask });
            self.last_live_sent = mask;
        }
    }

    /// Drain the input channel (MIDI or mic, whichever is active): update local
    /// (red) keys and forward each event to the remote peer.
    fn pump_input(&mut self) {
        // If the backend just switched (e.g. a MIDI cable was unplugged), force
        // every locally-held note off so nothing stays stuck lit, and tell the
        // peer. A yanked MIDI cable sends no Note Offs, so this is required.
        let epoch = self.input.switch_epoch();
        if epoch != self.notes_epoch {
            self.notes_epoch = epoch;
            // The mouse-held note (if any) is among the local keys cleared below;
            // silence it on the synth and forget it so the next press is clean.
            if let Some(m) = self.mouse_note.take() {
                self.synth_note_off(m, synth::Channel::Local);
            }
            for idx in 0..KEY_COUNT {
                if self.local[idx] {
                    self.local[idx] = false;
                    self.send_note(NoteMsg::Off(MIDI_LOW + idx as u8));
                }
            }
            // The matching note-offs will never arrive, so close the roll's
            // open marks too — and the pedal's (a yanked MIDI cable sends no
            // CC64 release either). The peer is told via the send-on-change
            // block below, which sees `local_pedal` snap to 0.
            self.roll.release_all(roll::Who::Local);
            self.roll.release_pedal(roll::Who::Local);
            self.local_pedal = 0;
        }

        // Only the microphone backend can hear the synth bleed back; a MIDI piano
        // makes its own sound and the mic isn't even running, so we never gate
        // genuine MIDI events.
        let mic_source = self.input.source() == Source::Microphone;
        while let Ok(msg) = self.input.notes.try_recv() {
            // Drop any mic transition for a note the synth is voicing (or just
            // stopped voicing): it's our own output echoing back, not a real
            // play. Both the On and the trailing Off are dropped so a note we
            // light from the mouse/peer isn't torn down by the echo's Off.
            if mic_source && self.echo_suppressed(msg.midi()) {
                continue;
            }
            // Muted mic: drop new onsets but let offs through, so anything
            // already sounding when the mute flipped on closes normally.
            if mic_source && self.mic_muted && matches!(msg, NoteMsg::On(..)) {
                continue;
            }
            apply(&mut self.local, msg);
            if let NoteMsg::On(midi, velocity) = msg {
                // Velocity only exists at the instant of attack; capture it
                // here for the evaluation scorer (see `frame_onsets`).
                self.frame_onsets.push((midi, velocity));
            }
            self.roll.note(roll::Who::Local, msg, self.local_color);
            self.send_note(msg);
        }

        // Sustain pedal (CC64). Only the MIDI backend is wired to this channel
        // (see input.rs), so nothing arrives on the mic fallback by
        // construction. Deliberately parallel to — never inside — the note
        // path: `Roll::pedal` must not reset the roll's idle timer.
        while let Ok(level) = self.input.pedal.try_recv() {
            if level == self.local_pedal {
                continue;
            }
            // Sensitivity deadzone (`prefs.pedal_deadzone`): mid-travel wiggles
            // smaller than the threshold don't register, settling jittery
            // analog pedals without discarding deliberate half-pedal moves.
            // Transitions to/from fully released always pass — eating a span's
            // open/close edge would leave the pedal stuck down. Local capture
            // only: remote levels arrive already filtered by the peer's own
            // deadzone, like the other per-player capture tunables.
            let deadzone = self.prefs.pedal_deadzone as i16;
            if level != 0
                && self.local_pedal != 0
                && (level as i16 - self.local_pedal as i16).abs() < deadzone
            {
                continue;
            }
            self.local_pedal = level;
            self.roll.pedal(roll::Who::Local, level);
        }
        // Forward on change only. `last_pedal_sent` is updated only when a
        // peer actually exists, so a level reached while disconnected is still
        // announced the moment a session comes up.
        if self.local_pedal != self.last_pedal_sent {
            if let Some(peer) = &self.peer {
                peer.send(Packet::Pedal { level: self.local_pedal });
                self.last_pedal_sent = self.local_pedal;
            }
        }
    }

    /// Whether a mic-detected transition for `midi` should be ignored as the
    /// synth's own sound echoing back: true while the synth is voicing that note,
    /// or within the echo hold-off of it having stopped (covers the release tail).
    fn echo_suppressed(&self, midi: u8) -> bool {
        let n = midi as usize;
        if n >= 128 {
            return false;
        }
        if self.echo_held[n][0] || self.echo_held[n][1] {
            return true;
        }
        matches!(self.echo_until[n], Some(t) if Instant::now() < t)
    }

    /// Begin voicing `midi` on the synth and mark it as a live echo source so the
    /// mic won't re-detect it.
    fn synth_note_on(&mut self, midi: u8, channel: synth::Channel) {
        self.synth.note_on(midi, channel);
        if (midi as usize) < 128 {
            self.echo_held[midi as usize][channel as usize] = true;
        }
    }

    /// Stop voicing `midi` on the synth. If no channel is voicing it anymore,
    /// arm the echo hold-off so the mic ignores it while the tone rings out.
    fn synth_note_off(&mut self, midi: u8, channel: synth::Channel) {
        self.synth.note_off(midi, channel);
        let n = midi as usize;
        if n >= 128 {
            return;
        }
        let was = self.echo_held[n][channel as usize];
        self.echo_held[n][channel as usize] = false;
        if was && !self.echo_held[n][0] && !self.echo_held[n][1] {
            let holdoff = Duration::from_millis(self.prefs.echo_holdoff_ms);
            self.echo_until[n] = Some(Instant::now() + holdoff);
        }
    }

    /// Record toggle + live status for the training-data capture harness. A
    /// session records mic audio to a WAV and, whenever a MIDI device is
    /// connected, its note/pedal events to a JSONL with a shared timeline —
    /// exactly the aligned audio+label pairs needed to train the model.
    fn record_controls(&mut self, ui: &mut egui::Ui) {
        let rec = &self.input.recorder;
        let armed = rec.is_armed();
        let recording = rec.is_recording();

        let label = if armed { "■ Stop recording" } else { "● Record training data" };
        let color = if armed {
            egui::Color32::from_rgb(220, 60, 60)
        } else {
            egui::Color32::from_gray(225)
        };
        if ui
            .add(egui::Button::new(egui::RichText::new(label).color(color)))
            .on_hover_text("Capture mic audio + MIDI labels to recordings/session_*")
            .clicked()
        {
            rec.set_armed(!armed);
        }

        if recording {
            ui.colored_label(egui::Color32::from_rgb(220, 60, 60), "● REC");
            let midi_n = rec.midi_event_count();
            let secs = rec.audio_seconds();
            ui.label(format!("{secs:.1}s audio · {midi_n} MIDI events"));
            if midi_n == 0 {
                ui.colored_label(
                    egui::Color32::from_rgb(210, 170, 60),
                    "(no MIDI device — audio only; connect a piano for labels)",
                );
            }
            ui.weak(format!("→ {}", rec.session_dir()));
            // Surface a setup/write failure instead of showing a healthy "REC"
            // over a session that isn't actually being written (L18).
            let err = rec.error();
            if !err.is_empty() {
                ui.colored_label(egui::Color32::from_rgb(220, 80, 80), format!("⚠ {err}"));
            }
        } else if armed {
            ui.label("Starting…");
        }
    }

    /// The File menu (save/open the piano roll), now just the menu button —
    /// it lives in the custom title bar's `menu::bar` (see `title_bar`). The
    /// "unsaved" chip rides alongside it there; the instance-rename field and
    /// last save/open results moved to `roll_status_row` in the config panel.
    /// Save As… lets the user pick between the interoperable MIDI export and
    /// the self-contained JSONL one (see roll.rs).
    fn file_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("File", |ui| {
            let save = egui::Button::new("Save roll")
                .shortcut_text(ui.ctx().format_shortcut(&SAVE_SHORTCUT));
            if ui
                .add_enabled(!self.roll.is_empty(), save)
                .on_hover_text("Write the roll to rolls/roll_<time>.mid (+ .json colors)")
                .clicked()
            {
                ui.close_menu();
                self.save_roll_quick();
            }
            if ui
                .add_enabled(!self.roll.is_empty(), egui::Button::new("Save roll as…"))
                .on_hover_text("Choose where to save, as MIDI (+ colors) or JSONL")
                .clicked()
            {
                ui.close_menu();
                self.save_roll_as();
            }
            ui.separator();
            if ui
                .button("Open…")
                .on_hover_text("Load a saved roll (.mid or .jsonl) for playback / practice")
                .clicked()
            {
                ui.close_menu();
                self.open_score();
            }
            if self.playback.is_some() && ui.button("Close file").clicked() {
                ui.close_menu();
                if let Some(pb) = &mut self.playback {
                    pb.silence(&self.synth);
                }
                self.playback = None;
                self.open_status.clear();
                self.falling_scrollback = None;
                self.falling_scrollback_idle_since = None;
                self.score_roll_origin_s = 0.0;
                // Explicit close is a deliberate "forget this file" signal —
                // don't reopen it on the next launch.
                if self.prefs.last_file_path.is_some() {
                    self.prefs.last_file_path = None;
                    self.prefs.save();
                }
            }
        });
    }

    /// The "● unsaved" chip. Rendered next to the File menu in the title bar so
    /// it's visible regardless of the config panel's collapse state.
    fn unsaved_chip(&self, ui: &mut egui::Ui) {
        if self.roll.has_unsaved() {
            ui.colored_label(egui::Color32::from_rgb(210, 170, 60), "● unsaved")
                .on_hover_text("The roll has notes that haven't been saved (File ▸ Save)");
        }
    }

    /// Instance rename field + last save/open results. Lives in the config
    /// panel (below the title bar), where there's room for the text field and
    /// the potentially-long status paths.
    fn roll_status_row(&mut self, ui: &mut egui::Ui) {
        // Rename whichever instance the live roll is currently in — the name
        // is baked into the file's markers on the next save.
        if !self.roll.is_empty() {
            ui.label("Instance:");
            let current = self.roll.current_instance_name();
            let edit =
                ui.add(egui::TextEdit::singleline(&mut self.instance_edit).desired_width(120.0));
            if edit.lost_focus() {
                if !self.instance_edit.is_empty() && self.instance_edit != current {
                    self.roll.rename_current_instance(self.instance_edit.clone());
                }
            } else if !edit.has_focus() {
                // Idle: mirror the app state (the current instance can change
                // under the field as the clock runs).
                self.instance_edit = current;
            }
        }
        if !self.roll_status.is_empty() {
            ui.weak(&self.roll_status);
        }
        if !self.open_status.is_empty() {
            ui.weak(&self.open_status);
        }
    }

    /// File > Open…: load a `.mid`/`.jsonl` into a playback engine. Playback
    /// and a live P2P session are mutually exclusive — a real peer's notes
    /// colliding with a score's "Remote" track visualization would be
    /// nonsense, so opening a file drops any session.
    fn open_score(&mut self) {
        std::fs::create_dir_all("rolls").ok();
        let start_dir = std::env::current_dir().unwrap_or_default().join("rolls");
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Piano rolls", &["mid", "midi", "jsonl"])
            .set_directory(start_dir)
            .pick_file()
        else {
            return; // cancelled
        };
        self.load_score_path(path);
    }

    /// Load a score file into a playback engine — the dialog-free tail of
    /// [`open_score`], reused by the startup "reopen last file" path.
    fn load_score_path(&mut self, path: PathBuf) {
        match score::Score::load(&path) {
            Ok(s) => {
                self.peer = None;
                self.my_ticket = None;
                self.clear_remote_keys();
                self.net_status = "Not connected".to_string();
                if let Some(pb) = &mut self.playback {
                    pb.silence(&self.synth); // replacing an already-open file
                }
                self.open_status = match &s.warning {
                    Some(w) => format!("opened {} ({w})", path.display()),
                    None => format!("opened {}", path.display()),
                };
                self.playback = Some(playback::PlaybackEngine::new(s, path.clone()));
                // Align the two strips' timelines: score-time zero now maps to
                // wherever the history roll's clock currently sits, so the
                // ruler labels on both sides of the keyboard read as one
                // continuous timeline (see `score_roll_origin_s`'s docs).
                self.score_roll_origin_s = self.roll.now_s();
                self.scrollback = None;
                self.scrollback_idle_since = None;
                self.falling_scrollback = None;
                self.falling_scrollback_idle_since = None;
                self.range_drag = None;
                if self.prefs.reopen_last_file {
                    self.prefs.last_file_path = Some(path);
                    self.prefs.save();
                }
            }
            Err(e) => self.open_status = format!("open failed: {e}"),
        }
    }

    /// One-click save to `rolls/roll_<unix>.mid` (+ color sidecar). Returns
    /// whether it succeeded, so the unsaved-close dialog can gate quitting on
    /// the file actually being on disk.
    fn save_roll_quick(&mut self) -> bool {
        if self.roll.is_empty() {
            return false;
        }
        match roll::save_quick(&self.roll, self.local_color, self.remote_color) {
            Ok(path) => {
                self.roll.mark_saved();
                self.roll_status = format!("saved → {}", path.display());
                true
            }
            Err(e) => {
                self.roll_status = format!("save failed: {e}");
                false
            }
        }
    }

    /// "Save As…" via the native dialog. Blocking on the GUI thread is fine:
    /// the frame simply freezes while the dialog is up, which is standard for
    /// eframe apps on Windows. The chosen extension picks the format.
    fn save_roll_as(&mut self) {
        // Make sure the default directory exists so the dialog lands there.
        std::fs::create_dir_all("rolls").ok();
        let start_dir = std::env::current_dir().unwrap_or_default().join("rolls");
        let Some(path) = rfd::FileDialog::new()
            .add_filter("MIDI + color sidecar", &["mid"])
            .add_filter("Piano-roll JSONL", &["jsonl"])
            .set_directory(start_dir)
            .set_file_name("roll")
            .save_file()
        else {
            return; // cancelled
        };
        let result = match path.extension().and_then(|e| e.to_str()) {
            Some("jsonl") => roll::save_jsonl(&self.roll, self.local_color, self.remote_color, &path),
            _ => roll::save_midi(&self.roll, self.local_color, self.remote_color, &path),
        };
        match result {
            Ok(()) => {
                self.roll.mark_saved();
                self.roll_status = format!("saved → {}", path.display());
            }
            Err(e) => self.roll_status = format!("save failed: {e}"),
        }
    }

    /// The "unsaved roll" confirmation shown when the window ✕ is clicked with
    /// unsaved notes. Deliberately has no titlebar ✕ (`.open()`): only the
    /// three buttons resolve it. A failed save stays open with the error
    /// visible rather than losing the roll.
    fn unsaved_dialog(&mut self, ctx: &egui::Context) {
        if !self.show_close_confirm {
            return;
        }
        egui::Window::new("Unsaved piano roll")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label("The piano roll has notes that haven't been saved.");
                if !self.roll_status.is_empty() {
                    ui.weak(&self.roll_status);
                }
                ui.add_space(6.0);
                // The same dialog serves a window close and a "Restart now"
                // (F22); word the confirm buttons for whichever is pending.
                let restart = self.pending_restart;
                let (save_label, discard_label) = if restart {
                    ("Save and restart", "Restart without saving")
                } else {
                    ("Save and quit", "Quit without saving")
                };
                ui.horizontal(|ui| {
                    if ui.button(save_label).clicked() && self.save_roll_quick() {
                        // Synchronous save: the file is on disk before the
                        // process is allowed to die. On failure we fall
                        // through with the error shown above.
                        if restart {
                            self.perform_restart();
                        }
                        self.allow_close = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    if ui.button(discard_label).clicked() {
                        if restart {
                            self.perform_restart();
                        }
                        self.allow_close = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    if ui.button("Cancel").clicked() {
                        self.show_close_confirm = false;
                        self.pending_restart = false;
                    }
                });
            });
    }

    /// Transport + mode + speed for the loaded score. Always visible while a
    /// file is open (core playback UI, unlike the hideable settings panels).
    /// Glyph scheme: barred ⏮/⏭ act on the whole piece, double-triangle
    /// ⏪/⏩ on one segment, and play/pause flips ▶/⏸ with state.
    fn playback_controls(&mut self, ui: &mut egui::Ui) {
        // Read before `self.playback` is borrowed: evaluation freezes the
        // velocity/pedal dimensions from the live source at take start.
        let live_midi = self.input.source() == Source::Midi;
        let Some(pb) = &mut self.playback else { return };

        if ui.button("⏮").on_hover_text("Jump to the start of the piece").clicked() {
            pb.jump_to(0.0, &self.synth);
        }
        if ui
            .button("⏪")
            .on_hover_text("Restart this segment (press again right away for the previous one)")
            .clicked()
        {
            pb.restart_or_previous(&self.synth);
        }
        let play_label = if pb.playing { "⏸" } else { "▶" };
        if ui
            .button(play_label)
            .on_hover_text(if pb.playing { "Pause" } else { "Play" })
            .clicked()
        {
            pb.set_playing(!pb.playing, &self.synth);
        }
        if ui.button("⏩").on_hover_text("Jump to the next segment").clicked() {
            pb.next_segment(&self.synth);
        }
        if ui.button("⏭").on_hover_text("Jump to the end of the piece").clicked() {
            let end = pb.score.duration_s;
            pb.jump_to(end, &self.synth);
        }

        ui.separator();
        // EvaluationReview is deliberately not a radio target: it's entered
        // automatically when a take finishes (while reviewing, none of the
        // three is selected — picking any exits the review).
        let prev_mode = pb.mode;
        ui.radio_value(&mut pb.mode, playback::Mode::Listen, "Listen");
        ui.radio_value(&mut pb.mode, playback::Mode::Learn, "Learn");
        ui.radio_value(&mut pb.mode, playback::Mode::Evaluation, "Evaluation")
            .on_hover_text("Play the piece through without stopping and get scored against it");
        if pb.mode != prev_mode {
            if pb.mode == playback::Mode::Evaluation {
                pb.start_evaluation(live_midi, &self.synth);
            } else if matches!(
                prev_mode,
                playback::Mode::Evaluation | playback::Mode::EvaluationReview
            ) {
                pb.exit_evaluation(&self.synth);
            }
        }
        if pb.mode == playback::Mode::EvaluationReview {
            ui.weak("— reviewing evaluation results");
            if ui
                .button("Retake")
                .on_hover_text("Discard this review and run the evaluation again")
                .clicked()
            {
                pb.start_evaluation(live_midi, &self.synth);
            }
        }
        ui.separator();
        ui.add(egui::Slider::new(&mut pb.speed, 0.25..=2.0).text("speed"));
        ui.separator();
        let eye =
            if self.segment_row_visible { "Hide segment row" } else { "Show segment row" };
        ui.checkbox(&mut self.segment_row_visible, eye)
            .on_hover_text("Show/hide the segment + key-range row");
        if pb.finished {
            ui.weak("— finished");
        }
    }

    /// The current segment's name (editable, persisted to the sidecar) and
    /// its loop controls. Shown in both modes — looping a passage is as
    /// useful for listening as for practicing.
    fn segment_controls(&mut self, ui: &mut egui::Ui) {
        let Some(pb) = &mut self.playback else { return };
        let idx = pb.current_segment_index();

        ui.label("Segment:");
        let edit =
            ui.add(egui::TextEdit::singleline(&mut self.segment_edit).desired_width(140.0));
        if edit.lost_focus() {
            // Commit to the segment that was pinned when editing began — the
            // playhead may have moved on since.
            let i = self.segment_edit_idx.min(pb.score.segments.len() - 1);
            if !self.segment_edit.is_empty() && self.segment_edit != pb.score.segments[i].name {
                pb.score.segments[i].name = self.segment_edit.clone();
                if let Err(e) = pb.score.save_segment_sidecar(&pb.source_path) {
                    self.open_status = format!("couldn't save segment names: {e}");
                }
            }
        } else if !edit.has_focus() {
            // Idle: follow the playhead's current segment.
            self.segment_edit = pb.score.segments[idx].name.clone();
            self.segment_edit_idx = idx;
        }
        ui.weak(format!("({}/{})", idx + 1, pb.score.segments.len()));

        ui.separator();
        // Scoring a looped partial pass isn't well-defined, so looping is
        // locked out during an evaluation take (start_evaluation also forces
        // it off).
        ui.add_enabled(
            pb.mode != playback::Mode::Evaluation,
            egui::Checkbox::new(&mut pb.loop_state.enabled, "Loop this segment"),
        )
        .on_disabled_hover_text("Looping is disabled while an evaluation take is running");
        if pb.loop_state.enabled {
            let mut finite = pb.loop_state.remaining.is_some();
            if ui
                .checkbox(&mut finite, "Limit repeats")
                .on_hover_text("Repeats *after* the pass currently playing")
                .changed()
            {
                pb.loop_state.remaining = if finite { Some(3) } else { None };
            }
            if let Some(n) = &mut pb.loop_state.remaining {
                ui.add(egui::DragValue::new(n).range(1..=99));
            }
            match pb.loop_state.pad_left_s {
                Some(left) => {
                    ui.weak(format!("looping in {left:.1}s"));
                }
                None => {
                    ui.weak(match pb.loop_state.remaining {
                        Some(n) => format!("🔁 {n} more"),
                        None => "🔁 ∞".into(),
                    });
                }
            }
        }
    }

    /// Learn-mode settings side panel: which tracks to practice and how
    /// strict the gating is (the key-range readout lives in the top config
    /// panel — see `key_range_panel`). Must be shown *before* the
    /// CentralPanel each frame (egui reserves panel space in show order).
    fn learn_panel(&mut self, ctx: &egui::Context) {
        let Some(pb) = &mut self.playback else { return };
        if pb.mode != playback::Mode::Learn {
            return;
        }

        // Collapsed/expanded variants animate between two *distinct* panel
        // ids (see the config panel's note in `update`). `pb` mutably borrows
        // `self.playback` for the whole closure, so the arrow click is
        // returned out of the closure and applied to `self` afterwards
        // instead of toggling in place.
        let collapsed = egui::SidePanel::right("learn_panel_collapsed")
            .resizable(false)
            .exact_width(18.0);
        let expanded = egui::SidePanel::right("learn_panel")
            .resizable(false)
            .default_width(220.0);
        let result = egui::SidePanel::show_animated_between(
            ctx,
            self.learn_panel_expanded,
            collapsed,
            expanded,
            |ui, how_expanded| -> bool {
                if how_expanded < 0.5 {
                    ui.vertical_centered(|ui| {
                        ui.add_space(6.0);
                        ui.small_button("‹").on_hover_text("Show Learn settings").clicked()
                    })
                    .inner
                } else {
                    let mut toggled = false;
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        toggled = ui
                            .small_button("›")
                            .on_hover_text("Hide Learn settings")
                            .clicked();
                        ui.strong("Learn settings");
                    });
                    ui.add_space(4.0);
                    ui.checkbox(&mut pb.learn.practice[0], "Practice Local track");
                    ui.checkbox(&mut pb.learn.practice[1], "Practice Remote track");
                    if !pb.learn.practice[0] && !pb.learn.practice[1] {
                        ui.weak("No track selected — behaves like Listen mode");
                    }
                    for (i, label) in [(0, "Local"), (1, "Remote")] {
                        if pb.learn.practice[i] && pb.score.tracks[i].notes.is_empty() {
                            ui.weak(format!("({label} track has no notes)"));
                        }
                    }
                    ui.separator();
                    ui.checkbox(&mut pb.learn.require_hold, "Require holding notes")
                        .on_hover_text(
                            "Off = wait for the right notes, then continue even if you release early",
                        );
                    ui.checkbox(&mut pb.learn.block_wrong, "Block on wrong notes")
                        .on_hover_text("Extra held keys also freeze playback");
                    toggled
                }
            },
        );
        if let Some(inner) = result {
            if inner.inner {
                self.learn_panel_expanded = !self.learn_panel_expanded;
            }
        }
    }

    /// Evaluation settings / review side panel — the Evaluation counterpart
    /// of `learn_panel` (same collapse/expand pair), shown in both the live
    /// take and the post-take review with different inner content. Any
    /// settings edit during a take restarts it from the top (a half-scored
    /// pass under changed rules isn't meaningful).
    fn evaluation_panel(&mut self, ctx: &egui::Context) {
        let live_midi = self.input.source() == Source::Midi;
        let synth = &self.synth;
        let Some(pb) = &mut self.playback else { return };
        if !matches!(pb.mode, playback::Mode::Evaluation | playback::Mode::EvaluationReview) {
            return;
        }

        let collapsed = egui::SidePanel::right("evaluation_panel_collapsed")
            .resizable(false)
            .exact_width(18.0);
        let expanded = egui::SidePanel::right("evaluation_panel")
            .resizable(false)
            .default_width(230.0);
        let result = egui::SidePanel::show_animated_between(
            ctx,
            self.evaluation_panel_expanded,
            collapsed,
            expanded,
            |ui, how_expanded| -> bool {
                if how_expanded < 0.5 {
                    ui.vertical_centered(|ui| {
                        ui.add_space(6.0);
                        ui.small_button("‹").on_hover_text("Show Evaluation settings").clicked()
                    })
                    .inner
                } else {
                    let mut toggled = false;
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        toggled = ui
                            .small_button("›")
                            .on_hover_text("Hide Evaluation settings")
                            .clicked();
                        ui.strong(if pb.mode == playback::Mode::Evaluation {
                            "Evaluation settings"
                        } else {
                            "Evaluation review"
                        });
                    });
                    ui.add_space(4.0);
                    if pb.mode == playback::Mode::Evaluation {
                        evaluation_settings_body(ui, pb, live_midi, synth);
                    } else {
                        review_settings_body(ui, pb, live_midi, synth);
                    }
                    toggled
                }
            },
        );
        if let Some(inner) = result {
            if inner.inner {
                self.evaluation_panel_expanded = !self.evaluation_panel_expanded;
            }
        }
    }

    /// The "Evaluation results" window, popped automatically when a take
    /// finishes; "Close" leaves the review browsable, "Retake" goes again.
    fn eval_results_window(&mut self, ctx: &egui::Context) {
        if !self.show_eval_results {
            return;
        }
        let live_midi = self.input.source() == Source::Midi;
        let synth = &self.synth;
        let Some(pb) = &mut self.playback else {
            self.show_eval_results = false;
            return;
        };
        if pb.eval_result.is_none() {
            self.show_eval_results = false;
            return;
        }
        let mut open = true;
        let mut close = false;
        let mut retake = false;
        egui::Window::new("Evaluation results")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                if let Some(result) = &pb.eval_result {
                    eval_results_body(ui, result);
                }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui
                        .button("Close")
                        .on_hover_text("Keep browsing the review (the breakdown stays in the side panel)")
                        .clicked()
                    {
                        close = true;
                    }
                    if ui.button("Retake").clicked() {
                        retake = true;
                    }
                });
            });
        if retake {
            pb.start_evaluation(live_midi, synth);
        }
        if !open || close || retake {
            self.show_eval_results = false;
        }
    }

    /// Key-range readout + clear button. Lives in the top config panel, not
    /// the Learn-only side panel: the drag-to-select gesture on the falling
    /// panel works in any mode, and the range now filters what's audible
    /// everywhere (see `drive_auto`), so the readout must be visible in
    /// Listen mode too.
    fn key_range_panel(&mut self, ui: &mut egui::Ui) {
        let Some(pb) = &mut self.playback else { return };
        match pb.learn.key_range {
            Some((lo, hi)) => {
                ui.label(format!(
                    "Key range: {} – {}",
                    note::solfege_name(lo),
                    note::solfege_name(hi)
                ));
                if ui.button("Clear range").clicked() {
                    pb.learn.key_range = None;
                }
            }
            None => {
                ui.weak("Key range: whole keyboard (drag across the falling notes to set one)");
            }
        }
    }

    /// Clicks and drags on the falling-notes panel: Ctrl+click / right-click
    /// insert a segment break at the clicked time (persisted via the
    /// sidecar); a horizontal drag selects the Learn key-range band; a plain
    /// click outside the band clears it; right-click also offers "Refine
    /// range…" while a band exists.
    fn falling_panel_interactions(
        &mut self,
        ui: &egui::Ui,
        resp: &egui::Response,
        keys: &[KeyRect],
    ) {
        // Read prefs before borrowing `self.playback` mutably below.
        let px_per_s = self.prefs.roll_px_per_s;
        let scrollback_idle_s = self.prefs.scrollback_idle_s;
        let Some(pb) = &mut self.playback else { return };
        let rect = resp.rect;
        // Capture the value, not `pb`, so the closure doesn't pin the borrow.
        // Includes the review offset so clicks land on the time actually drawn
        // under the pointer, not where the playhead would put it.
        let playhead = pb.playhead_s + self.falling_scrollback.unwrap_or(0.0);
        let t_of_y = move |y: f32| playhead + ((rect.bottom() - y) / px_per_s) as f64;

        // Wheel/trackpad scroll reviews time. Drag is reserved for the
        // key-range selection below (it only ever reads x), so a vertical
        // drag on the same gesture would be genuinely ambiguous — time
        // review here is wheel-only. The offset is view-only: the playhead
        // (and Learn gating) never move.
        let scroll_dy = if resp.hovered() {
            ui.input(|i| i.smooth_scroll_delta.y)
        } else {
            0.0
        };
        if scroll_dy != 0.0 {
            let lo = -pb.playhead_s;
            let hi = (pb.score.duration_s - pb.playhead_s).max(0.0);
            let cur = self.falling_scrollback.unwrap_or(0.0);
            self.falling_scrollback =
                Some((cur + (scroll_dy / px_per_s) as f64).clamp(lo, hi));
            self.falling_scrollback_idle_since = None;
        } else if let Some(cur) = self.falling_scrollback {
            // No input: hold the view for the idle window, then ease home.
            let idle = *self.falling_scrollback_idle_since.get_or_insert_with(Instant::now);
            if idle.elapsed().as_secs_f64() >= scrollback_idle_s {
                let dt = ui.input(|i| i.stable_dt) as f64;
                self.falling_scrollback = ease_toward(cur, 0.0, dt, px_per_s);
                if self.falling_scrollback.is_none() {
                    self.falling_scrollback_idle_since = None;
                }
            }
        }

        // Horizontal drag -> key-range selection.
        if resp.drag_started() {
            if let Some(pos) = resp.interact_pointer_pos() {
                self.range_drag = Some((pos.x, pos.x));
            }
        }
        if resp.dragged() {
            if let (Some(drag), Some(pos)) = (&mut self.range_drag, resp.interact_pointer_pos()) {
                drag.1 = pos.x;
            }
        }
        if resp.drag_stopped() {
            if let Some((a, b)) = self.range_drag.take() {
                let (x0, x1) = (a.min(b), a.max(b));
                // Keys whose lane intersects the dragged span; a sub-key drag
                // still grabs whatever key it touched.
                let mut lo = u8::MAX;
                let mut hi = u8::MIN;
                for k in keys {
                    if k.rect.right() >= x0 && k.rect.left() <= x1 {
                        lo = lo.min(k.midi);
                        hi = hi.max(k.midi);
                    }
                }
                if lo <= hi {
                    pb.learn.key_range = Some((lo, hi));
                }
            }
        }

        if resp.clicked() {
            if let Some(pos) = resp.interact_pointer_pos() {
                if ui.input(|i| i.modifiers.command) {
                    pb.score.insert_segment_break(t_of_y(pos.y));
                    if let Err(e) = pb.score.save_segment_sidecar(&pb.source_path) {
                        self.open_status = format!("couldn't save segment breaks: {e}");
                    }
                } else if let Some((lo, hi)) = pb.learn.key_range {
                    // A plain click outside the band dismisses it; inside is
                    // inert so stray clicks don't discard the selection.
                    if let Some(band) = range_band_x(keys, lo, hi) {
                        if !band.contains(pos.x) {
                            pb.learn.key_range = None;
                        }
                    }
                }
            }
        }

        if resp.secondary_clicked() {
            if let Some(pos) = resp.interact_pointer_pos() {
                self.pending_break_t = Some(t_of_y(pos.y));
            }
        }
        // NOTE: shown while *any* band exists (not only when the click landed
        // inside it) — simpler, and harmless to offer slightly outside.
        let has_range = pb.learn.key_range.is_some();
        let range_text = pb
            .learn
            .key_range
            .map(|(lo, hi)| (note::solfege_name(lo), note::solfege_name(hi)));
        resp.context_menu(|ui| {
            if ui.button("Insert segment break here").clicked() {
                if let Some(t) = self.pending_break_t.take() {
                    if let Some(pb) = &mut self.playback {
                        pb.score.insert_segment_break(t);
                        if let Err(e) = pb.score.save_segment_sidecar(&pb.source_path) {
                            self.open_status = format!("couldn't save segment breaks: {e}");
                        }
                    }
                }
                ui.close_menu();
            }
            if has_range && ui.button("Refine range…").clicked() {
                if let Some((lo, hi)) = &range_text {
                    self.refine_lo = lo.clone();
                    self.refine_hi = hi.clone();
                }
                self.show_refine_range = true;
                ui.close_menu();
            }
        });
    }

    /// The "Refine range…" dialog: exact solfège key names for the Learn
    /// key-range band. A field that doesn't parse just leaves that end as-is.
    fn refine_range_window(&mut self, ctx: &egui::Context) {
        if !self.show_refine_range {
            return;
        }
        let Some(pb) = &mut self.playback else {
            self.show_refine_range = false;
            return;
        };
        egui::Window::new("Refine key range")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label("Key names use solfège + octave, e.g. Do4 (middle C), Sol#2, La5.");
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("From:");
                    ui.add(egui::TextEdit::singleline(&mut self.refine_lo).desired_width(70.0));
                    ui.label("to:");
                    ui.add(egui::TextEdit::singleline(&mut self.refine_hi).desired_width(70.0));
                });
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.button("Set").clicked() {
                        let (cur_lo, cur_hi) =
                            pb.learn.key_range.unwrap_or((MIDI_LOW, MIDI_HIGH));
                        let lo = note::solfege_to_midi(&self.refine_lo).unwrap_or(cur_lo);
                        let hi = note::solfege_to_midi(&self.refine_hi).unwrap_or(cur_hi);
                        let lo = lo.clamp(MIDI_LOW, MIDI_HIGH);
                        let hi = hi.clamp(MIDI_LOW, MIDI_HIGH);
                        pb.learn.key_range = Some((lo.min(hi), lo.max(hi)));
                        self.show_refine_range = false;
                    }
                    if ui.button("Cancel").clicked() {
                        self.show_refine_range = false;
                    }
                });
            });
    }

    /// Synth volume + mute controls for the two sources with no acoustic origin:
    /// the keys you click on screen and the notes the peer plays. Each has a mute
    /// toggle and a 0–100% volume slider; changes are pushed to the synth as
    /// per-channel gains so even held notes follow the new level.
    fn synth_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        ui.label("Synth:");
        changed |= ui.checkbox(&mut self.screen_muted, "Mute screen").changed();
        changed |= ui
            .add_enabled(
                !self.screen_muted,
                egui::Slider::new(&mut self.screen_volume, 0.0..=1.0)
                    .show_value(false)
                    .text("screen"),
            )
            .changed();
        ui.separator();
        changed |= ui.checkbox(&mut self.peer_muted, "Mute peer").changed();
        changed |= ui
            .add_enabled(
                !self.peer_muted,
                egui::Slider::new(&mut self.peer_volume, 0.0..=1.0)
                    .show_value(false)
                    .text("peer"),
            )
            .changed();
        if self.playback.is_some() {
            ui.separator();
            changed |= ui.checkbox(&mut self.playback_muted, "Mute playback").changed();
            changed |= ui
                .add_enabled(
                    !self.playback_muted,
                    egui::Slider::new(&mut self.playback_volume, 0.0..=1.0)
                        .show_value(false)
                        .text("playback"),
                )
                .changed();
        }
        if changed {
            self.apply_synth_gains();
        }
    }

    /// Everything inside the *expanded* top config panel: file menu, update
    /// banner, networking row, playback transport, mic threshold, colors,
    /// synth volumes, and the record toggle. Split out of `update()` so the
    /// collapsible-panel wrapper there stays readable.
    fn config_panel_body(&mut self, ui: &mut egui::Ui) {
        // ---- Instance rename + last save/open status. (File/Edit menus live
        // in the custom title bar now — see `title_bar`.) Only drawn when
        // there's something to show, so the panel doesn't reserve a blank row. ----
        if !self.roll.is_empty() || !self.roll_status.is_empty() || !self.open_status.is_empty() {
            ui.horizontal(|ui| self.roll_status_row(ui));
            ui.separator();
        }
        // ---- Auto-update banner (only drawn when there's something to act on) ----
        let mut drawn = false;
        ui.horizontal(|ui| drawn = self.update_controls(ui));
        if drawn {
            ui.separator();
        }
        // ---- Play together: host a session or join one with an invite
        // code. No IPs or ports — iroh handles NAT traversal (net.rs).
        // Greyed out (not hidden — the row keeps its height, so opening a
        // file doesn't reflow the whole panel) while a file is open:
        // playback and live P2P are mutually exclusive (see `open_score`). ----
        ui.horizontal(|ui| {
            let net_enabled = self.playback.is_none();
            let disabled_hint = "Networking is disabled while a file is open (File ▸ Close file)";
            if ui
                .add_enabled(net_enabled, egui::Button::new("Host session"))
                .on_hover_text("Create an invite code to send to the other player")
                .on_disabled_hover_text(disabled_hint)
                .clicked()
            {
                self.host();
            }
            // `my_ticket` is always `None` while a file is open (cleared
            // in `open_score`), so this branch never renders mid-file.
            if let Some(code) = &self.my_ticket {
                if ui
                    .button("📋 Copy invite code")
                    .on_hover_text("Copy the code to the clipboard, then send it to the other player")
                    .clicked()
                {
                    ui.ctx().copy_text(code.clone());
                }
                // The code is 64 hex chars (or ~250 for the LAN-only
                // fallback ticket); show just enough to see it exists.
                // The Copy button is the real interface.
                ui.weak(format!("{}…", &code[..code.len().min(12)]));
            }
            ui.separator();
            ui.label("Invite code:");
            ui.add_enabled(
                net_enabled,
                egui::TextEdit::singleline(&mut self.join_ticket)
                    .desired_width(180.0)
                    .hint_text("paste code from the host"),
            )
            .on_disabled_hover_text(disabled_hint);
            if ui
                .add_enabled(net_enabled, egui::Button::new("Join"))
                .on_disabled_hover_text(disabled_hint)
                .clicked()
            {
                self.join();
            }
        });
        // ---- Playback transport + segment row (only with a file open) ----
        if self.playback.is_some() {
            ui.add_space(2.0);
            ui.horizontal(|ui| self.playback_controls(ui));
            if self.segment_row_visible {
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    self.segment_controls(ui);
                    ui.separator();
                    self.key_range_panel(ui);
                });
            }
        }
        // The detection threshold only affects ONNX transcription, so it's
        // only meaningful when the microphone fallback is active.
        if self.input.source() == Source::Microphone {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label("Detection threshold:");
                if ui
                    .add(egui::Slider::new(&mut self.threshold, 0.05..=0.95).step_by(0.01))
                    .changed()
                {
                    self.input.threshold.set(self.threshold);
                    // Keep the persisted default in sync with this live control.
                    // Debounced: the threshold is a slider dragged at frame rate.
                    self.prefs.threshold = self.threshold;
                    self.save_prefs_soon();
                }
                ui.separator();
                if ui
                    .checkbox(&mut self.mic_muted, "Mute mic")
                    .on_hover_text(
                        "Ignore mic-detected notes (stops ambient noise from \
                         painting the roll or counting as played keys)",
                    )
                    .changed()
                {
                    self.prefs.mic_muted = self.mic_muted;
                    self.prefs.save();
                }
            });
        }
        // ---- My name + color (broadcast to the peer) ----
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.label("My name:");
            if ui
                .add(egui::TextEdit::singleline(&mut self.local_name).desired_width(120.0))
                .changed()
            {
                // Push the change immediately; the heartbeat covers the rest.
                self.send_name();
                self.prefs.local_name = self.local_name.clone();
                self.prefs.save();
            }
            ui.label("My color:");
            if ui.color_edit_button_srgb(&mut self.local_color).changed() {
                // Push the change immediately; the heartbeat covers the rest.
                self.send_color();
                // Persist as the new default color.
                self.prefs.local_color = self.local_color;
                self.prefs.save();
            }
            ui.separator();
            ui.label(format!("Peer: {}", self.remote_name));
            let (r, g, b) = (self.remote_color[0], self.remote_color[1], self.remote_color[2]);
            ui.colored_label(egui::Color32::from_rgb(r, g, b), "■");
            // The peer is always drawn in the fixed remote color locally — the
            // peer's own color choice is intentionally not applied (see the
            // `Packet::Color` handler in pump_network), so this isn't "their"
            // color, it's the color they appear in on your screen (L6).
            ui.weak("(peer color on your screen)");
        });
        // ---- Synth volume / mute (screen + peer sources) ----
        ui.add_space(2.0);
        ui.horizontal(|ui| self.synth_controls(ui));
        // ---- Metronome (synced with the peer; each side mutes its own click) ----
        ui.add_space(2.0);
        ui.horizontal(|ui| self.metro_controls(ui));
        // ---- Training-data capture (record mic audio + MIDI labels) ----
        ui.add_space(2.0);
        ui.horizontal(|ui| self.record_controls(ui));
        ui.add_space(4.0);
    }

    /// Auto-update status line. The check + download run on a background thread
    /// (see `update.rs`); here we just render its latest state. We stay silent
    /// while `Checking` or `UpToDate` so the bar shows nothing in the common case,
    /// and only surface a row once there's a newer build staged (or a failure
    /// worth a tooltip). Returns whether anything was drawn, so the caller can
    /// skip the separator when there's nothing to show.
    fn update_controls(&mut self, ui: &mut egui::Ui) -> bool {
        match self.updater.state() {
            update::UpdateState::Ready { version } => {
                ui.colored_label(
                    egui::Color32::from_rgb(90, 180, 110),
                    format!("✓ Update ready: v{version}"),
                );
                if ui
                    .button("Restart now")
                    .on_hover_text("Relaunch into the new version (or just reopen the app later)")
                    .clicked()
                {
                    // Honor the unsaved-roll confirmation and flush a pending
                    // prefs save before the process exits (F12/F22/C1).
                    self.request_restart();
                }
                true
            }
            update::UpdateState::Failed { reason } => {
                ui.colored_label(egui::Color32::from_rgb(210, 170, 60), "Update check failed")
                    .on_hover_text(reason);
                true
            }
            update::UpdateState::Checking | update::UpdateState::UpToDate => false,
        }
    }

    /// Route a note transition to the built-in synth on a given channel (so the
    /// screen and peer sources can be muted/leveled independently).
    ///
    /// While recording, suppress note-*ons*: the local mic is capturing, and
    /// synth output bleeding through the speakers would contaminate the training
    /// audio with tones that have no MIDI label. Note-offs always pass through so
    /// nothing started before arming gets stuck sounding.
    fn play_synth(&mut self, msg: NoteMsg, channel: synth::Channel) {
        // Guard the note byte before it reaches the synth: peer datagrams are
        // untrusted, and an out-of-range note (e.g. `[0x90, 200, ..]`) would
        // start an aliased ultrasonic voice that `clear_remote_keys` — which
        // only releases MIDI 21..=108 — could never stop (L1). Every other
        // consumer already gates on `midi_to_key_index`; match it here.
        if midi_to_key_index(msg.midi()).is_none() {
            return;
        }
        match msg {
            NoteMsg::On(n, _) => {
                // Velocity is deliberately unused here: the built-in synth
                // has no per-voice level control (out of scope for now).
                if !self.input.recorder.is_armed() {
                    self.synth_note_on(n, channel);
                }
            }
            NoteMsg::Off(n) => self.synth_note_off(n, channel),
        }
    }

    /// Push the current screen/peer volume + mute state down to the synth as
    /// per-channel gains (muted → 0). Called once on startup and whenever a
    /// slider or mute toggle changes.
    fn apply_synth_gains(&self) {
        let screen = if self.screen_muted { 0.0 } else { self.screen_volume };
        let peer = if self.peer_muted { 0.0 } else { self.peer_volume };
        let playback = if self.playback_muted { 0.0 } else { self.playback_volume };
        self.synth.set_gain(synth::Channel::Local, screen);
        self.synth.set_gain(synth::Channel::Peer, peer);
        self.synth.set_gain(synth::Channel::Playback, playback);
        let metro = if self.metro.muted { 0.0 } else { self.metro.volume };
        self.synth.set_gain(synth::Channel::Metronome, metro);
    }

    /// Whether *this* instance owns the metronome grid: true when solo (no peer)
    /// or when we're the host. A follower defers to the host's markers.
    fn metro_authority(&self) -> bool {
        // A joiner whose host has gone (peer still `Some`, but disconnected)
        // reclaims authority so its metronome self-anchors instead of latching
        // on silently (M9).
        self.peer.is_none() || !self.peer_connected || self.is_host
    }

    /// (Re)anchor the metronome to the next roll-time-grid-aligned beat — the
    /// smallest multiple of the beat period (measured from roll-time zero) at
    /// or after the current roll time. Beat 0 of the grid always sits at
    /// roll-time zero, so beats land on round absolute positions (every BPM
    /// beats is exactly one whole minute of roll time) regardless of when the
    /// metronome was actually started: pressing "start" mid-beat waits for the
    /// next grid line rather than clicking immediately. Used on enable and
    /// after a long stall; once running, beats simply free-run by adding
    /// `period` each time (see `drive_metronome`), so a later BPM tweak doesn't
    /// cause an audible jump — only a fresh start re-snaps to the grid.
    fn metro_start_now(&mut self) {
        let (delta_s, beat_in_bar) =
            metro_grid_align(self.roll.now_s(), self.metro.period(), self.metro.beats_per_bar);
        self.metro.next_beat_at = Some(Instant::now() + Duration::from_secs_f64(delta_s));
        self.metro.next_beat_in_bar = beat_in_bar;
    }

    /// Advance the metronome and fire any clicks due this frame. Called once per
    /// frame from `update`. The authority free-runs its own schedule (and, when
    /// connected, broadcasts each beat); a follower fires from the schedule its
    /// last received marker anchored (see `on_metro_beat`).
    fn drive_metronome(&mut self) {
        if !self.metro.enabled {
            self.metro.next_beat_at = None;
            return;
        }
        let now = Instant::now();
        let authority = self.metro_authority();
        let period = Duration::from_secs_f64(self.metro.period());
        let bpb = self.metro.beats_per_bar.max(1);

        // The authority free-runs from a roll-time-grid-aligned anchor if idle.
        if authority && self.metro.next_beat_at.is_none() {
            self.metro_start_now();
        }
        // Recover from a long GUI stall (window drag, file dialog, sleep)
        // without machine-gunning every missed beat: re-anchor to the grid
        // instead of firing immediately.
        if let Some(at) = self.metro.next_beat_at {
            if now.saturating_duration_since(at) > Duration::from_secs(1) {
                self.metro_start_now();
            }
        }

        while let Some(at) = self.metro.next_beat_at {
            if now < at {
                break;
            }
            let beat_in_bar = self.metro.next_beat_in_bar;
            // Record every processed beat (sounded or muted) so a follower can
            // dedup a re-delivered marker (F3).
            self.metro.last_beat_at = Some(at);
            if !self.metro.muted {
                self.synth.tick(
                    self.metro.freq_for_beat(beat_in_bar),
                    beat_in_bar == 0,
                    self.metro.volume_for_beat(beat_in_bar),
                );
            }
            if authority {
                if let Some(peer) = &self.peer {
                    peer.send(Packet::MetroBeat {
                        bpm: self.metro.bpm,
                        beat_in_bar,
                        beats_per_bar: self.metro.beats_per_bar,
                        on: true,
                    });
                }
            }
            self.metro.next_beat_in_bar = (beat_in_bar + 1) % bpb;
            self.metro.next_beat_at = Some(at + period);
        }
    }

    /// Apply a metronome start/stop + tempo edit from the UI, routed by role: the
    /// authority changes its grid directly (and tells a connected follower); a
    /// follower sends a `MetroCtl` request the host adopts and echoes back.
    fn metro_set(&mut self, enabled: bool, bpm: u16) {
        let bpm = bpm.clamp(MIN_BPM, MAX_BPM);
        let was_enabled = self.metro.enabled;
        self.metro.bpm = bpm;
        self.metro.enabled = enabled;
        if !enabled {
            self.metro.next_beat_at = None;
        }
        if self.metro_authority() {
            if enabled && !was_enabled {
                // Fresh start: wait for the next roll-time-grid-aligned beat
                // rather than clicking immediately (see `metro_start_now`).
                self.metro_start_now();
            }
            // A bpm-only change keeps the running schedule (period is read live);
            // only broadcast the start/stop edge here — the per-beat markers
            // carry the new tempo to the follower on the next beat.
            if enabled != was_enabled {
                if let Some(peer) = &self.peer {
                    peer.send(Packet::MetroBeat {
                        bpm,
                        beat_in_bar: 0,
                        beats_per_bar: self.metro.beats_per_bar,
                        on: enabled,
                    });
                }
            }
        } else if let Some(peer) = &self.peer {
            // Follower: request the change; the host's markers are authoritative.
            peer.send(Packet::MetroCtl { on: enabled, bpm });
        }
    }

    /// Handle a metronome beat marker from the host (follower side). Anchors the
    /// *next* local click to the corrected marker time (receive time minus the
    /// one-way estimate), so our clicks land in phase with the host's. Ignored
    /// by the authority (it owns the grid).
    fn on_metro_beat(
        &mut self,
        bpm: u16,
        beat_in_bar: u8,
        beats_per_bar: u8,
        on: bool,
        one_way: Duration,
    ) {
        if self.metro_authority() {
            return;
        }
        self.metro.bpm = bpm.clamp(MIN_BPM, MAX_BPM);
        self.metro.beats_per_bar = beats_per_bar.max(1);
        if !on {
            self.metro.enabled = false;
            self.metro.next_beat_at = None;
            return;
        }
        self.metro.enabled = true;
        let now = Instant::now();
        let period = Duration::from_secs_f64(self.metro.period());
        // When (corrected for transit) the host played this beat, in our clock.
        let this_beat_at = now.checked_sub(one_way).unwrap_or(now);
        // Reduce the (untrusted) wire byte modulo the bar first, so a hostile
        // `beat_in_bar = 255` can't overflow the u8 add (a debug-build panic) —
        // `beat_in_bar % bpb < bpb`, so the `+ 1` is always in range (L2).
        let bpb = self.metro.beats_per_bar; // already `.max(1)` above
        let beat_in_bar = beat_in_bar % bpb;

        // Have we already handled ~this beat? On low-latency links the marker
        // for beat N arrives right as our local schedule for beat N comes due;
        // the old code overwrote the schedule with beat N+1, silently swallowing
        // N's click. Instead, if we haven't already sounded a beat within half a
        // period of this one, (re)schedule beat N *itself* at its corrected time
        // — which is at/just-before `now`, so `drive_metronome` (running right
        // after `pump_network` this same frame) sounds it and advances to N+1.
        // The dedup guard prevents a double-click when a marker is re-delivered
        // after we already generated N locally (F3).
        let already_sounded = self.metro.last_beat_at.is_some_and(|prev| {
            let d = if this_beat_at >= prev {
                this_beat_at - prev
            } else {
                prev - this_beat_at
            };
            d < period / 2
        });
        if already_sounded {
            // Same beat we already handled: just re-anchor the next one.
            self.metro.next_beat_at = Some(this_beat_at + period);
            self.metro.next_beat_in_bar = (beat_in_bar + 1) % bpb;
        } else {
            // Fire beat N now (via the schedule), then it advances to N+1.
            self.metro.next_beat_at = Some(this_beat_at);
            self.metro.next_beat_in_bar = beat_in_bar;
        }
    }

    /// Metronome row: on/off, tempo, and a local-only "mute click". Editing tempo
    /// or on/off on a follower is sent to the host (see `metro_set`).
    fn metro_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Metronome:");
        let mut enabled = self.metro.enabled;
        let label = if enabled { "⏸" } else { "▶" };
        if ui
            .toggle_value(&mut enabled, label)
            .on_hover_text("Start / stop the metronome (synced with the peer)")
            .changed()
        {
            self.metro_set(enabled, self.metro.bpm);
        }
        let mut bpm = self.metro.bpm;
        if ui
            .add(egui::DragValue::new(&mut bpm).range(MIN_BPM..=MAX_BPM).suffix(" BPM"))
            .on_hover_text("Tempo (both players hear the same beat)")
            .changed()
        {
            self.metro_set(self.metro.enabled, bpm);
            self.prefs.metro_bpm = bpm.clamp(MIN_BPM, MAX_BPM);
            self.save_prefs_soon(); // BPM is a drag-value — debounce the save
        }
        ui.separator();
        let mut changed = ui
            .checkbox(&mut self.metro.muted, "Mute click")
            .on_hover_text("Silence the click on *this* machine only (the peer still hears it)")
            .changed();
        changed |= ui
            .add_enabled(
                !self.metro.muted,
                egui::Slider::new(&mut self.metro.volume, 0.0..=1.0)
                    .show_value(false)
                    .text("click volume"),
            )
            .changed();
        if changed {
            self.apply_synth_gains();
        }
    }

    /// Apply the source-dependent default for the on-screen ("screen") synth:
    /// with a real MIDI piano connected the instrument makes its own sound, so
    /// the screen synth is muted by default; on the mic-only fallback it's
    /// unmuted so the on-screen keyboard is audible. Re-applied only when the
    /// MIDI connection state actually changes, so a manual mute toggle sticks
    /// until the next plug/unplug. The peer synth is untouched — the peer's
    /// notes never have a local acoustic source regardless of our input.
    fn sync_synth_to_source(&mut self) {
        let midi = self.input.midi_connected();
        if midi != self.was_midi {
            self.was_midi = midi;
            self.screen_muted = midi;
            self.apply_synth_gains();
        }
    }

    /// Play a mouse-triggered note: light it up locally, sound it on the synth,
    /// and forward it to the peer — the same visible effect a MIDI/mic event has
    /// in `pump_input`, plus the synth (clicks have no other sound source).
    fn local_note(&mut self, msg: NoteMsg) {
        apply(&mut self.local, msg);
        self.roll.note(roll::Who::Local, msg, self.local_color);
        self.play_synth(msg, synth::Channel::Local);
        self.send_note(msg);
    }

    /// Reconcile the mouse-held note with the key currently under the pressed
    /// pointer (`None` when the button is up or the pointer left the keys). A
    /// change releases the previous note and presses the new one, so dragging
    /// across the keyboard glides note-to-note.
    fn set_mouse_note(&mut self, target: Option<u8>) {
        if target == self.mouse_note {
            return;
        }
        if let Some(old) = self.mouse_note.take() {
            self.local_note(NoteMsg::Off(old));
        }
        if let Some(new) = target {
            self.mouse_note = Some(new);
            // A mouse click has no force behind it: flat placeholder velocity.
            self.local_note(NoteMsg::On(new, note::DEFAULT_VELOCITY));
        }
    }

    /// The custom window title bar, drawn as the *first* panel each frame so it
    /// stays the topmost row regardless of the config panel's collapse state:
    /// File/Edit menus + the unsaved chip on the left, a centered title, and our
    /// own minimize/maximize/close buttons on the right. Empty bar area drags
    /// the window; double-click maximizes/restores. Also installs the edge-resize
    /// handles the OS would provide (we dropped its chrome — see `main`).
    fn title_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("titlebar")
            .exact_height(TITLEBAR_H)
            .show(ctx, |ui| {
                let bar_rect = ui.max_rect();
                // Drag / double-click the bar. Interact FIRST so the menus and
                // buttons added afterwards sit on top and win their own clicks;
                // the leftover bar area drags the window.
                let drag = ui.interact(
                    bar_rect,
                    egui::Id::new("titlebar_drag"),
                    egui::Sense::click_and_drag(),
                );
                if drag.double_clicked() {
                    let max = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!max));
                } else if drag.dragged() {
                    // Drive the window ourselves each frame instead of handing
                    // off to the native SC_MOVE loop (which doesn't sustain a
                    // touch gesture — the actual cause of broken touch-move).
                    // Skip while maximized: repositioning a maximized window is
                    // meaningless, and the double-click above already toggles it.
                    let maximized = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
                    if let Some(local) = drag.interact_pointer_pos() {
                        // Pin the grab offset once, on the first drag frame.
                        let grab = *self.titlebar_drag.get_or_insert(local);
                        if !maximized {
                            // Solve the window origin absolutely so the grab
                            // point stays under the pointer. Delta-accumulation
                            // (the old approach) measured pointer motion in
                            // window-local space, which flips sign as the
                            // window moves under it — a feedback loop that
                            // jittered. Reading the live outer rect each frame
                            // makes this idempotent instead.
                            if let Some(outer) = ctx.input(|i| i.viewport().outer_rect) {
                                let target = outer.min + (local - grab);
                                ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(target));
                            }
                        }
                    }
                } else {
                    self.titlebar_drag = None;
                }

                // Centered title, painted behind the widgets (non-interactive).
                ui.painter().text(
                    bar_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    concat!("open-piano v", env!("CARGO_PKG_VERSION")),
                    egui::FontId::proportional(12.5),
                    ui.visuals().weak_text_color(),
                );

                // Window controls on the right. A child UI spanning the FULL bar
                // rect, laid out right-to-left, so Close lands at the true right
                // edge (a plain `menu::bar` shrinks to content and would bunch
                // these next to the menus instead).
                ui.allocate_new_ui(
                    egui::UiBuilder::new()
                        .max_rect(bar_rect)
                        .layout(egui::Layout::right_to_left(egui::Align::Center)),
                    |ui| {
                        if window_button(ui, WinBtn::Close) {
                            self.request_close(ctx);
                        }
                        let max = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
                        if window_button(ui, WinBtn::Maximize(max)) {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!max));
                        }
                        if window_button(ui, WinBtn::Minimize) {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                        }
                        // Compact toggle: flips intent (and persists it) only —
                        // `sync_compact_viewport`, called right after
                        // `title_bar()` returns, owns the actual resize.
                        if window_button(ui, WinBtn::Compact(self.compact_mode)) {
                            self.compact_mode = !self.compact_mode;
                            if self.prefs.remember_window_state {
                                self.prefs.compact_mode = self.compact_mode;
                                self.prefs.save();
                            }
                        }
                    },
                );

                // File/Edit menus + unsaved chip on the left, in a full-width
                // child UI so they anchor to the left edge independently of the
                // buttons above.
                ui.allocate_new_ui(
                    egui::UiBuilder::new()
                        .max_rect(bar_rect)
                        .layout(egui::Layout::left_to_right(egui::Align::Center)),
                    |ui| {
                        egui::menu::bar(ui, |ui| {
                            self.file_menu(ui);
                            self.edit_menu(ui);
                            self.unsaved_chip(ui);
                        });
                    },
                );
            });

        self.resize_handles(ctx);
    }

    /// Shared close decision for both the custom ✕ and the OS close path
    /// (Alt+F4, handled in `update` via `close_requested`): with unsaved roll
    /// notes, open the confirm dialog; otherwise close for real.
    fn request_close(&mut self, ctx: &egui::Context) {
        if self.roll.has_unsaved() && !self.allow_close {
            self.show_close_confirm = true;
        } else {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    /// Handle the "Restart now" button: honor the unsaved-roll confirmation
    /// (the update restart used to bypass it, destroying unsaved playing, F22).
    /// With no unsaved work, restart straight away.
    fn request_restart(&mut self) {
        if self.roll.has_unsaved() {
            self.pending_restart = true;
            self.show_close_confirm = true;
        } else {
            self.perform_restart();
        }
    }

    /// Actually relaunch into the updated build. `update::restart()` calls
    /// `process::exit`, which runs no destructors — so flush a pending debounced
    /// prefs save (F12) and finalize any in-progress recording (C1) first.
    fn perform_restart(&mut self) -> ! {
        if self.prefs_save_due.take().is_some() {
            self.prefs.save();
        }
        self.input.recorder.shutdown();
        update::restart()
    }

    /// Thin invisible drag handles on the window's side/bottom edges and four
    /// corners (no top edge — that strip is the title bar's move-drag; see the
    /// note on `handles`), each resizing the window — the affordance the OS
    /// gave us before we dropped its chrome. Each handle is its *own* foreground `Area` with a
    /// thin bounding rect, so only the edge strips capture the pointer and the
    /// window interior falls through to the app (a single screen-spanning Area
    /// would block everything — see egui's `layer_id_at`). Skipped while
    /// maximized (nothing to resize).
    fn resize_handles(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.viewport().maximized.unwrap_or(false)) {
            return;
        }
        let s = ctx.screen_rect();
        // Enlarged, grabbable strips only when the device is actually being used
        // as a tablet (a real touch event arrived recently); a mouse/trackpad
        // session keeps the original tight, invisible 6px zones.
        let b_thick: f32 = if self.touch_mode { 14.0 } else { 6.0 };
        let touch = self.touch_mode;
        use egui::CursorIcon as C;
        use egui::ResizeDirection as D;
        let (l, r, t, b) = (s.left(), s.right(), s.top(), s.bottom());
        let rect = egui::Rect::from_min_max;
        let p = egui::pos2;
        let bb = b_thick;
        // (handle rect, resize direction, cursor, is-corner) — corners then
        // edges, tiled so none overlap. Deliberately NO north edge handle: the
        // top strip belongs to the title bar's move-drag, and a resize handle
        // there (a foreground Area, so it wins the pointer) hijacked
        // move-drags that started near the top edge. The ENTIRE title-bar strip
        // (`TITLEBAR_H`) is reserved for it — the NW/NE corners begin just below
        // it, so a touch-sized corner handle no longer sits over the window
        // buttons / menus (a tap over ✕ started a resize instead of closing,
        // F23). Top resizing stays available from those just-below-the-bar
        // corners.
        let ct = t + TITLEBAR_H; // top of the resize zone (below the title bar)
        let handles = [
            (rect(p(l, ct), p(l + bb, ct + bb)), D::NorthWest, C::ResizeNorthWest, true),
            (rect(p(r - bb, ct), p(r, ct + bb)), D::NorthEast, C::ResizeNorthEast, true),
            (rect(p(l, b - bb), p(l + bb, b)), D::SouthWest, C::ResizeSouthWest, true),
            (rect(p(r - bb, b - bb), p(r, b)), D::SouthEast, C::ResizeSouthEast, true),
            (rect(p(l + bb, b - bb), p(r - bb, b)), D::South, C::ResizeSouth, false),
            (rect(p(l, ct + bb), p(l + bb, b - bb)), D::West, C::ResizeWest, false),
            (rect(p(r - bb, ct + bb), p(r, b - bb)), D::East, C::ResizeEast, false),
        ];
        // Minimum inner size to clamp against, matching what the OS enforced —
        // tighter in compact mode so the compact window can shrink further.
        let (min_w, min_h) = if self.compact_mode {
            (COMPACT_MIN_SIZE[0], COMPACT_MIN_SIZE[1])
        } else {
            (NORMAL_MIN_SIZE[0], NORMAL_MIN_SIZE[1])
        };

        let mut still_dragging = false;
        for (i, (hr, dir, cursor, corner)) in handles.into_iter().enumerate() {
            let resp = egui::Area::new(egui::Id::new(("resize_handle", i)))
                .order(egui::Order::Foreground)
                .fixed_pos(hr.min)
                .interactable(true)
                .show(ctx, |ui| {
                    let (_r, resp) = ui.allocate_exact_size(hr.size(), egui::Sense::drag());
                    if resp.hovered() || resp.dragged() {
                        ui.ctx().set_cursor_icon(cursor);
                    }
                    // Tablet-only visible grab affordance: two short strokes in
                    // the corner so the enlarged zone reads as draggable. Never
                    // drawn for a mouse/trackpad session (tight invisible zones).
                    if touch && corner {
                        let g = egui::Color32::from_gray(150);
                        let stroke = egui::Stroke::new(1.5, g);
                        let hr = ui.min_rect();
                        let inset = 4.0;
                        // A small "⌟"-ish bracket hugging the true window corner.
                        let (c1, c2, c3) = match dir {
                            D::NorthWest => (
                                egui::pos2(hr.left() + inset, hr.top() + inset + 5.0),
                                egui::pos2(hr.left() + inset, hr.top() + inset),
                                egui::pos2(hr.left() + inset + 5.0, hr.top() + inset),
                            ),
                            D::NorthEast => (
                                egui::pos2(hr.right() - inset - 5.0, hr.top() + inset),
                                egui::pos2(hr.right() - inset, hr.top() + inset),
                                egui::pos2(hr.right() - inset, hr.top() + inset + 5.0),
                            ),
                            D::SouthWest => (
                                egui::pos2(hr.left() + inset, hr.bottom() - inset - 5.0),
                                egui::pos2(hr.left() + inset, hr.bottom() - inset),
                                egui::pos2(hr.left() + inset + 5.0, hr.bottom() - inset),
                            ),
                            _ => (
                                egui::pos2(hr.right() - inset - 5.0, hr.bottom() - inset),
                                egui::pos2(hr.right() - inset, hr.bottom() - inset),
                                egui::pos2(hr.right() - inset, hr.bottom() - inset - 5.0),
                            ),
                        };
                        ui.painter().line_segment([c1, c2], stroke);
                        ui.painter().line_segment([c2, c3], stroke);
                    }
                    resp
                })
                .inner;

            if resp.dragged() {
                still_dragging = true;
                // Seed the accumulated target from the live outer position +
                // inner size on the first drag frame; thereafter apply deltas
                // to our own target so a lagging reported rect can't drop or
                // double-count movement.
                let seed = self.resize_drag.or_else(|| {
                    ctx.input(|i| {
                        let vp = i.viewport();
                        match (vp.outer_rect, vp.inner_rect) {
                            (Some(o), Some(inner)) => Some((o.min, inner.size())),
                            _ => None,
                        }
                    })
                });
                if let Some((mut pos, mut size)) = seed {
                    let d = resp.drag_delta();
                    let (west, east) = (
                        matches!(dir, D::NorthWest | D::SouthWest | D::West),
                        matches!(dir, D::NorthEast | D::SouthEast | D::East),
                    );
                    let (north, south) = (
                        matches!(dir, D::NorthWest | D::NorthEast | D::North),
                        matches!(dir, D::SouthWest | D::SouthEast | D::South),
                    );
                    if east {
                        size.x = (size.x + d.x).max(min_w);
                    }
                    if south {
                        size.y = (size.y + d.y).max(min_h);
                    }
                    // Left/top edges: shift the outer position by however much
                    // the edge actually moved (post-clamp) so the opposite edge
                    // stays anchored — what a native resize does.
                    if west {
                        let target_w = (size.x - d.x).max(min_w);
                        pos.x += size.x - target_w;
                        size.x = target_w;
                    }
                    if north {
                        let target_h = (size.y - d.y).max(min_h);
                        pos.y += size.y - target_h;
                        size.y = target_h;
                    }
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(size));
                    if west || north {
                        ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(pos));
                    }
                    self.resize_drag = Some((pos, size));
                }
            }
        }
        if !still_dragging {
            self.resize_drag = None;
        }
    }

    /// Reconcile the OS window against compact-mode intent, once per frame.
    /// `compact_applied` tracks what's actually been sent to the window, so a
    /// resize fires only on real transitions. Dialogs (Preferences included)
    /// simply float over a still-compact main window — its own scroll area
    /// degrades gracefully in a short pane, and the ever-present
    /// `resize_handles` let the user drag the window taller if they want.
    fn sync_compact_viewport(&mut self, ctx: &egui::Context) {
        // Always-on-top is reconciled first and *every* frame (before the size
        // early-return below), so flipping the preference while already compact
        // takes effect without a compact-mode transition to ride on.
        let want_on_top = self.compact_mode && self.prefs.compact_always_on_top;
        if want_on_top != self.on_top_applied {
            let level = if want_on_top {
                egui::WindowLevel::AlwaysOnTop
            } else {
                egui::WindowLevel::Normal
            };
            ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(level));
            self.on_top_applied = want_on_top;
        }

        let want_compact = self.compact_mode;
        if want_compact == self.compact_applied {
            return;
        }
        if want_compact {
            self.apply_compact_size(ctx);
        } else {
            self.apply_normal_size(ctx);
        }
        self.compact_applied = want_compact;
    }

    /// Shrink to keyboard + title bar: snapshot the current size for restore,
    /// keep the width, clamp the height.
    fn apply_compact_size(&mut self, ctx: &egui::Context) {
        let maximized = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
        // Clear the OS maximized flag first: a maximized window ignores
        // `InnerSize` and leaves the title-bar move-drag and every resize handle
        // disabled (they early-return while maximized), so the compact strip
        // would be stuck un-movable and un-resizable (F10).
        if maximized {
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(false));
        }
        // Snapshot the restore size only from a *non-maximized* rect: capturing
        // the full-screen maximized rect would persist a screen-sized "normal"
        // window that compact-restore then expands to (F10). While maximized we
        // keep the previously-captured `normal_size` (or the default).
        if !maximized {
            if let Some(rect) = ctx.input(|i| i.viewport().inner_rect) {
                self.normal_size = Some(rect.size());
                // Same opt-in policy as `compact_mode` itself: only persisted
                // when the user asked for window state to be remembered.
                if self.prefs.remember_window_state {
                    self.prefs.normal_window_size = Some([rect.size().x, rect.size().y]);
                    self.prefs.save();
                }
            }
        }
        let width = self.normal_size.map_or(DEFAULT_WINDOW_SIZE[0], |s| s.x);
        ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(COMPACT_MIN_SIZE.into()));
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
            width,
            COMPACT_WINDOW_H,
        )));
    }

    /// Restore normal size after compact mode.
    fn apply_normal_size(&mut self, ctx: &egui::Context) {
        ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(NORMAL_MIN_SIZE.into()));
        // Keep whatever width the user resized to while compact (still-resizable
        // per spec); only height needs restoring since compact clamps it down.
        let live_w = ctx.input(|i| i.viewport().inner_rect).map(|r| r.width());
        let height = self.normal_size.map_or(DEFAULT_WINDOW_SIZE[1], |s| s.y);
        let width = live_w
            .or(self.normal_size.map(|s| s.x))
            .unwrap_or(DEFAULT_WINDOW_SIZE[0]);
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(width, height)));
    }

    /// The Edit menu. Currently just Preferences; shares the title bar's menu
    /// bar with File.
    fn edit_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Edit", |ui| {
            let prefs = egui::Button::new("Preferences…")
                .shortcut_text(ui.ctx().format_shortcut(&PREFS_SHORTCUT));
            if ui
                .add(prefs)
                .on_hover_text("Roll timing, appearance, audio, and advanced tunables")
                .clicked()
            {
                ui.close_menu();
                self.show_prefs = true;
            }
        });
    }

    /// Edit ▸ Preferences: every persisted tunable (see prefs.rs), grouped
    /// into sidebar-navigated tabs (one `prefs_*_section` method per page).
    /// Each widget applies its change live to the relevant consumer and, if
    /// anything changed this frame, saves the prefs file (a tiny atomic write).
    /// The dialog is a floating, movable, resizable window: position/size are
    /// remembered by egui's per-window memory for the session (this app
    /// doesn't enable eframe persistence, so they reset on relaunch).
    fn preferences_window(&mut self, ctx: &egui::Context) {
        if !self.show_prefs {
            return;
        }
        // A local `open` for the window ✕ so the closure can still take `&mut
        // self` (unlike `about_window`, which only reads inside).
        let mut open = true;
        let mut changed = false;
        egui::Window::new("Preferences")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size([700.0, 520.0])
            .min_size([560.0, 320.0])
            // A default only — unlike the `.anchor(...)` this replaces, which
            // re-asserted center every frame and snapped any drag straight
            // back. From here on egui's own window memory owns position.
            .default_pos(ctx.screen_rect().center() - egui::vec2(350.0, 260.0))
            .show(ctx, |ui| {
                egui::SidePanel::left("prefs_nav")
                    .resizable(false)
                    .exact_width(150.0)
                    .show_inside(ui, |ui| {
                        for section in PrefsSection::ALL {
                            if ui
                                .selectable_label(self.prefs_section == section, section.label())
                                .clicked()
                            {
                                self.prefs_section = section;
                            }
                        }
                    });
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    // Cap the tab body to the viewport height (minus a reserve
                    // for chrome/margins) and scroll it — a sane ceiling for
                    // the scroll region *inside* the content pane; the window
                    // itself is sized by egui's resize memory, which this no
                    // longer fights.
                    let max_h = (ui.ctx().screen_rect().height() - 200.0).clamp(240.0, 640.0);
                    egui::ScrollArea::vertical()
                        // Per-tab scroll id, so switching tabs doesn't inherit
                        // the previous tab's scroll offset.
                        .id_salt(self.prefs_section.label())
                        .auto_shrink([false, false])
                        .max_height(max_h)
                        .show(ui, |ui| {
                            changed |= match self.prefs_section {
                                PrefsSection::StartupWindow => self.prefs_startup_section(ui),
                                PrefsSection::RollHistory => self.prefs_roll_section(ui),
                                PrefsSection::Appearance => self.prefs_appearance_section(ui),
                                PrefsSection::Pedal => self.prefs_pedal_section(ui),
                                PrefsSection::RollBehavior => self.prefs_roll_behavior_section(ui),
                                PrefsSection::AudioMic => self.prefs_audio_section(ui),
                                PrefsSection::Metronome => self.prefs_metronome_section(ui),
                                PrefsSection::Advanced => self.prefs_advanced_section(ui),
                            };
                        });
                });

                // Roll timing is read from `self.roll`'s own fields, so push
                // any timing edit down — once, whichever tab produced the
                // change (cheap and idempotent when it wasn't those fields).
                if changed {
                    self.roll.set_timing(
                        Duration::from_secs_f64(self.prefs.section_tail_s.max(0.0)),
                        Duration::from_secs_f64(self.prefs.section_lead_in_s.max(0.0)),
                        self.prefs.idle_pause.as_duration(),
                    );
                }
            });

        if changed {
            // Debounced: the Preferences dialog aggregates every control's
            // `.changed()` here, including sliders/drag-values that fire each
            // frame of a drag — an immediate save would write ~60×/s (M7).
            self.save_prefs_soon();
        }
        // The window ✕ (or Esc) clears `open`; mirror it back into our flag.
        if !open {
            self.show_prefs = false;
        }
    }

    /// The "Startup & window" tab: session-restore toggles.
    fn prefs_startup_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("Startup & window");
        if ui
            .checkbox(&mut self.prefs.remember_window_state, "Remember window state")
            .on_hover_text("Restore compact/normal window mode from your last session")
            .changed()
        {
            if self.prefs.remember_window_state {
                // Capture the current state immediately — otherwise it
                // only persists on the next toggle.
                self.prefs.compact_mode = self.compact_mode;
            }
            changed = true;
        }
        if ui
            .checkbox(&mut self.prefs.compact_always_on_top, "Keep compact window on top")
            .on_hover_text(
                "While in compact mode, float the window above other apps. \
                 Reconciled live — toggling this takes effect immediately.",
            )
            .changed()
        {
            // `sync_compact_viewport` reconciles the actual window level every
            // frame, so no explicit command is needed here.
            changed = true;
        }
        if ui
            .checkbox(&mut self.prefs.reopen_last_file, "Reopen last file on launch")
            .on_hover_text("Reload the most recently opened MIDI/JSONL file at startup")
            .changed()
        {
            self.prefs.last_file_path = if self.prefs.reopen_last_file {
                // Capture the currently-open file immediately, if any.
                self.playback.as_ref().map(|pb| pb.source_path.clone())
            } else {
                None // don't keep a stale path once the feature is off
            };
            changed = true;
        }
        changed
    }

    /// The "Roll & history" tab: view tunables (section-break timing lives in
    /// the Roll behavior tab).
    fn prefs_roll_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("Roll & history");
        ui.horizontal(|ui| {
            ui.label("Roll zoom (px/s):");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.prefs.roll_px_per_s)
                        .range(8.0..=200.0)
                        .speed(1.0),
                )
                .on_hover_text("Vertical scale of the history + falling-note strips")
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Scrollback hold (s):");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.prefs.scrollback_idle_s)
                        .range(0.0..=30.0)
                        .speed(0.1),
                )
                .on_hover_text("How long a scrolled-back view holds before easing to live")
                .changed();
        });
        changed
    }

    /// The "Appearance" tab: display name + note color.
    fn prefs_appearance_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("Appearance");
        ui.horizontal(|ui| {
            ui.label("My display name:");
            if ui
                .add(
                    egui::TextEdit::singleline(&mut self.prefs.local_name)
                        .desired_width(160.0),
                )
                .changed()
            {
                self.local_name = self.prefs.local_name.clone();
                self.send_name();
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("My note color:");
            if ui.color_edit_button_srgb(&mut self.prefs.local_color).changed() {
                self.local_color = self.prefs.local_color;
                self.send_color();
                changed = true;
            }
        });
        changed
    }

    /// The "Pedal" tab: lane visibility (relocated from Roll & history) +
    /// input sensitivity. Always editable — a user can dial in settings before
    /// plugging in a keyboard — but only *effective* on MIDI input, since the
    /// mic path has no CC64 signal (the lane's render gate enforces that).
    fn prefs_pedal_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("Pedal");
        changed |= ui
            .checkbox(&mut self.prefs.pedal_lane_visible, "Show pedal lane")
            .on_hover_text(
                "Draw sustain-pedal (CC64) activity as a slim strip at the \
                 roll's left edge, tinted by pedal depth. Only takes effect \
                 with a MIDI keyboard — the mic path has no CC64 signal.",
            )
            .changed();
        ui.horizontal(|ui| {
            ui.label("Pedal sensitivity deadzone:");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.prefs.pedal_deadzone)
                        .range(0..=32)
                        .speed(0.1),
                )
                .on_hover_text(
                    "Minimum CC64 change (out of 127) before a new pedal \
                     position registers. Raise it to settle a jittery \
                     analog pedal; too high coarsens half-pedaling. \
                     0 = record every distinct level. Only takes effect \
                     with a MIDI keyboard.",
                )
                .changed();
        });
        if self.input.source() != Source::Midi {
            ui.weak(
                "Not in effect on mic input — the mic path has no pedal signal. \
                 Settings are saved and apply once a MIDI keyboard is connected.",
            );
        }
        changed
    }

    /// The "Roll behavior" tab — section auto-break timing (see roll.rs:
    /// threshold fires the break, tail trims the old section's blank end,
    /// lead-in pads before the new section's first note).
    fn prefs_roll_behavior_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("Roll behavior");
        changed |= limit_row(
            ui,
            "Section break threshold",
            &mut self.prefs.idle_pause,
            "Silence before the roll clock pauses, so the next note starts a \
             new section. Gaps shorter than this show on the paper in full; \
             once crossed, the blank tail is trimmed to the section tail \
             below. ∞ = never break (one continuous section).",
        );
        ui.horizontal(|ui| {
            ui.label("Section tail (s):");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.prefs.section_tail_s)
                        .range(0.0..=10.0)
                        .speed(0.1),
                )
                .on_hover_text(
                    "Blank paper kept after a section's last note when a break \
                     fires (trailing silence at the end of a session is \
                     trimmed to this too)",
                )
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Section lead-in (s):");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.prefs.section_lead_in_s)
                        .range(0.0..=10.0)
                        .speed(0.1),
                )
                .on_hover_text(
                    "Blank paper between a section boundary and the first note \
                     of the section that resumes there",
                )
                .changed();
        });
        changed
    }

    /// The "Audio / mic" tab: detection threshold + echo guard.
    fn prefs_audio_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("Audio / mic");
        ui.horizontal(|ui| {
            ui.label("Detection threshold:");
            if ui
                .add(egui::Slider::new(&mut self.prefs.threshold, 0.05..=0.95).step_by(0.01))
                .on_hover_text("Mic sensitivity: lower detects quieter notes (more noise)")
                .changed()
            {
                self.threshold = self.prefs.threshold;
                self.input.threshold.set(self.prefs.threshold);
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Echo hold-off (ms):");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.prefs.echo_holdoff_ms)
                        .range(0..=10_000)
                        .speed(50.0),
                )
                .on_hover_text(
                    "How long after the synth stops a note the mic keeps ignoring \
                     that note (stops the speaker→mic echo loop)",
                )
                .changed();
        });
        if ui
            .checkbox(&mut self.prefs.mic_muted, "Mute mic by default")
            .changed()
        {
            self.mic_muted = self.prefs.mic_muted;
            changed = true;
        }
        changed
    }

    /// The "Metronome" tab: bar length + per-beat click pitch/volume tables.
    fn prefs_metronome_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("Metronome");
        ui.horizontal(|ui| {
            ui.label("Beats per bar:");
            let mut bpb = self.prefs.metro_beats_per_bar;
            // Host-authoritative: a follower's edit would be silently reverted
            // by the next host beat marker (which carries beats_per_bar), while
            // its own UI kept showing the changed value — both ends then
            // inconsistent. Disable it unless we own the grid (M8).
            let authority = self.metro_authority();
            let bpb_resp = ui
                .add_enabled(authority, egui::DragValue::new(&mut bpb).range(1..=12))
                .on_disabled_hover_text("The host sets beats per bar for both players");
            if bpb_resp.changed() {
                bpb = bpb.max(1);
                self.prefs.metro_beats_per_bar = bpb;
                resize_beat_table(&mut self.prefs.metro_beat_freqs, bpb as usize, 1200.0);
                resize_beat_table(&mut self.prefs.metro_beat_volumes, bpb as usize, 1.0);
                self.metro.beats_per_bar = bpb;
                self.metro.beat_freqs = self.prefs.metro_beat_freqs.clone();
                self.metro.beat_volumes = self.prefs.metro_beat_volumes.clone();
                self.metro.next_beat_in_bar %= bpb;
                self.send_metro_table();
                changed = true;
            }
        });
        ui.label(
            egui::RichText::new(
                "Pitch and level of each beat's click, synced with the peer so both \
                 sides sound identical. The first slider quick-picks a common pitch; \
                 fine-tune with the Hz field next to it.",
            )
            .weak(),
        );
        for i in 0..self.prefs.metro_beat_freqs.len() {
            let is_accent = i == 0;
            egui::Frame::none()
                .fill(if is_accent {
                    ui.visuals().faint_bg_color
                } else {
                    egui::Color32::TRANSPARENT
                })
                .inner_margin(egui::Margin::symmetric(6.0, 3.0))
                .rounding(egui::Rounding::same(4.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        // Fixed-width label column so the controls line up across
                        // rows regardless of the accent's extra sub-label or the
                        // Hz field's digit count.
                        ui.allocate_ui(egui::vec2(76.0, 0.0), |ui| {
                            ui.vertical(|ui| {
                                ui.label(format!("Beat {}", i + 1));
                                if is_accent {
                                    ui.label(egui::RichText::new("accent").small().weak());
                                }
                            });
                        });

                        // Quick-pick slider: snaps the Hz field to a common preset.
                        // Widths below are pinned via `ui.spacing_mut()` (egui 0.29 has
                        // no per-widget `desired_width` on Slider/DragValue) so a beat
                        // with more Hz digits doesn't push its neighbors out of column.
                        let freq = self.prefs.metro_beat_freqs[i];
                        let mut preset = METRO_FREQ_PRESETS
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| {
                                (**a - freq).abs().total_cmp(&(**b - freq).abs())
                            })
                            .map(|(idx, _)| idx)
                            .unwrap_or(0);
                        ui.spacing_mut().slider_width = 56.0;
                        if ui
                            .add(
                                egui::Slider::new(&mut preset, 0..=METRO_FREQ_PRESETS.len() - 1)
                                    .show_value(false),
                            )
                            .on_hover_text("Quick-pick a common click pitch")
                            .changed()
                        {
                            self.prefs.metro_beat_freqs[i] = METRO_FREQ_PRESETS[preset];
                            self.metro.beat_freqs = self.prefs.metro_beat_freqs.clone();
                            self.send_metro_table();
                            changed = true;
                        }

                        ui.spacing_mut().interact_size.x = 64.0;
                        if ui
                            .add(
                                egui::DragValue::new(&mut self.prefs.metro_beat_freqs[i])
                                    .range(100.0..=4000.0)
                                    .speed(5.0)
                                    .suffix(" Hz"),
                            )
                            .changed()
                        {
                            self.metro.beat_freqs = self.prefs.metro_beat_freqs.clone();
                            self.send_metro_table();
                            changed = true;
                        }

                        ui.separator();
                        ui.spacing_mut().slider_width = 80.0;
                        if ui
                            .add(
                                egui::Slider::new(&mut self.prefs.metro_beat_volumes[i], 0.0..=1.0)
                                    .show_value(false)
                                    .text("volume"),
                            )
                            .changed()
                        {
                            self.metro.beat_volumes = self.prefs.metro_beat_volumes.clone();
                            self.send_metro_table();
                            changed = true;
                        }
                    });
                });
        }
        if ui.button("Reset metronome to defaults").clicked() {
            let d = prefs::Prefs::default();
            self.prefs.metro_beats_per_bar = d.metro_beats_per_bar;
            self.prefs.metro_beat_freqs = d.metro_beat_freqs.clone();
            self.prefs.metro_beat_volumes = d.metro_beat_volumes.clone();
            self.metro.beats_per_bar = d.metro_beats_per_bar;
            self.metro.beat_freqs = d.metro_beat_freqs;
            self.metro.beat_volumes = d.metro_beat_volumes;
            self.metro.next_beat_in_bar %= self.metro.beats_per_bar.max(1);
            self.send_metro_table();
            changed = true;
        }
        changed
    }

    /// The "Advanced" tab (model / network). No longer behind a
    /// `CollapsingHeader` — having its own tab already gates visibility.
    fn prefs_advanced_section(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("Advanced");
        ui.label(
            egui::RichText::new(
                "These affect mic transcription directly — bad values can \
                 stop notes being detected. Defaults suit most rooms.",
            )
            .weak(),
        );
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Silence RMS:");
            if ui
                .add(
                    egui::DragValue::new(&mut self.prefs.silence_rms)
                        .range(0.0..=0.05)
                        .speed(0.0002)
                        .fixed_decimals(4),
                )
                .on_hover_text("Below this raw mic level a window is treated as silence")
                .changed()
            {
                self.input.tunables.silence_rms.set(self.prefs.silence_rms);
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Norm max gain:");
            if ui
                .add(
                    egui::DragValue::new(&mut self.prefs.norm_max_gain)
                        .range(1.0..=100.0)
                        .speed(0.5),
                )
                .on_hover_text("Cap on how much quiet input is amplified before inference")
                .changed()
            {
                self.input.tunables.norm_max_gain.set(self.prefs.norm_max_gain);
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Frame off:");
            if ui
                .add(
                    egui::DragValue::new(&mut self.prefs.frame_off)
                        .range(0.01..=0.9)
                        .speed(0.005)
                        .fixed_decimals(3),
                )
                .on_hover_text("Probability below which a sounding note is released")
                .changed()
            {
                self.input.tunables.frame_off.set(self.prefs.frame_off);
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("MIDI poll (ms):");
            if ui
                .add(
                    egui::DragValue::new(&mut self.prefs.midi_poll_ms)
                        .range(100..=5_000)
                        .speed(10.0),
                )
                .on_hover_text("How often the app rescans for MIDI devices")
                .changed()
            {
                self.input
                    .midi_poll_ms
                    .store(self.prefs.midi_poll_ms, std::sync::atomic::Ordering::Relaxed);
                changed = true;
            }
        });
        ui.add_space(4.0);
        if ui
            .button("Reset advanced to defaults")
            .on_hover_text("Restore Silence RMS / Norm max gain / Frame off / MIDI poll")
            .clicked()
        {
            let d = prefs::Prefs::default();
            self.prefs.silence_rms = d.silence_rms;
            self.prefs.norm_max_gain = d.norm_max_gain;
            self.prefs.frame_off = d.frame_off;
            self.prefs.midi_poll_ms = d.midi_poll_ms;
            self.input.tunables.silence_rms.set(d.silence_rms);
            self.input.tunables.norm_max_gain.set(d.norm_max_gain);
            self.input.tunables.frame_off.set(d.frame_off);
            self.input
                .midi_poll_ms
                .store(d.midi_poll_ms, std::sync::atomic::Ordering::Relaxed);
            changed = true;
        }
        changed
    }

    /// The About window: running version, update status, and project links.
    /// Opened from the version chip in the status bar; the titlebar ✕ (wired
    /// through `.open()`) closes it.
    fn about_window(&mut self, ctx: &egui::Context) {
        if !self.show_about {
            return;
        }
        // Resolved before `.open()` takes its mutable borrow of `show_about`.
        let update_line = match self.updater.state() {
            update::UpdateState::Checking => "Checking for updates…".to_string(),
            update::UpdateState::UpToDate => "Up to date — this is the latest release".to_string(),
            update::UpdateState::Ready { version } => {
                format!("Update ready: v{version} (restart to apply)")
            }
            update::UpdateState::Failed { reason } => format!("Update check failed: {reason}"),
        };
        egui::Window::new("About")
            .open(&mut self.show_about)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.heading(format!("open-piano v{VERSION}"));
                ui.label("Real-time, peer-to-peer acoustic piano visualizer.");
                ui.add_space(6.0);
                ui.label(update_line);
                ui.add_space(6.0);
                ui.hyperlink_to(
                    "Source & releases (GitHub)",
                    "https://github.com/ja-ortiz-uniandes/open-piano",
                );
                ui.label("License: MIT or Apache-2.0");
            });
    }

    /// Drain the network event channel: session status (invite code ready,
    /// connect/disconnect) plus the peer's packets (notes, color).
    fn pump_network(&mut self) {
        let mut events = Vec::new();
        if let Some(peer) = &self.peer {
            while let Ok(event) = peer.events.try_recv() {
                events.push(event);
            }
        }
        for event in events {
            match event {
                NetEvent::Ticket(code) => self.my_ticket = Some(code),
                NetEvent::Status(s) => self.net_status = s,
                NetEvent::Connected => {
                    // Fresh connection: remote state is unknown (this may be a
                    // reconnect mid-chord), and the peer needs our color.
                    self.peer_connected = true;
                    self.clear_remote_keys();
                    self.send_color();
                    self.send_name();
                    // Only the metronome authority (host) broadcasts the click
                    // table on connect — a follower staying silent is what stops
                    // the two sides *swapping* tables (each sending before it
                    // processes the other's) and never converging (F2). The
                    // follower adopts the host's, and pushes its own edits to the
                    // host afterwards.
                    if self.metro_authority() {
                        self.send_metro_table();
                    }
                    // Announce our current live notes, pinned keys, and pedal so
                    // a chord/pin/pedal held across the (re)connect lights up on
                    // the peer immediately instead of waiting to be re-struck
                    // (H1/H2/M1).
                    self.broadcast_live(true);
                    self.broadcast_held(true);
                    if let Some(peer) = &self.peer {
                        peer.send(Packet::Pedal { level: self.local_pedal });
                    }
                    self.last_pedal_sent = self.local_pedal;
                    // If we're the metronome authority, announce current state so
                    // a follower syncs immediately (even when it's off). The
                    // per-beat markers handle the running case.
                    if self.metro_authority() {
                        if let Some(peer) = &self.peer {
                            peer.send(Packet::MetroBeat {
                                bpm: self.metro.bpm,
                                beat_in_bar: 0,
                                beats_per_bar: self.metro.beats_per_bar,
                                on: self.metro.enabled,
                            });
                        }
                    }
                }
                NetEvent::Disconnected => {
                    self.peer_connected = false;
                    self.clear_remote_keys();
                    // Drop the peer's announced name so a stale one isn't shown
                    // while nobody's connected; it re-announces on reconnect.
                    self.remote_name = DEFAULT_REMOTE_NAME.to_string();
                }
                NetEvent::Packet(Packet::Note(msg, seq)) => {
                    // Note events always apply (per-key transitions are
                    // independent; reordered events for different keys must not
                    // cancel), and advance the high-water mark so a stale Live
                    // snapshot that reordered past this note is later ignored (F6).
                    self.remote_live_seq = self.remote_live_seq.max(seq);
                    apply(&mut self.remote, msg);
                    self.roll.note(roll::Who::Remote, msg, self.remote_color);
                    // The peer's notes have no local sound source, so voice them.
                    self.play_synth(msg, synth::Channel::Peer);
                }
                NetEvent::Packet(Packet::Pedal { level }) => {
                    self.remote_pedal = level;
                    self.roll.pedal(roll::Who::Remote, level);
                }
                // The peer's Ctrl+click-pinned keys — a whole-state snapshot, so
                // just adopt it (idempotent; an empty mask clears the overlay
                // when they release Ctrl). Display only: no synth, no roll.
                NetEvent::Packet(Packet::Held { seq, mask }) => {
                    // Ignore a snapshot that reordered behind a newer one (F6).
                    if seq >= self.remote_held_seq {
                        self.remote_held_seq = seq;
                        self.remote_held = unpack_held(&mask);
                    }
                }
                // The peer's live (sounding) notes as a whole-state snapshot —
                // reconcile so any dropped note-on/off self-heals (H1), unless it
                // reordered behind a newer note/snapshot, which would resurrect a
                // released note or extinguish a fresh press (F6).
                NetEvent::Packet(Packet::Live { seq, mask }) => {
                    if seq >= self.remote_live_seq {
                        self.remote_live_seq = seq;
                        self.reconcile_remote_live(mask);
                    }
                }
                // The peer's manual segment breaks — fold each into our roll
                // (idempotent; `insert_separator` dedupes) so the break shows on
                // both screens (M2). Drop any time beyond our own live edge
                // instead of clamping it there: clamping made every heartbeat
                // re-delivery land at the (ever-advancing) live edge, spawning a
                // fresh separator line every second (F1). A break the peer placed
                // ahead of our clock is simply inserted once our clock reaches it.
                NetEvent::Packet(Packet::Separators(times)) => {
                    let now = self.roll.now_s();
                    for at in times {
                        if at > 0.0 && at <= now {
                            self.roll.insert_separator(at);
                        }
                    }
                }
                // Intentionally *not* applied to `remote_color`: every fresh
                // install defaults `local_color` to the same red, so honoring
                // an un-customized peer's announced color would render both
                // sides identically and defeat the two-color visualization.
                // The peer is pinned to `DEFAULT_REMOTE_COLOR` (blue) locally
                // regardless of what they pick on their own screen. We still
                // accept (and drop) the packet so the wire protocol is
                // unchanged.
                NetEvent::Packet(Packet::Color(_rgb)) => {}
                NetEvent::Packet(Packet::Name(name)) => self.remote_name = name,
                NetEvent::Packet(Packet::MetroCtl { on, bpm }) => {
                    // Follower → host request: the authority adopts it as the new
                    // grid and echoes authoritative state back.
                    if self.metro_authority() {
                        let was = self.metro.enabled;
                        self.metro.bpm = bpm.clamp(MIN_BPM, MAX_BPM);
                        self.metro.enabled = on;
                        if on && !was {
                            // Wait for the next roll-time-grid-aligned beat
                            // rather than clicking immediately.
                            self.metro_start_now();
                        } else if !on {
                            self.metro.next_beat_at = None;
                        }
                        if let Some(peer) = &self.peer {
                            peer.send(Packet::MetroBeat {
                                bpm: self.metro.bpm,
                                beat_in_bar: 0,
                                beats_per_bar: self.metro.beats_per_bar,
                                on,
                            });
                        }
                    }
                }
                // Markers should always arrive as NetEvent::MetroBeat (net.rs
                // splits them out to stamp RTT); ignore any that slip through.
                NetEvent::Packet(Packet::MetroBeat { .. }) => {}
                NetEvent::Packet(Packet::MetroBeatTable { freqs, volumes }) => {
                    // No authority here (see Packet::MetroBeatTable): whoever
                    // last edited wins on both ends. Adopt (and persist) only
                    // when the table actually differs — this is what stops the
                    // two peers oscillating X→Y→X every second and turns the
                    // per-packet `prefs.save()` disk-write storm into a write
                    // only on a genuine change (H3). Values arrive already
                    // finite/in-range (sanitized in `Packet::decode`).
                    if self.metro.beat_freqs != freqs || self.metro.beat_volumes != volumes {
                        self.metro.beat_freqs = freqs.clone();
                        self.metro.beat_volumes = volumes.clone();
                        self.prefs.metro_beat_freqs = freqs;
                        self.prefs.metro_beat_volumes = volumes;
                        self.prefs.save();
                    }
                }
                NetEvent::MetroBeat { bpm, beat_in_bar, beats_per_bar, on, one_way } => {
                    self.on_metro_beat(bpm, beat_in_bar, beats_per_bar, on, one_way);
                }
            }
        }
    }
}

/// Apply a note transition to a key-state array.
fn apply(keys: &mut [bool; KEY_COUNT], msg: NoteMsg) {
    if let Some(idx) = midi_to_key_index(msg.midi()) {
        keys[idx] = matches!(msg, NoteMsg::On(..));
    }
}

impl eframe::App for PianoApp {
    /// Called once as the app shuts down (window closed / Alt+F4). A debounced
    /// prefs save may still be pending — a slider edited within the last
    /// `PREFS_SAVE_DEBOUNCE` — and nothing else flushes it once `update` stops
    /// running, so force it here or the last edit is silently lost (F12).
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if self.prefs_save_due.take().is_some() {
            self.prefs.save();
        }
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frame_onsets.clear(); // repopulated by pump_input below
        self.pump_input();
        self.pump_network();
        self.sync_synth_to_source();
        self.drive_metronome();
        self.roll.tick(Instant::now());
        self.flush_prefs_save();

        // Restart the evaluation take if the key range was edited mid-take: the
        // range is frozen into `EvaluationState` at take start (H7), so a live
        // edit would otherwise leave the frozen `required` list disagreeing with
        // what the UI now shows — pause-on-miss would freeze at a note the user
        // just excluded (the "roll froze" symptom). Every other evaluation
        // setting already restarts the take; the range is the one that didn't
        // (M13).
        let current_range = self.playback.as_ref().and_then(|pb| pb.learn.key_range);
        let range_changed_mid_take = self
            .playback
            .as_ref()
            .is_some_and(|pb| pb.mode == playback::Mode::Evaluation)
            && current_range != self.last_key_range;
        if range_changed_mid_take {
            let live_midi = self.input.midi_connected();
            if let Some(pb) = &mut self.playback {
                pb.start_evaluation(live_midi, &self.synth);
            }
        }
        self.last_key_range = current_range;

        // Advance the loaded score's playhead (Listen auto-play / Learn
        // gating). `self.local` is already current from the pumps above.
        // dt is clamped: after a GUI-thread stall (the blocking rfd file
        // dialog, a window drag, a system sleep) the first frame's dt is the
        // whole stall — unclamped it would fling the playhead past the end
        // of the piece the moment a file is opened.
        if let Some(pb) = &mut self.playback {
            let dt = (ctx.input(|i| i.stable_dt) as f64).min(0.1);
            let held: std::collections::BTreeSet<u8> = (0..KEY_COUNT)
                .filter(|&i| self.local[i])
                .map(|i| MIDI_LOW + i as u8)
                .collect();
            // The extra args feed Evaluation's scorer; Listen/Learn ignore them.
            pb.tick(dt, &held, &self.frame_onsets, self.local_pedal, &self.synth);
            // A take just finished (Evaluation → EvaluationReview): pop the
            // results window.
            if pb.take_review_transition() {
                self.show_eval_results = true;
            }
        }

        // Intercept the window ✕ while the roll has unsaved notes: cancel the
        // close (must happen this same frame) and ask instead. `allow_close`
        // lets the close we re-issue from the dialog through.
        if ctx.input(|i| i.viewport().close_requested())
            && self.roll.has_unsaved()
            && !self.allow_close
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.show_close_confirm = true;
        }

        if ctx.input_mut(|i| i.consume_shortcut(&SAVE_SHORTCUT)) {
            self.save_roll_quick();
        }
        if ctx.input_mut(|i| i.consume_shortcut(&PREFS_SHORTCUT)) {
            self.show_prefs = true;
        }

        // Track how long the pointer has held still. This drives the two-stage
        // "keyboard locked" tooltip during recording: a short label on hover that
        // expands into the full explanation after ~1 s of the mouse not moving.
        let hover_pos = ctx.input(|i| i.pointer.hover_pos());
        let pointer_moved = match (hover_pos, self.last_pointer_pos) {
            (Some(now), Some(prev)) => prev.distance(now) > 2.0,
            _ => true,
        };
        if pointer_moved {
            self.pointer_still_since = Instant::now();
        }
        self.last_pointer_pos = hover_pos;
        let still_secs = self.pointer_still_since.elapsed().as_secs_f32();

        // Track touch vs. mouse/trackpad usage to size the resize handles.
        // egui synthesizes primary-pointer events from single-finger touch (so
        // drag-based UI just works), but a genuine `Event::Touch` rides
        // alongside *only* for touch input — never for mouse/trackpad. So a
        // touch event this frame means tablet use; a real pointer event with no
        // touch means mouse/trackpad. Idle frames leave the last verdict
        // standing, so it tracks "most recently used," not flickers.
        let (saw_touch, saw_pointer) = ctx.input(|i| {
            let mut touch = false;
            let mut pointer = false;
            for ev in &i.raw.events {
                match ev {
                    egui::Event::Touch { .. } => touch = true,
                    egui::Event::PointerMoved(_)
                    | egui::Event::PointerButton { .. }
                    | egui::Event::MouseWheel { .. } => pointer = true,
                    _ => {}
                }
            }
            (touch, pointer)
        });
        if saw_touch {
            self.touch_mode = true;
        } else if saw_pointer {
            self.touch_mode = false;
        }

        // Low-rate heartbeat carrying every idempotent shared-surface snapshot,
        // so a dropped datagram self-heals within a second regardless of
        // connect order (and the QUIC connection never idles out). Everything
        // here is a whole-state snapshot the receiver reconciles idempotently,
        // so re-sending is safe.
        if self.peer.is_some() && self.last_color_send.elapsed() >= COLOR_HEARTBEAT {
            self.send_color();
            self.send_name();
            // Live notes + pinned keys, re-sent unconditionally: the pin-clear
            // (H2) and a phrase's final note-off (H1) have no "next event" to
            // correct a lost datagram, so the periodic snapshot is the only
            // self-heal. Idempotent — the receiver acts only on real diffs.
            self.broadcast_live(true);
            self.broadcast_held(true);
            // Current pedal level, so a level held constant across a reconnect,
            // and a lost final release, both heal (M1).
            if let Some(peer) = &self.peer {
                peer.send(Packet::Pedal { level: self.local_pedal });
            }
            self.last_pedal_sent = self.local_pedal;
            // Manually-inserted segment breaks, so a Ctrl+click break shows on
            // both screens and a dropped one still converges (M2).
            self.broadcast_separators();
            // Metronome state, authority-only so it can't oscillate (F2/F11):
            if self.metro_authority() {
                // The click table — a single broadcaster (the host) means the
                // two sides can't swap or oscillate, and a dropped table-edit
                // heals within a second (F2).
                self.send_metro_table();
                // On/off state: a *running* metronome already streams per-beat
                // markers, but an off authority produces none — so re-announce
                // the off state here, or a dropped connect/toggle marker would
                // leave a follower clicking a stale grid forever (F11). (Sending
                // an on:true MetroBeat here would inject a spurious beat, so we
                // rely on the per-beat markers for the running case.)
                if !self.metro.enabled {
                    if let Some(peer) = &self.peer {
                        peer.send(Packet::MetroBeat {
                            bpm: self.metro.bpm,
                            beat_in_bar: 0,
                            beats_per_bar: self.metro.beats_per_bar,
                            on: false,
                        });
                    }
                }
            }
        }

        // ---- Custom title bar (File/Edit menus, window controls, edge resize).
        // MUST be the first panel shown so it's the topmost row. ----
        self.title_bar(ctx);
        // Reconcile the OS window against the compact toggle immediately, so a
        // click on the title-bar button lands the same frame.
        self.sync_compact_viewport(ctx);

        // ---- Top: networking + audio config, collapsible to a title strip
        // via the chevron. The collapsed/expanded variants animate between
        // two *distinct* panel ids — sharing one would corrupt the height
        // lerp `show_animated_between` stores per id. ----
        // Hidden entirely in compact mode (keyboard + title bar only).
        if !self.compact_mode {
            let collapsed_panel = egui::TopBottomPanel::top("config_collapsed")
                .resizable(false)
                .exact_height(24.0);
            let expanded_panel = egui::TopBottomPanel::top("config").resizable(false);
            egui::TopBottomPanel::show_animated_between(
                ctx,
                !self.config_collapsed,
                collapsed_panel,
                expanded_panel,
                // Branch on `config_collapsed`, not `how_expanded`: contents are
                // only drawn at the fully-collapsed and fully-expanded endpoints,
                // which line up exactly with that flag.
                |ui, _how_expanded| {
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        // ⏶/⏷, not ⌃/⌄: the latter aren't in egui's default
                        // fonts (they render as tofu boxes); these come from the
                        // same block as the transport glyphs, which do render.
                        let (chevron, hover) = if self.config_collapsed {
                            ("⏷", "Show settings")
                        } else {
                            ("⏶", "Hide settings")
                        };
                        if ui.small_button(chevron).on_hover_text(hover).clicked() {
                            self.config_collapsed = !self.config_collapsed;
                        }
                        ui.label(
                            egui::RichText::new(concat!("open-piano v", env!("CARGO_PKG_VERSION")))
                                .weak(),
                        );
                    });
                    if !self.config_collapsed {
                        ui.add_space(4.0);
                        self.config_panel_body(ui);
                    }
                },
            );
        }

        // ---- Bottom status bar (hidden in compact mode) ----
        if !self.compact_mode {
            egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    let lc = self.local_color;
                    let rc = self.remote_color;
                    ui.colored_label(egui::Color32::from_rgb(lc[0], lc[1], lc[2]), "■");
                    ui.label(format!("{} (you)", self.local_name));
                    ui.colored_label(egui::Color32::from_rgb(rc[0], rc[1], rc[2]), "■");
                    ui.label(format!("{} (peer)", self.remote_name));
                    ui.separator();
                    let (device, model) = {
                        // Recover the guard if a backend thread panicked while
                        // holding it, rather than cascading that panic into the
                        // GUI thread every frame (L5).
                        let s = self
                            .input
                            .status
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        (s.device.clone(), s.model.clone())
                    };
                    ui.label(device);
                    ui.separator();
                    ui.label(model);
                    ui.separator();
                    ui.label(&self.net_status);
                    // Pending section break: the roll clock paused past the
                    // break threshold, so the next note starts a new section.
                    // Same chip affordance as the title bar's "● unsaved".
                    if self.roll.is_paused() && !self.roll.is_empty() {
                        ui.separator();
                        ui.colored_label(
                            egui::Color32::from_rgb(210, 170, 60),
                            "● section break on next keypress",
                        )
                        .on_hover_text(
                            "The roll has been idle past the section-break \
                             threshold; the next note starts a new section \
                             (Preferences ▸ Roll behavior)",
                        );
                    }
                    // While the history roll is scrolled back (or easing home), an
                    // instant way out: a deliberate click deserves an immediate
                    // snap, unlike the idle timer's gentle ease.
                    if self.scrollback.is_some() {
                        ui.separator();
                        if ui
                            .small_button("⏵ Live")
                            .on_hover_text("Jump back to the live edge")
                            .clicked()
                        {
                            self.scrollback = None;
                            self.scrollback_idle_since = None;
                        }
                    }
                    // Version chip pinned to the right edge; opens the About window.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button(format!("v{VERSION}"))
                            .on_hover_text("About open-piano")
                            .clicked()
                        {
                            self.show_about = true;
                        }
                    });
                });
                ui.add_space(2.0);
            });
        }

        self.about_window(ctx);
        self.preferences_window(ctx);
        self.unsaved_dialog(ctx);
        self.refine_range_window(ctx);
        self.eval_results_window(ctx);
        // Side panels reserve space in show order — must precede CentralPanel.
        // Hidden in compact mode (only the keyboard is drawn). At most one of
        // these two shows at a time (they're gated on disjoint modes).
        if !self.compact_mode {
            self.learn_panel(ctx);
            self.evaluation_panel(ctx);
        }

        // ---- Center: the 88-key keyboard (also playable with the mouse),
        // the piano-roll history strip on the paper below it, and — with a
        // score loaded — the falling-notes panel above it ----
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            // Compact mode: the keyboard IS the window — no falling panel or
            // roll strip below, so it takes the full central-panel height.
            let kb_h = if self.compact_mode {
                avail.y
            } else if let Some(frac) = self.keyboard_height_frac {
                let lo_h = MIN_KEYBOARD_H.min(avail.y);
                let hi_h = (avail.y * MAX_KEYBOARD_FRACTION).max(lo_h).min(avail.y);
                (frac * avail.y).clamp(lo_h, hi_h)
            } else {
                (avail.y * KEYBOARD_FRACTION).max(MIN_KEYBOARD_H).min(avail.y)
            };

            // Falling-notes panel first (it sits on top, ending at the keys).
            // Its height animates open/closed on file open/close — the one
            // layout change big enough (up to 55% of the space under the
            // keyboard, in one frame) to read as a jarring pop otherwise. The
            // animate call must run every frame regardless of playback state
            // so its stored value keeps decaying — satisfied here, since the
            // CentralPanel closure always runs.
            let falling_factor = ctx.animate_bool_with_time(
                egui::Id::new("falling_panel_visible"),
                self.playback.is_some(),
                0.2,
            );
            let falling_full_h = ((avail.y - kb_h).max(0.0) * FALLING_FRACTION).floor();
            let falling_h = (falling_full_h * falling_factor).floor();
            let falling_resp = if self.compact_mode {
                // Compact: no falling panel (kb_h == avail.y makes its height 0
                // anyway, but be explicit — the keyboard owns the whole panel).
                None
            } else if falling_h > 0.0 {
                Some(ui.allocate_response(
                    egui::vec2(avail.x, falling_h),
                    egui::Sense::click_and_drag(),
                ))
            } else {
                None
            };

            let response =
                ui.allocate_response(egui::vec2(avail.x, kb_h), egui::Sense::click_and_drag());
            let rect = response.rect;

            // Reserve a thin sliver at the keyboard's left edge for the live
            // pedal indicator (drawn below), leaving a gap so it reads as a
            // separate element rather than part of key 1. Unlike the roll's
            // pedal-history lane, this shows even on mic input (so the peer's
            // pedal is still visible) — hence no `Source::Midi` gate. The keys
            // lay out in the shrunk `kb_rect`, keeping keyboard and roll aligned.
            let show_pedal_indicator = self.prefs.pedal_lane_visible;
            let kb_rect = if show_pedal_indicator {
                egui::Rect::from_min_max(
                    egui::pos2(rect.left() + PEDAL_INDICATOR_W + PEDAL_INDICATOR_GAP, rect.top()),
                    rect.max,
                )
            } else {
                rect
            };

            // Drag-to-resize handles on the keyboard's edges: thin strips
            // straddling the top (against the falling panel) and bottom
            // (against the roll strip). Registered after `response` so they
            // win hover/drag priority over the keyboard's own click-and-drag
            // for their overlap (same interact-last-wins convention as the
            // title bar). The dragged height lands next frame — consistent
            // with the rest of the immediate-mode drag state here.
            let lo_h = MIN_KEYBOARD_H.min(avail.y);
            let hi_h = (avail.y * MAX_KEYBOARD_FRACTION).max(lo_h).min(avail.y);

            if !self.compact_mode {
                let bottom_handle_rect = egui::Rect::from_min_max(
                    egui::pos2(rect.left(), rect.bottom() - KB_RESIZE_HANDLE_H / 2.0),
                    egui::pos2(rect.right(), rect.bottom() + KB_RESIZE_HANDLE_H / 2.0),
                );
                let handle = ui.interact(
                    bottom_handle_rect,
                    response.id.with("kb_resize_bottom"),
                    egui::Sense::drag(),
                );
                if handle.hovered() || handle.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                }
                if handle.dragged() {
                    // Bottom edge moving down (positive dy) grows the keyboard.
                    let new_h = (kb_h + handle.drag_delta().y).clamp(lo_h, hi_h);
                    self.keyboard_height_frac = Some(new_h / avail.y.max(1.0));
                }
                if handle.drag_stopped() {
                    self.prefs.keyboard_height_frac = self.keyboard_height_frac;
                    self.prefs.save();
                }
            }

            if falling_resp.is_some() {
                let top_handle_rect = egui::Rect::from_min_max(
                    egui::pos2(rect.left(), rect.top() - KB_RESIZE_HANDLE_H / 2.0),
                    egui::pos2(rect.right(), rect.top() + KB_RESIZE_HANDLE_H / 2.0),
                );
                let handle = ui.interact(
                    top_handle_rect,
                    response.id.with("kb_resize_top"),
                    egui::Sense::drag(),
                );
                if handle.hovered() || handle.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                }
                if handle.dragged() {
                    // Top edge moving up (negative dy) grows the keyboard, so negate.
                    let new_h = (kb_h - handle.drag_delta().y).clamp(lo_h, hi_h);
                    self.keyboard_height_frac = Some(new_h / avail.y.max(1.0));
                }
                if handle.drag_stopped() {
                    self.prefs.keyboard_height_frac = self.keyboard_height_frac;
                    self.prefs.save();
                }
            }

            // `command` = Ctrl on Windows / Cmd on macOS — the modifier for
            // both pinning a key (below) and showing the pinned set (further
            // down). Detect the release edge here so the pinned set clears
            // the instant Ctrl goes up, before anything reads `held` this
            // frame — memoryless, so the next Ctrl-hold starts empty instead
            // of resurrecting a chord pinned in some earlier, unrelated hold.
            let cmd = ui.input(|i| i.modifiers.command);
            if self.held_cmd_down && !cmd {
                self.held = [false; KEY_COUNT];
                // Tell the peer the shared overlay just cleared (all-zero mask).
                self.broadcast_held(false);
            }
            self.held_cmd_down = cmd;

            // Mouse play is disabled while capturing training data (a click
            // makes no real sound and isn't in the MIDI labels) and while a
            // score is open (clicking keys would fight the score's own
            // highlights; real MIDI/mic input is the Learn-mode instrument).
            if self.input.recorder.is_armed() || self.playback.is_some() {
                self.set_mouse_note(None); // drop any note the mouse was holding
                if self.input.recorder.is_armed() {
                    response.on_hover_ui(|ui| {
                        ui.set_max_width(260.0);
                        ui.label(
                            egui::RichText::new("🔒 Recording — keyboard locked").strong(),
                        );
                        if still_secs >= 1.0 {
                            ui.add_space(4.0);
                            ui.label(
                                "A mouse click makes no sound and isn't written to the \
                                 MIDI labels, so it would show notes that aren't in the \
                                 recorded audio. Stop recording to play with the mouse.",
                            );
                        }
                    });
                }
            } else {
                // Ctrl+click pins/unpins a key "held" for display (see `held`),
                // so suppress mouse-play while Ctrl is down — otherwise the same
                // press would also sound a note.
                // While the button is held, the key under the pointer is played;
                // dragging across keys glides note-to-note.
                let target = if response.is_pointer_button_down_on() && !cmd {
                    response.interact_pointer_pos().and_then(|p| key_at(kb_rect, p))
                } else {
                    None
                };
                self.set_mouse_note(target);

                // Ctrl+click toggles the pinned-held state of the clicked key.
                // A display aid — no sound, no roll — but shared with the peer
                // (see `held` / `Packet::Held`) so both see the same overlay.
                if cmd && response.clicked() {
                    if let Some(idx) = response
                        .interact_pointer_pos()
                        .and_then(|p| key_at(kb_rect, p))
                        .and_then(midi_to_key_index)
                    {
                        self.held[idx] = !self.held[idx];
                        self.broadcast_held(false);
                    }
                }
            }

            // The keyboard's key layout doubles as lane geometry for both
            // roll panels, so marks are exactly x-aligned with their keys.
            let keys = layout_keys(kb_rect);

            // Ctrl+click-pinned keys render exactly like a live press (in that
            // player's color), so OR them into the key arrays used for drawing —
            // display only; nothing downstream (synth/roll) sees `held`. Local
            // pins show *while Ctrl is held*; releasing Ctrl hides and clears
            // `held` (see the release-edge check above). Remote pins show
            // whenever the peer has any (they broadcast the all-zero clear on
            // their own Ctrl release), keeping the shared overlay in sync.
            let mut local_shown = self.local;
            if cmd {
                for i in 0..KEY_COUNT {
                    local_shown[i] |= self.held[i];
                }
            }
            let mut remote_shown = self.remote;
            for i in 0..KEY_COUNT {
                remote_shown[i] |= self.remote_held[i];
            }

            if let Some(pb) = &self.playback {
                // Layered keyboard: your live presses, plus each unpracticed
                // (and, in review, visible) track's currently-playing notes in
                // its own color. `display_score` swaps in the original+played
                // pair while reviewing an evaluation.
                let mut layers: Vec<([bool; KEY_COUNT], egui::Color32)> = vec![(
                    local_shown,
                    egui::Color32::from_rgb(
                        self.local_color[0],
                        self.local_color[1],
                        self.local_color[2],
                    ),
                )];
                // The peer's pinned overlay is part of the shared surface, so it
                // shows even while we have a score open (our own pinning is
                // locked during playback, but theirs may not be).
                if self.remote_held.iter().any(|&h| h) {
                    layers.push((
                        self.remote_held,
                        egui::Color32::from_rgb(
                            self.remote_color[0],
                            self.remote_color[1],
                            self.remote_color[2],
                        ),
                    ));
                }
                for who in [roll::Who::Local, roll::Who::Remote] {
                    if !pb.practiced(who) && pb.track_visible(who) {
                        let [r, g, b] = pb.display_score().tracks[who.idx()].color;
                        layers.push((pb.active_key_array(who), egui::Color32::from_rgb(r, g, b)));
                    }
                }
                draw_keyboard_layered(ui.painter(), kb_rect, &keys, &layers);
            } else {
                let colors = KeyColors {
                    local: egui::Color32::from_rgb(self.local_color[0], self.local_color[1], self.local_color[2]),
                    remote: egui::Color32::from_rgb(self.remote_color[0], self.remote_color[1], self.remote_color[2]),
                };
                draw_keyboard(ui.painter(), kb_rect, &local_shown, &remote_shown, colors);
            }

            // Live pedal indicator: a thin sliver in the reserved strip left of
            // the keyboard, showing both players' current pedal depth (local +
            // remote), split diagonally like a simultaneous same-key press.
            if show_pedal_indicator {
                let strip = egui::Rect::from_min_max(
                    rect.left_top(),
                    egui::pos2(rect.left() + PEDAL_INDICATOR_W, rect.bottom()),
                );
                draw_pedal_indicator(
                    ui.painter(),
                    strip,
                    self.local_pedal,
                    self.remote_pedal,
                    self.local_color,
                    self.remote_color,
                );
            }

            // Falling-panel interactions + drawing. The background is painted
            // unconditionally: mid-close-animation the panel still has height
            // but no playback to draw, and a bare "hole" would flash there.
            // (`falling_panel_interactions` no-ops safely without playback.)
            if let Some(falling_resp) = falling_resp {
                ui.painter()
                    .rect_filled(falling_resp.rect, 0.0, egui::Color32::from_gray(18));
                self.falling_panel_interactions(ui, &falling_resp, &keys);
                if let Some(pb) = &self.playback {
                    // In EvaluationReview `display_score` is the synthetic
                    // original+played pair and the visibility flags follow the
                    // review toggles; everywhere else both tracks always show.
                    let visible =
                        [pb.track_visible(roll::Who::Local), pb.track_visible(roll::Who::Remote)];
                    draw_falling(
                        ui.painter(),
                        falling_resp.rect,
                        &keys,
                        pb.display_score(),
                        pb.playhead_s,
                        pb.learn.key_range,
                        self.range_drag,
                        self.falling_scrollback.unwrap_or(0.0),
                        self.prefs.roll_px_per_s,
                        self.score_roll_origin_s,
                        visible,
                    );
                }
            }

            // ---- The roll strip: everything below the keyboard. Dragging it
            // reviews history; releasing eases back to the live edge.
            // (Hidden in compact mode — the keyboard already took the full
            // height, so there's no space below it anyway.) ----
            if !self.compact_mode {
                let roll_resp =
                    ui.allocate_response(ui.available_size(), egui::Sense::click_and_drag());
                // Dragging up (negative delta.y) pulls older paper into view: a
                // mark sits at y ∝ (view_top − t), so moving marks up means
                // lowering the view-top time. Wheel/trackpad scroll does the same.
                let px_per_s = self.prefs.roll_px_per_s;
                let scrollback_idle_s = self.prefs.scrollback_idle_s;
                let roll_delta_px = scroll_or_drag_delta_y(ctx, &roll_resp);
                if roll_delta_px != 0.0 {
                    let top = self.scrollback.unwrap_or(self.roll.now_s());
                    let new_top = (top + (roll_delta_px / px_per_s) as f64)
                        .clamp(0.0, self.roll.now_s());
                    self.scrollback = Some(new_top);
                    self.scrollback_idle_since = None;
                } else if let Some(top) = self.scrollback {
                    // No input: hold the view for the idle window, then ease home.
                    let idle = *self.scrollback_idle_since.get_or_insert_with(Instant::now);
                    if idle.elapsed().as_secs_f64() >= scrollback_idle_s {
                        let dt = ctx.input(|i| i.stable_dt) as f64;
                        self.scrollback = ease_toward(top, self.roll.now_s(), dt, px_per_s);
                        if self.scrollback.is_none() {
                            self.scrollback_idle_since = None;
                        }
                    }
                }
                let view_top_s = self.scrollback.unwrap_or(self.roll.now_s());

                // Manual instance breaks on the history roll: Ctrl+click inserts
                // one at the clicked time; right-click offers the same via menu.
                let roll_t_of_y =
                    |y: f32| view_top_s - ((y - roll_resp.rect.top()) / px_per_s) as f64;
                if roll_resp.clicked() && ui.input(|i| i.modifiers.command) {
                    if let Some(pos) = roll_resp.interact_pointer_pos() {
                        self.insert_manual_separator(roll_t_of_y(pos.y));
                    }
                }
                if roll_resp.secondary_clicked() {
                    if let Some(pos) = roll_resp.interact_pointer_pos() {
                        self.pending_break_t = Some(roll_t_of_y(pos.y));
                    }
                }
                roll_resp.context_menu(|ui| {
                    if ui.button("Insert segment break here").clicked() {
                        if let Some(t) = self.pending_break_t.take() {
                            self.insert_manual_separator(t);
                        }
                        ui.close_menu();
                    }
                });

                // Pedal lane visibility follows the pref, and shows whenever
                // *either* side has pedal history — our own (MIDI input) or the
                // peer's. Gating on our local source alone hid a MIDI peer's
                // pedal from a mic-input player, contradicting the live
                // indicator's own rule and the shared-surface principle (M3).
                let have_remote_pedal =
                    self.roll.pedal_segments.iter().any(|p| p.who == roll::Who::Remote);
                let pedal_lane = (self.prefs.pedal_lane_visible
                    && (self.input.source() == Source::Midi || have_remote_pedal))
                    .then_some((self.local_color, self.remote_color));
                draw_roll(
                    ui.painter(),
                    roll_resp.rect,
                    &keys,
                    &self.roll,
                    view_top_s,
                    px_per_s,
                    pedal_lane,
                );
            }
        });

        // Keep redrawing for smooth real-time updates.
        ctx.request_repaint();
    }
}

/// The two players' currently-selected note colors, resolved for rendering.
#[derive(Clone, Copy)]
struct KeyColors {
    local: egui::Color32,
    remote: egui::Color32,
}

/// Paint one key according to who is pressing it. When *both* players hold the
/// same key, the key is split along its diagonal — local color in the upper-left
/// triangle, peer color in the lower-right — so a simultaneous press is
/// unmistakable regardless of which two colors were chosen (no ambiguous blend).
fn paint_key(
    painter: &egui::Painter,
    rect: egui::Rect,
    local_on: bool,
    remote_on: bool,
    colors: KeyColors,
    base: egui::Color32,
    rounding: f32,
    stroke: egui::Stroke,
) {
    match (local_on, remote_on) {
        (true, true) => {
            // Base fill + two triangles split along the anti-diagonal.
            painter.rect(rect, rounding, base, stroke);
            let tl = rect.left_top();
            let tr = rect.right_top();
            let bl = rect.left_bottom();
            let br = rect.right_bottom();
            let no_stroke = egui::Stroke::NONE;
            painter.add(egui::Shape::convex_polygon(vec![tl, tr, bl], colors.local, no_stroke));
            painter.add(egui::Shape::convex_polygon(vec![tr, br, bl], colors.remote, no_stroke));
            // Re-draw the border on top so the split sits inside a clean outline.
            painter.rect_stroke(rect, rounding, stroke);
        }
        (true, false) => {
            painter.rect(rect, rounding, colors.local, stroke);
        }
        (false, true) => {
            painter.rect(rect, rounding, colors.remote, stroke);
        }
        (false, false) => {
            painter.rect(rect, rounding, base, stroke);
        }
    }
}

/// Geometry of one painted key: its MIDI note, on-screen rect, and whether it is
/// a black key. Shared by `draw_keyboard` and pointer hit-testing (`key_at`) so
/// the *visible* layout and the *clickable* layout can never drift apart.
struct KeyRect {
    midi: u8,
    rect: egui::Rect,
    black: bool,
}

/// Lay out all 88 keys within `rect`. White keys tile the full width; black keys
/// overlay the white-key boundaries at 60% width and 62% height. Returned in
/// back-to-front paint order (white keys first, then black keys on top) — which
/// is also why `key_at` scans the slice *in reverse*, so the black keys drawn on
/// top win the hit-test.
fn layout_keys(rect: egui::Rect) -> Vec<KeyRect> {
    // White-key MIDI notes in order (52 of them across 88 keys).
    let white_midis: Vec<u8> = (MIDI_LOW..=MIDI_HIGH).filter(|m| !is_black_key(*m)).collect();
    let white_count = white_midis.len().max(1);
    let white_w = rect.width() / white_count as f32;
    let key_top = rect.top();
    let key_bottom = rect.bottom();

    let mut keys = Vec::with_capacity(KEY_COUNT);

    // White keys.
    for (col, &midi) in white_midis.iter().enumerate() {
        let x0 = rect.left() + col as f32 * white_w;
        keys.push(KeyRect {
            midi,
            rect: egui::Rect::from_min_max(
                egui::pos2(x0, key_top),
                egui::pos2(x0 + white_w, key_bottom),
            ),
            black: false,
        });
    }

    // Black keys: each sits over the boundary just right of white key (midi - 1).
    let black_w = white_w * 0.6;
    let black_h = (key_bottom - key_top) * 0.62;
    for midi in MIDI_LOW..=MIDI_HIGH {
        if !is_black_key(midi) {
            continue;
        }
        let Some(left_col) = white_midis.iter().position(|&m| m == midi - 1) else {
            continue;
        };
        let boundary_x = rect.left() + (left_col as f32 + 1.0) * white_w;
        let x0 = boundary_x - black_w / 2.0;
        keys.push(KeyRect {
            midi,
            rect: egui::Rect::from_min_max(
                egui::pos2(x0, key_top),
                egui::pos2(x0 + black_w, key_top + black_h),
            ),
            black: true,
        });
    }

    keys
}

/// The MIDI note of the key under `pos`, or `None`. Black keys sit on top, so we
/// test them first by scanning the back-to-front layout in reverse.
fn key_at(rect: egui::Rect, pos: egui::Pos2) -> Option<u8> {
    layout_keys(rect)
        .iter()
        .rev()
        .find(|k| k.rect.contains(pos))
        .map(|k| k.midi)
}

/// Render the 88-key piano: white keys first, then black keys on top.
fn draw_keyboard(
    painter: &egui::Painter,
    rect: egui::Rect,
    local: &[bool; KEY_COUNT],
    remote: &[bool; KEY_COUNT],
    colors: KeyColors,
) {
    // Background.
    painter.rect_filled(rect, 0.0, egui::Color32::from_gray(30));

    let white_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(60));
    let black_stroke = egui::Stroke::new(1.0, egui::Color32::BLACK);

    for key in layout_keys(rect) {
        let idx = midi_to_key_index(key.midi).unwrap();
        let (base, stroke) = if key.black {
            (egui::Color32::from_gray(20), black_stroke)
        } else {
            (egui::Color32::from_gray(245), white_stroke)
        };
        paint_key(painter, key.rect, local[idx], remote[idx], colors, base, 2.0, stroke);
    }
}

/// Render the piano-roll history strip: paper that scrolls down from the
/// keyboard, where every note leaves a mark in its key's lane.
///
/// `keys` is the *keyboard's* layout, so lanes are exactly x-aligned with the
/// keys above. `view_top_s` is the roll time at the strip's top edge — equal
/// to `roll.now_s()` when live, older while drag-reviewing.
///
/// Paint order: ruler (1 s / 10 s gridlines + timestamps), instance
/// separators, then marks — white-key lanes first and black on top, mirroring
/// the keyboard, with black marks naturally thinner (black-key width) and
/// drawn darker.
fn draw_roll(
    painter: &egui::Painter,
    rect: egui::Rect,
    keys: &[KeyRect],
    roll: &roll::Roll,
    view_top_s: f64,
    px_per_s: f32,
    pedal_lane: Option<([u8; 3], [u8; 3])>,
) {
    let painter = painter.with_clip_rect(rect);
    painter.rect_filled(rect, 0.0, egui::Color32::from_gray(18));

    // Roll time -> y. Subtract in f64 *before* the f32 cast: after a long
    // session the times are large and f32 subtraction would jitter the marks.
    let y_of = |t: f64| rect.top() + ((view_top_s - t) as f32) * px_per_s;

    // ---- Pedal lane (before the ruler, so its timestamps stay legible on
    // top of the lane fills).
    if let Some((local_color, remote_color)) = pedal_lane {
        draw_pedal_lane(&painter, rect, roll, view_top_s, px_per_s, local_color, remote_color);
    }

    // ---- Ruler: only the whole seconds inside the visible window (top edge
    // = view_top_s, bottom edge = view_top_s - height/speed), never the whole
    // session, so cost is O(strip height) regardless of duration.
    let bottom_s = view_top_s - (rect.height() / px_per_s) as f64;
    draw_ruler(&painter, rect, bottom_s, view_top_s, y_of, 0.0);

    // ---- Instance separators: full-width lines where the clock resumed
    // after an idle pause (or a break was inserted by hand).
    let sep_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(110));
    for sep in &roll.separators {
        let y = y_of(sep.at);
        if y >= rect.top() && y <= rect.bottom() {
            painter.hline(rect.x_range(), y, sep_stroke);
        }
    }

    // ---- Note marks. Lane x-extents per MIDI note, from the keyboard layout.
    let mut lanes: [Option<(egui::Rangef, bool)>; 128] = [None; 128];
    for k in keys {
        lanes[k.midi as usize] = Some((k.rect.x_range(), k.black));
    }

    // `segments` is sorted by `start_s`, so binary-search the slice that can
    // intersect the visible window instead of scanning the whole session every
    // frame (which grew unbounded — the ruler already fixed this for itself).
    // Upper bound: the first segment starting above the view top (future).
    // Lower bound: `bottom_s` minus a generous look-back, so a note held down
    // across the window's bottom edge (its `start_s` is earlier) is still
    // included; anything held longer than the look-back is not realistic (M20).
    let hi = roll.segments.partition_point(|s| s.start_s <= view_top_s);
    let lo = roll
        .segments
        .partition_point(|s| s.start_s < bottom_s - MARK_LOOKBACK_S);
    let visible = &roll.segments[lo..hi];

    // Fixed half-lane split: local always left, remote always right, so
    // simultaneous same-key presses sit side by side with no overlap logic —
    // the roll's analogue of `paint_key`'s diagonal split.
    for pass_black in [false, true] {
        for seg in visible {
            let Some((xr, black)) = lanes[seg.midi as usize] else { continue };
            if black != pass_black {
                continue;
            }
            let end = seg.end_s.unwrap_or(roll.now_s()); // held notes extend live
            let y_top = y_of(end);
            let y_bot = y_of(seg.start_s);
            if y_bot < rect.top() || y_top > rect.bottom() {
                continue;
            }
            let mid = xr.center();
            let (x0, x1) = match seg.who {
                roll::Who::Local => (xr.min + 1.0, mid),
                roll::Who::Remote => (mid, xr.max - 1.0),
            };
            // Velocity tint (saturation) first, then the black-key darken
            // (brightness) — orthogonal effects, applied in that order.
            let tinted = velocity_tint(seg.color, seg.velocity);
            let color = if black {
                // Black-key marks: same hue, dimmed — thinner comes free from
                // the black key's narrower lane.
                let dark = |c: u8| (c as f32 * 0.72) as u8;
                egui::Color32::from_rgb(dark(tinted.r()), dark(tinted.g()), dark(tinted.b()))
            } else {
                tinted
            };
            // Enforce a 2 px minimum so staccato notes still leave a sliver.
            let mark = egui::Rect::from_min_max(
                egui::pos2(x0, y_top),
                egui::pos2(x1, y_bot.max(y_top + 2.0)),
            );
            painter.rect_filled(mark, 2.0, color);
        }
    }
}

/// How far below the visible window's bottom edge (roll seconds) the mark /
/// pedal-span scan looks back, so a note or pedal held *across* that edge (its
/// start is earlier) is still drawn. Bounds the per-frame scan to the visible
/// window rather than the whole (uncapped) session; anything held longer than
/// this is not a realistic performance (M20).
const MARK_LOOKBACK_S: f64 = 600.0;

/// Width of the sustain-pedal lane at the history roll's left edge.
const PEDAL_LANE_W: f32 = 10.0;

/// Width of the live pedal indicator sliver drawn to the left of the keyboard,
/// and the gap between it and key 1 (so it reads as a separate element).
const PEDAL_INDICATOR_W: f32 = 8.0;
const PEDAL_INDICATOR_GAP: f32 = 4.0;

/// Paint the live pedal indicator: a thin vertical sliver showing both players'
/// current sustain-pedal depth. The sliver is split along its diagonal — local
/// in the upper-left triangle, remote in the lower-right — the same visual
/// language as `paint_key`'s simultaneous-press split, so a local + remote press
/// is legible at once. Each triangle fades from the unlit background toward its
/// player's full color as that side's pedal depth (CC64, 0..=127) rises, so a
/// side with the pedal up simply stays dark (e.g. yours, on mic input, while the
/// peer's half still lights).
fn draw_pedal_indicator(
    painter: &egui::Painter,
    strip: egui::Rect,
    local_pedal: u8,
    remote_pedal: u8,
    local_color: [u8; 3],
    remote_color: [u8; 3],
) {
    const BG: f32 = 26.0; // matches the unlit lane gray used elsewhere
    let bg = egui::Color32::from_gray(BG as u8);
    // Depth → brightness: lerp from the background gray toward the full color.
    let depth = |color: [u8; 3], level: u8| {
        let t = (level as f32 / 127.0).clamp(0.0, 1.0);
        let mix = |c: u8| (BG + (c as f32 - BG) * t) as u8;
        egui::Color32::from_rgb(mix(color[0]), mix(color[1]), mix(color[2]))
    };

    painter.rect_filled(strip, 2.0, bg);
    let tl = strip.left_top();
    let tr = strip.right_top();
    let bl = strip.left_bottom();
    let br = strip.right_bottom();
    let no_stroke = egui::Stroke::NONE;
    painter.add(egui::Shape::convex_polygon(
        vec![tl, tr, bl],
        depth(local_color, local_pedal),
        no_stroke,
    ));
    painter.add(egui::Shape::convex_polygon(
        vec![tr, br, bl],
        depth(remote_color, remote_pedal),
        no_stroke,
    ));
    // A faint outline so the sliver stays visible even fully unlit.
    painter.rect_stroke(strip, 2.0, egui::Stroke::new(1.0, egui::Color32::from_gray(60)));
}

/// Velocity → color mapping for roll marks: soft presses desaturate toward
/// gray, hard ones keep the player's full color. Never darkens toward black —
/// brightness is reserved for the black-key dimming, an orthogonal effect
/// applied after this. The curve is a visual tuning knob.
fn velocity_tint(color: [u8; 3], velocity: u8) -> egui::Color32 {
    let mut hsva = egui::ecolor::Hsva::from(egui::Color32::from_rgb(color[0], color[1], color[2]));
    let t = (velocity as f32 / 127.0).clamp(0.0, 1.0);
    // Ease-out before the linear remap, so saturation climbs toward the full
    // base color through the upper-middle of the range: a linear map left
    // typical hard presses (velocity ~90-115 — real playing rarely reaches
    // the literal 127) at a perceptibly dull 80-90% of base, while an
    // ease-out puts them near full color and leaves very soft touches close
    // to their old muted look. The exponent is a by-eye knob.
    let eased = 1.0 - (1.0 - t).powi(2);
    hsva.s *= 0.35 + 0.65 * eased;
    hsva.into()
}

/// The sustain-pedal lane: a slim strip at the roll's left edge sharing the
/// note lanes' time axis. One lane serves both players — where only one side
/// has the pedal down the strip fills with that side's color tinted by depth
/// (the same mapping as note velocity); where both are down at once, the
/// overlap is split along its diagonal, local upper-left / remote lower-right
/// — the same visual language as `paint_key`'s simultaneous-press split,
/// squeezed into one lane.
fn draw_pedal_lane(
    painter: &egui::Painter,
    rect: egui::Rect,
    roll: &roll::Roll,
    view_top_s: f64,
    px_per_s: f32,
    local_color: [u8; 3],
    remote_color: [u8; 3],
) {
    let lane = egui::Rect::from_min_max(
        rect.left_top(),
        egui::pos2(rect.left() + PEDAL_LANE_W, rect.bottom()),
    );
    painter.rect_filled(lane, 0.0, egui::Color32::from_gray(26));

    let y_of = |t: f64| rect.top() + ((view_top_s - t) as f32) * px_per_s;
    let bottom_s = view_top_s - (rect.height() / px_per_s) as f64;

    // Visible spans per player as (start, end, level); open ones extend live.
    // `pedal_segments` is start-sorted, so binary-search the visible window
    // rather than scanning every span each frame (M20, matching the note marks).
    let phi = roll.pedal_segments.partition_point(|p| p.start_s <= view_top_s);
    let plo = roll
        .pedal_segments
        .partition_point(|p| p.start_s < bottom_s - MARK_LOOKBACK_S);
    let spans = |who: roll::Who| -> Vec<(f64, f64, u8)> {
        roll.pedal_segments[plo..phi]
            .iter()
            .filter(|p| p.who == who)
            .map(|p| (p.start_s, p.end_s.unwrap_or(roll.now_s()), p.level))
            .filter(|(s, e, _)| *e >= bottom_s && *s <= view_top_s)
            .collect()
    };
    let local = spans(roll::Who::Local);
    let remote = spans(roll::Who::Remote);

    // 2 px minimum so a quick pedal tap still leaves a sliver.
    let rect_of = |s: f64, e: f64| {
        let y_top = y_of(e);
        egui::Rect::from_min_max(
            egui::pos2(lane.left(), y_top),
            egui::pos2(lane.right(), y_of(s).max(y_top + 2.0)),
        )
    };

    // Each side in full first; overlaps then repaint on top as the split.
    for &(s, e, level) in &local {
        painter.rect_filled(rect_of(s, e), 0.0, velocity_tint(local_color, level));
    }
    for &(s, e, level) in &remote {
        painter.rect_filled(rect_of(s, e), 0.0, velocity_tint(remote_color, level));
    }
    for &(ls, le, ll) in &local {
        for &(rs, re, rl) in &remote {
            let (s, e) = (ls.max(rs), le.min(re));
            if e <= s {
                continue;
            }
            let r = rect_of(s, e);
            let no_stroke = egui::Stroke::NONE;
            painter.add(egui::Shape::convex_polygon(
                vec![r.left_top(), r.right_top(), r.left_bottom()],
                velocity_tint(local_color, ll),
                no_stroke,
            ));
            painter.add(egui::Shape::convex_polygon(
                vec![r.right_top(), r.right_bottom(), r.left_bottom()],
                velocity_tint(remote_color, rl),
                no_stroke,
            ));
        }
    }
}

/// Net vertical input for a hand-rolled scrollable panel this frame: an
/// active drag wins, otherwise wheel/trackpad scroll while hovered. The hover
/// gate matters — `smooth_scroll_delta` is a global per-frame value, and
/// without it a scroll over the config panel would leak into the rolls.
fn scroll_or_drag_delta_y(ctx: &egui::Context, resp: &egui::Response) -> f32 {
    if resp.dragged() {
        resp.drag_delta().y
    } else if resp.hovered() {
        ctx.input(|i| i.smooth_scroll_delta.y)
    } else {
        0.0
    }
}

/// Which caption button to draw (`Maximize(true)` = currently maximized, so it
/// shows the "restore" glyph).
#[derive(Clone, Copy)]
enum WinBtn {
    Minimize,
    Maximize(bool),
    Close,
    /// Compact mode toggle (`true` = currently compact). An "alternate
    /// minimize" — see `PianoApp::sync_compact_viewport`.
    Compact(bool),
}

/// Draw one custom window-caption button and return whether it was clicked. The
/// glyphs are *painted* (line/rect strokes), not font characters, so they can't
/// fall back to a tofu box the way `🗕🗖🗙` might in egui's default font (the
/// codebase already learned some glyphs don't render — see the chevron note).
/// Close highlights red on hover, matching the OS button.
fn window_button(ui: &mut egui::Ui, kind: WinBtn) -> bool {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(32.0, 24.0), egui::Sense::click());
    let close = matches!(kind, WinBtn::Close);
    let hovered = resp.hovered();
    if hovered {
        let bg = if close {
            egui::Color32::from_rgb(200, 60, 60)
        } else {
            ui.visuals().widgets.hovered.bg_fill
        };
        ui.painter().rect_filled(rect, 2.0, bg);
    }
    let fg = if hovered && close {
        egui::Color32::WHITE
    } else {
        ui.visuals().text_color()
    };
    let stroke = egui::Stroke::new(1.2, fg);
    let c = rect.center();
    let e = 4.5;
    let painter = ui.painter();
    match kind {
        WinBtn::Minimize => {
            painter.hline(egui::Rangef::new(c.x - e, c.x + e), c.y + e - 0.5, stroke);
        }
        WinBtn::Maximize(false) => {
            painter.rect_stroke(
                egui::Rect::from_center_size(c, egui::vec2(2.0 * e, 2.0 * e)),
                0.0,
                stroke,
            );
        }
        WinBtn::Maximize(true) => {
            // Restore: two overlapping square outlines (back + front).
            let side = 2.0 * e - 1.5;
            painter.rect_stroke(
                egui::Rect::from_min_size(egui::pos2(c.x - e + 2.0, c.y - e), egui::vec2(side, side)),
                0.0,
                stroke,
            );
            painter.rect_stroke(
                egui::Rect::from_min_size(egui::pos2(c.x - e, c.y - e + 2.0), egui::vec2(side, side)),
                0.0,
                stroke,
            );
        }
        WinBtn::Close => {
            painter.line_segment([egui::pos2(c.x - e, c.y - e), egui::pos2(c.x + e, c.y + e)], stroke);
            painter.line_segment([egui::pos2(c.x - e, c.y + e), egui::pos2(c.x + e, c.y - e)], stroke);
        }
        WinBtn::Compact(active) => {
            // Keyboard-only silhouette: a wide low outline, with the bottom
            // "keys" bar filled in while compact mode is active.
            let outline = egui::Rect::from_center_size(c, egui::vec2(2.0 * e + 2.0, 2.0 * e - 2.0));
            painter.rect_stroke(outline, 0.0, stroke);
            if active {
                let keys = egui::Rect::from_min_max(
                    egui::pos2(outline.left(), outline.bottom() - 3.0),
                    outline.max,
                );
                painter.rect_filled(keys.shrink(1.0), 0.0, fg);
            }
        }
    }
    let clicked = resp.clicked();
    // Unlike Minimize/Maximize/Close this isn't a self-explanatory OS
    // convention, so it gets a tooltip.
    if let WinBtn::Compact(active) = kind {
        resp.on_hover_text(if active {
            "Exit compact mode"
        } else {
            "Compact mode — piano only"
        });
    }
    clicked
}

/// Inner content of the Evaluation side panel during a live take. Free fn
/// (not a method) because the panel closure already holds the playback
/// borrow. Any change restarts the take — a half-scored pass under changed
/// rules isn't meaningful, and `EvaluationState` freezes its rules at start.
fn evaluation_settings_body(
    ui: &mut egui::Ui,
    pb: &mut playback::PlaybackEngine,
    live_midi: bool,
    synth: &synth::Synth,
) {
    use playback::Strictness;
    let mut changed = false;

    ui.label("Track to perform:");
    changed |= ui.radio_value(&mut pb.eval.evaluate, None, "None").changed();
    changed |= ui
        .radio_value(&mut pb.eval.evaluate, Some(roll::Who::Local), "Local track")
        .changed();
    changed |= ui
        .radio_value(&mut pb.eval.evaluate, Some(roll::Who::Remote), "Remote track")
        .changed();
    match pb.eval.evaluate {
        None => {
            ui.weak("Pick a track to be scored on");
        }
        Some(who) if pb.score.tracks[who.idx()].notes.is_empty() => {
            ui.weak("(that track has no notes)");
        }
        Some(_) => {}
    }

    ui.separator();
    ui.label("Strictness:");
    for (value, label) in [
        (Strictness::Strict, "Strict"),
        (Strictness::Normal, "Normal"),
        (Strictness::Lenient, "Lenient"),
    ] {
        changed |= ui.radio_value(&mut pb.eval.strictness, value, label).changed();
    }
    // Custom can't be a plain radio_value: its payload is user-edited, so it
    // would never compare equal to a fixed target. Seed it from whatever tier
    // was active so switching to Custom changes nothing until a slider moves.
    let is_custom = matches!(pb.eval.strictness, Strictness::Custom { .. });
    if ui.radio(is_custom, "Custom").clicked() && !is_custom {
        let (temporal_tolerance_s, force_tolerance, pedal_tolerance) =
            pb.eval.strictness.tolerances();
        pb.eval.strictness =
            Strictness::Custom { temporal_tolerance_s, force_tolerance, pedal_tolerance };
        changed = true;
    }
    if let Strictness::Custom { temporal_tolerance_s, force_tolerance, pedal_tolerance } =
        &mut pb.eval.strictness
    {
        changed |= ui
            .add(
                egui::Slider::new(temporal_tolerance_s, 0.02..=0.5)
                    .text("timing ±s")
                    .logarithmic(true),
            )
            .on_hover_text("How far off a press can be and still match its note")
            .changed();
        changed |= ui
            .add(egui::Slider::new(force_tolerance, 0.05..=1.0).text("force tol."))
            .on_hover_text("How far off the key force can be before scoring zero")
            .changed();
        changed |= ui
            .add(egui::Slider::new(pedal_tolerance, 0.05..=1.0).text("pedal tol."))
            .on_hover_text("How far off the pedal can be before scoring zero")
            .changed();
    }

    ui.separator();
    // Greyed (not hidden) when inapplicable, with the reason on hover — the
    // same annotate-don't-hide convention as the Learn panel. Applicability
    // is re-frozen into the take by the restart below.
    changed |= ui
        .add_enabled(
            live_midi,
            egui::Checkbox::new(&mut pb.eval.evaluate_velocity, "Score key force"),
        )
        .on_hover_text("Also grade how hard each note is struck vs. the score")
        .on_disabled_hover_text("Needs a MIDI keyboard — mic input has no real velocity")
        .changed();
    let pedal_data = pb
        .eval
        .evaluate
        .is_some_and(|w| !pb.score.tracks[w.idx()].pedal_events.is_empty());
    changed |= ui
        .add_enabled(
            live_midi && pedal_data,
            egui::Checkbox::new(&mut pb.eval.evaluate_pedal, "Score sustain pedal"),
        )
        .on_hover_text("Also grade pedal state at each note's press (never during gaps)")
        .on_disabled_hover_text(if live_midi {
            "The evaluated track carries no pedal data to score against"
        } else {
            "Needs a MIDI keyboard — mic input has no pedal signal"
        })
        .changed();
    changed |= ui
        .checkbox(&mut pb.eval.pause_on_miss, "Pause on miss")
        .on_hover_text(
            "Freeze the roll on a note you haven't hit yet, instead of \
             scoring it a miss and moving on",
        )
        .changed();

    ui.add_space(4.0);
    ui.weak("The take restarts whenever a setting changes.");
    if changed {
        pb.start_evaluation(live_midi, synth);
    }
}

/// Inner content of the Evaluation side panel while reviewing a finished
/// take: per-side show/hear toggles, Retake, and the full breakdown.
fn review_settings_body(
    ui: &mut egui::Ui,
    pb: &mut playback::PlaybackEngine,
    live_midi: bool,
    synth: &synth::Synth,
) {
    ui.label("Compare (see and hear):");
    ui.checkbox(&mut pb.review.show_original, "Original track");
    ui.checkbox(&mut pb.review.show_played, "What you played");
    if !pb.review.show_original && !pb.review.show_played {
        ui.weak("Both hidden — nothing to compare");
    }
    ui.add_space(4.0);
    if ui
        .button("Retake")
        .on_hover_text("Discard this review and run the evaluation again")
        .clicked()
    {
        pb.start_evaluation(live_midi, synth);
        return; // the panel re-renders as Evaluation next frame
    }
    ui.separator();
    if let Some(result) = &pb.eval_result {
        ui.collapsing("Full breakdown", |ui| eval_results_body(ui, result));
    }
}

/// The finished take's breakdown — shared by the results window and the
/// review panel's "Full breakdown" expander. Lines for dimensions that
/// weren't evaluated are omitted entirely (no "N/A" noise).
fn eval_results_body(ui: &mut egui::Ui, result: &playback::EvaluationResult) {
    ui.heading(format!("{:.0}%", result.percent));
    let extra = match result.extra_hotspot {
        Some(m) => format!("{} extra (mostly {})", result.extra_press_count, note::solfege_name(m)),
        None => format!("{} extra", result.extra_press_count),
    };
    ui.label(format!(
        "{} matched · {} missed · {}",
        result.matched_count, result.missed_count, extra
    ));
    ui.label(format!("Longest clean streak: {}", result.longest_streak));
    if let Some(p) = &result.pause_stats {
        ui.label(format!(
            "Paused for missed notes: {:.1}s over {} pause{}",
            p.total_s,
            p.count,
            if p.count == 1 { "" } else { "s" }
        ));
    }
    if let Some(bias) = result.timing_bias_s {
        ui.label(format!(
            "Timing bias: {:.0} ms {}",
            (bias * 1000.0).abs(),
            if bias < 0.0 { "early" } else { "late" }
        ));
    }
    if let Some(v) = result.velocity_accuracy {
        ui.label(format!("Key-force accuracy: {:.0}%", v * 100.0));
    }
    if let Some(p) = result.pedal_accuracy {
        ui.label(format!("Pedal accuracy: {:.0}%", p * 100.0));
    }
    let names = |pitches: &[u8]| {
        pitches
            .iter()
            .map(|&m| match result.per_pitch.get(&m) {
                Some(s) => format!(
                    "{} ({:.0}% over {})",
                    note::solfege_name(m),
                    s.avg_score * 100.0,
                    s.attempts
                ),
                None => note::solfege_name(m),
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    if !result.best_pitches.is_empty() {
        ui.label(format!("Best keys: {}", names(&result.best_pitches)));
    }
    if !result.worst_pitches.is_empty() {
        ui.label(format!("Weakest keys: {}", names(&result.worst_pitches)));
    }
}

/// A Preferences row for a [`prefs::Limit`]: a seconds `DragValue` (greyed but
/// value-preserving while infinite) plus an `∞` toggle. Returns whether either
/// widget changed this frame. Free fn (not a method) so it can borrow one
/// `Limit` field while the caller still holds `&mut self`.
fn limit_row(ui: &mut egui::Ui, label: &str, limit: &mut prefs::Limit, tip: &str) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(format!("{label} (s):"));
        // Greyed while infinite, but `secs` is untouched, so flipping ∞ back
        // off restores the exact prior value.
        let dv = egui::DragValue::new(&mut limit.secs)
            .range(0.0..=3600.0)
            .speed(0.5);
        changed |= ui.add_enabled(!limit.infinite, dv).on_hover_text(tip).changed();
        changed |= ui.checkbox(&mut limit.infinite, "∞").on_hover_text(tip).changed();
    });
    changed
}

/// Pure grid-alignment math behind `PianoApp::metro_start_now`: given the
/// current roll time and the beat grid (period, beats per bar), returns
/// `(delta_s, beat_in_bar)` — how many seconds until the next beat that's an
/// exact multiple of `period` since roll-time zero, and that beat's position
/// in the bar. Because beat 0 always sits at roll-time zero, after exactly
/// `beats_per_bar` × (a whole number of bars) beats the target is a whole
/// multiple of `period * beats_per_bar` since zero — in particular, after
/// `bpm` beats at any tempo, exactly one minute has elapsed (`bpm * (60/bpm)
/// == 60`), so beats always land on round absolute positions.
fn metro_grid_align(roll_s: f64, period: f64, beats_per_bar: u8) -> (f64, u8) {
    let bpb = beats_per_bar.max(1) as u64;
    // Epsilon so a roll time that's already (almost) exactly on a grid line
    // fires now rather than waiting a full extra period.
    let beat_idx = ((roll_s / period) - 1e-6).ceil().max(0.0) as u64;
    let target_roll_s = beat_idx as f64 * period;
    let delta_s = (target_roll_s - roll_s).max(0.0);
    (delta_s, (beat_idx % bpb) as u8)
}

/// Resize a metronome per-beat table (pitch or volume) to `len` entries: extra
/// slots repeat the last configured value (a reasonable default for a
/// newly-added beat), shrinking just truncates. Used when `beats_per_bar`
/// changes; `default_fill` is used only when the table started out empty.
fn resize_beat_table(table: &mut Vec<f32>, len: usize, default_fill: f32) {
    let len = len.max(1);
    if table.len() < len {
        let fill = table.last().copied().unwrap_or(default_fill);
        table.resize(len, fill);
    } else {
        table.truncate(len);
    }
}

/// Ease `current` toward `target` (frame-rate independent exponential),
/// returning `None` once within half a roll-pixel — i.e. "arrived". The
/// arrival tolerance scales with the current roll zoom (`px_per_s`).
fn ease_toward(current: f64, target: f64, dt: f64, px_per_s: f32) -> Option<f64> {
    let eased = target + (current - target) * (-dt * SCROLLBACK_EASE_RATE).exp();
    if (target - eased).abs() < 0.5 / px_per_s as f64 {
        None
    } else {
        Some(eased)
    }
}

/// The time ruler shared by both roll panels: a minor gridline every second,
/// a major line + `mm:ss` label every ten. Iterates only the whole seconds
/// in `[lo_s, hi_s]` — O(panel height), never O(session length) — and leaves
/// the time→y mapping to the caller so the two panels' opposite scroll
/// directions both work.
///
/// `label_offset_s` shifts only the printed `mm:ss` text, not `y_of`'s
/// geometry: the falling panel's notes are positioned in *score* time, but its
/// labels are shown in the equivalent *roll* time (score time zero maps to
/// `PianoApp::score_roll_origin_s`), so the two strips' rulers read as one
/// continuous timeline across the keyboard. The history roll passes `0.0`
/// (its native time already *is* roll time).
fn draw_ruler(
    painter: &egui::Painter,
    rect: egui::Rect,
    lo_s: f64,
    hi_s: f64,
    y_of: impl Fn(f64) -> f32,
    label_offset_s: f64,
) {
    let minor = egui::Stroke::new(0.5, egui::Color32::from_gray(40));
    let major = egui::Stroke::new(1.0, egui::Color32::from_gray(60));
    let mut s = lo_s.max(0.0).ceil() as i64;
    let hi = hi_s.floor() as i64;
    while s <= hi {
        let y = y_of(s as f64);
        if s % 10 == 0 {
            painter.hline(rect.x_range(), y, major);
            let label_s = (s as f64 + label_offset_s).round() as i64;
            painter.text(
                egui::pos2(rect.left() + 4.0, y + 2.0),
                egui::Align2::LEFT_TOP,
                format!("{}:{:02}", label_s / 60, label_s % 60),
                egui::FontId::monospace(10.0),
                egui::Color32::from_gray(140),
            );
        } else {
            painter.hline(rect.x_range(), y, minor);
        }
        s += 1;
    }
}

/// The x-extent of the Learn key-range band: the union of the lanes of every
/// key in `lo..=hi`. `None` if no key falls in the range (can't happen for
/// ranges produced by drag/refine, which are clamped to real keys).
fn range_band_x(keys: &[KeyRect], lo: u8, hi: u8) -> Option<egui::Rangef> {
    let mut band: Option<egui::Rangef> = None;
    for k in keys {
        if (lo..=hi).contains(&k.midi) {
            let r = k.rect.x_range();
            band = Some(match band {
                None => r,
                Some(b) => egui::Rangef::new(b.min.min(r.min), b.max.max(r.max)),
            });
        }
    }
    band
}

/// The falling-notes panel: the loaded score's future, sliding down to reach
/// the keyboard exactly when each note starts. "Now" (the playhead) sits at
/// the *bottom* edge — the mirror of `draw_roll`, which hangs history off its
/// top edge. Both tracks normally draw (this shows the score, not who's
/// practicing what) with the same half-lane convention as the history roll;
/// `visible` (per `Who::idx()`) lets EvaluationReview skip a track entirely
/// (not dim it) — every other call site passes `[true, true]`.
#[allow(clippy::too_many_arguments)]
fn draw_falling(
    painter: &egui::Painter,
    rect: egui::Rect,
    keys: &[KeyRect],
    score: &score::Score,
    playhead_s: f64,
    key_range: Option<(u8, u8)>,
    range_drag: Option<(f32, f32)>,
    scroll_offset_s: f64,
    px_per_s: f32,
    label_origin_s: f64,
    visible: [bool; 2],
) {
    let painter = painter.with_clip_rect(rect);
    painter.rect_filled(rect, 0.0, egui::Color32::from_gray(18));

    // Wheel-scroll review shifts only this view time; the real playhead (and
    // with it Learn gating and auto-play) is untouched by construction.
    let view_playhead_s = playhead_s + scroll_offset_s;

    // Future time -> y (f64 subtraction before the f32 cast, as in draw_roll).
    let y_of = |t: f64| rect.bottom() - ((t - view_playhead_s) as f32) * px_per_s;
    let top_s = view_playhead_s + (rect.height() / px_per_s) as f64;

    // Labels only (not y_of's geometry) are shifted into roll-time — see
    // `draw_ruler`'s docs.
    draw_ruler(&painter, rect, view_playhead_s, top_s, y_of, label_origin_s);

    // Make a scrubbed view unmistakable: this is a preview, not a seek.
    if scroll_offset_s != 0.0 {
        painter.text(
            egui::pos2(rect.right() - 6.0, rect.top() + 4.0),
            egui::Align2::RIGHT_TOP,
            "previewing",
            egui::FontId::proportional(11.0),
            egui::Color32::from_gray(140),
        );
    }

    // The Learn key-range band (and the in-progress drag preview): a
    // translucent column over the full panel height.
    let band_fill = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 14);
    if let Some((lo, hi)) = key_range {
        if let Some(band) = range_band_x(keys, lo, hi) {
            painter.rect_filled(
                egui::Rect::from_x_y_ranges(band, rect.y_range()),
                0.0,
                band_fill,
            );
        }
    }
    if let Some((a, b)) = range_drag {
        painter.rect_filled(
            egui::Rect::from_x_y_ranges(egui::Rangef::new(a.min(b), a.max(b)), rect.y_range()),
            0.0,
            band_fill,
        );
    }

    // Segment boundaries ahead, same style as the history roll's separators.
    let sep_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(110));
    for seg in score.segments.iter().skip(1) {
        let y = y_of(seg.start_s);
        if y >= rect.top() && y <= rect.bottom() {
            painter.hline(rect.x_range(), y, sep_stroke);
        }
    }

    // Lane x-extents per MIDI note, from the keyboard layout.
    let mut lanes: [Option<(egui::Rangef, bool)>; 128] = [None; 128];
    for k in keys {
        lanes[k.midi as usize] = Some((k.rect.x_range(), k.black));
    }

    for pass_black in [false, true] {
        for who in [roll::Who::Local, roll::Who::Remote] {
            if !visible[who.idx()] {
                continue;
            }
            let track = &score.tracks[who.idx()];
            let [r, g, b] = track.color;
            for n in &track.notes {
                let Some((xr, black)) = lanes[n.midi as usize] else { continue };
                if black != pass_black {
                    continue;
                }
                // Cull: already fully past, or not yet inside the window.
                if n.end_s <= view_playhead_s || n.start_s > top_s {
                    continue;
                }
                let y_top = y_of(n.end_s);
                let y_bot = y_of(n.start_s);
                let mid = xr.center();
                let (x0, x1) = match who {
                    roll::Who::Local => (xr.min + 1.0, mid),
                    roll::Who::Remote => (mid, xr.max - 1.0),
                };
                let color = if black {
                    let dark = |c: u8| (c as f32 * 0.72) as u8;
                    egui::Color32::from_rgb(dark(r), dark(g), dark(b))
                } else {
                    egui::Color32::from_rgb(r, g, b)
                };
                let mark = egui::Rect::from_min_max(
                    egui::pos2(x0, y_top),
                    egui::pos2(x1, y_bot.max(y_top + 2.0)),
                );
                painter.rect_filled(mark, 2.0, color);
            }
        }
    }
}

/// Paint one key as up to N vertical color stripes — the playback-mode
/// analogue of `paint_key`'s two-way diagonal split, generalized to the three
/// possible sources there (your live press + each auto-played track).
fn paint_key_striped(
    painter: &egui::Painter,
    rect: egui::Rect,
    colors: &[egui::Color32],
    base: egui::Color32,
    rounding: f32,
    stroke: egui::Stroke,
) {
    if colors.is_empty() {
        painter.rect(rect, rounding, base, stroke);
        return;
    }
    painter.rect(rect, rounding, colors[0], egui::Stroke::NONE);
    if colors.len() > 1 {
        let w = rect.width() / colors.len() as f32;
        for (i, &c) in colors.iter().enumerate().skip(1) {
            let x0 = rect.left() + i as f32 * w;
            painter.rect_filled(
                egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x0 + w, rect.bottom())),
                0.0,
                c,
            );
        }
    }
    painter.rect_stroke(rect, rounding, stroke);
}

/// Playback-mode keyboard: each layer is (per-key on-states, color). Most
/// keys have 0–1 active layers; simultaneous 2–3 (e.g. practicing a note an
/// auto-played track also hits) render as vertical stripes — deliberately
/// distinct from live mode's diagonal split. `draw_keyboard` stays untouched
/// for live mode.
fn draw_keyboard_layered(
    painter: &egui::Painter,
    rect: egui::Rect,
    keys: &[KeyRect],
    layers: &[([bool; KEY_COUNT], egui::Color32)],
) {
    painter.rect_filled(rect, 0.0, egui::Color32::from_gray(30));

    let white_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(60));
    let black_stroke = egui::Stroke::new(1.0, egui::Color32::BLACK);

    for key in keys {
        let idx = midi_to_key_index(key.midi).unwrap();
        let (base, stroke) = if key.black {
            (egui::Color32::from_gray(20), black_stroke)
        } else {
            (egui::Color32::from_gray(245), white_stroke)
        };
        let active: Vec<egui::Color32> =
            layers.iter().filter(|(on, _)| on[idx]).map(|(_, c)| *c).collect();
        paint_key_striped(painter, key.rect, &active, base, 2.0, stroke);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metro_grid_aligns_to_next_beat_and_repeats_every_whole_minute() {
        // 120 BPM => period 0.5s (exact in f64). Starting mid-beat waits for
        // the next grid line instead of firing immediately.
        let (delta, beat) = metro_grid_align(17.3, 0.5, 4);
        assert!((delta - 0.2).abs() < 1e-9);
        assert_eq!(beat, 3); // beat_idx 35 (17.5 / 0.5); 35 % 4 == 3

        // Landing exactly on a grid line fires immediately.
        let (delta0, beat0) = metro_grid_align(18.0, 0.5, 4);
        assert!(delta0 < 1e-6);
        assert_eq!(beat0, 0);

        // Beat 0 of the grid always sits at roll-time zero, so after exactly
        // `bpm` beats the target lands on a whole minute — true for ANY bpm,
        // since bpm * (60 / bpm) == 60. Verified here at 120 BPM starting just
        // before the 60s mark.
        let (delta, beat) = metro_grid_align(59.9, 0.5, 4);
        assert_eq!(59.9 + delta, 60.0);
        assert_eq!(beat, 0);
    }

    #[test]
    fn resize_beat_table_extends_and_truncates() {
        let mut freqs = vec![1800.0, 1200.0];
        resize_beat_table(&mut freqs, 4, 1200.0);
        assert_eq!(freqs, vec![1800.0, 1200.0, 1200.0, 1200.0]); // repeats the last
        resize_beat_table(&mut freqs, 1, 1200.0);
        assert_eq!(freqs, vec![1800.0]);
        resize_beat_table(&mut freqs, 0, 1200.0);
        assert_eq!(freqs.len(), 1); // floored to at least 1
    }
}
