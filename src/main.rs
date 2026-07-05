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
mod record;
mod roll;
mod score;
mod synth;
mod update;

use eframe::egui;

use std::time::{Duration, Instant};

use input::{InputEngine, Source};
use net::{NetEvent, Peer};
use note::{is_black_key, midi_to_key_index, NoteMsg, Packet, KEY_COUNT, MIDI_HIGH, MIDI_LOW};

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

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 620.0])
            .with_min_inner_size([640.0, 420.0])
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
        Box::new(|_cc| Ok(Box::new(PianoApp::new()))),
    )
}

/// The version compiled into this build (from Cargo.toml) — what the About
/// dialog and window title report, and what the auto-updater compares against
/// GitHub Releases.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default model posterior probability above which a note counts as "on".
/// 0.3 matches Basic Pitch's own default frame threshold; lower = more
/// sensitive (use the slider to tune for your room/mic).
const DEFAULT_THRESHOLD: f32 = 0.3;

/// Default note colors (sRGB). Each player can change *their own* (the local
/// one); it is sent over the wire so the peer renders this player in this color.
const DEFAULT_LOCAL_COLOR: [u8; 3] = [220, 60, 60]; // warm red
const DEFAULT_REMOTE_COLOR: [u8; 3] = [60, 110, 230]; // blue (until the peer announces theirs)

/// How often to re-broadcast our color to the peer. A low-rate heartbeat means
/// color syncs regardless of who connects first, and recovers from a dropped
/// announcement, at a negligible 1 datagram/sec.
const COLOR_HEARTBEAT: Duration = Duration::from_secs(1);

/// How long, *after* the built-in synth stops voicing a note, to keep ignoring
/// mic-detected onsets of that same note (see the echo guard in `pump_input`).
/// The synth's own tone leaks through the speakers into the microphone, where
/// inference re-detects it as a played note — an echo loop that, combined with
/// the detector's release hysteresis, leaves a key lit long after you let go.
/// While a note is actively voiced it's suppressed unconditionally; this window
/// only has to cover the post-release tail (the synth's ~1.3 s release fade plus
/// margin), so the key doesn't flicker back on as the tone rings out.
const ECHO_HOLDOFF: Duration = Duration::from_millis(2000);

/// Pixels of paper per second of roll time in the history strip under the
/// keyboard. 40 px/s shows ~6 s in the default window, and even a staccato
/// click leaves a visible mark (plus `draw_roll` enforces a 2 px minimum).
const ROLL_PX_PER_S: f32 = 40.0;

/// How the central panel splits between the keyboard (top) and the roll
/// (bottom): the keyboard takes this fraction of the height, but never less
/// than `MIN_KEYBOARD_H` so keys stay playable in a short window.
const KEYBOARD_FRACTION: f32 = 0.45;
const MIN_KEYBOARD_H: f32 = 140.0;

/// With a score loaded, the space not taken by the keyboard splits between
/// the falling-notes panel (above the keys) and the history roll (below).
/// Biased toward the falling notes — the forward-looking practice aid — over
/// history review.
const FALLING_FRACTION: f32 = 0.55;

/// Ctrl+S (Cmd+S on mac) quick-saves the roll — same action as File ▸ Save.
const SAVE_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S);

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
    // mic-detected onset of that note is still ignored (`ECHO_HOLDOFF`).
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
    local_color: [u8; 3],  // our notes; editable, broadcast to the peer
    remote_color: [u8; 3], // the peer's notes; received from the peer
    last_color_send: Instant,

    // --- networking (see net.rs: host/join with a one-string invite code) ---
    // Our invite code, once the net thread reports it (hosting only). Shown
    // with a Copy button so it can be pasted to the other player.
    my_ticket: Option<String>,
    // The paste box for an invite code received from a host.
    join_ticket: String,
    peer: Option<Peer>,
    net_status: String,

    // --- in-app auto-update (checks GitHub Releases on launch) ---
    updater: update::Updater,

    // Whether the About window is open (toggled from the status bar).
    show_about: bool,

    // --- piano-roll history (see roll.rs + draw_roll) ---
    roll: roll::Roll,
    // Drag-to-review view state: `Some(t)` is the roll time rendered at the
    // strip's top edge while scrolled back (or animating home); `None` = live.
    scrollback: Option<f64>,
    // Result of the last save attempt, shown next to the File menu.
    roll_status: String,
    // Whether the "unsaved roll" confirmation is up (close was intercepted).
    show_close_confirm: bool,
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
    // Whether the segment row + Learn side panel are shown (👁 toggle).
    panels_visible: bool,
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
    fn new() -> Self {
        let input = input::start(DEFAULT_THRESHOLD);
        Self {
            input,
            synth: synth::Synth::start(),
            screen_volume: 1.0,
            screen_muted: false,
            peer_volume: 1.0,
            peer_muted: false,
            threshold: DEFAULT_THRESHOLD,
            mic_muted: false,
            notes_epoch: 0,
            was_midi: false,
            local: [false; KEY_COUNT],
            remote: [false; KEY_COUNT],
            echo_held: [[false; 2]; 128],
            echo_until: [None; 128],
            mouse_note: None,
            pointer_still_since: Instant::now(),
            last_pointer_pos: None,
            local_color: DEFAULT_LOCAL_COLOR,
            remote_color: DEFAULT_REMOTE_COLOR,
            last_color_send: Instant::now(),
            my_ticket: None,
            join_ticket: String::new(),
            peer: None,
            net_status: "Not connected".to_string(),
            // Kick off the background GitHub Releases check; the UI polls its
            // state each frame (see `update_controls`).
            updater: update::start(),
            show_about: false,
            roll: roll::Roll::new(),
            scrollback: None,
            roll_status: String::new(),
            show_close_confirm: false,
            allow_close: false,
            playback: None,
            playback_volume: 1.0,
            playback_muted: false,
            open_status: String::new(),
            panels_visible: true,
            pending_break_t: None,
            range_drag: None,
            show_refine_range: false,
            refine_lo: String::new(),
            refine_hi: String::new(),
            instance_edit: String::new(),
            segment_edit: String::new(),
            segment_edit_idx: 0,
        }
    }

    /// Send our chosen color to the peer (if connected) and reset the heartbeat.
    fn send_color(&mut self) {
        if let Some(peer) = &self.peer {
            peer.send(Packet::Color(self.local_color));
        }
        self.last_color_send = Instant::now();
    }

    /// Start hosting a session. Replaces any existing session (dropping the
    /// old `Peer` shuts its net thread down); the invite code arrives async
    /// as a `NetEvent::Ticket`.
    fn host(&mut self) {
        self.my_ticket = None;
        self.clear_remote_keys();
        self.net_status = "Starting…".into();
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
        self.net_status = "Joining…".into();
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
                    if let Some(peer) = &self.peer {
                        peer.send(Packet::Note(NoteMsg::Off(MIDI_LOW + idx as u8)));
                    }
                }
            }
            // The matching note-offs will never arrive, so close the roll's
            // open marks too.
            self.roll.release_all(roll::Who::Local);
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
            if mic_source && self.mic_muted && matches!(msg, NoteMsg::On(_)) {
                continue;
            }
            apply(&mut self.local, msg);
            self.roll.note(roll::Who::Local, msg, self.local_color);
            if let Some(peer) = &self.peer {
                peer.send(Packet::Note(msg));
            }
        }
    }

    /// Whether a mic-detected transition for `midi` should be ignored as the
    /// synth's own sound echoing back: true while the synth is voicing that note,
    /// or within `ECHO_HOLDOFF` of it having stopped (covers the release tail).
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
            self.echo_until[n] = Some(Instant::now() + ECHO_HOLDOFF);
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
        } else if armed {
            ui.label("Starting…");
        }
    }

    /// The File menu (save the piano roll) plus the "unsaved" chip and the
    /// last save result. Lives in a `menu::bar` row at the top of the config
    /// panel. Save As… lets the user pick between the interoperable MIDI
    /// export and the self-contained JSONL one (see roll.rs).
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
            }
        });
        if self.roll.has_unsaved() {
            ui.colored_label(egui::Color32::from_rgb(210, 170, 60), "● unsaved")
                .on_hover_text("The roll has notes that haven't been saved (File ▸ Save)");
        }
        // Rename whichever instance the live roll is currently in — the name
        // is baked into the file's markers on the next save.
        if !self.roll.is_empty() {
            ui.separator();
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
                self.playback = Some(playback::PlaybackEngine::new(s, path));
                self.scrollback = None;
                self.range_drag = None;
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
                ui.horizontal(|ui| {
                    if ui.button("Save and quit").clicked() && self.save_roll_quick() {
                        // Synchronous save: the file is on disk before the
                        // process is allowed to die. On failure we fall
                        // through with the error shown above.
                        self.allow_close = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    if ui.button("Quit without saving").clicked() {
                        self.allow_close = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    if ui.button("Cancel").clicked() {
                        self.show_close_confirm = false;
                    }
                });
            });
    }

    /// Transport + mode + speed for the loaded score. Always visible while a
    /// file is open (core playback UI, unlike the hideable settings panels).
    /// Glyph scheme: barred ⏮/⏭ act on the whole piece, double-triangle
    /// ⏪/⏩ on one segment, and play/pause flips ▶/⏸ with state.
    fn playback_controls(&mut self, ui: &mut egui::Ui) {
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
        ui.radio_value(&mut pb.mode, playback::Mode::Listen, "Listen");
        ui.radio_value(&mut pb.mode, playback::Mode::Learn, "Learn");
        ui.separator();
        ui.add(egui::Slider::new(&mut pb.speed, 0.25..=2.0).text("speed"));
        ui.separator();
        let eye = if self.panels_visible { "Hide panels" } else { "Show panels" };
        ui.checkbox(&mut self.panels_visible, eye)
            .on_hover_text("Show/hide the segment row and Learn settings panel");
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
        ui.checkbox(&mut pb.loop_state.enabled, "Loop this segment");
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

    /// Learn-mode settings side panel: which tracks to practice, how strict
    /// the gating is, and the key-range readout. Must be shown *before* the
    /// CentralPanel each frame (egui reserves panel space in show order).
    fn learn_panel(&mut self, ctx: &egui::Context) {
        let Some(pb) = &mut self.playback else { return };
        if pb.mode != playback::Mode::Learn || !self.panels_visible {
            return;
        }
        egui::SidePanel::right("learn_panel")
            .resizable(false)
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.strong("Learn settings");
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
                ui.separator();
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
                        ui.weak("Key range: whole keyboard");
                        ui.weak("(drag across the falling notes to set one)");
                    }
                }
            });
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
        let Some(pb) = &mut self.playback else { return };
        let rect = resp.rect;
        // Capture the value, not `pb`, so the closure doesn't pin the borrow.
        let playhead = pb.playhead_s;
        let t_of_y = move |y: f32| playhead + ((rect.bottom() - y) / ROLL_PX_PER_S) as f64;

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
                    update::restart();
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
        match msg {
            NoteMsg::On(n) => {
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
        if let Some(peer) = &self.peer {
            peer.send(Packet::Note(msg));
        }
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
            self.local_note(NoteMsg::On(new));
        }
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
                    self.clear_remote_keys();
                    self.send_color();
                }
                NetEvent::Disconnected => self.clear_remote_keys(),
                NetEvent::Packet(Packet::Note(msg)) => {
                    apply(&mut self.remote, msg);
                    self.roll.note(roll::Who::Remote, msg, self.remote_color);
                    // The peer's notes have no local sound source, so voice them.
                    self.play_synth(msg, synth::Channel::Peer);
                }
                NetEvent::Packet(Packet::Color(rgb)) => self.remote_color = rgb,
            }
        }
    }
}

/// Apply a note transition to a key-state array.
fn apply(keys: &mut [bool; KEY_COUNT], msg: NoteMsg) {
    if let Some(idx) = midi_to_key_index(msg.midi()) {
        keys[idx] = matches!(msg, NoteMsg::On(_));
    }
}

impl eframe::App for PianoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.pump_input();
        self.pump_network();
        self.sync_synth_to_source();
        self.roll.tick(Instant::now());

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
            pb.tick(dt, &held, &self.synth);
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

        // Low-rate color heartbeat so colors sync regardless of connect order.
        if self.peer.is_some() && self.last_color_send.elapsed() >= COLOR_HEARTBEAT {
            self.send_color();
        }

        // ---- Top: networking + audio config ----
        egui::TopBottomPanel::top("config").show(ctx, |ui| {
            ui.add_space(4.0);
            // ---- File menu (save the piano roll) + unsaved/status chips ----
            egui::menu::bar(ui, |ui| self.file_menu(ui));
            ui.separator();
            // ---- Auto-update banner (only drawn when there's something to act on) ----
            let mut drawn = false;
            ui.horizontal(|ui| drawn = self.update_controls(ui));
            if drawn {
                ui.separator();
            }
            // ---- Play together: host a session or join one with an invite
            // code. No IPs or ports — iroh handles NAT traversal (net.rs).
            // Hidden while a file is open: playback and live P2P are
            // mutually exclusive (see `open_score`). ----
            ui.horizontal(|ui| {
                if self.playback.is_some() {
                    ui.weak("Networking is disabled while a file is open (File ▸ Close file)");
                    return;
                }
                if ui
                    .button("Host session")
                    .on_hover_text("Create an invite code to send to the other player")
                    .clicked()
                {
                    self.host();
                }
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
                ui.add(
                    egui::TextEdit::singleline(&mut self.join_ticket)
                        .desired_width(180.0)
                        .hint_text("paste code from the host"),
                );
                if ui.button("Join").clicked() {
                    self.join();
                }
            });
            // ---- Playback transport + segment row (only with a file open) ----
            if self.playback.is_some() {
                ui.add_space(2.0);
                ui.horizontal(|ui| self.playback_controls(ui));
                if self.panels_visible {
                    ui.add_space(2.0);
                    ui.horizontal(|ui| self.segment_controls(ui));
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
                    }
                    ui.separator();
                    ui.checkbox(&mut self.mic_muted, "Mute mic").on_hover_text(
                        "Ignore mic-detected notes (stops ambient noise from \
                         painting the roll or counting as played keys)",
                    );
                });
            }
            // ---- My color (broadcast to the peer) ----
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label("My color:");
                if ui.color_edit_button_srgb(&mut self.local_color).changed() {
                    // Push the change immediately; the heartbeat covers the rest.
                    self.send_color();
                }
                ui.separator();
                ui.label("Peer's color:");
                let (r, g, b) = (self.remote_color[0], self.remote_color[1], self.remote_color[2]);
                ui.colored_label(egui::Color32::from_rgb(r, g, b), "■");
                ui.weak("(chosen by the peer)");
            });
            // ---- Synth volume / mute (screen + peer sources) ----
            ui.add_space(2.0);
            ui.horizontal(|ui| self.synth_controls(ui));
            // ---- Training-data capture (record mic audio + MIDI labels) ----
            ui.add_space(2.0);
            ui.horizontal(|ui| self.record_controls(ui));
            ui.add_space(4.0);
        });

        // ---- Bottom status bar ----
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                let lc = self.local_color;
                let rc = self.remote_color;
                ui.colored_label(egui::Color32::from_rgb(lc[0], lc[1], lc[2]), "■");
                ui.label("you");
                ui.colored_label(egui::Color32::from_rgb(rc[0], rc[1], rc[2]), "■");
                ui.label("peer");
                ui.separator();
                let (device, model) = {
                    let s = self.input.status.lock().unwrap();
                    (s.device.clone(), s.model.clone())
                };
                ui.label(device);
                ui.separator();
                ui.label(model);
                ui.separator();
                ui.label(&self.net_status);
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

        self.about_window(ctx);
        self.unsaved_dialog(ctx);
        self.refine_range_window(ctx);
        // Side panels reserve space in show order — must precede CentralPanel.
        self.learn_panel(ctx);

        // ---- Center: the 88-key keyboard (also playable with the mouse),
        // the piano-roll history strip on the paper below it, and — with a
        // score loaded — the falling-notes panel above it ----
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let kb_h = (avail.y * KEYBOARD_FRACTION).max(MIN_KEYBOARD_H).min(avail.y);

            // Falling-notes panel first (it sits on top, ending at the keys).
            let falling_resp = self.playback.as_ref().map(|_| {
                let h = ((avail.y - kb_h).max(0.0) * FALLING_FRACTION).floor();
                ui.allocate_response(egui::vec2(avail.x, h), egui::Sense::click_and_drag())
            });

            let response =
                ui.allocate_response(egui::vec2(avail.x, kb_h), egui::Sense::click_and_drag());
            let rect = response.rect;

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
                // While the button is held, the key under the pointer is played;
                // dragging across keys glides note-to-note.
                let target = if response.is_pointer_button_down_on() {
                    response.interact_pointer_pos().and_then(|p| key_at(rect, p))
                } else {
                    None
                };
                self.set_mouse_note(target);
            }

            // The keyboard's key layout doubles as lane geometry for both
            // roll panels, so marks are exactly x-aligned with their keys.
            let keys = layout_keys(rect);

            if let Some(pb) = &self.playback {
                // Layered keyboard: your live presses, plus each unpracticed
                // track's currently-playing notes in its own color.
                let mut layers: Vec<([bool; KEY_COUNT], egui::Color32)> = vec![(
                    self.local,
                    egui::Color32::from_rgb(
                        self.local_color[0],
                        self.local_color[1],
                        self.local_color[2],
                    ),
                )];
                for who in [roll::Who::Local, roll::Who::Remote] {
                    if !pb.practiced(who) {
                        let [r, g, b] = pb.score.tracks[who.idx()].color;
                        layers.push((pb.active_key_array(who), egui::Color32::from_rgb(r, g, b)));
                    }
                }
                draw_keyboard_layered(ui.painter(), rect, &keys, &layers);
            } else {
                let colors = KeyColors {
                    local: egui::Color32::from_rgb(self.local_color[0], self.local_color[1], self.local_color[2]),
                    remote: egui::Color32::from_rgb(self.remote_color[0], self.remote_color[1], self.remote_color[2]),
                };
                draw_keyboard(ui.painter(), rect, &self.local, &self.remote, colors);
            }

            // Falling-panel interactions + drawing.
            if let Some(falling_resp) = falling_resp {
                self.falling_panel_interactions(ui, &falling_resp, &keys);
                if let Some(pb) = &self.playback {
                    draw_falling(
                        ui.painter(),
                        falling_resp.rect,
                        &keys,
                        &pb.score,
                        pb.playhead_s,
                        pb.learn.key_range,
                        self.range_drag,
                    );
                }
            }

            // ---- The roll strip: everything below the keyboard. Dragging it
            // reviews history; releasing eases back to the live edge. ----
            let roll_resp =
                ui.allocate_response(ui.available_size(), egui::Sense::click_and_drag());
            if roll_resp.dragged() {
                // Dragging up (negative delta.y) pulls older paper into view:
                // a mark sits at y ∝ (view_top − t), so moving marks up means
                // lowering the view-top time.
                let top = self.scrollback.unwrap_or(self.roll.now_s());
                let new_top = (top + (roll_resp.drag_delta().y / ROLL_PX_PER_S) as f64)
                    .clamp(0.0, self.roll.now_s());
                self.scrollback = Some(new_top);
            } else if let Some(top) = self.scrollback {
                // Ease home exponentially (~0.2 s feel, frame-rate independent)
                // and snap to live once within half a pixel.
                let dt = ctx.input(|i| i.stable_dt) as f64;
                let now = self.roll.now_s();
                let eased = now + (top - now) * (-dt * 12.0).exp();
                self.scrollback = if (now - eased).abs() < 0.5 / ROLL_PX_PER_S as f64 {
                    None
                } else {
                    Some(eased)
                };
            }
            let view_top_s = self.scrollback.unwrap_or(self.roll.now_s());

            // Manual instance breaks on the history roll: Ctrl+click inserts
            // one at the clicked time; right-click offers the same via menu.
            let roll_t_of_y =
                |y: f32| view_top_s - ((y - roll_resp.rect.top()) / ROLL_PX_PER_S) as f64;
            if roll_resp.clicked() && ui.input(|i| i.modifiers.command) {
                if let Some(pos) = roll_resp.interact_pointer_pos() {
                    self.roll.insert_separator(roll_t_of_y(pos.y));
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
                        self.roll.insert_separator(t);
                    }
                    ui.close_menu();
                }
            });

            draw_roll(ui.painter(), roll_resp.rect, &keys, &self.roll, view_top_s);
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
) {
    let painter = painter.with_clip_rect(rect);
    painter.rect_filled(rect, 0.0, egui::Color32::from_gray(18));

    // Roll time -> y. Subtract in f64 *before* the f32 cast: after a long
    // session the times are large and f32 subtraction would jitter the marks.
    let y_of = |t: f64| rect.top() + ((view_top_s - t) as f32) * ROLL_PX_PER_S;

    // ---- Ruler: only the whole seconds inside the visible window (top edge
    // = view_top_s, bottom edge = view_top_s - height/speed), never the whole
    // session, so cost is O(strip height) regardless of duration.
    let bottom_s = view_top_s - (rect.height() / ROLL_PX_PER_S) as f64;
    draw_ruler(&painter, rect, bottom_s, view_top_s, y_of);

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

    // Fixed half-lane split: local always left, remote always right, so
    // simultaneous same-key presses sit side by side with no overlap logic —
    // the roll's analogue of `paint_key`'s diagonal split.
    for pass_black in [false, true] {
        for seg in &roll.segments {
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
            let [r, g, b] = seg.color;
            let color = if black {
                // Black-key marks: same hue, dimmed — thinner comes free from
                // the black key's narrower lane.
                let dark = |c: u8| (c as f32 * 0.72) as u8;
                egui::Color32::from_rgb(dark(r), dark(g), dark(b))
            } else {
                egui::Color32::from_rgb(r, g, b)
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

/// The time ruler shared by both roll panels: a minor gridline every second,
/// a major line + `mm:ss` label every ten. Iterates only the whole seconds
/// in `[lo_s, hi_s]` — O(panel height), never O(session length) — and leaves
/// the time→y mapping to the caller so the two panels' opposite scroll
/// directions both work.
fn draw_ruler(
    painter: &egui::Painter,
    rect: egui::Rect,
    lo_s: f64,
    hi_s: f64,
    y_of: impl Fn(f64) -> f32,
) {
    let minor = egui::Stroke::new(0.5, egui::Color32::from_gray(40));
    let major = egui::Stroke::new(1.0, egui::Color32::from_gray(60));
    let mut s = lo_s.max(0.0).ceil() as i64;
    let hi = hi_s.floor() as i64;
    while s <= hi {
        let y = y_of(s as f64);
        if s % 10 == 0 {
            painter.hline(rect.x_range(), y, major);
            painter.text(
                egui::pos2(rect.left() + 4.0, y + 2.0),
                egui::Align2::LEFT_TOP,
                format!("{}:{:02}", s / 60, s % 60),
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
/// top edge. Both tracks always draw (this shows the score, not who's
/// practicing what) with the same half-lane convention as the history roll.
fn draw_falling(
    painter: &egui::Painter,
    rect: egui::Rect,
    keys: &[KeyRect],
    score: &score::Score,
    playhead_s: f64,
    key_range: Option<(u8, u8)>,
    range_drag: Option<(f32, f32)>,
) {
    let painter = painter.with_clip_rect(rect);
    painter.rect_filled(rect, 0.0, egui::Color32::from_gray(18));

    // Future time -> y (f64 subtraction before the f32 cast, as in draw_roll).
    let y_of = |t: f64| rect.bottom() - ((t - playhead_s) as f32) * ROLL_PX_PER_S;
    let top_s = playhead_s + (rect.height() / ROLL_PX_PER_S) as f64;

    draw_ruler(&painter, rect, playhead_s, top_s, y_of);

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
            let track = &score.tracks[who.idx()];
            let [r, g, b] = track.color;
            for n in &track.notes {
                let Some((xr, black)) = lanes[n.midi as usize] else { continue };
                if black != pass_black {
                    continue;
                }
                // Cull: already fully past, or not yet inside the window.
                if n.end_s <= playhead_s || n.start_s > top_s {
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
