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
mod inference;
mod input;
mod midi;
mod net;
mod note;
mod record;
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

    // Point ONNX Runtime at the local `onnxruntime.dll` via env var only — do
    // NOT load it here. Loading ONNX Runtime spins up its own threads during
    // initialisation; doing that from the main thread before the event loop
    // deadlocks against the Windows loader lock and freezes the app before any
    // window appears. ort loads the DLL lazily the first time a Session is
    // built — which happens on the dedicated inference thread (see inference.rs)
    // where a slow or failing load can't block the GUI.
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        if let Ok(abs) = std::fs::canonicalize("onnxruntime.dll") {
            std::env::set_var("ORT_DYLIB_PATH", abs);
        }
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 380.0])
            .with_min_inner_size([640.0, 300.0])
            .with_title("open-piano — P2P acoustic piano visualizer"),
        ..Default::default()
    };

    eframe::run_native(
        "open-piano",
        options,
        Box::new(|_cc| Ok(Box::new(PianoApp::new()))),
    )
}

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
            apply(&mut self.local, msg);
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
        self.synth.set_gain(synth::Channel::Local, screen);
        self.synth.set_gain(synth::Channel::Peer, peer);
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
            // ---- Auto-update banner (only drawn when there's something to act on) ----
            let mut drawn = false;
            ui.horizontal(|ui| drawn = self.update_controls(ui));
            if drawn {
                ui.separator();
            }
            // ---- Play together: host a session or join one with an invite
            // code. No IPs or ports — iroh handles NAT traversal (net.rs). ----
            ui.horizontal(|ui| {
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
                    // The full code is long (~250 chars); show just enough to
                    // see it exists. The Copy button is the real interface.
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
            });
            ui.add_space(2.0);
        });

        // ---- Center: the 88-key keyboard (also playable with the mouse) ----
        egui::CentralPanel::default().show(ctx, |ui| {
            let response =
                ui.allocate_response(ui.available_size(), egui::Sense::click_and_drag());
            let rect = response.rect;

            // Mouse play is disabled while capturing training data: a click makes
            // no real sound and isn't written to the MIDI labels, so it would put
            // notes on screen that don't exist in the recorded audio.
            if self.input.recorder.is_armed() {
                self.set_mouse_note(None); // drop any note the mouse was holding
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

            let colors = KeyColors {
                local: egui::Color32::from_rgb(self.local_color[0], self.local_color[1], self.local_color[2]),
                remote: egui::Color32::from_rgb(self.remote_color[0], self.remote_color[1], self.remote_color[2]),
            };
            draw_keyboard(ui.painter(), rect, &self.local, &self.remote, colors);
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
