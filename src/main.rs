//! open-piano — a real-time, networked, peer-to-peer acoustic piano visualizer.
//!
//! * Note input comes from one of two backends, chosen automatically at startup
//!   (see `input.rs`): a connected **MIDI** device (preferred — exact events),
//!   or, as a fallback, **microphone** audio captured via cpal and transcribed
//!   by an ONNX model (Spotify Basic Pitch) on a dedicated inference thread.
//!   Either way the resulting note transitions arrive on one mpsc channel.
//! * Notes played locally light up RED; notes arriving over UDP from the remote
//!   peer light up BLUE (both -> purple).
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

use std::net::SocketAddr;

use eframe::egui;

use std::time::{Duration, Instant};

use input::{InputEngine, Source};
use net::Peer;
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

struct PianoApp {
    // --- note input (MIDI or microphone fallback) ---
    input: InputEngine,
    threshold: f32,
    // Last input-switch epoch we acted on; a change means the backend swapped
    // (e.g. MIDI unplugged) and we must force-release held notes.
    notes_epoch: u64,

    // --- key state ---
    local: [bool; KEY_COUNT],
    remote: [bool; KEY_COUNT],

    // --- colors (sRGB) ---
    local_color: [u8; 3],  // our notes; editable, broadcast to the peer
    remote_color: [u8; 3], // the peer's notes; received from the peer
    last_color_send: Instant,

    // --- networking config (UI fields) ---
    local_port: String,
    remote_ip: String,
    remote_port: String,
    peer: Option<Peer>,
    net_status: String,
}

impl PianoApp {
    fn new() -> Self {
        let input = input::start(DEFAULT_THRESHOLD);
        Self {
            input,
            threshold: DEFAULT_THRESHOLD,
            notes_epoch: 0,
            local: [false; KEY_COUNT],
            remote: [false; KEY_COUNT],
            local_color: DEFAULT_LOCAL_COLOR,
            remote_color: DEFAULT_REMOTE_COLOR,
            last_color_send: Instant::now(),
            local_port: "9000".to_string(),
            remote_ip: "127.0.0.1".to_string(),
            remote_port: "9001".to_string(),
            peer: None,
            net_status: "Not connected".to_string(),
        }
    }

    /// Send our chosen color to the peer (if connected) and reset the heartbeat.
    fn send_color(&mut self) {
        if let Some(peer) = &self.peer {
            peer.send(Packet::Color(self.local_color));
        }
        self.last_color_send = Instant::now();
    }

    /// Try to (re)bind the local port and target the remote peer.
    fn connect(&mut self) {
        let local_port: u16 = match self.local_port.trim().parse() {
            Ok(p) => p,
            Err(_) => {
                self.net_status = "Invalid local port".into();
                return;
            }
        };
        let remote_port: u16 = match self.remote_port.trim().parse() {
            Ok(p) => p,
            Err(_) => {
                self.net_status = "Invalid remote port".into();
                return;
            }
        };
        let ip = self.remote_ip.trim();
        let remote: SocketAddr = match format!("{ip}:{remote_port}").parse() {
            Ok(a) => a,
            Err(_) => {
                self.net_status = format!("Invalid remote address: {ip}:{remote_port}");
                return;
            }
        };

        match Peer::connect(local_port, remote) {
            Ok(peer) => {
                self.net_status = format!("Listening on :{local_port} → {remote}");
                self.peer = Some(peer);
                // A fresh bind means remote state is unknown; clear remote keys.
                self.remote = [false; KEY_COUNT];
                // Announce our color right away so the peer can render us.
                self.send_color();
            }
            Err(e) => {
                self.net_status = format!("Bind failed: {e}");
                self.peer = None;
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
            for idx in 0..KEY_COUNT {
                if self.local[idx] {
                    self.local[idx] = false;
                    if let Some(peer) = &self.peer {
                        peer.send(Packet::Note(NoteMsg::Off(MIDI_LOW + idx as u8)));
                    }
                }
            }
        }

        while let Ok(msg) = self.input.notes.try_recv() {
            apply(&mut self.local, msg);
            if let Some(peer) = &self.peer {
                peer.send(Packet::Note(msg));
            }
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

    /// Drain the network channel: update remote keys and the peer's color.
    fn pump_network(&mut self) {
        let mut packets = Vec::new();
        if let Some(peer) = &self.peer {
            while let Ok(packet) = peer.incoming.try_recv() {
                packets.push(packet);
            }
        }
        for packet in packets {
            match packet {
                Packet::Note(msg) => apply(&mut self.remote, msg),
                Packet::Color(rgb) => self.remote_color = rgb,
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

        // Low-rate color heartbeat so colors sync regardless of connect order.
        if self.peer.is_some() && self.last_color_send.elapsed() >= COLOR_HEARTBEAT {
            self.send_color();
        }

        // ---- Top: networking + audio config ----
        egui::TopBottomPanel::top("config").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Local Port:");
                ui.add(egui::TextEdit::singleline(&mut self.local_port).desired_width(70.0));
                ui.separator();
                ui.label("Remote Target IP:");
                ui.add(egui::TextEdit::singleline(&mut self.remote_ip).desired_width(120.0));
                ui.label("Remote Port:");
                ui.add(egui::TextEdit::singleline(&mut self.remote_port).desired_width(70.0));
                if ui.button("Connect").clicked() {
                    self.connect();
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

        // ---- Center: the 88-key keyboard ----
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let (_id, rect) = ui.allocate_space(avail);
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

    // Collect the white-key MIDI notes in order (52 of them across 88 keys).
    let white_midis: Vec<u8> = (MIDI_LOW..=MIDI_HIGH).filter(|m| !is_black_key(*m)).collect();
    let white_count = white_midis.len().max(1);
    let white_w = rect.width() / white_count as f32;
    let key_top = rect.top();
    let key_bottom = rect.bottom();

    let white_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(60));

    // Map a white-key MIDI note -> its column index, so black keys can be
    // positioned relative to the white key on their left.
    let white_index = |midi: u8| -> Option<usize> {
        white_midis.iter().position(|&m| m == midi)
    };

    // 1) White keys.
    for (col, &midi) in white_midis.iter().enumerate() {
        let x0 = rect.left() + col as f32 * white_w;
        let key_rect = egui::Rect::from_min_max(
            egui::pos2(x0, key_top),
            egui::pos2(x0 + white_w, key_bottom),
        );
        let idx = midi_to_key_index(midi).unwrap();
        paint_key(
            painter,
            key_rect,
            local[idx],
            remote[idx],
            colors,
            egui::Color32::from_gray(245),
            2.0,
            white_stroke,
        );
    }

    // 2) Black keys, drawn on top. A black key sits over the boundary between
    //    the white key to its left (midi - 1) and the next white key.
    let black_w = white_w * 0.6;
    let black_h = (key_bottom - key_top) * 0.62;
    for midi in MIDI_LOW..=MIDI_HIGH {
        if !is_black_key(midi) {
            continue;
        }
        // The white key immediately below a black key is always (midi - 1).
        let Some(left_col) = white_index(midi - 1) else {
            continue;
        };
        let boundary_x = rect.left() + (left_col as f32 + 1.0) * white_w;
        let x0 = boundary_x - black_w / 2.0;
        let key_rect = egui::Rect::from_min_max(
            egui::pos2(x0, key_top),
            egui::pos2(x0 + black_w, key_top + black_h),
        );
        let idx = midi_to_key_index(midi).unwrap();
        paint_key(
            painter,
            key_rect,
            local[idx],
            remote[idx],
            colors,
            egui::Color32::from_gray(20),
            2.0,
            egui::Stroke::new(1.0, egui::Color32::BLACK),
        );
    }
}
