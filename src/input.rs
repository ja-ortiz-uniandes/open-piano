//! Input orchestration: one note source, one channel, hot-swappable backends.
//!
//! A background **supervisor** thread continuously decides which backend feeds
//! the UI:
//!
//!   * **MIDI (preferred)** — if any MIDI input port is present, connect to it.
//!   * **Microphone (fallback)** — only when no MIDI device exists, capture the
//!     mic and transcribe with ONNX.
//!
//! Both backends push the same [`NoteMsg`] values into the *same* `mpsc`
//! channel, so the UI is agnostic about where a note came from.
//!
//!   ┌─ MIDI port (preferred) ─┐
//!   │                         ├──> mpsc::Sender<NoteMsg> ──> UI
//!   └─ mic + ONNX (fallback) ─┘
//!
//! Hot-plug: the supervisor polls the MIDI port list ~once a second. Plug in a
//! piano and it switches to MIDI (stopping the mic so it isn't held open);
//! unplug it and it releases any held notes and falls back to the mic. The mic
//! is started lazily — it never runs while a MIDI device is connected.
//!
//! Switching always bumps `epoch`; the UI watches it and force-releases every
//! locally-held note on a switch so nothing can stay stuck "on".

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use midir::MidiInputConnection;

use crate::audio::{self, AudioHandle, EngineStatus, Threshold};
use crate::midi;
use crate::note::NoteMsg;
use crate::record::Recorder;

/// How often the supervisor rescans the MIDI port list for hot-plug changes.
const POLL_INTERVAL: Duration = Duration::from_millis(1000);

// `source` atomic encoding (it's read by the UI thread).
const SRC_DETECTING: u8 = 0;
const SRC_MIDI: u8 = 1;
const SRC_MIC: u8 = 2;

/// The active note source, as observed by the UI.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    Midi,
    Microphone,
}

/// Handle held by the UI. Owns the note receiver and shared status/threshold.
/// The actual backends live on the supervisor thread; dropping this signals
/// that thread to tear everything down.
pub struct InputEngine {
    pub notes: Receiver<NoteMsg>,
    pub threshold: Threshold,
    pub status: Arc<Mutex<EngineStatus>>,
    /// Training-data capture harness. The UI arms/disarms it and reads its
    /// status; the supervisor drives the actual session lifecycle.
    pub recorder: Recorder,
    source: Arc<AtomicU8>,
    epoch: Arc<AtomicU64>,
    stop_all: Arc<AtomicBool>,
    _supervisor: JoinHandle<()>,
}

impl InputEngine {
    /// The currently-active input source (defaults to MIDI-ish while detecting;
    /// the only UI consequence is hiding the mic-only threshold slider).
    pub fn source(&self) -> Source {
        match self.source.load(Ordering::Relaxed) {
            SRC_MIC => Source::Microphone,
            _ => Source::Midi,
        }
    }

    /// A counter bumped on every backend switch. When it changes, the UI should
    /// force-release all held notes (see the module docs).
    pub fn switch_epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }
}

impl Drop for InputEngine {
    fn drop(&mut self) {
        // Tell the supervisor to stop; it tears down whatever backend is live.
        self.stop_all.store(true, Ordering::Relaxed);
    }
}

/// Spawn the input supervisor and return immediately. The first poll runs right
/// away, so a connected MIDI device or the mic fallback comes up within
/// milliseconds; thereafter the supervisor reacts to hot-plug changes.
pub fn start(initial_threshold: f32) -> InputEngine {
    let threshold = Threshold::new(initial_threshold);
    let (note_tx, note_rx) = mpsc::channel::<NoteMsg>();
    let status = Arc::new(Mutex::new(EngineStatus {
        device: "Input: detecting…".to_string(),
        model: String::new(),
    }));
    let source = Arc::new(AtomicU8::new(SRC_DETECTING));
    let epoch = Arc::new(AtomicU64::new(0));
    let stop_all = Arc::new(AtomicBool::new(false));
    let recorder = Recorder::new();

    let supervisor = {
        let threshold = threshold.clone();
        let status = Arc::clone(&status);
        let source = Arc::clone(&source);
        let epoch = Arc::clone(&epoch);
        let stop_all = Arc::clone(&stop_all);
        let recorder = recorder.clone();
        thread::Builder::new()
            .name("input-supervisor".into())
            .spawn(move || supervise(note_tx, threshold, status, source, epoch, stop_all, recorder))
            .expect("failed to spawn input supervisor thread")
    };

    InputEngine {
        notes: note_rx,
        threshold,
        status,
        recorder,
        source,
        epoch,
        stop_all,
        _supervisor: supervisor,
    }
}

/// The currently-running backend. Held only by the supervisor thread, so the
/// (non-`Send`) MIDI connection never crosses a thread boundary.
enum Active {
    None,
    Midi {
        _conn: MidiInputConnection<()>,
        name: String,
    },
    Mic(AudioHandle),
}

/// Supervisor loop: poll MIDI ports and keep exactly one backend live, MIDI
/// preferred. Runs until `stop_all` is set (i.e. the `InputEngine` is dropped).
fn supervise(
    note_tx: Sender<NoteMsg>,
    threshold: Threshold,
    status: Arc<Mutex<EngineStatus>>,
    source: Arc<AtomicU8>,
    epoch: Arc<AtomicU64>,
    stop_all: Arc<AtomicBool>,
    recorder: Recorder,
) {
    let mut active = Active::None;
    // Capture-only mic for the recording harness; runs alongside whatever the
    // live note source is. `Some` exactly while a recording session is open.
    let mut record_capture: Option<AudioHandle> = None;

    while !stop_all.load(Ordering::Relaxed) {
        let ports = midi::port_names();

        // 1) If the MIDI device we were using has vanished, drop it. Releasing
        //    held notes is handled by the epoch bump below (MIDI gives us no
        //    Note Offs for a yanked cable).
        if let Active::Midi { name, .. } = &active {
            if !ports.iter().any(|p| p == name) {
                eprintln!("[input] MIDI device '{name}' disconnected; switching to microphone");
                active = Active::None; // drops the connection -> closes the port
                set_detecting(&status, &source);
                epoch.fetch_add(1, Ordering::Relaxed);
            }
        }

        if let Some(want) = ports.first() {
            // 2) A MIDI device is available. Switch to it unless we're already
            //    connected to that exact port.
            let already = matches!(&active, Active::Midi { name, .. } if name == want);
            if !already {
                // Stop the mic first so it isn't held open while on MIDI; its
                // inference thread releases any held notes as it exits.
                if let Active::Mic(handle) = std::mem::replace(&mut active, Active::None) {
                    handle.stop();
                }
                epoch.fetch_add(1, Ordering::Relaxed);

                active = match midi::connect_to(note_tx.clone(), &status, want, recorder.clone()) {
                    Ok(conn) => {
                        source.store(SRC_MIDI, Ordering::Relaxed);
                        Active::Midi {
                            _conn: conn,
                            name: want.clone(),
                        }
                    }
                    Err(e) => {
                        eprintln!("[input] MIDI connect failed ({e}); using microphone");
                        start_mic(&note_tx, &threshold, &status, &source)
                    }
                };
            }
        } else if matches!(active, Active::None) {
            // 3) No MIDI device at all: bring up the mic fallback (lazily — only
            //    when there's nothing better, and only if not already running).
            active = start_mic(&note_tx, &threshold, &status, &source);
        }

        // Reconcile the recording harness with the UI's Record toggle. A session
        // runs whenever armed: the capture-only mic logs audio, and the MIDI
        // callback (already teeing) logs labels whenever a device is connected.
        reconcile_recording(&recorder, &mut record_capture);

        sleep_unless_stopped(&stop_all, POLL_INTERVAL);
    }

    // App is shutting down: finalize any open recording, then stop the live mic
    // if it's the active backend (a MIDI connection just drops with `active`).
    if record_capture.take().is_some() {
        recorder.end();
    }
    if let Active::Mic(handle) = active {
        handle.stop();
    }
}

/// Start or stop the recording session to match `recorder.is_armed()`.
/// `record_capture` is `Some` exactly while a session is open.
fn reconcile_recording(recorder: &Recorder, record_capture: &mut Option<AudioHandle>) {
    let want = recorder.is_armed();
    let active = record_capture.is_some();
    if want && !active {
        recorder.begin();
        *record_capture = Some(audio::start_record_capture(recorder.clone()));
    } else if !want && active {
        if let Some(handle) = record_capture.take() {
            handle.stop();
        }
        recorder.end();
    }
}

/// Start the microphone + ONNX backend and mark the source as the mic.
fn start_mic(
    note_tx: &Sender<NoteMsg>,
    threshold: &Threshold,
    status: &Arc<Mutex<EngineStatus>>,
    source: &Arc<AtomicU8>,
) -> Active {
    if let Ok(mut s) = status.lock() {
        s.device = "Mic: initializing…".to_string();
        s.model = "Model: loading…".to_string();
    }
    let handle = audio::start_into(note_tx.clone(), threshold.clone(), Arc::clone(status));
    source.store(SRC_MIC, Ordering::Relaxed);
    Active::Mic(handle)
}

/// Reset status to the transient "detecting" state shown between backends.
fn set_detecting(status: &Arc<Mutex<EngineStatus>>, source: &Arc<AtomicU8>) {
    source.store(SRC_DETECTING, Ordering::Relaxed);
    if let Ok(mut s) = status.lock() {
        s.device = "Input: detecting…".to_string();
        s.model = String::new();
    }
}

/// Sleep `total`, but wake early (within ~100 ms) if a shutdown is requested,
/// so dropping the `InputEngine` doesn't stall on a full poll interval.
fn sleep_unless_stopped(stop_all: &Arc<AtomicBool>, total: Duration) {
    let step = Duration::from_millis(100);
    let mut slept = Duration::ZERO;
    while slept < total {
        if stop_all.load(Ordering::Relaxed) {
            return;
        }
        thread::sleep(step);
        slept += step;
    }
}
