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

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use midir::MidiInputConnection;

use crate::audio::{self, AudioHandle, EngineStatus, InferenceTunables, Threshold};
use crate::bundle;
use crate::midi;
use crate::note::NoteMsg;
use crate::record::Recorder;

/// Floor on the MIDI rescan interval, so a preference of 0 can't spin the
/// supervisor. The interval itself is live-editable (see `midi_poll_ms`).
const MIN_POLL_MS: u64 = 100;

/// After a MIDI port fails to open (held exclusively by another app, flaky
/// virtual port), wait this long before retrying it — staying on the current
/// backend meanwhile instead of tearing down mic + ONNX + epoch every poll.
const MIDI_RETRY_COOLDOWN: Duration = Duration::from_secs(10);

/// After the microphone backend fails to start (no device, unsupported format)
/// or dies, wait this long before rebuilding it. Without this, a machine with no
/// usable mic rebuilds cpal + reloads the full ONNX model every ~1 s forever
/// (F4).
const MIC_RETRY_COOLDOWN: Duration = Duration::from_secs(10);

/// After the record-capture stream fails, wait this long before retrying it —
/// so a busy/absent mic doesn't respawn the capture thread every poll (F25).
const RECORD_RETRY_COOLDOWN: Duration = Duration::from_secs(5);

/// How often the supervisor refreshes the extracted ONNX Runtime's mtime, well
/// under the 24 h reap age, so a long-lived instance's runtime is never reaped
/// by a concurrent newer instance (R35).
const RUNTIME_TOUCH_INTERVAL: Duration = Duration::from_secs(60 * 60);

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
    /// Sustain-pedal (CC64) levels, 0..=127. Only the MIDI backend is ever
    /// handed the sending half (see [`start`]), so the mic path is
    /// *structurally* incapable of producing pedal events — no runtime gate.
    pub pedal: Receiver<u8>,
    pub threshold: Threshold,
    /// Live-editable ONNX/DSP tunables shared with the inference thread
    /// (Preferences ▸ Advanced writes them; the detector reads them each hop).
    pub tunables: InferenceTunables,
    /// Live-editable MIDI-port rescan interval (ms); the supervisor reads it
    /// each loop. Written by Preferences ▸ Advanced.
    pub midi_poll_ms: Arc<AtomicU64>,
    pub status: Arc<Mutex<EngineStatus>>,
    /// Training-data capture harness. The UI arms/disarms it and reads its
    /// status; the supervisor drives the actual session lifecycle.
    pub recorder: Recorder,
    source: Arc<AtomicU8>,
    epoch: Arc<AtomicU64>,
    stop_all: Arc<AtomicBool>,
    /// Joined in [`Drop`] so an orderly shutdown finalizes any open recording
    /// (WAV header + `meta.json`) before the process exits (C1).
    supervisor: Option<JoinHandle<()>>,
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

    /// Whether a MIDI device is *actually connected right now* — as opposed to
    /// the mic fallback or the brief initial "detecting" state. Distinct from
    /// [`Self::source`], which reports "detecting" as MIDI-ish; callers reacting
    /// to a real plug/unplug (e.g. auto-muting the on-screen synth) want this.
    pub fn midi_connected(&self) -> bool {
        self.source.load(Ordering::Relaxed) == SRC_MIDI
    }
}

impl Drop for InputEngine {
    fn drop(&mut self) {
        // Tell the supervisor to stop, then wait for it: its shutdown path
        // finalizes any open recording and joins the recorder writer thread, so
        // a WAV/meta.json is never left unwritten when the app closes (C1). The
        // supervisor wakes within ~100 ms (see `sleep_unless_stopped`).
        self.stop_all.store(true, Ordering::Relaxed);
        if let Some(j) = self.supervisor.take() {
            let _ = j.join();
        }
    }
}

/// Spawn the input supervisor and return immediately. The first poll runs right
/// away, so a connected MIDI device or the mic fallback comes up within
/// milliseconds; thereafter the supervisor reacts to hot-plug changes.
pub fn start(
    initial_threshold: f32,
    tunables: InferenceTunables,
    initial_midi_poll_ms: u64,
) -> InputEngine {
    let threshold = Threshold::new(initial_threshold);
    let (note_tx, note_rx) = mpsc::channel::<NoteMsg>();
    let (pedal_tx, pedal_rx) = mpsc::channel::<u8>();
    let status = Arc::new(Mutex::new(EngineStatus {
        device: "Input: detecting…".to_string(),
        model: String::new(),
    }));
    let source = Arc::new(AtomicU8::new(SRC_DETECTING));
    let epoch = Arc::new(AtomicU64::new(0));
    let stop_all = Arc::new(AtomicBool::new(false));
    let midi_poll_ms = Arc::new(AtomicU64::new(initial_midi_poll_ms));
    let recorder = Recorder::new();

    let supervisor = {
        let threshold = threshold.clone();
        let tunables = tunables.clone();
        let status = Arc::clone(&status);
        let source = Arc::clone(&source);
        let epoch = Arc::clone(&epoch);
        let stop_all = Arc::clone(&stop_all);
        let midi_poll_ms = Arc::clone(&midi_poll_ms);
        let recorder = recorder.clone();
        thread::Builder::new()
            .name("input-supervisor".into())
            .spawn(move || {
                supervise(
                    note_tx, pedal_tx, threshold, tunables, status, source, epoch, stop_all,
                    midi_poll_ms, recorder,
                )
            })
            .expect("failed to spawn input supervisor thread")
    };

    InputEngine {
        notes: note_rx,
        pedal: pedal_rx,
        threshold,
        tunables,
        midi_poll_ms,
        status,
        recorder,
        source,
        epoch,
        stop_all,
        supervisor: Some(supervisor),
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
#[allow(clippy::too_many_arguments)]
fn supervise(
    note_tx: Sender<NoteMsg>,
    pedal_tx: Sender<u8>,
    threshold: Threshold,
    tunables: InferenceTunables,
    status: Arc<Mutex<EngineStatus>>,
    source: Arc<AtomicU8>,
    epoch: Arc<AtomicU64>,
    stop_all: Arc<AtomicBool>,
    midi_poll_ms: Arc<AtomicU64>,
    recorder: Recorder,
) {
    let mut active = Active::None;
    // Capture-only mic for the recording harness; runs alongside whatever the
    // live note source is. `Some` exactly while a capture stream is running.
    let mut record_capture: Option<AudioHandle> = None;
    // MIDI ports we recently failed to open, keyed by name, with the time — so
    // we back off instead of thrashing them, and try *other* ports meanwhile so
    // an un-openable port 0 doesn't starve a working port 1 (H4/F9).
    let mut failed_midi: HashMap<String, Instant> = HashMap::new();
    // Time the mic backend last failed, so we back off rebuilding it instead of
    // reloading ONNX every poll (F4).
    let mut failed_mic: Option<Instant> = None;
    // Time the record-capture stream last failed, so a busy mic doesn't respawn
    // capture every poll (F25).
    let mut failed_record: Option<Instant> = None;
    // Arm/disarm edge count seen at the last reconcile, to catch a full toggle
    // cycle completed between two polls (R10).
    let mut last_arm_edges: u64 = recorder.arm_edges();
    // Periodically bump the extracted ONNX Runtime's mtime so a concurrent newer
    // instance's 24 h reaper never deletes a runtime this long-lived (possibly
    // MIDI-only, never-loads-ORT) instance still depends on (R35).
    let mut last_runtime_touch = Instant::now();
    bundle::touch_runtime();

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

        // 2) A mic backend that failed to start (no device / unsupported
        //    format) or died mid-session (device unplugged) reports it via the
        //    handle. Tear it down so the fallback below re-detects it, rather
        //    than leaving a silent zombie with healthy-looking status (H5/M11).
        //    Record the failure time so step 4 backs off instead of rebuilding
        //    cpal + reloading ONNX every poll for a permanently-dead mic (F4).
        if matches!(&active, Active::Mic(h) if h.failed()) {
            eprintln!("[input] microphone backend died; will re-detect");
            if let Active::Mic(h) = std::mem::replace(&mut active, Active::None) {
                h.stop();
            }
            // Preserve the failure text through the retry cooldown rather than
            // showing the generic "detecting…", which reads as a healthy
            // transient state on a machine with no usable mic (R29). Keep the
            // audio thread's error message if it left one.
            let prev = status.lock().ok().map(|s| s.device.clone()).unwrap_or_default();
            source.store(SRC_DETECTING, Ordering::Relaxed);
            if let Ok(mut s) = status.lock() {
                s.device = if prev.is_empty() || prev == "Input: detecting…" {
                    "Mic failed — retrying in 10 s".to_string()
                } else {
                    format!("{prev} — retrying in 10 s")
                };
                s.model = String::new();
            }
            failed_mic = Some(Instant::now());
        }

        // 3) A MIDI device is present and we're not already on one of its ports:
        //    switch to it. Connect *before* tearing down the mic, so an
        //    un-openable port (held exclusively by another app) never kills a
        //    working mic + ONNX every poll (H4); iterate all non-cooling ports
        //    so an un-openable port 0 doesn't starve a working port 1 (F9). On a
        //    successful connect, bump the epoch (which force-releases stale keys
        //    in the UI) *before* the blocking mic teardown, so genuine MIDI
        //    Note-Ons already arriving on the new connection aren't wiped by a
        //    switch that bumped the epoch only after `handle.stop()` returned (F8).
        let on_live_midi = matches!(&active, Active::Midi { name, .. } if ports.iter().any(|p| p == name));
        if !on_live_midi {
            let mut opened = None;
            for want in &ports {
                if failed_midi.get(want).is_some_and(|at| at.elapsed() < MIDI_RETRY_COOLDOWN) {
                    continue;
                }
                // Bump the epoch *before* `connect_to`, which starts delivering
                // Note-Ons into `note_tx` the instant it returns: a note struck
                // in the gap between the callback going live and a later bump was
                // consumed under the old epoch and then wiped by the force-release
                // (R28). A failed attempt over-bumps by one (a harmless spurious
                // force-release), but succeeding candidates open on the first try.
                epoch.fetch_add(1, Ordering::Relaxed);
                // Only MIDI gets the pedal sender: the mic backend is never
                // wired to it (the pedal feature's structural guarantee).
                match midi::connect_to(
                    note_tx.clone(),
                    &status,
                    want,
                    recorder.clone(),
                    pedal_tx.clone(),
                ) {
                    Ok(conn) => {
                        opened = Some((conn, want.clone()));
                        break;
                    }
                    Err(e) => {
                        eprintln!("[input] MIDI connect failed for '{want}' ({e}); trying next port");
                        failed_midi.insert(want.clone(), Instant::now());
                    }
                }
            }
            if let Some((conn, name)) = opened {
                failed_midi.remove(&name);
                failed_mic = None;
                source.store(SRC_MIDI, Ordering::Relaxed);
                // Install the MIDI backend and stop the old mic (if any) — the
                // epoch bump above already covers its trailing Note-Offs.
                if let Active::Mic(handle) =
                    std::mem::replace(&mut active, Active::Midi { _conn: conn, name })
                {
                    handle.stop();
                }
            }
            // All candidates failed (or none): `active` is unchanged — a working
            // mic keeps running (H4), and step 4 brings one up if there is none.
        }

        // 4) Nothing live (no MIDI, or every port is un-connectable and cooling
        //    down): bring up the mic fallback, unless it recently failed and is
        //    still cooling down (F4).
        if matches!(active, Active::None) {
            let cooling = failed_mic.is_some_and(|at| at.elapsed() < MIC_RETRY_COOLDOWN);
            if !cooling {
                failed_mic = None;
                active = start_mic(&note_tx, &threshold, &tunables, &status, &source);
            }
        }

        // A record-capture stream that failed to start / died yields a
        // labels-only stretch: tear the dead handle down (never leak it) and
        // surface the error, but keep the *session* open so MIDI labels keep
        // recording and a retried capture resumes audio into the same WAV
        // (`audio_start_s` is set by the first buffer that ever arrives). Back
        // off before retrying so a busy mic doesn't respawn capture every poll
        // (M11/F25).
        if record_capture.as_ref().is_some_and(|h| h.failed()) {
            recorder.report_error("microphone capture failed — retrying; audio may be missing");
            if let Some(h) = record_capture.take() {
                h.stop();
            }
            failed_record = Some(Instant::now());
        }

        // Reconcile the recording harness with the UI's Record toggle. A session
        // runs whenever armed: the capture-only mic logs audio, and the MIDI
        // callback (already teeing) logs labels whenever a device is connected.
        reconcile_recording(
            &recorder,
            &mut record_capture,
            &mut failed_record,
            &mut last_arm_edges,
        );

        // Keep the extracted ONNX Runtime's mtime fresh (~hourly) so a concurrent
        // newer instance's reaper doesn't delete it out from under us (R35).
        if last_runtime_touch.elapsed() >= RUNTIME_TOUCH_INTERVAL {
            bundle::touch_runtime();
            last_runtime_touch = Instant::now();
        }

        // Live poll interval (Preferences ▸ Advanced), floored so a tiny value
        // can't busy-spin the supervisor.
        let poll = Duration::from_millis(midi_poll_ms.load(Ordering::Relaxed).max(MIN_POLL_MS));
        sleep_unless_stopped(&stop_all, poll);
    }

    // App is shutting down: stop the record-capture stream (joining its thread
    // and closing the mic — never leak it), finalize any open recording, then
    // stop the live mic if it's the active backend (a MIDI connection just drops
    // with `active`) (F24).
    if let Some(handle) = record_capture.take() {
        handle.stop();
    }
    if recorder.is_recording() {
        recorder.end();
    }
    if let Active::Mic(handle) = active {
        handle.stop();
    }
    // Flush + join the recorder writer thread so the WAV header and meta.json
    // are always written before the process exits (C1).
    recorder.shutdown();
}

/// Start or stop the recording session to match `recorder.is_armed()`.
/// `record_capture` is `Some` exactly while a capture stream is running.
/// `failed_record` gates retries after a capture failure so a busy/absent mic
/// doesn't respawn capture every poll (F25).
fn reconcile_recording(
    recorder: &Recorder,
    record_capture: &mut Option<AudioHandle>,
    failed_record: &mut Option<Instant>,
    last_arm_edges: &mut u64,
) {
    let want = recorder.is_armed();
    // Did a full arm/disarm cycle complete since the last poll? Two-or-more edges
    // ending in the same level as our current session state means a take boundary
    // was toggled between polls (R10). Cut the session so a quick Stop→Record
    // doesn't merge two takes into one dir.
    let edges = recorder.arm_edges();
    let cycled = edges.wrapping_sub(*last_arm_edges) >= 2;
    *last_arm_edges = edges;
    if cycled && want && record_capture.is_some() && recorder.is_recording() {
        // Off→on happened while a session was running: finalize the current take
        // and start a fresh one so the two are separate sessions.
        if let Some(handle) = record_capture.take() {
            handle.stop();
        }
        recorder.end();
        *failed_record = None;
    }
    match (want, record_capture.is_some()) {
        (true, false) => {
            // Cooling down after a capture failure: keep the session open (MIDI
            // labels still record) and retry capture only once the cooldown
            // elapses.
            if failed_record.is_some_and(|at| at.elapsed() < RECORD_RETRY_COOLDOWN) {
                return;
            }
            *failed_record = None;
            // Open the session only if one isn't already running — a retry after
            // a mid-session capture failure resumes audio into the same session.
            if !recorder.is_recording() {
                recorder.begin();
            }
            *record_capture = Some(audio::start_record_capture(recorder.clone()));
        }
        (false, true) => {
            if let Some(handle) = record_capture.take() {
                handle.stop();
            }
            recorder.end();
            *failed_record = None;
        }
        (false, false) => {
            // Disarmed after a capture failure closed the stream but left the
            // session open: finalize it.
            if recorder.is_recording() {
                recorder.end();
            }
            *failed_record = None;
        }
        (true, true) => {}
    }
}

/// Start the microphone + ONNX backend and mark the source as the mic.
fn start_mic(
    note_tx: &Sender<NoteMsg>,
    threshold: &Threshold,
    tunables: &InferenceTunables,
    status: &Arc<Mutex<EngineStatus>>,
    source: &Arc<AtomicU8>,
) -> Active {
    if let Ok(mut s) = status.lock() {
        s.device = "Mic: initializing…".to_string();
        s.model = "Model: loading…".to_string();
    }
    let handle = audio::start_into(
        note_tx.clone(),
        threshold.clone(),
        tunables.clone(),
        Arc::clone(status),
    );
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
