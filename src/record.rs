//! Training-data capture harness.
//!
//! Records, *simultaneously*, the two halves of a supervised transcription
//! training pair:
//!
//!   * **`audio.wav`** — raw mono microphone audio at the device's native sample
//!     rate (float32). This is the model *input*.
//!   * **`midi.jsonl`** — every MIDI event from the connected digital piano
//!     (note on/off *with velocity*, and control changes *including CC64 sustain
//!     pedal*), each stamped with a time in seconds on a clock shared with the
//!     audio. These are the ground-truth *labels*.
//!   * **`meta.json`** — sample rate, channel count, the audio stream's start
//!     offset on the shared clock, device name, event counts, and the session's
//!     wall-clock start. Everything the offline alignment script needs.
//!
//! ## The shared clock
//!
//! Alignment is the make-or-break detail. Both the audio capture thread and the
//! MIDI callback thread stamp events with `Instant::now()`, which is a single
//! process-wide monotonic clock — so MIDI events and audio buffers are directly
//! comparable. The session's `t0` (also an `Instant`) is the zero point; every
//! logged time is `instant - t0` in seconds.
//!
//! There is still an unknown *fixed* latency offset between the two streams
//! (USB-MIDI latency, audio input buffering, OS scheduling). That residual is
//! measured **offline**, once, by recording a sharp staccato note and lining the
//! audio onset up against its MIDI timestamp — the harness's job is only to put
//! both streams on the same monotonic clock with the audio's per-sample timing
//! intact, which it does.
//!
//! ## Threading
//!
//! [`Recorder`] is a cheap-to-clone handle (all shared state is `Arc`/atomic and
//! the event channel `Sender` is `Clone`). It is held by the UI (to arm/disarm
//! and read status), the input supervisor (to drive the session lifecycle), the
//! capture-only mic thread (to push audio), and the MIDI callback (to push
//! events). All disk I/O happens on a dedicated **writer thread**, so neither the
//! realtime audio callback nor the MIDI callback ever blocks on the filesystem.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Directory (relative to the working directory / project root) under which each
/// session gets its own `session_<unix-seconds>` subfolder.
const RECORDINGS_DIR: &str = "recordings";

/// Messages from the hot paths (audio callback, MIDI callback, supervisor) to the
/// writer thread. Carrying the capture `Instant` lets the writer compute each
/// event's time against the session `t0` off the hot path.
enum RecEvent {
    /// Open a new session: create its directory and the MIDI log.
    Begin { t0: Instant, dir: PathBuf, wall_unix: u64 },
    /// Audio format is known (device opened) — open the WAV and note the audio
    /// stream's start offset on the shared clock.
    AudioFormat { sample_rate: u32, channels: u16, device: String, at: Instant },
    /// A buffer of mono samples captured at `at`.
    Audio { samples: Vec<f32>, at: Instant },
    /// A raw MIDI message captured at `at`.
    Midi { bytes: Vec<u8>, at: Instant },
    /// Finalize the current session (flush + fix WAV header + write meta.json).
    End,
}

/// Cheap-to-clone handle to the capture harness. See module docs.
#[derive(Clone)]
pub struct Recorder {
    /// User intent: the Record toggle in the UI. The supervisor reconciles this
    /// into actual session start/stop.
    armed: Arc<AtomicBool>,
    /// True between `begin()` and `end()` — i.e. a session is actively writing.
    recording: Arc<AtomicBool>,
    tx: Sender<RecEvent>,
    // --- live counters for the UI status line (updated by the writer) ---
    midi_events: Arc<AtomicU64>,
    audio_samples: Arc<AtomicU64>,
    sample_rate: Arc<AtomicU32>,
    /// Path of the current (or most recent) session directory, for the UI.
    session_dir: Arc<Mutex<String>>,
}

impl Recorder {
    /// Spawn the writer thread and return a handle. The thread lives for the
    /// process; it idles between sessions and exits only when every `Recorder`
    /// clone (hence every `Sender`) is dropped.
    pub fn new() -> Recorder {
        let (tx, rx) = mpsc::channel::<RecEvent>();
        let midi_events = Arc::new(AtomicU64::new(0));
        let audio_samples = Arc::new(AtomicU64::new(0));
        let sample_rate = Arc::new(AtomicU32::new(0));
        let session_dir = Arc::new(Mutex::new(String::new()));

        {
            let midi_events = Arc::clone(&midi_events);
            let audio_samples = Arc::clone(&audio_samples);
            let sample_rate = Arc::clone(&sample_rate);
            thread::Builder::new()
                .name("recorder-writer".into())
                .spawn(move || writer_loop(rx, midi_events, audio_samples, sample_rate))
                .expect("failed to spawn recorder writer thread");
        }

        Recorder {
            armed: Arc::new(AtomicBool::new(false)),
            recording: Arc::new(AtomicBool::new(false)),
            tx,
            midi_events,
            audio_samples,
            sample_rate,
            session_dir,
        }
    }

    // ---- UI-facing state ----

    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed)
    }

    pub fn set_armed(&self, armed: bool) {
        self.armed.store(armed, Ordering::Relaxed);
    }

    pub fn is_recording(&self) -> bool {
        self.recording.load(Ordering::Relaxed)
    }

    pub fn midi_event_count(&self) -> u64 {
        self.midi_events.load(Ordering::Relaxed)
    }

    /// Seconds of audio captured so far in the current session (0 if unknown).
    pub fn audio_seconds(&self) -> f32 {
        let sr = self.sample_rate.load(Ordering::Relaxed);
        if sr == 0 {
            return 0.0;
        }
        self.audio_samples.load(Ordering::Relaxed) as f32 / sr as f32
    }

    pub fn session_dir(&self) -> String {
        self.session_dir.lock().map(|s| s.clone()).unwrap_or_default()
    }

    // ---- session lifecycle (driven by the input supervisor) ----

    /// Begin a new session. Resets counters, picks a session directory, and tells
    /// the writer to open it. Audio/MIDI pushed after this point is recorded.
    pub fn begin(&self) {
        let t0 = Instant::now();
        let wall_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let dir = PathBuf::from(RECORDINGS_DIR).join(format!("session_{wall_unix}"));

        if let Ok(mut s) = self.session_dir.lock() {
            *s = dir.display().to_string();
        }
        self.midi_events.store(0, Ordering::Relaxed);
        self.audio_samples.store(0, Ordering::Relaxed);
        self.sample_rate.store(0, Ordering::Relaxed);

        self.recording.store(true, Ordering::Relaxed);
        let _ = self.tx.send(RecEvent::Begin { t0, dir, wall_unix });
    }

    /// End the current session: stop accepting data and finalize the files.
    pub fn end(&self) {
        // Stop the hot paths from enqueuing more first, then flush.
        self.recording.store(false, Ordering::Relaxed);
        let _ = self.tx.send(RecEvent::End);
    }

    // ---- hot-path pushes (no-ops unless recording) ----

    /// Announce the captured audio format. Call once per session, before audio.
    pub fn audio_format(&self, sample_rate: u32, channels: u16, device: String) {
        if !self.is_recording() {
            return;
        }
        let _ = self.tx.send(RecEvent::AudioFormat {
            sample_rate,
            channels,
            device,
            at: Instant::now(),
        });
    }

    /// Push a buffer of mono samples (called from the capture-only mic callback).
    pub fn push_audio(&self, samples: Vec<f32>) {
        if !self.is_recording() {
            return;
        }
        let _ = self.tx.send(RecEvent::Audio {
            samples,
            at: Instant::now(),
        });
    }

    /// Push a raw MIDI message (called from the MIDI input callback). Velocity
    /// and CC bytes are preserved verbatim; the writer parses them.
    pub fn push_midi(&self, bytes: &[u8]) {
        if !self.is_recording() {
            return;
        }
        let _ = self.tx.send(RecEvent::Midi {
            bytes: bytes.to_vec(),
            at: Instant::now(),
        });
    }
}

impl Default for Recorder {
    fn default() -> Self {
        Recorder::new()
    }
}

/// Per-session writer state, held only on the writer thread.
struct Session {
    t0: Instant,
    wall_unix: u64,
    dir: PathBuf,
    midi_log: BufWriter<File>,
    wav: Option<hound::WavWriter<BufWriter<File>>>,
    // Captured from `AudioFormat` so `meta.json` can be written at `End`.
    sample_rate: u32,
    channels: u16,
    device: String,
    /// Audio stream start, in seconds since `t0` (first buffer's arrival).
    audio_start_s: Option<f64>,
    midi_count: u64,
    sample_count: u64,
}

/// Seconds from `t0` to `at` on the shared monotonic clock (never negative).
fn secs_since(t0: Instant, at: Instant) -> f64 {
    at.saturating_duration_since(t0).as_secs_f64()
}

/// The writer thread. Owns all files; processes one [`RecEvent`] at a time.
fn writer_loop(
    rx: mpsc::Receiver<RecEvent>,
    midi_events: Arc<AtomicU64>,
    audio_samples: Arc<AtomicU64>,
    sample_rate: Arc<AtomicU32>,
) {
    let mut session: Option<Session> = None;

    while let Ok(ev) = rx.recv() {
        match ev {
            RecEvent::Begin { t0, dir, wall_unix } => {
                // A new Begin while one is open shouldn't happen (the supervisor
                // ends first), but be defensive: finalize the old one.
                if let Some(s) = session.take() {
                    finalize(s);
                }
                match open_session(t0, dir, wall_unix) {
                    Ok(s) => session = Some(s),
                    Err(e) => eprintln!("[record] failed to open session: {e}"),
                }
            }

            RecEvent::AudioFormat { sample_rate: sr, channels, device, at } => {
                if let Some(s) = session.as_mut() {
                    s.sample_rate = sr;
                    s.channels = channels;
                    s.device = device;
                    s.audio_start_s = Some(secs_since(s.t0, at));
                    sample_rate.store(sr, Ordering::Relaxed);
                    let wav_path = s.dir.join("audio.wav");
                    let spec = hound::WavSpec {
                        channels: 1, // we always down-mix to mono before recording
                        sample_rate: sr,
                        bits_per_sample: 32,
                        sample_format: hound::SampleFormat::Float,
                    };
                    match hound::WavWriter::create(&wav_path, spec) {
                        Ok(w) => s.wav = Some(w),
                        Err(e) => eprintln!("[record] failed to create {}: {e}", wav_path.display()),
                    }
                }
            }

            RecEvent::Audio { samples, at } => {
                if let Some(s) = session.as_mut() {
                    if s.audio_start_s.is_none() {
                        s.audio_start_s = Some(secs_since(s.t0, at));
                    }
                    if let Some(w) = s.wav.as_mut() {
                        for &sample in &samples {
                            // Ignore per-sample write errors after logging once
                            // would spam; hound only errors on I/O failure.
                            let _ = w.write_sample(sample);
                        }
                    }
                    s.sample_count += samples.len() as u64;
                    audio_samples.store(s.sample_count, Ordering::Relaxed);
                }
            }

            RecEvent::Midi { bytes, at } => {
                if let Some(s) = session.as_mut() {
                    let t = secs_since(s.t0, at);
                    if let Some(line) = midi_json_line(t, &bytes) {
                        let _ = writeln!(s.midi_log, "{line}");
                        s.midi_count += 1;
                        midi_events.store(s.midi_count, Ordering::Relaxed);
                    }
                }
            }

            RecEvent::End => {
                if let Some(s) = session.take() {
                    finalize(s);
                }
            }
        }
    }

    // Channel closed (app exiting): finalize anything still open.
    if let Some(s) = session.take() {
        finalize(s);
    }
}

/// Create the session directory and open the MIDI log.
fn open_session(t0: Instant, dir: PathBuf, wall_unix: u64) -> std::io::Result<Session> {
    fs::create_dir_all(&dir)?;
    let midi_log = BufWriter::new(File::create(dir.join("midi.jsonl"))?);
    Ok(Session {
        t0,
        wall_unix,
        dir,
        midi_log,
        wav: None,
        sample_rate: 0,
        channels: 0,
        device: String::new(),
        audio_start_s: None,
        midi_count: 0,
        sample_count: 0,
    })
}

/// Flush + close everything and write `meta.json`.
fn finalize(mut s: Session) {
    let _ = s.midi_log.flush();
    if let Some(w) = s.wav.take() {
        if let Err(e) = w.finalize() {
            eprintln!("[record] WAV finalize error: {e}");
        }
    }
    let audio_start = s.audio_start_s.unwrap_or(0.0);
    let duration = if s.sample_rate > 0 {
        s.sample_count as f64 / s.sample_rate as f64
    } else {
        0.0
    };
    // Hand-written JSON keeps the dependency surface tiny; the fields are simple
    // scalars/strings so there's nothing to escape beyond the device name.
    let meta = format!(
        "{{\n  \"sample_rate\": {sr},\n  \"channels\": {ch},\n  \"audio_start_s\": {astart},\n  \"audio_duration_s\": {dur},\n  \"midi_events\": {midi},\n  \"audio_samples\": {samp},\n  \"wall_clock_unix\": {wall},\n  \"device\": \"{dev}\",\n  \"clock\": \"audio.wav sample i is at time audio_start_s + i/sample_rate on the same clock as midi.jsonl 't'\"\n}}\n",
        sr = s.sample_rate,
        ch = s.channels,
        astart = audio_start,
        dur = duration,
        midi = s.midi_count,
        samp = s.sample_count,
        wall = s.wall_unix,
        dev = json_escape(&s.device),
    );
    if let Err(e) = fs::write(s.dir.join("meta.json"), meta) {
        eprintln!("[record] meta.json write error: {e}");
    }
    eprintln!(
        "[record] session saved: {} ({} MIDI events, {:.1}s audio)",
        s.dir.display(),
        s.midi_count,
        duration
    );
}

/// Format one raw MIDI message as a JSON object line, or `None` for messages we
/// don't log (anything that isn't a note on/off or control change). Velocity and
/// controller value are preserved so the offline pipeline has full ground truth,
/// including CC64 sustain pedal.
fn midi_json_line(t: f64, bytes: &[u8]) -> Option<String> {
    if bytes.len() < 3 {
        return None;
    }
    let status = bytes[0] & 0xF0;
    let channel = bytes[0] & 0x0F;
    let d1 = bytes[1];
    let d2 = bytes[2];
    match status {
        0x90 if d2 > 0 => Some(format!(
            "{{\"t\":{t:.6},\"type\":\"note_on\",\"note\":{d1},\"vel\":{d2},\"ch\":{channel}}}"
        )),
        // Note on with velocity 0 is, by convention, a note off.
        0x90 | 0x80 => Some(format!(
            "{{\"t\":{t:.6},\"type\":\"note_off\",\"note\":{d1},\"vel\":{d2},\"ch\":{channel}}}"
        )),
        // Control change — covers CC64 sustain, CC66 sostenuto, CC67 soft, etc.
        0xB0 => Some(format!(
            "{{\"t\":{t:.6},\"type\":\"cc\",\"ctrl\":{d1},\"val\":{d2},\"ch\":{channel}}}"
        )),
        _ => None,
    }
}

/// Minimal JSON string escaping for the one free-form field (device name).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
