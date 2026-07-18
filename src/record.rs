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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Directory (relative to the working directory / project root) under which each
/// session gets its own `session_<unix-seconds>` subfolder.
const RECORDINGS_DIR: &str = "recordings";

/// Flush the WAV header, the MIDI log, and rewrite meta.json at least this often
/// while recording, so a crash/kill leaves an *aligned, playable* session
/// (valid WAV header + a MIDI tail + a meta.json with `audio_start_s`) rather
/// than a WAV with a truncated midi.jsonl and no meta.json at all (F5).
const MAINTENANCE_INTERVAL: Duration = Duration::from_secs(1);

/// Refine the audio-start anchor over this many initial buffers, taking the
/// *minimum* origin estimate. OS scheduling only ever delays a callback, so the
/// least-delayed estimate is closest to the true stream start — this sheds the
/// per-session scheduler jitter a single callback wakeup would bake into the
/// alignment anchor (F18).
const ANCHOR_WINDOW_BUFFERS: u32 = 50;

/// Stop writing the WAV before hound's unchecked `u32` data-byte counter can
/// wrap at 4 GiB (mono f32 = 4 bytes/sample). Past this the header is corrupted
/// mid-recording (release) or the writer panics (debug); instead we stop audio
/// (MIDI keeps recording) and surface an error (F17). ~5.8 h @ 48 kHz.
const MAX_WAV_SAMPLES: u64 = 1_000_000_000;

/// Cap on audio bytes buffered toward the writer thread. A stalled disk (hung
/// SMB share, sleeping USB drive) otherwise lets the realtime callback queue
/// buffers without bound; past this we drop audio and surface a backpressure
/// error instead of growing memory forever (F26).
const MAX_QUEUED_AUDIO_BYTES: usize = 64 * 1024 * 1024;

/// Messages from the hot paths (audio callback, MIDI callback, supervisor) to the
/// writer thread. Carrying the capture `Instant` lets the writer compute each
/// event's time against the session `t0` off the hot path.
enum RecEvent {
    /// Open a new session: create its directory and the MIDI log.
    Begin { t0: Instant, dir: PathBuf, wall_unix: u64 },
    /// Audio format is known (device opened) — open the WAV. The audio stream's
    /// start offset on the shared clock is taken from the first `Audio` buffer,
    /// not this event, which is enqueued before `play()` (see H9 / `writer_loop`).
    AudioFormat { sample_rate: u32, channels: u16, device: String },
    /// A buffer of mono samples captured at `at`.
    Audio { samples: Vec<f32>, at: Instant },
    /// A raw MIDI message captured at `at`.
    Midi { bytes: Vec<u8>, at: Instant },
    /// Finalize the current session (flush + fix WAV header + write meta.json).
    End,
    /// Finalize any open session and stop the writer thread — the orderly
    /// app-shutdown signal, so the WAV header and `meta.json` are always
    /// written even when the process is about to exit (C1). Unlike `End` (which
    /// only closes the current session and keeps the thread idling), this
    /// breaks the writer loop so it can be joined.
    Shutdown,
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
    /// Last write/setup error for the current session, surfaced in the UI so a
    /// failed session isn't silently presented as a healthy "Recording" (L18).
    /// Empty = no error.
    error: Arc<Mutex<String>>,
    /// Bytes of audio currently queued toward the writer thread but not yet
    /// consumed. Bounds memory when the disk stalls (F26).
    queued_bytes: Arc<AtomicUsize>,
    /// Set when [`push_audio`] had to drop a buffer because the queue was full;
    /// the writer converts it into a surfaced error.
    overflowed: Arc<AtomicBool>,
    /// Writer thread handle, taken and joined by [`Recorder::shutdown`] so the
    /// files are always finalized before the process exits (C1).
    writer_join: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
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
        let error = Arc::new(Mutex::new(String::new()));
        let queued_bytes = Arc::new(AtomicUsize::new(0));
        let overflowed = Arc::new(AtomicBool::new(false));

        let join = {
            let midi_events = Arc::clone(&midi_events);
            let audio_samples = Arc::clone(&audio_samples);
            let sample_rate = Arc::clone(&sample_rate);
            let error = Arc::clone(&error);
            let queued_bytes = Arc::clone(&queued_bytes);
            let overflowed = Arc::clone(&overflowed);
            thread::Builder::new()
                .name("recorder-writer".into())
                .spawn(move || {
                    writer_loop(
                        rx,
                        midi_events,
                        audio_samples,
                        sample_rate,
                        error,
                        queued_bytes,
                        overflowed,
                    )
                })
                .expect("failed to spawn recorder writer thread")
        };

        Recorder {
            armed: Arc::new(AtomicBool::new(false)),
            recording: Arc::new(AtomicBool::new(false)),
            tx,
            midi_events,
            audio_samples,
            sample_rate,
            session_dir,
            error,
            queued_bytes,
            overflowed,
            writer_join: Arc::new(Mutex::new(Some(join))),
        }
    }

    /// Finalize any open session and join the writer thread. Called once on app
    /// shutdown (see `input::supervise`'s tail) so the WAV header and
    /// `meta.json` are always written even though the writer runs detached (C1).
    pub fn shutdown(&self) {
        self.recording.store(false, Ordering::Relaxed);
        let _ = self.tx.send(RecEvent::Shutdown);
        if let Some(j) = self.writer_join.lock().ok().and_then(|mut g| g.take()) {
            let _ = j.join();
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

    /// The current session's setup/write error, or empty when healthy. The UI
    /// shows this so a failed session isn't presented as a live recording (L18).
    pub fn error(&self) -> String {
        self.error.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Record an externally-detected session error (e.g. the supervisor seeing
    /// the record-capture stream fail to start). Idempotent for repeated calls
    /// with the same message.
    pub fn report_error(&self, msg: &str) {
        if let Ok(mut e) = self.error.lock() {
            if *e != msg {
                *e = msg.to_string();
            }
        }
    }

    // ---- session lifecycle (driven by the input supervisor) ----

    /// Begin a new session. Resets counters, picks a session directory, and tells
    /// the writer to open it. Audio/MIDI pushed after this point is recorded.
    pub fn begin(&self) {
        let t0 = Instant::now();
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let wall_unix = now.as_secs();
        // Millisecond suffix, not bare seconds: a quick off→on Record re-toggle
        // can land in the same wall second (the supervisor poll is >=100 ms
        // apart), and a same-named dir would truncate the just-finished
        // session's audio.wav / midi.jsonl (H8).
        let dir = PathBuf::from(RECORDINGS_DIR)
            .join(format!("session_{wall_unix}_{:03}", now.subsec_millis()));

        if let Ok(mut s) = self.session_dir.lock() {
            *s = dir.display().to_string();
        }
        if let Ok(mut e) = self.error.lock() {
            e.clear();
        }
        self.midi_events.store(0, Ordering::Relaxed);
        self.audio_samples.store(0, Ordering::Relaxed);
        self.sample_rate.store(0, Ordering::Relaxed);
        self.queued_bytes.store(0, Ordering::Relaxed);
        self.overflowed.store(false, Ordering::Relaxed);

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
        });
    }

    /// Push a buffer of mono samples (called from the capture-only mic callback).
    /// Drops the buffer (rather than growing memory without bound) if the writer
    /// has fallen too far behind on a stalled disk; the writer surfaces the
    /// resulting `overflowed` flag as an error (F26).
    pub fn push_audio(&self, samples: Vec<f32>) {
        if !self.is_recording() {
            return;
        }
        let bytes = samples.len() * std::mem::size_of::<f32>();
        if self.queued_bytes.load(Ordering::Relaxed) + bytes > MAX_QUEUED_AUDIO_BYTES {
            self.overflowed.store(true, Ordering::Relaxed);
            return;
        }
        self.queued_bytes.fetch_add(bytes, Ordering::Relaxed);
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
    /// Audio stream start, in seconds since `t0`. Refined over the first
    /// `ANCHOR_WINDOW_BUFFERS` buffers as the minimum origin estimate (F18).
    audio_start_s: Option<f64>,
    /// How many buffers have contributed to the `audio_start_s` estimate.
    anchor_buffers: u32,
    midi_count: u64,
    sample_count: u64,
    /// Last time the session was flushed to disk (WAV header + MIDI log +
    /// meta.json). Drives the ~1 s crash-safety cadence (F5).
    last_maintenance: Instant,
}

/// Seconds from `t0` to `at` on the shared monotonic clock (never negative).
fn secs_since(t0: Instant, at: Instant) -> f64 {
    at.saturating_duration_since(t0).as_secs_f64()
}

/// The writer thread. Owns all files; processes one [`RecEvent`] at a time.
#[allow(clippy::too_many_arguments)]
fn writer_loop(
    rx: mpsc::Receiver<RecEvent>,
    midi_events: Arc<AtomicU64>,
    audio_samples: Arc<AtomicU64>,
    sample_rate: Arc<AtomicU32>,
    error: Arc<Mutex<String>>,
    queued_bytes: Arc<AtomicUsize>,
    overflowed: Arc<AtomicBool>,
) {
    let set_error = |msg: String| {
        eprintln!("[record] {msg}");
        if let Ok(mut e) = error.lock() {
            *e = msg;
        }
    };
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
                    // Surface it: a failed dir/log create otherwise leaves the
                    // UI showing a live recording while nothing is written (L18).
                    Err(e) => set_error(format!("failed to open session: {e}")),
                }
            }

            RecEvent::AudioFormat { sample_rate: sr, channels, device } => {
                if let Some(s) = session.as_mut() {
                    s.sample_rate = sr;
                    s.channels = channels;
                    s.device = device;
                    // NOTE: do *not* stamp `audio_start_s` here. This event is
                    // enqueued at device-config time — before `play()` and the
                    // first real sample — so its `Instant` overstates the origin
                    // by the (per-session-variable) stream-build + driver
                    // latency. The first `Audio` buffer sets it instead (H9).
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
                        Err(e) => set_error(format!("failed to create {}: {e}", wav_path.display())),
                    }
                }
            }

            RecEvent::Audio { samples, at } => {
                // Release the queue reservation this buffer made, whether or not
                // it ends up written (F26).
                queued_bytes.fetch_sub(
                    samples.len() * std::mem::size_of::<f32>(),
                    Ordering::Relaxed,
                );
                if overflowed.swap(false, Ordering::Relaxed) {
                    set_error("disk too slow — dropping audio; recording may have gaps".into());
                }
                if let Some(s) = session.as_mut() {
                    let arrival = secs_since(s.t0, at);
                    // Stop before hound's u32 data-byte counter wraps at 4 GiB
                    // (F17): finalize the WAV so its header stays valid; MIDI
                    // keeps recording.
                    if s.wav.is_some() && s.sample_count >= MAX_WAV_SAMPLES {
                        set_error(
                            "recording reached the 4 GiB WAV limit — audio stopped (MIDI still recording)".into(),
                        );
                        if let Some(w) = s.wav.take() {
                            if let Err(e) = w.finalize() {
                                eprintln!("[record] WAV finalize error: {e}");
                            }
                        }
                    }
                    if let Some(w) = s.wav.as_mut() {
                        let mut wrote = true;
                        for &sample in &samples {
                            if w.write_sample(sample).is_err() {
                                wrote = false;
                                break;
                            }
                        }
                        if wrote {
                            // Count only what actually reached the writer, so
                            // `meta.json` never describes audio that isn't there.
                            s.sample_count += samples.len() as u64;
                            audio_samples.store(s.sample_count, Ordering::Relaxed);
                        } else {
                            set_error("audio write failed (disk full?); stopping capture".into());
                            s.wav = None; // stop trying; counters stop climbing
                        }
                    }
                    // Refine the audio-start anchor over the first few buffers by
                    // taking the minimum origin estimate (F18).
                    if s.sample_rate > 0 && s.anchor_buffers < ANCHOR_WINDOW_BUFFERS {
                        let est = (arrival - s.sample_count as f64 / s.sample_rate as f64).max(0.0);
                        s.audio_start_s = Some(match s.audio_start_s {
                            Some(cur) => cur.min(est),
                            None => est,
                        });
                        s.anchor_buffers += 1;
                    }
                    // Crash-safety flush cadence (F5).
                    maintain(s);
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
                    // Flush the MIDI log + meta.json on the crash-safety cadence
                    // even when no audio is arriving (MIDI-only recording) (F5).
                    maintain(s);
                }
            }

            RecEvent::End => {
                if let Some(s) = session.take() {
                    finalize(s);
                }
            }

            RecEvent::Shutdown => {
                if let Some(s) = session.take() {
                    finalize(s);
                }
                break; // orderly app exit: let the writer thread be joined
            }
        }
    }

    // Channel closed (all handles dropped): finalize anything still open.
    if let Some(s) = session.take() {
        finalize(s);
    }
}

/// Create the session directory and open the MIDI log.
fn open_session(t0: Instant, dir: PathBuf, wall_unix: u64) -> std::io::Result<Session> {
    fs::create_dir_all(&dir)?;
    let midi_log = BufWriter::new(File::create(dir.join("midi.jsonl"))?);
    let session = Session {
        t0,
        wall_unix,
        dir,
        midi_log,
        wav: None,
        sample_rate: 0,
        channels: 0,
        device: String::new(),
        audio_start_s: None,
        anchor_buffers: 0,
        midi_count: 0,
        sample_count: 0,
        last_maintenance: Instant::now(),
    };
    // Write meta.json up front (with zero counts) so a crash before the first
    // maintenance tick still leaves *some* alignment anchor on disk; it is
    // rewritten with live counts on the maintenance cadence and at finalize (F5).
    write_meta(&session);
    Ok(session)
}

/// Write (or rewrite) meta.json from the session's current counts. Called at
/// session open, on the maintenance cadence, and at finalize so a killed
/// process always leaves an alignment anchor (F5).
fn write_meta(s: &Session) {
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
}

/// Flush everything to disk (WAV header + MIDI log + meta.json) if the
/// maintenance interval has elapsed, so a crash leaves an aligned, playable
/// session (F5). Cheap and idempotent; called from the audio and MIDI handlers.
fn maintain(s: &mut Session) {
    if s.last_maintenance.elapsed() < MAINTENANCE_INTERVAL {
        return;
    }
    if let Some(w) = s.wav.as_mut() {
        let _ = w.flush();
    }
    let _ = s.midi_log.flush();
    write_meta(s);
    s.last_maintenance = Instant::now();
}

/// Flush + close everything and write the final `meta.json`.
fn finalize(mut s: Session) {
    let _ = s.midi_log.flush();
    if let Some(w) = s.wav.take() {
        if let Err(e) = w.finalize() {
            eprintln!("[record] WAV finalize error: {e}");
        }
    }
    // Rewrite meta.json with the final counts (it was written up front and on
    // the maintenance cadence for crash-safety) (F5).
    write_meta(&s);
    let duration = if s.sample_rate > 0 {
        s.sample_count as f64 / s.sample_rate as f64
    } else {
        0.0
    };
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
