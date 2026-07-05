//! Live microphone capture (cpal / WASAPI) feeding the ONNX inference thread.
//!
//! Architecture:
//!   cpal callback  --(raw mono samples @ device rate)-->  inference thread
//!       (see crate::inference)                                  |
//!                                                       (NoteMsg) v
//!                                                            egui UI
//!
//! The cpal callback only down-mixes interleaved frames to mono and forwards
//! them over an mpsc channel; all model work happens on the inference thread so
//! neither the audio driver callback nor the GUI render loop ever stalls.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;

use crate::inference;
use crate::note::NoteMsg;
use crate::record::Recorder;

/// A shared, lock-free `f32` (stored as its bit pattern in an [`AtomicU32`]):
/// the UI thread writes it, a realtime/inference thread reads it every hop with
/// no locking. Cloning shares the same cell. Originally just the detection
/// threshold; now the reusable primitive behind every live-editable audio knob.
#[derive(Clone)]
pub struct SharedF32(Arc<AtomicU32>);

impl SharedF32 {
    pub fn new(initial: f32) -> Self {
        SharedF32(Arc::new(AtomicU32::new(initial.to_bits())))
    }
    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
    pub fn set(&self, v: f32) {
        self.0.store(v.to_bits(), Ordering::Relaxed);
    }
}

/// The mic detection threshold (model posterior probability, 0..1). A
/// [`SharedF32`] under a name the rest of the app already uses.
pub type Threshold = SharedF32;

/// Live-editable ONNX/DSP tunables shared with the inference thread — the
/// former `SILENCE_RMS` / `NORM_MAX_GAIN` / `FRAME_OFF` constants, now cells the
/// Preferences ▸ Advanced tab writes and the detector reads each hop. Cloning
/// shares the underlying cells, so a UI edit takes effect on the next inference
/// pass with no restart. (The detection threshold stays a separate handle —
/// it's surfaced in the main config panel, not just Advanced.)
#[derive(Clone)]
pub struct InferenceTunables {
    pub silence_rms: SharedF32,
    pub norm_max_gain: SharedF32,
    pub frame_off: SharedF32,
}

impl InferenceTunables {
    pub fn new(silence_rms: f32, norm_max_gain: f32, frame_off: f32) -> Self {
        InferenceTunables {
            silence_rms: SharedF32::new(silence_rms),
            norm_max_gain: SharedF32::new(norm_max_gain),
            frame_off: SharedF32::new(frame_off),
        }
    }
}

/// Human-readable status lines shown in the UI status bar.
#[derive(Clone, Default)]
pub struct EngineStatus {
    pub device: String,
    pub model: String,
}

/// A running microphone backend. Dropping/stopping it tears down capture and
/// inference cleanly: stopping the capture thread drops the cpal stream and the
/// raw-audio sender, which ends the inference thread — and the inference thread
/// releases any still-held notes (emits Note Offs) as it exits.
pub struct AudioHandle {
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
}

impl AudioHandle {
    /// Signal the capture thread to stop and wait for it to wind down. The
    /// inference thread then exits on its own once the audio channel closes.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// Start microphone capture + inference, feeding detected notes into the
/// provided `note_tx`. Returns a handle that the caller stops when switching
/// away (e.g. a MIDI device was plugged in). Never blocks the GUI: all device
/// and model setup happens on background threads; progress is reported via
/// `status`.
///
/// The channel, threshold and status are owned by [`crate::input`] so that the
/// MIDI backend and this audio backend can share one unified note channel.
pub fn start_into(
    note_tx: Sender<NoteMsg>,
    threshold: Threshold,
    tunables: InferenceTunables,
    status: Arc<Mutex<EngineStatus>>,
) -> AudioHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let st = Arc::clone(&status);
    let stop_for_thread = Arc::clone(&stop);
    let join = thread::Builder::new()
        .name("audio-capture".into())
        .spawn(move || {
            if let Err(e) = capture_loop(note_tx, threshold, tunables, st.clone(), stop_for_thread) {
                eprintln!("[audio] fatal: {e}");
                if let Ok(mut s) = st.lock() {
                    s.device = format!("Audio ERROR: {e}");
                }
            }
        })
        .expect("failed to spawn audio thread");

    AudioHandle {
        stop,
        join: Some(join),
    }
}

/// Start a **capture-only** microphone stream for the training-data harness:
/// the same device the inference fallback would use, but the samples are written
/// to disk via `recorder` instead of being transcribed. No model is loaded and
/// no notes are emitted, so this is cheap and runs *alongside* the live MIDI
/// backend (mic and MIDI are different devices — no conflict).
///
/// Returns an [`AudioHandle`]; `stop()` it to end capture. The recorder session
/// itself is begun/ended by the caller (the input supervisor).
pub fn start_record_capture(recorder: Recorder) -> AudioHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_thread = Arc::clone(&stop);
    let join = thread::Builder::new()
        .name("record-capture".into())
        .spawn(move || {
            if let Err(e) = record_capture_loop(recorder, stop_for_thread) {
                eprintln!("[record] capture fatal: {e}");
            }
        })
        .expect("failed to spawn record-capture thread");

    AudioHandle {
        stop,
        join: Some(join),
    }
}

fn record_capture_loop(
    recorder: Recorder,
    stop: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "windows")]
    let host = cpal::host_from_id(cpal::HostId::Wasapi)?;
    #[cfg(not(target_os = "windows"))]
    let host = cpal::default_host();

    let device = host
        .default_input_device()
        .ok_or("no default input device (microphone) found")?;
    let dev_name = device.name().unwrap_or_else(|_| "unknown".into());
    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    let sample_format = config.sample_format();

    // Announce the format so the writer can open the WAV and stamp the audio
    // stream's start on the shared clock. Sent before `play()`, so it is enqueued
    // ahead of any audio buffer from the callback.
    recorder.audio_format(sample_rate, channels as u16, dev_name);

    let err_fn = |e| eprintln!("[record] capture stream error: {e}");
    let cfg = config.config();
    let stream = match sample_format {
        SampleFormat::F32 => {
            let rec = recorder.clone();
            device.build_input_stream(
                &cfg,
                move |data: &[f32], _| rec.push_audio(downmix_mono(data, channels, |s| s)),
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let rec = recorder.clone();
            device.build_input_stream(
                &cfg,
                move |data: &[i16], _| {
                    rec.push_audio(downmix_mono(data, channels, |s| s as f32 / i16::MAX as f32))
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let rec = recorder.clone();
            device.build_input_stream(
                &cfg,
                move |data: &[u16], _| {
                    rec.push_audio(downmix_mono(data, channels, |s| {
                        (s as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0)
                    }))
                },
                err_fn,
                None,
            )?
        }
        other => return Err(format!("unsupported sample format: {other:?}").into()),
    };

    stream.play()?;
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(100));
    }
    Ok(())
}

fn capture_loop(
    note_tx: Sender<NoteMsg>,
    threshold: Threshold,
    tunables: InferenceTunables,
    status: Arc<Mutex<EngineStatus>>,
    stop: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    // WASAPI on Windows; default host elsewhere (keeps it compiling on dev macs).
    #[cfg(target_os = "windows")]
    let host = cpal::host_from_id(cpal::HostId::Wasapi)?;
    #[cfg(not(target_os = "windows"))]
    let host = cpal::default_host();

    let device = host
        .default_input_device()
        .ok_or("no default input device (microphone) found")?;
    let dev_name = device.name().unwrap_or_else(|_| "unknown".into());

    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    let sample_format = config.sample_format();

    if let Ok(mut s) = status.lock() {
        s.device = format!("Mic: {dev_name}  ({sample_rate} Hz, {channels} ch, {sample_format:?})");
    }

    // Raw mono samples flow from the cpal callback to the inference thread.
    let (raw_tx, raw_rx) = mpsc::channel::<Vec<f32>>();

    // ---- Inference thread (loads model, resamples, runs ONNX, emits notes). ----
    {
        let note_tx = note_tx.clone();
        let threshold = threshold.clone();
        let tunables = tunables.clone();
        let status = Arc::clone(&status);
        thread::Builder::new()
            .name("inference".into())
            .spawn(move || {
                inference::run(raw_rx, note_tx, threshold, tunables, sample_rate, status);
            })
            .expect("failed to spawn inference thread");
    }

    // ---- Build the input stream. The callback only down-mixes + forwards. ----
    let err_fn = |e| eprintln!("[audio] stream error: {e}");
    let cfg = config.config();

    let stream = match sample_format {
        SampleFormat::F32 => {
            let tx = raw_tx.clone();
            device.build_input_stream(
                &cfg,
                move |data: &[f32], _| forward_mono(data, channels, &tx, |s| s),
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let tx = raw_tx.clone();
            device.build_input_stream(
                &cfg,
                move |data: &[i16], _| {
                    forward_mono(data, channels, &tx, |s| s as f32 / i16::MAX as f32)
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let tx = raw_tx.clone();
            device.build_input_stream(
                &cfg,
                move |data: &[u16], _| {
                    forward_mono(data, channels, &tx, |s| {
                        (s as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0)
                    })
                },
                err_fn,
                None,
            )?
        }
        other => return Err(format!("unsupported sample format: {other:?}").into()),
    };

    stream.play()?;

    // Keep the stream (and capture) alive until asked to stop. On stop we fall
    // out of the function: `stream` drops (cpal stops capturing) and the raw
    // sender(s) drop, which ends the inference thread — and that thread releases
    // any still-held notes as it exits, so nothing stays stuck on when we switch
    // to the MIDI backend.
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(100));
    }
    Ok(())
}

/// Down-mix an interleaved frame to a mono `Vec<f32>` in [-1, 1]. `convert` turns
/// one raw sample into f32. Shared by the inference and recording capture paths.
fn downmix_mono<T: Copy>(data: &[T], channels: usize, convert: impl Fn(T) -> f32) -> Vec<f32> {
    if channels == 0 {
        return Vec::new();
    }
    let frames = data.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut acc = 0.0f32;
        for c in 0..channels {
            acc += convert(data[f * channels + c]);
        }
        mono.push(acc / channels as f32);
    }
    mono
}

/// Down-mix an interleaved frame to mono and forward it to the inference thread.
fn forward_mono<T: Copy>(
    data: &[T],
    channels: usize,
    tx: &Sender<Vec<f32>>,
    convert: impl Fn(T) -> f32,
) {
    let _ = tx.send(downmix_mono(data, channels, convert)); // drop if inference gone
}
