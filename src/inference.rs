//! ONNX-based polyphonic note transcription.
//!
//! This replaces the previous hand-written FFT/DSP detector. Raw microphone
//! audio (at the device's native rate) is streamed in from [`crate::audio`],
//! linearly resampled to the model's expected rate, buffered into the fixed
//! input window, and run through a Spotify **Basic Pitch** ONNX model on a
//! dedicated background thread. The note posteriorgram is thresholded into
//! active MIDI notes which are pushed to the egui UI over the same
//! `std::sync::mpsc` channel the rest of the app already uses.
//!
//! Basic Pitch tensor contract (ICASSP-2022 model):
//!   * **input**:  `[1, AUDIO_N_SAMPLES, 1]` mono float samples @ 22050 Hz,
//!     where `AUDIO_N_SAMPLES = 22050*2 - 256 = 43844` (a ~2 s window).
//!   * **outputs**: three tensors; we use the *note* posteriorgram of shape
//!     `[1, n_frames (~172), 88]` — frame-level probabilities per piano key
//!     (MIDI 21..108). (The other outputs are onset and contour.)
//!
//! All ONNX work happens here, off the GUI and audio-callback threads, so the
//! UI stays responsive regardless of inference latency.

use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::audio::{EngineStatus, InferenceTunables, Threshold};
use crate::note::{NoteMsg, KEY_COUNT, MIDI_LOW};

/// Model input sample rate (Hz).
const TARGET_SR: u32 = 22050;

/// Fixed model input length in samples: `22050*2 - 256`.
const N_SAMPLES: usize = 43844;

/// Re-run inference once this many fresh resampled samples have arrived
/// (~0.05 s). Lower = lower latency and a higher refresh rate; the loop only
/// ever analyses the freshest audio, so if inference can't keep up it just runs
/// as fast as it can rather than backlogging. All the hop-based hysteresis
/// counters below are in *hops*, so a shorter hop shrinks their latency too
/// without weakening the flicker/sustain protection.
const HOP_TARGET: usize = 1102;

/// Number of trailing posteriorgram frames to average for the *note* (sustain)
/// grid when deciding which notes are *currently* sounding (the newest audio is
/// at the end of the window). Fewer = less smoothing lag on release (slightly
/// more jitter).
const FRAMES_TAIL: usize = 5;

/// Number of trailing frames to take the *max* over for the *onset* grid. Wider
/// than `FRAMES_TAIL` on purpose: Basic Pitch's onset spike for a fresh attack
/// lands a few frames inside the window edge (the last 1–3 output frames are
/// immature — incomplete future context — so they read low). A 5-frame tail
/// missed the real spike (measured onset 0.28 at the tail vs a 0.92 grid-wide
/// peak). ~12 frames (~140 ms at ~11.6 ms/frame) reliably catches the matured
/// attack peak, so a real onset finally clears the noise floor. Too wide and a
/// stale onset can re-trigger after release; 10–14 is the sweet spot.
const ONSET_TAIL: usize = 12;

/// Consecutive hops a note's frame probability must stay *below* `FRAME_OFF`
/// before we emit Note Off. At ~0.05 s/hop this (~0.1 s) rides over momentary
/// dips in a decaying note without clipping it short.
const RELEASE_HOPS: u8 = 2;

// NOTE: the former `FRAME_OFF`, `SILENCE_RMS` and `NORM_MAX_GAIN` constants are
// now live-editable [`InferenceTunables`] (Preferences ▸ Advanced), read from
// shared cells each hop instead of baked in. Their tuning guidance moved to the
// read sites in `run`/`infer` and to `prefs.rs`'s defaults:
//   * frame_off   — sustain/linger knob: lower = notes hold longer while
//                   decaying; higher = snappier release. Default 0.10.
//   * silence_rms — raw RMS below which a window is silence (inference skipped).
//                   Set just above the mic idle floor (~0.0011–0.0017); quietest
//                   real notes ≈ 0.002–0.008. Default 0.002.
//   * norm_max_gain — hard cap on the normalization gain; too high blows idle
//                   hiss up into phantom onsets. Default 10.0.

/// Consecutive hops the note/frame probability must hold above the trigger
/// threshold before Note On. This debounce is what suppresses near-threshold
/// flicker (e.g. an octave/harmonic ghost wobbling around the threshold): a real
/// note holds for several hops, a ghost doesn't. A clear onset spike bypasses it
/// (see `ONSET_TRIG`) so genuine attacks stay instant. Lower = snappier but more
/// flicker-prone; at ~0.05–0.085 s/hop, 2 hops ≈ 0.1–0.17 s.
const ATTACK_HOPS: u8 = 2;

/// Onset-grid level that, together with frame support, triggers a note
/// *instantly* (bypassing the `ATTACK_HOPS` debounce). The onset grid is too
/// noisy on a mic to gate notes by itself (measured real attacks 0.2–0.43 vs a
/// noise floor of ~0.2), so it only *accelerates* a note the frame grid already
/// agrees is there — a lone onset spike on noise, which has no frame support,
/// can't fire. Lower = more attacks go instant (less flicker protection).
const ONSET_TRIG: f32 = 0.3;

/// Normalization target: the window is scaled so its RMS approaches this before
/// inference, so you don't have to play loud. Higher = louder model input.
const NORM_TARGET_RMS: f32 = 0.1;

/// Detection thread entry point. Loads the model, then loops forever consuming
/// resampled audio and emitting note transitions. Returns (ending the thread)
/// only if the model fails to load or the audio channel closes.
pub fn run(
    raw_rx: Receiver<Vec<f32>>,
    note_tx: Sender<NoteMsg>,
    threshold: Threshold,
    tunables: InferenceTunables,
    device_sr: u32,
    status: Arc<Mutex<EngineStatus>>,
) {
    // ---- Load the ONNX model on this background thread. ----
    let mut session = match load_model() {
        Ok(s) => {
            set_model_status(&status, "Model: loaded (built-in)".to_string());
            s
        }
        Err(e) => {
            // The model is embedded, so this is an ONNX Runtime problem (the
            // extracted DLL failed to load), not a missing file.
            set_model_status(&status, format!("Model load FAILED: {e}"));
            // Drain audio so the sender never blocks, but emit nothing.
            while raw_rx.recv().is_ok() {}
            return;
        }
    };

    let mut resampler = Resampler::new(device_sr, TARGET_SR);
    let mut window: Vec<f32> = Vec::with_capacity(N_SAMPLES + 16384);
    let mut scratch: Vec<f32> = Vec::new();
    let mut new_count: usize = 0;

    // Per-MIDI-note emission state + hysteresis counters.
    let mut on = [false; 128];
    let mut present = [0u8; 128]; // consecutive hops above the on-threshold
    let mut absent = [0u8; 128]; // consecutive hops below the off-threshold

    loop {
        // Block for at least one chunk, then drain everything queued so we
        // always analyse the freshest audio.
        let first = match raw_rx.recv() {
            Ok(c) => c,
            Err(_) => break, // capture gone
        };
        scratch.clear();
        resampler.process(&first, &mut scratch);
        while let Ok(c) = raw_rx.try_recv() {
            resampler.process(&c, &mut scratch);
        }

        new_count += scratch.len();
        window.extend_from_slice(&scratch);
        if window.len() > N_SAMPLES {
            let drop = window.len() - N_SAMPLES;
            window.drain(..drop);
        }

        if new_count < HOP_TARGET {
            continue; // not enough new audio yet
        }
        new_count = 0;

        // Assemble exactly N_SAMPLES, left-padding with zeros during warm-up.
        let mut input = vec![0.0f32; N_SAMPLES];
        let take = window.len().min(N_SAMPLES);
        input[N_SAMPLES - take..].copy_from_slice(&window[window.len() - take..]);

        // Read the live tunables once per hop (see InferenceTunables): silence
        // gate, release threshold, and the normalization gain cap.
        let silence_rms = tunables.silence_rms.get();
        let frame_off = tunables.frame_off.get();
        let norm_max_gain = tunables.norm_max_gain.get();

        // Per-key note (frame) and onset probabilities (empty == silence).
        let Frames { note, onset } = if rms(&input) < silence_rms {
            Frames::default()
        } else {
            infer(&mut session, &input, norm_max_gain)
        };
        let onset_gating = !onset.is_empty();

        // Trigger threshold (slider): the note/frame probability needed to *start*
        // a note. The frame grid is the primary gate because on a mic it separates
        // real notes from noise far better than the onset grid (see `ONSET_TRIG`).
        let trig_thresh = threshold.get();
        for k in 0..KEY_COUNT {
            let m = (MIDI_LOW as usize) + k;
            let note_p = note.get(k).copied().unwrap_or(0.0);
            if on[m] {
                // Sustain on the frame grid: release only once it stays below
                // frame_off for RELEASE_HOPS (keeps decaying notes alive).
                if note_p < frame_off {
                    absent[m] = absent[m].saturating_add(1);
                    if absent[m] >= RELEASE_HOPS {
                        on[m] = false;
                        present[m] = 0;
                        let _ = note_tx.send(NoteMsg::Off(m as u8));
                    }
                } else {
                    absent[m] = 0;
                }
            } else {
                // Start a note when the frame grid clears the threshold. Debounce
                // across ATTACK_HOPS to ride over near-threshold flicker (octave/
                // harmonic ghosts), but let a clear onset spike at this key bypass
                // the debounce for an instant attack. The onset bypass still
                // requires frame support, so a lone onset spike on noise (no frame
                // support) can't fire a phantom note.
                let onset_p = onset.get(k).copied().unwrap_or(0.0);
                let frame_hit = note_p >= trig_thresh;
                let triggered = if frame_hit && onset_gating && onset_p >= ONSET_TRIG {
                    true // genuine attack: both grids agree — fire immediately
                } else if frame_hit {
                    present[m] = present[m].saturating_add(1);
                    present[m] >= ATTACK_HOPS
                } else {
                    present[m] = 0;
                    false
                };
                if triggered {
                    on[m] = true;
                    absent[m] = 0;
                    present[m] = 0;
                    let _ = note_tx.send(NoteMsg::On(m as u8));
                }
            }
        }
    }

    // Release anything still held on shutdown.
    for m in 0u8..128 {
        if on[m as usize] {
            let _ = note_tx.send(NoteMsg::Off(m));
        }
    }
}

fn load_model() -> ort::Result<Session> {
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(2)?;
    // The model ships inside the exe (see bundle.rs); nothing to find on disk.
    builder.commit_from_memory(crate::bundle::MODEL)
}

/// Per-key probabilities for one inference pass (index 0 == MIDI_LOW). `note`
/// is the sustained frame grid (tail-averaged); `onset` is the attack grid
/// (tail-*max*, since onsets are brief spikes). Either is empty when
/// unavailable; thresholding/hysteresis is the caller's job.
#[derive(Default)]
struct Frames {
    note: Vec<f32>,
    onset: Vec<f32>,
}

/// Run one inference pass and return the note (frame) and onset grids per key.
fn infer(session: &mut Session, audio: &[f32], norm_max_gain: f32) -> Frames {
    // Level-normalize so quiet mic input still drives the model (see fn below).
    let raw_rms = rms(audio);
    let normalized = normalize_for_model(audio, norm_max_gain);
    let tensor = match Tensor::from_array((vec![1_i64, normalized.len() as i64, 1_i64], normalized)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[inference] tensor build error: {e}");
            return Frames::default();
        }
    };

    let outputs = match session.run(ort::inputs![tensor]) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[inference] run error: {e}");
            return Frames::default();
        }
    };

    // Collect every 3-D output whose last dim is 88 (note + onset posteriorgrams;
    // the contour grid is 264-wide and excluded), keeping each output's name.
    let mut cands: Vec<Cand> = Vec::new();
    for (name, value) in &outputs {
        if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
            if shape.len() == 3 && *shape.last().unwrap() == 88 {
                cands.push(Cand {
                    name: name.to_string(),
                    frames: shape[1] as usize,
                    dim: shape[2] as usize,
                    data: data.to_vec(),
                });
            }
        }
    }
    if cands.is_empty() {
        return Frames::default();
    }

    // Identify the grids by NAME. Basic Pitch's ONNX exposes three generically
    // named outputs; the authoritative mapping (from basic_pitch/inference.py) is
    // `StatefulPartitionedCall:1` = note (frame), `:2` = onset, `:0` = the
    // 264-wide contour (already filtered out). Name matching is deterministic,
    // unlike the old density heuristic: the two grids share a ~0.10 diffuse floor
    // that dwarfs the sparse note content, so their per-pass means sit nearly
    // equal and *cross* during loud playing — which flipped note<->onset mid-
    // performance, breaking both triggering and sustain. Density survives only as
    // a fallback if a differently-exported model ever lacks these names.
    let note_idx = cands
        .iter()
        .position(|c| c.name.ends_with(":1"))
        .unwrap_or_else(|| densest(&cands, &[]));
    let onset_idx = cands
        .iter()
        .position(|c| c.name.ends_with(":2"))
        .or_else(|| (cands.len() >= 2).then(|| densest(&cands, &[note_idx])));
    let frames = Frames {
        note: tail_reduce(&cands[note_idx], Reduce::Mean, FRAMES_TAIL),
        onset: onset_idx
            .map(|i| tail_reduce(&cands[i], Reduce::Max, ONSET_TAIL))
            .unwrap_or_default(),
    };
    debug_log_outputs(&cands, note_idx, onset_idx, raw_rms, norm_max_gain, &frames);
    frames
}

/// (index, value) of the largest element, or (0, 0.0) for an empty slice.
fn argmax(v: &[f32]) -> (usize, f32) {
    let mut bi = 0;
    let mut bv = 0.0f32;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    (bi, bv)
}

/// One 88-wide model output for a single pass: its graph name (e.g.
/// `StatefulPartitionedCall:1`), frame count, key dimension, and the flat
/// `[frames * dim]` row-major probability data.
struct Cand {
    name: String,
    frames: usize,
    dim: usize,
    data: Vec<f32>,
}

/// Index of the highest-mean candidate, ignoring any in `exclude`. Fallback grid
/// identification when output names are missing (see `infer`).
fn densest(cands: &[Cand], exclude: &[usize]) -> usize {
    let mut best = 0;
    let mut best_mean = f32::MIN;
    for (i, c) in cands.iter().enumerate() {
        if exclude.contains(&i) {
            continue;
        }
        let mean = c.data.iter().sum::<f32>() / c.data.len().max(1) as f32;
        if mean > best_mean {
            best_mean = mean;
            best = i;
        }
    }
    best
}

enum Reduce {
    Mean,
    Max,
}

/// Reduce the trailing `tail` frames of a candidate grid into one probability
/// per key (the newest audio is at the window's end). Mean for the sustained
/// note grid (short tail); max for the spiky onset grid (wider tail, see
/// `ONSET_TAIL`).
fn tail_reduce(grid: &Cand, how: Reduce, tail: usize) -> Vec<f32> {
    let (frames, dim, data) = (grid.frames, grid.dim, &grid.data);
    let mut out = vec![0.0f32; KEY_COUNT];
    if frames == 0 {
        return out;
    }
    let tail = tail.min(frames);
    let start = frames - tail;
    for (k, slot) in out.iter_mut().enumerate().take(dim.min(KEY_COUNT)) {
        *slot = match how {
            Reduce::Mean => {
                let mut acc = 0.0f32;
                for f in start..frames {
                    acc += data[f * dim + k];
                }
                acc / tail as f32
            }
            Reduce::Max => {
                let mut mx = 0.0f32;
                for f in start..frames {
                    mx = mx.max(data[f * dim + k]);
                }
                mx
            }
        };
    }
    out
}

fn rms(buf: &[f32]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let s: f32 = buf.iter().map(|x| x * x).sum();
    (s / buf.len() as f32).sqrt()
}

/// Scale the window up toward a target RMS so quiet microphone input still
/// produces strong posteriorgrams (you shouldn't have to play loud). The gain
/// is capped so near-silence isn't blown up into noise — and the caller already
/// skips windows below `SILENCE_RMS`. Output is clamped to [-1, 1] to avoid
/// out-of-range distortion.
fn normalize_for_model(audio: &[f32], norm_max_gain: f32) -> Vec<f32> {
    let gain = norm_gain(rms(audio), norm_max_gain);
    audio.iter().map(|x| (x * gain).clamp(-1.0, 1.0)).collect()
}

/// Gain `normalize_for_model` applies for a given raw RMS (shared so the debug
/// line can report the exact gain without recomputing it differently).
fn norm_gain(raw_rms: f32, norm_max_gain: f32) -> f32 {
    if raw_rms > 1.0e-6 {
        (NORM_TARGET_RMS / raw_rms).min(norm_max_gain)
    } else {
        1.0
    }
}

/// Log each 88-wide output's name, shape, mean, and max activation plus the role
/// (note/onset) it was assigned this pass, prefixed with the raw mic RMS and the
/// applied normalization gain. Throttled to roughly once per 2 s (every 40th
/// inference at ~0.05 s/hop) so it stays readable while you play but reflects
/// *live* audio. Watch while playing: the role assignment should now stay fixed
/// (name-based), `raw=` tells you the true mic level (for setting SILENCE_RMS and
/// the gain cap), and the `onset` grid's max should spike at attacks.
fn debug_log_outputs(
    cands: &[Cand],
    note_idx: usize,
    onset_idx: Option<usize>,
    raw_rms: f32,
    norm_max_gain: f32,
    frames: &Frames,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    if COUNT.fetch_add(1, Ordering::Relaxed) % 40 != 0 {
        return;
    }
    let mut line = format!("raw={raw_rms:.5} gain={:.1}", norm_gain(raw_rms, norm_max_gain));
    for (i, c) in cands.iter().enumerate() {
        let mean = c.data.iter().sum::<f32>() / c.data.len().max(1) as f32;
        let max = c.data.iter().copied().fold(0.0f32, f32::max);
        let role = if i == note_idx {
            "note "
        } else if Some(i) == onset_idx {
            "onset"
        } else {
            "?    "
        };
        line.push_str(&format!(
            " | {} {role} [{}x{}] mean={mean:.4} max={max:.3}",
            c.name, c.frames, c.dim
        ));
    }
    // Hottest per-key value of the tail-reduced grids actually used for
    // detection (MIDI note that's loudest right now). For a sustained held note
    // the note-grid hot key should stay high throughout, while the onset-grid hot
    // value should fall back after the attack — the decisive note-vs-onset check.
    let (nk, nv) = argmax(&frames.note);
    let (ok, ov) = argmax(&frames.onset);
    line.push_str(&format!(
        "  ||  hot note=MIDI{} {nv:.3}  onset=MIDI{} {ov:.3}",
        MIDI_LOW as usize + nk,
        MIDI_LOW as usize + ok
    ));
    eprintln!("[inference] {line}");
}

/// Simple stateful linear-interpolation resampler (device rate -> model rate).
/// Carries one sample of history across blocks for continuity. Quality is
/// sufficient for note detection and adds no extra crate dependency.
struct Resampler {
    step: f32, // input samples advanced per output sample
    pos: f32,  // fractional read position within the current block
    last: f32, // final sample of the previous block (logical index -1)
}

impl Resampler {
    fn new(in_sr: u32, out_sr: u32) -> Self {
        Resampler {
            step: in_sr as f32 / out_sr as f32,
            pos: 0.0,
            last: 0.0,
        }
    }

    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let n = input.len();
        if n == 0 {
            return;
        }
        let sample = |idx: isize, last: f32| -> f32 {
            if idx < 0 {
                last
            } else if idx as usize >= n {
                input[n - 1]
            } else {
                input[idx as usize]
            }
        };
        while self.pos <= (n as f32 - 1.0) {
            let i0 = self.pos.floor() as isize;
            let frac = self.pos - i0 as f32;
            let a = sample(i0, self.last);
            let b = sample(i0 + 1, self.last);
            out.push(a + (b - a) * frac);
            self.pos += self.step;
        }
        self.pos -= n as f32;
        self.last = input[n - 1];
    }
}

fn set_model_status(status: &Arc<Mutex<EngineStatus>>, msg: String) {
    if let Ok(mut s) = status.lock() {
        s.model = msg;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end check of the self-contained-exe chain (`bundle.rs`): extract
    /// the embedded ONNX Runtime, point `ORT_DYLIB_PATH` at it, load it, and
    /// parse the embedded model into a Session. Loading ort on the test thread
    /// is fine — the loader-lock hazard is specific to a GUI main thread
    /// (see `main.rs`).
    #[test]
    fn embedded_model_and_runtime_load() {
        crate::bundle::prepare_ort_dylib();
        load_model().expect("embedded model should load via the extracted runtime");
    }
}
