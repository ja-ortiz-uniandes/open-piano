//! A tiny built-in polyphonic synth so the **on-screen keyboard makes sound**.
//!
//! open-piano is otherwise a pure *visualizer* — notes come from a real MIDI
//! piano or the microphone, both of which already produce their own sound. The
//! only notes with no acoustic source are the ones you click on screen and the
//! ones the peer plays, so those are what we synthesize here (see `main.rs`).
//!
//! Design mirrors the capture side (`audio.rs`): cpal owns a realtime callback
//! that must stay cheap and lock-free, so the GUI thread only *sends commands*
//! (note on/off) down an mpsc channel and the audio callback drains them and
//! renders. The cpal `Stream` is `!Send` on WASAPI, so it lives for the whole
//! program on its own thread, which just parks until `stop` is set.
//!
//! The voice is a small additive tone — four harmonics under a fast-attack,
//! exponentially-decaying envelope — which reads as a plucky, piano-ish keyboard
//! without any sample assets to bundle.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;

/// Maximum simultaneous notes. A two-hand chord plus the peer is well under this;
/// extra note-ons steal the quietest voice.
const MAX_VOICES: usize = 24;

/// Master output gain. Low enough that a fistful of voices won't clip after the
/// per-frame soft limiter.
const MASTER_GAIN: f32 = 0.22;

/// Relative amplitudes of harmonics 1..4. Falling off quickly keeps the tone
/// warm rather than buzzy.
const HARMONICS: [f32; 4] = [1.0, 0.45, 0.22, 0.12];

/// Which source a synthesized note belongs to. Notes carry their channel so
/// the audio callback can scale (or silence) each source independently — the
/// keys *you* click on screen, the ones the *peer* plays, and the notes a
/// loaded MIDI file plays back (see playback.rs).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// Notes from this player's on-screen (mouse) keyboard.
    Local = 0,
    /// Notes received from the remote peer.
    Peer = 1,
    /// Notes auto-played from a loaded MIDI file. NOTE: main.rs's mic-echo
    /// bookkeeping (`echo_held`) is sized for the two *live* channels only —
    /// playback must drive `Synth::note_on/off` directly, never through
    /// `synth_note_on`/`synth_note_off`.
    Playback = 2,
}

/// Number of distinct channels; sizes the per-channel gain table.
const CHANNELS: usize = 3;

/// A command from the GUI thread to the audio callback.
enum Cmd {
    On(u8, Channel),
    Off(u8, Channel),
    /// Set a channel's output gain (0.0 = silent). Applied per sample, so it
    /// fades notes already sounding, not just future ones.
    Gain(Channel, f32),
}

/// Where a voice is in its amplitude envelope.
#[derive(Clone, Copy, PartialEq)]
enum Stage {
    Attack,  // ramping up to full
    Decay,   // ringing down while held (piano-like)
    Release, // faster fade after note-off
    Idle,    // free slot
}

/// One synthesizer voice: a single note's phase and envelope state.
#[derive(Clone, Copy)]
struct Voice {
    midi: u8,
    channel: Channel, // which source plays this note (for per-channel gain)
    phase: f32,       // fundamental phase, 0..1
    inc: f32,         // phase advance per sample = freq / sample_rate
    env: f32,         // current envelope amplitude, 0..1
    stage: Stage,
}

impl Voice {
    const IDLE: Voice = Voice {
        midi: 0,
        channel: Channel::Local,
        phase: 0.0,
        inc: 0.0,
        env: 0.0,
        stage: Stage::Idle,
    };
}

/// Per-sample envelope coefficients, precomputed from the device sample rate.
struct EnvParams {
    attack_inc: f32,   // added per sample during attack
    decay_mult: f32,   // multiplied per sample while held (slow ring-down)
    release_mult: f32, // multiplied per sample after note-off (fast fade)
}

impl EnvParams {
    fn new(sample_rate: f32) -> Self {
        // Time constants in seconds → per-sample multipliers via exp(-1/(τ·sr)).
        EnvParams {
            attack_inc: 1.0 / (0.004 * sample_rate),
            decay_mult: (-1.0 / (2.0 * sample_rate)).exp(),
            release_mult: (-1.0 / (0.18 * sample_rate)).exp(),
        }
    }
}

/// The full set of voices, owned by the audio callback.
struct SynthState {
    voices: [Voice; MAX_VOICES],
    sample_rate: f32,
    /// Per-channel output gain (indexed by `Channel as usize`). 0.0 mutes.
    gain: [f32; CHANNELS],
}

impl SynthState {
    fn new(sample_rate: f32) -> Self {
        SynthState {
            voices: [Voice::IDLE; MAX_VOICES],
            sample_rate,
            gain: [1.0; CHANNELS],
        }
    }

    /// Start (or retrigger) the given note on a channel: reuse a voice already on
    /// this note+channel, else take a free slot, else steal the quietest voice.
    fn note_on(&mut self, midi: u8, channel: Channel) {
        let freq = 440.0 * 2f32.powf((midi as f32 - 69.0) / 12.0);
        let inc = freq / self.sample_rate;

        let slot = self
            .voices
            .iter()
            .position(|v| v.stage != Stage::Idle && v.midi == midi && v.channel == channel)
            .or_else(|| self.voices.iter().position(|v| v.stage == Stage::Idle))
            .unwrap_or_else(|| {
                // Steal the quietest voice.
                let mut quietest = 0;
                for i in 1..self.voices.len() {
                    if self.voices[i].env < self.voices[quietest].env {
                        quietest = i;
                    }
                }
                quietest
            });

        self.voices[slot] = Voice {
            midi,
            channel,
            phase: 0.0,
            inc,
            env: 0.0,
            stage: Stage::Attack,
        };
    }

    /// Release every still-sounding voice on the given note+channel. Scoping by
    /// channel means a note-off from one player doesn't cut the other player's
    /// voice when both happen to be holding the same key.
    fn note_off(&mut self, midi: u8, channel: Channel) {
        for v in self.voices.iter_mut() {
            if v.midi == midi
                && v.channel == channel
                && matches!(v.stage, Stage::Attack | Stage::Decay)
            {
                v.stage = Stage::Release;
            }
        }
    }

    /// Advance all voices one sample and return the mixed mono output.
    fn next_sample(&mut self, env: &EnvParams) -> f32 {
        let amp_sum: f32 = HARMONICS.iter().sum();
        let mut mix = 0.0;
        for v in self.voices.iter_mut() {
            match v.stage {
                Stage::Idle => continue,
                Stage::Attack => {
                    v.env += env.attack_inc;
                    if v.env >= 1.0 {
                        v.env = 1.0;
                        v.stage = Stage::Decay;
                    }
                }
                Stage::Decay => v.env *= env.decay_mult,
                Stage::Release => {
                    v.env *= env.release_mult;
                    if v.env < 0.0008 {
                        v.stage = Stage::Idle;
                        continue;
                    }
                }
            }

            // Additive waveform: sum the harmonics, normalized to unit peak-ish,
            // then scaled by this channel's gain (mute = 0).
            let mut s = 0.0;
            for (k, a) in HARMONICS.iter().enumerate() {
                let h = (k + 1) as f32;
                s += a * (std::f32::consts::TAU * v.phase * h).sin();
            }
            mix += (s / amp_sum) * v.env * self.gain[v.channel as usize];

            v.phase += v.inc;
            if v.phase >= 1.0 {
                v.phase -= 1.0;
            }
        }
        // Soft clamp so dense chords can't blow past full-scale.
        (mix * MASTER_GAIN).clamp(-1.0, 1.0)
    }
}

/// A running synth. Drop or `stop()` to tear down the output stream. Cloning the
/// command sender is cheap, so triggering from the GUI thread never blocks.
pub struct Synth {
    cmd_tx: Sender<Cmd>,
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
}

impl Synth {
    /// Open the default output device and start the audio thread. On failure
    /// (no output device, unsupported format) it logs and returns a handle whose
    /// `note_on`/`note_off` are silently no-ops — the app still runs, just mute.
    pub fn start() -> Synth {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);

        let join = thread::Builder::new()
            .name("synth-output".into())
            .spawn(move || {
                if let Err(e) = output_loop(cmd_rx, stop_for_thread) {
                    eprintln!("[synth] disabled: {e}");
                }
            })
            .expect("failed to spawn synth thread");

        Synth {
            cmd_tx,
            stop,
            join: Some(join),
        }
    }

    /// A synth whose commands go nowhere — for unit tests that need a
    /// `&Synth` without opening a real audio device.
    #[cfg(test)]
    pub fn disconnected() -> Synth {
        let (cmd_tx, _) = mpsc::channel::<Cmd>();
        Synth { cmd_tx, stop: Arc::new(AtomicBool::new(false)), join: None }
    }

    /// Begin sounding `midi` (MIDI note number) on `channel`.
    pub fn note_on(&self, midi: u8, channel: Channel) {
        let _ = self.cmd_tx.send(Cmd::On(midi, channel));
    }

    /// Stop sounding `midi` on `channel`.
    pub fn note_off(&self, midi: u8, channel: Channel) {
        let _ = self.cmd_tx.send(Cmd::Off(midi, channel));
    }

    /// Set a channel's output gain (0.0 = silent, 1.0 = full). Takes effect on
    /// the next audio block, fading notes already sounding on that channel.
    pub fn set_gain(&self, channel: Channel, gain: f32) {
        let _ = self.cmd_tx.send(Cmd::Gain(channel, gain));
    }

    /// Stop the audio thread and wait for it to wind down.
    #[allow(dead_code)]
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// Build the output stream and keep it alive until `stop` is set. The cpal
/// `Stream` is `!Send`, so it must be created and held on this one thread.
fn output_loop(
    cmd_rx: Receiver<Cmd>,
    stop: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    // WASAPI on Windows; default host elsewhere (keeps it compiling on dev macs).
    #[cfg(target_os = "windows")]
    let host = cpal::host_from_id(cpal::HostId::Wasapi)?;
    #[cfg(not(target_os = "windows"))]
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .ok_or("no default output device (speakers) found")?;
    let config = device.default_output_config()?;
    let sample_rate = config.sample_rate().0 as f32;
    let channels = config.channels() as usize;
    let sample_format = config.sample_format();
    let cfg = config.config();

    let mut state = SynthState::new(sample_rate);
    let env = EnvParams::new(sample_rate);
    let err_fn = |e| eprintln!("[synth] output stream error: {e}");

    // Drain queued note on/off commands, then render the block. Shared by every
    // sample format; the closures below only differ in how they write a sample.
    macro_rules! fill {
        ($data:expr, $channels:expr, $write:expr) => {{
            while let Ok(cmd) = cmd_rx.try_recv() {
                match cmd {
                    Cmd::On(m, ch) => state.note_on(m, ch),
                    Cmd::Off(m, ch) => state.note_off(m, ch),
                    Cmd::Gain(ch, g) => state.gain[ch as usize] = g,
                }
            }
            for frame in $data.chunks_mut($channels) {
                let s = state.next_sample(&env);
                for slot in frame.iter_mut() {
                    *slot = $write(s);
                }
            }
        }};
    }

    let stream = match sample_format {
        SampleFormat::F32 => device.build_output_stream(
            &cfg,
            move |data: &mut [f32], _| fill!(data, channels, |s: f32| s),
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_output_stream(
            &cfg,
            move |data: &mut [i16], _| fill!(data, channels, |s: f32| (s * i16::MAX as f32) as i16),
            err_fn,
            None,
        )?,
        SampleFormat::U16 => device.build_output_stream(
            &cfg,
            move |data: &mut [u16], _| {
                fill!(data, channels, |s: f32| {
                    ((s * 0.5 + 0.5) * u16::MAX as f32) as u16
                })
            },
            err_fn,
            None,
        )?,
        other => return Err(format!("unsupported output sample format: {other:?}").into()),
    };

    stream.play()?;
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(100));
    }
    Ok(())
}
