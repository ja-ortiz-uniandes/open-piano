"""Offline alignment + verification for open-piano capture sessions.

A capture session (see ``src/record.rs``) is a folder containing:

* ``audio.wav``   — mono float32 microphone audio at the device's native rate.
* ``midi.jsonl``  — one MIDI event per line, each with a time ``t`` (seconds) on
                    a clock shared with the audio.
* ``meta.json``   — ``sample_rate``, ``audio_start_s`` (when the audio stream
                    started on that shared clock), etc.

This script answers the make-or-break question before you collect hours of data:
**are the MIDI labels and the audio actually lined up in time?**

It does two things:

1. **Visual check** — draws the audio spectrogram with every MIDI note overlaid
   as a horizontal segment at its fundamental frequency, from key-down to key-up.
   If capture is aligned, each segment's left edge sits right where that pitch's
   energy switches on. (Saved as ``alignment.png`` in the session folder.)

2. **Offset estimate** — detects audio onsets (spectral flux), matches each MIDI
   note-on to the nearest audio onset, and reports the median delta. That delta
   is the fixed capture latency (USB-MIDI + audio buffering); subtract it from the
   MIDI times in your training pipeline. Best measured on a calibration take of
   isolated staccato notes.

Usage::

    python verify_alignment.py recordings/session_1718900000
    python verify_alignment.py recordings/session_1718900000 --no-show

Dependencies: ``numpy``, ``scipy``, ``matplotlib`` (``pip install numpy scipy matplotlib``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.io import wavfile


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

@dataclass
class Note:
    """A single key press: onset/offset in audio-relative seconds + velocity."""

    pitch: int
    velocity: int
    onset_s: float
    offset_s: float  # may equal onset_s if the matching note-off is missing


@dataclass
class Session:
    """A loaded capture session, with all times in audio-relative seconds."""

    sample_rate: int
    audio: np.ndarray  # mono float32, shape (n_samples,)
    notes: list[Note]
    onset_times: np.ndarray  # MIDI note-on times (audio-relative seconds)
    pedal_events: list[tuple[float, int]]  # (time_s, value) for CC64 sustain
    meta: dict[str, object]

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate


def load_session(session_dir: Path) -> Session:
    """Load ``audio.wav`` + ``midi.jsonl`` + ``meta.json`` from a session folder.

    MIDI event times (on the shared clock) are converted to *audio-relative*
    seconds by subtracting ``audio_start_s``, so 0.0 is the first audio sample.
    """
    meta_path = session_dir / "meta.json"
    meta: dict[str, object] = json.loads(meta_path.read_text(encoding="utf-8"))
    audio_start_s = float(meta.get("audio_start_s", 0.0))  # type: ignore[arg-type]

    sr_file, audio = wavfile.read(session_dir / "audio.wav")
    sample_rate: int = int(sr_file)
    audio = _to_mono_float(audio)

    notes, onset_times, pedal_events = _parse_midi(
        session_dir / "midi.jsonl", audio_start_s
    )

    return Session(
        sample_rate=sample_rate,
        audio=audio,
        notes=notes,
        onset_times=np.asarray(onset_times, dtype=np.float64),
        pedal_events=pedal_events,
        meta=meta,
    )


def _to_mono_float(audio: np.ndarray) -> np.ndarray:
    """Coerce a wavfile read into mono float32 in roughly [-1, 1]."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        max_val = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_val
    return audio.astype(np.float32, copy=False)


def _parse_midi(
    midi_path: Path, audio_start_s: float
) -> tuple[list[Note], list[float], list[tuple[float, int]]]:
    """Parse ``midi.jsonl`` into completed notes, onset times, and pedal events."""
    notes: list[Note] = []
    onset_times: list[float] = []
    pedal_events: list[tuple[float, int]] = []
    # Open key presses, keyed by (pitch, channel), holding (onset_s, velocity).
    active: dict[tuple[int, int], tuple[float, int]] = {}

    if not midi_path.exists():
        return notes, onset_times, pedal_events

    for raw_line in midi_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        event: dict[str, object] = json.loads(line)
        t_audio = float(event["t"]) - audio_start_s  # type: ignore[arg-type]
        kind = str(event.get("type", ""))

        if kind == "note_on":
            pitch = int(event["note"])  # type: ignore[arg-type]
            channel = int(event.get("ch", 0))  # type: ignore[arg-type]
            velocity = int(event.get("vel", 0))  # type: ignore[arg-type]
            active[(pitch, channel)] = (t_audio, velocity)
            onset_times.append(t_audio)
        elif kind == "note_off":
            pitch = int(event["note"])  # type: ignore[arg-type]
            channel = int(event.get("ch", 0))  # type: ignore[arg-type]
            started = active.pop((pitch, channel), None)
            if started is not None:
                onset_s, velocity = started
                notes.append(Note(pitch, velocity, onset_s, t_audio))
        elif kind == "cc" and int(event.get("ctrl", -1)) == 64:  # type: ignore[arg-type]
            pedal_events.append((t_audio, int(event.get("val", 0))))  # type: ignore[arg-type]

    # Any notes still held at end-of-session: close them at their onset so they
    # still render (zero-length) rather than being dropped.
    for (pitch, _channel), (onset_s, velocity) in active.items():
        notes.append(Note(pitch, velocity, onset_s, onset_s))

    return notes, onset_times, pedal_events


# --------------------------------------------------------------------------- #
# Signal processing
# --------------------------------------------------------------------------- #

def stft_magnitude(
    audio: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain STFT magnitude. Returns (frame_times, freqs, mag[frames, bins])."""
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(audio) - n_fft) // hop
    mag = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start : start + n_fft] * window
        mag[i] = np.abs(rfft(frame))
    frame_times = (np.arange(n_frames) * hop + n_fft / 2) / sr
    freqs = rfftfreq(n_fft, d=1.0 / sr)
    return frame_times, freqs, mag


# Onset timing uses a *finer* STFT than the spectrogram display: a short window
# and small hop localize attacks in time (≈6 ms), which matters because the whole
# point is measuring a tens-of-ms offset. The coarse spectrogram STFT (above) is
# kept for its frequency resolution in the visual. Verified on a synthetic take
# with a known 25 ms offset: these params recover it to within ~1 ms.
ONSET_N_FFT: int = 512
ONSET_HOP: int = 128


def onset_envelope(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Spectral-flux onset strength from a fine STFT: summed positive frame-to-
    frame magnitude increase, normalized to [0, 1]. Returns (times, strength)."""
    frame_times, _freqs, mag = stft_magnitude(audio, sr, n_fft=ONSET_N_FFT, hop=ONSET_HOP)
    diff = np.diff(mag, axis=0)
    flux = np.maximum(diff, 0.0).sum(axis=1)
    if flux.max() > 0:
        flux = flux / flux.max()
    return frame_times[1:], flux


def pick_onsets(
    times: np.ndarray,
    strength: np.ndarray,
    threshold: float = 0.12,
    min_gap_s: float = 0.05,
) -> np.ndarray:
    """Pick local maxima of the onset envelope above ``threshold``, spaced at
    least ``min_gap_s`` apart. Returns the onset times."""
    peaks: list[float] = []
    last_t = -np.inf
    for i in range(1, len(strength) - 1):
        s = strength[i]
        if s < threshold:
            continue
        if s >= strength[i - 1] and s >= strength[i + 1] and times[i] - last_t >= min_gap_s:
            peaks.append(float(times[i]))
            last_t = times[i]
    return np.asarray(peaks, dtype=np.float64)


def estimate_offset(
    midi_onsets: np.ndarray, audio_onsets: np.ndarray, max_lag_s: float = 0.10
) -> tuple[float, np.ndarray]:
    """For each MIDI note-on, find the nearest audio onset within ``max_lag_s``
    and collect ``audio_time - midi_time``. Returns (median_delta, all_deltas).

    A positive median means audio onsets land *after* the MIDI timestamps — i.e.
    MIDI arrives early relative to sound — so subtract it from MIDI times. NaN if
    there aren't enough matches to be meaningful.
    """
    if len(midi_onsets) == 0 or len(audio_onsets) == 0:
        return float("nan"), np.asarray([], dtype=np.float64)
    audio_sorted = np.sort(audio_onsets)
    deltas: list[float] = []
    for m in midi_onsets:
        idx = int(np.searchsorted(audio_sorted, m))
        candidates = audio_sorted[max(0, idx - 1) : idx + 1]
        if len(candidates) == 0:
            continue
        nearest = candidates[np.argmin(np.abs(candidates - m))]
        delta = float(nearest - m)
        if abs(delta) <= max_lag_s:
            deltas.append(delta)
    if len(deltas) < 3:
        return float("nan"), np.asarray(deltas, dtype=np.float64)
    return float(np.median(deltas)), np.asarray(deltas, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def pitch_to_freq(pitch: int) -> float:
    """MIDI note number to fundamental frequency in Hz (A4=69=440 Hz)."""
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


def make_figure(
    session: Session,
    audio_onsets: np.ndarray,
    median_delta: float,
    deltas: np.ndarray,
) -> plt.Figure:
    """Build the alignment figure: spectrogram + MIDI overlay, onset envelope,
    and a histogram of the per-note audio-vs-MIDI deltas."""
    frame_times, freqs, mag = stft_magnitude(session.audio, session.sample_rate)
    env_times, env = onset_envelope(session.audio, session.sample_rate)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], width_ratios=[3, 1])
    ax_spec = fig.add_subplot(gs[0, :])
    ax_env = fig.add_subplot(gs[1, :], sharex=ax_spec)
    ax_hist = fig.add_subplot(gs[2, 1])
    ax_info = fig.add_subplot(gs[2, 0])

    # --- Spectrogram (log magnitude) with MIDI notes overlaid. ---
    db = 20.0 * np.log10(mag.T + 1e-6)
    ax_spec.pcolormesh(frame_times, freqs, db, shading="auto", cmap="magma")
    ax_spec.set_yscale("log")
    ax_spec.set_ylim(50, 5000)
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title(
        "Spectrogram with MIDI notes overlaid — each note's left edge should sit "
        "at the onset of that pitch's energy"
    )
    for note in session.notes:
        f = pitch_to_freq(note.pitch)
        shade = 0.4 + 0.6 * (note.velocity / 127.0)
        ax_spec.hlines(
            f,
            note.onset_s,
            max(note.offset_s, note.onset_s + 0.01),
            color=(0.2, 1.0, 0.4),
            alpha=shade,
            linewidth=2.0,
        )

    # --- Onset envelope with MIDI onsets (red) and detected audio onsets. ---
    ax_env.plot(env_times, env, color="steelblue", linewidth=0.8, label="onset strength")
    for t in session.onset_times:
        ax_env.axvline(t, color="red", alpha=0.5, linewidth=0.8)
    for t in audio_onsets:
        ax_env.axvline(t, color="green", alpha=0.4, linewidth=0.8, linestyle="--")
    ax_env.set_ylabel("onset")
    ax_env.set_xlabel("Time (s, audio-relative)")
    ax_env.legend(loc="upper right", fontsize=8)

    # --- Delta histogram. ---
    if len(deltas) > 0:
        ax_hist.hist(deltas * 1000.0, bins=20, color="slateblue")
        ax_hist.axvline(median_delta * 1000.0, color="red", linewidth=1.5)
        ax_hist.set_xlabel("audio − MIDI (ms)")
        ax_hist.set_title("per-note offset", fontsize=9)
    else:
        ax_hist.text(0.5, 0.5, "no matches", ha="center", va="center")
        ax_hist.set_xticks([])
        ax_hist.set_yticks([])

    # --- Text summary. ---
    ax_info.axis("off")
    delta_str = "n/a" if np.isnan(median_delta) else f"{median_delta * 1000.0:+.1f} ms"
    lines = [
        f"duration:        {session.duration_s:.1f} s",
        f"sample rate:     {session.sample_rate} Hz",
        f"MIDI notes:      {len(session.notes)}",
        f"MIDI note-ons:   {len(session.onset_times)}",
        f"pedal events:    {len(session.pedal_events)}",
        f"audio onsets:    {len(audio_onsets)}",
        f"matched deltas:  {len(deltas)}",
        f"median offset:   {delta_str}",
    ]
    if not np.isnan(median_delta):
        lines.append(f"→ subtract {median_delta * 1000.0:+.1f} ms from MIDI times")
    ax_info.text(
        0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=10
    )

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="path to a session_* folder")
    parser.add_argument(
        "--no-show", action="store_true", help="save the PNG but don't open a window"
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.12,
        help="audio onset peak-picking threshold (0..1); lower = more onsets",
    )
    args = parser.parse_args()

    session_dir: Path = args.session
    if not session_dir.is_dir():
        raise SystemExit(f"not a directory: {session_dir}")

    session = load_session(session_dir)
    print(
        f"Loaded {session_dir}: {session.duration_s:.1f}s audio, "
        f"{len(session.notes)} notes, {len(session.onset_times)} note-ons, "
        f"{len(session.pedal_events)} pedal events."
    )

    env_times, env = onset_envelope(session.audio, session.sample_rate)
    audio_onsets = pick_onsets(env_times, env, threshold=args.onset_threshold)
    median_delta, deltas = estimate_offset(session.onset_times, audio_onsets)

    if np.isnan(median_delta):
        print(
            "Offset: not enough MIDI↔audio matches to estimate "
            "(need a take with clear, mostly-isolated notes — and a MIDI device "
            "connected during capture)."
        )
    else:
        print(
            f"Estimated capture offset: {median_delta * 1000.0:+.1f} ms "
            f"(median over {len(deltas)} matched onsets). "
            f"Subtract this from MIDI times when building training labels."
        )

    fig = make_figure(session, audio_onsets, median_delta, deltas)
    out_path = session_dir / "alignment.png"
    fig.savefig(out_path, dpi=110)
    print(f"Wrote {out_path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
