# Changelog

All notable changes to open-piano are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-29

First release: a working real-time, peer-to-peer acoustic-piano visualizer.

### Added

- **Dual note input, auto-selected.** A connected MIDI device is used instantly
  and preferred; with none, the app falls back to microphone transcription.
  Hot-plug is handled live — plug or unplug a piano mid-session and the active
  source switches, force-releasing any stuck notes.
- **Peer-to-peer networking over UDP.** Each instance binds a local port and
  targets the other's IP/port; note events are sent as fire-and-forget datagrams
  for lowest latency. See the README for the same-machine, LAN, and
  internet connection setups.
- **Per-player colors.** You choose your own color; it's broadcast to the peer (on
  a 1 s heartbeat so it syncs regardless of who connects first). When both players
  hold the same key, it splits diagonally so a simultaneous press is unmistakable.
- **ML transcription (microphone path).** Spotify's Basic Pitch model runs via
  ONNX Runtime on a dedicated inference thread.
- **Built-in synth.** A small polyphonic synth voices the notes with no acoustic
  source — the keys you click on the on-screen keyboard and the notes the peer
  plays — with independent volume/mute for each. MIDI and microphone notes are
  not synthesized (they already make their own sound).
- **Training-data capture harness.** A Record button logs microphone audio
  (`audio.wav`) and, when a MIDI device is connected, exact MIDI labels
  (`midi.jsonl`, including velocity and CC64 sustain) on a shared clock, plus an
  offline `verify_alignment.py` that measures the audio↔MIDI latency offset.
- **In-app auto-update.** On launch the app checks GitHub Releases and, if a newer
  version exists, downloads it and offers a one-click restart into the new build.

### Fixed

- **Microphone↔synth echo loop.** In microphone mode the synth's own output bled
  through the speakers into the mic and was re-detected as played notes, leaving
  keys lit after release. Notes the synth is voicing (and a short release-tail
  window after) are now ignored by mic detection, so the on-screen keyboard and
  peer notes no longer echo back onto the keyboard.

### Changed

- **On-screen synth muted by default with a MIDI device connected.** A real piano
  already makes its own sound, so the on-screen ("screen") synth auto-mutes while
  a MIDI device is connected and unmutes on the microphone fallback; a manual
  toggle sticks until the next plug/unplug. Highlighting of both local and remote
  notes always happens, and the synth stays disabled while recording training
  data.

### Known limitations

- The microphone path is laggy and imprecise: Basic Pitch is an offline model run
  in a sliding window, so attacks appear late, releases linger, and ghost notes
  occur. The MIDI path is exact. Replacing the windowed model with a trained
  causal/streaming one is the roadmap.
- Release binaries are unsigned, so Windows SmartScreen warns on first run and
  enforced Smart App Control blocks them outright. Code signing is on the roadmap.

[0.1.0]: https://github.com/ja-ortiz-uniandes/open-piano/releases/tag/v0.1.0
