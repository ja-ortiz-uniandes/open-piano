# CLAUDE.md — contributor & agent notes for open-piano

Orientation for working in this repo. Read alongside [README.md](README.md)
(user-facing) and [MODEL.md](MODEL.md) (how to obtain the model + ONNX Runtime).

## What this is

A real-time P2P acoustic-piano visualizer in **Rust** (egui/eframe GUI). Two
peers see one shared 88-key keyboard; each player's notes light up in that
player's chosen color. Input is either a MIDI device (preferred) or microphone
audio transcribed by an ONNX model. See README for the product goal.

## Architecture & data flow

```text
            ┌─ MIDI device (preferred) ──────────────┐
 input.rs ──┤                                         ├─→ mpsc<NoteMsg> ─→ UI
 (supervisor)└─ mic (cpal) ─→ inference.rs (ONNX) ────┘                    (main.rs)
                                                                            │  ▲
 record.rs ←── tee: mic audio + raw MIDI (when armed)                       │  │ NetEvent
                                                                            ▼  │
                                                             net.rs (iroh) ─┴──┘ peer
```

- **`main.rs`** — eframe app + all rendering. Owns key state (`local`/`remote`
  bool arrays), colors, and the `Peer`. `update()` pumps the input channel and
  the network channel each frame and repaints. Keyboard drawing + the diagonal
  split for simultaneous same-key presses live here (`paint_key`,
  `draw_keyboard`).
- **`input.rs`** — supervisor thread. Polls MIDI ports (~1 s), keeps exactly one
  *note source* live (MIDI preferred), and bumps an `epoch` on every switch so
  the UI force-releases stuck notes. Also drives the **recording session
  lifecycle** (runs a capture-only mic alongside MIDI while armed).
- **`midi.rs`** — MIDI input via `midir`. Translates note on/off → `NoteMsg` for
  the UI and **tees raw bytes to the recorder** (velocity + CC, incl. CC64).
- **`audio.rs`** — mic capture via `cpal` (WASAPI on Windows). Two entry points:
  `start_into` (capture → inference thread) and `start_record_capture`
  (capture-only → recorder, no model). `downmix_mono` is shared.
- **`inference.rs`** — ONNX Basic Pitch on a dedicated thread: resample → 2 s
  window → posteriorgram → thresholding/hysteresis → `NoteMsg`. Heavy with
  hand-tuned constants compensating for the model being offline (see roadmap —
  these go away with a causal model).
- **`net.rs`** — P2P over iroh (QUIC + NAT traversal). One side `host()`s and
  gets a one-string invite code; the other `join()`s with it — hole punching
  when possible, n0's public relays as fallback, so no port forwarding ever.
  The code is normally the bare `EndpointId` (64 hex chars; dial info comes
  from n0 discovery, which the `N0` preset publishes to), falling back to a
  full `EndpointTicket` when the host is offline/LAN-only; join accepts both.
  Each session runs a dedicated "net" thread with a
  current-thread tokio runtime; the UI receives `NetEvent`s (ticket, status,
  connect/disconnect, packets) on an mpsc channel and queues outgoing `Packet`s
  on an unbounded sender. Packets ride *unreliable QUIC datagrams* — the same
  fire-and-forget latency model (and identical wire bytes) as the original
  raw-UDP transport.
- **`note.rs`** — `NoteMsg` (On/Off), MIDI helpers, and the **wire protocol**
  (`Packet`): note bytes `[0x90|0x80, note]`, color `[0xC0, r, g, b]`, metronome
  beat `[0xB0, ...]` and control `[0xB1, ...]`. The synced metronome (host is the
  timing anchor; beat markers carry an RTT-derived one-way stamp added in
  `net.rs`) lives in `main.rs` (`Metronome`, `drive_metronome`) + `synth.rs`
  (`Channel::Metronome` click voice).
- **`prefs.rs`** — serde `Prefs` (+ `Limit`), persisted to
  `%LOCALAPPDATA%\open-piano\preferences.json` (atomic temp+rename;
  `#[serde(default)]` on every field). Loaded in `main`'s `new()`, edited via
  Edit ▸ Preferences, saved on change. Live-editable detector knobs reach the
  inference thread via the `SharedF32`/`InferenceTunables` atomics in `audio.rs`.
  The window uses **custom chrome** (`with_decorations(false)` + `title_bar`), so
  File/Edit and the min/max/close buttons are drawn by us.
- **`record.rs`** — `Recorder` handle + background writer thread. Writes
  `recordings/session_<unix>/{audio.wav, midi.jsonl, meta.json}`. All disk I/O is
  off the realtime callbacks.
- **`update.rs`** — in-app auto-update via `self_update`. A background thread
  checks GitHub Releases on launch and, if a newer tag exists, downloads the
  portable zip and self-replaces `open-piano.exe`; the UI polls `UpdateState` and
  offers a one-click restart. Only the exe is swapped — sufficient, because the
  exe embeds the model and runtime (see `bundle.rs`).
- **`bundle.rs`** — the exe is **self-contained**: `model.onnx` and
  `onnxruntime.dll` are `include_bytes!`-embedded at build time. The model is
  loaded from memory; the DLL is extracted on startup to
  `%LOCALAPPDATA%\open-piano\onnxruntime-<hash>.dll` (content-hash-named so
  concurrent old/new versions never clobber each other) and `ORT_DYLIB_PATH`
  points at it. Consequence: `python download_model.py` is a prerequisite for
  **every** build, not just the mic path.

## Threading model (important)

- GUI thread: egui `update()` only. Never blocks.
- Input supervisor thread: port polling + backend lifecycle.
- Inference thread: all ONNX work (mic path).
- Audio capture: `cpal` callback thread(s) — keep them cheap (downmix + channel
  send only).
- MIDI callback thread: `midir` — cheap (parse + channel send + recorder tee).
- Recorder writer thread: all file writes.
- Net thread (one per host/join session): a current-thread tokio runtime
  driving the iroh endpoint; shuts down when the UI drops its `Peer` handle.
- Auto-update thread: one-shot GitHub API check + download + self-replace.

The non-`Send` `midir` connection never crosses threads — it's owned by the
supervisor. Cross-thread timing uses `std::time::Instant` (one process-wide
monotonic clock), which is how the recorder aligns audio and MIDI.

## Conventions

- Keep realtime/callback paths allocation-light and lock-free where practical;
  push work to dedicated threads via channels (the existing pattern).
- Doc-comment modules and non-obvious constants — match the existing dense,
  explanatory comment style (see `inference.rs` for the bar).
- Prefer adding a typed channel message over sharing mutable state across
  threads.
- **Python** (tooling like `verify_alignment.py`): always add type hints to all
  Python code — type-hint every function signature (parameters and return type,
  including `-> None`) and add variable annotations where helpful.
- Don't commit `model.onnx`, `onnxruntime.dll`, or `recordings/` (gitignored).

## Build / run / test

```powershell
python download_model.py # fetch model.onnx + onnxruntime.dll — REQUIRED first:
                         # they're include_bytes!-embedded into every build
cargo build              # dev (opt-level 1 for the DSP loops)
cargo build --release    # release; what the CI release workflow ships
cargo run --release      # run the app
```

`cargo test` runs the one automated test: `net::tests::host_join_exchange_notes`
hosts and joins over real iroh (loopback + relay) and asserts packets flow both
ways — run it after touching `net.rs`; it needs a network stack and takes a few
seconds. Everything else is manual. The capture harness was validated against
a synthetic session; `verify_alignment.py` recovers a known injected offset to
within ~1 ms. When changing the recorder or alignment math, re-validate with a
synthetic session (sine tones at known times + a matching `midi.jsonl`).

## Releases

Push a `vX.Y.Z` tag → `.github/workflows/release.yml` builds a portable Windows
zip (exe + ONNX Runtime + model) and publishes a GitHub Release. Distribution and
the Windows SmartScreen/Smart App Control situation are documented in the README.

## Gotchas

- ONNX Runtime is loaded **lazily on the inference thread** via `ORT_DYLIB_PATH`;
  never load it on the main thread (Windows loader-lock deadlock — see the
  comment in `main.rs::main`).
- `midir`'s `Ignore::All` filters sysex/clock/active-sensing only — **note and CC
  messages still arrive**, which is why the recorder gets CC64 without changing
  the ignore flags.
- The Record toggle has up to ~1 s latency because the supervisor reconciles it
  on its poll interval.
- Colors are re-broadcast on a 1 s heartbeat so they sync regardless of who
  connects first; don't "optimize" that away without another sync mechanism.
  (It also keeps the QUIC connection from idling out.)
- iroh needs tokio, but only the net thread runs a runtime — never block the
  GUI thread on async work; talk to the net thread via the existing channels.
- `painter.rect` returns a `ShapeIdx` in egui 0.29 — match arms that mix it with
  unit need explicit `;`/blocks.
- **Smart App Control blocks local builds.** On a machine with Windows Smart App
  Control (SAC) *enforcing*, a from-scratch `cargo build` fails with `os error
  4551` ("An Application Control policy has blocked this file") because cargo
  compiles and runs **unsigned build-script executables** (e.g. `khronos_api`,
  `zerocopy`) that SAC kills. Incremental builds against an already-populated
  `target/` cache still work, which masks the problem. There are no per-folder
  exclusions for SAC. Implications: build from scratch on a machine with SAC off
  / in evaluation mode, **or rely on CI** — the GitHub Actions release workflow
  runs on GitHub's runners and is unaffected. Do **not** `cargo clean` on a SAC
  machine unless you can rebuild elsewhere.

## Next steps (see README Roadmap for context)

1. **Code signing** in the release workflow for SmartScreen/SAC.
2. **Training pipeline**: `sessions → framed (input, label) tensors` — apply the
   `verify_alignment.py` offset, render per-frame onset/sustain targets from
   `midi.jsonl` (account for CC64 pedal sustaining notes past key-up), optionally
   add Basic Pitch offline outputs as distillation targets. Then train a small
   **causal/streaming** model, export to ONNX, and replace the windowed model in
   `inference.rs` (deleting most of its hysteresis constants). This is the payoff
   that makes the mic path low-latency and accurate.
