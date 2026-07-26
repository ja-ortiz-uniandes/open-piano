# CLAUDE.md — contributor & agent notes for open-piano

Orientation for working in this repo. Read alongside [README.md](README.md)
(user-facing) and [MODEL.md](MODEL.md) (how to obtain the model + ONNX Runtime).

## What this is

A real-time P2P acoustic-piano visualizer in **Rust** (egui/eframe GUI). Two
peers see one shared 88-key keyboard; each player's notes light up in that
player's chosen color. Input is either a MIDI device (preferred) or microphone
audio transcribed by an ONNX model. See README for the product goal.

## Golden rule — one shared surface

The keyboard and both roll panels (top + bottom) are **a single thing both
players look at**, not two separate instances that merely share some state.
Whatever appears on one side must appear on the other, identically. When you add
anything that lights a key, draws a mark, or otherwise changes what's on that
surface — live notes, the sustain lane, Ctrl+click-pinned keys, the metronome
grid — it has to be broadcast to the peer and rendered the same on both screens.
Colors sync too: each player's real chosen color renders identically on both
screens (the peer's `Packet::Color` announcement is applied to `remote_color`,
not discarded) — a note played by one side must look the same regardless of
which screen you're looking at it on. The only deliberate asymmetry is a
fallback: if both sides are still at the untouched default color (two fresh
installs, neither has opened the color picker), the *joiner* alone switches to
a distinguishable fallback color and re-announces it, so an un-customized pair
doesn't render identically. This is computed the same way on both ends
(`is_host` is symmetric and known identically by construction, so there's no
race), never overrides a manual color choice, and never fires again once
`local_color` differs from the default. If a feature shows up on only one
side, that's a bug, not a local nicety.

Prefer **idempotent whole-state snapshots** over incremental deltas for anything
synced over the (unreliable-datagram) wire, and re-send on a heartbeat while the
state persists — a dropped packet must self-heal rather than leave the two
surfaces mismatched (see `Packet::Held`, the color/name heartbeat, and the
pedal send-on-change for the established patterns).

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
  Both the short `EndpointId` (64 hex chars) and the full `EndpointTicket`
  (relay + direct addresses inline, ~4× longer) are always useful, not
  "short with a fallback": `full` is available the instant the endpoint
  binds, `short` once a self-resolve confirms n0 discovery can actually serve
  it — the host publishes both and re-publishes on a heartbeat for as long as
  hosting continues, upgrading in place as reachability improves (see
  `publish_invite_code`/`confirm_published`). The UI always offers `full` too,
  as the "same-network code". Join accepts either form. Each session runs a
  dedicated "net" thread with a
  current-thread tokio runtime; the UI receives `NetEvent`s (ticket, status,
  connect/disconnect, packets) on an mpsc channel and queues outgoing `Packet`s
  on an unbounded sender. Packets ride *unreliable QUIC datagrams* — the same
  fire-and-forget latency model (and identical wire bytes) as the original
  raw-UDP transport. A joiner's `run_join` never permanently gives up: it
  redials forever on `JOIN_BACKOFF` (capping at 30s), both for the initial
  connect and for reconnecting after an established session drops — the only
  way to stop it is the Cancel button or a fresh Join click (a hard reset:
  drops and rebuilds the whole session). The endpoint's identity
  (`iroh::SecretKey`) is persisted in `Prefs::endpoint_secret` and reused
  across restarts (generated lazily on first Host/Join, via
  `net::generate_secret`), so the invite code stays the same after a crash or
  update; Edit ▸ Preferences ▸ Networking ▸ "Reset my identity" clears it. A
  dropped net thread (panic, or a spawn failure) always reports through
  `NetEvent::Disconnected` — either explicitly (`NetThreadGuard`'s `Drop`) or
  implicitly (`pump_network` noticing the event channel itself has closed) —
  so a dead thread can never leave the UI holding a zombie `Peer` or a stuck
  remote key.
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
  **every** build, not just the mic path. Also owns `app_dir()` — the shared
  `%LOCALAPPDATA%\open-piano` directory `prefs.rs` and `diag.rs` write into too.
- **`diag.rs`** — crash diagnostics: a write-through breadcrumb log
  (`%LOCALAPPDATA%\open-piano\open-piano.log`), a panic hook, a native
  (`SetUnhandledExceptionFilter`) handler for the crashes a panic hook can't
  see (access violations, OOM aborts), and unclean-exit detection via a
  per-pid marker file. `begin_session()` is the first statement in `main()`.
  Release builds are `windows_subsystem = "windows"` with no console and no
  OS crash dialogs, so this is the only record of a crash that exists.

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

## Commits, versioning & releases

**Commit messages** follow Conventional Commits: `type(scope): subject`, e.g.
`feat(playback): add pause-on-miss to Evaluation mode` or `fix(net): retry
relay handshake on timeout`. Common types: `feat` (new user-facing
capability), `fix`, `docs`, `chore`, `refactor` (no behavior change), `perf`,
`test`, `build`, `ci`. Scope is the touched module (`main`, `net`, `playback`,
`roll`, `prefs`, `midi`, `inference`, ...) — omit it only when a change
genuinely spans everything. Add a body paragraph explaining *why* when the
subject line alone doesn't carry the motivation.

**Version numbers** (`Cargo.toml`, and the release tag) bump on the size and
count of what changed since the last release — the project is pre-1.0, so
strict semver doesn't apply yet:

- **PATCH** (`0.0.X`): one small, self-contained feature, or a release that's
  fixes/docs/chores only with no new user-facing capability.
- **MINOR** (`0.X.0`): a release bundles multiple substantial features, or a
  single feature large enough to touch several modules or change core
  behavior meaningfully.
- **MAJOR** (`X.0.0`): reserved for 1.0 and a deliberate compatibility
  commitment (stable wire protocol / prefs format) — not expected before the
  causal/streaming model lands (see Next steps).

Bump `Cargo.toml`'s version to match before tagging.

**Tags**: always annotated and signed, with a real summary message — never a
bare lightweight tag:

```powershell
git tag -s -a v0.6.0 -m "v0.6.0: <one-line summary of what shipped>"
git push origin v0.6.0
```

`-s` signs the tag with your configured signing key (`git config
user.signingkey` / `gpg.format`), which is what makes it show Verified on
GitHub — a lightweight tag only inherits Verified from an already-signed
commit, and a plain (unsigned) annotated tag shows unverified even when the
commit underneath it is signed. Write the message like a mini changelog
entry (what shipped, not just the version number) — it's what renders on the
GitHub tag/release page. Pushing the tag triggers
`.github/workflows/release.yml`, which builds a portable Windows zip (exe +
ONNX Runtime + model) and publishes a GitHub Release. Distribution and the
Windows SmartScreen/Smart App Control situation are documented in the
README.

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
- **Diagnosing a report of "the app just closed"**: check `diag.rs`'s log first
  (Help ▸ Open log folder, or the status-bar chip after an unclean exit) — it
  has the panic/SEH record plus the breadcrumb trail leading up to it. If a
  user reports a crash with no log captured (an old build, or the log itself
  didn't survive), escalate manually:

  ```powershell
  # Step 1 — what Windows ALREADY recorded. Zero setup. Gives the faulting
  # MODULE NAME (onnxruntime.dll vs open-piano.exe vs a GPU driver), its
  # version, the exception code and offset. Often the whole answer.
  Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000} -MaxEvents 50 |
    Where-Object { $_.Message -match 'open-piano' } |
    Format-List TimeCreated, Message
  # Id=1002 is Application Hang, for freezes.

  # Step 2 — full dumps for the NEXT occurrence (ELEVATED shell)
  $key = 'HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\open-piano.exe'
  New-Item -Path $key -Force | Out-Null
  New-ItemProperty -Path $key -Name DumpFolder -PropertyType ExpandString `
    -Value "%LOCALAPPDATA%\open-piano\dumps" -Force | Out-Null
  New-ItemProperty -Path $key -Name DumpCount -PropertyType DWord -Value 5 -Force | Out-Null
  New-ItemProperty -Path $key -Name DumpType  -PropertyType DWord -Value 2 -Force | Out-Null  # 2=full
  # Undo: Remove-Item $key -Recurse
  ```

  A release build's `open-piano.pdb` (published as its own GitHub Release
  asset, not inside the zip — see `.github/workflows/release.yml`) symbolizes
  addresses from either source against that exact version.

## Pending manual verification (v0.5.0)

v0.5.0 (velocity-graded roll marks, the sustain-pedal lane, and Evaluation
mode + review) was committed and tagged **without hardware-in-the-loop
testing** — `cargo test`/`cargo build` pass, but nobody has yet played it on
a real MIDI keyboard or mic, or checked a live peer session. Before trusting
this release, run through:

- Resize the app below ~700px tall, open Preferences — title bar should stay
  visible, body scrolls.
- With a MIDI keyboard: check velocity-graded mark saturation, enable and
  exercise the pedal lane (including half-pedaling), then unplug/switch to
  mic and confirm the toggle and lane disappear. Check both over a live peer
  session if possible.
- Load a score, pick Evaluation, play through with some deliberate mistakes —
  the roll should never freeze, the results window should pop at the end, and
  the review toggles should switch between original/played/both for both eyes
  and ears.

Delete this section once it's been run through.

## Next steps (see README Roadmap for context)

1. **Code signing** in the release workflow for SmartScreen/SAC.
2. **Training pipeline**: `sessions → framed (input, label) tensors` — apply the
   `verify_alignment.py` offset, render per-frame onset/sustain targets from
   `midi.jsonl` (account for CC64 pedal sustaining notes past key-up), optionally
   add Basic Pitch offline outputs as distillation targets. Then train a small
   **causal/streaming** model, export to ONNX, and replace the windowed model in
   `inference.rs` (deleting most of its hysteresis constants). This is the payoff
   that makes the mic path low-latency and accurate.

## CodeGraph

This repo is indexed by [CodeGraph](https://github.com/colbymchenry/codegraph) (a `.codegraph/` directory at the repo root — a SQLite knowledge graph of the codebase's symbols, edges, and files). Reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

The index is local to each machine (`.codegraph/` is gitignored except its own `.gitignore`) and goes stale as files change. A local git hook re-syncs it after every commit (see "Replication" below). If the hook isn't present, run `codegraph sync` manually, or `codegraph status` to check for drift.

### Replication (new clone / new machine)

CodeGraph's index and sync hook are **local-only** and not committed to git, so set this up once per clone:

1. Install the CodeGraph CLI if needed (see the CodeGraph docs).
2. From the repo root, run `codegraph init` to build the initial index.
3. Add a local post-commit hook so the index stays current automatically:

   ```sh
   printf '#!/bin/sh\ncodegraph sync -q\n' > .git/hooks/post-commit
   chmod +x .git/hooks/post-commit
   ```
