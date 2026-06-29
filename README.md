# open-piano

A real-time, peer-to-peer acoustic-piano **visualizer**. Two people, each at
their own computer, see a shared 88-key keyboard light up live as either of them
plays — your notes in your color, your partner's in theirs.

## Goal

Let two musicians in different places watch each other play in real time, with
**very low latency** and **high accuracy**, using whatever input each has:

- A **digital piano / MIDI keyboard** (preferred) — exact note events, zero
  transcription.
- A **microphone + acoustic piano** (fallback) — audio is transcribed to notes
  by an on-device ML model.

The long game is making the *microphone* path as fast and accurate as the MIDI
path by training a piano-specific, low-latency transcription model on data
captured from the app itself (see [Roadmap](#roadmap)).

## Current state

Working today:

- **Dual input, auto-selected.** Plug in a MIDI device and it's used instantly;
  unplug it and the app falls back to the microphone. Hot-plug is handled live.
  (`src/input.rs`, `src/midi.rs`, `src/audio.rs`)
- **P2P networking over UDP.** Each side binds a local port and targets the
  other's IP\:port. Note events are 2-byte datagrams sent the instant they
  happen — chosen for lowest latency over guaranteed delivery. (`src/net.rs`)
- **Per-player colors.** You pick *your* color; it's sent over the wire so it
  shows up as your color on your partner's screen, and vice-versa. When you
  **both press the same key at once**, that key splits diagonally — your color in
  one half, theirs in the other — so a simultaneous press is unmistakable.
  (`src/main.rs`)
- **ML transcription (microphone path).** Spotify's **Basic Pitch** model runs
  via ONNX Runtime on a dedicated thread. (`src/inference.rs`)
- **Training-data capture harness.** A "Record" button logs microphone audio
  (`audio.wav`) and, when a MIDI device is connected, the exact MIDI labels
  (`midi.jsonl`, including velocity and CC64 sustain pedal) on a shared clock —
  the raw material for training a better model. (`src/record.rs`)
- **Offline alignment verifier.** A Python script overlays the captured MIDI on
  the audio spectrogram and measures the audio↔MIDI latency offset, so you can
  confirm a recording is well-aligned before relying on it.
  (`verify_alignment.py`)

Known limitation: the microphone path is **laggy and imprecise** today — Basic
Pitch is an offline model run in a sliding window, so attacks appear late,
releases linger, and ghost notes occur. Fixing this is the roadmap. The MIDI
path does not have this problem.

## Build & run from source

Requires a [Rust toolchain](https://rustup.rs/) and (for the microphone path)
Python 3.8+.

```powershell
# 1. Get the ML model + ONNX Runtime DLL into the project root (see MODEL.md).
python download_model.py

# 2. Build and run.
cargo run --release
```

The microphone path needs `model.onnx` and `onnxruntime.dll` in the working
directory; the status bar reports whether they loaded. **The MIDI path needs
neither** — if you only ever use a MIDI keyboard, you can skip step 1.

> **Building under Smart App Control:** if your machine has Windows Smart App
> Control *enforcing*, a from-scratch build fails with `os error 4551` — `cargo`
> runs unsigned build-script executables that SAC blocks. (Incremental builds on
> a warm `target/` cache still work, which hides this.) Build on a machine with
> SAC off / in evaluation mode, or just let CI build the release for you (the
> GitHub Actions runner is unaffected) and use the portable zip. Don't
> `cargo clean` on a SAC machine unless you can rebuild elsewhere.

### Connecting two computers

1. Both people launch the app.
2. Each sets **Local Port** (e.g. `9000`) and the **Remote IP / Port** of the
   other machine, then clicks **Connect**. (Across the internet you'll need to
   forward the UDP port or be on the same LAN/VPN.)
3. Pick your color. Play.

## Distribution & updates

Releases are built automatically by GitHub Actions
([`.github/workflows/release.yml`](.github/workflows/release.yml)). To cut one:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

That produces `open-piano-v0.1.0-win-x64.zip` on the repo's **Releases** page: a
**portable** folder containing `open-piano.exe`, `onnxruntime.dll`, `model.onnx`,
and the docs.

**To install or update** (e.g. on your professor's machine): download the latest
zip, unzip it anywhere (Desktop, a USB stick, wherever), and run
`open-piano.exe`. Updating is just replacing the old folder with the new one — no
installer, no admin rights, no settings to migrate.

### Windows security / Smart App Control — read this

The app is **portable and needs no elevated rights**, so you never have to turn
off antivirus or UAC. But be aware of code-signing reality:

- The executable is currently **unsigned**. On first run, Windows SmartScreen may
  show *"Windows protected your PC"* — click **More info → Run anyway**. This is
  normal for unsigned indie software and does **not** require disabling any
  security feature.
- **Smart App Control (SAC)**, if a machine has it in *enforced* mode, will
  **block unsigned executables outright** with no "run anyway." SAC can only be
  satisfied by a validly **code-signed** binary — there is no portable trick
  around it. Most machines have SAC off or in evaluation mode, so this usually
  isn't hit; if your professor's machine enforces SAC, the real fix is signing
  (see [Roadmap](#roadmap)), not asking them to weaken security.

In short: portable + unsigned clears SmartScreen with one click and never touches
security settings; only *enforced SAC* would require code signing.

## Capturing training data

With a MIDI piano connected, click **Record**. Play. Click again to stop. Each
session lands in `recordings/session_<unixtime>/` as `audio.wav` + `midi.jsonl` +
`meta.json`. Then verify alignment:

```powershell
pip install numpy scipy matplotlib
python verify_alignment.py recordings/session_<id>
```

It writes `alignment.png` and prints the capture latency offset. Do a short
calibration take of isolated staccato notes first to confirm sync before
collecting a lot of data. See `src/record.rs` for the file formats.

## Roadmap

Near-term, in rough order:

1. **In-app auto-update.** Add the [`self_update`](https://crates.io/crates/self_update)
   crate to check the GitHub Releases API on launch, download a newer portable
   build, and self-replace — so the professor's copy updates itself instead of
   needing a manual re-download. (Manual zip-replacement works today.)
2. **Code signing** so signed releases clear SmartScreen silently and satisfy
   enforced Smart App Control. An EV certificate gets instant reputation; a
   cheaper OV cert builds reputation over time.
3. **Train the fast piano model.** Collect 2–10 hours of aligned audio+MIDI,
   then train a small **causal/streaming** transcription network (so it doesn't
   need a look-ahead window like Basic Pitch) — optionally distilling from Basic
   Pitch plus the captured labels. Export to ONNX and drop it into
   `src/inference.rs`, replacing the windowed model and most of its hand-tuned
   hysteresis. This is what makes the microphone path low-latency *and* accurate.
   The data pipeline (sessions → framed training tensors, applying the measured
   offset and rendering per-frame onset/sustain targets) is the next script to
   write once a real session exists.

See [CLAUDE.md](CLAUDE.md) for architecture details and contributor notes.
