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
- **P2P networking with one-string invites.** One player clicks **Host** and
  sends the other a copy-pasteable invite code; the other pastes it and clicks
  **Join** — no IPs, no port forwarding, no router config. Under the hood it's
  [iroh](https://github.com/n0-computer/iroh): NAT hole punching when possible,
  a relay as fallback (so it works behind VPNs and CGNAT), authenticated by the
  host's public key. Note events still ride *unreliable datagrams* sent the
  instant they happen — chosen for lowest latency over guaranteed delivery.
  (`src/net.rs`)
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
- **Synced metronome.** A shared click both players hear together: the host is
  the timing anchor and either player can set the tempo or start/stop it (a
  guest's change is adopted by the host, so there's one grid, never two duelling
  clocks). Each player can mute *their own* click locally without silencing the
  other. Solo, it's just a plain metronome. (`src/synth.rs`, `src/main.rs`)
- **Preferences, persisted.** An **Edit ▸ Preferences** dialog (Ctrl+,)
  consolidates the roll timing, appearance, mic, and advanced model/network
  tunables, saved to `%LOCALAPPDATA%\open-piano\preferences.json` and restored
  on launch. An **Advanced** section (behind an expander, with a Reset) exposes
  the detector knobs that used to be compile-time constants, live-editable while
  you play. (`src/prefs.rs`)
- **Custom window chrome.** The app draws its own title bar — File/Edit menus on
  the left, minimize/maximize/close on the right — so the menus are always the
  topmost row, with window drag, double-click-to-maximize, and edge resizing.
  (`src/main.rs`)
- **Configurable roll timing.** The piano roll now preserves the *real* silence
  between phrases: a gap in your playing shows as that much blank paper, up to a
  cap (default 20 s) and an idle-pause threshold (default 30 s) — each settable
  in Preferences, and each switchable to **∞** (never clamp / never auto-pause)
  for a truly unbounded gap. (`src/roll.rs`)

Known limitation: the microphone path is **laggy and imprecise** today — Basic
Pitch is an offline model run in a sliding window, so attacks appear late,
releases linger, and ghost notes occur. Fixing this is the roadmap. The MIDI
path does not have this problem.

## Build & run from source

Requires a [Rust toolchain](https://rustup.rs/) and Python 3.8+.

```powershell
# 1. Get the ML model + ONNX Runtime DLL into the project root (see MODEL.md).
#    Required for every build: both files are compiled INTO the exe.
python download_model.py

# 2. Build and run.
cargo run --release
```

The model and ONNX Runtime are **embedded in the executable at build time**
(`src/bundle.rs`), which is why step 1 is mandatory — `include_bytes!` needs
the files present. The resulting exe is fully self-contained: nothing to ship
or place beside it.

> **Building under Smart App Control:** if your machine has Windows Smart App
> Control *enforcing*, a from-scratch build fails with `os error 4551` — `cargo`
> runs unsigned build-script executables that SAC blocks. (Incremental builds on
> a warm `target/` cache still work, which hides this.) Build on a machine with
> SAC off / in evaluation mode, or just let CI build the release for you (the
> GitHub Actions runner is unaffected) and use the portable zip. Don't
> `cargo clean` on a SAC machine unless you can rebuild elsewhere.

## Connecting two instances

One side hosts, the other joins — three steps, works the same on one machine,
one LAN, or across the internet:

1. **Player A** clicks **Host session**, waits a moment for the invite code,
   then clicks **📋 Copy invite code** and sends the code to Player B (chat,
   email, anything).
2. **Player B** pastes it into the **Invite code** box and clicks **Join**.
3. Pick your colors and play. The status bar on both sides shows
   `Connected to peer …` once the link is up.

There are no IPs or ports to exchange and **no router configuration**: the
invite code is the host's public key (64 characters), and everything else is
looked up automatically. Connections are carried by
[iroh](https://github.com/n0-computer/iroh) — the two machines rendezvous
through a public relay server, hole-punch a direct connection when the
networks allow it, and silently fall back to the relay when they don't (strict
NATs, VPNs, CGNAT). Traffic is end-to-end encrypted (QUIC/TLS) and the host is
authenticated by that same key, so a leaked code is the only way for a
stranger to join — generate a fresh one per session.

Details worth knowing:

- **Invite codes are per-session.** A code stays valid while that instance is
  hosting, and a new **Host session** click mints a new one. If the peer drops
  off (network blip), the host keeps listening — the joiner just presses
  **Join** again with the same code.
- **Joining right after hosting may take a few extra seconds.** The host's
  address is published to a lookup service when it starts; a joiner who pastes
  the code within seconds can race that. The app retries automatically
  ("Not reachable yet, retrying…" in the status bar).
- **No internet?** A host that can't reach the relay servers falls back to a
  longer (~250-character) invite code with its LAN addresses baked in, so
  same-network play works fully offline. Joining accepts both code forms.
- **Order doesn't matter for colors.** A 1 s color heartbeat syncs colors
  whenever both ends are up (it also keeps the connection warm).
- **Quick local test:** run the app twice on one machine, Host in one, paste
  the code in the other.

### If nothing lights up

- **Firewall:** the first time you host/join, Windows may prompt to allow
  `open-piano` through the firewall — say yes.
- **Status bar says "Contacting relay…" forever:** the machine can't reach the
  relay servers (offline, or a network blocking them). On a shared LAN it still
  works — the invite code carries direct addresses too.
- **"Could not reach host":** the host closed the app (or clicked Host again,
  which invalidates the old code). Ask for a fresh code.
- Notes are sent as fire-and-forget datagrams (lowest latency over guaranteed
  delivery), so an occasional dropped packet is expected and harmless; a key
  that never lights at all is a connection issue, not packet loss.

## Distribution & updates

Releases are built automatically by GitHub Actions
([`.github/workflows/release.yml`](.github/workflows/release.yml)). To cut one:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

That produces `open-piano-v0.1.0-win-x64.zip` on the repo's **Releases** page:
a **single self-contained `open-piano.exe`** (the ML model and ONNX Runtime are
embedded inside it) plus this README.

**To install** (e.g. on your professor's machine): download the latest zip, unzip
it anywhere (Desktop, a USB stick, wherever), and run `open-piano.exe`. No
installer, no admin rights, no settings to migrate.

**Updating is automatic and complete.** On launch the app checks the GitHub
Releases API; if a newer version exists it quietly downloads it and swaps in the
new `open-piano.exe`, then shows an **"Update ready — Restart now"** banner.
Click it (or just reopen the app later) to land on the new build. Because the
exe embeds the model and runtime, an update carries *everything* — there are no
side-files that can go stale. A failed check (offline, rate-limited) is silent —
the app just runs the current build.

(First-run detail: the app unpacks its embedded ONNX Runtime to
`%LOCALAPPDATA%\open-piano\` so Windows can load it; that cache cleans itself
up across versions and can always be deleted — it's recreated on next launch.)

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
  (in progress — see below), not asking them to weaken security.

In short: portable + unsigned clears SmartScreen with one click and never touches
security settings; only *enforced SAC* would require code signing.

**Code signing (in progress).** open-piano is applying to the
[SignPath Foundation](https://signpath.org/) free code-signing program for open
source. Once approved, release binaries will be signed automatically in CI — at
which point this section will be updated and the line below goes live:

> *Free code signing provided by [SignPath.io](https://signpath.io/), certificate
> by [SignPath Foundation](https://signpath.org/).*

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

1. **Code signing** so signed releases clear SmartScreen silently and satisfy
   enforced Smart App Control. Being set up for free via the
   [SignPath Foundation](https://signpath.org/) OSS program (the project's dual
   MIT/Apache-2.0 license and CI build qualify); the signing step plugs into the
   release workflow once approved (staged in
   `.github/workflows/release-signed.yml.disabled`).
2. **Train the fast piano model.** Collect 2–10 hours of aligned audio+MIDI,
   then train a small **causal/streaming** transcription network (so it doesn't
   need a look-ahead window like Basic Pitch) — optionally distilling from Basic
   Pitch plus the captured labels. Export to ONNX and drop it into
   `src/inference.rs`, replacing the windowed model and most of its hand-tuned
   hysteresis. This is what makes the microphone path low-latency *and* accurate.
   The data pipeline (sessions → framed training tensors, applying the measured
   offset and rendering per-frame onset/sustain targets) is the next script to
   write once a real session exists.

See [CHANGELOG.md](CHANGELOG.md) for the release history, and
[CLAUDE.md](CLAUDE.md) for architecture details and contributor notes.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option. Unless you explicitly state otherwise, any contribution you submit
for inclusion is dual-licensed as above, without additional terms.

The bundled ONNX Runtime and Basic Pitch model are third-party components under
their own permissive licenses — see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
