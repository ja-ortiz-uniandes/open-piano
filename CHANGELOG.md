# Changelog

All notable changes to open-piano are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-07-05

### Added

- **App icon.** The window/taskbar and the compiled `.exe`'s file icon now
  show the open-piano logo instead of the default.
- **Scroll the piano rolls.** Both the history roll and the falling-notes
  panel now respond to the wheel/trackpad, not just drag. A scrolled view
  holds still for a couple seconds, then eases back to live/now on its own;
  a "⏵ Live" button gives an instant way back. Scrubbing the falling panel is
  purely a preview — it never touches the real playhead or Learn-mode gating.
- **Key range now filters sound, not just gating.** The pitch-range band you
  drag across the falling notes (Learn's "Key range") now actually mutes
  out-of-range notes in both Listen mode's auto-play and Learn mode's
  unpracticed track — previously it only scoped which notes were *required*.
  The readout moved out of the Learn-only panel so it's visible in Listen
  mode too.
- **Collapsible panels.** The top settings panel collapses to a thin title
  strip via a chevron, and the Learn side panel collapses via a "‹"/"›"
  arrow — both to reclaim screen space.

### Changed

- Opening/closing a file no longer pops the layout instantly: the
  falling-notes panel now slides in/out, and the networking controls stay in
  place (greyed out) instead of being replaced by a status line.

## [0.3.0] - 2026-07-04

### Added

- **Piano-roll history.** A paper-roll strip below the keyboard records every
  note both players play — your color and the peer's, black keys thinner and
  darker — with a time ruler (1 s gridlines, `mm:ss` labels every 10 s). The
  roll pauses after 30 s of silence and draws a separator line when play
  resumes, splitting the session into named "instances": rename the current
  one inline (next to the File menu), Ctrl+click (or right-click) either roll
  to insert a break by hand, and drag the strip to review history (it eases
  back to live on release).
- **Save & open rolls.** File ▸ Save (Ctrl+S) writes a standard MIDI file plus
  a tiny color sidecar to `rolls/`; Save As… also offers a self-contained
  JSONL. Instance names are saved as standard MIDI markers, so they show up in
  any DAW. Closing the app with unsaved notes asks first.
- **Playback: Listen & Learn modes.** File ▸ Open loads a saved roll:
  falling notes descend onto the keyboard, auto-played through the built-in
  synth (own volume/mute), with transport (⏮ ⏪ ▶/⏸ ⏩ ⏭ — segment-aware,
  with the restart/previous double-tap convention) and a 0.25×–2× speed
  slider. In Learn mode you play instead: pick which track(s) to practice and
  the piece only advances while you're playing the right notes — strict
  hold-the-notes gating by default, or a wait-for-onset mode; optionally block
  on wrong notes; optionally restrict gating to a key range by dragging across
  the falling notes (refine it with exact solfège names, e.g. Do4–Sol5).
  Practice sessions record onto the live roll like normal play.
- **Segments.** A roll's instances become named segments on playback: rename
  them (persisted in a sidecar without touching the original file), jump
  between them, and loop the current one — indefinitely or N times — with a
  5-second breather between repeats.
- **Mute mic.** A checkbox next to the detection threshold stops mic-detected
  notes from painting the roll (or counting as played keys in Learn mode) —
  handy in noisy rooms.

## [0.2.2] - 2026-07-02

### Added

- **About window.** The status bar now shows a version chip (e.g. `v0.2.2`);
  clicking it opens an About dialog with the running version, live update
  status, and a link to the project. The window title shows the version too.

### Changed

- **Single self-contained exe.** The ML model and ONNX Runtime are now embedded
  inside `open-piano.exe`; the release zip is just the exe plus the README. On
  first launch the app unpacks its runtime to `%LOCALAPPDATA%\open-piano\`
  (self-cleaning across versions). Because updates swap the exe — and the exe
  now contains everything — auto-updates always carry the exactly-matching
  model and runtime; nothing beside the exe can go stale.

## [0.2.1] - 2026-07-02

### Changed

- **Invite codes are ~4× shorter** — 64 characters instead of ~250. The code is
  now just the host's public key; the joiner looks up the host's relay and
  addresses automatically through iroh's discovery service. A host with no
  internet (LAN-only play) still falls back to the long self-contained code,
  and joining accepts both forms — including codes from v0.2.0 hosts.
- Joining now retries for a few seconds with live status ("Not reachable yet,
  retrying…") instead of failing outright, which covers joining immediately
  after the host started.

Note: v0.2.0 can't read the new short codes — if your partner's app says
"Invalid invite code", have them restart it so it auto-updates.

## [0.2.0] - 2026-07-02

### Changed

- **Connecting is now a one-string invite code — no more IPs, ports, or router
  config.** One player clicks **Host session** and sends the copied invite code
  to the other, who pastes it and clicks **Join**. Connections are carried by
  [iroh](https://github.com/n0-computer/iroh): the peers meet through a public
  relay server, hole-punch a direct connection when the networks allow it, and
  fall back to the relay when they don't — so it works behind VPNs, CGNAT, and
  strict NATs with zero setup. Note events still travel as fire-and-forget
  datagrams, so the latency model is unchanged. The old Local Port / Remote IP /
  Remote Port fields are gone.

### Security

- Sessions are end-to-end encrypted and authenticated by the host's key (baked
  into the invite code). The previous transport accepted UDP packets from any
  sender that found the port.

### Fixed

- Notes the peer was holding no longer keep sounding on the built-in synth after
  a disconnect; remote keys and synth voices are released whenever the
  connection state resets.

## [0.1.1] - 2026-07-02

### Fixed

- **v0.1.0 release binary crashed instantly on launch** (no window, no error) on
  most machines. The checked-in `.cargo/config.toml` builds with
  `-C target-cpu=native`, so the CI-built exe contained instructions specific to
  the GitHub Actions runner's server CPU (e.g. AVX-512) and died with
  `STATUS_ILLEGAL_INSTRUCTION` on consumer hardware. Release builds now target
  the portable `x86-64-v2` baseline; local dev builds keep native codegen.

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

[0.1.1]: https://github.com/ja-ortiz-uniandes/open-piano/releases/tag/v0.1.1
[0.1.0]: https://github.com/ja-ortiz-uniandes/open-piano/releases/tag/v0.1.0
