# Changelog

All notable changes to open-piano are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-07-26

### Added

- **Crash diagnostics.** A write-through breadcrumb log
  (`%LOCALAPPDATA%\open-piano\open-piano.log`), a panic hook, and a native
  (`SetUnhandledExceptionFilter`) handler catch the crashes a panic hook can't
  see (access violations, OOM aborts), plus unclean-exit detection via a
  per-pid marker file. Release builds have no console and no OS crash
  dialogs, so "the app just closed" was previously undiagnosable — this is
  now the record. Release builds also publish `open-piano.pdb` as its own
  GitHub Release asset so a reported crash address can be symbolized against
  the exact shipped version. (`src/diag.rs`, `.github/workflows/release.yml`)
- **Persistent network identity and automatic reconnect.** The endpoint's
  iroh identity is now generated once and persisted
  (`Prefs::endpoint_secret`), so the invite code stays the same across
  restarts, crashes, and auto-updates instead of changing every launch.
  A joiner's redial now backs off (capping at 30 s) and never permanently
  gives up — neither on the initial connect nor after an established session
  drops. A "↻ Rejoin" button restores the last session's role and code, and
  an opt-in "auto-reconnect" preference restores it automatically at
  startup (also triggered by a crash-and-relaunch within 5 minutes). Edit ▸
  Preferences ▸ Networking ▸ "Reset my identity" clears the persisted key.
  (`src/net.rs`, `src/prefs.rs`, `src/main.rs`)
- **Both invite-code forms are kept live simultaneously.** The host now
  publishes the short code (once n0 discovery confirms it can serve it) and
  the long, self-contained ticket concurrently from the moment the endpoint
  binds, upgrading and re-publishing on a heartbeat for as long as hosting
  continues, instead of only ever offering one fallback form. (`src/net.rs`)

### Changed

- **Peer color now actually syncs.** The peer's announced color
  (`Packet::Color`) is applied to `remote_color` and rendered for real,
  instead of being silently discarded in favor of a fixed placeholder — a
  regression from v0.7.0 that meant the same note could render as a
  different color on each player's screen. The one deliberate asymmetry: if
  both sides are still at the untouched default color, the *joiner* alone
  switches to a distinguishable fallback (amber) and re-announces it, so two
  un-customized installs still render distinctly — computed identically on
  both ends and never overriding a manual color choice. (`src/main.rs`)
- **Touch-drag stability.** Title-bar and resize-handle dragging now follows
  the actual touch contact point and clamps runaway speed instead of
  occasionally teleporting the window, fixing artifacts on touch hardware
  the app hadn't previously been exercised on. (`src/main.rs`)
- **A dropped net thread can no longer leave a zombie session.** Whether it
  panics or fails to spawn, it now always reports through
  `NetEvent::Disconnected` (a drop guard, plus explicit handling when the
  event channel itself closes), so the UI can never keep holding a stale
  `Peer` handle or a permanently-stuck remote key. (`src/net.rs`)

## [0.8.2] - 2026-07-22

### Changed

- **Looped segments now scroll seamlessly in the falling roll.** Looping a
  segment (Listen or Learn mode) used to freeze the falling roll at the
  segment's end through the silent breather, then snap it back to the start.
  The panel now tracks a continuous view time that keeps advancing through
  the pad instead of the frozen playhead, and draws the looping segment's
  notes twice — once for real and once as a "ghost" copy one repeat-period
  ahead — so the next pass is already falling into view as the current one
  ends, and the wrap reads as continuous motion rather than a jump. Segment
  lines become per-repeat restart lines while looping. The keyboard also
  stays dark through the pad, so a note that ended exactly on the loop
  boundary no longer stays lit through the whole breather. Evaluation mode
  never loops, so it's unaffected. (`src/playback.rs`, `src/main.rs`)

## [0.8.1] - 2026-07-19

### Changed

- **Auto-update no longer installs silently.** A newer release now only
  surfaces as available; the download and exe swap happen exclusively after
  an explicit "Install" click. `self_update` verifies nothing about the
  downloaded payload beyond the TLS connection, so silently swapping the
  exe at every launch meant anyone able to publish a GitHub release could
  push arbitrary code to every installed copy with no user interaction.
  (`src/update.rs`, `src/main.rs`)
- **Persistent network problems are now visible.** A datagram send failure
  used to only print to a console release builds don't have. The status
  line now reports a sustained failure once (not every frame, and only when
  the failure kind changes) and clears itself once sending recovers, instead
  of silently showing "Connected" while some or all traffic has stopped
  flowing. A stalled or malicious incoming handshake no longer pins the net
  thread past a UI shutdown either. (`src/net.rs`)
- **Metronome click-table sync no longer fights itself.** The host's
  authoritative table now anchors to the connection itself and ignores
  stale echoes while a local edit is in flight, so dragging a pitch/volume
  slider can no longer be stomped by a heartbeat that hasn't caught up, and
  a follower's edit request can't be silently reverted by the host's own
  periodic re-broadcast.
- **Recorder self-heals through more failure modes.** The WAV header, MIDI
  log, and `meta.json` are now flushed on a roughly 1-second cadence (and
  `meta.json` is written up front), so a kill mid-session leaves an
  aligned, playable capture instead of a truncated one with no metadata.
  Recording now also stops cleanly before hound's internal WAV byte counter
  can wrap past 4 GiB (~5.8 h at 48 kHz) instead of corrupting the header,
  and queued-but-unwritten audio is capped so a stalled disk can't grow
  memory without bound. (`src/record.rs`)
- **Preferences sanitize hostile or corrupted values on load.** A garbled
  window size, echo hold-off, pedal deadzone, click frequency/level, or a
  hand-edited/truncated network identity is now clamped or dropped to a safe
  default at load instead of propagating NaNs or permanently breaking a
  feature; an unparseable preferences file is backed up to `.json.bak`
  before falling back to defaults, instead of being silently overwritten by
  the next save. (`src/prefs.rs`, `src/note.rs`)
- **Assorted playback/recording edge cases fixed**, found in a full-tree
  review: notes are matched against the evaluation window before the expiry
  sweep runs in the same frame (a same-frame hit no longer scores as a
  miss), loaded notes are clamped to a minimum audible length, a quick
  Stop→Record within one supervisor poll interval no longer merges two
  separate takes into a single session directory, and a saved `.jsonl`'s
  pedal/velocity values are sanitized before they're written rather than
  after. (`src/playback.rs`, `src/score.rs`, `src/input.rs`, `src/record.rs`)

## [0.8.0] - 2026-07-18

A hardening pass resolving 30 of 31 findings from a full-tree code review
(one, update-integrity via checksums, is deferred — tracked separately and
partly addressed in v0.8.1's consent gate).

### Changed

- **Shared-surface sync hardened against packet loss and reordering.**
  Manually-inserted segment breaks (Ctrl+click) are now reconciled against
  the roll before every broadcast and dropped if their timestamp is in the
  future rather than clamped-and-accepted, closing a bug where a lost ack
  could cause a break to be resent once a second forever. The metronome's
  per-beat pitch/volume table is now strictly host-authoritative (a
  follower's edit is adopted and re-broadcast by the host, never sent
  directly peer-to-peer), preventing the two sides from ever swapping or
  oscillating tables, and its on/off state now rides the same heartbeat so a
  dropped toggle self-heals. Note-related packets (including the `Live`/
  `Held` whole-state snapshots) now carry a per-sender sequence number, so a
  snapshot delivered out of order (a real risk during iroh's relay→direct
  migration) can no longer resurrect an already-released note or extinguish
  a fresh press. Persistent datagram send failures are now surfaced instead
  of only logged. (`src/net.rs`, `src/note.rs`, `src/main.rs`)
- **Input supervisor no longer thrashes on a flaky device.** A dying mic or
  record-capture stream now cools down between restart attempts instead of
  reloading the ONNX model on every ~1 s poll, and the ONNX session is only
  built once the stream format actually validates. Enumerating MIDI ports
  now keeps trying the rest of the list if one port fails to open, instead
  of a bad port 0 starving a working port 1. (`src/audio.rs`, `src/input.rs`)
- **Recorder hardened against crashes, huge sessions, and slow disks.** The
  WAV header, MIDI log, and `meta.json` are flushed on a steady cadence so a
  kill leaves an aligned session; recording stops cleanly before the WAV
  format's internal size limit; the audio-start alignment anchor is refined
  over the stream's first buffers instead of trusting a single, possibly
  jittery, first callback; and queued-but-unwritten audio is now bounded so
  a stalled disk can't grow memory indefinitely. (`src/record.rs`)
- **Preferences saves are now durable and debounce-safe.** The temp file is
  fsync'd before the atomic rename; window size, echo hold-off, and pedal
  deadzone are sanitized on load; and a pending debounced save is flushed
  before the app exits or restarts for an update, so a setting changed right
  before closing is no longer silently lost. (`src/prefs.rs`, `src/bundle.rs`)
- **Playback and scoring edge cases fixed.** Presses outside the 88-key
  range are ignored by the scorer instead of panicking or under/overflowing;
  pause-on-miss can now gate the final note of a take instead of letting it
  slip through free-run; loaded MIDI/JSONL notes and pedal events are sorted
  before use; Evaluation review can no longer be entered with no track
  actually evaluated; and velocity/pedal values from a wire packet are
  saturated to their valid range instead of silently wrapping.
  (`src/playback.rs`, `src/score.rs`)
- **Window chrome edge cases fixed.** The maximized flag is cleared before a
  compact-mode resize (a maximized rect could otherwise get snapshotted as
  the "restore to normal" size), "Restart now" from the update prompt now
  goes through the same unsaved-roll confirmation as closing the window
  instead of bypassing it, and the resize handles no longer overlap the
  title-bar strip. (`src/main.rs`)

## [0.7.3] - 2026-07-11

### Fixed

- **Ctrl-pinned key display is now memoryless.** Releasing Ctrl clears the
  pinned-key set instead of merely hiding it, so a chord pinned during one
  Ctrl-hold no longer resurfaces the next time Ctrl is pressed for something
  unrelated — each hold now starts from a clean slate, like a
  file-explorer-style Ctrl+click toggle rather than a persistent pin.
  (`src/main.rs`)

## [0.7.2] - 2026-07-11

### Fixed

- **Dragging the title bar to move the window no longer jitters.** The
  full-width top-edge resize handle (added for touch support in v0.7.0) sat
  above the title bar and intercepted move-drags that started near the top,
  fighting the move logic frame to frame. That handle is removed; top
  resizing stays available from the NW/NE corner handles, leaving the whole
  title bar exclusively for window moves. (`src/main.rs`)

## [0.7.1] - 2026-07-10

### Fixed

- **Window-move dragging is smooth and jitter-free.** v0.7.0's touch-friendly
  rewrite drove the frameless window by accumulating `drag_delta()` onto a
  running target, but that delta is measured in window-local coordinates —
  as the window moved under the pointer, the local pointer position shifted
  the opposite way, flipping the delta's sign the next frame and making the
  window oscillate (with mouse as well as touch). The window's target
  position is now solved absolutely each frame from the pointer's grab
  offset, so a one-frame lag in the reported rect self-corrects instead of
  compounding. (`src/main.rs`)
- **Ctrl-held keys now hide the instant Ctrl is released.** Ctrl+click-pinned
  keys were staying highlighted after the modifier came up; the render is
  now gated on the modifier being currently down. (`src/main.rs`)

## [0.7.0] - 2026-07-10

### Added

- **Live pedal indicator.** A thin sliver to the left of the keyboard (shown
  in both normal and compact mode) renders both players' current
  sustain-pedal depth, diagonally split the same way a simultaneous same-key
  press is — visible even on mic input, so the peer's pedal use stays legible
  when local input has no pedal signal at all. (`src/main.rs`)
- **Ctrl+click to pin keys for display.** Ctrl+click lights a key in your own
  color without playing it — a way to point at a chord while explaining
  something — gated by the same recording/evaluation/playback lock as
  mouse-play. Purely a local display aid: it never touches the synth, the
  roll, or the peer. (`src/main.rs`)
- **Touch-friendly window chrome.** Moving and resizing the frameless window
  is now driven manually every frame from egui's drag deltas
  (`ViewportCommand::OuterPosition`/`InnerSize`) instead of handing off to
  Windows' native move/resize loop, which doesn't sustain a
  touch-originated gesture — the cause of broken touch-move. The resize
  handles also enlarge to a visible, grabbable affordance in actual tablet
  use (detected from real touch events vs. mouse/trackpad), while staying
  tight and invisible for a mouse/trackpad session. (`src/main.rs`)
- **"Keep compact window on top" preference** (default on): while in compact
  mode the window floats above other applications, reconciled live so
  toggling it takes effect immediately without leaving compact mode.
  (`src/prefs.rs`, `src/main.rs`)

### Changed

- **Two un-customized peers no longer render identically.** The peer's
  announced color is no longer applied locally — the peer is always drawn
  in a fixed default (blue) regardless of what it announces — because every
  fresh install previously defaulted both sides to the same red, making a
  simultaneous press indistinguishable from a single one. (Superseded by a
  real per-peer color sync with a narrower fallback in v0.9.0.) (`src/main.rs`)
- **Mic input is muted by default on a fresh install**, so a new install
  doesn't start transcribing ambient audio before the user opts in.
  (`src/prefs.rs`)

## [0.6.1] - 2026-07-08

### Added

- **Drag-to-resize the keyboard.** The keyboard's top and bottom edges grow
  a thin invisible drag handle; dragging resizes the keyboard as a fraction
  of the central panel's height, persisted on release
  (`prefs.keyboard_height_frac`). The bottom handle works whenever not in
  compact mode; the top handle whenever the falling-notes panel is showing.
  (`src/main.rs`, `src/prefs.rs`)

### Changed

- **Color picker reverted to the stock popup swatch.** v0.6.0's always-inline
  hue/brightness/saturation picker permanently occupied space in the config
  bar and Preferences; both entry points are back to the click-to-open
  `color_edit_button_srgb` swatch. (`src/main.rs`)
- **Pedal lane and deadzone are now always editable in Preferences**, not
  only once a MIDI keyboard is connected, so they can be dialed in ahead of
  time — with a note that they have no effect on mic input. The lane's own
  MIDI-only render gate is unchanged. (`src/main.rs`)

## [0.6.0] - 2026-07-08

### Added

- **Pause-on-miss (Evaluation mode).** An optional setting that freezes the
  playhead at a missed note's tolerance-window edge until the note is
  actually struck, instead of scoring it a miss and free-running past it.
  The result card reports total frozen time and how many distinct freezes
  occurred. (`src/playback.rs`)
- **Tunable section breaks.** What was a single idle-pause/trailing-blank
  pair is now a break threshold plus two independently configurable margins:
  a tail kept after a section's last note, and a lead-in before the next
  section's first note. A status-bar chip ("● section break on next
  keypress") shows when a break is pending on the roll. (`src/roll.rs`,
  `src/prefs.rs`, `src/main.rs`)
- **Pedal sensitivity deadzone.** A new Preferences setting filters small
  mid-travel CC64 jitter from analog pedals without losing half-pedal moves
  or the fully-open/closed edges. (`src/main.rs`, `src/prefs.rs`)
- **Tabbed Preferences.** The dialog is reorganized from one long scrolling
  page into eight sidebar-navigated sections (Startup & window, Roll &
  history, Appearance, Pedal, Roll behavior, Audio/mic, Metronome,
  Advanced), and is now resizable and freely movable instead of pinned to
  screen center. (`src/main.rs`)

### Changed

- **Compact mode remembers its restore size.** The last normal-size window
  dimensions now persist (`prefs.normal_window_size`), so relaunching
  straight into compact mode restores to the right size instead of a
  mismatched default height; opening Preferences no longer force-expands a
  compact window — the dialog simply floats over it. (`src/main.rs`,
  `src/prefs.rs`)
- **Default blank-paper margin shrinks from 20 s to 2 s** now that section
  breaks trim the tail themselves — a pause longer than the (default 30 s)
  break threshold now leaves a tidy 2-second tail rather than up to 20
  seconds of dead paper. (`src/roll.rs`, `src/prefs.rs`)

## [0.5.0] - 2026-07-07

### Added

- **Evaluation mode.** A third playback mode alongside Listen/Learn: one score
  track goes silent while you play it live and the take is scored against the
  original (timing, and — MIDI input only — velocity and sustain-pedal use),
  with Strict/Normal/Lenient presets or custom tolerances. The playhead never
  gates in this mode; the run always completes and can't freeze. On take
  completion the engine flips into **Evaluation review**: a passive replay of
  a synthetic two-track score (original vs. what you actually played) with
  independent show/hear toggles per side, plus an "Evaluation results" window
  (score, streaks, best/worst pitches). (`src/playback.rs`, `src/main.rs`)
- **MIDI note velocity.** Real note-on velocity now flows end to end — MIDI
  input → `NoteMsg` → the wire protocol → the history roll — and tints each
  roll mark's saturation by how hard the key was struck. Older peers /
  velocity-less sources (mouse clicks, the mic path) fall back to a flat
  default rather than being rejected. (`src/note.rs`, `src/midi.rs`,
  `src/roll.rs`)
- **Sustain-pedal lane.** CC64 pedal activity is captured, drawn as a slim
  tinted strip at the roll's left edge (including half-pedaling as adjacent
  spans of differing depth), synced to the peer, and written into saved
  MIDI/jsonl sessions. Structurally MIDI-only end to end — the mic backend is
  never wired to the pedal channel, so the toggle and lane simply aren't
  present on that path. (`src/midi.rs`, `src/input.rs`, `src/roll.rs`,
  `src/note.rs`)

### Changed

- Loaded scores now also carry per-note velocity and per-track pedal streams
  (from SMF CC64 / jsonl), used only by Evaluation's scorer. (`src/score.rs`)

## [0.4.1] - 2026-07-05

### Added

- **Metronome click volume.** A dedicated volume slider next to "Mute click",
  independent of mute — matches the existing screen/peer/playback pattern.
- **Per-beat metronome pitch and volume tables (Preferences ▸ Metronome).**
  Beats per bar is now configurable (1–12), and each beat's click pitch and
  level are independently editable — beat 1 is the accent/downbeat, styled
  with a subtle background tint. A quick-pick slider snaps the pitch to one
  of four common presets before the precise Hz field; a Reset button restores
  the defaults. Both tables are synced with the peer (no host authority —
  whoever edits last wins on both ends), so the two players' clicks sound
  identical, not just land on the same beat.

### Changed

- **Metronome beats are grid-aligned, not free-run from "start".** Beat 0 of
  the metronome's grid always sits at the history roll's time zero, so
  pressing start mid-beat waits for the next round position (e.g. every `bpm`
  beats lands on a whole minute of roll time) instead of clicking immediately
  wherever you happened to press play. Once running it free-runs normally, so
  tempo tweaks don't cause an audible jump — only a fresh start re-snaps to
  the grid.
- **The falling-notes panel's ruler now reads on the same absolute timeline as
  the history roll below it.** Opening a file records where the history
  roll's clock currently sits as the score's time-zero, so the two strips'
  `mm:ss` labels line up continuously across the keyboard instead of each
  panel numbering from its own zero. Only the *printed labels* shift — note
  positions, Learn-mode gating, and looping are unaffected. The history roll
  stays the ground truth: it can run longer than the (fixed-length) score.

## [0.4.0] - 2026-07-05

### Added

- **Preferences dialog (Edit ▸ Preferences, Ctrl+,).** The app's scattered
  compile-time tunables are now editable and **persisted** across restarts to
  `%LOCALAPPDATA%\open-piano\preferences.json`. Sections: roll timing,
  appearance (default note color), audio/mic (threshold, echo hold-off, mute
  default), and an **Advanced** expander (collapsed, with a Reset) for the
  detector's silence/normalization/release knobs and MIDI-poll interval — all
  live-editable while you play. Older/partial preference files load without
  error. (`src/prefs.rs`)
- **Custom window title bar.** The OS chrome is replaced by our own title bar:
  File/Edit menus on the left (always the topmost row, independent of the
  settings panel's collapse state), the title centered, and minimize/maximize/
  close on the right. The bar drags the window, double-click maximizes/restores,
  and the window edges resize. (Windows 11 snap-layouts aren't available with
  custom chrome — an accepted trade-off.) (`src/main.rs`)
- **Synced metronome.** A shared click both players hear together. The host is
  the timing authority and broadcasts a beat marker each beat; a guest anchors a
  local click schedule to those markers (corrected by half the measured RTT), so
  clicks are generated locally and packet loss never drops or delays one. Either
  player can set the tempo (30–240 BPM) or start/stop — a guest's change is a
  request the host adopts and echoes back (one grid, last-writer-wins). Each
  player can mute *their own* click locally without affecting the peer's. Solo,
  it's a plain local metronome. (`src/synth.rs`, `src/note.rs`, `src/net.rs`,
  `src/main.rs`)

### Changed

- **Roll timing preserves real silence.** The trailing-blank cap now defaults
  to **20 s** (was 2 s), so a pause in your playing shows as roughly that much
  blank paper instead of snapping to a couple seconds. Both the trailing-blank
  cap and the idle-pause threshold (default 30 s) are now Preferences settings,
  and each can be switched to **∞** — no clamp / no auto-pause — for a truly
  unbounded gap (which needs both set to ∞). (`src/roll.rs`)
- The File/Edit menus moved out of the collapsible settings panel into the new
  title bar; the instance-rename field and save/open status stay in the panel.

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
