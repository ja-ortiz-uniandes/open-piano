//! Piano-roll history: a record of every note either player has played this
//! session, on a pausable "roll clock", plus save-to-disk.
//!
//! The roll is the data behind the paper-roll strip drawn under the keyboard
//! (see `draw_roll` in main.rs): each key press opens a [`Segment`] at the
//! current roll time and closes it on release. Gaps in the playing are shown
//! faithfully up to a section-break threshold ([`DEFAULT_IDLE_PAUSE`]) of
//! inactivity; once crossed, the clock formally pauses and the blank tail is
//! trimmed to [`DEFAULT_SECTION_TAIL`], so the next note opens a new
//! "instance" (separated in the record) a [`DEFAULT_SECTION_LEAD_IN`] margin
//! past the boundary — which also bounds the data and the on-screen paper.
//!
//! Saving writes an open, universally-readable Standard MIDI File (format 1,
//! one track per player, separators as SMF marker meta-events) plus a tiny
//! JSON sidecar for the one thing SMF cannot express: each player's display
//! color. Alternatively the whole roll (colors included) can be exported as a
//! single self-describing JSONL file.

use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::note::{midi_to_key_index, NoteMsg, KEY_COUNT};

/// Default section-break threshold: idle time (no note on/off from either
/// player, and no key held) after which the roll clock formally *pauses* —
/// trimming the blank tail to the section tail — so the next note opens a new
/// "instance" (a separator in the record). Gaps *shorter* than this are shown
/// faithfully: a 25 s breath leaves ~25 s of blank paper, because that is how
/// the phrase was played. A live, per-`Roll` setting (see
/// [`Roll::set_timing`]) seeded from user preferences; `None` disables the
/// auto-break entirely.
const DEFAULT_IDLE_PAUSE: Duration = Duration::from_secs(30);

/// Default blank margin kept after a section's last note when the break
/// fires: the clock snaps back to last-note + this, trimming the dead air the
/// threshold let accrue. The same snap trims trailing silence at the very end
/// of a session. See [`Roll::set_timing`].
const DEFAULT_SECTION_TAIL: Duration = Duration::from_secs(2);

/// Default blank margin inserted between a section boundary and the first
/// note of the section that resumes there, so a new section never starts
/// flush against its separator line. See [`Roll::set_timing`].
const DEFAULT_SECTION_LEAD_IN: Duration = Duration::from_secs(2);

/// Which player produced a segment. Doubles as an index (0 = local,
/// 1 = remote) into per-player tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Who {
    Local,
    Remote,
}

impl Who {
    pub(crate) fn idx(self) -> usize {
        match self {
            Who::Local => 0,
            Who::Remote => 1,
        }
    }
}

/// One "instance" boundary — where the roll clock resumed after an idle
/// pause (or where the user inserted a break by hand). `name: None`
/// renders/saves as an auto-generated "instance N".
pub struct Instance {
    pub at: f64,
    pub name: Option<String>,
}

/// One held note on the roll. Times are roll-clock seconds (see [`Roll::tick`]).
pub struct Segment {
    pub who: Who,
    pub midi: u8,
    pub start_s: f64,
    /// `None` while the key is still held; the renderer extends the mark to
    /// the live edge and a save synthesizes an off at save time.
    pub end_s: Option<f64>,
    /// The player's color at note-on, so changing a color mid-session doesn't
    /// repaint history. (A save persists only the *current* two player colors;
    /// mid-session color changes are a live-view nicety, not saved fidelity.)
    pub color: [u8; 3],
    /// Note-on velocity (1..=127). Real dynamics on the MIDI path; the flat
    /// [`crate::note::DEFAULT_VELOCITY`] placeholder everywhere else. Drives
    /// the roll's velocity→saturation tint and is written into saves.
    pub velocity: u8,
}

/// One sustain-pedal press on the roll, deliberately kept apart from the note
/// [`Segment`]s and never routed through [`Roll::note`]: pedal activity is
/// recorded, drawn, and saved, but must be structurally unable to reset the
/// idle timer that auto-pauses the clock (see [`Roll::pedal`]).
pub struct PedalSegment {
    pub who: Who,
    pub start_s: f64,
    /// `None` while the pedal is still down (the renderer extends the mark to
    /// the live edge; a save synthesizes the release, like open notes).
    pub end_s: Option<f64>,
    /// CC64 level over this span (1..=127). A depth change while down closes
    /// the span and opens a new one, so half-pedaling is preserved.
    pub level: u8,
}

/// The roll: an append-only list of note segments on a pausable clock.
///
/// Memory is deliberately uncapped: segments accrue at human playing speed
/// (~40 bytes each; two players at a frantic 10 notes/s combined for four
/// hours is ~144k segments ≈ 6 MB), and IDLE_PAUSE bounds the blank paper.
pub struct Roll {
    /// Every segment ever recorded, in note-on order (`start_s` is monotonic).
    pub segments: Vec<Segment>,
    /// Index into `segments` of the still-open segment per (player, key).
    open: [[Option<usize>; KEY_COUNT]; 2],
    /// How many segments are currently open (held keys, both players).
    /// Deliberately notes-only: a held *pedal* must not keep the clock alive.
    open_count: usize,
    /// Sustain-pedal history, parallel to `segments` (see [`PedalSegment`]).
    pub pedal_segments: Vec<PedalSegment>,
    /// Index into `pedal_segments` of the still-open span per player.
    open_pedal: [Option<usize>; 2],
    /// Last seen CC64 level per player, for no-op dedupe on repeated values.
    pedal_level: [u8; 2],
    /// Boundaries where the clock resumed after a pause (or a break was
    /// inserted by hand) — drawn as full-width lines separating "instances"
    /// of play, kept sorted by time. These delimit instance 2, 3, ...
    pub separators: Vec<Instance>,
    /// Name for instance 1 (everything before the first separator) — kept
    /// separately since there's no `Instance` entry to attach it to.
    pub first_instance_name: Option<String>,
    /// The paper position: seconds of roll time elapsed while unpaused.
    roll_now_s: f64,
    /// Wall time of the last `tick`, used both to advance the clock and as
    /// "now" for events (events arrive on the same thread between ticks, so
    /// they are at most one frame stale — imperceptible at repaint rate).
    last_frame: Instant,
    /// Wall time of the last note on/off, for idle detection.
    last_event: Option<Instant>,
    /// Roll time (`roll_now_s`) at the last note on/off. When the break
    /// fires, the clock snaps back to this plus [`Roll::section_tail`].
    last_event_roll_s: f64,
    /// Blank margin kept after a section's last note when the break fires.
    /// See [`Roll::set_timing`] and [`DEFAULT_SECTION_TAIL`].
    section_tail: Duration,
    /// Blank margin before the first note of a resumed section. See
    /// [`Roll::set_timing`] and [`DEFAULT_SECTION_LEAD_IN`].
    section_lead_in: Duration,
    /// Section-break threshold: idle window before the clock auto-pauses
    /// (`None` = never break). See [`Roll::set_timing`] and
    /// [`DEFAULT_IDLE_PAUSE`].
    idle_pause: Option<Duration>,
    /// Starts paused: no paper accumulates before the first note.
    paused: bool,
    /// Set on every note event, cleared by `mark_saved`.
    dirty: bool,
}

impl Roll {
    pub fn new() -> Self {
        Roll {
            segments: Vec::new(),
            open: [[None; KEY_COUNT]; 2],
            open_count: 0,
            pedal_segments: Vec::new(),
            open_pedal: [None; 2],
            pedal_level: [0; 2],
            separators: Vec::new(),
            first_instance_name: None,
            roll_now_s: 0.0,
            last_frame: Instant::now(),
            last_event: None,
            last_event_roll_s: 0.0,
            section_tail: DEFAULT_SECTION_TAIL,
            section_lead_in: DEFAULT_SECTION_LEAD_IN,
            idle_pause: Some(DEFAULT_IDLE_PAUSE),
            paused: true,
            dirty: false,
        }
    }

    /// Update the section timing live: the tail/lead-in margins and the break
    /// threshold (`idle_pause`; `None` = never auto-break, one continuous
    /// instance). Seeded from user preferences at startup and re-applied
    /// whenever they change.
    pub fn set_timing(
        &mut self,
        section_tail: Duration,
        section_lead_in: Duration,
        idle_pause: Option<Duration>,
    ) {
        self.section_tail = section_tail;
        self.section_lead_in = section_lead_in;
        self.idle_pause = idle_pause;
    }

    /// Advance the roll clock. Call once per frame, before rendering.
    pub fn tick(&mut self, now: Instant) {
        if !self.paused {
            // NOTE (L11): deliberately *not* clamped. Unlike the score playhead
            // (a fixed-length piece), the roll is a live recorder whose clock
            // tracks real elapsed wall time — a key held for minutes must scroll
            // the paper for those minutes (see `held_key_keeps_the_clock_running`).
            // The idle-pause auto-break already bounds blank paper when nothing
            // is held; clamping here would break that intended behavior.
            self.roll_now_s += now.saturating_duration_since(self.last_frame).as_secs_f64();
            // Freeze only when nothing is held: a key held for minutes is
            // still an extending mark, and the paper must keep moving under it.
            if self.open_count == 0 {
                if let Some(last) = self.last_event {
                    // Real elapsed time displays faithfully right up to the
                    // break threshold — a 25 s breath leaves 25 s of paper.
                    // Once crossed, the clock pauses (→ new instance on the
                    // next note) and snaps back to the section tail, trimming
                    // the dead air to a tidy margin. The same snap trims
                    // trailing silence at the very end of a session — no
                    // separate "last note" detection needed. Infinite (`None`)
                    // never auto-breaks, so play stays one instance.
                    if let Some(idle) = self.idle_pause {
                        if now.saturating_duration_since(last) > idle {
                            self.paused = true;
                            self.roll_now_s =
                                self.last_event_roll_s + self.section_tail.as_secs_f64();
                            // The snap-back rewinds the live edge into trimmed
                            // dead air: any separator the user inserted out
                            // there (e.g. a Ctrl+click near the pre-snap edge)
                            // now sits *beyond* the clock. Drop those — they'd
                            // otherwise leave `separators` unsorted (a resume
                            // pushes the new boundary at the rewound clock) and
                            // underflow the SMF delta math in `save_midi`.
                            self.separators.retain(|s| s.at <= self.roll_now_s);
                            // Pedal spans (unlike notes) can accrue during the
                            // trimmed dead air, since pedal traffic deliberately
                            // doesn't reset the idle timer. Clamp any that
                            // started or ended beyond the rewound clock back
                            // onto it — in place, so the open-span indices stay
                            // valid: a still-held pedal keeps showing from the
                            // new edge, phantom marks stop overlaying the next
                            // section, and `save_midi` never emits a release
                            // before its press (end >= start always holds).
                            let clock = self.roll_now_s;
                            for p in self.pedal_segments.iter_mut() {
                                if p.start_s > clock {
                                    p.start_s = clock;
                                }
                                if let Some(e) = p.end_s.as_mut() {
                                    if *e > clock {
                                        *e = clock;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        self.last_frame = now;
    }

    /// Record a note transition. `color` is that player's current color.
    pub fn note(&mut self, who: Who, msg: NoteMsg, color: [u8; 3]) {
        let Some(key) = midi_to_key_index(msg.midi()) else {
            return;
        };
        match msg {
            NoteMsg::On(midi, velocity) => {
                // Idempotent, like main.rs's `apply()`: the mic path's
                // hysteresis (or a reconnect race) can emit On twice.
                if self.open[who.idx()][key].is_some() {
                    return;
                }
                if self.paused {
                    self.paused = false;
                    // A resume after a pause starts a new "instance"; the
                    // very first note of the session gets no separator. The
                    // boundary sits at the tail-trimmed clock (see `tick`);
                    // the lead-in then pushes this first note a fixed margin
                    // past the separator line.
                    if !self.segments.is_empty() {
                        // Sorted insert, not a bare push: a manual separator can
                        // outlive the snap-back only when it's <= the rewound
                        // clock (tick drops the rest), so `separators` stays
                        // sorted for rendering and for `save_midi`'s delta math.
                        let pos = self.separators.partition_point(|s| s.at < self.roll_now_s);
                        self.separators.insert(pos, Instance { at: self.roll_now_s, name: None });
                        self.roll_now_s += self.section_lead_in.as_secs_f64();
                    }
                    // Open a span at this resume time for any pedal a player was
                    // holding across the pause, whose opening we deferred (R39).
                    for w in [Who::Local, Who::Remote] {
                        if self.pedal_level[w.idx()] > 0 && self.open_pedal[w.idx()].is_none() {
                            self.open_pedal[w.idx()] = Some(self.pedal_segments.len());
                            self.pedal_segments.push(PedalSegment {
                                who: w,
                                start_s: self.roll_now_s,
                                end_s: None,
                                level: self.pedal_level[w.idx()],
                            });
                            self.dirty = true;
                        }
                    }
                }
                self.open[who.idx()][key] = Some(self.segments.len());
                self.open_count += 1;
                self.segments.push(Segment {
                    who,
                    midi,
                    start_s: self.roll_now_s,
                    end_s: None,
                    color,
                    velocity,
                });
            }
            NoteMsg::Off(_) => {
                let Some(i) = self.open[who.idx()][key].take() else {
                    return;
                };
                self.segments[i].end_s = Some(self.roll_now_s);
                self.open_count -= 1;
            }
        }
        self.last_event = Some(self.last_frame);
        self.last_event_roll_s = self.roll_now_s;
        self.dirty = true;
    }

    /// Close all of `who`'s open segments at the current roll time. Called at
    /// the same points the UI force-releases stuck keys (input-backend epoch
    /// switch, peer connect/disconnect), where the matching note-offs will
    /// never arrive.
    pub fn release_all(&mut self, who: Who) {
        for slot in self.open[who.idx()].iter_mut() {
            if let Some(i) = slot.take() {
                self.segments[i].end_s = Some(self.roll_now_s);
                self.open_count -= 1;
                self.last_event = Some(self.last_frame);
                self.last_event_roll_s = self.roll_now_s;
                self.dirty = true;
            }
        }
    }

    /// Record a sustain-pedal (CC64) level for `who`. Opens a span on
    /// 0 → non-zero, closes it on → 0, and closes + reopens on a depth change
    /// so half-pedaling is preserved as adjacent spans; repeated identical
    /// levels are no-ops.
    ///
    /// Deliberately does NOT touch `last_event`/`last_event_roll_s` (unlike
    /// [`Roll::note`]), and pedal spans don't count into `open_count`: holding
    /// or pumping the pedal alone must never keep the idle clock alive or
    /// delay the auto-pause. This is load-bearing — see the test
    /// `pedal_never_resets_the_idle_timer`.
    pub fn pedal(&mut self, who: Who, level: u8) {
        if level == self.pedal_level[who.idx()] {
            return;
        }
        self.pedal_level[who.idx()] = level;
        // While the roll clock is frozen (paused after an idle snap-back, or
        // before the first note), every level change would open/close a span at
        // the same frozen `roll_now_s`, piling up as stacked zero-length spans —
        // a CC64 blip storm in the saved file (R39). Deliberately, pedal traffic
        // never unpauses the clock (unlike `note`). So while paused: just track
        // the level, close an open span on release, and defer *opening* a new
        // span to the resume path (`note`), which opens it at the real resume
        // time for any pedal still held.
        if self.paused {
            if level == 0 {
                if let Some(i) = self.open_pedal[who.idx()].take() {
                    self.pedal_segments[i].end_s = Some(self.roll_now_s);
                    self.dirty = true;
                }
            }
            return;
        }
        if let Some(i) = self.open_pedal[who.idx()].take() {
            self.pedal_segments[i].end_s = Some(self.roll_now_s);
        }
        if level > 0 {
            self.open_pedal[who.idx()] = Some(self.pedal_segments.len());
            self.pedal_segments.push(PedalSegment {
                who,
                start_s: self.roll_now_s,
                end_s: None,
                level,
            });
        }
        self.dirty = true;
    }

    /// Force-close any open pedal span for `who` — the pedal analogue of
    /// [`Roll::release_all`], for the same call sites (input-backend epoch
    /// switch, peer connect/disconnect), where the matching pedal-up will
    /// never arrive. Also never touches the idle timer.
    pub fn release_pedal(&mut self, who: Who) {
        if let Some(i) = self.open_pedal[who.idx()].take() {
            self.pedal_segments[i].end_s = Some(self.roll_now_s);
            self.dirty = true;
        }
        self.pedal_level[who.idx()] = 0;
    }

    /// The current roll time — the live (top) edge of the paper.
    pub fn now_s(&self) -> f64 {
        self.roll_now_s
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Whether the roll clock is formally paused — true from the moment the
    /// section-break threshold fires until the next note resumes it, i.e.
    /// exactly the "a break will be auto-inserted on the next keypress"
    /// condition the status indicator renders. (Also true before the first
    /// note; gate on [`Roll::is_empty`] where that matters.)
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// True when there are notes on the roll that haven't been saved since
    /// the last change — drives the "● unsaved" chip and the close warning.
    pub fn has_unsaved(&self) -> bool {
        self.dirty && !self.segments.is_empty()
    }

    pub fn mark_saved(&mut self) {
        self.dirty = false;
    }

    /// Which instance the roll clock is currently in: 0 while still in the
    /// first, 1 once the second has started, etc.
    pub fn current_instance(&self) -> usize {
        self.separators.iter().filter(|i| i.at <= self.roll_now_s).count()
    }

    /// Display name of the current instance — the custom name if one was set,
    /// else the same "instance N" auto-name the save path writes.
    pub fn current_instance_name(&self) -> String {
        let idx = self.current_instance();
        let name = if idx == 0 {
            self.first_instance_name.clone()
        } else {
            self.separators.get(idx - 1).and_then(|s| s.name.clone())
        };
        name.unwrap_or_else(|| format!("instance {}", idx + 1))
    }

    /// Rename whichever instance the roll is currently in. Counts as an
    /// unsaved change, same as a new note.
    pub fn rename_current_instance(&mut self, name: String) {
        let idx = self.current_instance();
        if idx == 0 {
            self.first_instance_name = Some(name);
        } else if let Some(sep) = self.separators.get_mut(idx - 1) {
            sep.name = Some(name);
        }
        self.dirty = true;
    }

    /// Insert a boundary at `at` (roll-clock seconds), splitting whatever
    /// instance currently spans that time. No-op at/before 0 or exactly on an
    /// existing boundary (avoids duplicate/zero-length instances). Unlike the
    /// auto-pause boundaries, this can land anywhere in already-recorded
    /// history, not just at the live edge.
    pub fn insert_separator(&mut self, at: f64) {
        let at = at.clamp(0.0, self.roll_now_s);
        if at <= 0.0 || self.separators.iter().any(|s| (s.at - at).abs() < 1e-9) {
            return;
        }
        let pos = self.separators.partition_point(|s| s.at < at);
        self.separators.insert(pos, Instance { at, name: None });
        self.dirty = true;
    }

    /// Whether a separator currently sits at `at`, within the same dedupe
    /// tolerance [`insert_separator`] uses. Lets the UI prune its broadcast list
    /// of breaks the idle snap-back trimmed away, so a dropped break stops being
    /// re-sent on the heartbeat instead of being re-created on the peer forever
    /// (F1).
    pub fn has_separator_at(&self, at: f64) -> bool {
        self.separators.iter().any(|s| (s.at - at).abs() < 1e-9)
    }

    /// Remove an *unnamed* separator at `at` (same dedupe tolerance), if present.
    /// Used to honor a peer's tombstone when its snap-back trimmed a shared
    /// manual break, so the break disappears on both surfaces rather than
    /// lingering as a permanent one-sided line (R13). Only `name.is_none()`
    /// breaks are eligible; named sections stay put, and the 1e-9 tolerance makes
    /// an accidental collision with a derived auto-pause boundary negligible.
    /// Returns whether anything was removed.
    pub fn remove_separator(&mut self, at: f64) -> bool {
        if let Some(pos) = self
            .separators
            .iter()
            .position(|s| s.name.is_none() && (s.at - at).abs() < 1e-9)
        {
            self.separators.remove(pos);
            self.dirty = true;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Saving. Synchronous on the caller's (GUI) thread: these are one-shot writes
// of tiny files (an evening of playing is well under a megabyte), so the cost
// is milliseconds at worst — and a synchronous result lets the caller clear
// the dirty flag only on success and surface the error, which the unsaved-
// close dialog depends on for correctness.
// ---------------------------------------------------------------------------

/// SMF timing: 120 BPM (500 000 µs per quarter) at 480 ticks per quarter gives
/// exactly 960 ticks per second, so roll seconds convert to ticks losslessly
/// at ~1 ms resolution.
const TICKS_PER_SEC: f64 = 960.0;
const PPQ: u16 = 480;
const TEMPO_US_PER_QUARTER: u32 = 500_000;

pub(crate) fn ticks(t_s: f64) -> u32 {
    (t_s * TICKS_PER_SEC).round().max(0.0) as u32
}

/// Inverse of `ticks`: general tick→seconds conversion for a *loaded* file
/// (see score.rs), whose tempo/PPQ may differ from this app's own fixed
/// write-side constants above.
pub(crate) fn seconds(ticks: u32, us_per_quarter: u32, ppq: u16) -> f64 {
    let ticks_per_sec = ppq as f64 * 1_000_000.0 / us_per_quarter.max(1) as f64;
    ticks as f64 / ticks_per_sec
}

/// Quick save: write `rolls/roll_<unix>.mid` (+ `.json` color sidecar) and
/// return the `.mid` path.
pub fn save_quick(roll: &Roll, local_color: [u8; 3], remote_color: [u8; 3]) -> io::Result<PathBuf> {
    std::fs::create_dir_all("rolls")?;
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = PathBuf::from(format!("rolls/roll_{unix}.mid"));
    save_midi(roll, local_color, remote_color, &path)?;
    Ok(path)
}

/// Write the roll as a Standard MIDI File at `path`, plus a `<stem>.json`
/// sidecar carrying the player colors (the one thing SMF cannot express).
///
/// Layout: format 1 with a conductor track (tempo + instance markers) and one
/// note track per player, named so any DAW shows who played what. Segments
/// still open at save time get a synthesized note-off at the live edge — in
/// the file only; the in-memory roll is untouched.
pub fn save_midi(
    roll: &Roll,
    local_color: [u8; 3],
    remote_color: [u8; 3],
    path: &Path,
) -> io::Result<()> {
    use midly::num::{u15, u24, u28, u4, u7};
    use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind};

    // Marker text is borrowed by the Smf, so the owned strings must outlive it.
    let marker_texts: Vec<String> = roll
        .separators
        .iter()
        .enumerate()
        .map(|(i, sep)| sep.name.clone().unwrap_or_else(|| format!("instance {}", i + 2)))
        .collect();

    let mut smf = Smf::new(Header::new(Format::Parallel, Timing::Metrical(u15::from(PPQ))));

    // Conductor track: tempo, then a marker at each instance boundary. A
    // custom name for instance 1 gets its own marker at tick 0 (there is no
    // separator entry to carry it); unnamed instance 1 emits nothing, keeping
    // unnamed exports byte-identical to before names existed.
    let mut conductor: Vec<TrackEvent> = vec![TrackEvent {
        delta: u28::from(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::from(TEMPO_US_PER_QUARTER))),
    }];
    if let Some(name) = &roll.first_instance_name {
        conductor.push(TrackEvent {
            delta: u28::from(0),
            kind: TrackEventKind::Meta(MetaMessage::Marker(name.as_bytes())),
        });
    }
    let mut prev = 0u32;
    for (sep, text) in roll.separators.iter().zip(&marker_texts) {
        let at = ticks(sep.at);
        conductor.push(TrackEvent {
            // `separators` is kept sorted, so `at >= prev`; saturating_sub is a
            // belt-and-suspenders guard against a stray out-of-order boundary
            // ever underflowing this delta into a garbage multi-hour value.
            delta: u28::from(at.saturating_sub(prev)),
            kind: TrackEventKind::Meta(MetaMessage::Marker(text.as_bytes())),
        });
        prev = at.max(prev);
    }
    conductor.push(TrackEvent {
        delta: u28::from(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    smf.tracks.push(conductor);

    for (who, name) in [
        (Who::Local, &b"open-piano: local player"[..]),
        (Who::Remote, &b"open-piano: remote player"[..]),
    ] {
        // Absolute-tick events: (tick, order, data, value) with order 0 = note
        // off, 1 = pedal CC64, 2 = note on. Sorted (stably) with offs before
        // ons at equal times so a retriggered note never nests; pedal sits
        // between so a re-pedal at a chord boundary releases before the ons.
        let mut events: Vec<(u32, u8, u8, u8)> = Vec::new();
        for seg in roll.segments.iter().filter(|s| s.who == who) {
            events.push((ticks(seg.start_s), 2, seg.midi, seg.velocity.max(1)));
            events.push((ticks(seg.end_s.unwrap_or(roll.now_s())), 0, seg.midi, 0));
        }
        // Pedal spans as CC64 level/0 pairs. A depth change is stored as
        // adjacent spans meeting at one instant; skip the release there so the
        // written CC stream is a clean level→level transition, not a blip to 0.
        let pedal: Vec<&PedalSegment> =
            roll.pedal_segments.iter().filter(|p| p.who == who).collect();
        for (i, ps) in pedal.iter().enumerate() {
            events.push((ticks(ps.start_s), 1, 64, ps.level));
            let end = ps.end_s.unwrap_or(roll.now_s());
            let continues = pedal.get(i + 1).is_some_and(|n| ticks(n.start_s) == ticks(end));
            if !continues {
                events.push((ticks(end), 1, 64, 0));
            }
        }
        events.sort_by_key(|&(t, order, data, _)| (t, order, data));

        let mut track: Vec<TrackEvent> = vec![TrackEvent {
            delta: u28::from(0),
            kind: TrackEventKind::Meta(MetaMessage::TrackName(name)),
        }];
        let mut prev = 0u32;
        for (at, order, data, value) in events {
            let message = match order {
                2 => MidiMessage::NoteOn { key: u7::from(data), vel: u7::from(value) },
                0 => MidiMessage::NoteOff { key: u7::from(data), vel: u7::from(0) },
                _ => MidiMessage::Controller { controller: u7::from(64), value: u7::from(value) },
            };
            track.push(TrackEvent {
                delta: u28::from(at.saturating_sub(prev)),
                kind: TrackEventKind::Midi { channel: u4::from(0), message },
            });
            prev = at.max(prev);
        }
        track.push(TrackEvent {
            delta: u28::from(0),
            kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
        });
        smf.tracks.push(track);
    }

    smf.save(path)?;

    let sidecar = serde_json::json!({
        "version": 1,
        "kind": "open-piano-roll-colors",
        "local_color": local_color,
        "remote_color": remote_color,
    });
    std::fs::write(path.with_extension("json"), format!("{sidecar}\n"))
}

/// Write the roll as a single self-contained JSONL file: a header line with
/// the player colors, then one line per event (note on/off + instance
/// separators) in time order. Human-readable; nothing lost, but no music
/// software opens it — the `.mid` export is the interoperable one.
pub fn save_jsonl(
    roll: &Roll,
    local_color: [u8; 3],
    remote_color: [u8; 3],
    path: &Path,
) -> io::Result<()> {
    // Milliseconds are plenty (and match the .mid resolution); rounding keeps
    // the f64s from printing as 17-digit noise.
    fn ms(t: f64) -> f64 {
        (t * 1000.0).round() / 1000.0
    }

    let mut lines = vec![serde_json::json!({
        "version": 1,
        "kind": "open-piano-roll",
        "local_color": local_color,
        "remote_color": remote_color,
        "first_instance_name": roll.first_instance_name,
    })];

    // (time, order-within-time, json): offs (0) before separators/pedal (1)
    // before ons (2) at equal timestamps, so instances read cleanly.
    let mut events: Vec<(f64, u8, serde_json::Value)> = Vec::new();
    for seg in &roll.segments {
        let who = match seg.who {
            Who::Local => "l",
            Who::Remote => "r",
        };
        events.push((
            seg.start_s,
            2,
            serde_json::json!({"t": ms(seg.start_s), "e": "on", "n": seg.midi, "who": who,
                               "v": seg.velocity}),
        ));
        let end = seg.end_s.unwrap_or(roll.now_s());
        events.push((
            end,
            0,
            serde_json::json!({"t": ms(end), "e": "off", "n": seg.midi, "who": who}),
        ));
    }
    // Pedal (CC64) levels: one event per span start, a 0 at each release —
    // except between adjacent spans (a depth change), which write as a clean
    // level→level transition. Same convention as the .mid export.
    for who in [Who::Local, Who::Remote] {
        let tag = match who {
            Who::Local => "l",
            Who::Remote => "r",
        };
        let spans: Vec<&PedalSegment> =
            roll.pedal_segments.iter().filter(|p| p.who == who).collect();
        for (i, ps) in spans.iter().enumerate() {
            events.push((
                ps.start_s,
                1,
                serde_json::json!({"t": ms(ps.start_s), "e": "pedal", "v": ps.level, "who": tag}),
            ));
            let end = ps.end_s.unwrap_or(roll.now_s());
            let continues = spans.get(i + 1).is_some_and(|n| (n.start_s - end).abs() < 1e-9);
            if !continues {
                events.push((
                    end,
                    1,
                    serde_json::json!({"t": ms(end), "e": "pedal", "v": 0, "who": tag}),
                ));
            }
        }
    }
    for sep in &roll.separators {
        events.push((
            sep.at,
            1,
            serde_json::json!({"t": ms(sep.at), "e": "sep", "name": sep.name}),
        ));
    }
    events.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    lines.extend(events.into_iter().map(|(_, _, v)| v));

    let mut out = String::new();
    for line in &lines {
        out.push_str(&line.to_string());
        out.push('\n');
    }
    std::fs::write(path, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the clock deterministically: `new()` starts paused, so wall time
    /// between `new()` and the first tick never reaches the roll.
    fn t(base: Instant, s: f64) -> Instant {
        base + Duration::from_secs_f64(s)
    }

    #[test]
    fn clock_pauses_after_threshold_trims_tail_and_separates_instances() {
        let base = Instant::now();
        let mut roll = Roll::new();
        // Explicit timing so the test is independent of the default values:
        // 2 s tail, 2 s lead-in, 30 s break threshold.
        roll.set_timing(
            Duration::from_secs(2),
            Duration::from_secs(2),
            Some(Duration::from_secs(30)),
        );
        roll.tick(t(base, 0.0));
        assert_eq!(roll.now_s(), 0.0);
        assert!(roll.is_paused()); // nothing played yet

        // First note: unpauses, no separator.
        roll.note(Who::Local, NoteMsg::On(60, 100), [1, 2, 3]);
        assert!(!roll.is_paused());
        roll.tick(t(base, 1.0));
        roll.note(Who::Local, NoteMsg::Off(60), [1, 2, 3]);
        assert!(roll.separators.is_empty());
        assert_eq!(roll.segments[0].start_s, 0.0);
        assert_eq!(roll.segments[0].end_s, Some(1.0));

        // Past the threshold the clock pauses and snaps back to the tail —
        // trimming the accrued dead air to last note (1.0) + tail (2.0).
        roll.tick(t(base, 40.0));
        assert!(roll.is_paused());
        let frozen = roll.now_s();
        assert_eq!(frozen, 3.0);
        roll.tick(t(base, 100.0));
        assert_eq!(roll.now_s(), frozen); // ...and stays there.

        // Next note resumes: a separator at the trimmed boundary, and the
        // lead-in pushes the new instance's first note 2 s past it.
        roll.note(Who::Local, NoteMsg::On(61, 100), [1, 2, 3]);
        assert!(!roll.is_paused());
        assert_eq!(roll.separators.len(), 1);
        assert_eq!(roll.separators[0].at, frozen);
        assert_eq!(roll.separators[0].name, None);
        assert_eq!(roll.segments[1].start_s, frozen + 2.0);
        roll.tick(t(base, 101.0));
        assert_eq!(roll.now_s(), frozen + 2.0 + 1.0);
    }

    #[test]
    fn gap_shorter_than_threshold_is_shown_faithfully() {
        let base = Instant::now();
        let mut roll = Roll::new();
        roll.set_timing(
            Duration::from_secs(2),
            Duration::from_secs(2),
            Some(Duration::from_secs(30)),
        );
        roll.tick(t(base, 0.0));
        roll.note(Who::Local, NoteMsg::On(60, 100), [0; 3]);
        roll.tick(t(base, 1.0));
        roll.note(Who::Local, NoteMsg::Off(60), [0; 3]);
        // A 25 s pause is under the 30 s threshold: nothing clamps, the full
        // gap stays on the paper — and the next note joins the same instance.
        roll.tick(t(base, 26.0));
        assert_eq!(roll.now_s(), 26.0);
        assert!(!roll.is_paused());
        roll.note(Who::Local, NoteMsg::On(61, 100), [0; 3]);
        assert!(roll.separators.is_empty());
        assert_eq!(roll.segments[1].start_s, 26.0);
    }

    /// The exact worked example the feature was specced against: 30 s
    /// threshold, 2 s tail, 2 s lead-in, end to end.
    #[test]
    fn section_break_end_to_end_threshold_tail_lead_in() {
        let base = Instant::now();
        let mut roll = Roll::new();
        roll.set_timing(
            Duration::from_secs(2),
            Duration::from_secs(2),
            Some(Duration::from_secs(30)),
        );
        // Section 1: a phrase from 0 to 1.
        roll.tick(t(base, 0.0));
        roll.note(Who::Local, NoteMsg::On(60, 100), [0; 3]);
        roll.tick(t(base, 1.0));
        roll.note(Who::Local, NoteMsg::Off(60), [0; 3]);
        // 35 s of silence: past the threshold, so the break is pending and
        // section 1 ends 2 s (the tail) after its last note.
        roll.tick(t(base, 36.0));
        assert!(roll.is_paused());
        assert_eq!(roll.now_s(), 3.0);
        // The next keypress inserts the boundary at exactly that trimmed
        // edge and opens section 2 with its first note 2 s (the lead-in)
        // past the line. No note's true recorded timestamp moved: only the
        // silent span between the sections was collapsed.
        roll.note(Who::Local, NoteMsg::On(62, 100), [0; 3]);
        assert_eq!(roll.separators.len(), 1);
        assert_eq!(roll.separators[0].at, 3.0);
        assert_eq!(roll.segments[1].start_s, 5.0);
        assert_eq!(roll.current_instance(), 1);
    }

    #[test]
    fn infinite_threshold_gives_an_unbounded_gap_and_no_separators() {
        let base = Instant::now();
        let mut roll = Roll::new();
        roll.set_timing(Duration::from_secs(2), Duration::from_secs(2), None);
        roll.tick(t(base, 0.0));
        roll.note(Who::Local, NoteMsg::On(60, 100), [0; 3]);
        roll.tick(t(base, 1.0));
        roll.note(Who::Local, NoteMsg::Off(60), [0; 3]);
        // Minutes of silence: the paper keeps scrolling and the clock never
        // pauses, so the real gap is preserved in full.
        roll.tick(t(base, 300.0));
        assert_eq!(roll.now_s(), 300.0);
        // The next note is part of the same instance — no separator inserted.
        roll.note(Who::Local, NoteMsg::On(61, 100), [0; 3]);
        assert!(roll.separators.is_empty());
    }

    #[test]
    fn instance_naming_and_manual_breaks() {
        let base = Instant::now();
        let mut roll = Roll::new();
        roll.tick(t(base, 0.0));
        roll.note(Who::Local, NoteMsg::On(60, 100), [0; 3]);
        assert_eq!(roll.current_instance_name(), "instance 1");
        roll.rename_current_instance("warmup".into());
        assert_eq!(roll.current_instance_name(), "warmup");
        assert_eq!(roll.first_instance_name.as_deref(), Some("warmup"));

        // A manual break splits recorded history and lands sorted.
        roll.tick(t(base, 10.0));
        roll.note(Who::Local, NoteMsg::Off(60), [0; 3]);
        roll.insert_separator(4.0);
        roll.insert_separator(2.0);
        roll.insert_separator(4.0); // duplicate: no-op
        roll.insert_separator(0.0); // at zero: no-op
        let ats: Vec<f64> = roll.separators.iter().map(|s| s.at).collect();
        assert_eq!(ats, vec![2.0, 4.0]);

        // The clock (at 10.0) is now past both breaks: current = instance 3.
        assert_eq!(roll.current_instance(), 2);
        assert_eq!(roll.current_instance_name(), "instance 3");
        roll.rename_current_instance("bridge".into());
        assert_eq!(roll.separators[1].name.as_deref(), Some("bridge"));
        assert_eq!(roll.separators[0].name, None);
    }

    #[test]
    fn manual_separator_in_dead_air_survives_snap_back_sorted() {
        // C2 regression: a note near t=0, a manual break inserted near the live
        // edge during a long idle, then the pause snaps the clock back behind
        // that break. The stale break must be dropped so `separators` stays
        // sorted and `save_midi` never underflows its delta math.
        let base = Instant::now();
        let mut roll = Roll::new();
        roll.set_timing(
            Duration::from_secs(2),
            Duration::from_secs(2),
            Some(Duration::from_secs(30)),
        );
        roll.tick(t(base, 0.0));
        roll.note(Who::Local, NoteMsg::On(60, 100), [0; 3]);
        roll.tick(t(base, 1.0));
        roll.note(Who::Local, NoteMsg::Off(60), [0; 3]);
        // Clock runs on during the idle; Ctrl+click a break near the live edge.
        roll.tick(t(base, 20.0));
        roll.insert_separator(19.0);
        assert_eq!(roll.separators.len(), 1);
        // Past the threshold: pause fires, snaps back to last note (1.0) + tail
        // (2.0) = 3.0, and the stale break at 19.0 (now beyond the clock) drops.
        roll.tick(t(base, 55.0));
        assert!(roll.is_paused());
        assert_eq!(roll.now_s(), 3.0);
        assert!(roll.separators.iter().all(|s| s.at <= roll.now_s()));
        // The next note resumes with a boundary at the rewound clock; the list
        // stays sorted.
        roll.note(Who::Local, NoteMsg::On(62, 100), [0; 3]);
        let ats: Vec<f64> = roll.separators.iter().map(|s| s.at).collect();
        assert!(ats.windows(2).all(|w| w[0] <= w[1]), "separators must stay sorted: {ats:?}");
        // And a save doesn't panic (dev) / corrupt (release).
        let dir = std::env::temp_dir().join(format!("op-roll-c2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c2.mid");
        save_midi(&roll, [0; 3], [0; 3], &path).expect("save must succeed");
        assert!(midly::Smf::parse(&std::fs::read(&path).unwrap()).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pedal_never_resets_the_idle_timer() {
        let base = Instant::now();
        let mut roll = Roll::new();
        roll.set_timing(
            Duration::from_secs(2),
            Duration::from_secs(2),
            Some(Duration::from_secs(30)),
        );
        roll.tick(t(base, 0.0));
        roll.note(Who::Local, NoteMsg::On(60, 100), [0; 3]);
        roll.tick(t(base, 1.0));
        roll.note(Who::Local, NoteMsg::Off(60), [0; 3]);

        // Pump the pedal across (and far past) the break threshold with no
        // note on/off: the clock must pause on schedule and trim to
        // last-note + tail, exactly as if the pedal were untouched (compare
        // `clock_pauses_after_threshold_trims_tail_and_separates_instances`).
        for (at, level) in [(5.0, 127u8), (10.0, 64), (20.0, 0), (40.0, 96)] {
            roll.tick(t(base, at));
            roll.pedal(Who::Local, level);
        }
        roll.tick(t(base, 100.0));
        assert_eq!(roll.now_s(), 3.0); // trimmed: last note (1.0) + tail (2.0)

        // The pause fired despite the pedal traffic: the next *note* opens a
        // new instance with a separator at the trimmed boundary.
        roll.note(Who::Local, NoteMsg::On(61, 100), [0; 3]);
        assert_eq!(roll.separators.len(), 1);
        assert_eq!(roll.separators[0].at, 3.0);
    }

    #[test]
    fn pedal_spans_in_trimmed_dead_air_are_clamped_to_the_clock() {
        // M4 regression: half-pedaling during a long idle, then the pause snaps
        // the clock back. Every pedal span must end up within [.., now] with
        // end >= start, so nothing overlays the next section or saves stuck-down.
        let base = Instant::now();
        let mut roll = Roll::new();
        roll.set_timing(
            Duration::from_secs(2),
            Duration::from_secs(2),
            Some(Duration::from_secs(30)),
        );
        roll.tick(t(base, 0.0));
        roll.note(Who::Local, NoteMsg::On(60, 100), [0; 3]);
        roll.tick(t(base, 1.0));
        roll.note(Who::Local, NoteMsg::Off(60), [0; 3]);
        // Pedal activity across the dead air (never resets the idle timer).
        for (at, level) in [(5.0, 127u8), (10.0, 64), (20.0, 0)] {
            roll.tick(t(base, at));
            roll.pedal(Who::Local, level);
        }
        // Past the threshold: pause fires, clock snaps back to 3.0.
        roll.tick(t(base, 55.0));
        assert_eq!(roll.now_s(), 3.0);
        let now = roll.now_s();
        for p in &roll.pedal_segments {
            assert!(p.start_s <= now, "pedal start {} beyond clock {now}", p.start_s);
            let end = p.end_s.unwrap_or(now);
            assert!(end <= now && end >= p.start_s, "span [{}, {end}] invalid", p.start_s);
        }
    }

    #[test]
    fn pedal_spans_open_close_and_dedupe() {
        let base = Instant::now();
        let mut roll = Roll::new();
        roll.tick(t(base, 0.0));
        // A note first, so the clock is running and times are meaningful.
        roll.note(Who::Local, NoteMsg::On(60, 100), [0; 3]);
        roll.pedal(Who::Local, 90);
        roll.pedal(Who::Local, 90); // repeat: no-op
        assert_eq!(roll.pedal_segments.len(), 1);
        roll.tick(t(base, 1.0));
        roll.pedal(Who::Local, 40); // depth change: close + reopen
        assert_eq!(roll.pedal_segments.len(), 2);
        assert_eq!(roll.pedal_segments[0].end_s, Some(1.0));
        assert_eq!(roll.pedal_segments[1].level, 40);
        roll.tick(t(base, 2.0));
        roll.pedal(Who::Local, 0); // release closes the open span
        assert_eq!(roll.pedal_segments[1].end_s, Some(2.0));
        // Each player's pedal is independent; release_pedal force-closes.
        roll.pedal(Who::Remote, 127);
        roll.tick(t(base, 3.0));
        roll.release_pedal(Who::Remote);
        assert_eq!(roll.pedal_segments[2].end_s, Some(3.0));
        // ...and a following press opens a fresh span (level was reset to 0).
        roll.pedal(Who::Remote, 127);
        assert_eq!(roll.pedal_segments.len(), 4);
    }

    #[test]
    fn held_key_keeps_the_clock_running() {
        let base = Instant::now();
        let mut roll = Roll::new();
        roll.tick(t(base, 0.0));
        roll.note(Who::Local, NoteMsg::On(60, 100), [0; 3]);
        // Held far past IDLE_PAUSE: the paper must keep moving.
        roll.tick(t(base, 120.0));
        assert_eq!(roll.now_s(), 120.0);
        roll.note(Who::Local, NoteMsg::Off(60), [0; 3]);
        assert_eq!(roll.segments[0].end_s, Some(120.0));
    }

    #[test]
    fn duplicate_on_is_ignored_and_release_all_closes() {
        let base = Instant::now();
        let mut roll = Roll::new();
        roll.tick(t(base, 0.0));
        roll.note(Who::Remote, NoteMsg::On(60, 100), [0; 3]);
        roll.note(Who::Remote, NoteMsg::On(60, 100), [0; 3]);
        assert_eq!(roll.segments.len(), 1);
        roll.note(Who::Remote, NoteMsg::On(72, 100), [0; 3]);
        roll.tick(t(base, 2.0));
        roll.release_all(Who::Remote);
        assert!(roll.segments.iter().all(|s| s.end_s == Some(2.0)));
        // Off with nothing open is a no-op.
        roll.note(Who::Remote, NoteMsg::Off(60), [0; 3]);
        assert_eq!(roll.segments.len(), 2);
    }

    #[test]
    fn dirty_tracks_note_events_and_saves() {
        let base = Instant::now();
        let mut roll = Roll::new();
        roll.tick(t(base, 0.0));
        assert!(!roll.has_unsaved()); // empty roll never warns
        roll.note(Who::Local, NoteMsg::On(60, 100), [0; 3]);
        assert!(roll.has_unsaved());
        roll.mark_saved();
        assert!(!roll.has_unsaved());
        // An Off after a save is a change too (the segment's end moved).
        roll.note(Who::Local, NoteMsg::Off(60), [0; 3]);
        assert!(roll.has_unsaved());
    }

    #[test]
    fn seconds_to_ticks_is_960_per_second() {
        assert_eq!(ticks(0.0), 0);
        assert_eq!(ticks(1.0), 960);
        assert_eq!(ticks(0.5), 480);
        assert_eq!(ticks(10.0), 9600);
    }

    /// Build a small two-player roll with a separator and an open note.
    fn sample_roll() -> Roll {
        let base = Instant::now();
        let mut roll = Roll::new();
        roll.tick(t(base, 0.0));
        roll.note(Who::Local, NoteMsg::On(60, 100), [220, 60, 60]);
        roll.tick(t(base, 0.5));
        roll.note(Who::Local, NoteMsg::Off(60), [220, 60, 60]);
        roll.note(Who::Remote, NoteMsg::On(64, 100), [60, 110, 230]);
        roll.tick(t(base, 1.0));
        roll.note(Who::Remote, NoteMsg::Off(64), [60, 110, 230]);
        roll.tick(t(base, 60.0)); // idle -> pause
        roll.note(Who::Local, NoteMsg::On(62, 100), [220, 60, 60]); // resume, separator
        roll.tick(t(base, 61.0)); // note 62 left open on purpose
        roll
    }

    #[test]
    fn saved_midi_is_valid_smf_with_expected_shape() {
        let roll = sample_roll();
        let dir = std::env::temp_dir().join(format!("open-piano-roll-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.mid");
        save_midi(&roll, [220, 60, 60], [60, 110, 230], &path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let smf = midly::Smf::parse(&bytes).expect("saved file must be valid SMF");
        assert_eq!(smf.tracks.len(), 3); // conductor + local + remote
        // One marker for the one separator.
        let markers = smf.tracks[0]
            .iter()
            .filter(|e| matches!(e.kind, midly::TrackEventKind::Meta(midly::MetaMessage::Marker(_))))
            .count();
        assert_eq!(markers, 1);
        // The open note got a synthesized off: ons == offs in each note track.
        for track in &smf.tracks[1..] {
            let (mut ons, mut offs) = (0, 0);
            for e in track {
                if let midly::TrackEventKind::Midi { message, .. } = e.kind {
                    match message {
                        midly::MidiMessage::NoteOn { .. } => ons += 1,
                        midly::MidiMessage::NoteOff { .. } => offs += 1,
                        _ => {}
                    }
                }
            }
            assert_eq!(ons, offs);
        }
        // Sidecar exists and holds the colors.
        let sidecar: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path.with_extension("json")).unwrap())
                .unwrap();
        assert_eq!(sidecar["local_color"], serde_json::json!([220, 60, 60]));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn saved_jsonl_is_well_formed_and_ordered() {
        let roll = sample_roll();
        let dir = std::env::temp_dir().join(format!("open-piano-jsonl-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.jsonl");
        save_jsonl(&roll, [220, 60, 60], [60, 110, 230], &path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).expect("every line must be valid JSON"))
            .collect();
        assert_eq!(lines[0]["kind"], "open-piano-roll");
        // 3 segments -> 6 note events, plus 1 separator.
        assert_eq!(lines.len(), 1 + 7);
        let times: Vec<f64> = lines[1..].iter().map(|l| l["t"].as_f64().unwrap()).collect();
        assert!(times.windows(2).all(|w| w[0] <= w[1]), "events must be time-ordered");
        assert!(lines[1..].iter().any(|l| l["e"] == "sep"));

        std::fs::remove_dir_all(&dir).ok();
    }
}
