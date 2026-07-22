//! Playback of a loaded score (see score.rs): one shared playhead driving
//! both tracks, in several modes.
//!
//! **Listen**: the playhead advances in real time (scaled by the speed
//! slider) and both tracks are auto-played through the built-in synth.
//! **Learn**: tracks marked "practice" are *not* auto-played — the player
//! produces them on their real input — and the playhead is *gated* on the
//! player actually playing what the score requires, in one of two ways:
//!
//! * `require_hold` (default): a literal per-frame check — the playhead only
//!   moves while every note the practiced tracks say should be sounding
//!   *right now* is held down. Release early and the paper freezes in place.
//! * onset-gate (Synthesia-style "wait mode", `require_hold` off): free-run
//!   between note onsets, freeze exactly at the next practiced onset until
//!   the required notes are struck, then continue — releasing early is fine.
//!
//! **Evaluation**: one chosen track goes silent and the playhead free-runs —
//! a full playthrough is scored live against the score ([`EvaluationState`]).
//! The optional pause-on-miss setting is the one exception to the free run:
//! it freezes the playhead at a missed note's tolerance-window edge until the
//! note is struck (the frozen time is reported on the result card). Either
//! way, when the take ends the engine flips itself into
//! **EvaluationReview**: a passive Listen-like replay of a synthetic
//! two-track score — the original part next to what was actually played —
//! with per-side show/hear toggles ([`ReviewSettings`]). Review is never a
//! mode the user picks directly; it's entered on take completion and left
//! via "Retake" or by picking another mode.
//!
//! The current segment (see `score::ScoreSegment`) can loop a chosen number
//! of times (or indefinitely) with a short silent pad before each repeat
//! (disabled during an evaluation take — scoring a looped pass isn't
//! well-defined).
//!
//! All synth output goes to `Channel::Playback` via `Synth::note_on/off`
//! directly — never through main.rs's `synth_note_on`/`synth_note_off`,
//! whose mic-echo bookkeeping is sized for the two live channels only.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::PathBuf;

use crate::note::{midi_to_key_index, KEY_COUNT};
use crate::roll::Who;
use crate::score::{Note, Score, Track};
use crate::synth::{Channel, Synth};

/// Silent breather inserted before each loop repeat, so the pickup of the
/// next pass doesn't slam into the tail of the previous one.
const LOOP_PAD_S: f64 = 5.0;

/// Nudge past a satisfied onset-gate so float equality can't re-trigger it.
const GATE_EPS: f64 = 1e-6;

/// Minimum length given to a played note recorded during a take, so a
/// same-frame press+release (staccato, or a GUI stall pumping On+Off in one
/// frame) isn't stored as a zero-length note — invisible/silent in review,
/// where `active_at`'s half-open `start <= t < end` is never true (F31).
const MIN_PLAYED_NOTE_S: f64 = 0.02;

/// Within this long after a segment's start, the "previous segment" button
/// goes back one segment instead of restarting the current one (the standard
/// media-player double-tap convention).
const PREV_SEGMENT_WINDOW_S: f64 = 0.5;

/// Evaluation: score deducted per extra (unmatched) press, in note-score
/// units — two stray presses cost as much as one missed note. Tunable.
const EXTRA_PENALTY_WEIGHT: f32 = 0.5;

/// Evaluation: a press that lands inside the temporal window is never worth
/// less than this, so an edge-of-window hit still beats a miss outright.
const TIMING_SCORE_FLOOR: f32 = 0.1;

/// Evaluation: a judged note must score at least this to count as "clean"
/// for the streak stat.
const STREAK_SCORE_MIN: f32 = 0.5;

/// Evaluation: minimum attempts before a pitch is eligible for the best/worst
/// lists, so one unlucky note can't dominate them.
const PITCH_MIN_ATTEMPTS: u32 = 2;

/// How many pitches the best/worst breakdown lists at most.
const PITCH_LIST_LEN: usize = 3;

/// Color of the review's "what you played" track (the original track keeps
/// its own color). Amber: distinct from both players' defaults.
pub const PLAYED_TRACK_COLOR: [u8; 3] = [235, 185, 60];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Listen,
    Learn,
    /// The live, free-running, scored take (see the module docs).
    Evaluation,
    /// The passive post-take dual-track replay. Never a direct radio target —
    /// entered automatically when an evaluation take finishes.
    EvaluationReview,
}

pub struct LearnSettings {
    /// Which track(s) the player must produce, indexed via `Who::idx()`.
    /// Neither selected = Learn behaves exactly like Listen (no gating).
    pub practice: [bool; 2],
    /// Continuous-hold gating (default) vs onset-gate "wait mode".
    pub require_hold: bool,
    /// Strict: extra held notes also block (the held set must *equal* the
    /// required set, not merely contain it).
    pub block_wrong: bool,
    /// Restrict gating to keys in this inclusive MIDI range — notes outside
    /// it still render and auto-play, they just aren't *required*. Session
    /// state only (not persisted); `None` = the whole keyboard.
    pub key_range: Option<(u8, u8)>,
}

pub struct LoopState {
    pub enabled: bool,
    /// `None` = loop indefinitely; `Some(n)` = n more repeats after the pass
    /// currently playing.
    pub remaining: Option<u32>,
    /// `Some(secs_left)` while silently pausing between repeats.
    pub pad_left_s: Option<f64>,
}

/// The looping segment, expressed for the falling-roll renderer so it can draw
/// the *next* repeat's notes falling into view (see [`FallingView`] and
/// `draw_falling`). Times are score seconds.
#[derive(Clone, Copy)]
pub struct LoopWrap {
    pub seg_start_s: f64,
    pub seg_end_s: f64,
    /// One whole repeat's span *including* the silent pad: `(end - start) +
    /// LOOP_PAD_S`. Ghost note copies are drawn shifted by this, so a copy sits
    /// exactly where the real note lands one repeat later — which makes the
    /// wrap seamless (the ghost's on-screen position at the wrap instant equals
    /// the real note's position right after it).
    pub period_s: f64,
}

/// What the falling-notes panel should render this frame. Normally the roll's
/// "now" edge is just the playhead, but while a segment loops the playhead
/// freezes at the segment end through the silent pad and then snaps back to the
/// start — which alone would freeze the roll and then jump it. [`FallingView`]
/// instead carries a *continuous* view time that keeps advancing through the
/// pad, plus the looping segment so the renderer can wrap the next repeat into
/// view; together they make the roll scroll seamlessly across the boundary.
pub struct FallingView {
    /// Score-time the roll's bottom (keyboard) edge sits at.
    pub view_s: f64,
    /// `Some` exactly while a segment loops.
    pub wrap: Option<LoopWrap>,
}

/// How forgiving evaluation scoring is. The presets fix all three tolerances
/// at once; `Custom` exposes them individually.
#[derive(Clone, Copy, PartialEq)]
pub enum Strictness {
    Strict,
    Normal,
    Lenient,
    Custom {
        /// Half-width (±s) of the window around a note's start in which a
        /// press can match it at all.
        temporal_tolerance_s: f64,
        /// Fraction of the full velocity range (0..1) at which the force
        /// score bottoms out.
        force_tolerance: f32,
        /// Fraction of the full CC64 range (0..1) at which the pedal score
        /// bottoms out.
        pedal_tolerance: f32,
    },
}

impl Strictness {
    /// `(temporal ±s, force tolerance, pedal tolerance)` — the preset tuples,
    /// or `Custom`'s own values. Starting points; tune by feel.
    pub fn tolerances(&self) -> (f64, f32, f32) {
        match *self {
            Strictness::Strict => (0.06, 0.25, 0.5),
            Strictness::Normal => (0.15, 0.5, 0.75),
            Strictness::Lenient => (0.30, 0.8, 1.0),
            Strictness::Custom { temporal_tolerance_s, force_tolerance, pedal_tolerance } => {
                (temporal_tolerance_s, force_tolerance, pedal_tolerance)
            }
        }
    }
}

/// Settings for [`Mode::Evaluation`]. Evaluation always scores exactly one
/// track — the take is "the original vs. what you played", not both parts at
/// once. Key-range scoping reuses `LearnSettings::key_range` (already
/// mode-agnostic) rather than duplicating it here.
pub struct EvaluationSettings {
    /// Which score track the player performs; `None` = nothing selected yet
    /// (the take free-runs but there's nothing to score).
    pub evaluate: Option<Who>,
    pub strictness: Strictness,
    /// Whether key force counts toward the score at all — the tier only sets
    /// how forgiving it is *when* it does. Ignored (no effect on scoring)
    /// unless the live input is MIDI: the mic path's flat placeholder
    /// velocity must not be graded as if genuine.
    pub evaluate_velocity: bool,
    /// Whether pedal use counts. Ignored unless the live input is MIDI *and*
    /// the evaluated track carries pedal data to score against. Pedal is only
    /// ever judged at the instant of a required note's press — never
    /// free-floating during gaps.
    pub evaluate_pedal: bool,
    /// Freeze the playhead at a missed note's tolerance-window edge until the
    /// note is actually struck, instead of scoring it a miss and moving on.
    /// The time spent frozen is reported on the take's result card.
    pub pause_on_miss: bool,
}

/// Per-side toggles for [`Mode::EvaluationReview`]: one flag per track drives
/// both its visibility on the falling panel and its audibility. Deliberately
/// not `LearnSettings::practice` — that means "silence until the player
/// produces it", which has no sense in a passive post-hoc comparison.
pub struct ReviewSettings {
    pub show_original: bool,
    pub show_played: bool,
}

impl Default for ReviewSettings {
    fn default() -> Self {
        ReviewSettings { show_original: true, show_played: true }
    }
}

// ---------------------------------------------------------------------------
// Evaluation scoring. Live state accrues in `EvaluationState` while the take
// free-runs; `finalize`/`result` turn it into the immutable
// `EvaluationResult` + the synthetic review score.
// ---------------------------------------------------------------------------

/// One note the evaluated track requires, precomputed at take start.
#[derive(Clone, Copy)]
struct RequiredNote {
    /// Index into the evaluated track's `notes` (unique per note — the key
    /// judgements are looked up by).
    note_index: usize,
    midi: u8,
    start_s: f64,
    target_velocity: u8,
    /// Whether the score wants the sustain pedal down at this note's start;
    /// `None` when the track has no pedal data (dimension not scored).
    required_pedal_down: Option<bool>,
}

enum Outcome {
    Correct,
    Missed,
}

/// The verdict on one required note.
struct NoteJudgement {
    note_index: usize,
    midi: u8,
    outcome: Outcome,
    /// Press time minus required time (matched notes only).
    press_delta_s: Option<f64>,
    timing_score: f32,
    /// `None` when force wasn't scored (dimension off / not applicable).
    velocity_score: Option<f32>,
    /// `None` when pedal wasn't scored.
    pedal_score: Option<f32>,
}

impl NoteJudgement {
    /// Equal-weight mean of whichever dimensions were scored, in [0, 1];
    /// a miss is flat 0. (Equal weight is a starting point, not sacred.)
    fn score(&self) -> f32 {
        if matches!(self.outcome, Outcome::Missed) {
            return 0.0;
        }
        let parts = [Some(self.timing_score), self.velocity_score, self.pedal_score];
        let (sum, n) = parts
            .iter()
            .flatten()
            .fold((0.0f32, 0u32), |(s, n), v| (s + v, n + 1));
        if n == 0 { 0.0 } else { sum / n as f32 }
    }
}

/// A press that matched no pending required note — penalized in the total.
struct ExtraPress {
    at_s: f64,
    midi: u8,
}

/// Live scoring state for one evaluation take. The applicability of the
/// velocity/pedal dimensions and the tolerances are *frozen at take start*,
/// so a mid-take input-backend flap (or settings edit — those restart the
/// take) can't retroactively invalidate already-recorded judgements.
struct EvaluationState {
    required: Vec<RequiredNote>,
    /// Indices into `required` not yet matched or missed, in score order.
    pending: VecDeque<usize>,
    judged: Vec<NoteJudgement>,
    extra_presses: Vec<ExtraPress>,
    /// Everything actually played, time-aligned to the playhead — becomes the
    /// review's "played" track. Built here, NOT from `roll::Roll`: the roll
    /// spans the whole app session across pieces and warm-ups and has no
    /// notion of "just this take".
    played_notes: Vec<Note>,
    /// Index into `played_notes` of the still-open note per MIDI number.
    open_played: [Option<usize>; 128],
    /// Last frame's held set, to diff for releases.
    prev_held: BTreeSet<u8>,
    // -- pause-on-miss bookkeeping (see `record_pause`) --
    /// Wall-clock seconds spent frozen waiting for missed notes.
    paused_time_s: f64,
    /// How many distinct freezes occurred (edge-counted via `was_paused`).
    pause_count: u32,
    was_paused: bool,
    // -- frozen at take start --
    score_velocity: bool,
    score_pedal: bool,
    pause_on_miss: bool,
    window_s: f64,
    force_tol: f32,
    pedal_tol: f32,
    /// The Learn key range frozen at take start: out-of-range notes aren't
    /// required *and* out-of-range presses aren't penalized as extras (H7).
    /// Frozen here so a mid-take range edit can't desync live scoring from the
    /// already-built `required` list (M13 restarts the take on any edit).
    key_range: Option<(u8, u8)>,
}

impl EvaluationState {
    /// Whether `midi` is inside the frozen key range (or always, if unset).
    fn in_key_range(&self, midi: u8) -> bool {
        self.key_range.map_or(true, |(lo, hi)| (lo..=hi).contains(&midi))
    }

    /// Playhead position at which the earliest pending note's tolerance window
    /// closes — mirrors the expiry-sweep condition in [`Self::record_frame`].
    /// `pending` is in score order and `window_s` is constant for the take, so
    /// checking only the front suffices.
    fn next_expiry_s(&self) -> Option<f64> {
        self.pending.front().map(|&i| self.required[i].start_s + self.window_s)
    }

    /// Where the take should end: the later of the piece's duration and the
    /// final required note's tolerance-window close. Running just past
    /// `duration_s` when the last note sits near the end gives that note its
    /// full window instead of truncating a slightly-late-but-in-tolerance
    /// press into a miss (L16). `required` is start-sorted, so the last entry
    /// has the greatest start.
    fn take_end_s(&self, duration_s: f64) -> f64 {
        self.required
            .last()
            .map_or(duration_s, |r| (r.start_s + self.window_s).max(duration_s))
    }

    /// Per-frame pause bookkeeping for the pause-on-miss gate: total frozen
    /// time (wall-clock, deliberately not scaled by speed — it's how long the
    /// *player* sat waiting) and an edge-counted number of distinct freezes.
    fn record_pause(&mut self, held_back: bool, dt_s: f64) {
        if held_back {
            if !self.was_paused {
                self.pause_count += 1;
            }
            self.paused_time_s += dt_s;
        }
        self.was_paused = held_back;
    }

    /// Feed one frame of live input. `playhead_s` is the post-advance playhead
    /// (used for scoring, expiry, and played-note ends); `pre_advance_s` is the
    /// playhead at the *start* of this frame, so an onset can still match a note
    /// whose tolerance window closed somewhere inside the frame (R36). `onsets`
    /// are the note-ons as `(midi, velocity)`.
    fn record_frame(
        &mut self,
        playhead_s: f64,
        pre_advance_s: f64,
        held: &BTreeSet<u8>,
        onsets: &[(u8, u8)],
        pedal_level: u8,
    ) {
        // Releases close their "played" note at the current playhead.
        for &m in self.prev_held.difference(held) {
            if let Some(i) = self.open_played[m as usize].take() {
                self.played_notes[i].end_s = playhead_s;
            }
        }

        // Match this frame's onsets *before* the expiry sweep (R36): a note whose
        // window closes inside this frame's advance (possible after a GUI stall,
        // where dt jumps up to 0.1 s) must be matchable by a press this frame,
        // instead of being swept into a Miss *and* then having the press booked
        // as an ExtraPress — one press double-charged.
        for &(midi, velocity) in onsets {
            // Outside the 88-key piano range a press can never be a required
            // note and renders on no lane — ignore it entirely (mirroring the
            // score loaders), so an octave-shifted controller or a >88-key
            // instrument isn't charged an extra press for perfect play (F13).
            if midi_to_key_index(midi).is_none() {
                continue;
            }
            // The played track records every press, matched or not.
            if self.open_played[midi as usize].is_none() {
                self.open_played[midi as usize] = Some(self.played_notes.len());
                self.played_notes.push(Note {
                    start_s: playhead_s,
                    end_s: playhead_s,
                    midi,
                    velocity,
                });
            }
            // Out-of-key-range presses take no part in scoring: those notes were
            // filtered out of `required` (not expected), so charging them as
            // extra presses would penalize perfect play of a range-scoped part
            // (H7). They still render on the played track above.
            if !self.in_key_range(midi) {
                continue;
            }
            // Match in FIFO score order — oldest pending first, a simple,
            // accepted approximation over full bipartite matching. A note counts
            // as reachable this frame if the playhead is within its window *or*
            // its window overlapped this frame's advance (R36).
            let hit = self.pending.iter().position(|&i| {
                let r = &self.required[i];
                if r.midi != midi {
                    return false;
                }
                (playhead_s - r.start_s).abs() <= self.window_s
                    || (r.start_s <= playhead_s && r.start_s + self.window_s >= pre_advance_s)
            });
            match hit {
                Some(pos) => {
                    let i = self.pending.remove(pos).expect("position came from iter");
                    self.judge_match(i, playhead_s, velocity, pedal_level);
                }
                None => self.extra_presses.push(ExtraPress { at_s: playhead_s, midi }),
            }
        }

        // Required notes whose window has fully passed (and weren't just matched
        // above) become misses as the playhead sweeps by (`pending` is in score
        // order, so only the front can have expired).
        while let Some(&i) = self.pending.front() {
            if self.required[i].start_s + self.window_s < playhead_s {
                self.pending.pop_front();
                self.judge_missed(i);
            } else {
                break;
            }
        }

        // A note pressed *and* released within one frame (staccato, or a stalled
        // frame) never enters `held`, so the prev/held release diff above never
        // closes its played note. Close any freshly-opened played note that
        // isn't currently held, so it doesn't stretch to end-of-piece in review
        // and its pitch can be pressed (and recorded) again (M15).
        for m in 0..128u8 {
            if let Some(i) = self.open_played[m as usize] {
                if !held.contains(&m) {
                    // Give a same-frame press+release a minimum length so it
                    // isn't a zero-length note (silent/invisible in review) (F31).
                    let n = &mut self.played_notes[i];
                    n.end_s = playhead_s.max(n.start_s + MIN_PLAYED_NOTE_S);
                    self.open_played[m as usize] = None;
                }
            }
        }

        self.prev_held = held.clone();
    }

    fn judge_match(&mut self, i: usize, playhead_s: f64, velocity: u8, pedal_level: u8) {
        let r = self.required[i];
        let delta = playhead_s - r.start_s;
        let timing_score =
            (1.0 - (delta.abs() / self.window_s.max(1e-9)) as f32).max(TIMING_SCORE_FLOOR);
        let velocity_score = self.score_velocity.then(|| {
            let diff = (velocity as f32 - r.target_velocity as f32).abs() / 127.0;
            (1.0 - diff / self.force_tol.max(1e-3)).clamp(0.0, 1.0)
        });
        // Pedal is judged only here — at the instant of a required note's
        // press — never independently during silence. Both sides are binarized
        // (the score's target is already down/up, from `pedal_down_at`): what
        // matters is whether the pedal was *down* (CC64 >= 64) as the score
        // wants, not hitting an exact analog level — so genuine half-pedaling
        // that is "down" scores full credit rather than near-failing (M12).
        let pedal_score = match (self.score_pedal, r.required_pedal_down) {
            (true, Some(down)) => {
                let played_down = pedal_level >= 64;
                Some(if played_down == down { 1.0 } else { 0.0 })
            }
            _ => None,
        };
        // `pedal_tol` no longer scales pedal scoring (it's binary now); the
        // Custom preset still carries it for forward-compat.
        let _ = self.pedal_tol;
        self.judged.push(NoteJudgement {
            note_index: r.note_index,
            midi: r.midi,
            outcome: Outcome::Correct,
            press_delta_s: Some(delta),
            timing_score,
            velocity_score,
            pedal_score,
        });
    }

    fn judge_missed(&mut self, i: usize) {
        let r = self.required[i];
        self.judged.push(NoteJudgement {
            note_index: r.note_index,
            midi: r.midi,
            outcome: Outcome::Missed,
            press_delta_s: None,
            timing_score: 0.0,
            velocity_score: None,
            pedal_score: None,
        });
    }

    /// Close the books at the end of the take: open played notes end at the
    /// piece's end, and everything still pending is a miss (even a final note
    /// whose window straddles the end — the take is over).
    fn finalize(&mut self, duration_s: f64) {
        for m in 0..128 {
            if let Some(i) = self.open_played[m].take() {
                // A note opened during the end-extension window (start_s past
                // duration_s) must not close before it opened (end_s < start_s):
                // that renders as an inverted sliver and never sounds in review
                // (F29). Close at the later of the two.
                self.played_notes[i].end_s = duration_s.max(self.played_notes[i].start_s);
            }
        }
        while let Some(i) = self.pending.pop_front() {
            self.judge_missed(i);
        }
    }

    /// Wipe everything recorded but keep `required` and the frozen
    /// tolerances/applicability — for restarting the take (transport jumps).
    fn restart(&mut self) {
        self.pending = (0..self.required.len()).collect();
        self.judged.clear();
        self.extra_presses.clear();
        self.played_notes.clear();
        self.open_played = [None; 128];
        self.prev_held.clear();
        // Pause history is per-take too — a scrub must not leak it forward.
        self.paused_time_s = 0.0;
        self.pause_count = 0;
        self.was_paused = false;
    }

    /// Crunch the judged take into the displayed breakdown.
    fn result(&self) -> EvaluationResult {
        let raw_points: f32 = self.judged.iter().map(|j| j.score()).sum();
        let penalty = self.extra_presses.len() as f32 * EXTRA_PENALTY_WEIGHT;
        let percent = if self.required.is_empty() {
            0.0
        } else {
            100.0 * ((raw_points - penalty) / self.required.len() as f32).clamp(0.0, 1.0)
        };

        let mut per_pitch: BTreeMap<u8, PitchStats> = BTreeMap::new();
        for j in &self.judged {
            let s = per_pitch.entry(j.midi).or_insert(PitchStats { attempts: 0, avg_score: 0.0 });
            s.attempts += 1;
            s.avg_score += j.score(); // sum for now; divided just below
        }
        for s in per_pitch.values_mut() {
            s.avg_score /= s.attempts as f32;
        }
        let mut eligible: Vec<(u8, f32)> = per_pitch
            .iter()
            .filter(|(_, s)| s.attempts >= PITCH_MIN_ATTEMPTS)
            .map(|(&midi, s)| (midi, s.avg_score))
            .collect();
        eligible.sort_by(|a, b| a.1.total_cmp(&b.1));
        let worst_pitches: Vec<u8> =
            eligible.iter().take(PITCH_LIST_LEN).map(|&(m, _)| m).collect();
        let best_pitches: Vec<u8> =
            eligible.iter().rev().take(PITCH_LIST_LEN).map(|&(m, _)| m).collect();

        let mean_of = |scores: &mut dyn Iterator<Item = f32>| -> Option<f32> {
            let (sum, n) = scores.fold((0.0f32, 0u32), |(s, n), v| (s + v, n + 1));
            (n > 0).then(|| sum / n as f32)
        };
        let velocity_accuracy = mean_of(&mut self.judged.iter().filter_map(|j| j.velocity_score));
        let pedal_accuracy = mean_of(&mut self.judged.iter().filter_map(|j| j.pedal_score));
        // Signed mean of the matched press deltas: a systematic rush (-) or
        // drag (+), as opposed to the unsigned spread timing_score captures.
        let deltas: Vec<f64> = self.judged.iter().filter_map(|j| j.press_delta_s).collect();
        let timing_bias_s = (!deltas.is_empty())
            .then(|| deltas.iter().sum::<f64>() / deltas.len() as f64);

        // Longest streak of clean *steps* in score order: chord notes sharing
        // an onset (same epsilon-grouping convention as `required_set`/
        // `gate_at_or_after`'s checkpoints) collapse to one step, which is
        // clean iff every note in it judged Correct above the threshold and
        // no extra press landed inside its window.
        let by_note: BTreeMap<usize, f32> = self
            .judged
            .iter()
            .map(|j| {
                let s = if matches!(j.outcome, Outcome::Correct) { j.score() } else { -1.0 };
                (j.note_index, s)
            })
            .collect();
        let (mut streak, mut longest_streak) = (0u32, 0u32);
        let mut k = 0;
        while k < self.required.len() {
            let t = self.required[k].start_s;
            let mut clean = true;
            while k < self.required.len() && (self.required[k].start_s - t).abs() < 1e-6 {
                clean &= by_note
                    .get(&self.required[k].note_index)
                    .is_some_and(|&s| s >= STREAK_SCORE_MIN);
                k += 1;
            }
            clean = clean
                && !self.extra_presses.iter().any(|e| (e.at_s - t).abs() <= self.window_s);
            if clean {
                streak += 1;
                longest_streak = longest_streak.max(streak);
            } else {
                streak = 0;
            }
        }

        // Where the stray presses cluster, if anywhere — often one wrong
        // neighbor key hit over and over.
        let mut extra_by_pitch: BTreeMap<u8, u32> = BTreeMap::new();
        for e in &self.extra_presses {
            *extra_by_pitch.entry(e.midi).or_insert(0) += 1;
        }
        let extra_hotspot = extra_by_pitch
            .into_iter()
            .max_by_key(|&(_, n)| n)
            .filter(|&(_, n)| n >= 2)
            .map(|(midi, _)| midi);

        let missed_count =
            self.judged.iter().filter(|j| matches!(j.outcome, Outcome::Missed)).count();
        EvaluationResult {
            percent,
            per_pitch,
            worst_pitches,
            best_pitches,
            longest_streak,
            velocity_accuracy,
            pedal_accuracy,
            timing_bias_s,
            extra_press_count: self.extra_presses.len(),
            extra_hotspot,
            missed_count,
            matched_count: self.judged.len() - missed_count,
            // The frozen per-take flag is exactly "was this setting active for
            // this completed take" — `None` (setting off) omits the line.
            pause_stats: self
                .pause_on_miss
                .then(|| PauseStats { total_s: self.paused_time_s, count: self.pause_count }),
        }
    }
}

pub struct PitchStats {
    pub attempts: u32,
    pub avg_score: f32,
}

/// How long (and how often) the pause-on-miss gate froze the take.
pub struct PauseStats {
    /// Total wall-clock seconds spent frozen.
    pub total_s: f64,
    /// Number of distinct freezes.
    pub count: u32,
}

/// The finished take's breakdown, shown in the results window and the review
/// panel. `velocity_accuracy`/`pedal_accuracy` are `None` when that dimension
/// wasn't evaluated (the UI omits the line entirely rather than showing N/A).
pub struct EvaluationResult {
    pub percent: f32,
    pub per_pitch: BTreeMap<u8, PitchStats>,
    pub worst_pitches: Vec<u8>,
    pub best_pitches: Vec<u8>,
    /// Longest run of consecutive clean onset-steps, in score order.
    pub longest_streak: u32,
    pub velocity_accuracy: Option<f32>,
    pub pedal_accuracy: Option<f32>,
    /// Signed mean press offset (matched notes only): negative = rushing,
    /// positive = dragging. `None` when nothing matched.
    pub timing_bias_s: Option<f64>,
    pub extra_press_count: usize,
    /// The key most of the stray presses landed on, when there's a repeat
    /// offender (>= 2 on the same pitch).
    pub extra_hotspot: Option<u8>,
    pub missed_count: usize,
    pub matched_count: usize,
    /// Pause-on-miss summary; `None` when the setting was off for the take
    /// (the UI omits the line, same convention as the accuracies above).
    pub pause_stats: Option<PauseStats>,
}

pub struct PlaybackEngine {
    pub score: Score,
    /// The score file this engine was loaded from — where the segment-name
    /// sidecar gets written on renames/breaks.
    pub source_path: PathBuf,
    pub playhead_s: f64,
    pub playing: bool,
    pub finished: bool,
    /// Playhead advance rate relative to wall clock (the speed slider).
    /// Everything else keys off `playhead_s`, so this is the only place
    /// speed exists.
    pub speed: f32,
    pub mode: Mode,
    pub learn: LearnSettings,
    pub loop_state: LoopState,
    pub eval: EvaluationSettings,
    pub review: ReviewSettings,
    /// Live scoring state; `Some` exactly while `mode == Evaluation`.
    eval_state: Option<EvaluationState>,
    /// The finished take's breakdown; `Some` while reviewing.
    pub eval_result: Option<EvaluationResult>,
    /// The synthetic original+played score EvaluationReview renders and
    /// sounds (see [`Self::display_score`]). `self.score` is never
    /// overwritten by evaluation.
    review_score: Option<Score>,
    /// One-shot edge flag set when a take finishes; main.rs consumes it (via
    /// [`Self::take_review_transition`]) to pop the results window.
    review_just_started: bool,
    /// Notes currently sounding on the synth per track (edge detection for
    /// note_on/note_off as the playhead moves).
    sounding: [BTreeSet<u8>; 2],
}

impl PlaybackEngine {
    pub fn new(score: Score, source_path: PathBuf) -> Self {
        PlaybackEngine {
            score,
            source_path,
            playhead_s: 0.0,
            playing: true,
            finished: false,
            speed: 1.0,
            mode: Mode::Listen,
            learn: LearnSettings {
                practice: [false, false],
                require_hold: true,
                block_wrong: false,
                key_range: None,
            },
            loop_state: LoopState { enabled: false, remaining: None, pad_left_s: None },
            eval: EvaluationSettings {
                evaluate: None,
                strictness: Strictness::Normal,
                evaluate_velocity: false,
                evaluate_pedal: false,
                pause_on_miss: false,
            },
            review: ReviewSettings::default(),
            eval_state: None,
            eval_result: None,
            review_score: None,
            review_just_started: false,
            sounding: [BTreeSet::new(), BTreeSet::new()],
        }
    }

    /// Whether the player (not the synth) is responsible for this track.
    pub fn practiced(&self, who: Who) -> bool {
        match self.mode {
            Mode::Learn => self.learn.practice[who.idx()],
            Mode::Evaluation => self.eval.evaluate == Some(who),
            Mode::Listen | Mode::EvaluationReview => false,
        }
    }

    /// The score the render/audio paths should read: the synthetic
    /// original+played pair while reviewing, the loaded file otherwise.
    pub fn display_score(&self) -> &Score {
        match (&self.mode, &self.review_score) {
            (Mode::EvaluationReview, Some(review)) => review,
            _ => &self.score,
        }
    }

    /// Whether a track should be drawn/heard right now. Only EvaluationReview
    /// makes tracks toggleable (per [`ReviewSettings`]: slot 0 = original,
    /// slot 1 = played); every other mode always shows both.
    pub fn track_visible(&self, who: Who) -> bool {
        match (self.mode, who) {
            (Mode::EvaluationReview, Who::Local) => self.review.show_original,
            (Mode::EvaluationReview, Who::Remote) => self.review.show_played,
            _ => true,
        }
    }

    fn active_at(&self, who: Who, t: f64) -> impl Iterator<Item = &Note> {
        self.display_score().tracks[who.idx()]
            .notes
            .iter()
            .filter(move |n| n.start_s <= t && t < n.end_s)
    }

    fn in_key_range(&self, midi: u8) -> bool {
        self.learn.key_range.map_or(true, |(lo, hi)| (lo..=hi).contains(&midi))
    }

    /// The notes the practiced tracks say should be sounding at `t` (scoped
    /// to the Learn key range, if one is set).
    fn required_set(&self, t: f64) -> BTreeSet<u8> {
        [Who::Local, Who::Remote]
            .into_iter()
            .filter(|&w| self.practiced(w))
            .flat_map(|w| self.active_at(w, t).map(|n| n.midi))
            .filter(|&m| self.in_key_range(m))
            .collect()
    }

    /// The union of practiced-track notes active anywhere in `[a, b]` (scoped
    /// to the Learn key range). Unlike [`required_set`](Self::required_set),
    /// which samples a single instant, this catches notes shorter than one
    /// frame's advance so hold-mode gating can't skip them (F32).
    fn required_over(&self, a: f64, b: f64) -> BTreeSet<u8> {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let score = self.display_score();
        [Who::Local, Who::Remote]
            .into_iter()
            .filter(|&w| self.practiced(w))
            .flat_map(|w| {
                score.tracks[w.idx()]
                    .notes
                    .iter()
                    .filter(move |n| n.start_s <= hi && n.end_s > lo)
                    .map(|n| n.midi)
            })
            .filter(|&m| self.in_key_range(m))
            .collect()
    }

    /// Smallest practiced-track note start time >= t, if any. Simultaneous
    /// chord notes (and cross-track ties) collapse to one checkpoint via min.
    fn gate_at_or_after(&self, t: f64) -> Option<f64> {
        [Who::Local, Who::Remote]
            .into_iter()
            .filter(|&w| self.practiced(w))
            .flat_map(|w| {
                self.score.tracks[w.idx()]
                    .notes
                    .iter()
                    .filter(|n| self.in_key_range(n.midi))
                    .map(|n| n.start_s)
                    .find(|&s| s >= t)
            })
            .fold(None, |acc: Option<f64>, t2| Some(acc.map_or(t2, |a| a.min(t2))))
    }

    fn gate_ok(&self, required: &BTreeSet<u8>, held: &BTreeSet<u8>) -> bool {
        if self.learn.block_wrong {
            held == required
        } else {
            required.is_subset(held)
        }
    }

    /// Advance one frame. `held` is the player's live held-key set (MIDI
    /// numbers); `onsets` are this frame's note-ons as `(midi, velocity)` and
    /// `pedal_level` the live CC64 level — both consumed only by Evaluation
    /// (velocity exists only at the instant of attack, so it must arrive
    /// here, not be reconstructed from `held`); `dt_s` is the frame's
    /// wall-clock delta.
    pub fn tick(
        &mut self,
        dt_s: f64,
        held: &BTreeSet<u8>,
        onsets: &[(u8, u8)],
        pedal_level: u8,
        synth: &Synth,
    ) {
        // Loop pad first, with an early return: a single frame must never
        // both finish padding *and* re-cross the segment end below, or a
        // finite repeat count would double-decrement. Gating is deliberately
        // not enforced during the pad — it's a breather, not a checkpoint.
        if let Some(left) = self.loop_state.pad_left_s {
            if left - dt_s <= 0.0 {
                self.loop_state.pad_left_s = None;
                self.playhead_s = self.score.segments[self.current_segment_index()].start_s;
            } else {
                self.loop_state.pad_left_s = Some(left - dt_s);
            }
            // The playhead is parked just *inside* the segment (see
            // begin_loop_pad), where its final notes still count as active —
            // so force silence rather than letting drive_auto sound them.
            self.silence(synth);
            return;
        }

        // The looping segment is whichever one the playhead is in *before*
        // this frame's advance: the moment the playhead touches a segment's
        // end, `current_segment_index` already reports the next one, so a
        // post-advance lookup would loop the wrong segment.
        let seg_end = self.score.segments[self.current_segment_index()].end_s;

        if self.playing && !self.finished {
            let no_practice = !self.learn.practice[0] && !self.learn.practice[1];
            match self.mode {
                Mode::Listen => self.advance(dt_s),
                // The review replay free-runs exactly like Listen.
                Mode::EvaluationReview => self.advance(dt_s),
                Mode::Evaluation => self.evaluation_advance(dt_s, held, onsets, pedal_level),
                // Nothing selected to practice: intentionally identical to
                // Listen (the Learn panel hints this).
                Mode::Learn if no_practice => self.advance(dt_s),
                Mode::Learn if self.learn.require_hold => self.hold_mode_advance(dt_s, held),
                Mode::Learn => self.wait_mode_advance(dt_s, held),
            }
        }

        // Loop-back once the playhead reaches that segment's end.
        if self.loop_state.enabled && self.playhead_s >= seg_end - GATE_EPS {
            match self.loop_state.remaining {
                // Repeats used up: disengage and keep playing forward.
                Some(0) => self.loop_state.enabled = false,
                Some(n) => {
                    self.loop_state.remaining = Some(n - 1);
                    self.begin_loop_pad(seg_end);
                }
                None => self.begin_loop_pad(seg_end),
            }
        }

        self.drive_auto(synth);
    }

    fn begin_loop_pad(&mut self, seg_end: f64) {
        // Park just *inside* the segment so current_segment_index still
        // resolves to the looping segment when the pad ends.
        self.playhead_s = (seg_end - GATE_EPS).max(0.0);
        self.finished = false;
        self.loop_state.pad_left_s = Some(LOOP_PAD_S);
    }

    fn advance(&mut self, dt_s: f64) {
        // Advance over the *displayed* score's duration, which in
        // EvaluationReview is the review score — extended to cover any correct
        // press made inside the end-extension window (R38). Clamping to the
        // original `self.score.duration_s` there left such a played note above
        // the end line, never crossed and never sounded. In Listen the displayed
        // score *is* `self.score`, so this is unchanged.
        let dur = self.display_score().duration_s;
        self.playhead_s = (self.playhead_s + dt_s * self.speed as f64).min(dur);
        self.finished = self.playhead_s >= dur;
    }

    /// Continuous-hold gate: freeze in place unless everything required across
    /// the whole tentative advance is held. Gating on the interval, not just the
    /// current instant, is what keeps a note shorter than one frame's step from
    /// starting and ending between two samples and never being required (F32) —
    /// the same class the wait-mode fix (M14) addresses.
    fn hold_mode_advance(&mut self, dt_s: f64, held: &BTreeSet<u8>) {
        let new = (self.playhead_s + dt_s * self.speed as f64).min(self.score.duration_s);
        let required = self.required_over(self.playhead_s, new);
        if self.gate_ok(&required, held) {
            self.advance(dt_s);
        }
    }

    /// Onset-gate: free-run up to the next practiced onset, freeze exactly
    /// there until satisfied, then jump past it (releasing early is fine).
    fn wait_mode_advance(&mut self, dt_s: f64, held: &BTreeSet<u8>) {
        let want = (self.playhead_s + dt_s * self.speed as f64).min(self.score.duration_s);
        match self.gate_at_or_after(self.playhead_s) {
            None => self.playhead_s = want,
            Some(g) if g > self.playhead_s + GATE_EPS => self.playhead_s = want.min(g),
            Some(g) => {
                let required = self.required_set(g);
                self.playhead_s = if self.gate_ok(&required, held) {
                    // Advance past the satisfied gate, but not past the *next*
                    // onset: a frame can carry the playhead up to one advance
                    // (0.1–0.2 s) beyond `g`, which would jump over — and never
                    // require — any onset closer than that (M14).
                    let next = self
                        .gate_at_or_after(g + GATE_EPS)
                        .unwrap_or(self.score.duration_s);
                    want.max(g + GATE_EPS).min(next).min(self.score.duration_s)
                } else {
                    g
                };
            }
        }
        self.finished = self.playhead_s >= self.score.duration_s;
    }

    /// The Evaluation frame: free-runs by default — the playhead is not gated
    /// on what's played — except that with `pause_on_miss` on, the advance is
    /// clamped at the earliest pending note's tolerance-window edge, so a
    /// missed note freezes the take until it's actually struck. Then feeds the
    /// frame's live input to the scorer. On the finished edge the take
    /// finalizes and the engine flips itself into review.
    ///
    /// The clamp lands *exactly* on `start_s + window_s`: the expiry sweep's
    /// strict `<` never fires there (the note stays pending indefinitely)
    /// while the match condition's `<=` still accepts a press, so once the
    /// note is hit and popped, `next_expiry_s` moves to the next note (or
    /// `None`) and the take free-runs again — no separate resume path. A
    /// wrong-pitch press while frozen just becomes an `ExtraPress` as usual.
    /// `want` is capped at `duration_s` first, so a final note whose window
    /// straddles the end never holds the take back (it finishes and the note
    /// is missed, exactly as in free-run).
    fn evaluation_advance(
        &mut self,
        dt_s: f64,
        held: &BTreeSet<u8>,
        onsets: &[(u8, u8)],
        pedal_level: u8,
    ) {
        // `tick` only dispatches here while !finished, so `self.finished`
        // turning true below IS the finished edge. By construction that edge
        // and `held_back` are mutually exclusive — a take never finalizes
        // mid-pause.
        // End the take at the later of the piece duration and the final note's
        // window close, so a last note near the end still gets its full
        // tolerance window instead of being truncated into a miss (L16).
        // Extend a hair past the final gate: when the last note's window
        // straddles the end, its gate (`start_s + window_s`) equals
        // `take_end_s`, so `held_back = gate < want` (with `want` capped at the
        // end) could never fire and pause-on-miss free-ran past the final note
        // instead of freezing on it like every other note (F14).
        let end = self
            .eval_state
            .as_ref()
            .map_or(self.score.duration_s, |s| s.take_end_s(self.score.duration_s))
            + GATE_EPS;
        let pre_advance = self.playhead_s;
        let want = (self.playhead_s + dt_s * self.speed as f64).min(end);
        let gate = self
            .eval_state
            .as_ref()
            .filter(|s| s.pause_on_miss)
            .and_then(EvaluationState::next_expiry_s);
        let held_back = gate.is_some_and(|g| g < want);
        self.playhead_s = if held_back { gate.unwrap() } else { want };
        self.finished = self.playhead_s >= end;
        let playhead_s = self.playhead_s;
        if let Some(state) = &mut self.eval_state {
            state.record_pause(held_back, dt_s);
            state.record_frame(playhead_s, pre_advance, held, onsets, pedal_level);
        }
        if self.finished {
            self.finalize_evaluation();
        }
    }

    /// (Re)start an evaluation take from the top. `live_midi` — whether the
    /// live input source is a real MIDI device — is what velocity/pedal
    /// applicability is frozen from (see [`EvaluationState`]). Called on
    /// entering the mode, on any evaluation-settings change, and by "Retake".
    pub fn start_evaluation(&mut self, live_midi: bool, synth: &Synth) {
        self.silence(synth);
        self.mode = Mode::Evaluation;
        self.playhead_s = 0.0;
        self.finished = false;
        // A take is running by definition (review parks paused, and a Retake
        // from there must not start frozen).
        self.playing = true;
        // Scoring a looped pass isn't well-defined; the loop checkbox is also
        // greyed out while evaluating.
        self.loop_state.enabled = false;
        self.loop_state.pad_left_s = None;
        self.eval_result = None;
        self.review_score = None;
        self.eval_state = Some(self.build_eval_state(live_midi));
    }

    /// Drop all evaluation/review state — for leaving via the Listen/Learn
    /// radios. (The mode itself was already changed by the radio.)
    pub fn exit_evaluation(&mut self, synth: &Synth) {
        self.silence(synth);
        self.eval_state = None;
        self.eval_result = None;
        self.review_score = None;
    }

    /// One-shot: true exactly once per Evaluation → EvaluationReview
    /// transition. main.rs polls this after `tick` to pop the results window.
    pub fn take_review_transition(&mut self) -> bool {
        std::mem::take(&mut self.review_just_started)
    }

    fn build_eval_state(&self, live_midi: bool) -> EvaluationState {
        let (window_s, force_tol, pedal_tol) = self.eval.strictness.tolerances();
        let mut required: Vec<RequiredNote> = Vec::new();
        if let Some(who) = self.eval.evaluate {
            let track = &self.score.tracks[who.idx()];
            for (note_index, n) in track.notes.iter().enumerate() {
                // Reuse the Learn key range: out-of-range notes still render
                // and auto-play, they just aren't required — same meaning the
                // one "Key range" control has everywhere else.
                if !self.in_key_range(n.midi) {
                    continue;
                }
                required.push(RequiredNote {
                    note_index,
                    midi: n.midi,
                    start_s: n.start_s,
                    target_velocity: n.velocity,
                    required_pedal_down: track.pedal_down_at(n.start_s),
                });
            }
            // Score notes are start-sorted already; keep the invariant
            // explicit — `pending`'s FIFO order and the expiry sweep rely on it.
            required.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
        }
        // Freeze applicability now (see EvaluationState docs): force needs
        // real (MIDI) velocity; pedal additionally needs score pedal data.
        let score_pedal = self.eval.evaluate_pedal
            && live_midi
            && self
                .eval
                .evaluate
                .is_some_and(|w| !self.score.tracks[w.idx()].pedal_events.is_empty());
        EvaluationState {
            pending: (0..required.len()).collect(),
            required,
            judged: Vec::new(),
            extra_presses: Vec::new(),
            played_notes: Vec::new(),
            open_played: [None; 128],
            prev_held: BTreeSet::new(),
            paused_time_s: 0.0,
            pause_count: 0,
            was_paused: false,
            score_velocity: self.eval.evaluate_velocity && live_midi,
            score_pedal,
            pause_on_miss: self.eval.pause_on_miss,
            window_s,
            force_tol,
            pedal_tol,
            key_range: self.learn.key_range,
        }
    }

    /// The take just ended: crunch the result, materialize the dual-track
    /// review score, and flip into EvaluationReview — parked paused at the
    /// top, ready to replay.
    fn finalize_evaluation(&mut self) {
        let Some(mut state) = self.eval_state.take() else { return };
        state.finalize(self.score.duration_s);

        // A take with no track selected to evaluate produced nothing meaningful:
        // building a review would blank the real (auto-played) track from the
        // falling panel, deselect all three mode radios, and park paused at 0
        // with no results window to explain any of it. Fall back to normal
        // Listen playback of the real score instead of entering review (F16).
        let Some(who) = self.eval.evaluate else {
            self.eval_result = None;
            self.review_score = None;
            self.mode = Mode::Listen;
            self.playhead_s = 0.0;
            self.finished = false;
            self.playing = false;
            self.review_just_started = false;
            return;
        };
        self.eval_result = Some(state.result());

        // Slot 0 = the original evaluated part, slot 1 = what was played.
        // Both review slots are spoken for, so the *other* (auto-played)
        // track, if any, is deliberately dropped from the review view.
        let original = &self.score.tracks[who.idx()];
        // Extend the review duration to cover any played note whose press landed
        // in the end-extension window (a late-but-in-tolerance final note), so it
        // is reachable and audible in review instead of stranded above the end
        // line (R38).
        let review_duration = state
            .played_notes
            .iter()
            .map(|n| n.end_s)
            .fold(self.score.duration_s, f64::max);
        self.review_score = Some(Score {
            tracks: [
                Track {
                    notes: original.notes.clone(),
                    color: original.color,
                    pedal_events: original.pedal_events.clone(),
                },
                Track {
                    notes: state.played_notes,
                    color: PLAYED_TRACK_COLOR,
                    pedal_events: Vec::new(),
                },
            ],
            duration_s: review_duration,
            markers: self.score.markers.clone(),
            first_marker_name: self.score.first_marker_name.clone(),
            segments: self.score.segments.clone(),
            warning: None,
        });

        self.mode = Mode::EvaluationReview;
        self.review = ReviewSettings::default();
        self.playhead_s = 0.0;
        self.finished = false;
        self.playing = false;
        // A track was evaluated (the `None` case returned early above), so the
        // results window always pops here.
        self.review_just_started = true;
    }

    /// Sound the unpracticed tracks: diff "should be sounding at the
    /// playhead" against "is sounding" and send the edges to the synth.
    /// Practiced tracks are the player's job — their sound (and their marks
    /// on the live history roll) comes from the real input path, unchanged.
    /// In EvaluationReview both tracks are passive; the [`ReviewSettings`]
    /// toggles gate audibility instead (visibility rides the same flags —
    /// see `track_visible`).
    fn drive_auto(&mut self, synth: &Synth) {
        for who in [Who::Local, Who::Remote] {
            let idx = who.idx();
            let audible = self.playing && !self.practiced(who) && self.track_visible(who);
            if !audible {
                let old = std::mem::take(&mut self.sounding[idx]);
                for m in old {
                    synth.note_off(m, Channel::Playback);
                }
                continue;
            }
            // The key range scopes what's *audible* too, not just what's
            // required: with a band set, out-of-range notes neither auto-play
            // here nor light up in `active_key_array` — one consistent meaning
            // for the single "Key range" control, in both modes.
            let now: BTreeSet<u8> = self
                .active_at(who, self.playhead_s)
                .map(|n| n.midi)
                .filter(|&m| self.in_key_range(m))
                .collect();
            for &m in now.difference(&self.sounding[idx]) {
                synth.note_on(m, Channel::Playback);
            }
            for &m in self.sounding[idx].difference(&now) {
                synth.note_off(m, Channel::Playback);
            }
            self.sounding[idx] = now;
        }
    }

    /// Stop every synth voice this engine started. Call before dropping the
    /// engine, jumping the playhead, or pausing.
    pub fn silence(&mut self, synth: &Synth) {
        for s in &mut self.sounding {
            let old = std::mem::take(s);
            for m in old {
                synth.note_off(m, Channel::Playback);
            }
        }
    }

    /// Pause/resume. Pausing silences immediately (nothing rings while
    /// paused); on resume the next `tick`'s `drive_auto` re-triggers whatever
    /// should be sounding at the unchanged playhead — `sounding` was just
    /// cleared, so it all reads as new.
    pub fn set_playing(&mut self, playing: bool, synth: &Synth) {
        if self.playing && !playing {
            self.silence(synth);
        }
        self.playing = playing;
    }

    pub fn jump_to(&mut self, t: f64, synth: &Synth) {
        self.silence(synth);
        // Scoring a scrubbed partial pass isn't well-defined: any jump while
        // evaluating wipes the scorer and restarts the take from the top.
        let t = if self.mode == Mode::Evaluation {
            if let Some(state) = &mut self.eval_state {
                state.restart();
            }
            0.0
        } else {
            t
        };
        self.playhead_s = t.clamp(0.0, self.score.duration_s);
        self.finished = self.playhead_s >= self.score.duration_s;
        // A stale pad countdown must not survive a jump — it would resume
        // padding logic against whatever segment the jump landed in.
        self.loop_state.pad_left_s = None;
    }

    /// True while silently padding between loop repeats — the breather where
    /// nothing sounds. The keyboard's playback lighting is suppressed here so a
    /// note that ends right on the segment boundary doesn't stay lit through
    /// the pad even though `silence` already killed its sound.
    pub fn in_loop_pad(&self) -> bool {
        self.loop_state.pad_left_s.is_some()
    }

    /// View state for the falling-notes panel (see [`FallingView`]). Off a
    /// loop this is simply the playhead; while looping it returns a continuous
    /// view time (advancing through the pad rather than freezing) and the
    /// looping segment, so `draw_falling` can wrap the next repeat into view
    /// and the roll never stalls-then-jumps at the loop boundary.
    pub fn falling_view(&self) -> FallingView {
        if !self.loop_state.enabled {
            return FallingView { view_s: self.playhead_s, wrap: None };
        }
        // During the pad the playhead is parked just *inside* the segment end
        // (see `begin_loop_pad`), so `current_segment_index` still resolves to
        // the looping segment.
        let seg = &self.score.segments[self.current_segment_index()];
        let (seg_start_s, seg_end_s) = (seg.start_s, seg.end_s);
        let period_s = (seg_end_s - seg_start_s) + LOOP_PAD_S;
        // While padding, drive the view off the countdown so it slides
        // continuously from the segment end toward the wrap point
        // (seg_end + LOOP_PAD_S); otherwise it's just the (advancing) playhead.
        let view_s = match self.loop_state.pad_left_s {
            Some(left) => seg_end_s + (LOOP_PAD_S - left).clamp(0.0, LOOP_PAD_S),
            None => self.playhead_s,
        };
        FallingView { view_s, wrap: Some(LoopWrap { seg_start_s, seg_end_s, period_s }) }
    }

    /// Segment containing the playhead (the last one at/after the end).
    pub fn current_segment_index(&self) -> usize {
        let n = self.score.segments.len(); // never 0: see Score::build_segments
        self.score
            .segments
            .iter()
            .position(|s| self.playhead_s < s.end_s)
            .unwrap_or(n - 1)
    }

    /// The ⏪ button: restart the current segment — or, within the first
    /// half-second of it, go to the previous segment instead.
    pub fn restart_or_previous(&mut self, synth: &Synth) {
        let idx = self.current_segment_index();
        let into = self.playhead_s - self.score.segments[idx].start_s;
        let target = if into > PREV_SEGMENT_WINDOW_S || idx == 0 {
            self.score.segments[idx].start_s
        } else {
            self.score.segments[idx - 1].start_s
        };
        self.jump_to(target, synth);
    }

    /// The ⏩ button: jump to the next segment's start (no-op on the last).
    pub fn next_segment(&mut self, synth: &Synth) {
        let idx = self.current_segment_index();
        if let Some(seg) = self.score.segments.get(idx + 1) {
            let start = seg.start_s;
            self.jump_to(start, synth);
        }
    }

    /// Key-state array for one track's currently-active notes, for the
    /// layered keyboard renderer (only meaningful for unpracticed tracks).
    /// Scoped to the key range like `drive_auto`, so the lit keys always
    /// match what's audible.
    pub fn active_key_array(&self, who: Who) -> [bool; KEY_COUNT] {
        let mut out = [false; KEY_COUNT];
        for n in self.active_at(who, self.playhead_s) {
            if self.in_key_range(n.midi) {
                if let Some(i) = midi_to_key_index(n.midi) {
                    out[i] = true;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::{ScoreSegment, Track};

    /// Local track: notes at [1,2) on 60 and [3,4) on 62+64 (a chord), with
    /// pedal down over the chord. Remote track: one note at [1.5, 2.5) on 70.
    /// Two segments split at 2.5.
    fn engine() -> PlaybackEngine {
        let n = |start_s: f64, end_s: f64, midi: u8| Note { start_s, end_s, midi, velocity: 80 };
        let score = Score {
            tracks: [
                Track {
                    notes: vec![n(1.0, 2.0, 60), n(3.0, 4.0, 62), n(3.0, 4.0, 64)],
                    color: [0; 3],
                    pedal_events: vec![(2.8, 127), (4.2, 0)],
                },
                Track {
                    notes: vec![n(1.5, 2.5, 70)],
                    color: [0; 3],
                    pedal_events: Vec::new(),
                },
            ],
            duration_s: 5.0,
            markers: Vec::new(),
            first_marker_name: None,
            segments: vec![
                ScoreSegment { name: "a".into(), start_s: 0.0, end_s: 2.5 },
                ScoreSegment { name: "b".into(), start_s: 2.5, end_s: 5.0 },
            ],
            warning: None,
        };
        PlaybackEngine::new(score, PathBuf::from("test.mid"))
    }

    fn held(notes: &[u8]) -> BTreeSet<u8> {
        notes.iter().copied().collect()
    }

    #[test]
    fn listen_mode_advances_and_finishes() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        for _ in 0..60 {
            pb.tick(0.1, &held(&[]), &[], 0, &synth);
        }
        assert!(pb.finished);
        assert_eq!(pb.playhead_s, 5.0);
    }

    #[test]
    fn speed_scales_the_playhead() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.speed = 0.5;
        pb.tick(1.0, &held(&[]), &[], 0, &synth);
        assert!((pb.playhead_s - 0.5).abs() < 1e-9);
    }

    #[test]
    fn hold_mode_freezes_without_required_notes() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.mode = Mode::Learn;
        pb.learn.practice = [true, false];
        pb.jump_to(1.2, &synth); // inside local note 60
        pb.tick(0.1, &held(&[]), &[], 0, &synth);
        assert_eq!(pb.playhead_s, 1.2); // frozen: 60 not held
        pb.tick(0.1, &held(&[60]), &[], 0, &synth);
        assert!((pb.playhead_s - 1.3).abs() < 1e-9); // held -> advances
        // Extra notes don't block unless block_wrong.
        pb.tick(0.1, &held(&[60, 99]), &[], 0, &synth);
        assert!((pb.playhead_s - 1.4).abs() < 1e-9);
        pb.learn.block_wrong = true;
        pb.tick(0.1, &held(&[60, 99]), &[], 0, &synth);
        assert!((pb.playhead_s - 1.4).abs() < 1e-9); // strict: extra blocks
        // Remote track is not practiced: its active note (70) is not required.
        pb.learn.block_wrong = false;
        pb.tick(0.1, &held(&[60]), &[], 0, &synth);
        assert!((pb.playhead_s - 1.5).abs() < 1e-9);
    }

    #[test]
    fn wait_mode_stops_at_onsets_then_releases() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.mode = Mode::Learn;
        pb.learn.practice = [true, false];
        pb.learn.require_hold = false;
        // Free-runs to the first onset (1.0) and freezes there.
        for _ in 0..30 {
            pb.tick(0.1, &held(&[]), &[], 0, &synth);
        }
        assert!((pb.playhead_s - 1.0).abs() < 1e-9);
        // Strike it: continues, and releasing immediately is fine.
        pb.tick(0.1, &held(&[60]), &[], 0, &synth);
        assert!(pb.playhead_s > 1.0);
        for _ in 0..30 {
            pb.tick(0.1, &held(&[]), &[], 0, &synth);
        }
        assert!((pb.playhead_s - 3.0).abs() < 1e-9); // next gate: the chord
        // The whole chord is required at once.
        pb.tick(0.1, &held(&[62]), &[], 0, &synth);
        assert!((pb.playhead_s - 3.0).abs() < 1e-9);
        pb.tick(0.1, &held(&[62, 64]), &[], 0, &synth);
        assert!(pb.playhead_s > 3.0);
    }

    #[test]
    fn key_range_scopes_gating() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.mode = Mode::Learn;
        pb.learn.practice = [true, false];
        pb.learn.require_hold = false;
        pb.learn.key_range = Some((63, 80)); // excludes 60 and 62, includes 64
        for _ in 0..30 {
            pb.tick(0.1, &held(&[]), &[], 0, &synth);
        }
        // Skipped the 1.0 onset (60 out of range); frozen at 3.0 needing only 64.
        assert!((pb.playhead_s - 3.0).abs() < 1e-9);
        pb.tick(0.1, &held(&[64]), &[], 0, &synth);
        assert!(pb.playhead_s > 3.0);
    }

    #[test]
    fn key_range_scopes_active_keys() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.jump_to(3.5, &synth); // inside the local 62+64 chord
        let idx = |m: u8| midi_to_key_index(m).unwrap();

        let keys = pb.active_key_array(Who::Local);
        assert!(keys[idx(62)] && keys[idx(64)]); // no range: whole chord lit

        pb.learn.key_range = Some((63, 80)); // excludes 62, includes 64
        let keys = pb.active_key_array(Who::Local);
        assert!(!keys[idx(62)], "out-of-range note must not light up");
        assert!(keys[idx(64)], "in-range note must still light up");
    }

    #[test]
    fn looping_pads_counts_down_and_disengages() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.jump_to(2.0, &synth);
        pb.loop_state.enabled = true;
        pb.loop_state.remaining = Some(1);
        // Reach the segment end (2.5): pad starts, one repeat consumed.
        for _ in 0..6 {
            pb.tick(0.1, &held(&[]), &[], 0, &synth);
        }
        assert!(pb.loop_state.pad_left_s.is_some());
        assert_eq!(pb.loop_state.remaining, Some(0));
        // The pad freezes the playhead just inside the segment.
        let parked = pb.playhead_s;
        pb.tick(1.0, &held(&[]), &[], 0, &synth);
        assert_eq!(pb.playhead_s, parked);
        // Burn the rest of the pad (3.9s left): snaps back to segment start.
        for _ in 0..4 {
            pb.tick(1.0, &held(&[]), &[], 0, &synth);
        }
        assert!((pb.playhead_s - 0.0).abs() < 1e-6);
        assert!(pb.loop_state.pad_left_s.is_none());
        // Second arrival at the end: remaining == 0 -> disengage, play on.
        for _ in 0..26 {
            pb.tick(0.1, &held(&[]), &[], 0, &synth);
        }
        assert!(!pb.loop_state.enabled);
        assert!(pb.playhead_s > 2.5);
    }

    #[test]
    fn falling_view_advances_through_the_pad_and_wraps_by_one_period() {
        // The roll's view time must keep sliding forward through the silent pad
        // (never freeze) and, at the loop boundary, drop by exactly one period —
        // which is what makes the next repeat's ghost notes, drawn one period
        // ahead, land seamlessly where the real notes then continue.
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.loop_state.enabled = true;
        pb.loop_state.remaining = None; // loop segment "a" (0..2.5) forever
        let dt = 0.1;
        let period = 2.5 + LOOP_PAD_S;

        // Off the pad the view is just the playhead, with the looping segment
        // reported so the renderer can wrap.
        let fv = pb.falling_view();
        assert_eq!(fv.view_s, 0.0);
        let w = fv.wrap.expect("looping => a wrap is present");
        assert!((w.period_s - period).abs() < 1e-9);
        assert_eq!((w.seg_start_s, w.seg_end_s), (0.0, 2.5));

        let mut prev = pb.falling_view().view_s;
        let mut wraps = 0;
        for _ in 0..120 {
            pb.tick(dt, &held(&[]), &[], 0, &synth);
            let cur = pb.falling_view().view_s;
            let d = cur - prev;
            if d < 0.0 {
                // The only permitted decrease is the wrap: it drops by ~one
                // period, and only after the view has reached ~the wrap point,
                // so the on-screen picture doesn't jump.
                wraps += 1;
                assert!((d + period).abs() <= dt + 1e-6, "wrap dropped {d:.3}, want ~-{period}");
                assert!((prev - period).abs() <= dt + 1e-6, "wrap fired early at view {prev:.3}");
            } else {
                // Otherwise the view only ever creeps forward by one frame —
                // including all the way through the pad (in_loop_pad true).
                assert!(d <= dt * pb.speed as f64 + 1e-6, "view jumped forward {d:.3}");
            }
            prev = cur;
        }
        assert!(wraps >= 1, "expected at least one loop wrap within 12s");
    }

    #[test]
    fn transport_jumps() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.jump_to(4.0, &synth);
        assert_eq!(pb.current_segment_index(), 1);
        // Deep into segment 1: ⏪ restarts it.
        pb.restart_or_previous(&synth);
        assert_eq!(pb.playhead_s, 2.5);
        // Within the double-tap window: ⏪ goes to the previous segment.
        pb.restart_or_previous(&synth);
        assert_eq!(pb.playhead_s, 0.0);
        // Already first: stays at its start.
        pb.restart_or_previous(&synth);
        assert_eq!(pb.playhead_s, 0.0);
        // ⏩ next / no-op on last.
        pb.next_segment(&synth);
        assert_eq!(pb.playhead_s, 2.5);
        pb.next_segment(&synth);
        assert_eq!(pb.playhead_s, 2.5);
        // ⏭ to the very end reports finished immediately.
        pb.jump_to(pb.score.duration_s, &synth);
        assert!(pb.finished);
    }

    /// Run one whole evaluation take: `plays` maps a playhead time (matched
    /// to the tick grid) to the onsets fired that frame.
    fn run_take(pb: &mut PlaybackEngine, synth: &Synth, plays: &[(f64, &[(u8, u8)], u8)]) {
        let mut t = 0.0;
        while pb.mode == Mode::Evaluation {
            t += 0.1;
            let frame = plays
                .iter()
                .find(|(at, _, _)| (t - at).abs() < 1e-9)
                .map_or((&[][..], 0), |&(_, onsets, pedal)| (onsets, pedal));
            pb.tick(0.1, &held(&[]), frame.0, frame.1, synth);
            assert!(t < 10.0, "take never finished");
        }
    }

    #[test]
    fn evaluation_free_runs_scores_and_flips_into_review() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.eval.evaluate = Some(Who::Local);
        pb.start_evaluation(true, &synth);
        assert_eq!(pb.playhead_s, 0.0);
        assert!(matches!(pb.mode, Mode::Evaluation));

        // Hit 60 dead on time (an extra 99 rings at 2.0); miss the chord
        // entirely. The playhead must never stall on any of it.
        run_take(
            &mut pb,
            &synth,
            &[(1.0, &[(60, 80)], 0), (2.0, &[(99, 90)], 0)],
        );

        // The take finished -> review, parked paused at the top, edge flagged.
        assert!(matches!(pb.mode, Mode::EvaluationReview));
        assert!(pb.take_review_transition());
        assert!(!pb.take_review_transition(), "edge flag is one-shot");
        assert_eq!(pb.playhead_s, 0.0);
        assert!(!pb.playing);

        let r = pb.eval_result.as_ref().expect("result must exist in review");
        assert_eq!((r.matched_count, r.missed_count, r.extra_press_count), (1, 2, 1));
        assert!(r.pause_stats.is_none(), "pause-on-miss off -> no pause line");
        // 60 was matched with |delta| ~0 and no velocity/pedal dimensions on:
        // 1.0 point, minus the extra-press penalty, over 3 required notes.
        let expect = 100.0 * (1.0 - EXTRA_PENALTY_WEIGHT) / 3.0;
        assert!((r.percent - expect).abs() < 1.0, "got {}", r.percent);
        assert_eq!(r.longest_streak, 1);

        // The review score pairs the original part with what was played.
        let review = pb.display_score();
        assert_eq!(review.tracks[0].notes.len(), 3);
        assert_eq!(review.tracks[1].notes.len(), 2); // 60 + the stray 99
        assert_eq!(review.tracks[1].color, PLAYED_TRACK_COLOR);
        // Review toggles gate visibility/audibility per side.
        assert!(pb.track_visible(Who::Local) && pb.track_visible(Who::Remote));
        pb.review.show_played = false;
        assert!(!pb.track_visible(Who::Remote));

        // Leaving review drops the synthetic score.
        pb.mode = Mode::Listen;
        pb.exit_evaluation(&synth);
        assert_eq!(pb.display_score().tracks[1].notes.len(), 1); // the real remote track
        assert!(pb.eval_result.is_none());
    }

    #[test]
    fn evaluation_scores_velocity_and_pedal_when_frozen_applicable() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.eval.evaluate = Some(Who::Local);
        pb.eval.evaluate_velocity = true;
        pb.eval.evaluate_pedal = true;
        pb.start_evaluation(true, &synth);

        // Perfect take: right notes, right times, exact target velocity (80),
        // pedal up at 1.0 and down at the 3.0 chord — matching the score's
        // pedal stream (down from 2.8).
        run_take(
            &mut pb,
            &synth,
            &[(1.0, &[(60, 80)], 0), (3.0, &[(62, 80), (64, 80)], 127)],
        );

        let r = pb.eval_result.as_ref().unwrap();
        assert_eq!(r.missed_count, 0);
        assert!(r.percent > 99.0, "got {}", r.percent);
        assert_eq!(r.velocity_accuracy, Some(1.0));
        assert_eq!(r.pedal_accuracy, Some(1.0));
        assert_eq!(r.longest_streak, 2); // note step + chord step

        // Mic input: both extra dimensions must freeze OFF even when the
        // checkboxes are on — a placeholder velocity is never graded.
        pb.start_evaluation(false, &synth);
        run_take(&mut pb, &synth, &[(1.0, &[(60, 100)], 0)]);
        let r = pb.eval_result.as_ref().unwrap();
        assert_eq!(r.velocity_accuracy, None);
        assert_eq!(r.pedal_accuracy, None);
    }

    /// The exact-representable tolerances the pause-on-miss tests use, so the
    /// gate boundary (note start + 0.5) is a clean float the assertions can
    /// compare against directly.
    fn half_second_window() -> Strictness {
        Strictness::Custom {
            temporal_tolerance_s: 0.5,
            force_tolerance: 0.5,
            pedal_tolerance: 0.75,
        }
    }

    #[test]
    fn pause_on_miss_freezes_at_the_window_edge_and_resumes_on_the_hit() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.eval.evaluate = Some(Who::Local);
        pb.eval.strictness = half_second_window();
        pb.eval.pause_on_miss = true;
        pb.start_evaluation(true, &synth);

        // Never play note 60 (start 1.0): the playhead must clamp exactly at
        // its window edge (1.5) and hold, however long we keep ticking.
        for _ in 0..40 {
            pb.tick(0.1, &held(&[]), &[], 0, &synth);
            assert!(pb.playhead_s <= 1.5, "ran past the gate: {}", pb.playhead_s);
        }
        assert_eq!(pb.playhead_s, 1.5);
        assert!(matches!(pb.mode, Mode::Evaluation), "gated take must not finish");

        // A wrong-pitch press while frozen: still frozen, still an extra.
        pb.tick(0.1, &held(&[99]), &[(99, 90)], 0, &synth);
        assert_eq!(pb.playhead_s, 1.5);

        // The late hit lands exactly on the window edge (`<=` matches where
        // the expiry sweep's `<` never fires). The hit frame itself is still
        // clamped — matching happens after the advance — and the gate then
        // releases on the next frame.
        pb.tick(0.1, &held(&[60]), &[(60, 80)], 0, &synth);
        assert_eq!(pb.playhead_s, 1.5);
        pb.tick(0.1, &held(&[]), &[], 0, &synth);
        assert!(pb.playhead_s > 1.5, "gate must release after the hit");

        // Play the chord on time so the rest of the take free-runs out.
        let mut fired = false;
        for _ in 0..200 {
            if !matches!(pb.mode, Mode::Evaluation) {
                break;
            }
            let fire = !fired && pb.playhead_s >= 3.0;
            fired |= fire;
            let onsets: &[(u8, u8)] = if fire { &[(62, 80), (64, 80)] } else { &[] };
            pb.tick(0.1, &held(&[]), onsets, 0, &synth);
        }
        assert!(matches!(pb.mode, Mode::EvaluationReview));

        let r = pb.eval_result.as_ref().unwrap();
        assert_eq!((r.matched_count, r.missed_count, r.extra_press_count), (3, 0, 1));
        let p = r.pause_stats.as_ref().expect("setting on -> stats present");
        assert!(p.total_s > 0.0, "frozen time must be reported");
        assert_eq!(p.count, 1, "one continuous freeze, edge-counted once");
    }

    #[test]
    fn evaluation_jump_mid_pause_resets_pause_stats() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.eval.evaluate = Some(Who::Local);
        pb.eval.strictness = half_second_window();
        pb.eval.pause_on_miss = true;
        pb.start_evaluation(true, &synth);

        // Run into the first gate and accumulate some frozen time.
        for _ in 0..30 {
            pb.tick(0.1, &held(&[]), &[], 0, &synth);
        }
        assert_eq!(pb.playhead_s, 1.5);

        // Scrub mid-pause: the take restarts from the top.
        pb.jump_to(4.0, &synth);
        assert_eq!(pb.playhead_s, 0.0);

        // Perfect take this time — every note on time, no freezes — so the
        // restarted take's stats must be clean: the pre-jump pause history
        // did not survive the restart.
        let (mut fired_note, mut fired_chord) = (false, false);
        for _ in 0..200 {
            if !matches!(pb.mode, Mode::Evaluation) {
                break;
            }
            let (fire_note, fire_chord) =
                (!fired_note && pb.playhead_s >= 1.0, !fired_chord && pb.playhead_s >= 3.0);
            fired_note |= fire_note;
            fired_chord |= fire_chord;
            let onsets: &[(u8, u8)] = if fire_note {
                &[(60, 80)]
            } else if fire_chord {
                &[(62, 80), (64, 80)]
            } else {
                &[]
            };
            pb.tick(0.1, &held(&[]), onsets, 0, &synth);
        }
        assert!(matches!(pb.mode, Mode::EvaluationReview));

        let r = pb.eval_result.as_ref().unwrap();
        assert_eq!(r.missed_count, 0);
        let p = r.pause_stats.as_ref().expect("setting on -> stats present");
        assert_eq!((p.count, p.total_s), (0, 0.0), "pause history leaked through restart");
    }

    #[test]
    fn out_of_range_press_is_not_penalized_as_extra() {
        // H7: with a key range set, out-of-range notes aren't required — and a
        // press of one must not cost an extra-press penalty.
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.eval.evaluate = Some(Who::Local);
        pb.learn.key_range = Some((63, 80)); // excludes 60 & 62, includes 64
        pb.start_evaluation(true, &synth);
        // 60 is out of range (at 1.0); 64 is the in-range chord note (at 3.0).
        run_take(&mut pb, &synth, &[(1.0, &[(60, 80)], 0), (3.0, &[(64, 80)], 0)]);
        let r = pb.eval_result.as_ref().unwrap();
        assert_eq!(r.extra_press_count, 0, "out-of-range press must not be an extra");
        assert_eq!(r.matched_count, 1, "the one in-range required note matched");
    }

    #[test]
    fn tap_within_one_frame_is_a_closed_played_note_not_a_hold() {
        // M15: a note pressed and released inside one frame (held stays empty)
        // must record as a bounded played note, and a later tap of the same
        // pitch must record as a second note — not be swallowed.
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.eval.evaluate = Some(Who::Local);
        pb.start_evaluation(true, &synth);
        run_take(&mut pb, &synth, &[(1.0, &[(60, 80)], 0), (2.0, &[(60, 80)], 0)]);
        let played: Vec<&Note> =
            pb.display_score().tracks[1].notes.iter().filter(|n| n.midi == 60).collect();
        assert_eq!(played.len(), 2, "two taps must be two played notes");
        assert!(
            played.iter().all(|n| n.end_s < pb.score.duration_s - 1e-6),
            "a tap must not stretch to end-of-piece"
        );
    }

    #[test]
    fn wait_mode_does_not_skip_a_close_second_onset() {
        // M14: two onsets closer than one frame's advance — the gate must stop
        // at the second, not jump over it.
        let n = |start_s: f64, midi: u8| Note { start_s, end_s: start_s + 0.5, midi, velocity: 80 };
        let score = Score {
            tracks: [
                Track { notes: vec![n(1.0, 60), n(1.05, 62)], color: [0; 3], pedal_events: vec![] },
                Track { notes: vec![], color: [0; 3], pedal_events: vec![] },
            ],
            duration_s: 3.0,
            markers: vec![],
            first_marker_name: None,
            segments: vec![ScoreSegment { name: "a".into(), start_s: 0.0, end_s: 3.0 }],
            warning: None,
        };
        let (mut pb, synth) = (PlaybackEngine::new(score, PathBuf::from("t.mid")), Synth::disconnected());
        pb.mode = Mode::Learn;
        pb.learn.practice = [true, false];
        pb.learn.require_hold = false;
        // Free-run to the first onset.
        for _ in 0..20 {
            pb.tick(0.1, &held(&[]), &[], 0, &synth);
        }
        assert!((pb.playhead_s - 1.0).abs() < 1e-9);
        // Satisfy it with a big frame: must land at the 1.05 gate, not past it.
        pb.tick(0.1, &held(&[60]), &[], 0, &synth);
        assert!(pb.playhead_s <= 1.05 + 1e-9, "skipped the close onset: {}", pb.playhead_s);
    }

    #[test]
    fn evaluation_jump_restarts_the_take_from_the_top() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.eval.evaluate = Some(Who::Local);
        pb.start_evaluation(true, &synth);
        // Play the first note (on time: the playhead sits at 1.0 after the
        // tenth tick), then scrub: everything recorded is wiped and the
        // playhead is forced back to 0 regardless of the jump target.
        for _ in 0..9 {
            pb.tick(0.1, &held(&[]), &[], 0, &synth);
        }
        pb.tick(0.1, &held(&[60]), &[(60, 80)], 0, &synth);
        pb.jump_to(4.0, &synth);
        assert_eq!(pb.playhead_s, 0.0);
        assert!(matches!(pb.mode, Mode::Evaluation));
        // Finish without playing anything: all three notes are misses — the
        // pre-jump match did not survive.
        run_take(&mut pb, &synth, &[]);
        let r = pb.eval_result.as_ref().unwrap();
        assert_eq!((r.matched_count, r.missed_count), (0, 3));
    }
}
