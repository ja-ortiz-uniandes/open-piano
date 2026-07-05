//! Playback of a loaded score (see score.rs): one shared playhead driving
//! both tracks, in two modes.
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
//! The current segment (see `score::ScoreSegment`) can loop a chosen number
//! of times (or indefinitely) with a short silent pad before each repeat.
//!
//! All synth output goes to `Channel::Playback` via `Synth::note_on/off`
//! directly — never through main.rs's `synth_note_on`/`synth_note_off`,
//! whose mic-echo bookkeeping is sized for the two live channels only.

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::note::{midi_to_key_index, KEY_COUNT};
use crate::roll::Who;
use crate::score::{Note, Score};
use crate::synth::{Channel, Synth};

/// Silent breather inserted before each loop repeat, so the pickup of the
/// next pass doesn't slam into the tail of the previous one.
const LOOP_PAD_S: f64 = 5.0;

/// Nudge past a satisfied onset-gate so float equality can't re-trigger it.
const GATE_EPS: f64 = 1e-6;

/// Within this long after a segment's start, the "previous segment" button
/// goes back one segment instead of restarting the current one (the standard
/// media-player double-tap convention).
const PREV_SEGMENT_WINDOW_S: f64 = 0.5;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Listen,
    Learn,
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
            sounding: [BTreeSet::new(), BTreeSet::new()],
        }
    }

    /// Whether the player (not the synth) is responsible for this track.
    pub fn practiced(&self, who: Who) -> bool {
        self.mode == Mode::Learn && self.learn.practice[who.idx()]
    }

    fn active_at(&self, who: Who, t: f64) -> impl Iterator<Item = &Note> {
        self.score.tracks[who.idx()]
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
    /// numbers); `dt_s` is the frame's wall-clock delta.
    pub fn tick(&mut self, dt_s: f64, held: &BTreeSet<u8>, synth: &Synth) {
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
        self.playhead_s = (self.playhead_s + dt_s * self.speed as f64).min(self.score.duration_s);
        self.finished = self.playhead_s >= self.score.duration_s;
    }

    /// Continuous-hold gate: freeze in place unless everything required
    /// right now is held.
    fn hold_mode_advance(&mut self, dt_s: f64, held: &BTreeSet<u8>) {
        let required = self.required_set(self.playhead_s);
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
                    want.max(g + GATE_EPS).min(self.score.duration_s)
                } else {
                    g
                };
            }
        }
        self.finished = self.playhead_s >= self.score.duration_s;
    }

    /// Sound the unpracticed tracks: diff "should be sounding at the
    /// playhead" against "is sounding" and send the edges to the synth.
    /// Practiced tracks are the player's job — their sound (and their marks
    /// on the live history roll) comes from the real input path, unchanged.
    fn drive_auto(&mut self, synth: &Synth) {
        for who in [Who::Local, Who::Remote] {
            let idx = who.idx();
            if self.practiced(who) || !self.playing {
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
        self.playhead_s = t.clamp(0.0, self.score.duration_s);
        self.finished = self.playhead_s >= self.score.duration_s;
        // A stale pad countdown must not survive a jump — it would resume
        // padding logic against whatever segment the jump landed in.
        self.loop_state.pad_left_s = None;
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

    /// Local track: notes at [1,2) on 60 and [3,4) on 62+64 (a chord).
    /// Remote track: one note at [1.5, 2.5) on 70. Two segments split at 2.5.
    fn engine() -> PlaybackEngine {
        let score = Score {
            tracks: [
                Track {
                    notes: vec![
                        Note { start_s: 1.0, end_s: 2.0, midi: 60 },
                        Note { start_s: 3.0, end_s: 4.0, midi: 62 },
                        Note { start_s: 3.0, end_s: 4.0, midi: 64 },
                    ],
                    color: [0; 3],
                },
                Track {
                    notes: vec![Note { start_s: 1.5, end_s: 2.5, midi: 70 }],
                    color: [0; 3],
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
            pb.tick(0.1, &held(&[]), &synth);
        }
        assert!(pb.finished);
        assert_eq!(pb.playhead_s, 5.0);
    }

    #[test]
    fn speed_scales_the_playhead() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.speed = 0.5;
        pb.tick(1.0, &held(&[]), &synth);
        assert!((pb.playhead_s - 0.5).abs() < 1e-9);
    }

    #[test]
    fn hold_mode_freezes_without_required_notes() {
        let (mut pb, synth) = (engine(), Synth::disconnected());
        pb.mode = Mode::Learn;
        pb.learn.practice = [true, false];
        pb.jump_to(1.2, &synth); // inside local note 60
        pb.tick(0.1, &held(&[]), &synth);
        assert_eq!(pb.playhead_s, 1.2); // frozen: 60 not held
        pb.tick(0.1, &held(&[60]), &synth);
        assert!((pb.playhead_s - 1.3).abs() < 1e-9); // held -> advances
        // Extra notes don't block unless block_wrong.
        pb.tick(0.1, &held(&[60, 99]), &synth);
        assert!((pb.playhead_s - 1.4).abs() < 1e-9);
        pb.learn.block_wrong = true;
        pb.tick(0.1, &held(&[60, 99]), &synth);
        assert!((pb.playhead_s - 1.4).abs() < 1e-9); // strict: extra blocks
        // Remote track is not practiced: its active note (70) is not required.
        pb.learn.block_wrong = false;
        pb.tick(0.1, &held(&[60]), &synth);
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
            pb.tick(0.1, &held(&[]), &synth);
        }
        assert!((pb.playhead_s - 1.0).abs() < 1e-9);
        // Strike it: continues, and releasing immediately is fine.
        pb.tick(0.1, &held(&[60]), &synth);
        assert!(pb.playhead_s > 1.0);
        for _ in 0..30 {
            pb.tick(0.1, &held(&[]), &synth);
        }
        assert!((pb.playhead_s - 3.0).abs() < 1e-9); // next gate: the chord
        // The whole chord is required at once.
        pb.tick(0.1, &held(&[62]), &synth);
        assert!((pb.playhead_s - 3.0).abs() < 1e-9);
        pb.tick(0.1, &held(&[62, 64]), &synth);
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
            pb.tick(0.1, &held(&[]), &synth);
        }
        // Skipped the 1.0 onset (60 out of range); frozen at 3.0 needing only 64.
        assert!((pb.playhead_s - 3.0).abs() < 1e-9);
        pb.tick(0.1, &held(&[64]), &synth);
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
            pb.tick(0.1, &held(&[]), &synth);
        }
        assert!(pb.loop_state.pad_left_s.is_some());
        assert_eq!(pb.loop_state.remaining, Some(0));
        // The pad freezes the playhead just inside the segment.
        let parked = pb.playhead_s;
        pb.tick(1.0, &held(&[]), &synth);
        assert_eq!(pb.playhead_s, parked);
        // Burn the rest of the pad (3.9s left): snaps back to segment start.
        for _ in 0..4 {
            pb.tick(1.0, &held(&[]), &synth);
        }
        assert!((pb.playhead_s - 0.0).abs() < 1e-6);
        assert!(pb.loop_state.pad_left_s.is_none());
        // Second arrival at the end: remaining == 0 -> disengage, play on.
        for _ in 0..26 {
            pb.tick(0.1, &held(&[]), &synth);
        }
        assert!(!pb.loop_state.enabled);
        assert!(pb.playhead_s > 2.5);
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
}
