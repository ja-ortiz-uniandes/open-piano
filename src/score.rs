//! Loading a saved piano roll back in as a *score* for playback (see
//! playback.rs): a deterministic, read-only timeline of two tracks of notes,
//! as opposed to roll.rs's live, pausable recording clock.
//!
//! Reads both formats the app writes — a Standard MIDI File (+ its `.json`
//! color sidecar) or the self-contained `.jsonl` — plus, best-effort, a
//! `.segments.json` sidecar holding playback-time segment renames and
//! manually inserted segment breaks. The original `.mid`/`.jsonl` is never
//! rewritten by playback features; all playback-side edits live in that
//! sidecar.

use std::path::Path;

use crate::note::{MIDI_HIGH, MIDI_LOW};
use crate::roll::seconds;

/// The velocity assumed for score notes from files that don't carry one
/// (older exports, hand-made jsonl) — the constant this app's own exports
/// historically wrote.
const FILE_DEFAULT_VELOCITY: u8 = 64;

/// Minimum length forced onto a loaded note. A press+release captured in one
/// GUI frame (mic hysteresis flap, or any stall pumping On+Off together) is
/// saved with `start_s == end_s`; playback's activity test is half-open
/// (`start <= t < end`), so a zero-length note is never "active" — it never
/// sounds or lights a key, yet Evaluation still *requires* it. Clamp it here so
/// it behaves like any other short note (mirrors playback's `MIN_PLAYED_NOTE_S`
/// for its own recorded notes) (R37).
const MIN_SCORE_NOTE_S: f64 = 0.02;

/// Clamp every note to at least [`MIN_SCORE_NOTE_S`] long (R37).
fn ensure_min_note_len(notes: &mut [Note]) {
    for n in notes {
        if n.end_s < n.start_s + MIN_SCORE_NOTE_S {
            n.end_s = n.start_s + MIN_SCORE_NOTE_S;
        }
    }
}

/// One note of a loaded score. Times are seconds from the start of the file.
#[derive(Clone)]
pub struct Note {
    pub start_s: f64,
    pub end_s: f64,
    pub midi: u8,
    /// Note-on velocity (1..=127), the evaluation scorer's force target.
    /// [`FILE_DEFAULT_VELOCITY`] when the file carries none.
    pub velocity: u8,
}

/// One player's part: notes sorted by `start_s`, plus that player's display
/// color (from the color sidecar / jsonl header, or the app defaults).
#[derive(Clone)]
pub struct Track {
    pub notes: Vec<Note>,
    pub color: [u8; 3],
    /// Sustain-pedal (CC64) stream as `(time, level)` pairs, sorted by time —
    /// from SMF CC64 events / jsonl `"pedal"` events. Empty when the file
    /// carries no pedal data, which disables pedal evaluation for the track.
    pub pedal_events: Vec<(f64, u8)>,
}

impl Track {
    /// Whether the sustain pedal is down (CC64 >= 64) at `t` per this track's
    /// pedal stream — `None` when the track carries no pedal data at all.
    pub fn pedal_down_at(&self, t: f64) -> Option<bool> {
        if self.pedal_events.is_empty() {
            return None;
        }
        let level = self
            .pedal_events
            .iter()
            .take_while(|(at, _)| *at <= t)
            .last()
            .map_or(0, |(_, level)| *level);
        Some(level >= 64)
    }
}

/// A contiguous, named span of the score — the playback-side counterpart of
/// roll.rs's "instances", derived from the file's markers plus any manual
/// breaks from the `.segments.json` sidecar. Always covers the whole score
/// end to end with no gaps.
#[derive(Clone)]
pub struct ScoreSegment {
    pub name: String,
    pub start_s: f64,
    pub end_s: f64,
}

/// A loaded, immutable-notes score. Segment *names/boundaries* are the one
/// mutable part (renames + manual breaks), persisted via `save_segment_names`.
pub struct Score {
    /// Indexed via `Who::idx()`: `[local, remote]`.
    pub tracks: [Track; 2],
    pub duration_s: f64,
    /// (time, marker text) pairs from the file, sorted — render/segment
    /// context only, not used for playback timing.
    pub markers: Vec<(f64, String)>,
    /// Marker text at exactly t=0, if any (a custom name for instance 1).
    pub first_marker_name: Option<String>,
    /// Derived from `markers` (+ sidecar breaks); never empty.
    pub segments: Vec<ScoreSegment>,
    /// One-line load caveat surfaced in the UI (tempo changes ignored,
    /// extra tracks dropped, ...). Loading still succeeds.
    pub warning: Option<String>,
}

impl Score {
    pub fn load(path: &Path) -> Result<Score, String> {
        // Case-insensitive: Windows (the app's only platform) happily hands us
        // `.MID`/`.JSONL`, and a foreign file may be upper-cased.
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());
        let mut score = match ext.as_deref() {
            Some("jsonl") => Self::load_jsonl(path)?,
            Some("mid") | Some("midi") => Self::load_midi(path)?,
            _ => return Err("expected a .mid, .midi, or .jsonl file".into()),
        };
        if score.tracks.iter().all(|t| t.notes.is_empty()) {
            return Err("the file contains no notes".into());
        }
        score.segments =
            Self::build_segments(score.first_marker_name.clone(), &score.markers, score.duration_s);
        score.apply_segment_sidecar(path); // best-effort; missing sidecar is fine
        Ok(score)
    }

    // -- segments ----------------------------------------------------------

    /// Turn (marker time, text) pairs + the score's total length into
    /// contiguous segments: [0, m0), [m0, m1), ..., [mN, duration]. Each
    /// segment's default name is its marker's own text when one exists
    /// (round-tripping names set live while recording) — "Segment N" is only
    /// a fallback for the first span (no tick-0 marker) or a markerless
    /// foreign file, which still yields exactly one whole-piece segment, so
    /// callers never special-case "no segments".
    fn build_segments(
        first_name: Option<String>,
        markers: &[(f64, String)],
        duration_s: f64,
    ) -> Vec<ScoreSegment> {
        let mut starts: Vec<(f64, Option<String>)> = vec![(0.0, first_name)];
        starts.extend(
            markers
                .iter()
                .filter(|(t, _)| *t > 1e-9 && *t < duration_s)
                .map(|(t, n)| (*t, (!n.is_empty()).then(|| n.clone()))),
        );
        starts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
        let mut ends: Vec<f64> = starts.iter().skip(1).map(|(t, _)| *t).collect();
        ends.push(duration_s);
        starts
            .into_iter()
            .zip(ends)
            .enumerate()
            .map(|(i, ((start_s, name), end_s))| ScoreSegment {
                name: name.unwrap_or_else(|| format!("Segment {}", i + 1)),
                start_s,
                end_s,
            })
            .collect()
    }

    /// Split whichever segment currently contains `at`. No-op outside
    /// (0, duration) or exactly on an existing boundary, so replaying the
    /// sidecar's breaks (which include the file's own markers) is harmless.
    pub fn insert_segment_break(&mut self, at: f64) {
        if at <= 1e-9 || at >= self.duration_s - 1e-9 {
            return;
        }
        if let Some(i) = self
            .segments
            .iter()
            .position(|s| s.start_s + 1e-9 < at && at < s.end_s - 1e-9)
        {
            let end = self.segments[i].end_s;
            self.segments[i].end_s = at;
            // Cosmetic placeholder; renumbered from scratch on the next load.
            let name = format!("Segment {}", self.segments.len() + 1);
            self.segments.insert(i + 1, ScoreSegment { name, start_s: at, end_s: end });
        }
    }

    /// Apply `<stem>.segments.json` if present: `{"names": [...],
    /// "extra_breaks": [...]}`. Breaks are replayed first (idempotent), then
    /// names applied by index — only when the count matches the resulting
    /// segment list, so a sidecar from a differently-segmented file with the
    /// same name degrades to defaults instead of mislabeling. All failure
    /// modes (missing file, bad JSON, wrong shapes) silently keep defaults.
    fn apply_segment_sidecar(&mut self, path: &Path) {
        let Ok(text) = std::fs::read_to_string(path.with_extension("segments.json")) else {
            return;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            return;
        };
        if let Some(breaks) = v["extra_breaks"].as_array() {
            for b in breaks.iter().filter_map(|b| b.as_f64()) {
                self.insert_segment_break(b);
            }
        }
        if let Some(names) = v["names"].as_array() {
            if names.len() == self.segments.len() {
                for (seg, name) in self.segments.iter_mut().zip(names) {
                    if let Some(n) = name.as_str() {
                        seg.name = n.to_string();
                    }
                }
            }
        }
    }

    /// Persist segment names + manual breaks next to the score file — called
    /// explicitly by the UI right after a rename or break insertion (no
    /// hidden IO in mutators, matching the roll::save_* convention). Breaks
    /// are stored as *every* boundary except the implicit 0.0 start; on load
    /// the ones that coincide with the file's own markers replay as no-ops.
    pub fn save_segment_sidecar(&self, path: &Path) -> std::io::Result<()> {
        let names: Vec<&str> = self.segments.iter().map(|s| s.name.as_str()).collect();
        let breaks: Vec<f64> = self.segments.iter().skip(1).map(|s| s.start_s).collect();
        let json = serde_json::json!({ "names": names, "extra_breaks": breaks });
        std::fs::write(path.with_extension("segments.json"), format!("{json}\n"))
    }

    // -- MIDI --------------------------------------------------------------

    fn load_midi(path: &Path) -> Result<Score, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let smf = midly::Smf::parse(&bytes).map_err(|e| format!("not a valid MIDI file: {e}"))?;

        let ppq = match smf.header.timing {
            midly::Timing::Metrical(t) => t.as_int(),
            midly::Timing::Timecode(..) => {
                return Err("SMPTE-timed MIDI files aren't supported".into());
            }
        };
        // Division 0 parses as legal metrical timing but makes ticks→seconds
        // `0/0 = NaN` (and later events +∞), so `duration_s` goes infinite and
        // an Evaluation take could never end. Reject it like SMPTE timing.
        if ppq == 0 {
            return Err("MIDI file has an invalid ticks-per-quarter of 0".into());
        }

        // Single constant tempo: the *earliest-in-time* Tempo event wins (500000
        // µs/quarter if none — the MIDI default). This app's own exports are
        // always single-tempo; a foreign multi-tempo file plays at its initial
        // tempo throughout, flagged via `warning`. We scan by absolute tick, not
        // track-scan order: in a format-1 file the real tick-0 conductor tempo
        // can live in a *later* track while an earlier track carries only a
        // mid-piece change — taking the first-seen would scale the whole piece by
        // the wrong tempo (R14). Ties (same abs_tick) keep the first track's.
        let mut tempo: Option<(u32, u32)> = None; // (abs_tick, us_per_quarter)
        let mut tempo_changes = 0u32;
        for track in &smf.tracks {
            let mut abs_ticks = 0u32;
            for ev in track {
                abs_ticks = abs_ticks.saturating_add(ev.delta.as_int());
                if let midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(us)) = ev.kind {
                    tempo_changes += 1;
                    let earlier = match tempo {
                        None => true,
                        Some((best_tick, _)) => abs_ticks < best_tick,
                    };
                    if earlier {
                        tempo = Some((abs_ticks, us.as_int()));
                    }
                }
            }
        }
        let tempo = tempo.map(|(_, us)| us).unwrap_or(500_000);
        let mut warnings: Vec<String> = Vec::new();
        if tempo_changes > 1 {
            warnings.push("tempo changes ignored (initial tempo used throughout)".into());
        }

        let mut markers: Vec<(f64, String)> = Vec::new();
        let mut first_marker_name: Option<String> = None;
        let mut tracks: Vec<(Vec<Note>, Vec<(f64, u8)>)> = Vec::new();
        let mut duration_s: f64 = 0.0;

        for track in &smf.tracks {
            let mut abs_ticks = 0u32;
            let mut notes: Vec<Note> = Vec::new();
            let mut pedal: Vec<(f64, u8)> = Vec::new();
            // Open-note index per MIDI number, same idea as roll::Roll::note.
            let mut open: [Option<usize>; 128] = [None; 128];
            for ev in track {
                abs_ticks = abs_ticks.saturating_add(ev.delta.as_int());
                let t = seconds(abs_ticks, tempo, ppq);
                match ev.kind {
                    midly::TrackEventKind::Midi { message, .. } => match message {
                        midly::MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                            let midi = key.as_int();
                            // Outside the 88-key range: no lane to draw and
                            // no key to press — drop (its Off is then a no-op).
                            if midi < MIDI_LOW || midi > MIDI_HIGH {
                                continue;
                            }
                            if open[midi as usize].is_none() {
                                open[midi as usize] = Some(notes.len());
                                notes.push(Note {
                                    start_s: t,
                                    end_s: t,
                                    midi,
                                    velocity: vel.as_int(),
                                });
                            }
                        }
                        // NoteOn with velocity 0 is a NoteOff by convention.
                        midly::MidiMessage::NoteOn { key, .. }
                        | midly::MidiMessage::NoteOff { key, .. } => {
                            if let Some(i) = open[key.as_int() as usize].take() {
                                notes[i].end_s = t;
                            }
                        }
                        // Sustain pedal (CC64): the evaluation scorer's
                        // pedal-intent stream. Other controllers are ignored.
                        midly::MidiMessage::Controller { controller, value }
                            if controller.as_int() == 64 =>
                        {
                            pedal.push((t, value.as_int()));
                        }
                        _ => {}
                    },
                    midly::TrackEventKind::Meta(midly::MetaMessage::Marker(text)) => {
                        let text = String::from_utf8_lossy(text).into_owned();
                        if abs_ticks == 0 {
                            first_marker_name.get_or_insert(text);
                        } else {
                            markers.push((t, text));
                        }
                    }
                    _ => {}
                }
                duration_s = duration_s.max(t);
            }
            // Anything left open closes at the track's last event.
            let track_end = seconds(abs_ticks, tempo, ppq);
            for slot in open.into_iter().flatten() {
                notes[slot].end_s = notes[slot].end_s.max(track_end);
            }
            // A same-tick on/off (or a zero-duration authored note) would never
            // sound or light a key — give it a minimum length (R37).
            ensure_min_note_len(&mut notes);
            // A track with pedal data but no notes is dropped along with any
            // other noteless track — pedal only means anything under notes.
            if !notes.is_empty() {
                tracks.push((notes, pedal));
            }
        }
        markers.sort_by(|a, b| a.0.total_cmp(&b.0));

        if tracks.len() > 2 {
            warnings.push(format!("loaded 2 of {} note tracks", tracks.len()));
            tracks.truncate(2);
        }
        let mut it = tracks.into_iter();
        let (local, remote) = (it.next().unwrap_or_default(), it.next().unwrap_or_default());

        let [local_color, remote_color] = load_color_sidecar(path);
        Ok(Score {
            tracks: [
                Track { notes: local.0, color: local_color, pedal_events: local.1 },
                Track { notes: remote.0, color: remote_color, pedal_events: remote.1 },
            ],
            duration_s,
            markers,
            first_marker_name,
            segments: Vec::new(), // filled in by `load`
            warning: (!warnings.is_empty()).then(|| warnings.join("; ")),
        })
    }

    // -- JSONL -------------------------------------------------------------

    /// The self-contained format: header line with colors, then time-sorted
    /// event lines. Times are already real seconds — no tempo math at all.
    fn load_jsonl(path: &Path) -> Result<Score, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());

        let header: serde_json::Value = lines
            .next()
            .ok_or("empty file")
            .and_then(|l| serde_json::from_str(l).map_err(|_| "malformed header line"))?;
        let color_or = |key: &str, default: [u8; 3]| -> [u8; 3] {
            header[key]
                .as_array()
                .and_then(|a| {
                    let v: Vec<u8> = a.iter().filter_map(|x| x.as_u64().map(|n| n as u8)).collect();
                    v.try_into().ok()
                })
                .unwrap_or(default)
        };
        let local_color = color_or("local_color", crate::DEFAULT_LOCAL_COLOR);
        let remote_color = color_or("remote_color", crate::DEFAULT_REMOTE_COLOR);
        let first_marker_name = header["first_instance_name"].as_str().map(String::from);

        let mut tracks: [Vec<Note>; 2] = [Vec::new(), Vec::new()];
        let mut pedal: [Vec<(f64, u8)>; 2] = [Vec::new(), Vec::new()];
        let mut open: [[Option<usize>; 128]; 2] = [[None; 128]; 2];
        let mut markers: Vec<(f64, String)> = Vec::new();
        let mut duration_s: f64 = 0.0;
        let mut sep_count = 0usize;

        for line in lines {
            let Ok(ev) = serde_json::from_str::<serde_json::Value>(line) else {
                return Err("malformed event line".into());
            };
            let Some(t) = ev["t"].as_f64() else { continue };
            // Reject non-finite / negative times: a negative start would drag
            // the playhead behind 0 (pause-on-miss clamps the gate to it), and
            // NaN/∞ would poison `duration_s` and all downstream timing.
            if !t.is_finite() || t < 0.0 {
                continue;
            }
            duration_s = duration_s.max(t);
            match ev["e"].as_str() {
                Some("sep") => {
                    sep_count += 1;
                    let name = ev["name"]
                        .as_str()
                        .map(String::from)
                        .unwrap_or_else(|| format!("instance {}", sep_count + 1));
                    markers.push((t, name));
                }
                Some(kind @ ("on" | "off")) => {
                    let (Some(n), Some(who)) = (ev["n"].as_u64(), ev["who"].as_str()) else {
                        continue;
                    };
                    // Range-check on the *u64* before narrowing: `n as u8` would
                    // wrap (note 300 → 44) and sneak past an after-the-cast check.
                    if n < MIDI_LOW as u64 || n > MIDI_HIGH as u64 {
                        continue;
                    }
                    let (midi, w) = (n as u8, if who == "r" { 1 } else { 0 });
                    if kind == "on" {
                        if open[w][midi as usize].is_none() {
                            open[w][midi as usize] = Some(tracks[w].len());
                            let velocity = ev["v"]
                                // Saturate on the *u64* before narrowing: `v as u8`
                                // would wrap (300 → 44, 256 → 0) and sneak past the
                                // clamp — the same class the `"n"` cast above fixes (F30).
                                .as_u64()
                                .map_or(FILE_DEFAULT_VELOCITY, |v| v.clamp(1, 127) as u8);
                            tracks[w].push(Note { start_s: t, end_s: t, midi, velocity });
                        }
                    } else if let Some(i) = open[w][midi as usize].take() {
                        tracks[w][i].end_s = t;
                    }
                }
                // Sustain-pedal (CC64) level events, as written by
                // roll::save_jsonl since pedal capture landed.
                Some("pedal") => {
                    let (Some(v), Some(who)) = (ev["v"].as_u64(), ev["who"].as_str()) else {
                        continue;
                    };
                    let w = if who == "r" { 1 } else { 0 };
                    // Saturate before narrowing (F30): `v as u8` would wrap
                    // (pedal 300 → 44 < 64), reporting an intended full press as
                    // pedal-up and corrupting `required_pedal_down` targets.
                    pedal[w].push((t, v.min(127) as u8));
                }
                _ => {}
            }
        }
        for w in 0..2 {
            for slot in open[w].into_iter().flatten() {
                tracks[w][slot].end_s = tracks[w][slot].end_s.max(duration_s);
            }
        }
        // Notes and pedal events must be sorted by time: `gate_at_or_after`
        // (wait-mode gating), `Track::pedal_down_at` (pedal grading), and the
        // segment builder all assume it. A hand-written / out-of-order jsonl
        // otherwise silently mis-gates and mis-grades (F15). `load_midi` already
        // emits them in order; sorting a sorted list is cheap.
        for w in 0..2 {
            tracks[w].sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
            pedal[w].sort_by(|a, b| a.0.total_cmp(&b.0));
            // A same-`t` on/off pair loads as a zero-length note that never
            // sounds/lights but is still required by Evaluation — clamp it (R37).
            ensure_min_note_len(&mut tracks[w]);
        }
        // Markers drive segment boundaries; `build_segments` assumes them
        // sorted (as `load_midi` guarantees). A hand-written / out-of-order
        // jsonl otherwise yields `end_s < start_s` segments.
        markers.sort_by(|a, b| a.0.total_cmp(&b.0));

        let [local, remote] = tracks;
        let [local_pedal, remote_pedal] = pedal;
        Ok(Score {
            tracks: [
                Track { notes: local, color: local_color, pedal_events: local_pedal },
                Track { notes: remote, color: remote_color, pedal_events: remote_pedal },
            ],
            duration_s,
            markers,
            first_marker_name,
            segments: Vec::new(), // filled in by `load`
            warning: None,
        })
    }
}

/// Colors from the `.mid`'s `<stem>.json` sidecar, or the app defaults when
/// it's missing/malformed (forgiving: a lost sidecar shouldn't block a load).
fn load_color_sidecar(path: &Path) -> [[u8; 3]; 2] {
    let defaults = [crate::DEFAULT_LOCAL_COLOR, crate::DEFAULT_REMOTE_COLOR];
    let Ok(text) = std::fs::read_to_string(path.with_extension("json")) else {
        return defaults;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return defaults;
    };
    let get = |key: &str, default: [u8; 3]| -> [u8; 3] {
        v[key]
            .as_array()
            .and_then(|a| {
                let c: Vec<u8> = a.iter().filter_map(|x| x.as_u64().map(|n| n as u8)).collect();
                c.try_into().ok()
            })
            .unwrap_or(default)
    };
    [get("local_color", defaults[0]), get("remote_color", defaults[1])]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::note::NoteMsg;
    use crate::roll::{self, Roll, Who};
    use std::time::{Duration, Instant};

    fn t(base: Instant, s: f64) -> Instant {
        base + Duration::from_secs_f64(s)
    }

    /// Two-player roll with a named first instance and one (renamed) resume.
    fn sample_roll() -> Roll {
        let base = Instant::now();
        let mut roll = Roll::new();
        roll.tick(t(base, 0.0));
        roll.note(Who::Local, NoteMsg::On(60, 100), [220, 60, 60]);
        roll.pedal(Who::Local, 90);
        roll.rename_current_instance("warmup".into());
        roll.tick(t(base, 0.5));
        roll.note(Who::Local, NoteMsg::Off(60), [220, 60, 60]);
        roll.pedal(Who::Local, 0);
        roll.note(Who::Remote, NoteMsg::On(64, 100), [60, 110, 230]);
        roll.tick(t(base, 1.0));
        roll.note(Who::Remote, NoteMsg::Off(64), [60, 110, 230]);
        roll.tick(t(base, 60.0)); // idle -> pause
        roll.note(Who::Local, NoteMsg::On(62, 100), [220, 60, 60]); // resume, separator
        roll.rename_current_instance("piece".into());
        roll.tick(t(base, 61.0));
        roll.note(Who::Local, NoteMsg::Off(62), [220, 60, 60]);
        roll
    }

    #[test]
    fn midi_round_trips_notes_names_and_colors() {
        let dir = std::env::temp_dir().join(format!("op-score-mid-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.mid");
        roll::save_midi(&sample_roll(), [1, 2, 3], [4, 5, 6], &path).unwrap();

        let score = Score::load(&path).expect("own export must load");
        assert_eq!(score.tracks[0].notes.len(), 2); // local: 60, 62
        assert_eq!(score.tracks[1].notes.len(), 1); // remote: 64
        assert_eq!(score.tracks[0].color, [1, 2, 3]);
        assert_eq!(score.tracks[1].color, [4, 5, 6]);
        let n0 = &score.tracks[0].notes[0];
        assert_eq!(n0.midi, 60);
        assert!((n0.start_s - 0.0).abs() < 2e-3 && (n0.end_s - 0.5).abs() < 2e-3);
        assert_eq!(n0.velocity, 100); // real velocity survives the round trip
        // The pedal press round-trips as a CC64 pair: 90 at 0.0, 0 at 0.5.
        let pe = &score.tracks[0].pedal_events;
        assert_eq!(pe.len(), 2);
        assert_eq!((pe[0].1, pe[1].1), (90, 0));
        assert!((pe[0].0 - 0.0).abs() < 2e-3 && (pe[1].0 - 0.5).abs() < 2e-3);
        assert_eq!(score.tracks[0].pedal_down_at(0.2), Some(true));
        assert_eq!(score.tracks[0].pedal_down_at(0.7), Some(false));
        assert_eq!(score.tracks[1].pedal_down_at(0.2), None); // no data at all
        // Both instance names survived: tick-0 marker + the separator marker.
        assert_eq!(score.segments.len(), 2);
        assert_eq!(score.segments[0].name, "warmup");
        assert_eq!(score.segments[1].name, "piece");
        assert!((score.segments[1].end_s - score.duration_s).abs() < 1e-9);
        assert!(score.warning.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn jsonl_round_trips_and_breaks_persist_via_sidecar() {
        let dir = std::env::temp_dir().join(format!("op-score-jsonl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.jsonl");
        roll::save_jsonl(&sample_roll(), [7, 8, 9], [10, 11, 12], &path).unwrap();

        let mut score = Score::load(&path).unwrap();
        assert_eq!(score.tracks[0].color, [7, 8, 9]);
        assert_eq!(score.tracks[0].notes[0].velocity, 100);
        assert_eq!(score.tracks[0].pedal_events, vec![(0.0, 90), (0.5, 0)]);
        assert_eq!(score.segments.len(), 2);
        assert_eq!(score.segments[0].name, "warmup");
        assert_eq!(score.segments[1].name, "piece");

        // Manual break + rename, persisted through the sidecar.
        let mid = (score.segments[1].start_s + score.duration_s) / 2.0;
        score.insert_segment_break(mid);
        assert_eq!(score.segments.len(), 3);
        score.segments[2].name = "coda".into();
        score.save_segment_sidecar(&path).unwrap();

        let again = Score::load(&path).unwrap();
        let names: Vec<&str> = again.segments.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["warmup", "piece", "coda"]);
        assert!((again.segments[2].start_s - mid).abs() < 1e-9);
        // Boundaries stay contiguous.
        for w in again.segments.windows(2) {
            assert!((w[0].end_s - w[1].start_s).abs() < 1e-9);
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn jsonl_notes_and_pedal_sorted_and_values_saturated() {
        // Hand-written, out-of-order jsonl with hostile velocity/pedal values.
        // Notes and pedal must load time-sorted (F15) and their values must
        // saturate before narrowing, not wrap (F30: `300 as u8` == 44).
        let jsonl = concat!(
            "{}\n",                                                        // header
            "{\"t\":2.0,\"e\":\"on\",\"n\":64,\"v\":300,\"who\":\"l\"}\n", // later note first
            "{\"t\":2.5,\"e\":\"off\",\"n\":64,\"who\":\"l\"}\n",
            "{\"t\":0.5,\"e\":\"on\",\"n\":60,\"v\":100,\"who\":\"l\"}\n",  // earlier note second
            "{\"t\":1.0,\"e\":\"off\",\"n\":60,\"who\":\"l\"}\n",
            "{\"t\":1.5,\"e\":\"pedal\",\"v\":300,\"who\":\"l\"}\n",        // out-of-order pedal
            "{\"t\":0.2,\"e\":\"pedal\",\"v\":80,\"who\":\"l\"}\n",
        );
        let dir = std::env::temp_dir().join(format!("op-score-sort-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hand.jsonl");
        std::fs::write(&path, jsonl).unwrap();

        let score = Score::load(&path).unwrap();
        let notes = &score.tracks[0].notes;
        assert_eq!(notes.len(), 2);
        // Sorted by start_s (F15): note 60 (t=0.5) before note 64 (t=2.0).
        assert_eq!(notes[0].midi, 60);
        assert_eq!(notes[1].midi, 64);
        // v:300 saturates to 127, not `300 as u8` == 44 (F30).
        assert_eq!(notes[1].velocity, 127);
        // Pedal sorted by time (F15), values saturated (F30): 300 -> 127 (down).
        let pedal = &score.tracks[0].pedal_events;
        assert_eq!(pedal.len(), 2);
        assert_eq!(pedal[0], (0.2, 80));
        assert_eq!(pedal[1], (1.5, 127));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn zero_length_notes_are_given_a_minimum_length_at_load() {
        // A same-`t` on/off pair (a mic-hysteresis flap, or a stall pumping both
        // in one frame) would load as `start_s == end_s` — never active, so it
        // never sounds or lights a key, yet Evaluation still requires it. It must
        // be clamped to a minimum length at load (R37).
        let jsonl = concat!(
            "{}\n",
            "{\"t\":1.0,\"e\":\"on\",\"n\":60,\"v\":100,\"who\":\"l\"}\n",
            "{\"t\":1.0,\"e\":\"off\",\"n\":60,\"who\":\"l\"}\n",
        );
        let dir = std::env::temp_dir().join(format!("op-score-zerolen-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zero.jsonl");
        std::fs::write(&path, jsonl).unwrap();

        let score = Score::load(&path).unwrap();
        let notes = &score.tracks[0].notes;
        assert_eq!(notes.len(), 1);
        assert!(
            notes[0].end_s >= notes[0].start_s + MIN_SCORE_NOTE_S,
            "zero-length note should be clamped: {} -> {}",
            notes[0].start_s,
            notes[0].end_s
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_zero_division_midi() {
        use midly::num::{u15, u28, u4, u7};
        use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind};
        // A legally-parsing but nonsensical division-0 file: loading must fail
        // rather than produce NaN/∞ note times (H6).
        let mut smf = Smf::new(Header::new(Format::SingleTrack, Timing::Metrical(u15::from(0))));
        smf.tracks.push(vec![
            TrackEvent {
                delta: u28::from(0),
                kind: TrackEventKind::Midi {
                    channel: u4::from(0),
                    message: MidiMessage::NoteOn { key: u7::from(60), vel: u7::from(100) },
                },
            },
            TrackEvent {
                delta: u28::from(10),
                kind: TrackEventKind::Midi {
                    channel: u4::from(0),
                    message: MidiMessage::NoteOff { key: u7::from(60), vel: u7::from(0) },
                },
            },
            TrackEvent { delta: u28::from(0), kind: TrackEventKind::Meta(MetaMessage::EndOfTrack) },
        ]);
        let dir = std::env::temp_dir().join(format!("op-score-ppq0-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("z.mid");
        smf.save(&path).unwrap();
        assert!(Score::load(&path).is_err(), "division-0 file must be rejected");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn initial_tempo_is_earliest_in_time_not_first_track_scanned() {
        use midly::num::{u15, u24, u28, u4, u7};
        use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind};
        // Format-1 file: track 0 carries only a *mid-piece* tempo change (at a
        // late tick), while the real tick-0 conductor tempo lives in track 1. The
        // whole piece must scale by the tick-0 tempo (500000 µs/qtr = 120 BPM),
        // not the first-scanned one (R14). A note at tick 480 (one quarter at
        // ppq 480) must therefore land at 0.5 s, not 0.25 s.
        let ppq = 480u16;
        let mut smf = Smf::new(Header::new(Format::Parallel, Timing::Metrical(u15::from(ppq))));
        // Track 0: a mid-piece tempo (250000 = 240 BPM) at a late tick, plus the note.
        smf.tracks.push(vec![
            TrackEvent {
                delta: u28::from(4800),
                kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::from(250_000))),
            },
            TrackEvent { delta: u28::from(0), kind: TrackEventKind::Meta(MetaMessage::EndOfTrack) },
        ]);
        // Track 1: the real conductor tempo at tick 0, then a one-quarter note.
        smf.tracks.push(vec![
            TrackEvent {
                delta: u28::from(0),
                kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::from(500_000))),
            },
            TrackEvent {
                delta: u28::from(0),
                kind: TrackEventKind::Midi {
                    channel: u4::from(0),
                    message: MidiMessage::NoteOn { key: u7::from(60), vel: u7::from(100) },
                },
            },
            TrackEvent {
                delta: u28::from(480),
                kind: TrackEventKind::Midi {
                    channel: u4::from(0),
                    message: MidiMessage::NoteOff { key: u7::from(60), vel: u7::from(0) },
                },
            },
            TrackEvent { delta: u28::from(0), kind: TrackEventKind::Meta(MetaMessage::EndOfTrack) },
        ]);
        let dir = std::env::temp_dir().join(format!("op-score-tempo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.mid");
        smf.save(&path).unwrap();

        let score = Score::load(&path).unwrap();
        let note = score.tracks.iter().flat_map(|t| &t.notes).next().expect("a note");
        // 120 BPM: one quarter = 0.5 s. (At the wrong 240 BPM it would be 0.25 s.)
        assert!((note.end_s - note.start_s - 0.5).abs() < 1e-3, "duration {}", note.end_s - note.start_s);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn segment_breaks_are_idempotent_and_bounded() {
        let mut score = Score {
            tracks: [
                Track {
                    notes: vec![Note { start_s: 0.0, end_s: 10.0, midi: 60, velocity: 64 }],
                    color: [0; 3],
                    pedal_events: Vec::new(),
                },
                Track { notes: Vec::new(), color: [0; 3], pedal_events: Vec::new() },
            ],
            duration_s: 10.0,
            markers: Vec::new(),
            first_marker_name: None,
            segments: Score::build_segments(None, &[], 10.0),
            warning: None,
        };
        assert_eq!(score.segments.len(), 1); // markerless -> one whole-piece segment
        score.insert_segment_break(5.0);
        score.insert_segment_break(5.0); // duplicate: no-op
        score.insert_segment_break(0.0); // at start: no-op
        score.insert_segment_break(10.0); // at end: no-op
        assert_eq!(score.segments.len(), 2);
        assert_eq!(score.segments[1].start_s, 5.0);
    }
}
