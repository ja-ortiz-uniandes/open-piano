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

/// One note of a loaded score. Times are seconds from the start of the file.
pub struct Note {
    pub start_s: f64,
    pub end_s: f64,
    pub midi: u8,
}

/// One player's part: notes sorted by `start_s`, plus that player's display
/// color (from the color sidecar / jsonl header, or the app defaults).
pub struct Track {
    pub notes: Vec<Note>,
    pub color: [u8; 3],
}

/// A contiguous, named span of the score — the playback-side counterpart of
/// roll.rs's "instances", derived from the file's markers plus any manual
/// breaks from the `.segments.json` sidecar. Always covers the whole score
/// end to end with no gaps.
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
        let mut score = match path.extension().and_then(|e| e.to_str()) {
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

        // Single constant tempo: the first Tempo event anywhere wins (500000
        // µs/quarter if none — the MIDI default). This app's own exports are
        // always single-tempo; a foreign multi-tempo file plays at its
        // initial tempo throughout, flagged via `warning`.
        let mut tempo: Option<u32> = None;
        let mut tempo_changes = 0u32;
        for track in &smf.tracks {
            for ev in track {
                if let midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(us)) = ev.kind {
                    tempo_changes += 1;
                    if tempo.is_none() {
                        tempo = Some(us.as_int());
                    }
                }
            }
        }
        let tempo = tempo.unwrap_or(500_000);
        let mut warnings: Vec<String> = Vec::new();
        if tempo_changes > 1 {
            warnings.push("tempo changes ignored (initial tempo used throughout)".into());
        }

        let mut markers: Vec<(f64, String)> = Vec::new();
        let mut first_marker_name: Option<String> = None;
        let mut tracks: Vec<Vec<Note>> = Vec::new();
        let mut duration_s: f64 = 0.0;

        for track in &smf.tracks {
            let mut abs_ticks = 0u32;
            let mut notes: Vec<Note> = Vec::new();
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
                                notes.push(Note { start_s: t, end_s: t, midi });
                            }
                        }
                        // NoteOn with velocity 0 is a NoteOff by convention.
                        midly::MidiMessage::NoteOn { key, .. }
                        | midly::MidiMessage::NoteOff { key, .. } => {
                            if let Some(i) = open[key.as_int() as usize].take() {
                                notes[i].end_s = t;
                            }
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
            if !notes.is_empty() {
                tracks.push(notes);
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
                Track { notes: local, color: local_color },
                Track { notes: remote, color: remote_color },
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
        let mut open: [[Option<usize>; 128]; 2] = [[None; 128]; 2];
        let mut markers: Vec<(f64, String)> = Vec::new();
        let mut duration_s: f64 = 0.0;
        let mut sep_count = 0usize;

        for line in lines {
            let Ok(ev) = serde_json::from_str::<serde_json::Value>(line) else {
                return Err("malformed event line".into());
            };
            let Some(t) = ev["t"].as_f64() else { continue };
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
                    let (midi, w) = (n as u8, if who == "r" { 1 } else { 0 });
                    if midi < MIDI_LOW || midi > MIDI_HIGH {
                        continue;
                    }
                    if kind == "on" {
                        if open[w][midi as usize].is_none() {
                            open[w][midi as usize] = Some(tracks[w].len());
                            tracks[w].push(Note { start_s: t, end_s: t, midi });
                        }
                    } else if let Some(i) = open[w][midi as usize].take() {
                        tracks[w][i].end_s = t;
                    }
                }
                _ => {}
            }
        }
        for w in 0..2 {
            for slot in open[w].into_iter().flatten() {
                tracks[w][slot].end_s = tracks[w][slot].end_s.max(duration_s);
            }
        }

        let [local, remote] = tracks;
        Ok(Score {
            tracks: [
                Track { notes: local, color: local_color },
                Track { notes: remote, color: remote_color },
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
        roll.note(Who::Local, NoteMsg::On(60), [220, 60, 60]);
        roll.rename_current_instance("warmup".into());
        roll.tick(t(base, 0.5));
        roll.note(Who::Local, NoteMsg::Off(60), [220, 60, 60]);
        roll.note(Who::Remote, NoteMsg::On(64), [60, 110, 230]);
        roll.tick(t(base, 1.0));
        roll.note(Who::Remote, NoteMsg::Off(64), [60, 110, 230]);
        roll.tick(t(base, 60.0)); // idle -> pause
        roll.note(Who::Local, NoteMsg::On(62), [220, 60, 60]); // resume, separator
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
    fn segment_breaks_are_idempotent_and_bounded() {
        let mut score = Score {
            tracks: [
                Track { notes: vec![Note { start_s: 0.0, end_s: 10.0, midi: 60 }], color: [0; 3] },
                Track { notes: Vec::new(), color: [0; 3] },
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
