//! User preferences: the app's scattered compile-time tunables, consolidated
//! into one serde-serializable struct that survives restarts.
//!
//! Persisted as JSON at `%LOCALAPPDATA%\open-piano\preferences.json` — the same
//! per-user directory `bundle.rs` extracts the ONNX Runtime into. Loading is
//! fault-tolerant by design: a missing file, a parse error, or a field that a
//! newer build added but an older JSON lacks all fall back to the compile-time
//! default (`#[serde(default)]` on every field), so the file is always
//! forward/backward compatible and a corrupt file never blocks startup.
//!
//! The Edit ▸ Preferences dialog (see `main.rs`) is the sole editor: it mutates
//! the live `Prefs`, applies each change to its consumer immediately, and calls
//! [`Prefs::save`] — a tiny atomic write (temp file + rename) so a crash mid-save
//! can't leave a half-written file.

use std::path::PathBuf;
use std::time::Duration;

/// A toggleable limit that **remembers its numeric value across toggles**.
///
/// `infinite = true` means "no limit" ([`as_duration`](Self::as_duration)
/// returns `None`), but `secs` is stored independently, so a UI can grey out the
/// numeric box while infinite is on and restore its exact prior value when the
/// switch flips back off.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Limit {
    // Per-field serde defaults so a *partial* nested object — a natural hand
    // edit like `{"idle_pause":{"infinite":true}}` — fills the missing field
    // instead of failing the deserialize and throwing away the *whole* prefs
    // file (R32). `sanitize` then re-clamps `secs` to a sane value.
    #[serde(default)]
    pub infinite: bool,
    #[serde(default)]
    pub secs: f64,
}

impl Limit {
    pub const fn finite(secs: f64) -> Self {
        Limit { infinite: false, secs }
    }

    /// The limit as a `Duration`, or `None` when infinite (no limit).
    pub fn as_duration(&self) -> Option<Duration> {
        (!self.infinite).then(|| Duration::from_secs_f64(self.secs.max(0.0)))
    }
}

/// All persisted preferences. Every field carries a `#[serde(default)]` (backed
/// by the module-level `default_*` fns and [`Default`]) so partial/older JSON
/// loads cleanly.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Prefs {
    // ---- Roll & history ----
    /// Section-break threshold: idle time before the roll clock pauses and
    /// the next note starts a new instance (`infinite` = never auto-break).
    /// Gaps shorter than this show on the paper in full; once crossed, the
    /// blank tail is trimmed to `section_tail_s`.
    pub idle_pause: Limit,
    /// Blank paper (s) kept after a section's last note when a break fires.
    pub section_tail_s: f64,
    /// Blank paper (s) before the first note of the section that resumes
    /// after a break.
    pub section_lead_in_s: f64,
    /// Pixels of paper per second in the roll / falling panels (zoom).
    pub roll_px_per_s: f32,
    /// Seconds a scrolled-back roll view holds before easing home.
    pub scrollback_idle_s: f64,
    /// Show the sustain-pedal (CC64) lane at the history roll's left edge.
    /// Opt-in (the roll's default look is unchanged), and only meaningful on
    /// MIDI input — the mic path has no pedal signal, so the toggle is hidden
    /// and the lane not drawn on the mic fallback.
    pub pedal_lane_visible: bool,
    /// Minimum change in CC64 level (0..=127) required to register as a new
    /// pedal position. Filters analog-pedal jitter without losing half-pedal
    /// nuance. 0 = accept every distinct value (today's behavior).
    pub pedal_deadzone: u8,

    // ---- Appearance ----
    /// This player's note color (sRGB), broadcast to the peer.
    pub local_color: [u8; 3],
    /// This player's display name, shown in the status bar and broadcast to the
    /// peer (which renders it next to the peer color). Persists across sessions.
    pub local_name: String,
    /// User-dragged keyboard height, as a fraction of the central panel's
    /// height so it scales with the window. Set by dragging the keyboard's
    /// top/bottom edge; saved on drag release. `None` = the built-in default
    /// split (`KEYBOARD_FRACTION` in main.rs).
    pub keyboard_height_frac: Option<f32>,

    // ---- Audio / mic ----
    /// Mic detection threshold (model posterior probability, 0..1).
    pub threshold: f32,
    /// How long after the synth stops voicing a note the mic keeps ignoring
    /// that note (echo guard), in milliseconds.
    pub echo_holdoff_ms: u64,
    /// Default state of the "Mute mic" toggle. Defaults to muted so a fresh
    /// install doesn't start transcribing ambient audio before the user opts in.
    pub mic_muted: bool,

    // ---- Metronome ----
    /// Default metronome tempo (BPM).
    pub metro_bpm: u16,
    /// Number of beats per bar (size of `metro_beat_freqs`); also broadcast to
    /// a connected peer with each beat, so both sides use the same bar length.
    pub metro_beats_per_bar: u8,
    /// Pitch (Hz) of each beat's click, indexed by position in the bar — index
    /// 0 is the accent/downbeat. Broadcast to a connected peer (`Packet::
    /// MetroBeatTable`) whenever it changes, so both sides' clicks sound
    /// identical — unlike `metro_bpm`/`metro_beats_per_bar` there's no host
    /// authority; whichever side edits last wins on both ends.
    pub metro_beat_freqs: Vec<f32>,
    /// Per-beat click level (0..1), indexed the same way as `metro_beat_freqs`.
    /// Multiplies the accent/plain amplitude in `synth::Synth::tick`, so beats
    /// can be dialed back (or muted) individually — e.g. a downbeat much
    /// louder than the rest. Synced with the peer alongside `metro_beat_freqs`.
    pub metro_beat_volumes: Vec<f32>,

    // ---- Advanced (model / network) ----
    /// Below this raw RMS a mic window is treated as silence (inference skipped).
    pub silence_rms: f32,
    /// Hard cap on the mic normalization gain.
    pub norm_max_gain: f32,
    /// Frame-grid probability below which a sounding note is released.
    pub frame_off: f32,
    /// How often the input supervisor rescans MIDI ports, in milliseconds.
    pub midi_poll_ms: u64,

    // ---- Startup & window ----
    /// Restore compact/normal window mode from the last session at startup.
    pub remember_window_state: bool,
    /// Last active compact-mode state; only consulted at startup when
    /// `remember_window_state` is true.
    pub compact_mode: bool,
    /// Keep the compact window above other applications (always-on-top). Only
    /// applied while in compact mode; reconciled live so toggling it takes
    /// effect immediately without leaving compact mode.
    pub compact_always_on_top: bool,
    /// Last known normal-mode window size, snapshotted on every compact-mode
    /// entry. Seeds `PianoApp::normal_size` at startup so a session that
    /// launches directly into compact mode never falls back to a mismatched
    /// default height. Default `None`; only meaningful with
    /// `remember_window_state`.
    pub normal_window_size: Option<[f32; 2]>,
    /// Reload the most recently opened MIDI/JSONL file at startup.
    pub reopen_last_file: bool,
    /// The most recently opened score file; only consulted at startup when
    /// `reopen_last_file` is true. Cleared on explicit File ▸ Close.
    pub last_file_path: Option<PathBuf>,
}

// Defaults mirror the former compile-time constants in roll.rs / main.rs /
// inference.rs / input.rs. Kept as free fns so `#[serde(default)]` can name them
// per field and `Default` can reuse them.
fn default_idle_pause() -> Limit {
    Limit::finite(30.0)
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs {
            idle_pause: default_idle_pause(),
            section_tail_s: 2.0,
            section_lead_in_s: 2.0,
            roll_px_per_s: 40.0,
            scrollback_idle_s: 2.5,
            pedal_lane_visible: false,
            pedal_deadzone: 0,
            local_color: [220, 60, 60],
            local_name: "Player".to_string(),
            keyboard_height_frac: None,
            threshold: 0.30,
            echo_holdoff_ms: 2000,
            mic_muted: true,
            metro_bpm: 120,
            metro_beats_per_bar: 4,
            metro_beat_freqs: vec![1800.0, 1200.0, 1200.0, 1200.0],
            metro_beat_volumes: vec![1.0, 1.0, 1.0, 1.0],
            silence_rms: 0.002,
            norm_max_gain: 10.0,
            frame_off: 0.10,
            midi_poll_ms: 1000,
            remember_window_state: false,
            compact_mode: false,
            compact_always_on_top: true,
            normal_window_size: None,
            reopen_last_file: false,
            last_file_path: None,
        }
    }
}

impl Prefs {
    /// Load preferences, falling back to [`Default`] on any error (missing file,
    /// parse failure, unreadable dir). Never fails — a bad file just resets.
    /// Whatever parses is then [`sanitize`](Self::sanitize)d, so even a
    /// well-formed-but-hostile file can't panic startup or poison rendering.
    pub fn load() -> Self {
        let Some(path) = prefs_path() else {
            return Prefs::default();
        };
        let mut prefs = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                eprintln!("[prefs] {} is unreadable ({e}); using defaults", path.display());
                // Preserve the unparseable file as `.bak` before proceeding: the
                // first preference edit `save()`s over the original, so without
                // this a user who hand-edited one value loses every setting with
                // no way to recover the file (R31).
                let backup = path.with_extension("json.bak");
                if let Err(e) = std::fs::rename(&path, &backup) {
                    eprintln!("[prefs] could not back up corrupt prefs to {}: {e}", backup.display());
                }
                Prefs::default()
            }),
            // Missing file is the normal first-run case — silent.
            Err(_) => Prefs::default(),
        };
        prefs.sanitize();
        prefs
    }

    /// Clamp every numeric field to a finite, in-range value and reconcile the
    /// two per-beat metronome arrays to equal length. `#[serde(default)]` keeps
    /// a *missing* field safe; this keeps a *present but hostile* one safe —
    /// values like `1e30` (which would panic `Duration::from_secs_f64` on every
    /// launch), non-finite floats, `roll_px_per_s: 0` (a coordinate-mapping
    /// divisor), or a `metro_beat_volumes` shorter than `metro_beat_freqs`
    /// (an index-out-of-bounds panic when the Metronome tab opens).
    fn sanitize(&mut self) {
        fn f64_or(v: f64, lo: f64, hi: f64, default: f64) -> f64 {
            if v.is_finite() { v.clamp(lo, hi) } else { default }
        }
        fn f32_or(v: f32, lo: f32, hi: f32, default: f32) -> f32 {
            if v.is_finite() { v.clamp(lo, hi) } else { default }
        }

        // Section timing (seconds): finite and within a day, so `Duration`
        // conversions can't overflow/panic.
        self.idle_pause.secs = f64_or(self.idle_pause.secs, 0.0, 86_400.0, 30.0);
        self.section_tail_s = f64_or(self.section_tail_s, 0.0, 86_400.0, 2.0);
        self.section_lead_in_s = f64_or(self.section_lead_in_s, 0.0, 86_400.0, 2.0);
        self.scrollback_idle_s = f64_or(self.scrollback_idle_s, 0.0, 86_400.0, 2.5);
        // Zoom is a divisor in the roll coordinate map — must be strictly > 0.
        self.roll_px_per_s = f32_or(self.roll_px_per_s, 1.0, 5000.0, 40.0);

        // Audio / detector knobs.
        self.threshold = f32_or(self.threshold, 0.05, 0.95, 0.30);
        self.silence_rms = f32_or(self.silence_rms, 0.0, 1.0, 0.002);
        self.norm_max_gain = f32_or(self.norm_max_gain, 1.0, 100.0, 10.0);
        self.frame_off = f32_or(self.frame_off, 0.0, 1.0, 0.10);

        // Metronome: clamp tempo, and make the two per-beat arrays equal length
        // (the Preferences loop iterates `freqs` and indexes `volumes`).
        self.metro_bpm = self.metro_bpm.clamp(30, 240);
        self.metro_beats_per_bar = self.metro_beats_per_bar.clamp(1, 32);
        if self.metro_beat_freqs.is_empty() {
            self.metro_beat_freqs = vec![1200.0; self.metro_beats_per_bar as usize];
        }
        for f in &mut self.metro_beat_freqs {
            *f = f32_or(*f, 20.0, 8000.0, 1200.0);
        }
        self.metro_beat_volumes.resize(self.metro_beat_freqs.len(), 1.0);
        for v in &mut self.metro_beat_volumes {
            *v = f32_or(*v, 0.0, 1.0, 1.0);
        }

        // Window size fed straight into `ViewportCommand::InnerSize` when
        // leaving compact mode: a non-finite/degenerate value produces a
        // broken/vanished window (and is re-saved, so it persists). Drop the
        // hint entirely if either axis is unusable; otherwise clamp to a sane
        // pixel range (F27).
        self.normal_window_size = self.normal_window_size.and_then(|[w, h]| {
            if w.is_finite() && h.is_finite() {
                Some([w.clamp(200.0, 10_000.0), h.clamp(150.0, 10_000.0)])
            } else {
                None
            }
        });
        // Keyboard height fraction flows straight into every keyboard rect
        // (main.rs); `f32::clamp` returns NaN for NaN input, so a non-finite
        // value would propagate NaN into layout and be re-saved (R33). Drop a
        // non-finite hint; clamp a finite one to a usable range.
        self.keyboard_height_frac = self
            .keyboard_height_frac
            .filter(|f| f.is_finite())
            .map(|f| f.clamp(0.05, 0.85));

        // Echo holdoff: a huge value leaves the mic permanently deaf after any
        // synth note. Cap at 60 s (F27).
        self.echo_holdoff_ms = self.echo_holdoff_ms.min(60_000);
        // Pedal deadzone is a CC delta (0..=127); anything above makes the pedal
        // lane permanently dead (F27).
        self.pedal_deadzone = self.pedal_deadzone.min(127);
    }

    /// Save preferences atomically (temp file + rename). Errors are logged but
    /// not fatal — a failed save just means this change won't persist.
    pub fn save(&self) {
        let Some(path) = prefs_path() else { return };
        if let Some(dir) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                eprintln!("[prefs] could not create {}: {e}", dir.display());
                return;
            }
        }
        let json = match serde_json::to_string_pretty(self) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("[prefs] serialize failed: {e}");
                return;
            }
        };
        let tmp = path.with_extension(format!("json.tmp-{}", std::process::id()));
        // Write + fsync the temp file's *data* before the rename, so a power
        // loss can't commit the rename (metadata) while the file's data blocks
        // are still empty — which would truncate preferences.json to nothing and
        // silently reset every preference next launch (F19). Mirrors bundle.rs.
        match std::fs::File::create(&tmp) {
            Ok(mut f) => {
                use std::io::Write as _;
                if let Err(e) = f.write_all(json.as_bytes()).and_then(|()| f.sync_all()) {
                    eprintln!("[prefs] write/sync failed: {e}");
                    let _ = std::fs::remove_file(&tmp);
                    return;
                }
            }
            Err(e) => {
                eprintln!("[prefs] write failed: {e}");
                return;
            }
        }
        if let Err(e) = std::fs::rename(&tmp, &path) {
            eprintln!("[prefs] rename failed: {e}");
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

/// `%LOCALAPPDATA%\open-piano\preferences.json` (falling back to the system temp
/// dir if `LOCALAPPDATA` is unset), matching `bundle.rs`'s cache directory.
fn prefs_path() -> Option<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    Some(base.join("open-piano").join("preferences.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_remembers_secs_across_infinite_toggle() {
        let mut l = Limit::finite(20.0);
        assert_eq!(l.as_duration(), Some(Duration::from_secs(20)));
        l.infinite = true;
        assert_eq!(l.as_duration(), None);
        // The numeric value survives the toggle, so flipping back restores it.
        l.infinite = false;
        assert_eq!(l.as_duration(), Some(Duration::from_secs(20)));
    }

    #[test]
    fn partial_json_fills_missing_fields_from_defaults() {
        // Only one field present: everything else must fall back to Default.
        let parsed: Prefs = serde_json::from_str(r#"{"threshold": 0.5}"#).unwrap();
        assert_eq!(parsed.threshold, 0.5);
        assert_eq!(parsed.roll_px_per_s, 40.0);
        assert_eq!(parsed.metro_bpm, 120);
        assert_eq!(parsed.metro_beats_per_bar, 4);
        assert_eq!(parsed.metro_beat_freqs, vec![1800.0, 1200.0, 1200.0, 1200.0]);
        assert!(!parsed.idle_pause.infinite);
        assert_eq!(parsed.idle_pause.secs, 30.0);
        assert_eq!(parsed.section_tail_s, 2.0);
        assert_eq!(parsed.section_lead_in_s, 2.0);
        assert_eq!(parsed.pedal_deadzone, 0);
        assert_eq!(parsed.normal_window_size, None);
    }

    #[test]
    fn sanitize_reconciles_metro_array_lengths() {
        // 5 freqs but 4 volumes: the Preferences loop would index out of bounds.
        let mut p: Prefs = serde_json::from_str(
            r#"{"metro_beat_freqs":[1800,1200,1200,1200,1200],"metro_beat_volumes":[1,1,1,1]}"#,
        )
        .unwrap();
        p.sanitize();
        assert_eq!(p.metro_beat_volumes.len(), p.metro_beat_freqs.len());
    }

    #[test]
    fn sanitize_clamps_hostile_numerics() {
        let mut p: Prefs =
            serde_json::from_str(r#"{"section_tail_s":1e30,"roll_px_per_s":0.0}"#).unwrap();
        p.sanitize();
        assert!(p.section_tail_s.is_finite() && p.section_tail_s <= 86_400.0);
        assert!(p.roll_px_per_s >= 1.0);
        // The duration conversion that used to panic on 1e30 is now safe.
        let _ = Duration::from_secs_f64(p.section_tail_s);
        let _ = p.idle_pause.as_duration();
    }

    #[test]
    fn roundtrips_through_json() {
        let mut p = Prefs::default();
        p.idle_pause.infinite = true;
        p.local_color = [1, 2, 3];
        let json = serde_json::to_string(&p).unwrap();
        let back: Prefs = serde_json::from_str(&json).unwrap();
        assert!(back.idle_pause.infinite);
        assert_eq!(back.local_color, [1, 2, 3]);
    }
}
