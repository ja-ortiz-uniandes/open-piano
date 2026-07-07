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
    pub infinite: bool,
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
    /// Cap on trailing blank paper past the last note (`infinite` = never clamp).
    pub trailing_blank: Limit,
    /// Idle time before the clock pauses / a new instance is separated
    /// (`infinite` = never auto-pause). A truly unbounded gap needs BOTH this
    /// and `trailing_blank` infinite — a finite idle pause freezes the clock
    /// when it fires, which itself caps blank paper at the idle length.
    pub idle_pause: Limit,
    /// Pixels of paper per second in the roll / falling panels (zoom).
    pub roll_px_per_s: f32,
    /// Seconds a scrolled-back roll view holds before easing home.
    pub scrollback_idle_s: f64,
    /// Show the sustain-pedal (CC64) lane at the history roll's left edge.
    /// Opt-in (the roll's default look is unchanged), and only meaningful on
    /// MIDI input — the mic path has no pedal signal, so the toggle is hidden
    /// and the lane not drawn on the mic fallback.
    pub pedal_lane_visible: bool,

    // ---- Appearance ----
    /// This player's note color (sRGB), broadcast to the peer.
    pub local_color: [u8; 3],
    /// This player's display name, shown in the status bar and broadcast to the
    /// peer (which renders it next to the peer color). Persists across sessions.
    pub local_name: String,

    // ---- Audio / mic ----
    /// Mic detection threshold (model posterior probability, 0..1).
    pub threshold: f32,
    /// How long after the synth stops voicing a note the mic keeps ignoring
    /// that note (echo guard), in milliseconds.
    pub echo_holdoff_ms: u64,
    /// Default state of the "Mute mic" toggle.
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
    /// Reload the most recently opened MIDI/JSONL file at startup.
    pub reopen_last_file: bool,
    /// The most recently opened score file; only consulted at startup when
    /// `reopen_last_file` is true. Cleared on explicit File ▸ Close.
    pub last_file_path: Option<PathBuf>,
}

// Defaults mirror the former compile-time constants in roll.rs / main.rs /
// inference.rs / input.rs. Kept as free fns so `#[serde(default)]` can name them
// per field and `Default` can reuse them.
fn default_trailing_blank() -> Limit {
    Limit::finite(20.0)
}
fn default_idle_pause() -> Limit {
    Limit::finite(30.0)
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs {
            trailing_blank: default_trailing_blank(),
            idle_pause: default_idle_pause(),
            roll_px_per_s: 40.0,
            scrollback_idle_s: 2.5,
            pedal_lane_visible: false,
            local_color: [220, 60, 60],
            local_name: "Player".to_string(),
            threshold: 0.30,
            echo_holdoff_ms: 2000,
            mic_muted: false,
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
            reopen_last_file: false,
            last_file_path: None,
        }
    }
}

impl Prefs {
    /// Load preferences, falling back to [`Default`] on any error (missing file,
    /// parse failure, unreadable dir). Never fails — a bad file just resets.
    pub fn load() -> Self {
        let Some(path) = prefs_path() else {
            return Prefs::default();
        };
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                eprintln!("[prefs] {} is unreadable ({e}); using defaults", path.display());
                Prefs::default()
            }),
            // Missing file is the normal first-run case — silent.
            Err(_) => Prefs::default(),
        }
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
        if let Err(e) = std::fs::write(&tmp, json) {
            eprintln!("[prefs] write failed: {e}");
            return;
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
        assert!(!parsed.trailing_blank.infinite);
        assert_eq!(parsed.trailing_blank.secs, 20.0);
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
