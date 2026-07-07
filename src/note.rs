//! Shared note message type + MIDI helpers used across the audio, network and
//! UI layers.

/// The full 88-key piano spans MIDI note 21 (A0) .. 108 (C8) inclusive.
pub const MIDI_LOW: u8 = 21; // A0
pub const MIDI_HIGH: u8 = 108; // C8
pub const KEY_COUNT: usize = (MIDI_HIGH - MIDI_LOW + 1) as usize; // 88

/// The velocity used for note-ons that have no real velocity behind them —
/// mouse clicks on the on-screen keyboard and the mic/ONNX path (a
/// posteriorgram has no force signal). Chosen mezzo-forte-ish so the flat
/// placeholder doesn't render as either a ghost note or a hammered one.
pub const DEFAULT_VELOCITY: u8 = 100;

/// A note-on / note-off transition. This is the unit of communication on every
/// channel in the app: audio-thread -> UI, net-thread -> UI, and the wire
/// format sent to the peer (as an unreliable QUIC datagram) — 3 bytes for an
/// On (`[0x90, note, velocity]`), 2 for an Off (`[0x80, note]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteMsg {
    /// Note on: `(midi, velocity 1..=127)`. Callers with no real velocity
    /// (mouse clicks, the mic path) use [`DEFAULT_VELOCITY`].
    On(u8, u8),
    Off(u8),
}

impl NoteMsg {
    pub fn midi(&self) -> u8 {
        match self {
            NoteMsg::On(n, _) | NoteMsg::Off(n) => *n,
        }
    }

    /// Encode to the wire format (see the type docs), using MIDI-style
    /// status bytes (0x90 = note on, 0x80 = note off).
    pub fn encode(&self) -> Vec<u8> {
        match self {
            NoteMsg::On(n, v) => vec![0x90, *n, *v],
            NoteMsg::Off(n) => vec![0x80, *n],
        }
    }

    /// Decode a received datagram. Returns `None` for malformed / unknown
    /// data. A velocity-less 2-byte On (an older peer) decodes with
    /// [`DEFAULT_VELOCITY`] rather than being dropped.
    pub fn decode(buf: &[u8]) -> Option<NoteMsg> {
        if buf.len() < 2 {
            return None;
        }
        match buf[0] {
            0x90 => Some(NoteMsg::On(buf[1], buf.get(2).copied().unwrap_or(DEFAULT_VELOCITY))),
            0x80 => Some(NoteMsg::Off(buf[1])),
            _ => None,
        }
    }
}

/// A packet on the P2P wire: either a note transition or a peer announcing the
/// color it wants its notes drawn in. Colors travel over the network so each
/// player picks *their own* color and the other end renders it — see `net.rs`.
#[derive(Debug, Clone, PartialEq)]
pub enum Packet {
    Note(NoteMsg),
    /// The sender's chosen display color, as sRGB `[r, g, b]`.
    Color([u8; 3]),
    /// The sender's chosen display name (UTF-8). Travels over the network the
    /// same way colors do — each player picks their own name and the other end
    /// renders it next to the peer color. Re-sent on the color heartbeat so a
    /// dropped datagram doesn't leave the peer showing a stale name.
    Name(String),
    /// Metronome beat marker, broadcast by the host on **each** beat. Carries
    /// enough to (re)derive the grid (`bpm`, `beats_per_bar`), the accent
    /// (`beat_in_bar == 0` = downbeat), and to toggle the follower on/off
    /// (`on`). The follower anchors its local click schedule to when this
    /// arrives (see `net.rs` / `main.rs`). Status byte `0xB0`.
    MetroBeat {
        bpm: u16,
        beat_in_bar: u8,
        beats_per_bar: u8,
        on: bool,
    },
    /// Metronome control request, follower → host: set the running state and
    /// tempo. The host adopts it as the new authoritative grid and echoes it
    /// back via `MetroBeat`s (last-writer-wins, single scheduler). Status `0xB1`.
    MetroCtl { on: bool, bpm: u16 },
    /// Per-beat click pitch (Hz) and level (0..1) tables, indexed the same way
    /// as `Metronome::beat_freqs`/`beat_volumes` in `main.rs`. Unlike
    /// `MetroBeat`/`MetroCtl` there's no host authority here — whichever side
    /// edits its Preferences last broadcasts its tables and the other side
    /// adopts them verbatim, so both players' clicks sound identical. Sent on
    /// every edit and re-sent on the color heartbeat so a dropped datagram
    /// doesn't leave the two sides mismatched. Status `0xB2`.
    MetroBeatTable { freqs: Vec<f32>, volumes: Vec<f32> },
    /// The sender's sustain-pedal (CC64) level, 0..=127. Deliberately its own
    /// `Packet` variant, never folded into [`NoteMsg`]: pedal events must be
    /// structurally unable to reach `Roll::note()` — and with it the roll's
    /// idle-timer reset (see `roll::Roll::pedal`). Sent on change only (no
    /// heartbeat: CC64 fires frequently while half-pedaling, so a dropped
    /// datagram self-heals on the next change). Status `0xB3`.
    Pedal { level: u8 },
}

impl Packet {
    /// Encode to the wire format. Notes are the 2/3-byte `NoteMsg` form
    /// (`0x90`/`0x80`); a color is `[0xC0, r, g, b]`; metronome markers/
    /// control use `0xB0`/`0xB1`; pedal is `[0xB3, level]`. None of these
    /// status bytes collide, so `decode` is unambiguous. (`bpm` is big-endian
    /// u16.)
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Packet::Note(n) => n.encode(),
            Packet::Color([r, g, b]) => vec![0xC0, *r, *g, *b],
            Packet::Name(name) => {
                // `[0xC1, len, ..utf8..]`. Length-prefixed with a single byte so
                // the frame is self-delimiting; names longer than 255 bytes are
                // truncated on a UTF-8 char boundary so the wire stays valid.
                let mut bytes = name.as_bytes();
                if bytes.len() > u8::MAX as usize {
                    let mut end = u8::MAX as usize;
                    while end > 0 && (bytes[end] & 0xC0) == 0x80 {
                        end -= 1; // back off to a char boundary
                    }
                    bytes = &bytes[..end];
                }
                let mut buf = Vec::with_capacity(2 + bytes.len());
                buf.push(0xC1);
                buf.push(bytes.len() as u8);
                buf.extend_from_slice(bytes);
                buf
            }
            Packet::MetroBeat { bpm, beat_in_bar, beats_per_bar, on } => {
                let [hi, lo] = bpm.to_be_bytes();
                vec![0xB0, hi, lo, *beat_in_bar, *beats_per_bar, *on as u8]
            }
            Packet::MetroCtl { on, bpm } => {
                let [hi, lo] = bpm.to_be_bytes();
                vec![0xB1, *on as u8, hi, lo]
            }
            Packet::MetroBeatTable { freqs, volumes } => {
                // Both tables always have equal length (main.rs keeps them in
                // lockstep); clamp defensively so a stray mismatch can't
                // corrupt the frame instead of just truncating it.
                let len = freqs.len().min(volumes.len()).min(u8::MAX as usize);
                let mut buf = Vec::with_capacity(2 + len * 8);
                buf.push(0xB2);
                buf.push(len as u8);
                for f in freqs.iter().take(len) {
                    buf.extend_from_slice(&f.to_be_bytes());
                }
                for v in volumes.iter().take(len) {
                    buf.extend_from_slice(&v.to_be_bytes());
                }
                buf
            }
            Packet::Pedal { level } => vec![0xB3, *level],
        }
    }

    /// Decode a received datagram, or `None` for malformed / unknown data.
    pub fn decode(buf: &[u8]) -> Option<Packet> {
        match buf.first()? {
            0x90 | 0x80 => NoteMsg::decode(buf).map(Packet::Note),
            0xC0 if buf.len() >= 4 => Some(Packet::Color([buf[1], buf[2], buf[3]])),
            0xC1 if buf.len() >= 2 => {
                let len = buf[1] as usize;
                if buf.len() < 2 + len {
                    return None;
                }
                // Lossy so a corrupt byte can't drop the whole (valid-length) frame.
                Some(Packet::Name(String::from_utf8_lossy(&buf[2..2 + len]).into_owned()))
            }
            0xB0 if buf.len() >= 6 => Some(Packet::MetroBeat {
                bpm: u16::from_be_bytes([buf[1], buf[2]]),
                beat_in_bar: buf[3],
                beats_per_bar: buf[4],
                on: buf[5] != 0,
            }),
            0xB1 if buf.len() >= 4 => Some(Packet::MetroCtl {
                on: buf[1] != 0,
                bpm: u16::from_be_bytes([buf[2], buf[3]]),
            }),
            0xB2 if buf.len() >= 2 => {
                let len = buf[1] as usize;
                if buf.len() < 2 + len * 8 {
                    return None;
                }
                let read_f32 = |i: usize| {
                    let off = 2 + i * 4;
                    f32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
                };
                let freqs = (0..len).map(read_f32).collect();
                let volumes = (0..len).map(|i| read_f32(len + i)).collect();
                Some(Packet::MetroBeatTable { freqs, volumes })
            }
            0xB3 if buf.len() >= 2 => Some(Packet::Pedal { level: buf[1] }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_encode_decode_roundtrips() {
        for p in [
            Packet::Note(NoteMsg::On(60, 100)),
            Packet::Note(NoteMsg::On(108, 1)),
            Packet::Note(NoteMsg::Off(21)),
            Packet::Pedal { level: 0 },
            Packet::Pedal { level: 127 },
            Packet::Color([220, 60, 60]),
            Packet::Name(String::new()),
            Packet::Name("Ada".to_string()),
            Packet::Name("Grace — 音楽".to_string()),
            Packet::MetroBeat { bpm: 120, beat_in_bar: 0, beats_per_bar: 4, on: true },
            Packet::MetroBeat { bpm: 240, beat_in_bar: 3, beats_per_bar: 4, on: false },
            Packet::MetroCtl { on: true, bpm: 90 },
            Packet::MetroCtl { on: false, bpm: 200 },
            Packet::MetroBeatTable { freqs: vec![], volumes: vec![] },
            Packet::MetroBeatTable {
                freqs: vec![1800.0, 1200.0, 1200.0, 1200.0],
                volumes: vec![1.0, 0.5, 0.75, 0.25],
            },
        ] {
            assert_eq!(Packet::decode(&p.encode()), Some(p.clone()), "roundtrip {p:?}");
        }
    }

    #[test]
    fn decode_rejects_truncated_and_unknown() {
        assert_eq!(Packet::decode(&[]), None);
        assert_eq!(Packet::decode(&[0xB0, 0, 120]), None); // too short for MetroBeat
        assert_eq!(Packet::decode(&[0xB1, 1]), None); // too short for MetroCtl
        assert_eq!(Packet::decode(&[0xB2, 2, 0, 0, 0, 0]), None); // too short for MetroBeatTable
        assert_eq!(Packet::decode(&[0xC1, 5, b'h', b'i']), None); // name len exceeds payload
        assert_eq!(Packet::decode(&[0xB3]), None); // too short for Pedal
        assert_eq!(Packet::decode(&[0x7F, 0, 0]), None); // unknown status
    }

    #[test]
    fn velocity_less_note_on_decodes_with_the_default() {
        // An older peer's 2-byte On: degrade to the flat placeholder velocity
        // instead of dropping the note.
        assert_eq!(
            Packet::decode(&[0x90, 60]),
            Some(Packet::Note(NoteMsg::On(60, DEFAULT_VELOCITY)))
        );
    }

    #[test]
    fn name_truncates_on_char_boundary() {
        // A multibyte name longer than 255 bytes must truncate to valid UTF-8.
        let long = "é".repeat(200); // 400 bytes
        let encoded = Packet::Name(long).encode();
        assert!(encoded.len() <= 2 + u8::MAX as usize);
        // The truncated payload still decodes to a valid string (no split char).
        match Packet::decode(&encoded) {
            Some(Packet::Name(s)) => assert!(s.chars().all(|c| c == 'é')),
            other => panic!("expected a Name, got {other:?}"),
        }
    }
}

/// Convert a MIDI note into a keyboard index in `0..KEY_COUNT`, if in range.
pub fn midi_to_key_index(midi: u8) -> Option<usize> {
    if midi < MIDI_LOW || midi > MIDI_HIGH {
        return None;
    }
    Some((midi - MIDI_LOW) as usize)
}

/// True if the given MIDI note is a black (sharp/flat) key.
pub fn is_black_key(midi: u8) -> bool {
    matches!(midi % 12, 1 | 3 | 6 | 8 | 10)
}

/// Fixed-do solfège note names, the app's naming convention for keys
/// (Do = C). Sharps use the ASCII '#' so names are typeable in text fields.
const SOLFEGE: [&str; 12] = [
    "Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si",
];

/// Solfège name + scientific-pitch octave for a MIDI note: 60 → "Do4"
/// (middle C), 21 → "La0" (the piano's lowest key).
pub fn solfege_name(midi: u8) -> String {
    let octave = (midi as i32 / 12) - 1;
    format!("{}{}", SOLFEGE[(midi % 12) as usize], octave)
}

/// Inverse of `solfege_name`: case-insensitive "NameOctave" (e.g. "sol3",
/// "do#5"; '♯' is accepted as an alias for '#'). `None` if it doesn't parse
/// or falls outside MIDI 0..=127.
pub fn solfege_to_midi(s: &str) -> Option<u8> {
    let s = s.trim().replace('♯', "#").to_lowercase();
    // Longest match first so "do#" isn't read as "do" + garbage octave.
    let (pc, rest) = SOLFEGE
        .iter()
        .enumerate()
        .filter(|(_, name)| s.starts_with(&name.to_lowercase()))
        .max_by_key(|(_, name)| name.len())
        .map(|(i, name)| (i as i32, &s[name.len()..]))?;
    let octave: i32 = rest.parse().ok()?;
    let midi = (octave + 1) * 12 + pc;
    u8::try_from(midi).ok().filter(|&m| m <= 127)
}
