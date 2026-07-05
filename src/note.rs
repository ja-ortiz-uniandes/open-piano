//! Shared note message type + MIDI helpers used across the audio, network and
//! UI layers.

/// The full 88-key piano spans MIDI note 21 (A0) .. 108 (C8) inclusive.
pub const MIDI_LOW: u8 = 21; // A0
pub const MIDI_HIGH: u8 = 108; // C8
pub const KEY_COUNT: usize = (MIDI_HIGH - MIDI_LOW + 1) as usize; // 88

/// A note-on / note-off transition. This is the unit of communication on every
/// channel in the app: audio-thread -> UI, net-thread -> UI, and the
/// 2-byte wire format sent to the peer (as an unreliable QUIC datagram).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteMsg {
    On(u8),
    Off(u8),
}

impl NoteMsg {
    pub fn midi(&self) -> u8 {
        match self {
            NoteMsg::On(n) | NoteMsg::Off(n) => *n,
        }
    }

    /// Encode to the 2-byte wire format: `[status, note]`, using MIDI-style
    /// status bytes (0x90 = note on, 0x80 = note off).
    pub fn encode(&self) -> [u8; 2] {
        match self {
            NoteMsg::On(n) => [0x90, *n],
            NoteMsg::Off(n) => [0x80, *n],
        }
    }

    /// Decode a received datagram. Returns `None` for malformed / unknown data.
    pub fn decode(buf: &[u8]) -> Option<NoteMsg> {
        if buf.len() < 2 {
            return None;
        }
        match buf[0] {
            0x90 => Some(NoteMsg::On(buf[1])),
            0x80 => Some(NoteMsg::Off(buf[1])),
            _ => None,
        }
    }
}

/// A packet on the P2P wire: either a note transition or a peer announcing the
/// color it wants its notes drawn in. Colors travel over the network so each
/// player picks *their own* color and the other end renders it — see `net.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Packet {
    Note(NoteMsg),
    /// The sender's chosen display color, as sRGB `[r, g, b]`.
    Color([u8; 3]),
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
}

impl Packet {
    /// Encode to the wire format. Notes are the existing 2-byte form (`0x90`/
    /// `0x80`); a color is `[0xC0, r, g, b]`; metronome markers/control use
    /// `0xB0`/`0xB1`. None of these status bytes collide, so `decode` is
    /// unambiguous. (`bpm` is big-endian u16.)
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Packet::Note(n) => n.encode().to_vec(),
            Packet::Color([r, g, b]) => vec![0xC0, *r, *g, *b],
            Packet::MetroBeat { bpm, beat_in_bar, beats_per_bar, on } => {
                let [hi, lo] = bpm.to_be_bytes();
                vec![0xB0, hi, lo, *beat_in_bar, *beats_per_bar, *on as u8]
            }
            Packet::MetroCtl { on, bpm } => {
                let [hi, lo] = bpm.to_be_bytes();
                vec![0xB1, *on as u8, hi, lo]
            }
        }
    }

    /// Decode a received datagram, or `None` for malformed / unknown data.
    pub fn decode(buf: &[u8]) -> Option<Packet> {
        match buf.first()? {
            0x90 | 0x80 => NoteMsg::decode(buf).map(Packet::Note),
            0xC0 if buf.len() >= 4 => Some(Packet::Color([buf[1], buf[2], buf[3]])),
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
            Packet::Note(NoteMsg::On(60)),
            Packet::Note(NoteMsg::Off(21)),
            Packet::Color([220, 60, 60]),
            Packet::MetroBeat { bpm: 120, beat_in_bar: 0, beats_per_bar: 4, on: true },
            Packet::MetroBeat { bpm: 240, beat_in_bar: 3, beats_per_bar: 4, on: false },
            Packet::MetroCtl { on: true, bpm: 90 },
            Packet::MetroCtl { on: false, bpm: 200 },
        ] {
            assert_eq!(Packet::decode(&p.encode()), Some(p), "roundtrip {p:?}");
        }
    }

    #[test]
    fn decode_rejects_truncated_and_unknown() {
        assert_eq!(Packet::decode(&[]), None);
        assert_eq!(Packet::decode(&[0xB0, 0, 120]), None); // too short for MetroBeat
        assert_eq!(Packet::decode(&[0xB1, 1]), None); // too short for MetroCtl
        assert_eq!(Packet::decode(&[0x7F, 0, 0]), None); // unknown status
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
