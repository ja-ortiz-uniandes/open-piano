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
}

impl Packet {
    /// Encode to the wire format. Notes are the existing 2-byte form (`0x90`/
    /// `0x80`); a color is `[0xC0, r, g, b]`. `0xC0` never collides with the note
    /// status bytes, so `decode` is unambiguous.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Packet::Note(n) => n.encode().to_vec(),
            Packet::Color([r, g, b]) => vec![0xC0, *r, *g, *b],
        }
    }

    /// Decode a received datagram, or `None` for malformed / unknown data.
    pub fn decode(buf: &[u8]) -> Option<Packet> {
        match buf.first()? {
            0x90 | 0x80 => NoteMsg::decode(buf).map(Packet::Note),
            0xC0 if buf.len() >= 4 => Some(Packet::Color([buf[1], buf[2], buf[3]])),
            _ => None,
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
