//! Shared note message type + MIDI helpers used across the audio, network and
//! UI layers.

/// The full 88-key piano spans MIDI note 21 (A0) .. 108 (C8) inclusive.
pub const MIDI_LOW: u8 = 21; // A0
pub const MIDI_HIGH: u8 = 108; // C8
pub const KEY_COUNT: usize = (MIDI_HIGH - MIDI_LOW + 1) as usize; // 88

/// Bytes needed to pack one bool per key into a bitmask (88 keys → 11 bytes).
/// Used by [`Packet::Held`] to snapshot the whole pinned-key set in one frame.
pub const HELD_MASK_BYTES: usize = KEY_COUNT.div_ceil(8); // 11

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
    /// [`DEFAULT_VELOCITY`] rather than being dropped. A note-on carrying an
    /// explicit velocity of 0 is, by the MIDI convention, a note-off — so a
    /// hostile/buggy peer can't strand a key "on" with an un-releasable
    /// zero-velocity voice (see `synth.rs` / the `on`/`remote` arrays).
    pub fn decode(buf: &[u8]) -> Option<NoteMsg> {
        if buf.len() < 2 {
            return None;
        }
        match buf[0] {
            0x90 => match buf.get(2) {
                Some(0) => Some(NoteMsg::Off(buf[1])),
                Some(&v) => Some(NoteMsg::On(buf[1], v)),
                None => Some(NoteMsg::On(buf[1], DEFAULT_VELOCITY)),
            },
            0x80 => Some(NoteMsg::Off(buf[1])),
            _ => None,
        }
    }
}

/// A packet on the P2P wire: a note transition, or one of several small
/// shared-surface announcements (color, name, metronome, pedal, pins, …).
#[derive(Debug, Clone, PartialEq)]
pub enum Packet {
    /// A note transition plus a per-sender monotonic sequence number. The seq
    /// totally orders the sender's note-related traffic (notes *and* the
    /// [`Packet::Live`] snapshots), so a whole-state snapshot delivered out of
    /// order — reordered datagrams, notably during iroh's relay→direct path
    /// migration — can't resurrect a note a later `Note::Off` already cleared
    /// (or extinguish a fresh press). Note events always apply and advance the
    /// receiver's high-water mark; snapshots apply only if not stale (F6).
    /// Older peers omit the seq on the wire → decodes as 0, which still applies
    /// (notes are unconditional) but doesn't protect their snapshots.
    Note(NoteMsg, u32),
    /// The sender's chosen display color, as sRGB `[r, g, b]`. NOTE: the
    /// *receiver currently ignores the payload* — each player renders the peer
    /// in a fixed local remote color (see `main.rs`'s `Packet::Color` handler),
    /// because fresh installs share one default color. The variant is kept on
    /// the wire (and still sent) so the protocol stays stable if per-peer colors
    /// are ever honored.
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
    /// as `Metronome::beat_freqs`/`beat_volumes` in `main.rs`. The **host is
    /// authoritative** (like `MetroBeat`/`MetroCtl`): only the host broadcasts
    /// the table — on connect and on the color heartbeat — so the two sides
    /// can't *swap* tables (each sending before it processes the other's) or
    /// oscillate. A follower adopts what the host sends and pushes its own edits
    /// to the host, which adopts them and re-broadcasts on the next heartbeat,
    /// so a dropped edit heals within a second. Status `0xB2`.
    MetroBeatTable { freqs: Vec<f32>, volumes: Vec<f32> },
    /// The sender's sustain-pedal (CC64) level, 0..=127. Deliberately its own
    /// `Packet` variant, never folded into [`NoteMsg`]: pedal events must be
    /// structurally unable to reach `Roll::note()` — and with it the roll's
    /// idle-timer reset (see `roll::Roll::pedal`). Sent on change only (no
    /// heartbeat: CC64 fires frequently while half-pedaling, so a dropped
    /// datagram self-heals on the next change). Status `0xB3`.
    Pedal { level: u8 },
    /// The sender's full set of Ctrl+click-**pinned** keys, packed as an 88-bit
    /// mask (`HELD_MASK_BYTES` bytes, key `i` = bit `i&7` of byte `i>>3`). A
    /// display-only overlay — the receiver lights these keys in the sender's
    /// color exactly as it does live presses, so the pinned "point at this
    /// chord" gesture is a single shared thing both peers see (see `main.rs`
    /// `held`/`remote_held`). Sent as a whole-state **snapshot** on every
    /// change (pin/unpin, and the all-zero clear when Ctrl is released) and
    /// re-sent on the color heartbeat while any key is pinned: idempotent, so a
    /// dropped datagram can't leave the two views mismatched. Status `0xB4`.
    Held { seq: u32, mask: [u8; HELD_MASK_BYTES] },
    /// The sender's full set of **currently-sounding live notes**, packed as the
    /// same 88-bit mask as [`Packet::Held`]. An idempotent whole-state snapshot
    /// riding the heartbeat: the per-event [`NoteMsg`] datagrams are
    /// fire-and-forget, so a dropped note-**off** would otherwise leave a remote
    /// key (and its synth voice) stuck forever, and a chord held across a
    /// reconnect would never re-light. The receiver reconciles its `remote`
    /// array against this (see `main.rs` `reconcile_remote_live`), so any lost
    /// transition self-heals within a heartbeat. Status `0xB5`.
    Live { seq: u32, mask: [u8; HELD_MASK_BYTES] },
    /// The sender's manually-inserted segment-break times (roll-clock seconds),
    /// so a Ctrl+click break is one shared thing both players see rather than a
    /// line on one screen only. A whole-list snapshot re-sent on the heartbeat;
    /// the receiver folds each time into its own roll (deduped), so a dropped
    /// datagram converges. Status `0xB6`, `u8` count then that many big-endian
    /// `f64`s (count capped at 255 — far more manual breaks than a session has).
    Separators(Vec<f64>),
}

/// Pack the per-key pinned flags into the wire bitmask (see [`Packet::Held`]).
pub fn pack_held(keys: &[bool; KEY_COUNT]) -> [u8; HELD_MASK_BYTES] {
    let mut mask = [0u8; HELD_MASK_BYTES];
    for (i, &on) in keys.iter().enumerate() {
        if on {
            mask[i >> 3] |= 1 << (i & 7);
        }
    }
    mask
}

/// Unpack the wire bitmask back into per-key pinned flags (see [`Packet::Held`]).
pub fn unpack_held(mask: &[u8; HELD_MASK_BYTES]) -> [bool; KEY_COUNT] {
    let mut keys = [false; KEY_COUNT];
    for (i, key) in keys.iter_mut().enumerate() {
        *key = mask[i >> 3] & (1 << (i & 7)) != 0;
    }
    keys
}

/// Clamp a wire-received click frequency (Hz) to the audible range the UI
/// itself enforces, mapping non-finite values to a safe default. Keeps a
/// hostile/garbled [`Packet::MetroBeatTable`] from ever reaching the synth or
/// the persisted prefs with a NaN/∞ pitch.
pub(crate) fn sanitize_freq(f: f32) -> f32 {
    if f.is_finite() {
        f.clamp(20.0, 8000.0)
    } else {
        1200.0
    }
}

/// Clamp a wire-received per-beat click level to `0..=1`, mapping non-finite
/// values to full level. See [`sanitize_freq`].
pub(crate) fn sanitize_volume(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

impl Packet {
    /// Encode to the wire format. Notes are the 2/3-byte `NoteMsg` form
    /// (`0x90`/`0x80`); a color is `[0xC0, r, g, b]`; metronome markers/
    /// control use `0xB0`/`0xB1`; pedal is `[0xB3, level]`. None of these
    /// status bytes collide, so `decode` is unambiguous. (`bpm` is big-endian
    /// u16.)
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Packet::Note(n, seq) => {
                // The 2/3-byte note form with the seq appended big-endian. An
                // older peer's `NoteMsg::decode` reads only the leading bytes and
                // ignores the trailing seq, so this stays wire-compatible.
                let mut buf = n.encode();
                buf.extend_from_slice(&seq.to_be_bytes());
                buf
            }
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
            Packet::Held { seq, mask } => {
                let mut buf = Vec::with_capacity(5 + HELD_MASK_BYTES);
                buf.push(0xB4);
                buf.extend_from_slice(&seq.to_be_bytes());
                buf.extend_from_slice(mask);
                buf
            }
            Packet::Live { seq, mask } => {
                let mut buf = Vec::with_capacity(5 + HELD_MASK_BYTES);
                buf.push(0xB5);
                buf.extend_from_slice(&seq.to_be_bytes());
                buf.extend_from_slice(mask);
                buf
            }
            Packet::Separators(times) => {
                let len = times.len().min(u8::MAX as usize);
                let mut buf = Vec::with_capacity(2 + len * 8);
                buf.push(0xB6);
                buf.push(len as u8);
                for t in times.iter().take(len) {
                    buf.extend_from_slice(&t.to_be_bytes());
                }
                buf
            }
        }
    }

    /// Decode a received datagram, or `None` for malformed / unknown data.
    pub fn decode(buf: &[u8]) -> Option<Packet> {
        // Read a big-endian u32 at `off`, or 0 if the buffer is too short (an
        // older peer that didn't stamp a seq).
        let read_u32 = |off: usize| -> u32 {
            if buf.len() >= off + 4 {
                u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
            } else {
                0
            }
        };
        match buf.first()? {
            0x90 | 0x80 => {
                let msg = NoteMsg::decode(buf)?;
                // The seq follows the note bytes: 3 for a 0x90 On, 2 for a 0x80
                // Off. (A velocity-less 0x90 has no seq → 0.)
                let seq = read_u32(if buf[0] == 0x90 { 3 } else { 2 });
                Some(Packet::Note(msg, seq))
            }
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
                // Sanitize at the decode boundary so consumers (and the local
                // prefs the adopt path persists) can never see a NaN/∞ click
                // frequency or level: a non-finite freq would emit NaN samples
                // straight to the audio device, and serde would later write it
                // as JSON `null`, wiping every local preference on next launch.
                let freqs = (0..len).map(|i| sanitize_freq(read_f32(i))).collect();
                let volumes = (0..len).map(|i| sanitize_volume(read_f32(len + i))).collect();
                Some(Packet::MetroBeatTable { freqs, volumes })
            }
            // Clamp to a valid CC value at the decode boundary (like the other
            // sanitized packets): an out-of-range 128–255 otherwise flows into
            // the persistent session record — the `.mid` export masks it to a
            // wrong depth and the `.jsonl` stores the raw invalid value (R24).
            0xB3 if buf.len() >= 2 => Some(Packet::Pedal { level: buf[1].min(127) }),
            0xB4 if buf.len() >= 5 + HELD_MASK_BYTES => {
                let mut mask = [0u8; HELD_MASK_BYTES];
                mask.copy_from_slice(&buf[5..5 + HELD_MASK_BYTES]);
                Some(Packet::Held { seq: read_u32(1), mask })
            }
            0xB5 if buf.len() >= 5 + HELD_MASK_BYTES => {
                let mut mask = [0u8; HELD_MASK_BYTES];
                mask.copy_from_slice(&buf[5..5 + HELD_MASK_BYTES]);
                Some(Packet::Live { seq: read_u32(1), mask })
            }
            0xB6 if buf.len() >= 2 => {
                let len = buf[1] as usize;
                if buf.len() < 2 + len * 8 {
                    return None;
                }
                let times = (0..len)
                    .map(|i| {
                        let off = 2 + i * 8;
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&buf[off..off + 8]);
                        f64::from_be_bytes(b)
                    })
                    // Drop non-finite times so a garbled frame can't poison the
                    // roll's separator math.
                    .filter(|t| t.is_finite())
                    .collect();
                Some(Packet::Separators(times))
            }
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
            Packet::Note(NoteMsg::On(60, 100), 0),
            Packet::Note(NoteMsg::On(108, 1), 42),
            Packet::Note(NoteMsg::Off(21), 4_000_000_000),
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
            Packet::Held { seq: 0, mask: [0; HELD_MASK_BYTES] },
            Packet::Held { seq: 7, mask: [0xFF; HELD_MASK_BYTES] },
            Packet::Live { seq: 0, mask: [0; HELD_MASK_BYTES] },
            Packet::Live { seq: 65_536, mask: [0b1010_1010; HELD_MASK_BYTES] },
            Packet::Separators(vec![]),
            Packet::Separators(vec![3.0, 5.5, 128.25]),
            Packet::Held {
                seq: 1,
                mask: pack_held(&{
                    let mut k = [false; KEY_COUNT];
                    k[0] = true; // A0 (lowest)
                    k[39] = true; // middle C
                    k[KEY_COUNT - 1] = true; // C8 (highest)
                    k
                }),
            },
        ] {
            assert_eq!(Packet::decode(&p.encode()), Some(p.clone()), "roundtrip {p:?}");
        }
    }

    #[test]
    fn held_mask_roundtrips_through_pack_unpack() {
        let mut keys = [false; KEY_COUNT];
        keys[0] = true;
        keys[39] = true;
        keys[KEY_COUNT - 1] = true;
        assert_eq!(unpack_held(&pack_held(&keys)), keys);
        // No bits beyond the 88 keys are ever set: the top mask byte only holds
        // the last KEY_COUNT % 8 keys.
        let full = pack_held(&[true; KEY_COUNT]);
        assert_eq!(unpack_held(&full), [true; KEY_COUNT]);
    }

    #[test]
    fn decode_rejects_truncated_and_unknown() {
        assert_eq!(Packet::decode(&[]), None);
        assert_eq!(Packet::decode(&[0xB0, 0, 120]), None); // too short for MetroBeat
        assert_eq!(Packet::decode(&[0xB1, 1]), None); // too short for MetroCtl
        assert_eq!(Packet::decode(&[0xB2, 2, 0, 0, 0, 0]), None); // too short for MetroBeatTable
        assert_eq!(Packet::decode(&[0xC1, 5, b'h', b'i']), None); // name len exceeds payload
        assert_eq!(Packet::decode(&[0xB3]), None); // too short for Pedal
        assert_eq!(Packet::decode(&[0xB4, 0, 0]), None); // too short for Held (needs 4 seq + 11 mask)
        assert_eq!(Packet::decode(&[0x7F, 0, 0]), None); // unknown status
    }

    #[test]
    fn pedal_level_is_clamped_at_decode() {
        // A hostile 128–255 level must be clamped to a valid CC value at the
        // wire boundary, not flow into the saved session record (R24).
        assert_eq!(Packet::decode(&[0xB3, 200]), Some(Packet::Pedal { level: 127 }));
        assert_eq!(Packet::decode(&[0xB3, 255]), Some(Packet::Pedal { level: 127 }));
        assert_eq!(Packet::decode(&[0xB3, 64]), Some(Packet::Pedal { level: 64 }));
    }

    #[test]
    fn solfege_to_midi_rejects_out_of_range_octaves_without_overflow() {
        // Valid notes still parse.
        assert_eq!(solfege_to_midi("Do4"), Some(60));
        assert_eq!(solfege_to_midi("La0"), Some(21));
        // A huge octave must return None, not overflow `(octave + 1) * 12` (a
        // debug-build panic) or wrap into a bogus in-range note in release (R25).
        assert_eq!(solfege_to_midi("do357913946"), None);
        assert_eq!(solfege_to_midi("do178956970"), None);
        assert_eq!(solfege_to_midi("Do-99999"), None);
    }

    #[test]
    fn velocity_less_note_on_decodes_with_the_default() {
        // An older peer's 2-byte On: degrade to the flat placeholder velocity
        // instead of dropping the note.
        assert_eq!(
            Packet::decode(&[0x90, 60]),
            Some(Packet::Note(NoteMsg::On(60, DEFAULT_VELOCITY), 0))
        );
    }

    #[test]
    fn note_on_velocity_zero_decodes_as_off() {
        // A note-on carrying an explicit velocity of 0 is a note-off by MIDI
        // convention — never a stuck On(n, 0) a peer could use to strand a key.
        assert_eq!(Packet::decode(&[0x90, 60, 0]), Some(Packet::Note(NoteMsg::Off(60), 0)));
        // A non-zero explicit velocity still decodes as a real On.
        assert_eq!(Packet::decode(&[0x90, 60, 77]), Some(Packet::Note(NoteMsg::On(60, 77), 0)));
    }

    #[test]
    fn metro_table_nan_is_sanitized_on_decode() {
        // A crafted table with a NaN freq and ∞ volume must decode to finite,
        // in-range values — never reaching the synth or the persisted prefs.
        let mut buf = vec![0xB2, 1];
        buf.extend_from_slice(&f32::NAN.to_be_bytes());
        buf.extend_from_slice(&f32::INFINITY.to_be_bytes());
        match Packet::decode(&buf) {
            Some(Packet::MetroBeatTable { freqs, volumes }) => {
                assert!(freqs[0].is_finite() && (20.0..=8000.0).contains(&freqs[0]));
                assert!(volumes[0].is_finite() && (0.0..=1.0).contains(&volumes[0]));
            }
            other => panic!("expected a MetroBeatTable, got {other:?}"),
        }
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
    // Reject absurd octaves before the arithmetic: a huge parsed value overflows
    // `(octave + 1) * 12` — a debug-build panic from a text field, and a value
    // that wraps into 0..=127 and is wrongly accepted in release (R25). Any octave
    // outside the MIDI range can't yield a valid note anyway.
    if !(-2..=9).contains(&octave) {
        return None;
    }
    let midi = (octave + 1) * 12 + pc;
    u8::try_from(midi).ok().filter(|&m| m <= 127)
}
