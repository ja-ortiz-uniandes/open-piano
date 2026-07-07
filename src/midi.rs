//! USB / class-compliant MIDI input (midir, WinMM on Windows).
//!
//! This is the *preferred* input source: a connected digital piano emits exact
//! Note On / Note Off events, so there is no transcription, no model, and no
//! microphone bleed. The [`crate::input`] supervisor polls [`port_names`] to
//! discover devices (including hot-plugged ones) and calls [`connect_to`] to
//! attach to one, translating raw MIDI into the same [`NoteMsg`] values the
//! rest of the app speaks — routed into the *same* mpsc channel the audio
//! fallback uses, so the UI never knows the difference.

use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use midir::{Ignore, MidiInput, MidiInputConnection};

use crate::audio::EngineStatus;
use crate::note::NoteMsg;
use crate::record::Recorder;

/// List the names of all currently-available MIDI input ports. Returns an empty
/// vec if the MIDI subsystem can't be initialised or no devices are present.
/// Cheap enough to call on a poll loop (it enumerates, it doesn't open a port).
pub fn port_names() -> Vec<String> {
    let midi_in = match MidiInput::new("open-piano-scan") {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };
    midi_in
        .ports()
        .iter()
        .filter_map(|p| midi_in.port_name(p).ok())
        .collect()
}

/// Connect to the MIDI input port whose name is `want_name` and start
/// forwarding note events into `note_tx` and sustain-pedal (CC64) levels into
/// `pedal_tx`. Only this backend ever gets a `pedal_tx` — the mic path has no
/// pedal signal, which is what makes the pedal feature structurally MIDI-only
/// (see `input::start`).
///
/// On success returns the live connection — the caller **must keep it alive**
/// (dropping it closes the port, exactly like dropping a cpal stream). Fails if
/// the named port has gone away in the race between scanning and connecting, or
/// the MIDI stack errors.
pub fn connect_to(
    note_tx: Sender<NoteMsg>,
    status: &Arc<Mutex<EngineStatus>>,
    want_name: &str,
    recorder: Recorder,
    pedal_tx: Sender<u8>,
) -> Result<MidiInputConnection<()>, Box<dyn std::error::Error>> {
    let mut midi_in = MidiInput::new("open-piano")?;
    // Ignore sysex/timing/active-sensing chatter. Note on/off *and* control
    // changes (incl. CC64 sustain pedal) still come through — `Ignore` never
    // filters those — which the recorder needs for full ground-truth labels.
    midi_in.ignore(Ignore::All);

    // Find the port matching the requested name (the device may have been
    // unplugged between the scan and now).
    let port = midi_in
        .ports()
        .into_iter()
        .find(|p| midi_in.port_name(p).ok().as_deref() == Some(want_name))
        .ok_or("MIDI port disappeared before connect")?;

    let conn = midi_in
        .connect(
            &port,
            "open-piano-in",
            move |_timestamp, message, _| {
                // Tee every message to the capture harness first (a no-op unless
                // a recording session is active). It logs velocity + CC64 pedal
                // straight from the raw bytes, independent of the UI path below.
                recorder.push_midi(message);
                if let Some(msg) = parse_midi(message) {
                    // UI gone -> ignore; nothing else to do from the MIDI callback.
                    let _ = note_tx.send(msg);
                }
                if let Some(level) = parse_pedal(message) {
                    let _ = pedal_tx.send(level);
                }
            },
            (),
        )
        // ConnectError isn't guaranteed to be a boxable std::error::Error
        // (it carries the MidiInput back), so render it via Display.
        .map_err(|e| format!("MIDI connect failed: {e}"))?;

    if let Ok(mut s) = status.lock() {
        s.device = format!("MIDI: {want_name}");
        s.model = "Model: not used (direct MIDI input)".to_string();
    }

    Ok(conn)
}

/// Translate a raw MIDI channel-voice message into a [`NoteMsg`].
///
/// Handles the standard convention that a Note On with velocity 0 is really a
/// Note Off. Returns `None` for anything that isn't a note on/off (control
/// changes, pitch bend, running status fragments, etc.).
fn parse_midi(message: &[u8]) -> Option<NoteMsg> {
    if message.len() < 3 {
        return None;
    }
    let status = message[0] & 0xF0; // strip the channel nibble
    let note = message[1];
    let velocity = message[2];
    match status {
        0x90 if velocity > 0 => Some(NoteMsg::On(note, velocity)),
        0x90 => Some(NoteMsg::Off(note)), // note-on, velocity 0 == note-off
        0x80 => Some(NoteMsg::Off(note)),
        _ => None,
    }
}

/// The sustain-pedal (CC64) level from a raw MIDI message, if that's what the
/// message is. Deliberately a sibling of [`parse_midi`] that can never produce
/// a [`NoteMsg`]: pedal events travel their own channel end to end so they are
/// structurally unable to reach the roll's note path (and its idle-timer
/// reset — see `roll::Roll::pedal`).
fn parse_pedal(message: &[u8]) -> Option<u8> {
    if message.len() < 3 {
        return None;
    }
    // 0xB0 = control change (any channel); controller 64 = sustain pedal.
    (message[0] & 0xF0 == 0xB0 && message[1] == 64).then(|| message[2])
}
