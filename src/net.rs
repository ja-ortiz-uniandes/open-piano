//! Peer-to-peer UDP transport.
//!
//! A single UDP socket is bound to the local port. A listener thread reads
//! incoming datagrams and forwards decoded [`Packet`]s (note events *and* color
//! announcements) to the UI over an mpsc channel; the same socket (cloned) is
//! used to send local packets to the configured remote peer.
//!
//! UDP is deliberate: for a live duet visualizer the lowest possible latency
//! beats guaranteed delivery — a dropped note-on is corrected by the next event
//! within milliseconds, whereas TCP head-of-line blocking would add lag. Each
//! event is its own datagram, sent immediately, so there is no batching delay.

use std::net::{SocketAddr, UdpSocket};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;

use crate::note::Packet;

/// An active P2P connection: an owned send-socket plus the receiver of remote
/// packets (notes and color announcements).
pub struct Peer {
    socket: Arc<UdpSocket>,
    pub remote: SocketAddr,
    pub incoming: Receiver<Packet>,
}

impl Peer {
    /// Bind `0.0.0.0:local_port`, start the listener thread, and prepare to send
    /// to `remote`.
    pub fn connect(local_port: u16, remote: SocketAddr) -> std::io::Result<Peer> {
        let socket = UdpSocket::bind(("0.0.0.0", local_port))?;
        let socket = Arc::new(socket);

        let (tx, rx) = mpsc::channel::<Packet>();

        // Listener thread.
        let recv_sock = Arc::clone(&socket);
        thread::Builder::new()
            .name("udp-listener".into())
            .spawn(move || {
                let mut buf = [0u8; 64];
                loop {
                    match recv_sock.recv_from(&mut buf) {
                        Ok((n, _from)) => {
                            if let Some(msg) = Packet::decode(&buf[..n]) {
                                if tx.send(msg).is_err() {
                                    break; // UI gone.
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[net] recv error: {e}");
                            break;
                        }
                    }
                }
            })
            .expect("failed to spawn udp listener thread");

        Ok(Peer {
            socket,
            remote,
            incoming: rx,
        })
    }

    /// Send a packet (note event or color announcement) to the remote peer.
    /// Best-effort; errors are logged but not fatal.
    pub fn send(&self, packet: Packet) {
        let bytes = packet.encode();
        if let Err(e) = self.socket.send_to(&bytes, self.remote) {
            eprintln!("[net] send error: {e}");
        }
    }
}
