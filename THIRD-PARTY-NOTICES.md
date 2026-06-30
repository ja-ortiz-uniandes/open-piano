# Third-party notices

open-piano is distributed with the following third-party components. Their
licenses are permissive and travel with the release; this file collects the
attributions they ask for.

## Bundled in the release zip

- **ONNX Runtime** (`onnxruntime.dll`) — © Microsoft, licensed under the MIT
  License. https://github.com/microsoft/onnxruntime
- **Basic Pitch model** (`model.onnx`) — © Spotify, licensed under the Apache
  License 2.0. https://github.com/spotify/basic-pitch

## Rust dependencies

The Rust crates linked into `open-piano.exe` are each licensed under permissive
terms (MIT and/or Apache-2.0), including `eframe`/`egui`, `cpal`, `midir`,
`ort`, `hound`, and `self_update`. See each crate's repository for its full
license text; `cargo about` or `cargo license` can regenerate the complete
list.
