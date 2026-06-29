#!/usr/bin/env python3
"""Fetch the files open-piano needs at runtime and drop them in the project root:

  * model.onnx       - Spotify Basic Pitch polyphonic transcription model
  * onnxruntime.dll  - the ONNX Runtime shared library (we link it dynamically)

Both are downloaded straight from their authoritative release sources, so there
is no pip install, no build step, and no dependency on a particular Python
version. Run this once:

    python download_model.py

Requires Python 3.8+ and internet access. Safe to re-run (it overwrites). No venv necessary.
"""

import io
import pathlib
import sys
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent

# Basic Pitch polyphonic transcription model, straight from Spotify's repo.
MODEL_URL = (
    "https://github.com/spotify/basic-pitch/raw/main/"
    "basic_pitch/saved_models/icassp_2022/nmp.onnx"
)

# Pin ONNX Runtime to the version the `ort` crate's `api-24` feature targets so
# the loaded DLL and the Rust bindings agree on the C API version.
ORT_VERSION = "1.24.2"
ORT_ZIP_URL = (
    f"https://github.com/microsoft/onnxruntime/releases/download/"
    f"v{ORT_VERSION}/onnxruntime-win-x64-{ORT_VERSION}.zip"
)


def download(url: str) -> bytes:
    print(f"  GET {url}")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted, hard-coded URLs)
        return resp.read()


def write_file(data: bytes, dest: pathlib.Path, what: str) -> None:
    dest.write_bytes(data)
    size = dest.stat().st_size / (1024 * 1024)
    print(f"  wrote {what}: {dest}  ({size:.1f} MiB)")


def main() -> None:
    print("[1/2] Downloading model.onnx...")
    model_bytes = download(MODEL_URL)
    write_file(model_bytes, ROOT / "model.onnx", "Basic Pitch ONNX model")

    print(f"[2/2] Downloading onnxruntime.dll (v{ORT_VERSION})...")
    zip_bytes = download(ORT_ZIP_URL)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        dll_names = [n for n in zf.namelist() if n.endswith("/lib/onnxruntime.dll")]
        if not dll_names:
            sys.exit("ERROR: onnxruntime.dll not found inside the release zip.")
        write_file(zf.read(dll_names[0]), ROOT / "onnxruntime.dll", "ONNX Runtime DLL")

    print("\nDone. `cargo run` should now find both files in the project root.")


if __name__ == "__main__":
    main()
