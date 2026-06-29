# Getting the model + ONNX Runtime

`open-piano` does **machine-learning** note transcription with Spotify's
**Basic Pitch** model, run via ONNX Runtime. Because the project links ONNX
Runtime *dynamically* (so `cargo build` never depends on a binary-download CDN),
you must place **two** files in the project root before running:

| File              | What it is                                       |
|-------------------|--------------------------------------------------|
| `model.onnx`      | Basic Pitch polyphonic piano transcription model |
| `onnxruntime.dll` | ONNX Runtime shared library (Windows x64)        |

The app looks for both in the current working directory (the project root when
you `cargo run`).

---

## Option A — one command (recommended)

Requires Python 3.8+. This downloads both files directly from their
authoritative release sources (Spotify's repo and Microsoft's ONNX Runtime
releases) using only the Python standard library — no pip, no build step, and
no dependency on a particular Python version:

```powershell
python download_model.py
```

That writes both `model.onnx` and `onnxruntime.dll` into the project root.

---

## Option B — manual download (PowerShell)

### 1. The model

Pull the Basic Pitch ONNX model straight from Spotify's repository:

```powershell
Invoke-WebRequest `
  -Uri "https://github.com/spotify/basic-pitch/raw/main/basic_pitch/saved_models/icassp_2022/nmp.onnx" `
  -OutFile "model.onnx"
```

> If that path 404s (Spotify occasionally reorganizes the repo), use Option A —
> the pip wheel is the authoritative source for the model file.

### 2. ONNX Runtime DLL

Download the official Windows x64 build and copy the DLL to the project root:

```powershell
$ver = "1.24.2"
Invoke-WebRequest `
  -Uri "https://github.com/microsoft/onnxruntime/releases/download/v$ver/onnxruntime-win-x64-$ver.zip" `
  -OutFile "ort.zip"
Expand-Archive ort.zip -DestinationPath ort_tmp -Force
Copy-Item "ort_tmp\onnxruntime-win-x64-$ver\lib\onnxruntime.dll" ".\onnxruntime.dll"
Remove-Item -Recurse -Force ort.zip, ort_tmp
```

Use ONNX Runtime **1.24.x** — it matches the `api-24` feature pinned for the
`ort` crate in `Cargo.toml`. A very different runtime version may report a C API
mismatch at load time.

---

## Verifying

```powershell
cargo run
```

The status bar at the bottom of the window shows model state:

* `Model: loaded model.onnx` — success.
* `Model load FAILED: ...` — the message includes the real cause (missing
  `model.onnx`, missing/incompatible `onnxruntime.dll`, etc.). The UI stays
  fully responsive either way; you just won't see detected notes until both
  files are in place.

### Pointing elsewhere

If you keep `onnxruntime.dll` somewhere else, set `ORT_DYLIB_PATH` to its full
path instead of copying it into the project root:

```powershell
$env:ORT_DYLIB_PATH = "C:\path\to\onnxruntime.dll"
cargo run
```
