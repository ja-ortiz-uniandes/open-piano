# Getting the model + ONNX Runtime

`open-piano` does **machine-learning** note transcription with Spotify's
**Basic Pitch** model, run via ONNX Runtime. Both are **embedded into the
executable at build time** (`src/bundle.rs`, via `include_bytes!`), so you must
place **two** files in the project root before **building**:

| File              | What it is                                       |
|-------------------|--------------------------------------------------|
| `model.onnx`      | Basic Pitch polyphonic piano transcription model |
| `onnxruntime.dll` | ONNX Runtime shared library (Windows x64)        |

Every `cargo build` needs them (the build fails with a clear `include_bytes!`
error if they're missing). End users never see these files — the shipped exe is
self-contained; at startup it unpacks the runtime to `%LOCALAPPDATA%\open-piano\`
(a DLL must be a real file for Windows to load it) and loads the model straight
from memory.

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

* `Model: loaded (built-in)` — success.
* `Model load FAILED: ...` — the message includes the real cause (an
  incompatible ONNX Runtime version, a failed unpack to `%LOCALAPPDATA%`,
  etc.). The UI stays fully responsive either way; you just won't see detected
  notes.

### Pointing elsewhere

To test against a different ONNX Runtime build without recompiling, set
`ORT_DYLIB_PATH` to its full path — it takes precedence over the embedded copy:

```powershell
$env:ORT_DYLIB_PATH = "C:\path\to\onnxruntime.dll"
cargo run
```
