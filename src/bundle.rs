//! Assets compiled into the executable, so the shipped program is a **single
//! self-contained exe**.
//!
//! Two things used to travel beside the exe in the portable zip:
//!
//! * `model.onnx` — embedded here and handed to ort straight from memory
//!   (`commit_from_memory` in `inference.rs`). Never touches disk.
//! * `onnxruntime.dll` — also embedded, but a DLL can only be loaded from a
//!   *file*, so [`prepare_ort_dylib`] extracts it to a per-user cache dir on
//!   startup and points `ORT_DYLIB_PATH` at it. The extracted file is named
//!   by a content hash: a new app version carrying a new runtime writes a new
//!   file (never overwriting one an older running instance has mapped), and
//!   an unchanged runtime is reused across versions without rewriting.
//!
//! This is also what makes the auto-updater complete: it swaps only the exe
//! (`update.rs`), and since the exe now *contains* the model and runtime,
//! every update atomically carries its matching pair — they can never go
//! stale or mismatch the code.
//!
//! Build-time requirement: both files must exist in the project root when
//! compiling (`python download_model.py`); `include_bytes!` fails otherwise.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// Basic Pitch transcription model, loaded by the inference thread.
pub const MODEL: &[u8] = include_bytes!("../model.onnx");

/// ONNX Runtime shared library (Windows x64).
const ONNXRUNTIME_DLL: &[u8] = include_bytes!("../onnxruntime.dll");

/// Make the embedded ONNX Runtime loadable and set `ORT_DYLIB_PATH` to it.
///
/// Called once from `main()` **before** the inference thread starts. This is
/// plain file I/O — it never loads the DLL, so it's safe on the main thread
/// (loading is deferred to the inference thread; see the loader-lock note in
/// `main.rs`).
///
/// A pre-set `ORT_DYLIB_PATH` wins, so a developer can still test against a
/// different runtime without rebuilding. Extraction failures only disable the
/// microphone path: we leave the env var unset and the inference thread
/// reports the load error in the status bar, same as a missing DLL used to.
pub fn prepare_ort_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }
    match extract_dll() {
        Ok(path) => std::env::set_var("ORT_DYLIB_PATH", &path),
        Err(e) => eprintln!("[bundle] could not extract onnxruntime.dll: {e}"),
    }
}

/// Write the embedded DLL to `<cache>/open-piano/onnxruntime-<hash>.dll` if
/// it isn't there yet, and return its path.
fn extract_dll() -> std::io::Result<PathBuf> {
    // %LOCALAPPDATA% on Windows; fall back to the system temp dir so this
    // still works in odd environments (it just may re-extract more often).
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join("open-piano");
    fs::create_dir_all(&dir)?;

    let name = format!("onnxruntime-{:016x}.dll", fnv1a(ONNXRUNTIME_DLL));
    let path = dir.join(&name);
    if path.exists() {
        return Ok(path);
    }

    // Two instances may start at once (e.g. the run-the-app-twice local
    // test), so write to a unique temp name and rename into place; the loser
    // of that race just discards its copy.
    let tmp = dir.join(format!("{name}.tmp-{}", std::process::id()));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(ONNXRUNTIME_DLL)?;
    }
    if fs::rename(&tmp, &path).is_err() {
        let _ = fs::remove_file(&tmp);
        if !path.exists() {
            return Err(std::io::Error::other("rename failed and target missing"));
        }
    }

    // Housekeeping: drop runtimes older app versions extracted. A file still
    // mapped by a running instance won't delete on Windows — fine, skip it.
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let fname = entry.file_name();
            let fname = fname.to_string_lossy();
            if fname.starts_with("onnxruntime-") && fname != name.as_str() {
                let _ = fs::remove_file(entry.path());
            }
        }
    }

    Ok(path)
}

/// FNV-1a over the embedded bytes: stable, dependency-free content id for the
/// cache filename (not security — just change detection).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
