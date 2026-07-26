//! Crash diagnostics: a write-through breadcrumb log, a Rust panic hook, a
//! native (SEH) crash handler for the exceptions a panic hook can't see, and
//! unclean-exit detection so the next launch can tell the user something went
//! wrong last time.
//!
//! Release builds are `windows_subsystem = "windows"` (no console) with no
//! crash dialogs and no logging of any kind — a panic or an access violation
//! just closes the window with nothing left behind. This module is the fix:
//! [`begin_session`] must run as the very first statement in `main()`, before
//! anything else can fail.
//!
//! **Write-through, not an in-memory ring.** Every breadcrumb is appended to
//! disk immediately (no `fsync` — that cost is reserved for the panic/crash
//! record itself). An in-memory-only log is lost in exactly the crashes that
//! matter most: an access violation in `onnxruntime.dll`, an OOM `abort()`, a
//! Task-Manager kill. A 200-entry ring is *also* kept, purely so a panic
//! record can embed the recent trail inline without re-reading the file.
//!
//! **Realtime discipline**: never call [`log`] from the cpal audio
//! capture/output callbacks, the midir callback, or the inference inner loop
//! — it allocates, locks, and does file I/O. Those paths report through their
//! existing channels/status cells; the input supervisor logs on their behalf.
//! The one documented exception is a cpal *stream-error* callback (fires once
//! when a device dies, not per buffer) — use [`log_from_realtime_callback`]
//! there instead, so the exception is explicit at the call site.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{Read as _, Seek, SeekFrom, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const LOG_FILE_NAME: &str = "open-piano.log";
/// Rotate the live log past this size (two files, so ~2 MiB total on disk).
const LOG_MAX_BYTES: u64 = 1024 * 1024;
/// Recent breadcrumbs kept in memory (in addition to on disk) purely so a
/// panic record can embed the trail inline.
const RING_CAPACITY: usize = 200;
/// How often [`heartbeat`] should be called (by the input supervisor) to keep
/// this session's marker looking alive.
pub const MARKER_HEARTBEAT: Duration = Duration::from_secs(30);
/// A per-pid marker untouched for this long is presumed dead (crashed, hung,
/// or killed), not just sleeping — 3x [`MARKER_HEARTBEAT`], so a slow disk or
/// one missed tick can't fake a crash report.
const MARKER_STALE_AFTER: Duration = Duration::from_secs(90);

/// Breadcrumb subsystem, prefixed onto every logged line.
pub enum Area {
    Net,
    Input,
    Audio,
    Inference,
    Record,
    Update,
    Prefs,
    Bundle,
    Ui,
}

impl Area {
    fn tag(&self) -> &'static str {
        match self {
            Area::Net => "net",
            Area::Input => "input",
            Area::Audio => "audio",
            Area::Inference => "inference",
            Area::Record => "record",
            Area::Update => "update",
            Area::Prefs => "prefs",
            Area::Bundle => "bundle",
            Area::Ui => "ui",
        }
    }
}

/// What [`begin_session`] found about the *previous* run of this app.
pub struct LastRun {
    pub unclean: Option<UncleanRun>,
}

/// A previous session whose per-pid marker was found stale (crashed, hung, or
/// killed without a chance to clean up after itself).
pub struct UncleanRun {
    pub pid: u32,
    pub version: String,
    pub started_unix: u64,
    /// Byte offset into the (possibly since-rotated) log where that session's
    /// own breadcrumbs begin — feed to [`read_since`] to show its tail.
    pub log_offset: u64,
}

struct LogState {
    file: File,
    ring: VecDeque<String>,
    /// This process's own `session-<pid>.lock`, removed by [`mark_clean_exit`].
    marker_path: PathBuf,
}

static LOG: OnceLock<Mutex<LogState>> = OnceLock::new();
/// Null-terminated UTF-16 log path, precomputed once so the native crash
/// handler never allocates (it may run on a corrupt heap).
#[cfg(windows)]
static LOG_PATH_W: OnceLock<Vec<u16>> = OnceLock::new();

/// Must be the first statement in `main()`. Rotates the log, scans for a
/// stale marker left by a previous unclean exit, writes this session's own
/// marker + header, and installs the panic hook and (on Windows) the native
/// crash handler.
pub fn begin_session() -> LastRun {
    let dir = crate::bundle::app_dir();
    let _ = fs::create_dir_all(&dir);
    let path = dir.join(LOG_FILE_NAME);
    rotate_if_needed(&path);

    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) else {
        // No log file — diagnostics degrade to debug-eprintln only, but
        // startup must never block on this.
        return LastRun { unclean: None };
    };

    let unclean = scan_stale_markers(&dir);

    let pid = std::process::id();
    let started_unix = unix_now();
    let log_offset = file.metadata().map(|m| m.len()).unwrap_or(0);
    let marker_path = dir.join(format!("session-{pid}.lock"));
    let _ = fs::write(
        &marker_path,
        format!("{} {started_unix} {log_offset}", env!("CARGO_PKG_VERSION")),
    );

    // The ORT dylib path isn't known yet here — `begin_session` runs before
    // `bundle::prepare_ort_dylib` deliberately (the panic hook must install
    // before ORT extraction can fail) — so it's logged separately as a
    // breadcrumb once `main` sets it, not baked into this header.
    let header = format!(
        "\n==== SESSION START ====\nversion: {}\npid:     {pid}\ntime:    {}\nexe:     {}\n",
        env!("CARGO_PKG_VERSION"),
        format_unix_iso(started_unix),
        std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_default(),
    );
    let _ = file.write_all(header.as_bytes());
    let _ = file.sync_all();

    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let _ = LOG_PATH_W.set(wide);
    }

    let _ = LOG.set(Mutex::new(LogState {
        file,
        ring: VecDeque::with_capacity(RING_CAPACITY),
        marker_path,
    }));

    install_panic_hook();
    #[cfg(windows)]
    install_native_handler();

    unclean
}

/// Remove this process's own marker — a normal, non-crashing exit. Call from
/// every clean-shutdown path: `on_exit`, `perform_restart` (which bypasses
/// destructors via `process::exit`), and once more right after `run_native`
/// returns, so no exit path is missed.
pub fn mark_clean_exit() {
    if let Some(lock) = LOG.get() {
        let state = lock.lock().unwrap_or_else(|e| e.into_inner());
        let _ = fs::remove_file(&state.marker_path);
    }
}

/// Refresh this process's marker mtime so the *next* launch's staleness scan
/// treats a still-running instance as alive, not crashed. Call periodically
/// (not every frame) from the input supervisor — deliberately not the GUI
/// thread, which can block arbitrarily long on the blocking `rfd` dialog.
pub fn heartbeat() {
    if let Some(lock) = LOG.get() {
        let state = lock.lock().unwrap_or_else(|e| e.into_inner());
        if let Ok(f) = OpenOptions::new().write(true).open(&state.marker_path) {
            let _ = f.set_modified(SystemTime::now());
        }
    }
}

/// Append a breadcrumb. Safe from any thread the app itself spawns and named
/// (see the module docs for the realtime exceptions). In debug builds this
/// also `eprintln!`s, matching every former call site's dev-time behavior.
pub fn log(area: Area, msg: impl Into<String>) {
    debug_assert!(
        std::thread::current().name().is_some(),
        "diag::log called from an unnamed thread — likely a realtime audio/MIDI \
         callback (cpal/midir spawn these without a name). Those must report \
         through their existing channels/status cells instead of logging \
         directly. If this really is one of the documented one-shot \
         stream-error exceptions, call log_from_realtime_callback instead."
    );
    log_impl(area, msg.into());
}

/// Same as [`log`], without the unnamed-thread assert, for the documented
/// one-shot cpal stream-error callbacks (`audio.rs`'s two capture `err_fn`s
/// and `synth.rs`'s output `err_fn`) — each fires once when a device dies, not
/// per audio buffer, so the allocation/lock/file-I/O cost is a non-issue. Do
/// not add new call sites here without updating this comment and CLAUDE.md's
/// realtime-discipline note.
pub fn log_from_realtime_callback(area: Area, msg: impl Into<String>) {
    log_impl(area, msg.into());
}

fn log_impl(area: Area, msg: String) {
    let thread = std::thread::current().name().unwrap_or("?").to_string();
    let line = format!("{} [{thread}] [{}] {msg}", format_unix_iso(unix_now()), area.tag());
    #[cfg(debug_assertions)]
    eprintln!("{line}");
    if let Some(lock) = LOG.get() {
        let mut state = lock.lock().unwrap_or_else(|e| e.into_inner());
        let _ = writeln!(state.file, "{line}");
        if state.ring.len() >= RING_CAPACITY {
            state.ring.pop_front();
        }
        state.ring.push_back(line);
    }
}

/// `%LOCALAPPDATA%\open-piano\open-piano.log` (or the fallback temp dir — see
/// `bundle::app_dir`).
pub fn log_path() -> PathBuf {
    crate::bundle::app_dir().join(LOG_FILE_NAME)
}

/// Read up to `max_bytes` of the log starting at `offset`, for the crash-chip
/// UI. `offset` is clamped to the current file length, so a since-rotated
/// offset degrades to "from the top of the current file" instead of nothing.
pub fn read_since(offset: u64, max_bytes: usize) -> String {
    let Ok(mut file) = File::open(log_path()) else {
        return String::new();
    };
    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let start = offset.min(len);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return String::new();
    }
    let to_read = ((len - start) as usize).min(max_bytes);
    let mut buf = vec![0u8; to_read];
    let n = file.read(&mut buf).unwrap_or(0);
    buf.truncate(n);
    String::from_utf8_lossy(&buf).into_owned()
}

/// Open Explorer with the log file pre-selected. `explorer.exe` returns exit
/// code 1 on success, so the spawn result (not its status) is all we check.
pub fn reveal_log_in_explorer() {
    let _ = std::process::Command::new("explorer")
        .arg(format!("/select,{}", log_path().display()))
        .spawn();
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Days-since-epoch -> proleptic-Gregorian (y, m, d), UTC. Howard Hinnant's
/// well-known `civil_from_days` (public domain) — used so a wall-clock
/// timestamp can be rendered without pulling in a date/time crate.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn format_unix_iso(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let tod = secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    let (h, mi, s) = (tod / 3600, (tod / 60) % 60, tod % 60);
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

fn rotate_if_needed(path: &Path) {
    let Ok(meta) = fs::metadata(path) else { return };
    if meta.len() < LOG_MAX_BYTES {
        return;
    }
    let rotated = path.with_extension("log.1");
    let _ = fs::remove_file(&rotated);
    let _ = fs::rename(path, &rotated);
}

/// Find every OTHER process's `session-<pid>.lock` in `dir`. A marker whose
/// mtime is fresh means that pid is genuinely alive right now (a second
/// instance running side by side) and is left alone. A stale one means that
/// process never got to clean up — the strongest signal available that it
/// crashed, hung, or was killed. Stale markers are consumed (removed) as
/// they're read, so a crash is only ever reported once; the most recent
/// candidate (by its own start time) is returned.
fn scan_stale_markers(dir: &Path) -> LastRun {
    let my_pid = std::process::id();
    let mut best: Option<UncleanRun> = None;
    let Ok(entries) = fs::read_dir(dir) else {
        return LastRun { unclean: None };
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(pid_str) = name.strip_prefix("session-").and_then(|s| s.strip_suffix(".lock")) else {
            continue;
        };
        let Ok(pid) = pid_str.parse::<u32>() else { continue };
        if pid == my_pid {
            continue; // can't happen for a fresh process; defensive
        }
        let stale = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|mtime| SystemTime::now().duration_since(mtime).ok())
            .is_some_and(|age| age >= MARKER_STALE_AFTER);
        if !stale {
            continue; // another instance is genuinely alive right now
        }
        let path = entry.path();
        if let Ok(text) = fs::read_to_string(&path) {
            let mut parts = text.split_whitespace();
            let version = parts.next().unwrap_or_default().to_string();
            let started_unix = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let log_offset = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let candidate = UncleanRun { pid, version, started_unix, log_offset };
            let better = match &best {
                None => true,
                Some(b) => candidate.started_unix >= b.started_unix,
            };
            if better {
                best = Some(candidate);
            }
        }
        let _ = fs::remove_file(&path); // consumed — never reported twice
    }
    LastRun { unclean: best }
}

// ---- Panic hook ----

fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // A panic *inside* the hook would abort immediately and lose the
        // record, so the actual work is wrapped and any panic swallowed.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| record_panic(info)));
        prev(info); // chain, so debug builds keep their default stderr output
    }));
}

fn record_panic(info: &std::panic::PanicHookInfo<'_>) {
    // Manual downcast, not `PanicHookInfo::payload_as_str` (stable only since
    // Rust 1.81; this repo pins no MSRV).
    let payload = info.payload();
    let msg: &str = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>");
    let location = info
        .location()
        .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
        .unwrap_or_else(|| "<unknown location>".to_string());
    let thread = std::thread::current().name().unwrap_or("<unnamed>").to_string();
    // `force_capture`, not `capture` — `capture` is gated on `RUST_BACKTRACE`,
    // which a double-clicked GUI app never has set.
    let bt = std::backtrace::Backtrace::force_capture();

    let Some(lock) = LOG.get() else { return };
    let mut state = lock.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    text.push_str("\n==== PANIC ====\n");
    let _ = writeln!(text, "thread:   {thread}");
    let _ = writeln!(text, "location: {location}");
    let _ = writeln!(text, "message:  {msg}");
    text.push_str("backtrace:\n");
    let _ = writeln!(text, "{bt}");
    if !state.ring.is_empty() {
        text.push_str("recent breadcrumbs:\n");
        for line in &state.ring {
            text.push_str(line);
            text.push('\n');
        }
    }
    // One write, one sync, under the file's mutex — concurrent panics on
    // different threads can't interleave their records.
    let _ = state.file.write_all(text.as_bytes());
    let _ = state.file.sync_all();
}

// ---- Native (non-Rust) crash handler: Windows only ----
//
// A Rust panic hook sees none of: an access violation in `onnxruntime.dll` /
// WASAPI / a GPU driver, an OOM abort, a double-panic abort, or a stack
// overflow. `SetUnhandledExceptionFilter` catches those. It runs on the
// faulting thread over a possibly-corrupt heap, so this handler allocates
// nothing, formats nothing through `format!`, and takes no mutex (the
// faulting thread may already hold the log mutex — taking it here would
// deadlock the crash handler). It hand-formats hex into a stack buffer and
// writes via raw `CreateFileW`/`WriteFile`/`CloseHandle`. The breadcrumbs are
// already on disk (that's the point of write-through), so this only needs to
// name the exception code and faulting address.
//
// Deliberately NOT implemented: `AddVectoredExceptionHandler` (fires on
// *first-chance* exceptions, including 0xE06D7363 — how Rust panics unwind on
// MSVC — a flood of normal exceptions); in-process `MiniDumpWriteDump`
// (writing a dump from a possibly-corrupt heap is exactly when it fails;
// doing it properly needs an out-of-process monitor). See README/CLAUDE.md
// for the manual Windows Event Log / WER LocalDumps escalation instead.
#[cfg(windows)]
mod native {
    use std::ffi::c_void;

    const STATUS_BREAKPOINT: u32 = 0x8000_0003;
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
    const FILE_APPEND_DATA: u32 = 0x0004;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_SHARE_WRITE: u32 = 0x0000_0002;
    const OPEN_ALWAYS: u32 = 4;
    const FILE_ATTRIBUTE_NORMAL: u32 = 0x0000_0080;

    type Handle = *mut c_void;
    const INVALID_HANDLE_VALUE: Handle = -1isize as Handle;

    #[repr(C)]
    struct ExceptionRecord {
        exception_code: u32,
        exception_flags: u32,
        exception_record: *mut ExceptionRecord,
        exception_address: *mut c_void,
        number_parameters: u32,
        exception_information: [usize; 15],
    }

    #[repr(C)]
    struct ExceptionPointers {
        exception_record: *mut ExceptionRecord,
        context_record: *mut c_void,
    }

    type TopLevelExceptionFilter = unsafe extern "system" fn(*mut ExceptionPointers) -> i32;

    #[link(name = "kernel32")]
    extern "system" {
        fn SetUnhandledExceptionFilter(
            filter: Option<TopLevelExceptionFilter>,
        ) -> Option<TopLevelExceptionFilter>;
        fn GetCurrentThreadId() -> u32;
        fn CreateFileW(
            file_name: *const u16,
            desired_access: u32,
            share_mode: u32,
            security_attributes: *mut c_void,
            creation_disposition: u32,
            flags_and_attributes: u32,
            template_file: Handle,
        ) -> Handle;
        fn WriteFile(
            file: Handle,
            buffer: *const u8,
            bytes_to_write: u32,
            bytes_written: *mut u32,
            overlapped: *mut c_void,
        ) -> i32;
        fn CloseHandle(object: Handle) -> i32;
    }

    /// Append `s` (ASCII) into `buf` at `*pos`, truncating silently if it
    /// wouldn't fit — never panics, never allocates.
    fn push_str(buf: &mut [u8], pos: &mut usize, s: &[u8]) {
        for &b in s {
            if *pos >= buf.len() {
                return;
            }
            buf[*pos] = b;
            *pos += 1;
        }
    }

    /// Append `0x` + hex digits of `val` into `buf` at `*pos`. No allocation.
    fn push_hex(buf: &mut [u8], pos: &mut usize, mut val: usize) {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        push_str(buf, pos, b"0x");
        if val == 0 {
            push_str(buf, pos, b"0");
            return;
        }
        let mut tmp = [0u8; 16];
        let mut n = 0;
        while val != 0 {
            tmp[n] = HEX[val & 0xf];
            val >>= 4;
            n += 1;
        }
        for i in (0..n).rev() {
            push_str(buf, pos, &[tmp[i]]);
        }
    }

    // The whole body is already an unsafe context (this is an `unsafe fn`),
    // so raw-pointer derefs and extern calls below need no nested `unsafe {}`
    // — edition 2021 (this crate's edition) treats an unsafe fn's body that
    // way; adding one would just trip the `unused_unsafe` lint.
    unsafe extern "system" fn seh_handler(ptrs: *mut ExceptionPointers) -> i32 {
        let (code, address) = if ptrs.is_null() || (*ptrs).exception_record.is_null() {
            (0u32, 0usize)
        } else {
            let rec = &*(*ptrs).exception_record;
            (rec.exception_code, rec.exception_address as usize)
        };
        // A breakpoint is expected under a debugger, not a real crash — don't
        // spend the write on it.
        if code == STATUS_BREAKPOINT {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        if let Some(path) = super::LOG_PATH_W.get() {
            let mut buf = [0u8; 512];
            let mut pos = 0usize;
            push_str(&mut buf, &mut pos, b"\r\n==== NATIVE CRASH (SEH) ====\r\ncode=");
            push_hex(&mut buf, &mut pos, code as usize);
            push_str(&mut buf, &mut pos, b" address=");
            push_hex(&mut buf, &mut pos, address);
            push_str(&mut buf, &mut pos, b" thread=");
            push_hex(&mut buf, &mut pos, GetCurrentThreadId() as usize);
            push_str(&mut buf, &mut pos, b"\r\n");
            let handle = CreateFileW(
                path.as_ptr(),
                FILE_APPEND_DATA,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null_mut(),
                OPEN_ALWAYS,
                FILE_ATTRIBUTE_NORMAL,
                std::ptr::null_mut(),
            );
            if handle != INVALID_HANDLE_VALUE {
                let mut written: u32 = 0;
                WriteFile(handle, buf.as_ptr(), pos as u32, &mut written, std::ptr::null_mut());
                CloseHandle(handle);
            }
        }
        // Not EXECUTE_HANDLER: returning CONTINUE_SEARCH lets WER still run,
        // so LocalDumps (see the README escalation steps) still produces a
        // real minidump alongside this text record.
        EXCEPTION_CONTINUE_SEARCH
    }

    pub(super) fn install() {
        unsafe {
            SetUnhandledExceptionFilter(Some(seh_handler));
        }
    }
}

#[cfg(windows)]
fn install_native_handler() {
    native::install();
}
