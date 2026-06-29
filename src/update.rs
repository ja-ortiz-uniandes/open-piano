//! In-app auto-update against GitHub Releases.
//!
//! On launch a background thread asks the GitHub Releases API whether a tag
//! newer than the running `CARGO_PKG_VERSION` exists. If so it downloads that
//! release's portable zip and atomically swaps the running `open-piano.exe` in
//! place (via `self_update` → `self_replace`: rename the live exe, drop the new
//! one beside it). The swap takes effect on the next launch, so the UI surfaces
//! "update ready" with a one-click **Restart now** ([`restart`]) — or the user
//! can just reopen the app later. This is the hands-off path so the professor's
//! copy updates itself instead of needing a manual zip-replacement (which still
//! works — see the README).
//!
//! Only the **executable** is replaced. `onnxruntime.dll` and `model.onnx` ship
//! in the same zip, but `self_update` touches just the binary; those files
//! change across releases only rarely, and a code update never *requires* a new
//! model. If a release ever needs a new runtime/model, fall back to the manual
//! zip-replacement documented in the README.
//!
//! Everything here runs **off the GUI thread** — the GitHub API call, the
//! download, and the file swap are all blocking I/O that would freeze the window
//! (and the check shouldn't gate startup either way). It never loads ONNX
//! Runtime, so it's unaffected by the Windows loader-lock constraint that
//! governs the rest of `main` (see `main.rs`). A failed check (offline,
//! rate-limited, read-only folder) is non-fatal: the app runs normally and the
//! status just reads "update check failed".

use std::sync::{Arc, Mutex};
use std::thread;

/// GitHub repo the releases are published to (see `.github/workflows/release.yml`).
const REPO_OWNER: &str = "ja-ortiz-uniandes";
const REPO_NAME: &str = "open-piano";

/// Substring that identifies our single Windows release asset
/// (`open-piano-vX.Y.Z-win-x64.zip`, see the release workflow). We match the
/// asset on this rather than `self_update`'s default (the build target triple,
/// e.g. `x86_64-pc-windows-msvc`), which never appears in the asset name.
const ASSET_TARGET: &str = "win-x64";

/// Where the (single, one-shot) update attempt currently stands. Polled
/// read-only by the UI each frame to render a status line.
#[derive(Clone, Debug, Default)]
pub enum UpdateState {
    /// The check is in flight (or hasn't finished its first run yet).
    #[default]
    Checking,
    /// The running build is already the newest release.
    UpToDate,
    /// A newer build was downloaded and staged; relaunch to run it.
    Ready { version: String },
    /// The check or download failed (offline, rate-limited, read-only folder,
    /// …). Non-fatal — the app keeps running on the current build.
    Failed { reason: String },
}

/// UI-side handle to the auto-updater. The work happens on a detached background
/// thread; this just exposes the latest [`UpdateState`].
pub struct Updater {
    state: Arc<Mutex<UpdateState>>,
}

impl Updater {
    /// A snapshot of the current update state for rendering.
    pub fn state(&self) -> UpdateState {
        self.state.lock().unwrap().clone()
    }
}

/// Spawn the update check on a background thread and return immediately. The
/// thread is detached (one-shot work): if the app exits mid-download the OS
/// reaps it, leaving at worst a temp file that `self_update` cleans up next run.
pub fn start() -> Updater {
    let state = Arc::new(Mutex::new(UpdateState::Checking));
    {
        let state = Arc::clone(&state);
        thread::Builder::new()
            .name("auto-update".into())
            .spawn(move || {
                let next = match run_update() {
                    Ok(Some(version)) => UpdateState::Ready { version },
                    Ok(None) => UpdateState::UpToDate,
                    Err(e) => UpdateState::Failed { reason: e.to_string() },
                };
                if let Ok(mut s) = state.lock() {
                    *s = next;
                }
            })
            .expect("failed to spawn auto-update thread");
    }
    Updater { state }
}

/// Check GitHub, and if a newer release exists download it and swap the running
/// exe. Returns `Ok(Some(version))` when a newer build was staged, `Ok(None)`
/// when already current, and `Err` on any network/IO failure (offline,
/// rate-limited, non-writable install folder, …).
fn run_update() -> Result<Option<String>, Box<dyn std::error::Error>> {
    let status = self_update::backends::github::Update::configure()
        .repo_owner(REPO_OWNER)
        .repo_name(REPO_NAME)
        .bin_name("open-piano.exe")
        // Pick our single asset by the constant `win-x64` substring.
        .target(ASSET_TARGET)
        // Path to the exe *inside* the zip. The release workflow wraps everything
        // in a single folder named after the tag — `open-piano-v0.2.0-win-x64/`
        // (`$name` = `open-piano-<github.ref_name>-win-x64`). `self_update`
        // expands `{{ version }}` to the release version with the leading `v`
        // stripped (confirmed against its source: `tag.trim_start_matches('v')`),
        // so `open-piano-v{{ version }}-win-x64/{{ bin }}` reconstructs that path
        // exactly. `{{ bin }}` → the `bin_name` below.
        .bin_path_in_archive("open-piano-v{{ version }}-win-x64/{{ bin }}")
        // No TTY in a windowed app, and the check is meant to be invisible.
        .show_download_progress(false)
        .no_confirm(true)
        .current_version(self_update::cargo_crate_version!())
        .build()?
        .update()?;

    // `.update()` returns a `Status`; in self_update 0.44 the query method is
    // `updated()` (no `is_` prefix). The `is_updated()` name does not exist on
    // this version, so do NOT rename it back — it won't compile.
    Ok(if status.updated() {
        Some(status.version().to_string())
    } else {
        None
    })
}

/// Relaunch the (already-swapped) executable and exit, so the user lands on the
/// new build without manually reopening. Call from the GUI thread once the state
/// is [`UpdateState::Ready`].
///
/// `self_replace` left the new exe at the original `current_exe()` path, so
/// spawning it starts the updated binary; we then exit to release the old one. If
/// the relaunch can't be set up we still exit — reopening by hand runs the new
/// build all the same.
pub fn restart() -> ! {
    if let Ok(exe) = std::env::current_exe() {
        let _ = std::process::Command::new(exe).spawn();
    }
    std::process::exit(0);
}
