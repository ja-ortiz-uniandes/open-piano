//! In-app auto-update against GitHub Releases.
//!
//! On launch a background thread asks the GitHub Releases API whether a newer
//! release exists (see [`pick_update`] for the stable-vs-preview policy). If so
//! it downloads that release's portable zip and atomically swaps the running
//! `open-piano.exe` in
//! place (via `self_update` → `self_replace`: rename the live exe, drop the new
//! one beside it). The swap takes effect on the next launch, so the UI surfaces
//! "update ready" with a one-click **Restart now** ([`restart`]) — or the user
//! can just reopen the app later. This is the hands-off path so the professor's
//! copy updates itself instead of needing a manual zip-replacement (which still
//! works — see the README).
//!
//! Only the **executable** is replaced — which since v0.2.2 is the whole
//! program: the ONNX model and ONNX Runtime are embedded in the exe (see
//! `bundle.rs`), so an exe swap atomically updates the code *and* its matching
//! model/runtime. Nothing beside the exe can go stale.
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
///
/// We don't let `self_update`'s `.update()` pick the release: its built-in logic
/// just takes the newest release by date among higher versions, which can't
/// express the stable-vs-preview policy we want. Instead we list every release,
/// choose one ourselves ([`pick_update`]), and pin that exact tag.
fn run_update() -> Result<Option<String>, Box<dyn std::error::Error>> {
    let current = self_update::cargo_crate_version!();

    // List *all* published releases — the GitHub `/releases` endpoint, which
    // includes pre-releases — keeping only those carrying our Windows asset
    // (matched on the `win-x64` substring, since the build target triple never
    // appears in the asset name).
    let releases = self_update::backends::github::ReleaseList::configure()
        .repo_owner(REPO_OWNER)
        .repo_name(REPO_NAME)
        .with_target(ASSET_TARGET)
        .build()?
        .fetch()?;

    let Some(target) = pick_update(current, &releases)? else {
        return Ok(None);
    };

    // Pin the exact tag we chose (releases carry the version sans `v`; the tag is
    // `vX.Y.Z`). With a target version set, `self_update` downloads and swaps
    // that release unconditionally — it does *no* internal newer-than gate — so
    // `pick_update` is solely responsible for only ever returning a strictly
    // newer release.
    let tag = format!("v{target}");
    self_update::backends::github::Update::configure()
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
        .current_version(current)
        .target_version_tag(&tag)
        .build()?
        .update()?;

    Ok(Some(target))
}

/// Choose which release (if any) to update `current` to, under our pre-1.0
/// policy:
///
/// * A **pre-release** is any `0.x.y` tag — the same `v0.*` rule the release
///   workflow uses to set GitHub's pre-release flag. `1.0.0`+ tags are stable.
///   `self_update`'s `Release` doesn't expose GitHub's `prerelease` boolean, so
///   we classify by the version number; because the workflow keys off the same
///   number, the two can't disagree.
/// * Prefer the highest **stable** release newer than `current`. A stable
///   install only ever moves to a higher stable release.
/// * Only when there is no higher stable release *and* `current` is itself a
///   pre-release do we fall back to the highest **pre-release** newer than
///   `current`. So preview users roll forward across previews, but a stable
///   install is never pulled back onto a preview.
///
/// Returns the chosen version string (no leading `v`), or `None` when there's
/// nothing newer to move to.
fn pick_update(
    current: &str,
    releases: &[self_update::update::Release],
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    use self_update::version::bump_is_greater;

    // Highest release strictly newer than `current` among those `keep` accepts.
    // Compares by version number (not publish date), so an out-of-order release
    // list can't trick us into picking a lower version.
    let newest = |keep: &dyn Fn(&str) -> bool| -> Result<Option<String>, Box<dyn std::error::Error>> {
        let mut best: Option<String> = None;
        for r in releases {
            // A single malformed tag (e.g. `nightly-2026-07-01`, `v0.8`) must
            // not fail the whole check — that would brick auto-update for every
            // installed copy, *including* against the later well-formed release
            // that would fix it. Skip unparseable / not-newer releases instead.
            if !keep(&r.version) || !matches!(bump_is_greater(current, &r.version), Ok(true)) {
                continue;
            }
            let is_better = match &best {
                None => true,
                Some(b) => matches!(bump_is_greater(b, &r.version), Ok(true)),
            };
            if is_better {
                best = Some(r.version.clone());
            }
        }
        Ok(best)
    };

    // Pre-release = any `0.x.y` tag (the workflow's own `v0.*` rule) *or* any
    // tag carrying a semver pre-release suffix (`1.0.0-rc.1`). Classifying the
    // latter as stable would pull stable installs onto release candidates.
    let is_pre = |v: &str| v.starts_with("0.") || v.contains('-');
    let is_stable = |v: &str| !is_pre(v);

    // 1. A higher stable release always wins (for stable and preview users alike).
    if let Some(stable) = newest(&is_stable)? {
        return Ok(Some(stable));
    }
    // 2. Otherwise a preview install may roll forward to a higher preview.
    if is_pre(current) {
        if let Some(pre) = newest(&is_pre)? {
            return Ok(Some(pre));
        }
    }
    Ok(None)
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
