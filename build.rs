//! Embed `assets/icon.ico` as the executable's file icon (what File Explorer
//! and the taskbar show for the `.exe` itself). The *window* icon is separate:
//! `assets/icon.png`, embedded via `bundle.rs` and set on the viewport.

fn main() {
    #[cfg(windows)]
    {
        println!("cargo:rerun-if-changed=assets/icon.ico");
        let mut res = winres::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        res.compile().expect("failed to embed exe icon resource");
    }
}
