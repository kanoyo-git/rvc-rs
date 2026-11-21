use std::io;
use std::ops::ControlFlow;
use std::path::Path;
use std::process::{Command, Stdio};
use std::{thread, time::Duration};

use anyhow::{Context, Result, bail, ensure};
use tempfile::NamedTempFile;

const MIB: f64 = 1_048_576.0;

/// Size of the file behind `url` after redirects, when the server tells it.
fn remote_size(url: &str) -> Option<u64> {
    let output = Command::new("curl")
        .args([
            "--disable", "--head", "--location", "--silent", "--proto", "=http,https",
            "--proto-redir", "=http,https", "--max-redirs", "8", "--connect-timeout", "20",
            "--max-time", "30", "--",
        ])
        .arg(url)
        .output()
        .ok()?;
    let headers = String::from_utf8_lossy(&output.stdout);
    // Hugging Face's first hop carries the real size of LFS files.
    let linked = headers.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case("x-linked-size").then(|| value.trim().parse().ok())?
    });
    linked.or_else(|| {
        headers.lines().rev().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().ok())?
                .filter(|&size: &u64| size > 0)
        })
    })
}

/// Download `url` to `output`, reporting `Downloading <label> · <percent>% · <done> / <total> MiB`
/// four times a second. `size` skips the size request when the caller knows it.
pub(crate) fn download(
    url: &str,
    output: &Path,
    label: &str,
    size: Option<u64>,
    report: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<()> {
    let label = if label.is_empty() { "file" } else { label };
    ensure!(
        report(&format!("Downloading {label}")).is_continue(),
        "download cancelled"
    );
    let total = size.or_else(|| remote_size(url)).filter(|&size| size > 0);
    let mut stderr = NamedTempFile::new()?;
    let mut child = Command::new("curl")
        .args([
            "--disable",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--proto",
            "=http,https",
            "--proto-redir",
            "=http,https",
            "--retry",
            "2",
            "--retry-all-errors",
            "--max-redirs",
            "8",
            "--connect-timeout",
            "20",
            "--max-time",
            "3600",
            "--max-filesize",
            "4294967296",
            "--output",
        ])
        .arg(output)
        .arg("--")
        .arg(url)
        .stdout(Stdio::null())
        .stderr(stderr.reopen()?)
        .spawn()
        .context("starting curl to download models")?;
    loop {
        if let Some(status) = child.try_wait()? {
            let error = io::read_to_string(&mut stderr)?;
            ensure!(status.success(), "downloading {label} failed: {}", error.trim());
            return Ok(());
        }
        let bytes = output.metadata().map_or(0, |file| file.len());
        let line = match total {
            Some(total) => format!(
                "Downloading {label} · {:.0}% · {:.1} / {:.1} MiB",
                (100.0 * bytes as f64 / total as f64).min(100.0),
                bytes as f64 / MIB,
                total as f64 / MIB
            ),
            None => format!("Downloading {label} · {:.1} MiB", bytes as f64 / MIB),
        };
        if report(&line).is_break() {
            let _ = child.kill();
            let _ = child.wait();
            bail!("download cancelled");
        }
        thread::sleep(Duration::from_millis(250));
    }
}
