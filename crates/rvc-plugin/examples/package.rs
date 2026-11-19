use std::{env, fs, path::PathBuf};

use anyhow::{Context, Result, bail, ensure};

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).map(PathBuf::from).collect();
    let [library, destination] = args.as_slice() else {
        bail!("usage: package <native-library> <destination>");
    };
    ensure!(
        library.is_file(),
        "library does not exist: {}",
        library.display()
    );
    let (vst2, platform, vst3) = match library.extension().and_then(|ext| ext.to_str()) {
        Some("so") => ("RVC Realtime.so", "x86_64-linux", "RVCRealtime.so"),
        Some("dll") => ("RVC Realtime.dll", "x86_64-win", "RVCRealtime.vst3"),
        _ => bail!("expected a Linux .so or Windows .dll"),
    };
    let bundle = destination.join("RVCRealtime.vst3/Contents").join(platform);
    fs::create_dir_all(&bundle)?;
    for target in [destination.join(vst2), bundle.join(vst3)] {
        fs::copy(library, &target).with_context(|| format!("write {}", target.display()))?;
        println!("{}", target.display());
    }
    Ok(())
}
