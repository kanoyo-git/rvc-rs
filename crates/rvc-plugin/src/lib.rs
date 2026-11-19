//! RVC Realtime VST2 and single-component VST3 entry points.

pub mod daw;
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod editor;
pub mod vst2;
pub mod vst3;
#[cfg(target_os = "windows")]
mod windows_paths;

use std::{
    env,
    path::{Path, PathBuf},
    ptr,
};

use anyhow::{Context, Result, ensure};
use rvc::{
    catalog::{ModelInfo, Paths},
    model::{self, ConversionSettings, PitchMethod, Rvc},
    realtime::context::{RealtimeContext, RealtimeOptions},
};

use daw::{LiveControls, ModelPaths, PluginModelSettings, Processor};

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn VSTPluginMain(host: Option<vst2::HostCallbackProc>) -> *mut vst2::AEffect {
    host.map_or(ptr::null_mut(), |host| {
        vst2::create(host, model_paths(), Box::new(load_processor))
    })
}

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "system" fn GetPluginFactory() -> *mut vst3::FactoryInterface {
    vst3::create_factory(model_paths(), Box::new(|| Box::new(load_processor)))
}

#[cfg(target_os = "windows")]
#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "system" fn InitDll() -> bool {
    true
}

#[cfg(target_os = "windows")]
#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "system" fn ExitDll() -> bool {
    true
}

#[cfg(target_os = "linux")]
#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn ModuleEntry(_: *mut std::ffi::c_void) -> bool {
    true
}

#[cfg(target_os = "linux")]
#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn ModuleExit() -> bool {
    true
}

fn load_processor(config: PluginModelSettings) -> Result<Processor> {
    ensure!(
        !config.paths.model.as_os_str().is_empty(),
        "select an RVC model in the preset or set RVC_PLUGIN_MODEL"
    );
    let root = if config.paths.rvc_root.as_os_str().is_empty() {
        Path::new(".")
    } else {
        &config.paths.rvc_root
    };
    let paths = Paths::new(root)?;
    let voice = paths.model(&config.paths.model)?;
    let speaker = ModelInfo::read(&voice)?.speaker(None)?;
    let index = if config.paths.index.as_os_str().is_empty() {
        paths.select_index(&voice, Some(speaker))
    } else {
        paths.require_index(&voice, speaker, Some(&config.paths.index), 1.0)?
    };
    let device = env::var("RVC_PLUGIN_DEVICE").unwrap_or_else(|_| "auto".to_owned());
    let mut engine = Rvc::load(
        &paths,
        &voice,
        Some(speaker),
        index.as_deref(),
        settings(config.initial_controls),
        model::weights::device(&device)?,
    )
    .with_context(|| format!("load voice {}", voice.display()))?;
    let mut context = RealtimeContext::new_vst(
        config.format,
        engine.rate(),
        RealtimeOptions {
            crossfade_ms: config.crossfade_ms,
            extra_ms: config.extra_ms,
            threshold_db: config.initial_controls.threshold_db,
            rms_mix: config.initial_controls.rms_mix,
        },
    )?;
    let frame = (config.format.sample_rate / 100) as usize;
    let block = config.format.block_frames / frame;
    Ok(Box::new(move |input, output, discontinuity, live| {
        let settings = settings(live);
        engine.update_pitch_method(&paths, settings.f0_method)?;
        if discontinuity {
            engine.reset_stream();
        }
        context.set_levels(live.threshold_db, live.rms_mix)?;
        context.process(input, output, discontinuity, |samples, skip, frames| {
            engine.convert_window(samples, settings, skip, frames, block)
        })
    }))
}

fn settings(controls: LiveControls) -> ConversionSettings {
    ConversionSettings {
        pitch: controls.pitch as f32,
        formant: controls.formant,
        index_rate: controls.index_rate,
        f0_method: match controls.f0_method {
            1 => PitchMethod::Fcpe,
            2 => PitchMethod::Pm,
            _ => PitchMethod::Rmvpe,
        },
        ..Default::default()
    }
}

fn model_paths() -> ModelPaths {
    let mut paths = legacy_paths();
    for (value, key) in [
        (&mut paths.rvc_root, "RVC_PLUGIN_ROOT"),
        (&mut paths.model, "RVC_PLUGIN_MODEL"),
        (&mut paths.index, "RVC_PLUGIN_INDEX"),
    ] {
        if let Some(path) = env::var_os(key) {
            *value = PathBuf::from(path);
        }
    }
    paths
}

#[cfg(not(target_os = "windows"))]
fn legacy_paths() -> ModelPaths {
    ModelPaths::default()
}

#[cfg(target_os = "windows")]
fn legacy_paths() -> ModelPaths {
    windows_paths::read()
}
