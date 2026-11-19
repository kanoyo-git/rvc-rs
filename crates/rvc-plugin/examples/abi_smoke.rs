//! Load the finished shared library as a host would.

use std::{
    env,
    ffi::c_void,
    path::PathBuf,
    ptr,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use libloading::Library;
use rvc_realtime::{
    daw::{BLOCK_MS, ENGINE, PARAMETERS},
    vst3::COMPONENT_UID,
};
use vst::{
    api::{AEffect, HostCallbackProc, consts::VST_MAGIC},
    host, plugin,
};
use vst3::{
    ComPtr, Interface,
    Steinberg::{Vst::*, *},
};

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let path = args
        .next()
        .context("usage: abi_smoke <native-library> [voice-input]")?;
    let source = args
        .next()
        .map(|path| rvc::audio::read_mono(&PathBuf::from(path), 48_000))
        .transpose()?;
    if let Some(source) = &source {
        ensure!(!source.is_empty(), "voice input is empty");
    }
    // SAFETY: the smoke program is explicitly invoked on the locally built RVC
    // library. Its entry points and dependent libraries are trusted here.
    let library = unsafe { Library::new(path)? };
    smoke_vst2(&library, source.as_deref())?;
    smoke_vst3(&library, source.as_deref())?;
    println!("VST2 and VST3 exports, layouts, parameters, latency and dry PCM verified");
    if source.is_some() {
        println!("VST2 and VST3 produced synthesized PCM with the configured model");
    }
    Ok(())
}

extern "C" fn host_callback(
    _: *mut AEffect,
    event: i32,
    _: i32,
    _: isize,
    _: *mut c_void,
    _: f32,
) -> isize {
    if event == host::OpCode::Version as i32 {
        2400
    } else {
        0
    }
}

struct Vst2Instance(*mut AEffect);

impl Vst2Instance {
    fn dispatch(&self, event: plugin::OpCode, value: isize, float: f32) -> isize {
        // SAFETY: the smoke host owns a live plugin until Drop.
        unsafe { ((*self.0).dispatcher)(self.0, event as i32, 0, value, ptr::null_mut(), float) }
    }
}

impl Drop for Vst2Instance {
    fn drop(&mut self) {
        self.dispatch(plugin::OpCode::StateChanged, 0, 0.0);
        self.dispatch(plugin::OpCode::Shutdown, 0, 0.0);
    }
}

fn smoke_vst2(library: &Library, voice: Option<&[f32]>) -> Result<()> {
    type Main = unsafe extern "C" fn(Option<HostCallbackProc>) -> *mut AEffect;
    // SAFETY: this is the exported VST2.4 entry point and its documented ABI.
    let entry = unsafe { library.get::<Main>(b"VSTPluginMain")? };
    let raw = unsafe { entry(Some(host_callback)) };
    ensure!(!raw.is_null(), "VSTPluginMain returned null");
    let host = Vst2Instance(raw);
    // SAFETY: host owns a live AEffect for the rest of the test.
    let effect = unsafe { &*raw };
    ensure!(
        effect.magic == VST_MAGIC
            && effect.uniqueId == i32::from_be_bytes(*b"Rvcr")
            && effect.numInputs == 2
            && effect.numOutputs == 2
            && effect.numParams == 12,
        "unexpected VST2 class properties"
    );
    host.dispatch(plugin::OpCode::Initialize, 0, 0.0);
    host.dispatch(plugin::OpCode::SetSampleRate, 0, 48_000.0);
    host.dispatch(plugin::OpCode::SetBlockSize, 64, 0.0);
    host.dispatch(plugin::OpCode::StateChanged, 1, 0.0);
    ensure!(
        unsafe { (*raw).initialDelay } == 12_480,
        "wrong VST2 latency"
    );

    let (source_l, source_r) = ([0.25_f32; 64], [-0.125_f32; 64]);
    let mut left = [0.0; 64];
    let mut right = [0.0; 64];
    let inputs = [source_l.as_ptr(), source_r.as_ptr()];
    let mut outputs = [left.as_mut_ptr(), right.as_mut_ptr()];
    // SAFETY: two valid input/output channels of 64 frames match the current
    // stereo host layout. Calls read and write the indicated arrays.
    unsafe { ((*raw).processReplacing)(raw, inputs.as_ptr(), outputs.as_mut_ptr(), 64) };
    ensure!(
        (left, right) == (source_l, source_r),
        "VST2 dry audio changed"
    );
    if let Some(voice) = voice {
        host.dispatch(plugin::OpCode::SetBlockSize, VOICE_BLOCK as isize, 0.0);
        // SAFETY: the effect and input/output arrays remain valid in every host call.
        unsafe {
            ((*raw).setParameter)(raw, BLOCK_MS as i32, PARAMETERS[BLOCK_MS].normalize(200.0));
            ((*raw).setParameter)(raw, 8, PARAMETERS[8].normalize(600.0));
            ((*raw).setParameter)(raw, ENGINE as i32, 1.0);
            check_voice(voice, |input, dry_right, left, right| {
                let channels = [input.as_ptr(), dry_right.as_ptr()];
                let mut output = [left.as_mut_ptr(), right.as_mut_ptr()];
                ((*raw).processReplacing)(
                    raw,
                    channels.as_ptr(),
                    output.as_mut_ptr(),
                    VOICE_BLOCK as i32,
                );
                Ok(())
            })
            .context("VST2 native voice")?;
        }
    }
    Ok(())
}

fn smoke_vst3(library: &Library, voice: Option<&[f32]>) -> Result<()> {
    type GetFactory = unsafe extern "system" fn() -> *mut IPluginFactory;
    // SAFETY: these are the documented exports from the built bundle.
    unsafe {
        #[cfg(target_os = "linux")]
        let exit = {
            type Entry = unsafe extern "C" fn(*mut c_void) -> bool;
            type Exit = unsafe extern "C" fn() -> bool;
            let entry = library.get::<Entry>(b"ModuleEntry")?;
            ensure!(entry(ptr::null_mut()), "VST3 module failed to initialize");
            library.get::<Exit>(b"ModuleExit")?
        };
        #[cfg(target_os = "windows")]
        let exit = {
            type Lifecycle = unsafe extern "system" fn() -> bool;
            let entry = library.get::<Lifecycle>(b"InitDll")?;
            ensure!(entry(), "VST3 DLL failed to initialize");
            library.get::<Lifecycle>(b"ExitDll")?
        };
        let entry = library.get::<GetFactory>(b"GetPluginFactory")?;
        let factory = ComPtr::from_raw(entry()).context("VST3 factory null")?;
        ensure!(factory.countClasses() == 1, "wrong VST3 class count");
        let mut raw = ptr::null_mut::<c_void>();
        ensure!(
            factory.createInstance(
                COMPONENT_UID.as_ptr(),
                IComponent::IID.as_ptr().cast(),
                &mut raw
            ) == kResultOk,
            "expected the original VST3 class and component interface"
        );
        let component = ComPtr::from_raw(raw.cast::<IComponent>()).context("null component")?;
        let controller = component
            .cast::<IEditController>()
            .context("no controller interface")?;
        let processor = component
            .cast::<IAudioProcessor>()
            .context("no audio processor interface")?;
        ensure!(component.initialize(ptr::null_mut()) == kResultOk);
        ensure!(
            controller.getParameterCount() == 13,
            "wrong VST3 parameter count"
        );
        let mut input = SpeakerArr::kMono;
        let mut output = SpeakerArr::kStereo;
        ensure!(processor.setBusArrangements(&mut input, 1, &mut output, 1) == kResultOk);
        ensure!(
            processor.getLatencySamples() == 12_480,
            "wrong VST3 latency"
        );
        let mut setup = ProcessSetup {
            processMode: ProcessModes_::kRealtime as i32,
            symbolicSampleSize: SymbolicSampleSizes_::kSample32 as i32,
            maxSamplesPerBlock: 64,
            sampleRate: 48_000.0,
        };
        ensure!(processor.setupProcessing(&mut setup) == kResultOk);
        ensure!(component.setActive(1) == kResultOk);
        ensure!(processor.setProcessing(1) == kResultOk);

        let mut source = [0.25_f32; 64];
        let mut left = [0.0_f32; 64];
        let mut right = [0.0_f32; 64];
        let mut input_channels = [source.as_mut_ptr()];
        let mut output_channels = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut input = AudioBusBuffers {
            numChannels: 1,
            silenceFlags: 0,
            __field0: AudioBusBuffers__type0 {
                channelBuffers32: input_channels.as_mut_ptr(),
            },
        };
        let mut output = AudioBusBuffers {
            numChannels: 2,
            silenceFlags: 0,
            __field0: AudioBusBuffers__type0 {
                channelBuffers32: output_channels.as_mut_ptr(),
            },
        };
        let mut data = ProcessData {
            processMode: setup.processMode,
            symbolicSampleSize: setup.symbolicSampleSize,
            numSamples: 64,
            numInputs: 1,
            numOutputs: 1,
            inputs: &mut input,
            outputs: &mut output,
            inputParameterChanges: ptr::null_mut(),
            outputParameterChanges: ptr::null_mut(),
            inputEvents: ptr::null_mut(),
            outputEvents: ptr::null_mut(),
            processContext: ptr::null_mut(),
        };
        ensure!(processor.process(&mut data) == kResultOk);
        ensure!(
            (left, right) == (source, source),
            "VST3 mono→stereo dry audio changed"
        );
        if let Some(voice) = voice {
            ensure!(processor.setProcessing(0) == kResultOk);
            ensure!(component.setActive(0) == kResultOk);
            let mut capture_position = SpeakerArr::kStereo;
            let mut render_position = SpeakerArr::kStereo;
            ensure!(
                processor.setBusArrangements(&mut capture_position, 1, &mut render_position, 1)
                    == kResultOk
            );
            setup.maxSamplesPerBlock = VOICE_BLOCK as i32;
            ensure!(processor.setupProcessing(&mut setup) == kResultOk);
            ensure!(
                controller.setParamNormalized(
                    BLOCK_MS as u32,
                    f64::from(PARAMETERS[BLOCK_MS].normalize(200.0)),
                ) == kResultOk
            );
            ensure!(
                controller.setParamNormalized(8, f64::from(PARAMETERS[8].normalize(600.0)))
                    == kResultOk
            );
            ensure!(controller.setParamNormalized(ENGINE as u32, 1.0) == kResultOk);
            ensure!(component.setActive(1) == kResultOk);
            ensure!(processor.setProcessing(1) == kResultOk);
            check_voice(voice, |input, dry_right, left, right| {
                let mut input_channels = [input.as_ptr().cast_mut(), dry_right.as_ptr().cast_mut()];
                let mut output_channels = [left.as_mut_ptr(), right.as_mut_ptr()];
                let mut capture = AudioBusBuffers {
                    numChannels: 2,
                    silenceFlags: 0,
                    __field0: AudioBusBuffers__type0 {
                        channelBuffers32: input_channels.as_mut_ptr(),
                    },
                };
                let mut render = AudioBusBuffers {
                    numChannels: 2,
                    silenceFlags: 0,
                    __field0: AudioBusBuffers__type0 {
                        channelBuffers32: output_channels.as_mut_ptr(),
                    },
                };
                let mut data = ProcessData {
                    numSamples: VOICE_BLOCK as i32,
                    inputs: &mut capture,
                    outputs: &mut render,
                    ..data
                };
                ensure!(processor.process(&mut data) == kResultOk);
                Ok(())
            })
            .context("VST3 native voice")?;
        }
        ensure!(processor.setProcessing(0) == kResultOk);
        ensure!(component.setActive(0) == kResultOk);
        ensure!(component.terminate() == kResultOk);
        drop(processor);
        drop(controller);
        drop(component);
        drop(factory);
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        ensure!(exit(), "VST3 module failed to deinitialize");
    }
    Ok(())
}

const VOICE_BLOCK: usize = 960;

fn check_voice(
    audio: &[f32],
    mut process: impl FnMut(&[f32], &[f32], &mut [f32], &mut [f32]) -> Result<()>,
) -> Result<()> {
    let mut input = [0.0; VOICE_BLOCK];
    let mut dry_right = [0.0; VOICE_BLOCK];
    let mut out_left = [0.0; VOICE_BLOCK];
    let mut out_right = [0.0; VOICE_BLOCK];
    let mut phase = 0;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        for (left, right) in input.iter_mut().zip(&mut dry_right) {
            *left = audio[phase];
            *right = 0.5 * audio[phase];
            phase = (phase + 1) % audio.len();
        }
        process(&input, &dry_right, &mut out_left, &mut out_right)?;
        ensure!(
            out_left
                .iter()
                .chain(&out_right)
                .all(|sample| sample.is_finite()),
            "found non-finite samples"
        );
        if out_left == out_right
            && out_left.iter().any(|sample| sample.abs() > 1e-4)
            && out_left
                .iter()
                .zip(&input)
                .any(|(wet, dry)| (wet - dry).abs() > 1e-3)
        {
            return Ok(());
        }
        ensure!(
            Instant::now() < deadline,
            "no non-silent wet audio within 30 seconds"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}
