//! Native VST 2.4 adapter. Model construction is supplied by the application;
//! the ABI layer owns host callbacks and does not run any Python process.

use std::ffi::{CStr, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicI32, Ordering},
};

pub use vst::api::{AEffect, HostCallbackProc};
use vst::{
    api::{ChannelFlags, ChannelProperties, PluginFlags, SpeakerArrangementType, consts},
    host, plugin,
};

use crate::daw::{
    self, AudioSlot, BLOCK_MS, MAX_STATE_BYTES, ModelPaths, NativeSample, PARAMETER_COUNT,
    PARAMETERS, PluginControls, PluginWorker, ProcessorFactory, model_state, parse_model_state,
    render_channels, state_path,
};
#[cfg(any(target_os = "linux", target_os = "windows"))]
use crate::editor::{self, EditorAttachment};

const NAME: &str = "RVC Realtime";
const VENDOR: &str = "RVC Project";
pub const EFFECT_ID: i32 = i32::from_be_bytes(*b"Rvcr");
const VERSION: i32 = 10_000;
const API_VERSION: isize = 2400;
const PRESET_MAGIC: i32 = i32::from_be_bytes(*b"pfft");
const PRESET_VERSION: i32 = 0x010000;

struct Instance {
    host: HostCallbackProc,
    controls: Arc<PluginControls>,
    audio: AudioSlot,
    presets: Mutex<Presets>,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    editor: EditorAttachment,
    _worker: PluginWorker,
}

#[derive(Default)]
struct Presets {
    single: Vec<u8>,
    bank: Vec<u8>,
}

/// Construct the host-owned effect. Export a platform VSTPluginMain that
/// forwards to this with the real RVC factory. A disabled engine is dry.
pub fn create(
    host: HostCallbackProc,
    default_paths: ModelPaths,
    processor: ProcessorFactory,
) -> *mut AEffect {
    catch_unwind(AssertUnwindSafe(|| {
        if host(
            ptr::null_mut(),
            host::OpCode::Version as i32,
            0,
            0,
            ptr::null_mut(),
            0.0,
        ) == 0
        {
            return ptr::null_mut();
        }
        let Ok((controls, audio, worker)) = daw::start(default_paths, processor) else {
            return ptr::null_mut();
        };
        let flags = PluginFlags::CAN_REPLACING
            | PluginFlags::CAN_DOUBLE_REPLACING
            | PluginFlags::PROGRAM_CHUNKS;
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        let flags = flags | PluginFlags::HAS_EDITOR;
        let effect = AEffect {
            magic: consts::VST_MAGIC,
            dispatcher,
            _process: process_accumulating,
            setParameter: set_parameter,
            getParameter: get_parameter,
            numPrograms: 1,
            numParams: PARAMETER_COUNT as i32,
            numInputs: 2,
            numOutputs: 2,
            flags: flags.bits(),
            reserved1: 0,
            reserved2: 0,
            initialDelay: controls.latency_frames() as i32,
            _realQualities: 0,
            _offQualities: 0,
            _ioRatio: 1.0,
            object: Box::into_raw(Box::new(Instance {
                host,
                #[cfg(any(target_os = "linux", target_os = "windows"))]
                editor: EditorAttachment::new(Arc::clone(&controls)),
                controls,
                audio: AudioSlot::new(audio),
                presets: Mutex::new(Presets::default()),
                _worker: worker,
            }))
            .cast(),
            user: ptr::null_mut(),
            uniqueId: EFFECT_ID,
            version: VERSION,
            processReplacing: process_f32,
            processReplacingF64: process_f64,
            future: [0; 56],
        };
        Box::into_raw(Box::new(effect))
    }))
    .unwrap_or(ptr::null_mut())
}

unsafe fn instance<'a>(effect: *mut AEffect) -> Option<&'a Instance> {
    // SAFETY: callers are VST callbacks with a live effect pointer returned
    // by create. A null pointer is allowed for a rejected host call.
    if effect.is_null() {
        return None;
    }
    unsafe { (*effect).object.cast::<Instance>().as_ref() }
}

extern "C" fn set_parameter(effect: *mut AEffect, index: i32, normalized: f32) {
    // SAFETY: the host owns a live effect until it dispatches Shutdown.
    let Some(state) = (unsafe { instance(effect) }) else {
        return;
    };
    state.controls.set_normalized(index as usize, normalized);
    if index as usize == BLOCK_MS {
        update_latency(effect, state);
    }
}

extern "C" fn get_parameter(effect: *mut AEffect, index: i32) -> f32 {
    // SAFETY: the host owns a live effect until it dispatches Shutdown.
    let Some(state) = (unsafe { instance(effect) }) else {
        return 0.0;
    };
    state
        .controls
        .get(index as usize)
        .map_or(0.0, |value| PARAMETERS[index as usize].normalize(value))
}

fn update_latency(effect: *mut AEffect, state: &Instance) {
    notify_latency(effect, state.host, state.controls.latency_frames() as i32);
}

fn notify_latency(effect: *mut AEffect, host: HostCallbackProc, latency: i32) {
    // SAFETY: aligned i32 storage on a live AEffect. Parameter/control calls
    // serialize changes via atomic access, and the host is notified afterward.
    let delay = unsafe { AtomicI32::from_ptr(&raw mut (*effect).initialDelay) };
    if delay.swap(latency, Ordering::Relaxed) != latency {
        host(
            effect,
            host::OpCode::IOChanged as i32,
            0,
            0,
            ptr::null_mut(),
            0.0,
        );
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[repr(C)]
struct EditorRect {
    top: i16,
    left: i16,
    bottom: i16,
    right: i16,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
static EDITOR_RECT: EditorRect = EditorRect {
    top: 0,
    left: 0,
    bottom: editor::HEIGHT as i16,
    right: editor::WIDTH as i16,
};

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn open_editor(effect: *mut AEffect, state: &Instance, parent: *mut c_void) -> isize {
    if state.editor.is_open() {
        return 1;
    }
    #[cfg(target_os = "windows")]
    let sink = {
        let raw = effect as usize;
        let callback = state.host;
        let controls = Arc::clone(&state.controls);
        // Baseview dispatches on the Win32 host GUI thread. Closing the editor
        // synchronously destroys the window before the effect is released.
        let sink: editor::EditSink = Arc::new(move |edits| {
            notify_edits(raw as *mut AEffect, callback, &controls, edits);
        });
        Some(sink)
    };
    #[cfg(target_os = "linux")]
    let sink = None;

    match state
        .editor
        .open(parent, editor::WIDTH, editor::HEIGHT, sink)
    {
        Ok(()) => {
            (state.host)(
                effect,
                host::OpCode::_NeedIdle as i32,
                0,
                0,
                ptr::null_mut(),
                0.0,
            );
            1
        }
        Err(error) => {
            eprintln!("RVC Realtime: {error:#}");
            0
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn flush_editor(effect: *mut AEffect, state: &Instance) {
    let edits = state.editor.model().take_edits();
    notify_edits(effect, state.host, &state.controls, edits);
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn notify_edits(
    effect: *mut AEffect,
    callback: HostCallbackProc,
    controls: &PluginControls,
    edits: editor::EditBatch,
) {
    if edits[BLOCK_MS].is_some() {
        notify_latency(effect, callback, controls.latency_frames() as i32);
    }
    for (index, value) in edits.into_iter().enumerate() {
        let Some(value) = value else { continue };
        for opcode in [
            host::OpCode::BeginEdit,
            host::OpCode::Automate,
            host::OpCode::EndEdit,
        ] {
            callback(
                effect,
                opcode as i32,
                index as i32,
                0,
                ptr::null_mut(),
                value,
            );
        }
    }
}

extern "C" fn dispatcher(
    effect: *mut AEffect,
    opcode: i32,
    index: i32,
    value: isize,
    data: *mut c_void,
    float: f32,
) -> isize {
    catch_unwind(AssertUnwindSafe(|| {
        use plugin::OpCode::*;
        let Ok(opcode) = plugin::OpCode::try_from(opcode) else {
            return 0;
        };
        if effect.is_null() {
            return 0;
        }
        if matches!(opcode, Shutdown) {
            // SAFETY: Shutdown ends the host's ownership after processing is
            // stopped. Each allocation is uniquely recovered exactly once.
            unsafe {
                let effect = Box::from_raw(effect);
                drop(Box::from_raw(effect.object.cast::<Instance>()));
            }
            return 1;
        }
        // SAFETY: non-shutdown callbacks use the live instance allocation.
        let Some(state) = (unsafe { instance(effect) }) else {
            return 0;
        };

        match opcode {
            Initialize => 1,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            EditorGetRect if !data.is_null() => {
                // SAFETY: effEditGetRect receives a slot for a borrowed ERect pointer.
                unsafe {
                    data.cast::<*const EditorRect>()
                        .write_unaligned(&EDITOR_RECT)
                };
                1
            }
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            EditorOpen if !data.is_null() => open_editor(effect, state, data),
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            EditorIdle => {
                flush_editor(effect, state);
                1
            }
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            EditorClose => {
                state.editor.close();
                flush_editor(effect, state);
                1
            }
            ChangePreset if value == 0 => 1,
            GetCurrentPresetNum => 0,
            GetCurrentPresetName => copy_text(data, "Default", 24),
            SetSampleRate if float.is_finite() && float > 0.0 => {
                state.controls.set_sample_rate(float.round() as u32);
                update_latency(effect, state);
                1
            }
            SetBlockSize if value > 0 => 1,
            StateChanged => {
                state.controls.set_active(value != 0);
                1
            }
            StartProcess => 1,
            StopProcess => 1,
            GetCategory => plugin::Category::Effect as isize,
            GetApiVersion => API_VERSION,
            GetVendorVersion => VERSION as isize,
            GetVendorName => copy_text(data, VENDOR, 64),
            GetProductName => copy_text(data, NAME, 64),
            GetEffectName => copy_text(data, NAME, 32),
            SoftBypass => {
                state.controls.set_bypass(value != 0);
                1
            }
            CanBeAutomated => isize::from((0..PARAMETER_COUNT as i32).contains(&index)),
            GetParameterName | GetParameterLabel | GetParameterDisplay => {
                let Some(parameter) = PARAMETERS.get(index as usize) else {
                    return 0;
                };
                match opcode {
                    GetParameterName => copy_text(data, parameter.name, 8),
                    GetParameterLabel => copy_text(data, parameter.unit, 8),
                    GetParameterDisplay => {
                        let value = state
                            .controls
                            .get(index as usize)
                            .unwrap_or(parameter.default);
                        if index as usize == daw::ENGINE {
                            copy_text(data, state.controls.engine_display(value >= 0.5), 8)
                        } else {
                            copy_text(data, &parameter.display(index as usize, value), 8)
                        }
                    }
                    _ => 0,
                }
            }
            GetInputInfo | GetOutputInfo => {
                let (inputs, outputs) = state.controls.layout();
                let channels = if matches!(opcode, GetInputInfo) {
                    inputs
                } else {
                    outputs
                };
                get_pin(data, index as usize, channels)
            }
            SetSpeakerArrangement => set_arrangement(state, value, data),
            GetSpeakerArrangement => get_arrangement(state, value, data),
            GetData => get_preset(state, index == 0, data),
            SetData => {
                let accepted = set_preset(state, index == 0, value, data);
                if accepted {
                    update_latency(effect, state);
                }
                isize::from(accepted)
            }
            CanDo if !data.is_null() => {
                // SAFETY: CanDo carries a host-owned NUL-terminated string.
                let query = unsafe { CStr::from_ptr(data.cast()) };
                match query.to_bytes() {
                    b"plugAsChannelInsert"
                    | b"plugAsSend"
                    | b"mixDryWet"
                    | b"1in1out"
                    | b"1in2out"
                    | b"2in2out"
                    | b"bypass" => 1,
                    b"receiveVstMidiEvent" | b"sendVstMidiEvent" => -1,
                    _ => 0,
                }
            }
            GetTailSize => 1,
            _ => 0,
        }
    }))
    .unwrap_or(0)
}

fn copy_text(data: *mut c_void, text: &str, capacity: usize) -> isize {
    if data.is_null() {
        return 0;
    }
    let count = text.len().min(capacity - 1);
    let output = data.cast::<u8>();
    // SAFETY: the opcode's documented host output is a writable C buffer of
    // at least capacity bytes. The terminator is within the advertised bound.
    unsafe {
        ptr::copy_nonoverlapping(text.as_ptr(), output, count);
        output.add(count).write(0);
    }
    1
}

fn get_pin(data: *mut c_void, index: usize, channels: usize) -> isize {
    if data.is_null() || index >= channels {
        return 0;
    }
    let mut name = [0u8; consts::MAX_LABEL];
    let mut short = [0u8; consts::MAX_SHORT_LABEL];
    let label: &[u8] = if channels == 1 {
        b"Mono".as_slice()
    } else if index == 0 {
        b"Left"
    } else {
        b"Right"
    };
    name[..label.len()].copy_from_slice(label);
    short[..label.len()].copy_from_slice(label);
    let stereo = channels == 2;
    let flags = ChannelFlags::ACTIVE
        | if stereo && index == 0 {
            ChannelFlags::STEREO
        } else {
            ChannelFlags::empty()
        };
    let channel = ChannelProperties {
        name,
        flags: flags.bits(),
        arrangement_type: if stereo {
            SpeakerArrangementType::Stereo
        } else {
            SpeakerArrangementType::Mono
        },
        short_name: short,
        future: [0; 48],
    };
    // SAFETY: the host supplies a writable VstPinProperties for this opcode.
    unsafe { data.cast::<ChannelProperties>().write_unaligned(channel) };
    1
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Speaker {
    position: [f32; 4],
    name: [u8; 64],
    kind: i32,
    reserved: [u8; 28],
}

impl Speaker {
    const EMPTY: Self = Self {
        position: [0.0; 4],
        name: [0; 64],
        kind: 0,
        reserved: [0; 28],
    };
}

#[repr(C)]
struct Arrangement {
    kind: i32,
    channels: i32,
    speakers: [Speaker; 8],
}

static MONO: Arrangement = Arrangement {
    kind: SpeakerArrangementType::Mono as i32,
    channels: 1,
    speakers: [Speaker::EMPTY; 8],
};
static STEREO: Arrangement = Arrangement {
    kind: SpeakerArrangementType::Stereo as i32,
    channels: 2,
    speakers: [
        Speaker {
            kind: 1,
            ..Speaker::EMPTY
        },
        Speaker {
            kind: 2,
            ..Speaker::EMPTY
        },
        Speaker::EMPTY,
        Speaker::EMPTY,
        Speaker::EMPTY,
        Speaker::EMPTY,
        Speaker::EMPTY,
        Speaker::EMPTY,
    ],
};

fn set_arrangement(state: &Instance, input: isize, output: *mut c_void) -> isize {
    let (mut inputs, mut outputs) = state.controls.layout();
    let input = input as *const i32;
    let output = output.cast::<i32>();
    // SAFETY: provided pointers are VstSpeakerArrangement records from the
    // host. Only the fixed-size channel-count field is read, even for mono.
    unsafe {
        if !input.is_null() {
            inputs = input.add(1).read_unaligned() as usize;
        }
        if !output.is_null() {
            outputs = output.add(1).read_unaligned() as usize;
        }
    }
    isize::from(state.controls.set_layout(inputs, outputs))
}

fn get_arrangement(state: &Instance, input: isize, output: *mut c_void) -> isize {
    let (inputs, outputs) = state.controls.layout();
    let input = input as *mut *const Arrangement;
    let output = output.cast::<*const Arrangement>();
    // SAFETY: non-null arguments point to host slots for borrowed layout
    // pointers. Our mono/stereo descriptors are immutable and static.
    unsafe {
        if !input.is_null() {
            input.write_unaligned(if inputs == 1 { &MONO } else { &STEREO });
        }
        if !output.is_null() {
            output.write_unaligned(if outputs == 1 { &MONO } else { &STEREO });
        }
    }
    1
}

fn get_preset(state: &Instance, bank: bool, data: *mut c_void) -> isize {
    if data.is_null() {
        return 0;
    }
    let mut presets = state.presets.lock().unwrap_or_else(|err| err.into_inner());
    let destination = if bank {
        &mut presets.bank
    } else {
        &mut presets.single
    };
    let (paths, values) = state.controls.snapshot();
    let Some(chunk) = encode_preset(paths, values, bank) else {
        return 0;
    };
    *destination = chunk;
    // SAFETY: the host's GetChunk pointer location is writable. The chunk
    // allocation stays alive until the next GetChunk of this kind or Shutdown.
    unsafe {
        data.cast::<*const u8>()
            .write_unaligned(destination.as_ptr())
    };
    destination.len() as isize
}

fn set_preset(state: &Instance, bank: bool, len: isize, data: *mut c_void) -> bool {
    if data.is_null() || len <= 0 || len as usize > MAX_STATE_BYTES {
        return false;
    }
    // SAFETY: the host owns the read-only chunk of the provided size.
    let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), len as usize) };
    let Some((paths, params)) = decode_preset(bytes, bank) else {
        return false;
    };
    state.controls.restore(paths, params);
    true
}

fn encode_preset(
    paths: ModelPaths,
    parameters: [f32; PARAMETER_COUNT],
    bank: bool,
) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    out.extend(PRESET_MAGIC.to_le_bytes());
    out.extend(PRESET_VERSION.to_le_bytes());
    if bank {
        out.extend(7i32.to_le_bytes());
        out.extend(b"Default");
        out.push(1);
    }
    out.extend(model_state(paths, parameters)?);
    (out.len() <= MAX_STATE_BYTES).then_some(out)
}

fn decode_preset(mut data: &[u8], bank: bool) -> Option<(ModelPaths, [f32; PARAMETER_COUNT])> {
    if data.first_chunk::<4>() == Some(&PRESET_MAGIC.to_le_bytes()) {
        data = data.get(4..)?;
        let version = i32::from_le_bytes(data.split_off(..4)?.try_into().ok()?);
        if bank && version >= PRESET_VERSION {
            let _name = state_path(&mut data)?;
            if *data.split_off_first()? != 1 {
                return None;
            }
        }
    }
    parse_model_state(data)
}

extern "C" fn process_f32(
    effect: *mut AEffect,
    input: *const *const f32,
    output: *mut *mut f32,
    frames: i32,
) {
    process(effect, input, output, frames, false);
}

extern "C" fn process_f64(
    effect: *mut AEffect,
    input: *const *const f64,
    output: *mut *mut f64,
    frames: i32,
) {
    process(effect, input, output, frames, false);
}

extern "C" fn process_accumulating(
    effect: *mut AEffect,
    input: *const *const f32,
    output: *mut *mut f32,
    frames: i32,
) {
    process(effect, input, output, frames, true);
}

fn process<T: NativeSample>(
    effect: *mut AEffect,
    input: *const *const T,
    output: *mut *mut T,
    frames: i32,
    accumulate: bool,
) {
    if frames <= 0 || output.is_null() {
        return;
    }
    // SAFETY: each process callback receives a live plugin instance.
    let Some(state) = (unsafe { instance(effect) }) else {
        return;
    };
    let (inputs, outputs) = state.controls.layout();
    // SAFETY: VST provides valid channel pointers matching the current layout.
    unsafe {
        render_channels(
            &state.audio,
            (inputs, outputs),
            input,
            output,
            frames as usize,
            accumulate,
        )
    };
}

#[cfg(test)]
mod tests {
    use std::{
        sync::mpsc,
        time::{Duration, Instant},
    };

    use anyhow::{Result, ensure};

    use super::*;

    extern "C" fn host(
        _: *mut AEffect,
        opcode: i32,
        _: i32,
        _: isize,
        _: *mut c_void,
        _: f32,
    ) -> isize {
        if opcode == host::OpCode::Version as i32 {
            API_VERSION
        } else {
            0
        }
    }

    struct TestEffect(*mut AEffect);

    impl TestEffect {
        fn new(paths: ModelPaths, factory: ProcessorFactory) -> Self {
            let effect = create(host, paths, factory);
            assert!(!effect.is_null());
            Self(effect)
        }

        fn effect(&self) -> &AEffect {
            // SAFETY: TestEffect owns a live VST host instance.
            unsafe { &*self.0 }
        }

        fn dispatch(
            &self,
            opcode: plugin::OpCode,
            index: i32,
            value: isize,
            data: *mut c_void,
            float: f32,
        ) -> isize {
            (self.effect().dispatcher)(self.0, opcode as i32, index, value, data, float)
        }

        fn set(&self, index: usize, value: f32) {
            (self.effect().setParameter)(self.0, index as i32, PARAMETERS[index].normalize(value));
        }
    }

    impl Drop for TestEffect {
        fn drop(&mut self) {
            self.dispatch(plugin::OpCode::Shutdown, 0, 0, ptr::null_mut(), 0.0);
        }
    }

    #[test]
    fn vst2_off_preserves_stereo_and_reports_live_latency_and_bus_layout() {
        let instance = TestEffect::new(
            ModelPaths::default(),
            Box::new(|_| anyhow::bail!("engine should be disabled")),
        );
        let effect = instance.effect();
        assert_eq!(
            (effect.magic, effect.uniqueId),
            (consts::VST_MAGIC, EFFECT_ID)
        );
        assert_eq!(
            (effect.numInputs, effect.numOutputs, effect.numParams),
            (2, 2, 12)
        );
        assert_eq!(effect.initialDelay, 12_480);
        assert_eq!(
            instance.dispatch(plugin::OpCode::GetApiVersion, 0, 0, ptr::null_mut(), 0.0),
            2400
        );
        instance.dispatch(
            plugin::OpCode::SetSampleRate,
            0,
            0,
            ptr::null_mut(),
            44_100.0,
        );
        assert_eq!(instance.effect().initialDelay, 11_466);
        instance.set(BLOCK_MS, 100.0);
        assert_eq!(instance.effect().initialDelay, 8_820);

        let source_l = [0.1234567890123_f64, -0.25];
        let source_r = [0.333333333333333, 0.37];
        let inputs = [source_l.as_ptr(), source_r.as_ptr()];
        let mut out_l = [0.0; 2];
        let mut out_r = [0.0; 2];
        let mut outputs = [out_l.as_mut_ptr(), out_r.as_mut_ptr()];
        (instance.effect().processReplacingF64)(
            instance.0,
            inputs.as_ptr(),
            outputs.as_mut_ptr(),
            2,
        );
        assert_eq!((out_l, out_r), (source_l, source_r));

        assert_eq!(
            instance.dispatch(
                plugin::OpCode::SetSpeakerArrangement,
                0,
                (&raw const MONO) as isize,
                (&raw const STEREO).cast_mut().cast(),
                0.0,
            ),
            1
        );
        let mut reported_input: *const Arrangement = ptr::null();
        let mut reported_output: *const Arrangement = ptr::null();
        instance.dispatch(
            plugin::OpCode::GetSpeakerArrangement,
            0,
            (&raw mut reported_input) as isize,
            (&raw mut reported_output).cast(),
            0.0,
        );
        assert_eq!(
            (reported_input, reported_output),
            (&raw const MONO, &raw const STEREO)
        );
        (instance.effect().processReplacingF64)(
            instance.0,
            inputs.as_ptr(),
            outputs.as_mut_ptr(),
            2,
        );
        assert_eq!((out_l, out_r), (source_l, source_l));
    }

    #[test]
    fn vst2_engine_uses_the_supplied_worker_with_current_controls() -> Result<()> {
        let (controls, monitor) = mpsc::channel();
        let paths = ModelPaths {
            model: "voice.pth".into(),
            ..Default::default()
        };
        let instance = TestEffect::new(
            paths.clone(),
            Box::new(move |spec| {
                assert_eq!(spec.paths, paths);
                assert_eq!(spec.format.block_frames, 960);
                let controls = controls.clone();
                Ok(Box::new(move |input, output, gap, settings| {
                    controls.send((settings.pitch, gap, input.len(), input[0]))?;
                    for (sample, source) in output.iter_mut().zip(input) {
                        *sample = -*source * 2.0;
                    }
                    Ok(())
                }))
            }),
        );
        instance.set(BLOCK_MS, 20.0);
        instance.set(1, -7.0);
        instance.set(0, 1.0);
        instance.dispatch(plugin::OpCode::StateChanged, 0, 1, ptr::null_mut(), 0.0);

        let source_l = [0.125_f32; 240];
        let source_r = [0.25_f32; 240];
        let inputs = [source_l.as_ptr(), source_r.as_ptr()];
        let mut out_l = [0.0; 240];
        let mut out_r = [0.0; 240];
        let mut outputs = [out_l.as_mut_ptr(), out_r.as_mut_ptr()];
        let deadline = Instant::now() + Duration::from_secs(2);
        let wet = -0.375;
        loop {
            (instance.effect().processReplacing)(
                instance.0,
                inputs.as_ptr(),
                outputs.as_mut_ptr(),
                240,
            );
            if out_l.iter().all(|&s| s == wet) && out_r.iter().all(|&s| s == wet) {
                break;
            }
            ensure!(
                Instant::now() < deadline,
                "DAW never received synthesized fixture output"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let warmup = (-7, true, 960, 0.0);
        assert_eq!(monitor.try_recv()?, warmup);
        // Host notifications can supersede a warmed engine before it receives live audio.
        let first_live = monitor.try_iter().find(|call| *call != warmup);
        assert_eq!(first_live, Some((-7, true, 960, 0.1875)));
        instance.dispatch(plugin::OpCode::SoftBypass, 0, 1, ptr::null_mut(), 0.0);
        (instance.effect().processReplacing)(
            instance.0,
            inputs.as_ptr(),
            outputs.as_mut_ptr(),
            240,
        );
        assert_eq!((out_l, out_r), (source_l, source_r));
        Ok(())
    }

    #[test]
    fn vst2_preset_and_bank_chunks_round_trip_with_legacy_layout() {
        let paths = ModelPaths {
            model: "голос.pth".into(),
            index: "added_voice.index".into(),
            rvc_root: "/tmp/rvc".into(),
            legacy_python: "runtime\\python.exe".into(),
        };
        let instance = TestEffect::new(
            paths.clone(),
            Box::new(|_| anyhow::bail!("engine should be disabled")),
        );
        let defaults = PARAMETERS.map(|spec| spec.default);
        for bank in [false, true] {
            let mut chunk = ptr::null::<u8>();
            let index = i32::from(!bank);
            let size = instance.dispatch(
                plugin::OpCode::GetData,
                index,
                0,
                (&raw mut chunk).cast(),
                0.0,
            );
            assert!(size > 0);
            // SAFETY: take an owned copy while the live plugin owns the chunk.
            let state = unsafe { std::slice::from_raw_parts(chunk, size as usize) }.to_vec();
            assert_eq!(decode_preset(&state, bank), Some((paths.clone(), defaults)));
            assert!(decode_preset(&state[..state.len() - 1], bank).is_none());
            instance.set(1, -10.0);
            assert_eq!(
                instance.dispatch(
                    plugin::OpCode::SetData,
                    index,
                    size,
                    state.as_ptr().cast_mut().cast(),
                    0.0
                ),
                1
            );
            assert_eq!((instance.effect().getParameter)(instance.0, 1), 0.75);
        }
    }
}
