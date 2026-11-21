//! Single-component VST3 adapter using RVC's native DAW backend.

use std::{
    ffi::{c_char, c_void},
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use ::vst3::{
    Class, ComPtr, ComRef, ComWrapper,
    Steinberg::{Vst::*, *},
    uid,
};

use crate::daw::{
    self, AudioSlot, BLOCK_MS, MAX_STATE_BYTES, ModelPaths, PARAMETER_COUNT, PARAMETERS,
    PluginControls, PluginWorker, ProcessorFactory, model_state, parse_model_state,
    render_channels,
};

pub use ::vst3::Steinberg::IPluginFactory as FactoryInterface;

#[cfg(any(target_os = "linux", target_os = "windows"))]
mod editor_view;

const NAME: &str = "RVC Realtime";
const VENDOR: &str = "RVC Project";
const CATEGORY: &str = "Audio Module Class";
const BYPASS: u32 = 65_536;
pub const COMPONENT_UID: TUID = uid(
    0xF2AEE70D,
    0x00DE4F4E,
    u32::from_be_bytes(*b"Rvcp"),
    u32::from_be_bytes(*b"Rvcr"),
);

pub type FactoryProvider = Box<dyn Fn() -> ProcessorFactory + Send + Sync + 'static>;

/// A GetPluginFactory export can return this pointer. The host owns a COM
/// reference. Each component receives a fresh native model factory.
pub fn create_factory(paths: ModelPaths, processors: FactoryProvider) -> *mut FactoryInterface {
    ComWrapper::new(Factory { paths, processors })
        .to_com_ptr::<IPluginFactory>()
        .map_or(ptr::null_mut(), ComPtr::into_raw)
}

struct Factory {
    paths: ModelPaths,
    processors: FactoryProvider,
}

impl Class for Factory {
    type Interfaces = (IPluginFactory2,);
}

impl IPluginFactoryTrait for Factory {
    unsafe fn getFactoryInfo(&self, info: *mut PFactoryInfo) -> tresult {
        let info_value = PFactoryInfo {
            vendor: narrow(VENDOR),
            url: narrow("https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"),
            email: narrow(""),
            flags: PFactoryInfo_::FactoryFlags_::kUnicode as i32,
        };
        // SAFETY: a non-null pointer is a host-allocated factory info record.
        unsafe { put(info, info_value) }
    }

    unsafe fn countClasses(&self) -> i32 {
        1
    }

    unsafe fn getClassInfo(&self, index: i32, info: *mut PClassInfo) -> tresult {
        if index != 0 {
            return kInvalidArgument;
        }
        // SAFETY: VST3 supplies a writable descriptor for this call.
        unsafe {
            put(
                info,
                PClassInfo {
                    cid: COMPONENT_UID,
                    cardinality: PClassInfo_::ClassCardinality_::kManyInstances as i32,
                    category: narrow(CATEGORY),
                    name: narrow(NAME),
                },
            )
        }
    }

    unsafe fn createInstance(
        &self,
        cid: FIDString,
        iid: FIDString,
        object: *mut *mut c_void,
    ) -> tresult {
        if cid.is_null() || iid.is_null() || object.is_null() {
            return kInvalidArgument;
        }
        // SAFETY: the two identifers are 16-byte VST3 TUIDs. The result pointer
        // is writable and set to null when the request is invalid.
        if unsafe { cid.cast::<TUID>().read_unaligned() } != COMPONENT_UID {
            unsafe { object.write(ptr::null_mut()) };
            return kInvalidArgument;
        }
        // SAFETY: createInstance owns this output slot for the call.
        unsafe { object.write(ptr::null_mut()) };
        let Ok((controls, audio, worker)) = daw::start(self.paths.clone(), (self.processors)())
        else {
            return kInternalError;
        };
        let instance = ComWrapper::new(Component {
            controls,
            audio: AudioSlot::new(audio),
            component_active: AtomicBool::new(false),
            processing: AtomicBool::new(false),
            handler: Arc::new(Mutex::new(None)),
            _worker: worker,
        });
        let Some(unknown) = instance.to_com_ptr::<FUnknown>() else {
            return kNoInterface;
        };
        let raw = unknown.as_ptr();
        // SAFETY: raw owns a live COM instance and the requested IID and
        // writable result pointer are valid for queryInterface.
        unsafe { ((*(*raw).vtbl).queryInterface)(raw, iid.cast_mut().cast(), object) }
    }
}

impl IPluginFactory2Trait for Factory {
    #[allow(clippy::unnecessary_cast)] // The generated enum is signed on Windows.
    unsafe fn getClassInfo2(&self, index: i32, info: *mut PClassInfo2) -> tresult {
        if index != 0 {
            return kInvalidArgument;
        }
        // SAFETY: VST3 supplies a writable descriptor for this call.
        unsafe {
            put(
                info,
                PClassInfo2 {
                    cid: COMPONENT_UID,
                    cardinality: PClassInfo_::ClassCardinality_::kManyInstances as i32,
                    category: narrow(CATEGORY),
                    name: narrow(NAME),
                    classFlags: ComponentFlags_::kSimpleModeSupported as u32,
                    subCategories: narrow("Fx"),
                    vendor: narrow(VENDOR),
                    version: narrow(env!("CARGO_PKG_VERSION")),
                    sdkVersion: narrow("VST 3.8.0"),
                },
            )
        }
    }
}

struct Component {
    controls: Arc<PluginControls>,
    audio: AudioSlot,
    component_active: AtomicBool,
    processing: AtomicBool,
    handler: Arc<Mutex<Option<ComPtr<IComponentHandler>>>>,
    _worker: PluginWorker,
}

impl Class for Component {
    type Interfaces = (IComponent, IAudioProcessor, IEditController);
}

impl Component {
    fn update_active(&self) {
        self.controls.set_active(
            self.component_active.load(Ordering::Relaxed)
                && self.processing.load(Ordering::Relaxed),
        );
    }

    fn channels(&self, direction: BusDirection) -> Option<usize> {
        let (input, output) = self.controls.layout();
        match direction {
            direction if direction == BusDirections_::kInput as BusDirection => Some(input),
            direction if direction == BusDirections_::kOutput as BusDirection => Some(output),
            _ => None,
        }
    }

    fn set_param(&self, id: ParamID, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        if id == BYPASS {
            self.controls.set_bypass(value > 0.5);
            true
        } else if id < PARAMETER_COUNT as u32 {
            self.controls.set_normalized(id as usize, value as f32);
            true
        } else {
            false
        }
    }

    fn latency_changed(&self, prior_latency: usize) {
        if self.controls.latency_frames() == prior_latency {
            return;
        }
        let handler = self
            .handler
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone();
        if let Some(handler) = handler {
            // SAFETY: the controller owns an add-ref'd host component handler.
            unsafe { handler.restartComponent(RestartFlags_::kLatencyChanged) };
        }
    }

    unsafe fn apply_automation(&self, changes: *mut IParameterChanges) {
        // SAFETY: the pointer is a borrowed interface during process().
        let Some(changes) = (unsafe { ComRef::from_raw(changes) }) else {
            return;
        };
        let count = unsafe { changes.getParameterCount() };
        // Each RVC control takes the last queued value, matching the legacy
        // blockwise automation and leaving neural inference off the host thread.
        for index in 0..count.clamp(0, PARAMETER_COUNT as i32 + 1) {
            // SAFETY: the queue belongs to the host for the duration of process().
            let Some(queue) = (unsafe { ComRef::from_raw(changes.getParameterData(index)) }) else {
                continue;
            };
            let last = unsafe { queue.getPointCount() } - 1;
            if last < 0 {
                continue;
            }
            let mut offset = 0;
            let mut normalized = 0.0;
            // SAFETY: last is the index of the final existing queue entry.
            if unsafe { queue.getPoint(last, &mut offset, &mut normalized) } == kResultOk {
                self.set_param(unsafe { queue.getParameterId() }, normalized);
            }
        }
    }
}

impl IPluginBaseTrait for Component {
    unsafe fn initialize(&self, _: *mut FUnknown) -> tresult {
        kResultOk
    }

    unsafe fn terminate(&self) -> tresult {
        self.component_active.store(false, Ordering::Relaxed);
        self.processing.store(false, Ordering::Relaxed);
        self.update_active();
        *self.handler.lock().unwrap_or_else(|err| err.into_inner()) = None;
        kResultOk
    }
}

impl IComponentTrait for Component {
    unsafe fn getControllerClassId(&self, _: *mut TUID) -> tresult {
        // This instance exposes IEditController directly.
        kResultFalse
    }

    unsafe fn setIoMode(&self, _: IoMode) -> tresult {
        kResultOk
    }

    unsafe fn getBusCount(&self, media: MediaType, direction: BusDirection) -> i32 {
        i32::from(media == MediaTypes_::kAudio as i32 && self.channels(direction).is_some())
    }

    #[allow(clippy::unnecessary_cast)] // The generated enum is signed on Windows.
    unsafe fn getBusInfo(
        &self,
        media: MediaType,
        direction: BusDirection,
        index: i32,
        info: *mut BusInfo,
    ) -> tresult {
        if media != MediaTypes_::kAudio as i32 || index != 0 {
            return kInvalidArgument;
        }
        let Some(channels) = self.channels(direction) else {
            return kInvalidArgument;
        };
        let name = if direction == BusDirections_::kInput as i32 {
            "Input"
        } else {
            "Output"
        };
        // SAFETY: the host supplied writable bus information.
        unsafe {
            put(
                info,
                BusInfo {
                    mediaType: media,
                    direction,
                    channelCount: channels as i32,
                    name: wide(name),
                    busType: BusTypes_::kMain as i32,
                    flags: BusInfo_::BusFlags_::kDefaultActive as u32,
                },
            )
        }
    }

    unsafe fn getRoutingInfo(&self, _: *mut RoutingInfo, _: *mut RoutingInfo) -> tresult {
        kNotImplemented
    }

    unsafe fn activateBus(
        &self,
        media: MediaType,
        direction: BusDirection,
        index: i32,
        _: TBool,
    ) -> tresult {
        if index == 0 && media == MediaTypes_::kAudio as i32 && self.channels(direction).is_some() {
            kResultOk
        } else {
            kInvalidArgument
        }
    }

    unsafe fn setActive(&self, value: TBool) -> tresult {
        self.component_active.store(value != 0, Ordering::Relaxed);
        self.update_active();
        kResultOk
    }

    unsafe fn setState(&self, state: *mut IBStream) -> tresult {
        // SAFETY: the host provides a live stream for the duration of this call.
        let Some(mut bytes) = (unsafe { read_state(state) }) else {
            return kInvalidArgument;
        };
        let Some(payload) = bytes.len().checked_sub(4) else {
            return kInvalidArgument;
        };
        let bypassed = bytes.split_off(payload) == 1i32.to_le_bytes();
        let Some((paths, params)) = parse_model_state(&bytes) else {
            return kInvalidArgument;
        };
        let prior = self.controls.latency_frames();
        self.controls.restore(paths, params);
        self.controls.set_bypass(bypassed);
        self.latency_changed(prior);
        kResultOk
    }

    unsafe fn getState(&self, state: *mut IBStream) -> tresult {
        let (paths, params) = self.controls.snapshot();
        let Some(mut bytes) = model_state(paths, params) else {
            return kInternalError;
        };
        bytes.extend(i32::from(self.controls.bypassed()).to_le_bytes());
        // SAFETY: state is the host's writable stream for the call.
        unsafe { write_state(state, &mut bytes) }
    }
}

impl IAudioProcessorTrait for Component {
    unsafe fn setBusArrangements(
        &self,
        inputs: *mut SpeakerArrangement,
        input_count: i32,
        outputs: *mut SpeakerArrangement,
        output_count: i32,
    ) -> tresult {
        if input_count != 1 || output_count != 1 || inputs.is_null() || outputs.is_null() {
            return kResultFalse;
        }
        fn channels(speakers: SpeakerArrangement) -> usize {
            match speakers {
                SpeakerArr::kMono => 1,
                SpeakerArr::kStereo => 2,
                _ => 0,
            }
        }
        // SAFETY: the supported request has one speaker descriptor per bus.
        if self
            .controls
            .set_layout(unsafe { channels(*inputs) }, unsafe { channels(*outputs) })
        {
            kResultOk
        } else {
            kResultFalse
        }
    }

    unsafe fn getBusArrangement(
        &self,
        direction: BusDirection,
        index: i32,
        arrangement: *mut SpeakerArrangement,
    ) -> tresult {
        if index != 0 {
            return kInvalidArgument;
        }
        let Some(channels) = self.channels(direction) else {
            return kInvalidArgument;
        };
        let value = if channels == 1 {
            SpeakerArr::kMono
        } else {
            SpeakerArr::kStereo
        };
        // SAFETY: non-null pointer is the caller's output descriptor.
        unsafe { put(arrangement, value) }
    }

    unsafe fn canProcessSampleSize(&self, size: i32) -> tresult {
        if sample_size_supported(size) {
            kResultOk
        } else {
            kResultFalse
        }
    }

    unsafe fn getLatencySamples(&self) -> u32 {
        self.controls.latency_frames() as u32
    }

    unsafe fn setupProcessing(&self, setup: *mut ProcessSetup) -> tresult {
        // SAFETY: setup is borrowed from the host during this call.
        let Some(setup) = (unsafe { setup.as_ref() }) else {
            return kInvalidArgument;
        };
        if !sample_size_supported(setup.symbolicSampleSize)
            || !(8_000.0..=384_000.0).contains(&setup.sampleRate)
            || setup.maxSamplesPerBlock <= 0
        {
            return kInvalidArgument;
        }
        self.controls
            .set_sample_rate(setup.sampleRate.round() as u32);
        kResultOk
    }

    unsafe fn setProcessing(&self, value: TBool) -> tresult {
        self.processing.store(value != 0, Ordering::Relaxed);
        self.update_active();
        kResultOk
    }

    unsafe fn process(&self, data: *mut ProcessData) -> tresult {
        // SAFETY: VST3 lends this process descriptor for the callback.
        let Some(data) = (unsafe { data.as_mut() }) else {
            return kInvalidArgument;
        };
        unsafe { self.apply_automation(data.inputParameterChanges) };
        if data.numSamples == 0 || data.numOutputs == 0 {
            return kResultOk;
        }
        if data.numSamples < 0
            || !(0..=1).contains(&data.numInputs)
            || data.numOutputs != 1
            || !sample_size_supported(data.symbolicSampleSize)
        {
            return kInvalidArgument;
        }
        // SAFETY: the valid topology uses at most one input and one output bus.
        let Some(out) = (unsafe { data.outputs.as_mut() }) else {
            return kInvalidArgument;
        };
        let input = if data.numInputs == 1 {
            unsafe { data.inputs.as_ref() }
        } else {
            None
        };
        let (inputs, outputs) = self.controls.layout();
        if out.numChannels != outputs as i32
            || input.is_some_and(|item| item.numChannels != inputs as i32)
        {
            return kInvalidArgument;
        }
        let frames = data.numSamples as usize;
        out.silenceFlags = 0;
        // SAFETY: symbolicSampleSize selects the active union fields. Channel
        // counts were validated against the currently negotiated bus layout.
        unsafe {
            if data.symbolicSampleSize == SymbolicSampleSizes_::kSample32 as i32 {
                let input = input.map_or(ptr::null(), |bus| {
                    bus.__field0
                        .channelBuffers32
                        .cast::<*const f32>()
                        .cast_const()
                });
                render_channels(
                    &self.audio,
                    (inputs, outputs),
                    input,
                    out.__field0.channelBuffers32,
                    frames,
                    false,
                );
            } else {
                let input = input.map_or(ptr::null(), |bus| {
                    bus.__field0
                        .channelBuffers64
                        .cast::<*const f64>()
                        .cast_const()
                });
                render_channels(
                    &self.audio,
                    (inputs, outputs),
                    input,
                    out.__field0.channelBuffers64,
                    frames,
                    false,
                );
            }
        }
        kResultOk
    }

    unsafe fn getTailSamples(&self) -> u32 {
        kNoTail
    }
}

impl IEditControllerTrait for Component {
    unsafe fn setComponentState(&self, state: *mut IBStream) -> tresult {
        // SAFETY: in a combined component this is the same processor state format.
        unsafe { IComponentTrait::setState(self, state) }
    }

    unsafe fn setState(&self, _: *mut IBStream) -> tresult {
        kResultOk
    }

    unsafe fn getState(&self, _: *mut IBStream) -> tresult {
        kResultOk
    }

    unsafe fn getParameterCount(&self) -> i32 {
        PARAMETER_COUNT as i32 + 1
    }

    unsafe fn getParameterInfo(&self, index: i32, info: *mut ParameterInfo) -> tresult {
        if index < 0 || index > PARAMETER_COUNT as i32 {
            return kInvalidArgument;
        }
        let parameter = if index == 0 {
            ParameterInfo {
                id: BYPASS,
                title: wide("Bypass"),
                shortTitle: wide("Bypass"),
                units: wide(""),
                stepCount: 1,
                defaultNormalizedValue: 0.0,
                unitId: kRootUnitId,
                flags: ParameterInfo_::ParameterFlags_::kCanAutomate
                    | ParameterInfo_::ParameterFlags_::kIsBypass,
            }
        } else {
            let id = index as usize - 1;
            let metadata = PARAMETERS[id];
            ParameterInfo {
                id: id as ParamID,
                title: wide(metadata.name),
                shortTitle: wide(metadata.name),
                units: wide(metadata.unit),
                stepCount: metadata.step_count(),
                defaultNormalizedValue: f64::from(metadata.normalize(metadata.default)),
                unitId: kRootUnitId,
                flags: ParameterInfo_::ParameterFlags_::kCanAutomate,
            }
        };
        // SAFETY: non-null parameter metadata pointer is a host output slot.
        unsafe { put(info, parameter) }
    }

    unsafe fn getParamStringByValue(
        &self,
        id: ParamID,
        normalized: ParamValue,
        output: *mut String128,
    ) -> tresult {
        if !normalized.is_finite() {
            return kInvalidArgument;
        }
        let text = if id == BYPASS {
            if normalized > 0.5 {
                "On".into()
            } else {
                "Off".into()
            }
        } else if id == daw::ENGINE as u32 {
            self.controls.engine_display(normalized >= 0.5).into()
        } else if let Some(spec) = PARAMETERS.get(id as usize) {
            spec.display(id as usize, spec.denormalize(normalized as f32))
        } else {
            return kInvalidArgument;
        };
        // SAFETY: VST3 expects a writable 128-code-unit string buffer.
        unsafe { put(output, wide(&text)) }
    }

    unsafe fn getParamValueByString(
        &self,
        id: ParamID,
        value: *mut TChar,
        normalized: *mut ParamValue,
    ) -> tresult {
        if value.is_null() || normalized.is_null() {
            return kInvalidArgument;
        }
        let mut count = 0;
        // SAFETY: VST3 provides a NUL-terminated string with up to 128 code units.
        unsafe {
            while count < 128 && value.add(count).read() != 0 {
                count += 1;
            }
        }
        let units = unsafe { std::slice::from_raw_parts(value, count) };
        let Ok(value) = String::from_utf16(units) else {
            return kInvalidArgument;
        };
        let value = value.trim();
        let number = if id == BYPASS || id == daw::ENGINE as u32 {
            match value.to_ascii_lowercase().as_str() {
                "on" | "ready" | "loading" | "error" | "1" => 1.0,
                "off" | "0" => 0.0,
                _ => return kInvalidArgument,
            }
        } else if id == 9 {
            let Some(&(plain, _)) = daw::PITCH_METHODS
                .iter()
                .find(|(_, name)| name.eq_ignore_ascii_case(value))
            else {
                return kInvalidArgument;
            };
            f64::from(PARAMETERS[9].normalize(plain))
        } else {
            let Some(spec) = PARAMETERS.get(id as usize) else {
                return kInvalidArgument;
            };
            let Ok(plain) = value.parse::<f32>() else {
                return kInvalidArgument;
            };
            if !plain.is_finite() {
                return kInvalidArgument;
            }
            f64::from(spec.normalize(plain))
        };
        // SAFETY: host requests writing the parsed normalized output here.
        unsafe { put(normalized, number) }
    }

    unsafe fn normalizedParamToPlain(&self, id: ParamID, value: ParamValue) -> ParamValue {
        if !value.is_finite() {
            return 0.0;
        }
        if id == BYPASS {
            return f64::from(value > 0.5);
        }
        PARAMETERS
            .get(id as usize)
            .map_or(0.0, |p| f64::from(p.denormalize(value as f32)))
    }

    unsafe fn plainParamToNormalized(&self, id: ParamID, value: ParamValue) -> ParamValue {
        if !value.is_finite() {
            return 0.0;
        }
        if id == BYPASS {
            return f64::from(value > 0.5);
        }
        PARAMETERS
            .get(id as usize)
            .map_or(0.0, |p| f64::from(p.normalize(value as f32)))
    }

    unsafe fn getParamNormalized(&self, id: ParamID) -> ParamValue {
        if id == BYPASS {
            return f64::from(self.controls.bypassed());
        }
        self.controls.get(id as usize).map_or(0.0, |value| {
            f64::from(PARAMETERS[id as usize].normalize(value))
        })
    }

    unsafe fn setParamNormalized(&self, id: ParamID, value: ParamValue) -> tresult {
        let prior = self.controls.latency_frames();
        if self.set_param(id, value) {
            if id == BLOCK_MS as u32 {
                self.latency_changed(prior);
            }
            kResultOk
        } else {
            kInvalidArgument
        }
    }

    unsafe fn setComponentHandler(&self, handler: *mut IComponentHandler) -> tresult {
        // SAFETY: the host's reference is borrowed until setComponentHandler
        // returns. We take our own reference for future latency notifications.
        *self.handler.lock().unwrap_or_else(|err| err.into_inner()) =
            unsafe { ComRef::from_raw(handler) }.map(|handler| handler.to_com_ptr());
        kResultOk
    }

    unsafe fn createView(&self, name: FIDString) -> *mut IPlugView {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if !name.is_null() {
            // SAFETY: createView receives a valid NUL-terminated VST3 view type.
            if unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes() == b"editor" {
                return editor_view::create(Arc::clone(&self.controls), Arc::clone(&self.handler));
            }
        }
        ptr::null_mut()
    }
}

fn narrow<const N: usize>(text: &str) -> [c_char; N] {
    let mut out = [0; N];
    for (destination, source) in out[..N - 1].iter_mut().zip(text.bytes()) {
        *destination = source as c_char;
    }
    out
}

fn wide(text: &str) -> String128 {
    let mut out = [0; 128];
    for (destination, source) in out[..127].iter_mut().zip(text.encode_utf16()) {
        *destination = source;
    }
    out
}

unsafe fn put<T>(target: *mut T, value: T) -> tresult {
    if target.is_null() {
        return kInvalidArgument;
    }
    // SAFETY: the caller declares the non-null output slot writable.
    unsafe { target.write(value) };
    kResultOk
}

fn sample_size_supported(size: i32) -> bool {
    size == SymbolicSampleSizes_::kSample32 as i32 || size == SymbolicSampleSizes_::kSample64 as i32
}

unsafe fn read_state(stream: *mut IBStream) -> Option<Vec<u8>> {
    // SAFETY: VST3 passes a live stream for the callback.
    let stream = unsafe { ComRef::from_raw(stream) }?;
    let mut result = Vec::new();
    let mut buffer = [0u8; 512];
    loop {
        let mut transferred = 0;
        // SAFETY: buffer is writable for the requested size, with a count slot.
        let status = unsafe {
            stream.read(
                buffer.as_mut_ptr().cast(),
                buffer.len() as i32,
                &mut transferred,
            )
        };
        let bytes = usize::try_from(transferred).ok()?;
        if bytes > buffer.len() || result.len().checked_add(bytes)? > MAX_STATE_BYTES + 4 {
            return None;
        }
        result.extend_from_slice(&buffer[..bytes]);
        if bytes == 0 {
            return Some(result);
        }
        if status != kResultOk {
            return Some(result);
        }
    }
}

unsafe fn write_state(stream: *mut IBStream, bytes: &mut [u8]) -> tresult {
    // SAFETY: VST3 passes a live stream for the callback.
    let Some(stream) = (unsafe { ComRef::from_raw(stream) }) else {
        return kInvalidArgument;
    };
    let mut written = 0;
    while written < bytes.len() {
        let mut transferred = 0;
        let remainder = &mut bytes[written..];
        // SAFETY: the stream can read this complete buffer.
        let status = unsafe {
            stream.write(
                remainder.as_mut_ptr().cast(),
                remainder.len() as i32,
                &mut transferred,
            )
        };
        if status != kResultOk || transferred <= 0 || transferred as usize > remainder.len() {
            return kResultFalse;
        }
        written += transferred as usize;
    }
    kResultOk
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Cursor, Read, Seek, SeekFrom, Write},
        mem::MaybeUninit,
        time::{Duration, Instant},
    };

    use ::vst3::Interface;
    use anyhow::{Context, Result, ensure};

    use super::*;

    #[test]
    fn existing_host_pitch_automation_and_new_midpoints_round_trip() {
        let spec = PARAMETERS[9];
        for (normalized, index) in [(0.0, 0), (0.5, 1), (1.0, 2), (0.25, 3), (0.75, 4)] {
            let value = spec.denormalize(normalized);
            assert_eq!(daw::pitch_method_index(value), index);
            assert_eq!(spec.normalize(daw::PITCH_METHODS[index].0), normalized);
        }
        assert_eq!(spec.step_count(), 4);
    }

    #[test]
    fn vst3_class_processes_a_real_worker_and_keeps_legacy_state() -> Result<()> {
        let paths = ModelPaths {
            model: "voice.pth".into(),
            ..Default::default()
        };
        let expected = paths.clone();
        let factory = create_factory(
            paths.clone(),
            Box::new(move || {
                let expected = expected.clone();
                Box::new(move |config| {
                    assert_eq!(config.paths, expected);
                    assert_eq!(config.format.block_frames, 960);
                    Ok(Box::new(|input, output, _, control| {
                        assert_eq!(control.pitch, 12);
                        for (source, destination) in input.iter().zip(output) {
                            *destination = 2.0 * source;
                        }
                        Ok(())
                    }))
                })
            }),
        );
        // SAFETY: this test owns the mock host buffers and COM references for
        // every callback; pointers remain valid through each call.
        unsafe {
            let factory = ComPtr::from_raw(factory).context("factory construction")?;
            assert_eq!(factory.countClasses(), 1);
            let mut info = MaybeUninit::<PClassInfo2>::uninit();
            let factory2 = factory.cast::<IPluginFactory2>().context("IFactory2")?;
            assert_eq!(factory2.getClassInfo2(0, info.as_mut_ptr()), kResultOk);
            assert_eq!(info.assume_init().cid, COMPONENT_UID);
            let mut object: *mut c_void = ptr::null_mut();
            assert_eq!(
                factory.createInstance(
                    COMPONENT_UID.as_ptr(),
                    IComponent::IID.as_ptr().cast(),
                    &mut object,
                ),
                kResultOk
            );
            let component = ComPtr::from_raw(object.cast::<IComponent>()).context("IComponent")?;
            let controller = component
                .cast::<IEditController>()
                .context("IEditController")?;
            let processor = component
                .cast::<IAudioProcessor>()
                .context("IAudioProcessor")?;
            assert_eq!(component.initialize(ptr::null_mut()), kResultOk);
            assert_eq!(controller.getParameterCount(), 13);
            let mut parameter = MaybeUninit::<ParameterInfo>::uninit();
            assert_eq!(
                controller.getParameterInfo(0, parameter.as_mut_ptr()),
                kResultOk
            );
            assert_eq!(parameter.assume_init().id, BYPASS);

            let mut mono = SpeakerArr::kMono;
            let mut stereo = SpeakerArr::kStereo;
            assert_eq!(
                processor.setBusArrangements(&mut mono, 1, &mut stereo, 1),
                kResultOk
            );
            let mut setup = ProcessSetup {
                processMode: ProcessModes_::kRealtime as i32,
                symbolicSampleSize: SymbolicSampleSizes_::kSample32 as i32,
                maxSamplesPerBlock: 240,
                sampleRate: 48_000.0,
            };
            assert_eq!(processor.setupProcessing(&mut setup), kResultOk);
            assert_eq!(processor.getLatencySamples(), 12_480);
            assert_eq!(
                controller.setParamNormalized(BLOCK_MS as ParamID, 0.0),
                kResultOk
            );
            assert_eq!(processor.getLatencySamples(), 1920);
            assert_eq!(
                controller.setParamNormalized(daw::ENGINE as ParamID, 1.0),
                kResultOk
            );
            assert_eq!(component.setActive(1), kResultOk);
            assert_eq!(processor.setProcessing(1), kResultOk);

            let mut source = [0.125_f32; 240];
            let mut input_channels = [source.as_mut_ptr()];
            let mut left = [0.0_f32; 240];
            let mut right = [0.0_f32; 240];
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
                numSamples: 240,
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
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                assert_eq!(processor.process(&mut data), kResultOk);
                if left.iter().all(|&x| x == 0.25) && right.iter().all(|&x| x == 0.25) {
                    break;
                }
                ensure!(
                    Instant::now() < deadline,
                    "native worker did not publish audio"
                );
                std::thread::sleep(Duration::from_millis(5));
            }

            assert_eq!(controller.setParamNormalized(BYPASS, 1.0), kResultOk);
            assert_eq!(processor.process(&mut data), kResultOk);
            assert_eq!((left, right), (source, source));

            let chunk = ComWrapper::new(MemoryStream::default());
            let stream = chunk.to_com_ptr::<IBStream>().context("IBStream")?;
            assert_eq!(
                IComponentTrait::getState(&component, stream.as_ptr()),
                kResultOk
            );
            let bytes = chunk.0.lock().unwrap().get_ref().clone();
            let (payload, bypass) = bytes.split_at(bytes.len() - 4);
            assert_eq!(bypass, &1i32.to_le_bytes());
            let (saved_paths, saved_params) = parse_model_state(payload).context("state layout")?;
            assert_eq!(saved_paths, paths);
            assert_eq!(saved_params[daw::ENGINE], 1.0);

            assert_eq!(controller.setParamNormalized(BYPASS, 0.0), kResultOk);
            assert_eq!(stream.seek(0, 0, ptr::null_mut()), kResultOk);
            assert_eq!(
                IComponentTrait::setState(&component, stream.as_ptr()),
                kResultOk
            );
            assert_eq!(controller.getParamNormalized(BYPASS), 1.0);
            assert_eq!(processor.setProcessing(0), kResultOk);
            assert_eq!(component.terminate(), kResultOk);
        }
        Ok(())
    }

    #[derive(Default)]
    struct MemoryStream(Mutex<Cursor<Vec<u8>>>);

    impl Class for MemoryStream {
        type Interfaces = (IBStream,);
    }

    impl IBStreamTrait for MemoryStream {
        unsafe fn read(&self, out: *mut c_void, size: i32, count: *mut i32) -> tresult {
            // SAFETY: the invocation's buffer is writable for size bytes.
            let output = unsafe { std::slice::from_raw_parts_mut(out.cast(), size as usize) };
            let read = self.0.lock().unwrap().read(output).unwrap();
            if !count.is_null() {
                unsafe { count.write(read as i32) };
            }
            kResultOk
        }

        unsafe fn write(&self, data: *mut c_void, size: i32, count: *mut i32) -> tresult {
            // SAFETY: the invocation's buffer is readable for size bytes.
            let input = unsafe { std::slice::from_raw_parts(data.cast(), size as usize) };
            self.0.lock().unwrap().write_all(input).unwrap();
            if !count.is_null() {
                unsafe { count.write(size) };
            }
            kResultOk
        }

        unsafe fn seek(&self, offset: i64, mode: i32, pos: *mut i64) -> tresult {
            let offset = match mode {
                0 => SeekFrom::Start(offset as u64),
                1 => SeekFrom::Current(offset),
                2 => SeekFrom::End(offset),
                _ => return kInvalidArgument,
            };
            let cursor = self.0.lock().unwrap().seek(offset).unwrap();
            if !pos.is_null() {
                unsafe { pos.write(cursor as i64) };
            }
            kResultOk
        }

        unsafe fn tell(&self, pos: *mut i64) -> tresult {
            unsafe { put(pos, self.0.lock().unwrap().position() as i64) }
        }
    }
}
