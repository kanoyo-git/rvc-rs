//! Event-driven WASAPI exclusive capture and rendering. Windows owns the
//! device periods; RVC's worker still consumes fixed model blocks.

use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::ptr;
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use windows::Win32::Foundation::{
    E_INVALIDARG, HANDLE, S_OK, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows::Win32::Media::Audio::{
    AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY, AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_E_BUFFER_ERROR,
    AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED, AUDCLNT_E_DEVICE_IN_USE,
    AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED, AUDCLNT_E_UNSUPPORTED_FORMAT,
    AUDCLNT_SHAREMODE_EXCLUSIVE, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, EDataFlow, IAudioCaptureClient,
    IAudioClient, IAudioRenderClient, IMMDevice, IMMDeviceEnumerator, IMMEndpoint,
    MMDeviceEnumerator, WAVE_FORMAT_PCM, WAVEFORMATEX, WAVEFORMATEXTENSIBLE,
    WAVEFORMATEXTENSIBLE_0, eCapture, eConsole, eRender,
};
use windows::Win32::Media::KernelStreaming::{
    KSDATAFORMAT_SUBTYPE_PCM, SPEAKER_FRONT_CENTER, SPEAKER_FRONT_LEFT, SPEAKER_FRONT_RIGHT,
    WAVE_FORMAT_EXTENSIBLE,
};
use windows::Win32::Media::Multimedia::{KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT};
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
    CoUninitialize,
};
use windows::Win32::System::Threading::{
    AvRevertMmThreadCharacteristics, AvSetMmThreadCharacteristicsW, CreateEventW, SetEvent,
    WaitForMultipleObjects,
};
use windows::core::{Error as WinError, HSTRING, Interface, PCWSTR, w};

use super::wasapi_pcm::Pcm;
use super::{
    AudioWorker, Capture, DeviceOptions, Playback, RealtimeStream, StreamFormat, StreamStatus,
    Streams,
};

const DEVICE_TIMEOUT: u32 = 2_000;
const REFERENCE_TIME_SECOND: i64 = 10_000_000;

pub(super) struct ExclusiveStream {
    stop: Arc<Event>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for ExclusiveStream {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            // SAFETY: the event remains live until the device thread joins.
            let _ = unsafe { SetEvent(self.stop.handle()) };
            let _ = thread.join();
        }
    }
}

pub(super) fn start<F>(
    options: &DeviceOptions,
    input_id: Option<&str>,
    output_id: Option<&str>,
    make_processor: impl FnOnce(StreamFormat) -> Result<F>,
) -> Result<RealtimeStream>
where
    F: FnMut(&[f32], &mut [f32], bool) -> Result<()> + Send + 'static,
{
    let stop = Arc::new(Event::new(true)?);
    let status = Arc::new(StreamStatus::default());
    let request = Request {
        input: input_id.map(str::to_owned),
        output: output_id.map(str::to_owned),
        rate: options.sample_rate,
        block_ms: options.block_ms,
    };
    let (format_tx, format_rx) = mpsc::channel();
    let (endpoints_tx, endpoints_rx) = mpsc::channel();
    let (started_tx, started_rx) = mpsc::channel();
    let thread_stop = Arc::clone(&stop);
    let thread_status = Arc::clone(&status);
    let thread = thread::Builder::new()
        .name("rvc-wasapi-exclusive".into())
        .spawn(move || {
            let mut device = match Prepared::open(&request) {
                Ok(device) => device,
                Err(error) => {
                    let _ = format_tx.send(Err(error));
                    return;
                }
            };
            if format_tx.send(Ok(device.format)).is_err() {
                return;
            }
            let Ok((capture, playback)) = endpoints_rx.recv() else {
                return;
            };
            let started = device.endpoints.start();
            let ready = started.is_ok();
            if started_tx.send(started).is_err() || !ready {
                return;
            }
            let _priority = AudioPriority::try_set();
            if let Err(error) =
                device
                    .endpoints
                    .run(capture, playback, &thread_stop, &thread_status)
            {
                thread_status.record_error(error);
            }
        })
        .context("start WASAPI exclusive device thread")?;
    let stream = ExclusiveStream {
        stop,
        thread: Some(thread),
    };
    // Own the sender inside this scope so initialization errors release the
    // device thread's receiver before the stream destructor joins it.
    let connect = move || {
        let format = format_rx
            .recv()
            .context("WASAPI exclusive device thread stopped during format negotiation")??;
        let (worker, capture, playback) =
            AudioWorker::start_ready(format.block_frames, make_processor(format)?)?;
        endpoints_tx
            .send((capture, playback))
            .map_err(|_| anyhow!("WASAPI exclusive device thread stopped before starting"))?;
        started_rx
            .recv()
            .context("WASAPI exclusive device thread stopped during startup")??;
        Ok::<_, anyhow::Error>((format, worker))
    };
    let (format, worker) = connect()?;
    Ok(RealtimeStream {
        devices: Streams::Exclusive { _stream: stream },
        worker,
        status,
        format,
    })
}

struct Request {
    input: Option<String>,
    output: Option<String>,
    rate: Option<u32>,
    block_ms: u32,
}

struct Prepared {
    endpoints: Endpoints,
    format: StreamFormat,
    _apartment: Apartment,
}

impl Prepared {
    fn open(request: &Request) -> Result<Self> {
        let apartment = Apartment::new()?;
        // SAFETY: all COM audio objects are activated and used on this thread.
        let enumerator: IMMDeviceEnumerator =
            unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
                .context("open WASAPI device enumerator")?;
        let input = endpoint(&enumerator, request.input.as_deref(), eCapture)
            .context("select WASAPI exclusive input")?;
        let output = endpoint(&enumerator, request.output.as_deref(), eRender)
            .context("select WASAPI exclusive output")?;
        let input_client = activate(&input).context("activate WASAPI input")?;
        let output_client = activate(&output).context("activate WASAPI output")?;
        let (input_format, output_format) = negotiate(&input_client, &output_client, request.rate)?;
        let format = StreamFormat::new(input_format.rate(), request.block_ms)?;

        let input = AudioEndpoint::new(&input, input_client, input_format)
            .context("open WASAPI exclusive input stream")?;
        let output = AudioEndpoint::new(&output, output_client, output_format)
            .context("open WASAPI exclusive output stream")?;
        // SAFETY: both clients have successfully initialized their endpoints.
        let capture = unsafe { input.client.GetService::<IAudioCaptureClient>() }
            .context("get WASAPI capture service")?;
        let render = unsafe { output.client.GetService::<IAudioRenderClient>() }
            .context("get WASAPI rendering service")?;
        Ok(Self {
            endpoints: Endpoints {
                capture,
                render,
                input,
                output,
            },
            format,
            _apartment: apartment,
        })
    }
}

struct Apartment;

impl Apartment {
    fn new() -> Result<Self> {
        // SAFETY: the device thread has no prior COM apartment or COM objects.
        unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED).ok() }
            .context("initialize WASAPI COM apartment")?;
        Ok(Self)
    }
}

impl Drop for Apartment {
    fn drop(&mut self) {
        // SAFETY: all audio interfaces have been released on this same thread.
        unsafe { CoUninitialize() };
    }
}

struct Event(OwnedHandle);

impl Event {
    fn new(manual_reset: bool) -> Result<Self> {
        // SAFETY: CreateEventW returns a new owned handle.
        let handle = unsafe { CreateEventW(None, manual_reset, false, PCWSTR::null()) }
            .context("create WASAPI event")?;
        Ok(Self(unsafe { OwnedHandle::from_raw_handle(handle.0) }))
    }

    fn handle(&self) -> HANDLE {
        HANDLE(self.0.as_raw_handle())
    }
}

struct AudioPriority(HANDLE);

impl AudioPriority {
    fn try_set() -> Option<Self> {
        let mut task = 0;
        // SAFETY: MMCSS registers the calling audio thread until reverted.
        unsafe { AvSetMmThreadCharacteristicsW(w!("Pro Audio"), &mut task) }
            .ok()
            .map(Self)
    }
}

impl Drop for AudioPriority {
    fn drop(&mut self) {
        // SAFETY: this registration belongs to the current thread.
        let _ = unsafe { AvRevertMmThreadCharacteristics(self.0) };
    }
}

fn endpoint(
    enumerator: &IMMDeviceEnumerator,
    id: Option<&str>,
    flow: EDataFlow,
) -> Result<IMMDevice> {
    // SAFETY: the optional wide string remains alive through GetDevice.
    let device = unsafe {
        match id {
            Some(id) => enumerator.GetDevice(&HSTRING::from(id)),
            None => enumerator.GetDefaultAudioEndpoint(flow, eConsole),
        }
    }?;
    let endpoint = device.cast::<IMMEndpoint>()?;
    // SAFETY: endpoint retains a valid reference to the selected device.
    ensure!(
        unsafe { endpoint.GetDataFlow() }? == flow,
        "selected audio endpoint has the wrong direction"
    );
    Ok(device)
}

fn activate(endpoint: &IMMDevice) -> Result<IAudioClient> {
    // SAFETY: the live endpoint belongs to the device thread's COM apartment.
    Ok(unsafe { endpoint.Activate::<IAudioClient>(CLSCTX_ALL, None) }?)
}

#[derive(Clone, Copy)]
struct NativeFormat {
    wave: WAVEFORMATEXTENSIBLE,
    pcm: Pcm,
}

impl NativeFormat {
    fn new(rate: u32, channels: u16, mask: u32, pcm: Pcm, extended: bool) -> Option<Self> {
        let frame_bytes = channels.checked_mul(pcm.bits() / 8)?;
        let tag = match (extended, pcm == Pcm::Float32) {
            (true, _) => WAVE_FORMAT_EXTENSIBLE,
            (false, true) => WAVE_FORMAT_IEEE_FLOAT,
            (false, false) => WAVE_FORMAT_PCM,
        };
        Some(Self {
            wave: WAVEFORMATEXTENSIBLE {
                Format: WAVEFORMATEX {
                    wFormatTag: tag as u16,
                    nChannels: channels,
                    nSamplesPerSec: rate,
                    nAvgBytesPerSec: rate.checked_mul(u32::from(frame_bytes))?,
                    nBlockAlign: frame_bytes,
                    wBitsPerSample: pcm.bits(),
                    cbSize: if extended { WAVE_EXTENSION_BYTES } else { 0 },
                },
                Samples: WAVEFORMATEXTENSIBLE_0 {
                    wValidBitsPerSample: pcm.valid_bits(),
                },
                dwChannelMask: mask,
                SubFormat: if pcm == Pcm::Float32 {
                    KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
                } else {
                    KSDATAFORMAT_SUBTYPE_PCM
                },
            },
            pcm,
        })
    }

    fn rate(self) -> u32 {
        self.wave.Format.nSamplesPerSec
    }

    fn frame_bytes(self) -> usize {
        usize::from(self.wave.Format.nBlockAlign)
    }

    fn as_ptr(&self) -> *const WAVEFORMATEX {
        &raw const self.wave.Format
    }
}

const WAVE_EXTENSION_BYTES: u16 =
    (size_of::<WAVEFORMATEXTENSIBLE>() - size_of::<WAVEFORMATEX>()) as u16;

struct MixFormat {
    rate: u32,
    channels: u16,
    mask: u32,
}

impl MixFormat {
    fn read(client: &IAudioClient) -> Result<Self> {
        // SAFETY: WASAPI specifies a COM allocation containing at least WAVEFORMATEX.
        let pointer = unsafe { client.GetMixFormat() }?;
        ensure!(!pointer.is_null(), "WASAPI returned an empty mix format");
        let wave = unsafe { pointer.read_unaligned() };
        let mask = if u32::from(wave.wFormatTag) == WAVE_FORMAT_EXTENSIBLE
            && wave.cbSize >= WAVE_EXTENSION_BYTES
        {
            unsafe { pointer.cast::<WAVEFORMATEXTENSIBLE>().read_unaligned() }.dwChannelMask
        } else {
            channel_mask(wave.nChannels)
        };
        // SAFETY: GetMixFormat transfers the returned allocation to the caller.
        unsafe { CoTaskMemFree(Some(pointer.cast())) };
        ensure!(wave.nChannels > 0, "WASAPI mix format has no channels");
        Ok(Self {
            rate: wave.nSamplesPerSec,
            channels: wave.nChannels,
            mask,
        })
    }

    fn exclusive_at(&self, client: &IAudioClient, rate: u32) -> Result<Option<NativeFormat>> {
        const FORMATS: [Pcm; 8] = [
            Pcm::Float32,
            Pcm::Integer {
                bits: 32,
                valid: 24,
            },
            Pcm::Integer {
                bits: 24,
                valid: 24,
            },
            Pcm::Integer {
                bits: 32,
                valid: 32,
            },
            Pcm::Integer {
                bits: 16,
                valid: 16,
            },
            Pcm::Integer {
                bits: 24,
                valid: 20,
            },
            Pcm::Integer {
                bits: 32,
                valid: 20,
            },
            Pcm::Integer { bits: 8, valid: 8 },
        ];
        let layouts = [
            (self.channels, self.mask),
            (2, channel_mask(2)),
            (1, channel_mask(1)),
        ];
        for (index, &(channels, mask)) in layouts.iter().enumerate() {
            if layouts[..index].contains(&(channels, mask)) {
                continue;
            }
            for pcm in FORMATS {
                for extended in [true, false] {
                    if !extended && (channels > 2 || pcm.valid_bits() != pcm.bits()) {
                        continue;
                    }
                    let Some(format) = NativeFormat::new(rate, channels, mask, pcm, extended)
                    else {
                        continue;
                    };
                    // SAFETY: format points to a complete descriptor. Exclusive
                    // probing requires a null closest-match output pointer.
                    let result = unsafe {
                        client.IsFormatSupported(AUDCLNT_SHAREMODE_EXCLUSIVE, format.as_ptr(), None)
                    };
                    if result == S_OK {
                        return Ok(Some(format));
                    }
                    if result != AUDCLNT_E_UNSUPPORTED_FORMAT && result != E_INVALIDARG {
                        result
                            .ok()
                            .map_err(exclusive_error)
                            .context("probe WASAPI exclusive PCM format")?;
                    }
                }
            }
        }
        Ok(None)
    }
}

fn channel_mask(channels: u16) -> u32 {
    match channels {
        1 => SPEAKER_FRONT_CENTER,
        2 => SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT,
        _ => 0,
    }
}

fn negotiate(
    input: &IAudioClient,
    output: &IAudioClient,
    requested: Option<u32>,
) -> Result<(NativeFormat, NativeFormat)> {
    let left = MixFormat::read(input).context("read WASAPI input mix format")?;
    let right = MixFormat::read(output).context("read WASAPI output mix format")?;
    let defaults = [
        left.rate, right.rate, 48_000, 44_100, 32_000, 40_000, 96_000, 88_200, 192_000, 176_400,
        384_000, 352_800, 24_000, 22_050, 16_000, 11_025, 8_000,
    ];
    let candidates = requested
        .as_ref()
        .map_or(defaults.as_slice(), std::slice::from_ref);
    for (index, &rate) in candidates.iter().enumerate() {
        if !(8_000..=384_000).contains(&rate) || candidates[..index].contains(&rate) {
            continue;
        }
        let Some(capture) = left.exclusive_at(input, rate)? else {
            continue;
        };
        if let Some(playback) = right.exclusive_at(output, rate)? {
            return Ok((capture, playback));
        }
    }
    if let Some(rate) = requested {
        bail!("WASAPI exclusive input and output have no PCM formats at {rate} Hz");
    }
    bail!("WASAPI exclusive input and output have no common PCM sample rate");
}

struct AudioEndpoint {
    client: IAudioClient,
    event: Event,
    format: NativeFormat,
    frames: u32,
    running: bool,
}

impl AudioEndpoint {
    fn new(device: &IMMDevice, client: IAudioClient, format: NativeFormat) -> Result<Self> {
        let mut default_period = 0;
        let mut minimum_period = 0;
        // SAFETY: both output pointers are valid while the client is live.
        unsafe { client.GetDevicePeriod(Some(&mut default_period), Some(&mut minimum_period)) }
            .context("get WASAPI device period")?;
        let period = default_period.max(minimum_period);
        ensure!(
            period > 0,
            "WASAPI device reported an invalid hardware period"
        );

        let client = match initialize(&client, period, &format) {
            Ok(()) => client,
            Err(error) if error.code() == AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED => {
                // SAFETY: after an alignment error WASAPI reports the correct
                // frame count on the uninitialized client. A fresh client is
                // required for the retry.
                let aligned = unsafe { client.GetBufferSize() }?;
                ensure!(aligned > 0, "WASAPI device reported a zero aligned buffer");
                let rate = i64::from(format.rate());
                let period = (i64::from(aligned) * REFERENCE_TIME_SECOND + rate / 2) / rate;
                drop(client);
                let client = activate(device)?;
                initialize(&client, period, &format)
                    .map_err(exclusive_error)
                    .context("retry WASAPI exclusive initialization with an aligned buffer")?;
                client
            }
            Err(error) => {
                return Err(exclusive_error(error)).context("initialize WASAPI exclusive audio");
            }
        };
        let frames = unsafe { client.GetBufferSize() }.context("get WASAPI audio buffer size")?;
        ensure!(
            frames > 0,
            "WASAPI device reported an empty endpoint buffer"
        );
        buffer_bytes(frames, format)?;
        let endpoint = Self {
            client,
            event: Event::new(false)?,
            format,
            frames,
            running: false,
        };
        // SAFETY: the initialized client is dropped before the attached event.
        unsafe { endpoint.client.SetEventHandle(endpoint.event.handle()) }
            .context("set WASAPI audio event")?;
        Ok(endpoint)
    }

    fn start(&mut self) -> Result<()> {
        // SAFETY: the event and the same-thread client are ready for streaming.
        unsafe { self.client.Start() }?;
        self.running = true;
        Ok(())
    }
}

impl Drop for AudioEndpoint {
    fn drop(&mut self) {
        if self.running {
            // SAFETY: this endpoint was started and belongs to this thread.
            let _ = unsafe { self.client.Stop() };
        }
    }
}

fn initialize(
    client: &IAudioClient,
    period: i64,
    format: &NativeFormat,
) -> windows::core::Result<()> {
    // SAFETY: the same buffer duration and device period are required by
    // WASAPI's exclusive event mode. The descriptor lives through the call.
    unsafe {
        client.Initialize(
            AUDCLNT_SHAREMODE_EXCLUSIVE,
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
            period,
            period,
            format.as_ptr(),
            None,
        )
    }
}

fn exclusive_error(error: WinError) -> anyhow::Error {
    match error.code() {
        AUDCLNT_E_DEVICE_IN_USE => {
            anyhow::Error::new(error).context("another stream is using this audio endpoint")
        }
        AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED => anyhow::Error::new(error)
            .context("Windows has disabled exclusive access to this endpoint"),
        _ => error.into(),
    }
}

struct Endpoints {
    capture: IAudioCaptureClient,
    render: IAudioRenderClient,
    input: AudioEndpoint,
    output: AudioEndpoint,
}

impl Endpoints {
    fn start(&mut self) -> Result<()> {
        // SAFETY: exclusive event render must prime one full buffer before Start.
        unsafe {
            self.render.GetBuffer(self.output.frames)?;
            self.render
                .ReleaseBuffer(self.output.frames, AUDCLNT_BUFFERFLAGS_SILENT.0 as u32)?;
        }
        self.output
            .start()
            .context("start WASAPI exclusive output")?;
        self.input.start().context("start WASAPI exclusive input")?;
        Ok(())
    }

    fn run(
        &self,
        mut capture: Capture,
        mut playback: Playback,
        stop: &Event,
        status: &StreamStatus,
    ) -> Result<()> {
        let handles = [
            stop.handle(),
            self.input.event.handle(),
            self.output.event.handle(),
        ];
        let mut last_input = Instant::now();
        let mut last_output = last_input;
        let stall = Duration::from_millis(u64::from(DEVICE_TIMEOUT));
        loop {
            // SAFETY: all handles remain live during this wait. The stop event
            // has priority over ready capture or playback.
            let ready = unsafe { WaitForMultipleObjects(&handles, false, DEVICE_TIMEOUT) };
            if ready == WAIT_OBJECT_0 {
                break;
            }
            if ready == WAIT_FAILED {
                return Err(std::io::Error::last_os_error())
                    .context("wait for WASAPI audio events");
            }
            if ready.0 == WAIT_OBJECT_0.0 + 1 {
                self.capture_packet(&mut capture, status)
                    .context("read WASAPI exclusive input")?;
                last_input = Instant::now();
            } else if ready.0 == WAIT_OBJECT_0.0 + 2 {
                self.render_packet(&mut playback, status)
                    .context("write WASAPI exclusive output")?;
                last_output = Instant::now();
            } else if ready != WAIT_TIMEOUT {
                bail!("unexpected WASAPI wait status {}", ready.0);
            }
            ensure!(
                last_input.elapsed() < stall,
                "WASAPI exclusive input stopped signaling"
            );
            ensure!(
                last_output.elapsed() < stall,
                "WASAPI exclusive output stopped signaling"
            );
        }
        Ok(())
    }

    fn capture_packet(&self, capture: &mut Capture, status: &StreamStatus) -> Result<()> {
        let mut pointer = ptr::null_mut();
        let mut frames = 0;
        let mut flags = 0;
        let mut position = 0;
        // SAFETY: the client owns the returned packet until ReleaseBuffer.
        let buffer = unsafe {
            self.capture.GetBuffer(
                &mut pointer,
                &mut frames,
                &mut flags,
                Some(&mut position),
                None,
            )
        };
        match buffer {
            Ok(()) => {}
            Err(error) if error.code() == AUDCLNT_E_BUFFER_ERROR => {
                status.xrun();
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        }
        if frames == 0 {
            return Ok(());
        }
        if position != 0 && flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32 != 0 {
            status.xrun();
        }
        if frames > self.input.frames {
            unsafe { self.capture.ReleaseBuffer(0) }?;
            bail!("WASAPI returned an oversized capture packet");
        }
        let count = frames as usize;
        if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
            capture.write(std::iter::repeat_n(0.0, count));
        } else if pointer.is_null() {
            unsafe { self.capture.ReleaseBuffer(0) }?;
            bail!("WASAPI returned an invalid capture packet");
        } else {
            let format = self.input.format;
            let length = count * format.frame_bytes();
            // SAFETY: non-silent packets provide exactly frames interleaved samples.
            let bytes = unsafe { std::slice::from_raw_parts(pointer, length) };
            capture.write(
                bytes
                    .chunks_exact(format.frame_bytes())
                    .map(|frame| format.pcm.downmix(frame)),
            );
        }
        unsafe { self.capture.ReleaseBuffer(frames) }?;
        Ok(())
    }

    fn render_packet(&self, playback: &mut Playback, status: &StreamStatus) -> Result<()> {
        let output = &self.output;
        // SAFETY: on each ready event the driver supplies one full exclusive buffer.
        let pointer = match unsafe { self.render.GetBuffer(output.frames) } {
            Ok(pointer) => pointer,
            Err(error) if error.code() == AUDCLNT_E_BUFFER_ERROR => {
                status.xrun();
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        if pointer.is_null() {
            unsafe {
                self.render
                    .ReleaseBuffer(output.frames, AUDCLNT_BUFFERFLAGS_SILENT.0 as u32)
            }?;
            bail!("WASAPI returned an invalid playback packet");
        }
        let count = output.frames as usize;
        let length = count * output.format.frame_bytes();
        // SAFETY: WASAPI owns a writable interleaved packet of this size. Make
        // the bytes initialized before borrowing them as a Rust slice.
        let bytes = unsafe {
            pointer.write_bytes(0, length);
            std::slice::from_raw_parts_mut(pointer, length)
        };
        let mut frames = bytes.chunks_exact_mut(output.format.frame_bytes());
        playback.read(count, |sample| {
            if let Some(frame) = frames.next() {
                output.format.pcm.render(frame, sample);
            }
        });
        unsafe { self.render.ReleaseBuffer(output.frames, 0) }?;
        Ok(())
    }
}

fn buffer_bytes(frames: u32, format: NativeFormat) -> Result<usize> {
    (frames as usize)
        .checked_mul(format.frame_bytes())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .context("WASAPI device buffer is too large")
}
