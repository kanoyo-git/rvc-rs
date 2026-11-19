use std::{
    cell::Cell,
    ffi::{CStr, c_void},
    ptr,
    sync::{Arc, Mutex},
};

#[cfg(target_os = "linux")]
use ::vst3::ComRef;
use ::vst3::{
    Class, ComPtr, ComWrapper,
    Steinberg::{Vst::*, *},
};
#[cfg(target_os = "linux")]
use std::cell::RefCell;

use crate::{
    daw::{BLOCK_MS, PluginControls},
    editor::{self, EditBatch, EditorAttachment},
};

type Handler = Arc<Mutex<Option<ComPtr<IComponentHandler>>>>;

struct View {
    state: EditorAttachment,
    controller: Handler,
    size: Cell<ViewRect>,
    #[cfg(target_os = "linux")]
    frame: RefCell<Option<ComPtr<IPlugFrame>>>,
    #[cfg(target_os = "linux")]
    polling: RefCell<Option<HostTimer>>,
}

pub(super) fn create(controls: Arc<PluginControls>, controller: Handler) -> *mut IPlugView {
    let view = View {
        state: EditorAttachment::new(controls),
        controller,
        size: Cell::new(rect(editor::WIDTH, editor::HEIGHT)),
        #[cfg(target_os = "linux")]
        frame: RefCell::new(None),
        #[cfg(target_os = "linux")]
        polling: RefCell::new(None),
    };
    ComWrapper::new(view)
        .to_com_ptr::<IPlugView>()
        .map_or(ptr::null_mut(), ComPtr::into_raw)
}

impl Class for View {
    type Interfaces = (IPlugView,);
}

impl View {
    fn flush(&self) {
        notify(&self.controller, self.state.model().take_edits());
    }

    #[cfg(target_os = "linux")]
    fn restart_timer(&self) {
        self.stop_timer();
        if self.state.is_open()
            && let Some(frame) = self.frame.borrow().as_ref()
            && let Some(timer) =
                HostTimer::start(frame, self.state.model(), Arc::clone(&self.controller))
        {
            *self.polling.borrow_mut() = Some(timer);
        }
    }

    #[cfg(target_os = "linux")]
    fn stop_timer(&self) {
        let timer = self.polling.borrow_mut().take();
        drop(timer);
    }
}

impl IPlugViewTrait for View {
    unsafe fn isPlatformTypeSupported(&self, kind: FIDString) -> tresult {
        // SAFETY: the VST host owns a NUL-terminated platform type string.
        if unsafe { supported(kind) } {
            kResultTrue
        } else {
            kResultFalse
        }
    }

    unsafe fn attached(&self, parent: *mut c_void, kind: FIDString) -> tresult {
        // SAFETY: the VST host owns a NUL-terminated platform type string.
        if parent.is_null() || !unsafe { supported(kind) } {
            return kInvalidArgument;
        }
        let (width, height) = dimensions(self.size.get());
        #[cfg(target_os = "linux")]
        let sink = None;
        #[cfg(target_os = "windows")]
        let sink = {
            // The HWND child dispatches on the host's GUI thread.
            let controller = Arc::clone(&self.controller);
            let sink: editor::EditSink = Arc::new(move |edits| notify(&controller, edits));
            Some(sink)
        };
        match self.state.open(parent, width, height, sink) {
            Ok(()) => {
                #[cfg(target_os = "linux")]
                self.restart_timer();
                kResultOk
            }
            Err(error) => {
                eprintln!("RVC Realtime: {error:#}");
                kResultFalse
            }
        }
    }

    unsafe fn removed(&self) -> tresult {
        #[cfg(target_os = "linux")]
        self.stop_timer();
        self.state.close();
        self.flush();
        kResultOk
    }

    unsafe fn onWheel(&self, _: f32) -> tresult {
        kResultFalse
    }
    unsafe fn onKeyDown(&self, _: u16, _: i16, _: i16) -> tresult {
        kResultFalse
    }
    unsafe fn onKeyUp(&self, _: u16, _: i16, _: i16) -> tresult {
        kResultFalse
    }

    unsafe fn getSize(&self, output: *mut ViewRect) -> tresult {
        if output.is_null() {
            return kInvalidArgument;
        }
        // SAFETY: output is the writable rectangle requested by the host.
        unsafe { output.write(self.size.get()) };
        kResultOk
    }

    unsafe fn onSize(&self, requested: *mut ViewRect) -> tresult {
        let Some(size) = (unsafe { requested.as_ref() }) else {
            return kInvalidArgument;
        };
        let (width, height) = dimensions(*size);
        if self.state.resize(width, height).is_err() {
            return kResultFalse;
        }
        self.size.set(rect(width, height));
        self.flush();
        kResultOk
    }

    unsafe fn onFocus(&self, _: TBool) -> tresult {
        self.flush();
        kResultOk
    }

    unsafe fn setFrame(&self, _frame: *mut IPlugFrame) -> tresult {
        #[cfg(target_os = "linux")]
        {
            // SAFETY: the optional frame is borrowed by the host during this call.
            *self.frame.borrow_mut() =
                unsafe { ComRef::from_raw(_frame) }.map(|frame| frame.to_com_ptr());
            self.restart_timer();
        }
        kResultOk
    }

    unsafe fn canResize(&self) -> tresult {
        kResultTrue
    }

    unsafe fn checkSizeConstraint(&self, size: *mut ViewRect) -> tresult {
        let Some(size) = (unsafe { size.as_mut() }) else {
            return kInvalidArgument;
        };
        let (width, height) = dimensions(*size);
        *size = rect(width, height);
        kResultOk
    }
}

impl Drop for View {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        self.stop_timer();
        self.state.close();
    }
}

unsafe fn supported(kind: FIDString) -> bool {
    if kind.is_null() {
        return false;
    }
    // SAFETY: the host's ABI supplies a readable NUL-terminated string.
    let kind = unsafe { CStr::from_ptr(kind) }.to_bytes();
    #[cfg(target_os = "linux")]
    return kind == b"X11EmbedWindowID";
    #[cfg(target_os = "windows")]
    return kind == b"HWND";
}

fn dimensions(size: ViewRect) -> (u32, u32) {
    (
        size.right.saturating_sub(size.left).clamp(600, 1500) as u32,
        size.bottom.saturating_sub(size.top).clamp(455, 1250) as u32,
    )
}

fn rect(width: u32, height: u32) -> ViewRect {
    ViewRect {
        left: 0,
        top: 0,
        right: width as i32,
        bottom: height as i32,
    }
}

fn notify(controller: &Handler, edits: EditBatch) {
    if edits.iter().all(Option::is_none) {
        return;
    }
    let handler = controller
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .clone();
    let Some(handler) = handler else { return };
    // SAFETY: controller is retained by the host interface and this function is
    // called on its GUI thread (Windows child dispatch or Linux host timer).
    unsafe {
        for (index, value) in edits.into_iter().enumerate() {
            let Some(value) = value else { continue };
            let id = index as ParamID;
            handler.beginEdit(id);
            handler.performEdit(id, f64::from(value));
            handler.endEdit(id);
        }
        if edits[BLOCK_MS].is_some() {
            handler.restartComponent(RestartFlags_::kLatencyChanged);
        }
    }
}

#[cfg(target_os = "linux")]
use ::vst3::Steinberg::Linux::{IRunLoop, IRunLoopTrait, ITimerHandler, ITimerHandlerTrait};

#[cfg(target_os = "linux")]
struct HostTimer {
    run_loop: ComPtr<IRunLoop>,
    handler: ComPtr<ITimerHandler>,
}

#[cfg(target_os = "linux")]
impl HostTimer {
    fn start(
        frame: &ComPtr<IPlugFrame>,
        events: Arc<editor::EditorModel>,
        controller: Handler,
    ) -> Option<Self> {
        let run_loop = frame.cast::<IRunLoop>()?;
        let handler =
            ComWrapper::new(Timer { events, controller }).to_com_ptr::<ITimerHandler>()?;
        // SAFETY: both COM pointers remain retained for the registration lifetime.
        if unsafe { run_loop.registerTimer(handler.as_ptr(), 25) } != kResultOk {
            return None;
        }
        Some(Self { run_loop, handler })
    }
}

#[cfg(target_os = "linux")]
impl Drop for HostTimer {
    fn drop(&mut self) {
        // SAFETY: the live run loop owns this timer registration until removal.
        unsafe { self.run_loop.unregisterTimer(self.handler.as_ptr()) };
    }
}

#[cfg(target_os = "linux")]
struct Timer {
    events: Arc<editor::EditorModel>,
    controller: Handler,
}

#[cfg(target_os = "linux")]
impl Class for Timer {
    type Interfaces = (ITimerHandler,);
}

#[cfg(target_os = "linux")]
impl ITimerHandlerTrait for Timer {
    unsafe fn onTimer(&self) {
        notify(&self.controller, self.events.take_edits());
    }
}
