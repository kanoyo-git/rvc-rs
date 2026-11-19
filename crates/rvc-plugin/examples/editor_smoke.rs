//! X11 host harness for the VST2 and VST3 native editors.

#[cfg(target_os = "linux")]
mod x11 {
    use std::{
        cell::{Cell, RefCell},
        env,
        ffi::c_void,
        fs,
        path::Path,
        ptr,
        rc::Rc,
        sync::atomic::{AtomicU32, Ordering},
        thread,
        time::{Duration, Instant},
    };

    use anyhow::{Context, Result, ensure};
    use libloading::Library;
    use rvc_realtime::vst3::COMPONENT_UID;
    use vst::{
        api::{AEffect, HostCallbackProc, PluginFlags},
        host, plugin,
    };
    use vst3::{
        Class, ComPtr, ComRef, ComWrapper, Interface,
        Steinberg::{Linux::*, Vst::*, *},
    };
    use x11rb::{
        COPY_DEPTH_FROM_PARENT,
        connection::Connection,
        protocol::{
            xproto::{self, ConnectionExt, CreateWindowAux, ImageFormat, WindowClass},
            xtest,
        },
        rust_connection::RustConnection,
    };

    type Exit = unsafe extern "C" fn() -> bool;
    static VST2_PITCH_EDIT: AtomicU32 = AtomicU32::new(f32::NAN.to_bits());

    pub fn run() -> Result<()> {
        let mut args = env::args_os().skip(1);
        let library = args
            .next()
            .context("usage: editor_smoke <library> [capture-folder] [fixture-model]")?;
        let capture = args.next().map(std::path::PathBuf::from);
        // RVC_PLUGIN_ROOT must point to the fixture's RVC folder when supplied.
        let fixture = args.next().map(std::path::PathBuf::from);
        if let Some(capture) = &capture {
            fs::create_dir_all(capture)?;
        }
        // SAFETY: the caller supplies the built local RVC plug-in library.
        let lib = unsafe { Library::new(library)? };
        let (connection, screen_num) = x11rb::connect(None).context("connect to X11")?;
        let screen = &connection.setup().roots[screen_num];
        let parent = connection.generate_id()?;
        connection
            .create_window(
                COPY_DEPTH_FROM_PARENT,
                parent,
                screen.root,
                32,
                28,
                820,
                670,
                0,
                WindowClass::INPUT_OUTPUT,
                0,
                &CreateWindowAux::new().background_pixel(0x101010),
            )?
            .check()?;
        connection.map_window(parent)?.check()?;
        run_vst2(
            &lib,
            &connection,
            parent,
            capture.as_deref(),
            fixture.as_deref(),
        )?;
        run_vst3(&lib, &connection, parent, capture.as_deref())?;
        connection.destroy_window(parent)?.check()?;
        println!("VST2 and VST3 drew and edited pitch in X11 hosts, then detached cleanly");
        Ok(())
    }

    extern "C" fn host_callback(
        _: *mut AEffect,
        event: i32,
        index: i32,
        _: isize,
        _: *mut c_void,
        value: f32,
    ) -> isize {
        if event == host::OpCode::Version as i32 {
            2400
        } else {
            if event == host::OpCode::Automate as i32 && index == 1 {
                VST2_PITCH_EDIT.store(value.to_bits(), Ordering::Relaxed);
            }
            0
        }
    }

    fn run_vst2(
        lib: &Library,
        connection: &RustConnection,
        parent: u32,
        capture: Option<&Path>,
        fixture: Option<&Path>,
    ) -> Result<()> {
        type Main = unsafe extern "C" fn(Option<HostCallbackProc>) -> *mut AEffect;
        // SAFETY: this is the VST2 ABI exported by the local library.
        let main = unsafe { lib.get::<Main>(b"VSTPluginMain")? };
        let instance = unsafe { main(Some(host_callback)) };
        ensure!(!instance.is_null(), "VST2 factory rejected test host");
        let effect = Vst2Effect(instance);
        // SAFETY: the test owns this effect for the complete editor lifecycle.
        ensure!(unsafe { (*instance).flags } & PluginFlags::HAS_EDITOR.bits() != 0);
        ensure!(effect.call(plugin::OpCode::Initialize, ptr::null_mut()) == 1);
        let mut bounds = ptr::null::<[i16; 4]>();
        ensure!(effect.call(plugin::OpCode::EditorGetRect, (&raw mut bounds).cast()) == 1);
        ensure!(!bounds.is_null() && unsafe { *bounds } == [0, 0, 630, 780]);
        ensure!(
            effect.call(plugin::OpCode::EditorOpen, parent as usize as *mut c_void) == 1,
            "VST2 could not open a child in the X11 host"
        );
        let child = child_window(connection, parent)?;
        let geometry = connection.get_geometry(child)?.reply()?;
        ensure!((geometry.width, geometry.height) == (780, 630));
        wait_for_drawing(
            connection,
            child,
            capture.map(|dir| dir.join("vst2.ppm")).as_deref(),
        )?;
        let pitch = unsafe { ((*instance).getParameter)(instance, 1) };
        click(connection, child, 35, 335)?;
        wait_until("VST2 pitch automation", || {
            effect.call(plugin::OpCode::EditorIdle, ptr::null_mut());
            let edit = f32::from_bits(VST2_PITCH_EDIT.load(Ordering::Relaxed));
            let value = unsafe { ((*instance).getParameter)(instance, 1) };
            Ok(edit.is_finite() && (edit - value).abs() < 1e-6 && (edit - pitch).abs() > 1e-3)
        })?;
        if let Some(model) = fixture {
            browse_vst2(&effect, connection, child, capture, model)?;
        }
        ensure!(effect.call(plugin::OpCode::EditorClose, ptr::null_mut()) == 1);
        wait_until("VST2 editor removal", || {
            Ok(connection.query_tree(parent)?.reply()?.children.is_empty())
        })?;
        Ok(())
    }

    struct Vst2Effect(*mut AEffect);

    impl Vst2Effect {
        fn call(&self, event: plugin::OpCode, data: *mut c_void) -> isize {
            // SAFETY: the test owns this effect until shutdown.
            unsafe { ((*self.0).dispatcher)(self.0, event as i32, 0, 0, data, 0.0) }
        }

        fn contains_path(&self, path: &Path) -> bool {
            let mut chunk = ptr::null::<u8>();
            let count = self.call(plugin::OpCode::GetData, (&raw mut chunk).cast());
            if chunk.is_null() || count <= 0 {
                return false;
            }
            // SAFETY: the test owns the effect and copies before the next call.
            let bytes = unsafe { std::slice::from_raw_parts(chunk, count as usize) };
            let path = path.to_string_lossy();
            bytes
                .windows(path.len())
                .any(|window| window == path.as_bytes())
        }
    }

    fn browse_vst2(
        effect: &Vst2Effect,
        connection: &RustConnection,
        child: u32,
        capture: Option<&Path>,
        model: &Path,
    ) -> Result<()> {
        ensure!(model.is_file() && model.is_absolute());
        ensure!(!effect.contains_path(model), "fixture is already selected");
        click(connection, child, 661, 176)?;
        thread::sleep(Duration::from_millis(250));
        wait_for_drawing(
            connection,
            child,
            capture.map(|dir| dir.join("model-browser.ppm")).as_deref(),
        )?;
        click(connection, child, 73, 131)?;
        wait_until("VST2 browser model saved in state", || {
            Ok(effect.contains_path(model))
        })
    }

    impl Drop for Vst2Effect {
        fn drop(&mut self) {
            self.call(plugin::OpCode::EditorClose, ptr::null_mut());
            self.call(plugin::OpCode::Shutdown, ptr::null_mut());
        }
    }

    fn run_vst3(
        lib: &Library,
        connection: &RustConnection,
        parent: u32,
        capture: Option<&Path>,
    ) -> Result<()> {
        type Entry = unsafe extern "C" fn(*mut c_void) -> bool;
        type Factory = unsafe extern "system" fn() -> *mut IPluginFactory;
        // SAFETY: every pointer below represents a local exported interface or
        // host-owned object with a lifetime covering the call.
        unsafe {
            let entry = lib.get::<Entry>(b"ModuleEntry")?;
            let exit = lib.get::<Exit>(b"ModuleExit")?;
            let factory = lib.get::<Factory>(b"GetPluginFactory")?;
            ensure!(entry(ptr::null_mut()));
            let factory = ComPtr::from_raw(factory()).context("missing VST3 factory")?;
            let mut instance = ptr::null_mut::<c_void>();
            ensure!(
                factory.createInstance(
                    COMPONENT_UID.as_ptr(),
                    IComponent::IID.as_ptr().cast(),
                    &mut instance,
                ) == kResultOk
            );
            let component =
                ComPtr::from_raw(instance.cast::<IComponent>()).context("no VST3 component")?;
            ensure!(component.initialize(ptr::null_mut()) == kResultOk);
            let controller = component
                .cast::<IEditController>()
                .context("no VST3 controller")?;
            let view = ComPtr::from_raw(controller.createView(c"editor".as_ptr()))
                .context("no plug-in editor view")?;
            let calls = Rc::new(HostCalls::default());
            let host = ComWrapper::new(EditorHost(Rc::clone(&calls)));
            let frame = host
                .to_com_ptr::<IPlugFrame>()
                .context("missing host frame")?;
            let handler = host
                .to_com_ptr::<IComponentHandler>()
                .context("missing host handler")?;
            ensure!(controller.setComponentHandler(handler.as_ptr()) == kResultOk);
            ensure!(view.setFrame(frame.as_ptr()) == kResultOk);
            ensure!(view.isPlatformTypeSupported(c"HWND".as_ptr()) == kResultFalse);
            ensure!(view.isPlatformTypeSupported(c"X11EmbedWindowID".as_ptr()) == kResultTrue);
            ensure!(
                view.attached(parent as usize as *mut c_void, c"X11EmbedWindowID".as_ptr())
                    == kResultOk
            );
            let child = child_window(connection, parent)?;
            let mut size = ViewRect {
                left: 0,
                top: 0,
                right: 800,
                bottom: 640,
            };
            ensure!(view.canResize() == kResultTrue);
            ensure!(view.onSize(&mut size) == kResultOk);
            let geometry = connection.get_geometry(child)?.reply()?;
            ensure!((geometry.width, geometry.height) == (800, 640));
            wait_for_drawing(
                connection,
                child,
                capture.map(|dir| dir.join("vst3.ppm")).as_deref(),
            )?;
            let pitch = controller.getParamNormalized(1);
            click(connection, child, 35, 335)?;
            wait_until("VST3 pitch automation", || {
                let timer = calls
                    .timer
                    .borrow()
                    .clone()
                    .context("timer not registered")?;
                timer.onTimer();
                let edit = calls.pitch.get();
                let value = controller.getParamNormalized(1);
                Ok(edit
                    .is_some_and(|edit| (edit - value).abs() < 1e-6 && (edit - pitch).abs() > 1e-3))
            })?;
            ensure!(view.removed() == kResultOk);
            ensure!(calls.timer.borrow().is_none());
            wait_until("VST3 editor removal", || {
                Ok(connection.query_tree(parent)?.reply()?.children.is_empty())
            })?;
            drop(view);
            drop(controller);
            ensure!(component.terminate() == kResultOk);
            drop(component);
            drop(factory);
            ensure!(exit());
        }
        Ok(())
    }

    #[derive(Default)]
    struct HostCalls {
        pitch: Cell<Option<f64>>,
        timer: RefCell<Option<ComPtr<ITimerHandler>>>,
    }

    struct EditorHost(Rc<HostCalls>);

    impl Class for EditorHost {
        type Interfaces = (IPlugFrame, IRunLoop, IComponentHandler);
    }

    impl IPlugFrameTrait for EditorHost {
        unsafe fn resizeView(&self, _: *mut IPlugView, _: *mut ViewRect) -> tresult {
            kResultOk
        }
    }

    impl IRunLoopTrait for EditorHost {
        unsafe fn registerEventHandler(&self, _: *mut IEventHandler, _: FileDescriptor) -> tresult {
            kNotImplemented
        }
        unsafe fn unregisterEventHandler(&self, _: *mut IEventHandler) -> tresult {
            kNotImplemented
        }
        unsafe fn registerTimer(&self, handler: *mut ITimerHandler, _: TimerInterval) -> tresult {
            *self.0.timer.borrow_mut() =
                unsafe { ComRef::from_raw(handler) }.map(|handler| handler.to_com_ptr());
            kResultOk
        }
        unsafe fn unregisterTimer(&self, _: *mut ITimerHandler) -> tresult {
            self.0.timer.borrow_mut().take();
            kResultOk
        }
    }

    impl IComponentHandlerTrait for EditorHost {
        unsafe fn beginEdit(&self, _: ParamID) -> tresult {
            kResultOk
        }
        unsafe fn performEdit(&self, id: ParamID, value: ParamValue) -> tresult {
            if id == 1 {
                self.0.pitch.set(Some(value));
            }
            kResultOk
        }
        unsafe fn endEdit(&self, _: ParamID) -> tresult {
            kResultOk
        }
        unsafe fn restartComponent(&self, _: i32) -> tresult {
            kResultOk
        }
    }

    fn click(connection: &RustConnection, child: u32, x: i16, y: i16) -> Result<()> {
        let root = connection.get_geometry(child)?.reply()?.root;
        let position = connection
            .translate_coordinates(child, root, x, y)?
            .reply()?;
        for (kind, button) in [
            (xproto::MOTION_NOTIFY_EVENT, 0),
            (xproto::BUTTON_PRESS_EVENT, 1),
            (xproto::BUTTON_RELEASE_EVENT, 1),
        ] {
            xtest::fake_input(
                connection,
                kind,
                button,
                x11rb::CURRENT_TIME,
                root,
                position.dst_x,
                position.dst_y,
                0,
            )?
            .check()?;
            if kind == xproto::BUTTON_PRESS_EVENT {
                // The GUI samples input on its draw cycle.
                thread::sleep(Duration::from_millis(90));
            }
        }
        Ok(())
    }

    fn wait_until(label: &str, mut check: impl FnMut() -> Result<bool>) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(3);
        while !check()? {
            ensure!(Instant::now() < deadline, "{label} did not complete");
            thread::sleep(Duration::from_millis(30));
        }
        Ok(())
    }

    fn child_window(connection: &RustConnection, parent: u32) -> Result<u32> {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if let Some(&child) = connection.query_tree(parent)?.reply()?.children.first() {
                return Ok(child);
            }
            ensure!(
                Instant::now() < deadline,
                "X11 parent has no plug-in editor child"
            );
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn wait_for_drawing(
        connection: &RustConnection,
        window: xproto::Window,
        capture: Option<&Path>,
    ) -> Result<()> {
        let geometry = connection.get_geometry(window)?.reply()?;
        let deadline = Instant::now() + Duration::from_secs(4);
        loop {
            let image = connection
                .get_image(
                    ImageFormat::Z_PIXMAP,
                    window,
                    0,
                    0,
                    geometry.width,
                    geometry.height,
                    u32::MAX,
                )?
                .reply()?;
            let width = usize::from(geometry.width);
            let height = usize::from(geometry.height);
            ensure!(
                image.data.len() == width * height * 4,
                "harness expects a 24-bit X11 visual"
            );
            let (pixels, _) = image.data.as_chunks::<4>();
            let mixed = pixels.iter().any(|pixel| pixel[..3] != image.data[..3]);
            if mixed {
                if let Some(path) = capture {
                    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
                    for pixel in pixels {
                        ppm.extend([pixel[2], pixel[1], pixel[0]]);
                    }
                    fs::write(path, ppm)?;
                }
                return Ok(());
            }
            ensure!(
                Instant::now() < deadline,
                "editor child remained blank for four seconds"
            );
            thread::sleep(Duration::from_millis(50));
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    x11::run()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("editor_smoke uses an X11 test host");
}
