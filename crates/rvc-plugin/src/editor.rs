//! Model paths and RVC automation controls shared by the VST editor adapters.

use std::{
    cell::{Cell, RefCell},
    ffi::c_void,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use anyhow::{Context, Result, anyhow, ensure};
use baseview::{
    Event, EventStatus, HandlerError, MouseButton, MouseEvent, ScrollDelta, Window, WindowContext,
    WindowEvent, WindowHandler, WindowSettings, WindowSize,
    dpi::PhysicalSize,
    gl::{GlConfig, GlContext},
};
use egui::{Color32, FontId, PointerButton, Pos2, Rect, RichText, TextStyle, Vec2};
use egui_glow::{Painter, glow};
use keyboard_types::{Key as NativeKey, KeyState, Modifiers, NamedKey};
use raw_window_handle::{HandleError, HasWindowHandle, RawWindowHandle, WindowHandle};

use crate::daw::{
    BLOCK_MS, ENGINE, ModelPaths, PARAMETER_COUNT, PARAMETERS, PITCH_METHODS, PluginControls,
    WorkerStatus, pitch_method_index,
};

pub(crate) const WIDTH: u32 = 780;
pub(crate) const HEIGHT: u32 = 630;

pub(crate) type EditBatch = [Option<f32>; PARAMETER_COUNT];
pub(crate) type EditSink = Arc<dyn Fn(EditBatch) + Send + Sync>;

pub(crate) struct EditorModel {
    controls: Arc<PluginControls>,
    pending: Mutex<EditBatch>,
    closing: AtomicBool,
}

impl EditorModel {
    fn new(controls: Arc<PluginControls>) -> Self {
        Self {
            controls,
            pending: Mutex::new([None; PARAMETER_COUNT]),
            closing: AtomicBool::new(false),
        }
    }

    fn set_plain(&self, index: usize, value: f32) {
        let spec = PARAMETERS[index];
        let normalized = spec.normalize(value);
        self.controls.set_normalized(index, normalized);
        self.enqueue(index, normalized);
    }

    fn update_path(&self, kind: PathKind, path: PathBuf) -> Result<()> {
        let (mut paths, _) = self.controls.snapshot();
        kind.validate(&paths.rvc_root, &path)?;
        *kind.path_mut(&mut paths) = path;
        if self.controls.set_model_paths(paths) {
            self.enqueue(ENGINE, 0.0);
        }
        Ok(())
    }

    fn enqueue(&self, index: usize, value: f32) {
        self.pending.lock().unwrap_or_else(|err| err.into_inner())[index] = Some(value);
    }

    pub(crate) fn take_edits(&self) -> EditBatch {
        std::mem::take(&mut *self.pending.lock().unwrap_or_else(|err| err.into_inner()))
    }
}

/// Window creation, resize and removal are called by the host's GUI thread.
pub(crate) struct EditorAttachment {
    model: Arc<EditorModel>,
    window: RefCell<Option<Window>>,
}

impl EditorAttachment {
    pub(crate) fn new(controls: Arc<PluginControls>) -> Self {
        Self {
            model: Arc::new(EditorModel::new(controls)),
            window: RefCell::new(None),
        }
    }

    pub(crate) fn model(&self) -> Arc<EditorModel> {
        Arc::clone(&self.model)
    }

    pub(crate) fn open(
        &self,
        parent: *mut c_void,
        width: u32,
        height: u32,
        sink: Option<EditSink>,
    ) -> Result<()> {
        ensure!(!self.is_open(), "editor is already attached");
        let parent = EditorParent::new(parent).context("invalid host window")?;
        self.close();
        self.model.closing.store(false, Ordering::Release);
        let model = Arc::clone(&self.model);
        let settings = WindowSettings::new()
            .with_title("RVC Realtime")
            .with_size(PhysicalSize::new(width, height))
            .with_parent(&parent)
            .with_resizable(true)
            .with_gl_config(GlConfig::default());
        let window = Window::create(settings, move |window| {
            EditorWindow::new(window, model, sink)
        })
        .map_err(|error| anyhow!("create RVC editor: {error}"))?;
        let shown = window.show();
        *self.window.borrow_mut() = Some(window);
        if let Err(error) = shown {
            self.close();
            return Err(anyhow!("show RVC editor: {error}"));
        }
        Ok(())
    }

    pub(crate) fn is_open(&self) -> bool {
        self.window.borrow().as_ref().is_some_and(Window::is_open)
    }

    pub(crate) fn resize(&self, width: u32, height: u32) -> Result<()> {
        if let Some(window) = self.window.borrow().as_ref() {
            window
                .resize(PhysicalSize::new(width, height))
                .map_err(|error| anyhow!("resize RVC editor: {error}"))?;
        }
        Ok(())
    }

    pub(crate) fn close(&self) {
        let Some(window) = self.window.borrow_mut().take() else {
            return;
        };
        #[cfg(target_os = "linux")]
        {
            // Request closure inside the X11 event loop. Once it stops, join
            // explicitly before the host is allowed to unload this library.
            self.model.closing.store(true, Ordering::Release);
            let _ = window.request_poll();
            while window.is_open() {
                std::thread::sleep(Duration::from_millis(1));
            }
            let _ = window.run_until_closed();
        }
        #[cfg(target_os = "windows")]
        window.close();
    }
}

impl Drop for EditorAttachment {
    fn drop(&mut self) {
        self.close();
    }
}

struct EditorParent(RawWindowHandle);

impl EditorParent {
    #[cfg(target_os = "linux")]
    fn new(parent: *mut c_void) -> Option<Self> {
        use raw_window_handle::XcbWindowHandle;
        let id = u32::try_from(parent as usize).ok()?.try_into().ok()?;
        Some(Self(RawWindowHandle::Xcb(XcbWindowHandle::new(id))))
    }

    #[cfg(target_os = "windows")]
    fn new(parent: *mut c_void) -> Option<Self> {
        use raw_window_handle::Win32WindowHandle;
        let id = (parent as isize).try_into().ok()?;
        Some(Self(RawWindowHandle::Win32(Win32WindowHandle::new(id))))
    }
}

impl HasWindowHandle for EditorParent {
    fn window_handle(&self) -> std::result::Result<WindowHandle<'_>, HandleError> {
        // SAFETY: the DAW supplies the parent for the entire attachment.
        Ok(unsafe { WindowHandle::borrow_raw(self.0) })
    }
}

#[derive(Clone, Copy)]
enum PathKind {
    Root,
    Model,
    Index,
}

impl PathKind {
    const ALL: [Self; 3] = [Self::Root, Self::Model, Self::Index];

    fn label(self) -> &'static str {
        match self {
            Self::Root => "RVC folder",
            Self::Model => "Voice model",
            Self::Index => "Retrieval index",
        }
    }

    fn extension(self) -> Option<&'static str> {
        match self {
            Self::Root => None,
            Self::Model => Some("pth"),
            Self::Index => Some("index"),
        }
    }

    fn get(self, paths: &ModelPaths) -> &Path {
        match self {
            Self::Root => &paths.rvc_root,
            Self::Model => &paths.model,
            Self::Index => &paths.index,
        }
    }

    fn path_mut(self, paths: &mut ModelPaths) -> &mut PathBuf {
        match self {
            Self::Root => &mut paths.rvc_root,
            Self::Model => &mut paths.model,
            Self::Index => &mut paths.index,
        }
    }

    fn accepts(self, path: &Path) -> bool {
        match self.extension() {
            Some(ext) => path
                .extension()
                .is_some_and(|suffix| suffix.eq_ignore_ascii_case(ext)),
            None => false,
        }
    }

    fn validate(self, root: &Path, path: &Path) -> Result<()> {
        if path.as_os_str().is_empty() {
            return Ok(());
        }
        let Some(extension) = self.extension() else {
            ensure!(
                path.is_dir(),
                "RVC folder does not exist: {}",
                path.display()
            );
            return Ok(());
        };
        ensure!(
            self.accepts(path),
            "{} requires a .{extension} file",
            self.label()
        );
        let resolved = root.join(path);
        let present = resolved.is_file()
            || matches!(self, Self::Model) && root.join("assets/weights").join(path).is_file();
        ensure!(present, "file does not exist: {}", resolved.display());
        Ok(())
    }
}

struct DirectoryEntry {
    path: PathBuf,
    folder: bool,
}

struct Browser {
    kind: PathKind,
    directory: PathBuf,
    entries: Vec<DirectoryEntry>,
    error: Option<String>,
}

impl Browser {
    fn open(kind: PathKind, paths: &ModelPaths) -> Self {
        let root = if paths.rvc_root.is_dir() {
            paths.rvc_root.clone()
        } else {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        };
        let preferred = match kind {
            PathKind::Root => root.clone(),
            PathKind::Model => root.join("assets/weights"),
            PathKind::Index => root.join("assets/indices"),
        };
        let selected = root.join(kind.get(paths));
        let initial = if selected.is_file() {
            selected.parent().map_or(root.clone(), Path::to_path_buf)
        } else if preferred.is_dir() {
            preferred
        } else {
            root
        };
        let mut browser = Self {
            kind,
            directory: PathBuf::new(),
            entries: Vec::new(),
            error: None,
        };
        browser.navigate(initial);
        browser
    }

    fn navigate(&mut self, path: PathBuf) {
        self.directory = path;
        self.entries.clear();
        self.error = None;
        let files = match fs::read_dir(&self.directory) {
            Ok(files) => files,
            Err(error) => {
                self.error = Some(error.to_string());
                return;
            }
        };
        for item in files {
            match item {
                Ok(item) => {
                    let path = item.path();
                    let folder = path.is_dir();
                    if folder || path.is_file() && self.kind.accepts(&path) {
                        self.entries.push(DirectoryEntry { path, folder });
                    }
                }
                Err(error) => self.error = Some(error.to_string()),
            }
        }
        self.entries
            .sort_by(|a, b| b.folder.cmp(&a.folder).then(a.path.cmp(&b.path)));
    }

    fn draw(&mut self, ctx: &egui::Context) -> (bool, Option<PathBuf>) {
        let mut open = true;
        let mut selected = None;
        let mut navigate = None;
        egui::Window::new(format!("Choose {}", self.kind.label()))
            .id(egui::Id::new("rvc-file-browser"))
            .open(&mut open)
            .default_size([570.0, 390.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if let Some(parent) = self.directory.parent()
                        && ui.button("Parent folder").clicked()
                    {
                        navigate = Some(parent.to_path_buf());
                    }
                    if matches!(self.kind, PathKind::Root)
                        && self.directory.is_dir()
                        && ui.button("Use this folder").clicked()
                    {
                        selected = Some(self.directory.clone());
                    }
                });
                ui.label(self.directory.display().to_string());
                if let Some(error) = &self.error {
                    ui.colored_label(Color32::LIGHT_RED, error);
                }
                ui.separator();
                egui::ScrollArea::vertical()
                    .max_height(315.0)
                    .show(ui, |ui| {
                        for entry in &self.entries {
                            let name = entry.path.file_name().unwrap_or_default().to_string_lossy();
                            let label = if entry.folder {
                                format!("{name}/")
                            } else {
                                name.into_owned()
                            };
                            if ui.selectable_label(false, label).clicked() {
                                if entry.folder {
                                    navigate = Some(entry.path.clone());
                                } else {
                                    selected = Some(entry.path.clone());
                                }
                            }
                        }
                        if self.entries.is_empty() && self.error.is_none() {
                            ui.weak("No matching files");
                        }
                    });
            });
        if let Some(next) = navigate {
            self.navigate(next);
        }
        (open, selected)
    }
}

struct Panel {
    model: Arc<EditorModel>,
    paths: ModelPaths,
    fields: [String; 3],
    browser: Option<Browser>,
    error: Option<String>,
}

impl Panel {
    fn new(model: Arc<EditorModel>) -> Self {
        let (paths, _) = model.controls.snapshot();
        let fields = PathKind::ALL.map(|kind| kind.get(&paths).to_string_lossy().into_owned());
        Self {
            model,
            paths,
            fields,
            browser: None,
            error: None,
        }
    }

    fn draw(&mut self, ctx: &egui::Context) {
        let (paths, values) = self.model.controls.snapshot();
        if self.paths != paths {
            self.paths = paths;
            self.fields =
                PathKind::ALL.map(|kind| kind.get(&self.paths).to_string_lossy().into_owned());
        }

        let frame = egui::Frame::central_panel(&ctx.style()).inner_margin(18);
        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.header(ui, values[ENGINE] >= 0.5);
                ui.separator();
                let mut changes = None;
                for (index, kind) in PathKind::ALL.into_iter().enumerate() {
                    ui.label(RichText::new(kind.label()).strong());
                    ui.horizontal(|ui| {
                        let field = ui.add_sized(
                            [(ui.available_width() - 142.0).max(120.0), 26.0],
                            egui::TextEdit::singleline(&mut self.fields[index]),
                        );
                        let submit = field.lost_focus()
                            && ui.input(|input| input.key_pressed(egui::Key::Enter));
                        if ui.button("Browse…").clicked() {
                            self.browser = Some(Browser::open(kind, &self.paths));
                        }
                        if ui.button("Apply").clicked() || submit {
                            changes = Some((kind, PathBuf::from(self.fields[index].trim())));
                        }
                    });
                }

                if let Some(browser) = &mut self.browser {
                    let (keep_open, chosen) = browser.draw(ctx);
                    if let Some(chosen) = chosen {
                        changes = Some((browser.kind, chosen));
                        self.browser = None;
                    } else if !keep_open {
                        self.browser = None;
                    }
                }
                if let Some((kind, path)) = changes {
                    self.error = self
                        .model
                        .update_path(kind, path)
                        .err()
                        .map(|err| err.to_string());
                }
                if let Some(error) = self.error.as_ref() {
                    ui.colored_label(Color32::LIGHT_RED, error);
                } else if let Some(error) = self.model.controls.last_error() {
                    ui.colored_label(Color32::LIGHT_RED, error);
                }

                ui.add_space(12.0);
                ui.separator();
                ui.heading("Voice and timing");
                ui.add_space(8.0);
                let columns = if ui.available_width() >= 650.0 { 3 } else { 2 };
                for row in [1, 2, 3, 4, 5, 10, BLOCK_MS, 7, 8].chunks(columns) {
                    ui.columns(columns, |columns| {
                        for (&index, ui) in row.iter().zip(columns) {
                            self.slider(ui, index, values[index]);
                        }
                    });
                    ui.add_space(10.0);
                }
                ui.columns(columns, |columns| {
                    self.slider(&mut columns[0], 11, values[11]);
                    self.pitch_method(&mut columns[1], values[9]);
                });
            });
        });
    }

    fn header(&self, ui: &mut egui::Ui, enabled: bool) {
        ui.horizontal(|ui| {
            ui.heading(RichText::new("RVC Realtime").size(27.0).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let label = if enabled {
                    "Stop engine"
                } else {
                    "Start engine"
                };
                let button = egui::Button::new(RichText::new(label).color(Color32::WHITE)).fill(
                    if enabled {
                        Color32::from_rgb(86, 65, 74)
                    } else {
                        Color32::from_rgb(35, 93, 87)
                    },
                );
                if ui.add_sized([114.0, 30.0], button).clicked() {
                    self.model.set_plain(ENGINE, f32::from(!enabled));
                }
            });
        });
        let (status, color) = match self.model.controls.status() {
            WorkerStatus::Off => ("Off", Color32::GRAY),
            WorkerStatus::Loading => ("Loading model", Color32::LIGHT_YELLOW),
            WorkerStatus::Ready => ("Ready", Color32::LIGHT_GREEN),
            WorkerStatus::Error => ("Engine error", Color32::LIGHT_RED),
        };
        let (milliseconds, drops) = self.model.controls.performance();
        ui.horizontal(|ui| {
            ui.colored_label(color, status);
            ui.weak(format!(
                "{milliseconds:.1} ms inference · {drops} dropped blocks"
            ));
        });
    }

    fn slider(&self, ui: &mut egui::Ui, index: usize, mut value: f32) {
        let parameter = PARAMETERS[index];
        ui.label(RichText::new(parameter.name).strong());
        ui.spacing_mut().slider_width = (ui.available_width() - 96.0).clamp(90.0, 180.0);
        if ui
            .add(
                egui::Slider::new(&mut value, parameter.range())
                    .step_by(f64::from(parameter.increment()))
                    .suffix(parameter.unit),
            )
            .changed()
        {
            self.model.set_plain(index, value);
        }
    }

    fn pitch_method(&self, ui: &mut egui::Ui, previous: f32) {
        let mut algorithm = previous;
        ui.label(RichText::new("Pitch extraction").strong());
        egui::ComboBox::from_id_salt("rvc-f0-method")
            .selected_text(PITCH_METHODS[pitch_method_index(previous)].1)
            .show_ui(ui, |ui| {
                for (value, name) in PITCH_METHODS {
                    ui.selectable_value(&mut algorithm, value, name);
                }
            });
        if algorithm != previous {
            self.model.set_plain(9, algorithm);
        }
    }
}

struct EditorWindow {
    window: WindowContext,
    gl: GlContext,
    painter: RefCell<Painter>,
    ctx: egui::Context,
    panel: RefCell<Panel>,
    events: RefCell<Vec<egui::Event>>,
    modifiers: Cell<egui::Modifiers>,
    pointer: Cell<Pos2>,
    sink: Option<EditSink>,
}

impl EditorWindow {
    fn new(
        window: WindowContext,
        model: Arc<EditorModel>,
        sink: Option<EditSink>,
    ) -> std::result::Result<Self, HandlerError> {
        let Some(gl) = window.gl_context() else {
            return Err(std::io::Error::other("no OpenGL editor context").into());
        };
        // SAFETY: the window worker uses the GL context only on its GUI thread.
        unsafe { gl.make_current()? };
        let context = unsafe {
            glow::Context::from_loader_function(|symbol| gl.get_proc_address_from_str(symbol))
        };
        let painter = Painter::new(Arc::new(context), "", None, false)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        unsafe { gl.make_not_current()? };
        let ctx = egui::Context::default();
        let mut theme = egui::Visuals::dark();
        theme.panel_fill = Color32::from_rgb(21, 25, 31);
        theme.selection.bg_fill = Color32::from_rgb(40, 104, 99);
        theme.widgets.active.bg_fill = Color32::from_rgb(40, 104, 99);
        theme.widgets.noninteractive.fg_stroke.color = Color32::from_rgb(189, 198, 204);
        ctx.set_visuals(theme);
        ctx.style_mut(|style| {
            style.spacing.item_spacing = Vec2::new(8.0, 5.0);
            style.spacing.button_padding = Vec2::new(7.0, 4.0);
            style
                .text_styles
                .insert(TextStyle::Body, FontId::proportional(15.0));
            style
                .text_styles
                .insert(TextStyle::Button, FontId::proportional(14.0));
        });
        Ok(Self {
            window,
            gl,
            painter: RefCell::new(painter),
            ctx,
            panel: RefCell::new(Panel::new(model)),
            events: RefCell::new(Vec::new()),
            modifiers: Cell::new(egui::Modifiers::default()),
            pointer: Cell::new(Pos2::ZERO),
            sink,
        })
    }

    fn mouse(&self, event: MouseEvent) {
        let mut events = self.events.borrow_mut();
        match event {
            MouseEvent::CursorMoved {
                position,
                modifiers,
            } => {
                self.modifiers.set(egui_modifiers(modifiers));
                let scale = self.window.scale_factor() as f32;
                let position = Pos2::new(position.x as f32 / scale, position.y as f32 / scale);
                self.pointer.set(position);
                events.push(egui::Event::PointerMoved(position));
            }
            MouseEvent::ButtonPressed { button, modifiers }
            | MouseEvent::ButtonReleased { button, modifiers } => {
                let modifiers = egui_modifiers(modifiers);
                self.modifiers.set(modifiers);
                let button = match button {
                    MouseButton::Left => PointerButton::Primary,
                    MouseButton::Right => PointerButton::Secondary,
                    MouseButton::Middle => PointerButton::Middle,
                    MouseButton::Back => PointerButton::Extra1,
                    MouseButton::Forward => PointerButton::Extra2,
                    _ => return,
                };
                let pressed = matches!(event, MouseEvent::ButtonPressed { .. });
                events.push(egui::Event::PointerButton {
                    pos: self.pointer.get(),
                    button,
                    pressed,
                    modifiers,
                });
                if pressed {
                    let _ = self.window.focus();
                }
            }
            MouseEvent::WheelScrolled { delta, modifiers } => {
                let modifiers = egui_modifiers(modifiers);
                self.modifiers.set(modifiers);
                let (unit, x, y) = match delta {
                    ScrollDelta::Lines { x, y } => (egui::MouseWheelUnit::Line, x, y),
                    ScrollDelta::Pixels { x, y } => (egui::MouseWheelUnit::Point, x, y),
                };
                events.push(egui::Event::MouseWheel {
                    unit,
                    delta: Vec2::new(x, y),
                    modifiers,
                });
            }
            MouseEvent::CursorLeft => events.push(egui::Event::PointerGone),
            _ => {}
        }
    }

    fn key(&self, event: keyboard_types::KeyboardEvent) -> EventStatus {
        let modifiers = egui_modifiers(event.modifiers);
        self.modifiers.set(modifiers);
        if !self.ctx.wants_keyboard_input() {
            return EventStatus::Ignored;
        }
        let mut events = self.events.borrow_mut();
        let pressed = event.state == KeyState::Down;
        if let Some(key) = egui_key(&event.key) {
            events.push(egui::Event::Key {
                key,
                physical_key: None,
                pressed,
                repeat: event.repeat,
                modifiers,
            });
        }
        if let NativeKey::Character(text) = event.key
            && pressed
            && !modifiers.ctrl
            && !modifiers.command
            && !text.chars().any(char::is_control)
        {
            events.push(egui::Event::Text(text));
        }
        EventStatus::Captured
    }
}

impl WindowHandler for EditorWindow {
    fn poll(&self) {
        if self.panel.borrow().model.closing.load(Ordering::Acquire) {
            self.window.request_close();
        }
    }

    fn draw(&self) -> std::result::Result<(), HandlerError> {
        let size = self.window.size();
        let scale = self.window.scale_factor() as f32;
        let size: PhysicalSize<u32> = size.physical;
        self.ctx.set_pixels_per_point(scale);
        let mut panel = self.panel.borrow_mut();
        let output = self.ctx.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(
                    Pos2::ZERO,
                    Vec2::new(size.width as f32 / scale, size.height as f32 / scale),
                )),
                events: self.events.take(),
                modifiers: self.modifiers.get(),
                focused: self.window.has_focus(),
                ..Default::default()
            },
            |ctx| panel.draw(ctx),
        );
        for command in &output.platform_output.commands {
            if let egui::OutputCommand::CopyText(text) = command {
                baseview::copy_to_clipboard(text);
            }
        }
        let edits = self.sink.as_ref().map(|_| panel.model.take_edits());
        drop(panel);
        let primitives = self.ctx.tessellate(output.shapes, output.pixels_per_point);
        // SAFETY: GL calls and resource mutation are confined to this window thread.
        unsafe { self.gl.make_current()? };
        let mut painter = self.painter.borrow_mut();
        let physical = [size.width, size.height];
        painter.clear(physical, [0.08, 0.10, 0.12, 1.0]);
        painter.paint_and_update_textures(
            physical,
            output.pixels_per_point,
            &primitives,
            &output.textures_delta,
        );
        let swap = self.gl.swap_buffers();
        unsafe { self.gl.make_not_current()? };
        swap?;
        self.window.request_redraw_after(Duration::from_millis(100));
        if let (Some(sink), Some(edits)) = (&self.sink, edits) {
            sink(edits);
        }
        Ok(())
    }

    fn resized(&self, _: WindowSize) -> std::result::Result<(), HandlerError> {
        self.window.request_redraw();
        Ok(())
    }

    fn on_event(&self, event: Event) -> EventStatus {
        let status = match event {
            Event::Mouse(event) => {
                self.mouse(event);
                EventStatus::Captured
            }
            Event::Keyboard(event) => self.key(event),
            Event::Window(WindowEvent::WillClose) => EventStatus::Ignored,
            _ => EventStatus::Ignored,
        };
        self.window.request_redraw();
        status
    }
}

impl Drop for EditorWindow {
    fn drop(&mut self) {
        // SAFETY: destruction runs on the GUI thread before the GL window is released.
        if unsafe { self.gl.make_current() }.is_ok() {
            self.painter.get_mut().destroy();
            let _ = unsafe { self.gl.make_not_current() };
        }
    }
}

fn egui_modifiers(input: Modifiers) -> egui::Modifiers {
    let ctrl = input.contains(Modifiers::CONTROL);
    egui::Modifiers {
        alt: input.contains(Modifiers::ALT),
        ctrl,
        shift: input.contains(Modifiers::SHIFT),
        command: ctrl,
        mac_cmd: false,
    }
}

fn egui_key(key: &NativeKey) -> Option<egui::Key> {
    use egui::Key;
    match key {
        NativeKey::Named(NamedKey::ArrowDown) => Some(Key::ArrowDown),
        NativeKey::Named(NamedKey::ArrowLeft) => Some(Key::ArrowLeft),
        NativeKey::Named(NamedKey::ArrowRight) => Some(Key::ArrowRight),
        NativeKey::Named(NamedKey::ArrowUp) => Some(Key::ArrowUp),
        NativeKey::Named(NamedKey::Backspace) => Some(Key::Backspace),
        NativeKey::Named(NamedKey::Delete) => Some(Key::Delete),
        NativeKey::Named(NamedKey::Enter) => Some(Key::Enter),
        NativeKey::Named(NamedKey::Escape) => Some(Key::Escape),
        NativeKey::Named(NamedKey::Home) => Some(Key::Home),
        NativeKey::Named(NamedKey::End) => Some(Key::End),
        NativeKey::Named(NamedKey::Tab) => Some(Key::Tab),
        NativeKey::Character(character) => Key::from_name(character),
        _ => None,
    }
}
