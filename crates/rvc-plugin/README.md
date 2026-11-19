# RVC Realtime plug-in (Rust)

The shared library exports VST2 and VST3 entry points for x86-64 Linux and Windows. Both use the native Rust RVC engine. The audio callback exchanges bounded PCM blocks with an inference worker. It also handles the dry path, latency report, channel layout, gain and wet mix.

## Build and verify

Build from the workspace root with the same LibTorch setup as the native RVC application.

```sh
cargo build --release -p rvc-plugin
cargo run --release -p rvc-plugin --example abi_smoke -- target/release/librvc_realtime.so
cargo run --release -p rvc-plugin --example package -- target/release/librvc_realtime.so target/plugins
```

On Windows use `target/release/rvc_realtime.dll` for the last two commands and make the LibTorch runtime available to the DAW. The packaging command copies the library to `RVC Realtime.so` or `RVC Realtime.dll` for VST2 and creates the matching `RVCRealtime.vst3` bundle. It leaves additional runtime libraries in the existing LibTorch installation.

## Select a voice

Open the RVC Realtime editor in the DAW. Set **RVC folder** to the workspace that contains your model assets. Set **Voice model** to a `.pth` checkpoint and, optionally, **Retrieval index** to a matching `.index` file. Use **Browse** to select the paths, or type each path and click **Apply**. A model name under `assets/weights` also works.

Click **Start engine** after selecting the model. The editor reports the loading state, inference time and dropped blocks. It offers the host-automatable pitch, formant, index, RMS mix, noise gate, wet mix, block, crossfade, context and output controls, plus RMVPE, FCPE and PM for pitch extraction. The DAW saves the parameters and paths in its session state. Detailed loading and inference failures also go to the host process's standard error stream.

The editor requires an OpenGL-capable X11 or XWayland host on Linux and an HWND host on Windows. The host's built-in parameter panel is also supported. For sessions configured without the editor, set `RVC_PLUGIN_ROOT`, `RVC_PLUGIN_MODEL` and optionally `RVC_PLUGIN_INDEX` in the DAW environment. The Windows build reads the original `%LOCALAPPDATA%\RVCRealtime\settings.ini` and saves the last successfully loaded paths there. `RVC_PLUGIN_DEVICE` overrides automatic compute-device selection.

## Host tests

The ABI smoke test loads the built library and exercises both formats. Pass an audio file after the library path to verify non-silent converted audio with the configured model. The unit tests drive the worker callbacks with deterministic processors.

On Linux, run the editor harness with `DISPLAY` pointing to a test X11 server with GLX and XTEST. It embeds both editor formats, verifies rendering and an automation edit, resizes VST3 and checks that both windows detach. Pass a capture folder to save PPM screenshots. An optional third argument points to a test model file in an otherwise empty `$RVC_PLUGIN_ROOT/assets/weights` directory and checks browsing and session-state storage.

```sh
cargo run -p rvc-plugin --example editor_smoke -- target/debug/librvc_realtime.so
```
