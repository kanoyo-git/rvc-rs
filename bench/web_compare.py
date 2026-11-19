"""Measure cold and warm conversion through the Gradio HTTP API on CPU."""

import argparse
import base64
import io
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import tempfile
import time
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener
import wave

from compare import command_environment, sha256, wait_usage


BASELINE_LAUNCH = """
import os
import gradio
import gradio.utils

launch = gradio.Blocks.launch
def loopback(app, **_):
    return launch(app, server_name="127.0.0.1",
                  server_port=int(os.environ["RVC_BENCHMARK_PORT"]),
                  share=False, inbrowser=False)
gradio.Blocks.launch = loopback
gradio.utils.version_check = lambda: None
import webui
"""
HTTP = build_opener(ProxyHandler({}))


def ready_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}", str(port)


def read(url, body=None, timeout=60):
    request = Request(url)
    if body is not None:
        request.data = json.dumps(body).encode()
        request.add_header("Content-Type", "application/json")
    with HTTP.open(request, timeout=timeout) as response:
        return response.read()


def predict(url, function, data):
    body = {"data": data, "session_hash": "rvc-web-benchmark"}
    return json.loads(read(f"{url}/api/{function}", body))["data"]


def measure(url, model, audio, calls):
    results = []
    timings = []
    start = time.perf_counter()
    predict(url, "infer_change_voice", [model, 0.33, 0.33])
    for _ in range(calls):
        data = predict(url, "infer_convert", [0, None, audio, 0, "rmvpe", "", 0, 0, 0.25, 0.33])
        if len(data) != 2 or not data[1] or not data[1]["is_file"]:
            raise RuntimeError(f"conversion failed: {data[0]}")
        name = quote(data[1]["name"], safe="/")
        output = read(f"{url}/file={name}")
        stop = time.perf_counter()
        timings.append(stop - start)
        with wave.open(io.BytesIO(output), "rb") as wav:
            results.append([wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()])
        start = time.perf_counter()
    return timings, results


def trial(args, language, environment, audio):
    with tempfile.TemporaryDirectory(prefix="rvc-web-bench-") as temporary:
        root = Path(temporary)
        for name in ("configs", "docs", "i18n"):
            (root / name).symlink_to(args.baseline / name, target_is_directory=True)
        for name in ("logs", "assets/weights", "assets/pretrained", "assets/pretrained_v2", "tmp"):
            (root / name).mkdir(parents=True)
        for name in ("hubert_base", "indices", "pymss_weights", "rmvpe"):
            (root / "assets" / name).symlink_to(args.baseline / "assets" / name, target_is_directory=True)
        (root / "assets/weights" / args.model.name).symlink_to(args.model)
        url, port = ready_url()
        environment = dict(environment, RVC_BENCHMARK_PORT=port,
                           TMPDIR=str(root / "tmp"), GRADIO_ANALYTICS_ENABLED="False",
                           no_proxy="localhost,127.0.0.1", NO_PROXY="localhost,127.0.0.1")
        environment.update(weight_root=str(root / "assets/weights"), index_root=str(root / "logs"))
        if language == "Python":
            python_path = os.environ.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(args.baseline), python_path]))
            command = [args.python, "-u", "-c", BASELINE_LAUNCH]
        else:
            command = [args.rust_binary, "serve", "--port", port]

        with (root / "server.log").open("w+b") as log:
            start = time.perf_counter()
            child = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = start + 60
                while True:
                    try:
                        config = json.loads(read(f"{url}/config", timeout=1))
                        break
                    except (URLError, TimeoutError):
                        if child.poll() is not None or time.perf_counter() >= deadline:
                            log.seek(0, os.SEEK_END)
                            log.seek(max(0, log.tell() - 4096))
                            raise RuntimeError(f"{language} startup failed\n{log.read().decode(errors='replace')}")
                        time.sleep(0.05)
                startup = time.perf_counter() - start
                timings, geometry = measure(url, args.model.name, audio, args.calls)
            finally:
                if child.returncode is None:
                    child.terminate()
                    try:
                        _, status, usage = wait_usage(child.pid, time.perf_counter() + 15)
                    except TimeoutError:
                        os.killpg(child.pid, signal.SIGKILL)
                        _, status, _ = os.wait4(child.pid, 0)
                        child.returncode = os.waitstatus_to_exitcode(status)
                        raise
                    child.returncode = os.waitstatus_to_exitcode(status)

    return {
        "startup_s": startup,
        "cold_s": timings[0],
        "warm_s": timings[1:],
        "cpu_s": usage.ru_utime + usage.ru_stime,
        "peak_rss_mib": usage.ru_maxrss / 1024,
        "geometry": geometry,
        "dependencies": [{key: item.get(key) for key in ("api_name", "inputs", "outputs", "queue")}
                         for item in config["dependencies"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "python", "rust-binary", "model", "input"):
        parser.add_argument(f"--{name}", required=True, type=lambda value: Path(value).absolute())
    parser.add_argument("--rust-lib-dir", type=lambda value: Path(value).absolute())
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.threads < 1 or args.calls < 2:
        parser.error("runs and threads must be positive, calls at least 2")
    args.device = "cpu"
    environment = command_environment(args, "-1")
    audio = {
        "name": args.input.name, "is_file": False,
        "data": "data:audio/wav;base64," + base64.b64encode(args.input.read_bytes()).decode(),
    }
    results = {"Python": [], "Rust": []}
    for repeat in range(args.runs):
        sequence = list(results) if repeat % 2 == 0 else list(reversed(results))
        for language in sequence:
            results[language].append(trial(args, language, environment[language], audio))
        python, rust = (results[language][-1] for language in results)
        for field in ("geometry", "dependencies"):
            if python[field] != rust[field]:
                raise RuntimeError(f"{field} differs across implementations")
    medians = {}
    for language, trials in results.items():
        medians[language] = {
            key: statistics.median(run[key] for run in trials)
            for key in ("startup_s", "cold_s", "cpu_s", "peak_rss_mib")
        }
        medians[language]["warm_s"] = statistics.median(value for run in trials for value in run["warm_s"])
    print(json.dumps(medians, indent=2))
    if args.json:
        args.json.write_text(json.dumps({
            "input_sha256": sha256(args.input),
            "model_sha256": sha256(args.model),
            "cpus": sorted(os.sched_getaffinity(0)),
            "threads": args.threads,
            "medians": medians,
            "results": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
