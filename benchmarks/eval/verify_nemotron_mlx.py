# SPDX-License-Identifier: Apache-2.0
"""Real-weight Torch CPU / Apple backend token parity followed by real HTTP/SSE serving checks.

Run with the Apple environment from the repository root:
  python -m benchmarks.eval.verify_nemotron_mlx --model-path /path/to/model \
      --output /tmp/nemotron-validation.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import torch


def verify_streaming(reference, native, audio_paths):
    """Compare real batched cache-aware decoding on CPU and MPS, chunk by chunk."""
    from sglang_omni.models.nemotron3_5_asr.streaming import (
        Nemotron3_5ASRStreamingChunkSpec,
        Nemotron3_5ASRStreamState,
    )
    from sglang_omni.utils.audio import load_audio

    rows = []
    for lookahead in (0, 3, 6, 13):
        for runner in (reference, native):
            runner.processor.set_num_lookahead_tokens(lookahead)
        for language in ("en-US", "auto"):
            streams = []
            for path in audio_paths:
                audio = load_audio(
                    str(path),
                    source_name="MPS streaming validation",
                    target_sample_rate=16000,
                )
                state = Nemotron3_5ASRStreamState(
                    request_id=path.name,
                    payload=None,
                    language=language,
                    spec=Nemotron3_5ASRStreamingChunkSpec(
                        **native.streaming_chunk_spec
                    ),
                    decode=native.new_streaming_decode_state(),
                )
                pcm = torch.from_numpy((np.clip(audio, -1, 1) * 32767).astype(np.int16))
                state.append_pcm16(pcm, {"sample_rate": 16000})
                state.mark_done()
                streams.append(state)
            cpu_states = [reference.new_streaming_decode_state() for _ in streams]
            chunks_checked = 0
            while True:
                active = [
                    i
                    for i, state in enumerate(streams)
                    if state.has_ready_window(finalizing=True)
                ]
                if not active:
                    break
                windows = [streams[i].pop_ready_window(finalizing=True) for i in active]
                for runner, states in (
                    (reference, [cpu_states[i] for i in active]),
                    (native, [streams[i].decode for i in active]),
                ):
                    chunks = [
                        runner.prepare_streaming_chunk(
                            w.waveform, language=language, is_first=w.is_first
                        )
                        for w in windows
                    ]
                    runner.run_streaming_batch(
                        states, chunks, requested_languages=[language] * len(active)
                    )
                for i in active:
                    assert (
                        streams[i].decode.tokens == cpu_states[i].tokens
                    ), f"Streaming token mismatch: {audio_paths[i].name} {language} lookahead={lookahead}"
                    assert streams[i].decode.durations == cpu_states[i].durations
                    chunks_checked += 1
            rows.append(
                dict(
                    lookahead=lookahead,
                    language=language,
                    streams=len(streams),
                    chunks_checked=chunks_checked,
                )
            )
            print("Streaming parity: " + json.dumps(rows[-1]), flush=True)
    return rows


def verify_components(model_path, audio_paths, backend="mlx"):
    from sglang_omni.models.nemotron3_5_asr.model_runner import (
        Nemotron3_5ASRModelRunner,
    )
    from sglang_omni.serve.transcription_adapters.nemotron3_5_asr import (
        Nemotron3_5ASRTranscriptionAdapter,
    )
    from sglang_omni.utils.audio import load_audio

    torch.set_num_threads(4)
    reference = Nemotron3_5ASRModelRunner(model_path, device="cpu")
    if backend == "mlx":
        import mlx.core as mx

        from sglang_omni.models.nemotron3_5_asr.mlx.runner import (
            Nemotron3_5ASRMLXRunner,
        )

        native = Nemotron3_5ASRMLXRunner(model_path)
    else:
        assert torch.backends.mps.is_available(), "MPS is not available"
        native = Nemotron3_5ASRModelRunner(model_path, device="mps")
        assert all(p.device.type == "mps" for p in native.model.parameters())
    adapter = Nemotron3_5ASRTranscriptionAdapter()
    results = []
    for lookahead in (0, 3, 6, 13):
        native.processor.set_num_lookahead_tokens(lookahead)
        for path in audio_paths:
            audio = load_audio(
                str(path), source_name="Nemotron validation", target_sample_rate=16000
            )
            for language in ("en-US", "auto"):
                inputs = dict(
                    native.processor(
                        audio,
                        sampling_rate=16000,
                        language=language,
                        return_tensors="pt",
                    )
                )
                started = time.perf_counter()
                expected = reference._generate_sequences(inputs, max_new_tokens=None)[
                    0
                ].tolist()
                torch_s = time.perf_counter() - started
                started = time.perf_counter()
                actual = native._generate_sequences(inputs, max_new_tokens=None)[0]
                backend_s = time.perf_counter() - started
                if isinstance(actual, torch.Tensor):
                    actual = actual.tolist()
                assert (
                    actual == expected
                ), f"Token mismatch: {path.name} {language} lookahead={lookahead}"
                raw = native.processor.decode(actual, skip_special_tokens=False)
                text = adapter.postprocess_text(raw)
                assert text, f"Empty speech transcript: {path}"
                row = dict(
                    audio=path.name,
                    language=language,
                    lookahead=lookahead,
                    tokens=actual,
                    text=text,
                    raw_text=raw,
                    torch_s=torch_s,
                    **{f"{backend}_s": backend_s},
                )
                results.append(row)
                print(
                    json.dumps({k: v for k, v in row.items() if k != "tokens"}),
                    flush=True,
                )
    streaming = (
        verify_streaming(reference, native, audio_paths) if backend == "mps" else []
    )
    peak = mx.get_peak_memory() if backend == "mlx" else None
    reference.close()
    native.close()
    del reference, native
    gc.collect()
    return results, peak, streaming


def verify_http(model_path, audio_paths, component_results, output, backend="mlx"):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(
        os.environ,
        SGLANG_USE_MLX="1" if backend == "mlx" else "0",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
    )
    env["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/ffmpeg@7/lib" + (
        ":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else ""
    )
    server_log = output.with_suffix(".server.log")
    rows = []
    output.with_suffix(".http.jsonl").write_text("")
    started = time.perf_counter()
    with server_log.open("w") as log:
        proc = subprocess.Popen(
            [
                str(Path(sys.executable).parent / "sgl-omni"),
                "serve",
                "--model-path",
                model_path,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            with httpx.Client(base_url=base, timeout=180) as client:
                deadline = time.monotonic() + 180
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Server exited; see {server_log}")
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Server readiness timed out; see {server_log}"
                        )
                    time.sleep(0.5)
                startup_s = time.perf_counter() - started

                def request(audio, **params):
                    data = dict(
                        model=model_path, language="en-US", response_format="json"
                    )
                    data.update(params)
                    tick = time.perf_counter()
                    response = client.post(
                        "/v1/audio/transcriptions",
                        data=data,
                        files={"file": ("audio.wav", audio, "audio/wav")},
                    )
                    row = dict(
                        params=data,
                        status=response.status_code,
                        elapsed_s=time.perf_counter() - tick,
                        body=response.text,
                    )
                    rows.append(row)
                    with output.with_suffix(".http.jsonl").open("a") as records:
                        records.write(json.dumps(row) + "\n")
                    return response

                expected = {
                    (row["audio"], row["language"]): row["text"]
                    for row in component_results
                    if row["lookahead"] == 3
                }
                for path in audio_paths:
                    audio = path.read_bytes()
                    for language in ("en-US", "auto"):
                        for fmt in ("json", "text", "verbose_json"):
                            response = request(
                                audio, language=language, response_format=fmt
                            )
                            response.raise_for_status()
                            text = (
                                response.text
                                if fmt == "text"
                                else response.json()["text"]
                            )
                            assert text.strip() == expected[path.name, language]
                            if fmt == "verbose_json":
                                assert response.json()["language"] == "en-US"
                    response = request(audio, stream="true")
                    response.raise_for_status()
                    events = [
                        json.loads(line[6:])
                        for line in response.text.splitlines()
                        if line.startswith("data: ") and line != "data: [DONE]"
                    ]
                    text = "".join(
                        e.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        for e in events
                    )
                    # The endpoint may use transcription delta events instead of chat deltas.
                    if not text:
                        text = "".join(
                            e.get("delta", "")
                            for e in events
                            if e.get("type") == "transcript.text.delta"
                        )
                    done = [
                        e for e in events if e.get("type") == "transcript.text.done"
                    ]
                    assert (
                        len(done) == 1
                        and done[0]["text"] == expected[path.name, "en-US"]
                    ), events
                    assert "data: [DONE]" in response.text
                    if text:
                        assert text.strip() == expected[path.name, "en-US"], events

                # Queued peers and repeat requests must not inherit predictor state.
                def concurrent_request(path):
                    response = request(path.read_bytes())
                    response.raise_for_status()
                    assert response.json()["text"] == expected[path.name, "en-US"]

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(concurrent_request, audio_paths * 2))

                for params in (
                    {"language": "xx-XX"},
                    {"temperature": "0.5"},
                    {"prompt": "context"},
                ):
                    assert (
                        request(audio_paths[0].read_bytes(), **params).status_code
                        == 400
                    )
                assert request(b"not audio").status_code == 400
                empty = io.BytesIO()
                sf.write(empty, np.zeros(0, dtype=np.float32), 16000, format="WAV")
                assert request(empty.getvalue()).status_code == 400
                if backend == "mlx":
                    too_long = io.BytesIO()
                    sf.write(
                        too_long,
                        np.zeros(61 * 16000, dtype=np.float32),
                        16000,
                        format="WAV",
                    )
                    assert request(too_long.getvalue()).status_code == 400
                # Recovery after bad requests.
                concurrent_request(audio_paths[0])
                return dict(
                    startup_s=startup_s, requests=rows, server_log=str(server_log)
                )
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("mlx", "mps"), default="mlx")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    audio_paths = [
        root / "tests/data/query_to_cars.wav",
        root / "tests/data/query_to_draw.wav",
    ]
    report = dict(
        hardware=platform.platform(),
        versions={
            name: importlib.metadata.version(name)
            for name in (("mlx",) if args.backend == "mlx" else ())
            + ("torch", "transformers", "sglang")
        },
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        model_path=args.model_path,
        backend=args.backend,
    )
    if args.backend == "mlx":
        import mlx.core as mx

        report["metal_device"] = mx.device_info()
    else:
        report["mps_available"] = torch.backends.mps.is_available()
    report["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "sglang_omni/models/nemotron3_5_asr").rglob("*.py"))
    }
    report["audio_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in audio_paths
    }
    manifest = Path(args.model_path) / "conversion.json"
    if manifest.exists():
        report["conversion"] = json.loads(manifest.read_text())
    try:
        (
            report["components"],
            report[f"{args.backend}_peak_bytes"],
            report["streaming"],
        ) = verify_components(args.model_path, audio_paths, args.backend)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["http"] = verify_http(
            args.model_path,
            audio_paths,
            report["components"],
            args.output,
            args.backend,
        )
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validation passed: {args.output}")


if __name__ == "__main__":
    main()
