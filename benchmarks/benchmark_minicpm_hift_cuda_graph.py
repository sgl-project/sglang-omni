# SPDX-License-Identifier: Apache-2.0
"""Compare checkpoint HiFT eager, compiled and CUDA-graph decode."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sglang_omni.models.minicpm_o.components.token2wav.hift import HiFTGenerator
from sglang_omni.models.minicpm_o.components.token2wav.hift_cuda_graph import (
    HiftBodyCudaGraphRunner,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", default="1:64,1:100,1:237,1:400,2:100,4:100")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    arguments = parser.parse_args()
    shapes = tuple(
        tuple(int(dimension) for dimension in shape.split(":"))
        for shape in arguments.shapes.split(",")
    )
    if arguments.iterations < 2 or any(len(shape) != 2 for shape in shapes):
        raise ValueError("Use at least two iterations and batch:frames shape pairs")
    else:
        pass
    torch.manual_seed(7)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = arguments.deterministic
    device = torch.device("cuda", 0)
    hift = HiFTGenerator().to(device).eval()
    weights = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    hift.load_state_dict(
        {name.removeprefix("generator."): value for name, value in weights.items()},
        strict=True,
    )
    eager_body = hift.decode_body
    compiled_body = torch.compile(
        eager_body,
        dynamic=True,
        fullgraph=True,
        options={"triton.cudagraphs": False},
    )
    inputs = {
        shape: (
            torch.randn(shape[0], 80, shape[1], device=device),
            torch.randn(
                shape[0], 1, shape[1] * int(hift.f0_upsamp.scale_factor), device=device
            )
            * 0.01,
        )
        for shape in shapes
    }
    runners: dict[str, HiftBodyCudaGraphRunner] = {}
    startup_seconds: dict[str, float] = {}
    reserved_bytes: dict[str, int] = {}
    rows: list[dict[str, str | int | float]] = []
    for mode, body in (("eager_graph", eager_body), ("compiled_graph", compiled_body)):
        runner = HiftBodyCudaGraphRunner(body)
        torch.cuda.synchronize()
        started = time.perf_counter()
        memory_before = torch.cuda.memory_reserved()
        runner.capture(
            shapes,
            device=device,
            mel_channels=80,
            samples_per_frame=int(hift.f0_upsamp.scale_factor),
            fft_samples=hift.istft_params["n_fft"],
            hop_samples=hift.istft_params["hop_len"],
        )
        torch.cuda.synchronize()
        startup_seconds[mode] = time.perf_counter() - started
        reserved_bytes[mode] = torch.cuda.memory_reserved() - memory_before
        runners[mode] = runner
    with torch.inference_mode():
        for (batch_size, mel_frames), (mel, source) in inputs.items():
            hift.decode_body = eager_body
            hift.body_graph_runner = None
            reference = hift.decode(mel, source).clone()
            modes = ("eager", "compiled", "eager_graph", "compiled_graph")
            for mode in reversed(modes) if arguments.reverse else modes:
                hift.decode_body = compiled_body if "compiled" in mode else eager_body
                hift.body_graph_runner = None
                expected = hift.decode(mel, source).clone()
                hift.body_graph_runner = runners.get(mode)
                for _ in range(5):
                    actual = hift.decode(mel, source)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-4,
                    atol=1e-5,
                    msg=f"{mode}, batch={batch_size}, frames={mel_frames}",
                )
                relative_error = (
                    (actual - reference).norm() / reference.norm().clamp_min(1e-12)
                ).item()
                if not torch.isfinite(actual).all() or relative_error >= 1e-2:
                    raise AssertionError(f"{mode} waveform error: {relative_error}")
                else:
                    pass
                wall_milliseconds: list[float] = []
                device_milliseconds: list[float] = []
                for _ in range(arguments.iterations):
                    torch.cuda.synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    started = time.perf_counter()
                    start_event.record()
                    hift.decode(mel, source)
                    end_event.record()
                    end_event.synchronize()
                    wall_milliseconds.append((time.perf_counter() - started) * 1000)
                    device_milliseconds.append(start_event.elapsed_time(end_event))
                rows.append(
                    {
                        "mode": mode,
                        "batch_size": batch_size,
                        "mel_frames": mel_frames,
                        "wall_median_ms": statistics.median(wall_milliseconds),
                        "device_median_ms": statistics.median(device_milliseconds),
                        "relative_waveform_error": relative_error,
                        "same_body_max_absolute_error": (actual - expected)
                        .abs()
                        .max()
                        .item(),
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
    arguments.output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "iterations": arguments.iterations,
                "deterministic": arguments.deterministic,
                "reverse": arguments.reverse,
                "capture_seconds": startup_seconds,
                "additional_reserved_bytes": reserved_bytes,
                "input_kind": "seeded synthetic mel and shared excitation; checkpoint weights",
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
