# SPDX-License-Identifier: Apache-2.0
"""Compare identical Drama3 codec codes without stochastic service generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sglang_omni.models.fishaudio_drama3.channels_last_decoder import (
    ChannelsLastDecoder,
)
from sglang_omni.models.fishaudio_drama3.stateful_decoder import StatefulDACDecoder
from sglang_omni.models.fishaudio_drama3.stateful_graphs import StatefulGraphDecoder
from sglang_omni.models.fishaudio_drama3.vocoder import WindowGraphDecoder
from sglang_omni.models.fishaudio_s2_pro.stages import load_codec
from sglang_omni.scheduling.audio_stream import StreamVocoderState, decode_stream_window


def frame_partitions(frame_count: int) -> list[int]:
    if frame_count < 1:
        raise ValueError("Codes must contain at least one frame")
    else:
        pass
    counts: list[int] = []
    remaining = frame_count
    for count in (10, 10, 12, 32):
        emitted = min(count, remaining)
        if emitted:
            counts.append(emitted)
            remaining -= emitted
        else:
            break
    while remaining:
        emitted = min(64, remaining)
        counts.append(emitted)
        remaining -= emitted
    return counts


def read_codes(path: Path) -> torch.Tensor:
    codes = np.load(path, allow_pickle=False)
    if codes.ndim != 2 or codes.dtype.kind not in "iu" or codes.shape[1] < 1:
        raise ValueError("Expected an integer NumPy array shaped [codebooks, frames]")
    elif np.any(codes < 0):
        raise ValueError("Codec IDs must be nonnegative")
    else:
        return torch.from_numpy(codes.astype(np.int64, copy=True)).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-path", required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    options = parser.parse_args()
    codes = read_codes(options.codes)
    output = options.output_dir
    output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=repository)
    report = {
        "scope": "Identical-code reconstruction, not service latency or quality noninferiority",
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "codes_sha256": hashlib.sha256(options.codes.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_sources": {
            str(path.relative_to(repository)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (repository / "sglang_omni/models/fishaudio_drama3").glob(
                "*.py"
            )
        },
        "codec_path": options.codec_path,
        "device": options.device,
        "torch": torch.__version__,
        "frames": codes.shape[-1],
        "partitions": frame_partitions(codes.shape[-1]),
        "status": "started",
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    try:
        run_comparison(options.codec_path, options.device, codes, output)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        raise
    else:
        report["status"] = "complete"
        manifest.write_text(json.dumps(report, indent=2) + "\n")


@torch.inference_mode()
def run_comparison(
    codec_path: str, device: str, codes: torch.Tensor, output: Path
) -> None:
    codec = load_codec(codec_path, device)
    codec.fold_weight_norm()
    window_backend = ChannelsLastDecoder(codec, dtype=torch.float16)
    backend = StatefulDACDecoder(
        codec,
        dtype=torch.float16,
        capacity=1,
        gemm_upsample=True,
        folded_codebooks=True,
        fuse_snake_history=True,
        compile_attention=True,
    )
    codes = codes.to(device)
    validation_state = backend.create_state()
    try:
        backend.validate_delta(codes[..., :1], validation_state)
    finally:
        backend.release_state(validation_state)
    graph = StatefulGraphDecoder(backend)
    window = WindowGraphDecoder(
        window_backend,
        num_codebooks=backend.num_codebooks,
        frame_length=codec.frame_length,
        device=torch.device(device),
        window_lengths=(10, 20, 32, 64, 96),
    )
    native = codec.from_indices(codes.clone()).float().cpu().flatten()
    state = graph.create_state()
    stream = StreamVocoderState()
    position = 0
    window_chunks: list[torch.Tensor] = []
    stateful_chunks: list[torch.Tensor] = []
    try:
        for count in frame_partitions(codes.shape[-1]):
            delta = codes[..., position : position + count]
            stateful_chunks.append(
                graph.decode_delta(delta, state).float().cpu().flatten()
            )
            stream.codes.append(
                torch.cat((torch.zeros_like(delta[0, :1]), delta[0]), dim=0)
            )
            stream.total_tokens += count
            position += count
            chunk = decode_stream_window(
                stream,
                codec=codec,
                device=torch.device(device),
                stream_overlap_tokens=32,
                stream_crossfade_samples=0,
                is_final=position == codes.shape[-1],
                decode_window=window,
            )
            assert chunk is not None
            window_chunks.append(
                torch.from_numpy(
                    np.frombuffer(chunk["audio_waveform"], dtype=np.float32).copy()
                )
            )
        graph.decode_delta(codes[..., :0], state, is_final=True)
    finally:
        graph.release_state(state)
    waves = {
        "window": torch.cat(window_chunks),
        "stateful": torch.cat(stateful_chunks),
        "native": native,
    }
    results: dict[str, dict[str, float | int | str]] = {}
    for name, waveform in waves.items():
        assert waveform.numel() == codes.shape[-1] * codec.frame_length
        assert torch.isfinite(waveform).all()
        path = output / f"{name}.wav"
        sf.write(path, waveform.numpy(), codec.sample_rate, subtype="FLOAT")
        results[name] = {
            "samples": waveform.numel(),
            "maximum_error_vs_native": (waveform - native).abs().max().item(),
            "rms_error_vs_native": (waveform - native).square().mean().sqrt().item(),
            "waveform_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    torch.testing.assert_close(waves["stateful"], native, atol=0.02, rtol=0.02)


if __name__ == "__main__":
    main()
