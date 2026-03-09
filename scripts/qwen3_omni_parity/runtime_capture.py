# SPDX-License-Identifier: Apache-2.0
"""Capture one canonical runtime speech-parity run."""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.qwen3_omni_parity.common import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OUT_DIR,
    DEFAULT_PROMPT,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEAKER,
    add_repo_root_to_syspath,
    apply_env_overrides,
    collect_runtime_artifacts,
    extract_audio_waveform,
    file_sha256,
    find_base_port,
    parse_env_overrides,
    save_json,
)

add_repo_root_to_syspath()

from sglang_omni.client.audio import encode_wav
from sglang_omni.config.mp_runner import MultiProcessPipelineRunner
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage


def build_config(model_path: str) -> Qwen3OmniSpeechPipelineConfig:
    cfg = Qwen3OmniSpeechPipelineConfig(
        model_path=model_path,
        relay_backend="shm",
        name="qwen3_omni_parity_runtime_capture",
    )
    cfg.endpoints.scheme = "tcp"
    cfg.endpoints.base_port = find_base_port()
    cfg.gpu_placement.update(
        {
            "thinker": 0,
            "talker_ar": 1,
            "code_predictor": 2,
            "code2wav": 3,
        }
    )
    for stage in cfg.stages:
        if stage.name == "image_encoder":
            stage.executor.args["device"] = "cuda:4"
            stage.relay.device = "cuda:4"
        elif stage.name == "audio_encoder":
            stage.executor.args["device"] = "cuda:5"
            stage.relay.device = "cuda:5"
        elif stage.name == "thinker":
            stage.executor.args["thinker_max_seq_len"] = 768
            stage.executor.args.setdefault("server_args_overrides", {})[
                "mem_fraction_static"
            ] = 0.72
            stage.relay.device = "cuda:0"
        elif stage.name == "talker_ar":
            stage.executor.args["talker_max_seq_len"] = 384
            stage.executor.args.setdefault("server_args_overrides", {})[
                "mem_fraction_static"
            ] = 0.72
            stage.relay.device = "cuda:1"
        elif stage.name == "code_predictor":
            stage.executor.args["code_predictor_max_seq_len"] = 64
            stage.relay.device = "cuda:2"
        elif stage.name == "code2wav":
            stage.executor.args["device"] = "cuda:3"
            stage.relay.device = "cuda:3"
    return cfg


async def run_capture(args: argparse.Namespace) -> dict:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    cfg = build_config(str(args.model_path))
    runner = MultiProcessPipelineRunner(cfg)
    started = False
    audio_chunks: list[np.ndarray] = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        await runner.start(timeout=args.start_timeout_sec)
        started = True

        request_id = args.request_id or f"runtime-capture-{int(time.time())}"
        request = OmniRequest(
            inputs=[{"role": "user", "content": args.prompt}],
            params={
                "stream": True,
                "max_new_tokens": args.thinker_max_new_tokens,
                "temperature": 0.0,
                "speaker": args.speaker,
                "talker_max_new_tokens": args.talker_max_new_tokens,
                "talker_temperature": 0.0,
                "talker_top_k": 1,
                "talker_top_p": 1.0,
                "talker_repetition_penalty": args.talker_repetition_penalty,
            },
            metadata={"output_modalities": ["text", "audio"]},
        )

        runtime_token_ids: list[int] = []
        delta_text_parts: list[str] = []
        final_text_from_event = ""
        decode_complete_text = ""

        async for msg in runner.coordinator.stream(request_id, request):
            if isinstance(msg, StreamMessage):
                if msg.from_stage == "thinker":
                    chunk = msg.chunk if isinstance(msg.chunk, dict) else {}
                    token_id = chunk.get("token_id")
                    if token_id is not None:
                        runtime_token_ids.append(int(token_id))
                    for event in chunk.get("events", []):
                        if not isinstance(event, dict):
                            continue
                        payload = event.get("payload") or {}
                        if event.get("modality") != "text":
                            continue
                        text = payload.get("text") or ""
                        if not text:
                            continue
                        if event.get("is_final"):
                            final_text_from_event = text
                        elif event.get("type") == "text_delta":
                            delta_text_parts.append(text)
                elif msg.from_stage == "code2wav":
                    chunk = msg.chunk if isinstance(msg.chunk, dict) else {}
                    audio = extract_audio_waveform(chunk)
                    if audio is not None and audio.size:
                        audio_chunks.append(audio)
            elif isinstance(msg, CompleteMessage) and msg.from_stage == "decode":
                result = msg.result if isinstance(msg.result, dict) else {}
                text = result.get("text")
                if isinstance(text, str):
                    decode_complete_text = text

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks).astype(np.float32, copy=False)
        else:
            full_audio = np.zeros((0,), dtype=np.float32)

        wav_path = out_dir / f"runtime_capture_{request_id}.wav"
        wav_path.write_bytes(encode_wav(full_audio, sample_rate=DEFAULT_SAMPLE_RATE))
        artifacts = collect_runtime_artifacts(request_id, out_dir=out_dir)

        summary = {
            "request_id": request_id,
            "prompt": args.prompt,
            "speaker": args.speaker,
            "thinker_generated_ids": runtime_token_ids,
            "runtime_delta_text": "".join(delta_text_parts),
            "runtime_final_text_from_event": final_text_from_event,
            "runtime_decode_complete_text": decode_complete_text,
            "audio_path": str(wav_path),
            "audio_sha256": file_sha256(wav_path),
            "num_samples": int(full_audio.shape[0]),
            "duration_sec": float(full_audio.shape[0] / DEFAULT_SAMPLE_RATE),
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "max_abs": float(np.abs(full_audio).max()) if full_audio.size else 0.0,
            "rms": float(np.sqrt(np.mean(np.square(full_audio)))) if full_audio.size else 0.0,
            "artifacts": artifacts,
        }
        save_json(summary, out_dir / f"runtime_capture_{request_id}.json")
        return summary
    finally:
        if started:
            await runner.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--thinker-max-new-tokens", type=int, default=48)
    parser.add_argument("--talker-max-new-tokens", type=int, default=256)
    parser.add_argument("--talker-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--start-timeout-sec", type=float, default=1800.0)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra env override in KEY=VALUE form. May be repeated.",
    )
    args = parser.parse_args()

    apply_env_overrides(parse_env_overrides(args.env))
    summary = asyncio.run(run_capture(args))
    print(summary["request_id"])


if __name__ == "__main__":
    main()
