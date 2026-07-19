# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from typing import Any

from ._common import (
    LauncherPreset,
    add_mem_fraction,
    add_model_path,
    add_offline_args,
    add_server_args,
    apply_stage_factory_updates,
    parser,
    run_async,
    run_speech_request,
    set_stage_gpu,
    set_stage_tp_size,
    validate_fraction,
)

_TEXT_SERVER_DESCRIPTION = """Launch Qwen3-Omni with text-only OpenAI responses.

Examples:
  python examples/run_omni.py qwen3-text-server \\
      --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000

  curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":256}'
"""

_SPEECH_SERVER_DESCRIPTION = """Launch Qwen3-Omni with speech output.

Each stage runs in its own process with explicit GPU placement.

Examples:
  python examples/run_omni.py qwen3-speech-server

  python examples/run_omni.py qwen3-speech-server \\
      --gpu-thinker 0 --gpu-talker 1 --gpu-code2wav 1

  python examples/run_omni.py qwen3-speech-server \\
      --thinker-tp-size 2 --gpu-thinker-tp 0,1 \\
      --gpu-talker 2 --gpu-code2wav 2
"""

_SPEECH_DESCRIPTION = """Run one Qwen3-Omni text-to-speech request.

Examples:
  python examples/run_omni.py qwen3-speech \\
      --prompt "Tell me about what makes a beautiful sunset."

  python examples/run_omni.py qwen3-speech \\
      --prompt "Hello, how are you?" --gpu-thinker 0 --gpu-talker 1

  python examples/run_omni.py qwen3-speech \\
      --prompt "Read me a bedtime story." --output audio.wav
"""


def _parse_thinker_tp_gpu_list(spec: str, tp_size: int) -> list[int]:
    try:
        gpu_ids = [int(piece.strip()) for piece in spec.split(",") if piece.strip()]
    except ValueError as exc:
        raise ValueError(
            f"--gpu-thinker-tp must be a comma-separated list of integers, got {spec!r}"
        ) from exc
    if any(gpu < 0 for gpu in gpu_ids):
        raise ValueError(f"--gpu-thinker-tp GPU ids must be >= 0, got {gpu_ids}")
    if len(gpu_ids) != tp_size:
        raise ValueError(
            f"--gpu-thinker-tp has {len(gpu_ids)} entries but --thinker-tp-size="
            f"{tp_size} requires exactly {tp_size}"
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpu-thinker-tp must list distinct GPU ids, got {gpu_ids}")
    return gpu_ids


def _resolve_speech_mem_fractions(
    config: Any,
    *,
    global_mem_fraction_static: float | None,
    thinker_mem_fraction_static: float | None,
    talker_mem_fraction_static: float | None,
) -> None:
    values = (
        ("--mem-fraction-static", global_mem_fraction_static),
        ("--thinker-mem-fraction-static", thinker_mem_fraction_static),
        ("--talker-mem-fraction-static", talker_mem_fraction_static),
    )
    for flag_name, value in values:
        validate_fraction(flag_name, value)
    thinker_value = (
        thinker_mem_fraction_static
        if thinker_mem_fraction_static is not None
        else global_mem_fraction_static
    )
    talker_value = (
        talker_mem_fraction_static
        if talker_mem_fraction_static is not None
        else global_mem_fraction_static
    )
    if thinker_value is not None:
        apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates={"mem_fraction_static": thinker_value},
        )
    if talker_value is not None:
        apply_stage_factory_updates(
            config,
            stage_name="talker_ar",
            server_arg_updates={"mem_fraction_static": talker_value},
        )


def _build_qwen_text_server_parser() -> argparse.ArgumentParser:
    target = parser(_TEXT_SERVER_DESCRIPTION)
    add_model_path(target, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    target.add_argument(
        "--thinker-max-seq-len",
        type=int,
        default=None,
        help="Context length for preprocessing and the thinker stage.",
    )
    target.add_argument(
        "--cpu-offload-gb",
        type=int,
        default=0,
        help="GB of thinker weights to offload to CPU.",
    )
    add_mem_fraction(
        target,
        "Set mem_fraction_static for the thinker stage.",
    )
    target.add_argument(
        "--encoder-mem-reserve",
        type=float,
        default=None,
        help=(
            "GPU-memory fraction kept outside SGLang's static pool for the "
            "colocated vision and audio encoders. With neither memory flag, "
            "SGLang auto-selects the pool and reserves 0.05. This flag changes "
            "that reserve; --mem-fraction-static instead pins the pool directly. "
            "Passing both flags is rejected."
        ),
    )
    add_server_args(target, model_name=None)
    target.add_argument(
        "--enable-realtime",
        action="store_true",
        help="Mount the WebSocket /v1/realtime endpoint.",
    )
    return target


def launch_qwen_text_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
    from sglang_omni.serve import launch_server

    if args.mem_fraction_static is not None and args.encoder_mem_reserve is not None:
        raise ValueError(
            "--mem-fraction-static and --encoder-mem-reserve are mutually exclusive"
        )
    validate_fraction("--mem-fraction-static", args.mem_fraction_static)
    if (
        args.encoder_mem_reserve is not None
        and not 0.0 <= args.encoder_mem_reserve < 1.0
    ):
        raise ValueError(
            f"--encoder-mem-reserve must be in [0, 1), got {args.encoder_mem_reserve}"
        )

    config = Qwen3OmniPipelineConfig(model_path=args.model_path)
    thinker_updates: dict[str, object] = {}
    preprocessing_updates: dict[str, object] = {}
    if args.thinker_max_seq_len is not None:
        max_seq_len = int(args.thinker_max_seq_len)
        thinker_updates["thinker_max_seq_len"] = max_seq_len
        preprocessing_updates["thinker_max_seq_len"] = max_seq_len
    if args.encoder_mem_reserve is not None:
        thinker_updates["encoder_mem_reserve"] = args.encoder_mem_reserve

    server_updates: dict[str, object] = {}
    if args.cpu_offload_gb:
        server_updates["cpu_offload_gb"] = int(args.cpu_offload_gb)
    if args.mem_fraction_static is not None:
        server_updates["mem_fraction_static"] = args.mem_fraction_static
    if thinker_updates or server_updates:
        apply_stage_factory_updates(
            config,
            stage_name="thinker",
            updates=thinker_updates,
            server_arg_updates=server_updates,
        )
    if preprocessing_updates:
        apply_stage_factory_updates(
            config,
            stage_name="preprocessing",
            updates=preprocessing_updates,
        )
    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        enable_realtime=args.enable_realtime,
    )


def _add_qwen_speech_mem_args(target: argparse.ArgumentParser) -> None:
    add_mem_fraction(
        target,
        "Set mem_fraction_static for both Qwen AR stages. If omitted, SGLang "
        "chooses automatically.",
    )
    target.add_argument(
        "--thinker-mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Set mem_fraction_static only for the thinker stage. Overrides "
            "--mem-fraction-static for thinker."
        ),
    )
    target.add_argument(
        "--talker-mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Set mem_fraction_static only for the talker stage. Overrides "
            "--mem-fraction-static for talker."
        ),
    )


def _build_qwen_speech_server_parser() -> argparse.ArgumentParser:
    target = parser(_SPEECH_SERVER_DESCRIPTION)
    add_model_path(target, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    target.add_argument("--gpu-thinker", type=int, default=0)
    target.add_argument("--gpu-talker", type=int, default=None)
    target.add_argument("--gpu-code-predictor", type=int, default=None)
    target.add_argument("--gpu-code2wav", type=int, default=None)
    target.add_argument("--gpu-image-encoder", type=int, default=None)
    target.add_argument("--gpu-audio-encoder", type=int, default=None)
    target.add_argument(
        "--thinker-tp-size",
        type=int,
        default=1,
        help=(
            "Tensor-parallel size for the thinker stage. Must be >= 1. When "
            "> 1, also pass --gpu-thinker-tp with exactly that many GPU ids."
        ),
    )
    target.add_argument(
        "--gpu-thinker-tp",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU ids for thinker when --thinker-tp-size > 1. "
            "Length must equal --thinker-tp-size and overrides --gpu-thinker."
        ),
    )
    target.add_argument(
        "--thinker-max-seq-len",
        type=int,
        default=8192,
        help=(
            "Context length for the thinker stage, preprocessing, and talker "
            "context guards."
        ),
    )
    target.add_argument(
        "--talker-max-seq-len",
        type=int,
        default=None,
        help=(
            "Context length for the talker_ar KV pool. Uses the pipeline "
            "default when omitted."
        ),
    )
    _add_qwen_speech_mem_args(target)
    target.add_argument(
        "--enable-partial-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable partial-prefix talker startup. Defaults to on for the "
            "disaggregated topology and off for --colocated."
        ),
    )
    target.add_argument(
        "--partial-start-min-chunks",
        type=int,
        default=5,
        help=(
            "Chunk threshold for partial-start (default 5). When partial-start "
            "is enabled, it must be >= MIN_PARTIAL_START_CHUNKS (3)."
        ),
    )
    target.add_argument(
        "--colocated",
        action="store_true",
        help=(
            "Use the single-GPU colocated topology. All GPU stage flags must "
            "point to the same device."
        ),
    )
    add_server_args(target, model_name="qwen3-omni")
    return target


def launch_qwen_speech_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.qwen3_omni.config import (
        MIN_PARTIAL_START_CHUNKS,
        Qwen3OmniSpeechColocatedPipelineConfig,
        Qwen3OmniSpeechPipelineConfig,
    )
    from sglang_omni.serve import launch_server
    from sglang_omni.utils.gpu_compat import should_disable_custom_all_reduce_for_gpus

    for flag_name, value in (
        ("--mem-fraction-static", args.mem_fraction_static),
        ("--thinker-mem-fraction-static", args.thinker_mem_fraction_static),
        ("--talker-mem-fraction-static", args.talker_mem_fraction_static),
    ):
        validate_fraction(flag_name, value)

    enable_partial_start = (
        not args.colocated
        if args.enable_partial_start is None
        else bool(args.enable_partial_start)
    )
    if (
        enable_partial_start
        and args.partial_start_min_chunks < MIN_PARTIAL_START_CHUNKS
    ):
        raise ValueError(
            f"--partial-start-min-chunks must be >= {MIN_PARTIAL_START_CHUNKS}, "
            f"got {args.partial_start_min_chunks}"
        )

    gpu_talker = args.gpu_talker
    if gpu_talker is None:
        gpu_talker = args.gpu_thinker if args.colocated else 1
    gpu_code2wav = args.gpu_code2wav
    if gpu_code2wav is None:
        gpu_code2wav = args.gpu_thinker if args.colocated else 0
    gpu_image_encoder = args.gpu_image_encoder
    if gpu_image_encoder is None:
        gpu_image_encoder = args.gpu_thinker if args.colocated else 0
    gpu_audio_encoder = args.gpu_audio_encoder
    if gpu_audio_encoder is None:
        gpu_audio_encoder = args.gpu_thinker if args.colocated else 0

    if args.colocated:
        colocated_gpus = {
            "--gpu-thinker": args.gpu_thinker,
            "--gpu-talker": gpu_talker,
            "--gpu-code2wav": gpu_code2wav,
            "--gpu-image-encoder": gpu_image_encoder,
            "--gpu-audio-encoder": gpu_audio_encoder,
        }
        if len(set(colocated_gpus.values())) != 1:
            raise ValueError(
                "--colocated requires all GPU stage flags to use the same GPU, "
                f"got {colocated_gpus}"
            )

    gpu_code_predictor = args.gpu_code_predictor
    if gpu_code_predictor is None:
        gpu_code_predictor = gpu_talker
    if gpu_code_predictor != gpu_talker:
        raise ValueError(
            "Qwen3 speech pipeline does not expose a separate code_predictor "
            "stage. Use the same GPU for --gpu-code-predictor and --gpu-talker."
        )

    config_cls = (
        Qwen3OmniSpeechColocatedPipelineConfig
        if args.colocated
        else Qwen3OmniSpeechPipelineConfig
    )
    config = config_cls(model_path=args.model_path)
    set_stage_gpu(config, "image_encoder", gpu_image_encoder)
    set_stage_gpu(config, "audio_encoder", gpu_audio_encoder)

    if args.thinker_tp_size < 1:
        raise ValueError(f"--thinker-tp-size must be >= 1, got {args.thinker_tp_size}")
    if args.thinker_tp_size > 1:
        if args.gpu_thinker_tp is None:
            raise ValueError(
                "--thinker-tp-size > 1 requires --gpu-thinker-tp "
                "(comma-separated GPU ids, one per TP rank)."
            )
        thinker_gpus = _parse_thinker_tp_gpu_list(
            args.gpu_thinker_tp,
            args.thinker_tp_size,
        )
        set_stage_tp_size(config, "thinker", args.thinker_tp_size)
        set_stage_gpu(config, "thinker", thinker_gpus)
        apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates={
                "disable_custom_all_reduce": should_disable_custom_all_reduce_for_gpus(
                    thinker_gpus
                )
            },
        )
    else:
        if args.gpu_thinker_tp is not None:
            raise ValueError(
                "--gpu-thinker-tp only applies when --thinker-tp-size > 1; "
                "for TP=1, use --gpu-thinker."
            )
        set_stage_gpu(config, "thinker", args.gpu_thinker)

    set_stage_gpu(config, "talker_ar", gpu_talker)
    set_stage_gpu(config, "code2wav", gpu_code2wav)
    _resolve_speech_mem_fractions(
        config,
        global_mem_fraction_static=args.mem_fraction_static,
        thinker_mem_fraction_static=args.thinker_mem_fraction_static,
        talker_mem_fraction_static=args.talker_mem_fraction_static,
    )
    if args.thinker_max_seq_len is not None:
        updates = {"thinker_max_seq_len": int(args.thinker_max_seq_len)}
        apply_stage_factory_updates(config, stage_name="thinker", updates=updates)
        apply_stage_factory_updates(
            config,
            stage_name="preprocessing",
            updates=updates,
        )
    if args.talker_max_seq_len is not None:
        apply_stage_factory_updates(
            config,
            stage_name="talker_ar",
            updates={"talker_max_seq_len": int(args.talker_max_seq_len)},
        )
    partial_start_updates: dict[str, object] = {
        "enable_partial_start": enable_partial_start
    }
    if enable_partial_start:
        partial_start_updates["partial_start_min_chunks"] = int(
            args.partial_start_min_chunks
        )
    apply_stage_factory_updates(
        config,
        stage_name="talker_ar",
        updates=partial_start_updates,
    )
    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


def _build_qwen_speech_parser() -> argparse.ArgumentParser:
    target = parser(_SPEECH_DESCRIPTION)
    add_model_path(target, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    add_offline_args(
        target,
        prompt="Hello! Tell me something interesting.",
        system="You are a friendly assistant. Speak naturally and warmly.",
        max_new_tokens=64,
        output=None,
    )
    target.add_argument("--gpu-thinker", type=int, default=0)
    target.add_argument("--gpu-talker", type=int, default=1)
    target.add_argument("--gpu-code-predictor", type=int, default=None)
    target.add_argument("--gpu-code2wav", type=int, default=0)
    target.add_argument("--gpu-image-encoder", type=int, default=0)
    target.add_argument("--gpu-audio-encoder", type=int, default=0)
    _add_qwen_speech_mem_args(target)
    return target


async def run_qwen_speech(args: argparse.Namespace) -> None:
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

    config = Qwen3OmniSpeechPipelineConfig(model_path=args.model_path)
    if args.gpu_code_predictor not in (None, args.gpu_talker):
        raise ValueError(
            "Qwen3 speech pipeline does not expose a separate code_predictor "
            "stage. Use the same GPU for --gpu-code-predictor and --gpu-talker."
        )
    for stage_name, gpu_id in (
        ("thinker", args.gpu_thinker),
        ("talker_ar", args.gpu_talker),
        ("code2wav", args.gpu_code2wav),
        ("image_encoder", args.gpu_image_encoder),
        ("audio_encoder", args.gpu_audio_encoder),
    ):
        set_stage_gpu(config, stage_name, gpu_id)
    _resolve_speech_mem_fractions(
        config,
        global_mem_fraction_static=args.mem_fraction_static,
        thinker_mem_fraction_static=args.thinker_mem_fraction_static,
        talker_mem_fraction_static=args.talker_mem_fraction_static,
    )
    request = {
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ],
        "images": [],
        "videos": [],
        "audios": [],
    }
    await run_speech_request(
        config,
        request=request,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        output=args.output,
        label="Qwen3-Omni",
    )


PRESETS = {
    "qwen3-text-server": LauncherPreset(
        "Qwen3-Omni text server",
        _build_qwen_text_server_parser,
        launch_qwen_text_server,
    ),
    "qwen3-speech-server": LauncherPreset(
        "Qwen3-Omni speech server",
        _build_qwen_speech_server_parser,
        launch_qwen_speech_server,
        spawn=True,
    ),
    "qwen3-speech": LauncherPreset(
        "One Qwen3-Omni speech request",
        _build_qwen_speech_parser,
        lambda args: run_async(run_qwen_speech, args),
        spawn=True,
    ),
}
