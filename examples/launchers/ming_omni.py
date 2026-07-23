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

_MING_TEXT_SERVER_DESCRIPTION = """Launch Ming-Omni with text-only OpenAI responses.

Examples:
  python examples/run_omni.py ming-text-server \\
      --model-path inclusionAI/Ming-flash-omni-2.0 --port 8000

  curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"ming-omni","messages":[{"role":"user","content":"你好！"}]}'
"""

_MING_SPEECH_SERVER_DESCRIPTION = """Launch Ming-Omni with text and audio OpenAI responses.

Each stage runs in its own process with explicit GPU placement.

Examples:
  python examples/run_omni.py ming-speech-server \\
      --gpu-thinker 0 --gpu-talker 1 --cpu-offload-gb 80

  python examples/run_omni.py ming-speech-server \\
      --gpu-thinker 0 --gpu-talker 1 --enable-streaming-tts

  curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"ming-omni","messages":[{"role":"user","content":"你好！"}],"modalities":["text","audio"]}'
"""

_MING_SPEECH_DESCRIPTION = """Run one Ming-Omni text-to-speech request.

The talker contains its own LLM, CFM, and AudioVAE and generates speech from
the thinker's decoded text.

Examples:
  python examples/run_omni.py ming-speech --prompt "请给我讲一个故事。"

  python examples/run_omni.py ming-speech \\
      --prompt "你好，今天天气怎么样？" --gpu-thinker 0 --gpu-talker 1

  python examples/run_omni.py ming-speech \\
      --prompt "朗读一首古诗。" --output audio.wav
"""

_MING_TEXT_DESCRIPTION = """Run one Ming-Omni request with text output.

Examples:
  python examples/run_omni.py ming-text \\
      --prompt "你好，请介绍一下你自己。"

  python examples/run_omni.py ming-text \\
      --audio-path /path/to/audio.wav --prompt "请描述这段音频的内容。"
"""


def _build_ming_text_server_parser() -> argparse.ArgumentParser:
    target = parser(_MING_TEXT_SERVER_DESCRIPTION)
    add_model_path(target, "inclusionAI/Ming-flash-omni-2.0")
    target.add_argument(
        "--thinker-max-seq-len",
        type=int,
        default=8192,
        help="Context length for the thinker stage (default: 8192).",
    )
    target.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help=(
            "Tensor parallel size for thinker (must be >= 1). TP ranks use "
            "GPU ids 0 through N-1."
        ),
    )
    target.add_argument(
        "--gpu-audio-encoder",
        type=int,
        default=None,
        help="GPU id for the audio encoder stage.",
    )
    target.add_argument(
        "--gpu-image-encoder",
        type=int,
        nargs="+",
        default=None,
        help=(
            "GPU id(s) for the image encoder stage. For --image-encoder-tp N, "
            "pass N GPU ids."
        ),
    )
    target.add_argument(
        "--image-encoder-tp",
        type=int,
        default=1,
        help="Tensor parallel size for image encoder (must be >= 1)",
    )
    target.add_argument(
        "--thinker-only",
        action="store_true",
        help="Launch a text-only smoke pipeline without media encoders.",
    )
    target.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Thinker quantization method, for example fp8.",
    )
    target.add_argument(
        "--cpu-offload-gb",
        type=int,
        default=80,
        help=(
            "GB of thinker weights to offload to CPU "
            "(default: 80 for Ming-flash-omni-2.0)."
        ),
    )
    add_mem_fraction(target, "Set mem_fraction_static for the thinker stage.")
    add_server_args(target, model_name="ming-omni")
    return target


def _configure_ming_thinker_only(config: Any) -> None:
    stages = {stage.name: stage for stage in config.stages}
    preprocessing = stages["preprocessing"]
    aggregate = stages["mm_aggregate"]
    preprocessing.next = "mm_aggregate"
    preprocessing.project_payload = {
        "mm_aggregate": (
            "sglang_omni.models.ming_omni.stages.project_preprocessing_to_mm_aggregate"
        )
    }
    aggregate.wait_for = ["preprocessing"]
    config.stages = [
        stage
        for stage in config.stages
        if stage.name not in {"audio_encoder", "image_encoder"}
    ]


def launch_ming_text_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig
    from sglang_omni.serve import launch_server

    tp_size = int(args.tp_size)
    if tp_size < 1:
        raise ValueError(f"--tp-size must be >= 1, got {args.tp_size}")
    image_tp = int(args.image_encoder_tp)
    if image_tp < 1:
        raise ValueError("--image-encoder-tp must be >= 1")
    validate_fraction("--mem-fraction-static", args.mem_fraction_static)

    config = MingOmniPipelineConfig(model_path=args.model_path)
    if args.thinker_only:
        if args.gpu_audio_encoder is not None or args.gpu_image_encoder is not None:
            raise ValueError(
                "--gpu-audio-encoder/--gpu-image-encoder cannot be used "
                "with --thinker-only"
            )
        _configure_ming_thinker_only(config)

    server_updates: dict[str, object] = {}
    if tp_size > 1:
        set_stage_tp_size(config, "thinker", tp_size)
        set_stage_gpu(config, "thinker", list(range(tp_size)))
        server_updates["disable_custom_all_reduce"] = True
    if args.gpu_audio_encoder is not None:
        set_stage_gpu(config, "audio_encoder", int(args.gpu_audio_encoder))

    if image_tp > 1 and args.thinker_only:
        raise ValueError("--thinker-only cannot be used with --image-encoder-tp > 1")
    if image_tp > 1:
        if args.gpu_image_encoder is None:
            raise ValueError(
                "--gpu-image-encoder must be specified when --image-encoder-tp > 1"
            )
        if len(args.gpu_image_encoder) != image_tp:
            raise ValueError(
                f"--gpu-image-encoder requires exactly {image_tp} GPU ids "
                f"(matching --image-encoder-tp), got {len(args.gpu_image_encoder)}"
            )
        if len(set(args.gpu_image_encoder)) != len(args.gpu_image_encoder):
            raise ValueError("--gpu-image-encoder GPU ids must be unique")
        set_stage_tp_size(config, "image_encoder", image_tp)
        set_stage_gpu(config, "image_encoder", args.gpu_image_encoder)
    elif args.gpu_image_encoder is not None:
        set_stage_gpu(config, "image_encoder", int(args.gpu_image_encoder[0]))

    if args.quantization:
        server_updates["quantization"] = args.quantization
    if args.cpu_offload_gb:
        server_updates["cpu_offload_gb"] = int(args.cpu_offload_gb)
    if args.mem_fraction_static is not None:
        server_updates["mem_fraction_static"] = args.mem_fraction_static
    apply_stage_factory_updates(
        config,
        stage_name="thinker",
        updates={"thinker_max_seq_len": int(args.thinker_max_seq_len)},
        server_arg_updates=server_updates,
    )
    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


def _build_ming_speech_server_parser() -> argparse.ArgumentParser:
    target = parser(_MING_SPEECH_SERVER_DESCRIPTION)
    add_model_path(target, "inclusionAI/Ming-flash-omni-2.0")
    target.add_argument(
        "--gpu-thinker",
        type=int,
        default=0,
        help=(
            "Thinker GPU id. With TP > 1, this is the first GPU rank and the "
            "thinker uses the contiguous range starting here."
        ),
    )
    target.add_argument(
        "--gpu-talker",
        type=int,
        default=1,
        help="Talker GPU id; it must be outside the thinker TP range.",
    )
    target.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help=(
            "Tensor parallel size for thinker (must be >= 1). "
            "--gpu-thinker is the first GPU rank."
        ),
    )
    target.add_argument(
        "--voice",
        type=str,
        default="DB30",
        help="Voice ID for the talker (default: DB30).",
    )
    add_mem_fraction(target, "Set mem_fraction_static for the thinker stage.")
    target.add_argument(
        "--cpu-offload-gb",
        type=int,
        default=None,
        help=(
            "Offload N GiB of thinker weights to CPU. Ming-flash-omni-2.0 has "
            "about 200 GB of MoE weights and normally requires offload on one GPU."
        ),
    )
    target.add_argument(
        "--enable-streaming-tts",
        action="store_true",
        help=(
            "Use the 8-stage streaming-TTS path with a segmenter and streaming "
            "talker for sub-second time-to-first-audio. The default uses the "
            "non-streaming 7-stage speech path."
        ),
    )
    add_server_args(target, model_name="ming-omni")
    return target


def launch_ming_speech_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.ming_omni.config import (
        MingOmniSpeechPipelineConfig,
        MingOmniStreamingSpeechPipelineConfig,
    )
    from sglang_omni.serve import launch_server

    validate_fraction("--mem-fraction-static", args.mem_fraction_static)
    tp_size = int(args.tp_size)
    if tp_size < 1:
        raise ValueError(f"--tp-size must be >= 1, got {args.tp_size}")

    if args.enable_streaming_tts:
        config = MingOmniStreamingSpeechPipelineConfig(model_path=args.model_path)
        talker_stage = "talker_stream"
        validate_gpus = config._validate_talker_stream_gpu_not_in_thinker_tp_range
    else:
        config = MingOmniSpeechPipelineConfig(model_path=args.model_path)
        talker_stage = "talker"
        validate_gpus = config._validate_talker_gpu_not_in_thinker_tp_range

    set_stage_tp_size(config, "thinker", tp_size)
    thinker_gpus: int | list[int] = int(args.gpu_thinker)
    if tp_size > 1:
        thinker_gpus = list(
            range(int(args.gpu_thinker), int(args.gpu_thinker) + tp_size)
        )
    set_stage_gpu(config, "thinker", thinker_gpus)
    set_stage_gpu(config, talker_stage, int(args.gpu_talker))
    validate_gpus()

    server_updates: dict[str, object] = {}
    if tp_size > 1:
        server_updates["disable_custom_all_reduce"] = True
    if args.mem_fraction_static is not None:
        server_updates["mem_fraction_static"] = args.mem_fraction_static
    if args.cpu_offload_gb is not None:
        server_updates["cpu_offload_gb"] = args.cpu_offload_gb
    if server_updates:
        apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates=server_updates,
        )
    apply_stage_factory_updates(
        config,
        stage_name=talker_stage,
        updates={"voice": args.voice},
    )
    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


def _build_ming_speech_parser() -> argparse.ArgumentParser:
    target = parser(_MING_SPEECH_DESCRIPTION)
    add_model_path(target, "inclusionAI/Ming-flash-omni-2.0")
    add_offline_args(
        target,
        prompt="你好！给我讲一个有趣的事情。",
        system="你是一个友好的AI助手。请用自然、温暖的语气说话。",
        max_new_tokens=256,
        output="./output_audio.wav",
    )
    target.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Optional audio input path.",
    )
    target.add_argument(
        "--voice",
        type=str,
        default="DB30",
        help="Voice ID for the talker (default: DB30).",
    )
    target.add_argument(
        "--gpu-thinker",
        type=int,
        default=0,
        help=(
            "Thinker GPU id. With TP > 1, this is the first GPU rank and the "
            "thinker uses the contiguous range starting here."
        ),
    )
    target.add_argument(
        "--gpu-talker",
        type=int,
        default=1,
        help="Talker GPU id; it must be outside the thinker TP range.",
    )
    target.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=0,
        help="GiB of thinker weights to offload to CPU.",
    )
    add_mem_fraction(target, "Set mem_fraction_static for the thinker stage.")
    target.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help=(
            "Tensor parallel size for thinker (must be >= 1). "
            "--gpu-thinker is the first GPU rank."
        ),
    )
    return target


async def run_ming_speech(args: argparse.Namespace) -> None:
    from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

    tp_size = int(args.tp_size)
    if tp_size < 1:
        raise ValueError(f"--tp-size must be >= 1, got {args.tp_size}")
    validate_fraction("--mem-fraction-static", args.mem_fraction_static)

    config = MingOmniSpeechPipelineConfig(model_path=args.model_path)
    set_stage_tp_size(config, "thinker", tp_size)
    thinker_gpus: int | list[int] = args.gpu_thinker
    if tp_size > 1:
        thinker_gpus = list(range(args.gpu_thinker, args.gpu_thinker + tp_size))
    set_stage_gpu(config, "thinker", thinker_gpus)
    set_stage_gpu(config, "talker", args.gpu_talker)
    config._validate_talker_gpu_not_in_thinker_tp_range()

    overrides: dict[str, object] = {}
    if tp_size > 1:
        overrides["disable_custom_all_reduce"] = True
    if args.cpu_offload_gb:
        overrides["cpu_offload_gb"] = args.cpu_offload_gb
    if args.mem_fraction_static is not None:
        overrides["mem_fraction_static"] = args.mem_fraction_static
    if overrides:
        apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates=overrides,
        )
    apply_stage_factory_updates(
        config,
        stage_name="talker",
        updates={"voice": args.voice},
    )

    content: object = args.prompt
    if args.audio_path:
        content = [
            {"type": "audio_url", "audio_url": {"url": args.audio_path}},
            {"type": "text", "text": args.prompt},
        ]
    request = {
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": content},
        ],
        "audios": [args.audio_path] if args.audio_path else [],
    }
    await run_speech_request(
        config,
        request=request,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        output=args.output,
        label="Ming-Omni",
    )


def _build_ming_text_parser() -> argparse.ArgumentParser:
    target = parser(_MING_TEXT_DESCRIPTION)
    add_model_path(target, "inclusionAI/Ming-flash-omni-2.0")
    target.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    target.add_argument("--thinker-max-seq-len", type=int, default=8192)
    target.add_argument("--max-new-tokens", type=int, default=1024)
    target.add_argument("--temperature", type=float, default=0.8)
    target.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Optional audio input path.",
    )
    target.add_argument(
        "--cpu-offload-gb",
        type=int,
        default=80,
        help=(
            "GB of thinker weights to offload to CPU "
            "(default: 80 for Ming-flash-omni-2.0)."
        ),
    )
    add_mem_fraction(target, "Set mem_fraction_static for the thinker stage.")
    return target


def build_ming_text_config(args: argparse.Namespace) -> Any:
    from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig

    validate_fraction("--mem-fraction-static", args.mem_fraction_static)
    config = MingOmniPipelineConfig(model_path=args.model_path)
    overrides: dict[str, object] = {}
    if args.cpu_offload_gb:
        overrides["cpu_offload_gb"] = args.cpu_offload_gb
    if args.mem_fraction_static is not None:
        overrides["mem_fraction_static"] = args.mem_fraction_static
    apply_stage_factory_updates(
        config,
        stage_name="thinker",
        updates={"thinker_max_seq_len": int(args.thinker_max_seq_len)},
        server_arg_updates=overrides,
    )
    return config


async def run_ming_text(args: argparse.Namespace) -> None:
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import OmniRequest

    config = build_ming_text_config(args)

    content: object = args.prompt
    if args.audio_path:
        content = [
            {"type": "audio_url", "audio_url": {"url": args.audio_path}},
            {"type": "text", "text": args.prompt},
        ]
    runner = MultiProcessPipelineRunner(config)
    await runner.start()
    try:
        result = await runner.coordinator.submit(
            "ming-omni-text-first",
            OmniRequest(
                inputs={
                    "messages": [{"role": "user", "content": content}],
                    "audios": [args.audio_path] if args.audio_path else [],
                },
                params={
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                },
            ),
        )
        print(result)
    finally:
        await runner.stop()


PRESETS = {
    "ming-text-server": LauncherPreset(
        "Ming-Omni text server",
        _build_ming_text_server_parser,
        launch_ming_text_server,
        spawn=True,
    ),
    "ming-speech-server": LauncherPreset(
        "Ming-Omni speech server",
        _build_ming_speech_server_parser,
        launch_ming_speech_server,
        spawn=True,
    ),
    "ming-speech": LauncherPreset(
        "One Ming-Omni speech request",
        _build_ming_speech_parser,
        lambda args: run_async(run_ming_speech, args),
        spawn=True,
    ),
    "ming-text": LauncherPreset(
        "One Ming-Omni text request",
        _build_ming_text_parser,
        lambda args: run_async(run_ming_text, args),
        default_log_level="DEBUG",
    ),
}
