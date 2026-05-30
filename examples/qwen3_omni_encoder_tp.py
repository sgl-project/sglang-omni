# SPDX-License-Identifier: Apache-2.0
"""Reference script: launch Qwen3-Omni speech with SGLang-backed encoder TP.

This is the Plan B deployment recipe described in
``docs/developer_reference/encoder_tp_path_b_design.md`` (RFC #375).
It shows how to flip the image and audio encoder stages from the local
HF tower to the SGLang-native encoders without changing any other
stage.

Usage (single-host, 8-GPU layout):

    python examples/qwen3_omni_encoder_tp.py \\
        --model Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --image-tp 2 --audio-tp 2 --port 8000

    # Long-video validation on hosts with constrained shared GPU headroom:
    python examples/qwen3_omni_encoder_tp.py \\
        --model Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --image-tp 2 --audio-tp 2 \\
        --thinker-mem-fraction-static 0.45 \\
        --talker-mem-fraction-static 0.45 \\
        --thinker-max-seq-len 32768 \\
        --port 8000

    # Same PR-branch command shape for a controlled tp=1 SGLang baseline:
    python examples/qwen3_omni_encoder_tp.py \\
        --model Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --image-tp 1 --audio-tp 1 --port 8000

The encoders run with ``tp_size=2`` each on dedicated GPU pairs. The
thinker uses the next GPU, while talker and code2wav share the following
GPU as in the default speech config. The selected encoder backend is honored
at every TP size: ``sglang`` uses the SGLang encoder runner, and ``local``
uses the main local encoder path where supported.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _validate_optional_fraction(name: str, value: float | None) -> None:
    if value is None:
        return
    if not 0.0 < value < 1.0:
        raise SystemExit(f"{name} must be in (0, 1)")


def _apply_ar_mem_fraction(
    stage,
    fraction: float | None,
    *,
    typed_resource: bool = True,
):
    """Set SGLang static AR memory and optionally the typed placement budget."""

    if fraction is None:
        return stage
    resources_update = (
        {"total_gpu_memory_fraction": fraction}
        if typed_resource
        else {"total_gpu_memory_fraction": None}
    )
    runtime = stage.runtime.model_copy(
        update={
            "resources": stage.runtime.resources.model_copy(
                update=resources_update
            ),
            "sglang_server_args": stage.runtime.sglang_server_args.model_copy(
                update={"mem_fraction_static": fraction}
            ),
        }
    )
    return stage.model_copy(update={"runtime": runtime})


def _apply_encoder_runtime(
    stage,
    *,
    activation_budget_bytes: int,
    total_gpu_memory_fraction: float | None,
    encoder_max_batch_size: int | None,
    typed_resource: bool = True,
):
    resources_update = {
        "encoder_activation_budget_bytes": activation_budget_bytes,
    }
    if typed_resource and total_gpu_memory_fraction is not None:
        resources_update["total_gpu_memory_fraction"] = total_gpu_memory_fraction
    elif not typed_resource:
        resources_update["total_gpu_memory_fraction"] = None
    if encoder_max_batch_size is not None:
        resources_update["encoder_max_batch_size"] = encoder_max_batch_size
    runtime = stage.runtime.model_copy(
        update={
            "resources": stage.runtime.resources.model_copy(update=resources_update),
        }
    )
    return stage.model_copy(update={"runtime": runtime})


def _gib_to_positive_bytes(name: str, value: float) -> int:
    if value <= 0:
        raise SystemExit(f"{name} must be positive")
    return int(value * 1024**3)


def _resolve_layout(
    *,
    layout: str,
    image_tp: int,
    audio_tp: int,
) -> tuple[list[int], list[int], int, int]:
    if layout == "separate":
        if image_tp + audio_tp + 2 > 8:
            raise SystemExit(
                "The separate recipe assumes 8 GPUs: image_encoder TP + "
                "audio_encoder TP, then one thinker GPU and one shared "
                "talker/code2wav GPU."
            )
        image_gpus = list(range(0, image_tp))
        audio_gpus = list(range(image_tp, image_tp + audio_tp))
        thinker_gpu = image_tp + audio_tp
        talker_gpu = thinker_gpu + 1
        return image_gpus, audio_gpus, thinker_gpu, talker_gpu

    if layout == "colocated-2gpu":
        if image_tp > 2 or audio_tp > 2:
            raise SystemExit(
                "--layout colocated-2gpu supports only --image-tp/--audio-tp <= 2"
            )
        image_gpus = list(range(image_tp))
        audio_gpus = list(range(audio_tp))
        return image_gpus, audio_gpus, 0, 1

    raise SystemExit(f"unknown layout {layout!r}")


def _resolve_encoder_max_batch_size(
    *,
    explicit: int | None,
    layout: str,
    image_tp: int,
    audio_tp: int,
) -> int | None:
    del layout, image_tp, audio_tp
    if explicit is not None:
        return explicit
    return None


def _resolve_effective_encoder_backend(
    requested: str,
    *,
    tp_size: int,
) -> str:
    del tp_size
    if requested == "auto":
        return "sglang"
    return requested


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument(
        "--encoder-backend",
        choices=("local", "sglang", "auto"),
        default="sglang",
        help="Backend to use for both image and audio encoder stages.",
    )
    p.add_argument("--image-tp", type=int, default=2)
    p.add_argument("--audio-tp", type=int, default=2)
    p.add_argument(
        "--layout",
        choices=("separate", "colocated-2gpu"),
        default="separate",
        help=(
            "GPU placement recipe. 'separate' uses the original 8-GPU layout; "
            "'colocated-2gpu' uses two visible GPUs, maps image/audio encoder "
            "TP ranks onto both GPUs, places thinker on GPU 0, and places "
            "talker/code2wav on GPU 1."
        ),
    )
    p.add_argument(
        "--encoder-activation-budget-gib",
        type=float,
        default=10.0,
        help=(
            "Default typed runtime.resources.encoder_activation_budget_bytes "
            "for each encoder stage."
        ),
    )
    p.add_argument(
        "--image-encoder-activation-budget-gib",
        type=float,
        default=None,
        help=(
            "Optional image_encoder-specific activation budget. Defaults to "
            "--encoder-activation-budget-gib."
        ),
    )
    p.add_argument(
        "--audio-encoder-activation-budget-gib",
        type=float,
        default=None,
        help=(
            "Optional audio_encoder-specific activation budget. Defaults to "
            "--encoder-activation-budget-gib."
        ),
    )
    p.add_argument(
        "--encoder-total-gpu-memory-fraction",
        type=float,
        default=None,
        help=(
            "Optional typed runtime.resources.total_gpu_memory_fraction "
            "override for each encoder rank. Useful for colocated validation "
            "where activation headroom is budgeted separately."
        ),
    )
    p.add_argument(
        "--encoder-max-batch-size",
        type=int,
        default=None,
        help=(
            "Optional runtime.resources.encoder_max_batch_size for each encoder. "
            "When omitted, encoder batch size is decided by activation-budget "
            "admission plus the runtime whole-GPU guard."
        ),
    )
    p.add_argument(
        "--thinker-mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Optional typed runtime.resources.total_gpu_memory_fraction and "
            "runtime.sglang_server_args.mem_fraction_static override for the "
            "thinker stage. Useful for validation hosts with limited free KV "
            "headroom."
        ),
    )
    p.add_argument(
        "--talker-mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Optional typed runtime.resources.total_gpu_memory_fraction and "
            "runtime.sglang_server_args.mem_fraction_static override for the "
            "talker_ar stage."
        ),
    )
    p.add_argument(
        "--thinker-max-seq-len",
        type=int,
        default=None,
        help="Optional thinker_max_seq_len override for long-video validation.",
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    _validate_optional_fraction(
        "--thinker-mem-fraction-static", args.thinker_mem_fraction_static
    )
    _validate_optional_fraction(
        "--talker-mem-fraction-static", args.talker_mem_fraction_static
    )
    _validate_optional_fraction(
        "--encoder-total-gpu-memory-fraction",
        args.encoder_total_gpu_memory_fraction,
    )
    if args.thinker_max_seq_len is not None and args.thinker_max_seq_len <= 0:
        raise SystemExit("--thinker-max-seq-len must be positive")
    if args.encoder_max_batch_size is not None and args.encoder_max_batch_size <= 0:
        raise SystemExit("--encoder-max-batch-size must be positive")

    # Import after argparse so --help is fast.
    from sglang_omni.models.qwen3_omni.config import (
        Qwen3OmniSpeechPipelineConfig,
    )
    from sglang_omni.serve.launcher import launch_server

    image_backend = _resolve_effective_encoder_backend(
        args.encoder_backend,
        tp_size=args.image_tp,
    )
    audio_backend = _resolve_effective_encoder_backend(
        args.encoder_backend,
        tp_size=args.audio_tp,
    )
    if image_backend == "local" and args.image_tp != 1:
        raise SystemExit("image backend='local' supports only --image-tp 1")
    if audio_backend == "local" and args.audio_tp != 1:
        raise SystemExit("audio backend='local' supports only --audio-tp 1")
    if args.encoder_backend == "local" and (
        args.image_tp != 1 or args.audio_tp != 1
    ):
        raise SystemExit("backend='local' supports only --image-tp 1 --audio-tp 1")
    default_encoder_activation_budget_bytes = _gib_to_positive_bytes(
        "--encoder-activation-budget-gib", args.encoder_activation_budget_gib
    )
    image_encoder_activation_budget_bytes = _gib_to_positive_bytes(
        "--image-encoder-activation-budget-gib",
        args.image_encoder_activation_budget_gib,
    ) if args.image_encoder_activation_budget_gib is not None else (
        default_encoder_activation_budget_bytes
    )
    audio_encoder_activation_budget_bytes = _gib_to_positive_bytes(
        "--audio-encoder-activation-budget-gib",
        args.audio_encoder_activation_budget_gib,
    ) if args.audio_encoder_activation_budget_gib is not None else (
        default_encoder_activation_budget_bytes
    )
    image_gpus, audio_gpus, thinker_gpu, talker_gpu = _resolve_layout(
        layout=args.layout,
        image_tp=args.image_tp,
        audio_tp=args.audio_tp,
    )
    encoder_max_batch_size = _resolve_encoder_max_batch_size(
        explicit=args.encoder_max_batch_size,
        layout=args.layout,
        image_tp=args.image_tp,
        audio_tp=args.audio_tp,
    )
    typed_ar_memory = image_backend == "sglang" or audio_backend == "sglang"

    # Build the canonical config and flip just the two encoder stages.
    cfg = Qwen3OmniSpeechPipelineConfig(model_path=args.model)
    new_stages = []
    for s in cfg.stages:
        if s.name == "preprocessing" and args.thinker_max_seq_len is not None:
            s = s.model_copy(
                update={
                    "factory_args": {
                        **s.factory_args,
                        "thinker_max_seq_len": args.thinker_max_seq_len,
                    }
                }
            )
        elif s.name == "image_encoder":
            s = s.model_copy(
                update={
                    "factory_args": {**s.factory_args, "backend": image_backend},
                    "tp_size": args.image_tp,
                    "parallelism": s.parallelism.model_copy(
                        update={"tp": args.image_tp}
                    ),
                    "gpu": image_gpus,
                }
            )
            s = _apply_encoder_runtime(
                s,
                activation_budget_bytes=image_encoder_activation_budget_bytes,
                total_gpu_memory_fraction=args.encoder_total_gpu_memory_fraction,
                encoder_max_batch_size=encoder_max_batch_size,
                typed_resource=image_backend == "sglang",
            )
        elif s.name == "audio_encoder":
            s = s.model_copy(
                update={
                    "factory_args": {**s.factory_args, "backend": audio_backend},
                    "tp_size": args.audio_tp,
                    "parallelism": s.parallelism.model_copy(
                        update={"tp": args.audio_tp}
                    ),
                    "gpu": audio_gpus,
                }
            )
            s = _apply_encoder_runtime(
                s,
                activation_budget_bytes=audio_encoder_activation_budget_bytes,
                total_gpu_memory_fraction=args.encoder_total_gpu_memory_fraction,
                encoder_max_batch_size=encoder_max_batch_size,
                typed_resource=audio_backend == "sglang",
            )
        elif s.name == "thinker":
            factory_args = dict(s.factory_args)
            if args.thinker_max_seq_len is not None:
                factory_args["thinker_max_seq_len"] = args.thinker_max_seq_len
            s = _apply_ar_mem_fraction(
                s.model_copy(update={"gpu": thinker_gpu, "factory_args": factory_args}),
                args.thinker_mem_fraction_static,
                typed_resource=typed_ar_memory,
            )
        elif s.name == "talker_ar":
            s = _apply_ar_mem_fraction(
                s.model_copy(update={"gpu": talker_gpu}),
                args.talker_mem_fraction_static,
                typed_resource=typed_ar_memory,
            )
        elif s.name == "code2wav":
            s = s.model_copy(update={"gpu": talker_gpu})
        new_stages.append(s)
    cfg = cfg.model_copy(update={"stages": new_stages})

    launch_server(cfg, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
