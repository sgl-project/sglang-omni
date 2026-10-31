# SPDX-License-Identifier: Apache-2.0 AND MIT
# Inference recipe adapted from Tencent-Hunyuan/AuK, Copyright (C) 2026 Tencent.
# See LICENSE for the upstream MIT permission notice.
"""Independent conditioning, DiT sampling, and audio decoding stages for AuK."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import torch

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.batch_compute import (
    BatchCompute,
    condition_batch,
    decode_batch,
    sample_batch,
)
from sglang_omni.models.auk.hf_config import make_runtime_config
from sglang_omni.models.auk.offload_scheduler import AuKOffloadScheduler
from sglang_omni.models.auk.reference_encode import AuKConditionEncoder
from sglang_omni.models.auk.request_builders import (
    AuKPreprocessingContext,
    preprocess_auk_payload,
    set_auk_preprocessing_context,
)
from sglang_omni.models.auk.runtime import (
    load_flow,
    load_fusion,
    load_vae,
    pin_attention_backend,
    resolve_dtype,
    warmup_flow,
)
from sglang_omni.models.auk.serial_offload import get_coordinator
from sglang_omni.models.auk.step_cuda_graph import (
    DEFAULT_CAPTURE_SHAPES,
    build_step_graph_runner,
)
from sglang_omni.models.auk.weight_loader import resolve_vae_file, resolve_weight_file
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_residency import torch_memory_saver
from sglang_omni.scheduling.weight_cache import checkpoint_identity
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device


def scheduler(
    compute_batch: BatchCompute,
    device: torch.device,
    max_batch_size: int,
    max_batch_wait_ms: int,
) -> SimpleScheduler[StagePayload, StagePayload]:
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    @torch.inference_mode()
    def run(payloads: list[StagePayload]) -> list[StagePayload]:
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            return compute_batch(payloads)

    return SimpleScheduler(
        lambda payload: run([payload])[0],
        batch_compute_fn=run,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_wait_when_idle=False,
    )


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 8,
    default_seconds: float = C.DEFAULT_SECONDS,
    max_seconds: float = C.MAX_SECONDS,
) -> SimpleScheduler[StagePayload, StagePayload]:
    config = make_runtime_config(resolve_checkpoint(model_path))
    set_auk_preprocessing_context(
        AuKPreprocessingContext(
            config=config,
            default_seconds=default_seconds,
            max_seconds=max_seconds,
        )
    )
    return SimpleScheduler(preprocess_auk_payload, max_concurrency=max_concurrency)


def create_conditioning_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    text_encoder_path: str = C.DEFAULT_TEXT_ENCODER,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 10,
    enable_serial_offload: bool = False,
    serial_offload_source: Literal["mmap", "ram"] | None = None,
    serial_offload_cache_dir: str | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    compute_dtype = resolve_dtype(field="dtype", name=dtype)
    device = resolve_concrete_device(device, gpu_id)
    pin_attention_backend(device)
    checkpoint = resolve_checkpoint(model_path)
    fusion = load_fusion(checkpoint, str(device))
    offload = None
    if enable_serial_offload:
        if device.type != "cuda" or torch_memory_saver is None:
            raise ValueError("AuK offload requires CUDA and torch_memory_saver==0.0.10")
        elif version("torch_memory_saver") != "0.0.10":
            raise ValueError("AuK offload requires torch_memory_saver==0.0.10")
        elif serial_offload_source is None:
            raise ValueError("AuK offload requires serial_offload_source")
        else:
            pass
        offload = get_coordinator(checkpoint, str(device))
        if offload.residencies:
            raise RuntimeError("AuK offload is already initialized in this process")
        else:
            pass
        encoder_checkpoint = resolve_checkpoint(text_encoder_path)
        with torch_memory_saver.region(tag="auk_conditioning", enable_cpu_backup=False):
            encoder = AuKConditionEncoder(
                encoder_checkpoint, device=device, dtype=torch.bfloat16
            )
        offload.register_conditioning(
            encoder.model,
            checkpoint_identity(
                sorted(Path(encoder_checkpoint).glob("*.safetensors"))
                + sorted(Path(encoder_checkpoint).glob("*.bin"))
                + sorted(Path(encoder_checkpoint).glob("*.json"))
            ),
            source=serial_offload_source,
            cache_dir=serial_offload_cache_dir,
        )
        offload.switch(None)
        with torch_memory_saver.region(tag="auk_vae", enable_cpu_backup=False):
            vae = load_vae(checkpoint, str(device), fold_weight_norm=True)
        vae_path = resolve_vae_file(checkpoint)
        assert vae_path is not None
        offload.register(
            "vae",
            {"vae": vae},
            checkpoint_identity([vae_path, *Path(checkpoint).glob("config.*")]),
            folding_backend=f"cuda-{torch.cuda.get_device_capability(device)}-weight-norm-v1",
        )
    else:
        encoder = AuKConditionEncoder(
            text_encoder_path, device=device, dtype=torch.bfloat16
        )
        vae = load_vae(checkpoint, str(device))
    compute = lambda payloads: condition_batch(
        payloads, encoder, vae, fusion, device, compute_dtype, offload
    )
    if offload is not None:
        return AuKOffloadScheduler(compute, offload, "conditioning")
    else:
        return scheduler(compute, device, max_batch_size, max_batch_wait_ms)


def create_auk_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    nfe: int = C.DEFAULT_NFE,
    enable_dit_fused_qk_norm_rope: bool = True,
    cfg_strength: float = C.DEFAULT_CFG_STRENGTH,
    sway_sampling_coef: float | None = C.DEFAULT_SWAY_SAMPLING_COEF,
    max_seconds: float = C.MAX_SECONDS,
    max_batch_size: int = 16,
    max_batch_wait_ms: int = 10,
    weight_dtype: str = "float32",
    enable_dit_torch_compile: bool = False,
    enable_dit_cuda_graph: bool = False,
    dit_cuda_graph_capture_shapes: Sequence[Sequence[int]] | None = None,
    enable_serial_offload: bool = False,
) -> SimpleScheduler[StagePayload, StagePayload]:
    """Build the DiT sampling stage.

    A float32 weight_dtype keeps the weights under dtype autocast, the
    upstream-exact recipe; bfloat16 stores the backbone in bf16 and skips
    autocast. See docs/cookbook/auk.md, Sampling, for the compile and graph
    options and the capture shape format.
    """
    # Named dtypes are checked before resolve_checkpoint, which downloads.
    compute_dtype = resolve_dtype(field="dtype", name=dtype)
    backbone_dtype = resolve_dtype(field="weight_dtype", name=weight_dtype)
    device = resolve_concrete_device(device, gpu_id)
    pin_attention_backend(device)
    checkpoint = resolve_checkpoint(model_path)
    config = make_runtime_config(checkpoint)
    # note(Dayuxiaoshui): autocast reads fp32 as off, and on a non-fp32
    # backbone it would only re-cast per op and force the norms back to fp32.
    autocast_dtype = compute_dtype if backbone_dtype == torch.float32 else torch.float32
    offload = None
    if enable_serial_offload:
        offload = get_coordinator(checkpoint, str(device))
        offload.require_vae()
        assert torch_memory_saver is not None
        offload.switch(None)
        with torch_memory_saver.region(tag="auk_dit", enable_cpu_backup=False):
            flow = load_flow(
                checkpoint,
                str(device),
                backbone_dtype,
                enable_dit_torch_compile,
                finalize_on_cpu=True,
            )
    else:
        flow = load_flow(
            checkpoint, str(device), backbone_dtype, enable_dit_torch_compile
        )
    sampling = dict(
        steps=C.FLASH_NFE if config.is_flash else nfe,
        cfg_strength=C.FLASH_CFG_STRENGTH if config.is_flash else cfg_strength,
        sway_sampling_coef=None if config.is_flash else sway_sampling_coef,
        t_grid=C.FLASH_T_GRID if config.is_flash else None,
    )
    # note(Dayuxiaoshui): installed before the blocks compile and before the
    # step graph captures them, so both carry the fused kernel.
    if enable_dit_fused_qk_norm_rope and device.type == "cuda" and not config.is_flash:
        from sglang_omni.models.auk.fused_qk_norm_rope import fused_qk_norm_rope

        flow.transformer.enable_fused_qk_norm_rope(fused_qk_norm_rope)
    else:
        pass
    step_graph = None
    if enable_dit_cuda_graph:
        if not flow.transformer.attn_mask_enabled:
            raise ValueError(
                "AuK enable_dit_cuda_graph needs attn_mask_enabled: without the "
                "attention bias, padded rows would reach the valid ones"
            )
        else:
            pass
        if enable_serial_offload:
            if dit_cuda_graph_capture_shapes is None:
                dit_cuda_graph_capture_shapes = [
                    shape for shape in DEFAULT_CAPTURE_SHAPES if shape.batch == 1
                ]
            elif any(shape[0] != 1 for shape in dit_cuda_graph_capture_shapes):
                raise ValueError("AuK offload capture shapes require batch size 1")
            else:
                pass
        else:
            pass
        step_graph = build_step_graph_runner(device, dit_cuda_graph_capture_shapes)
    else:
        pass
    if enable_dit_torch_compile or step_graph is not None:
        if enable_serial_offload or max_batch_size == 1:
            warmup_flow(flow, device, autocast_dtype, sampling, step_graph, serial=True)
        else:
            warmup_flow(flow, device, autocast_dtype, sampling, step_graph)
    else:
        pass
    if step_graph is not None:
        sampling["step_graph"] = step_graph
    else:
        pass
    if offload is not None:
        offload.register(
            "dit",
            {"flow": flow},
            checkpoint_identity(
                [resolve_weight_file(checkpoint), *Path(checkpoint).glob("config.*")]
            ),
        )
    else:
        pass
    compute = lambda payloads: sample_batch(
        payloads,
        flow,
        device,
        autocast_dtype,
        config.seconds_to_frames(max_seconds),
        sampling,
        offload,
    )
    if offload is not None:
        return AuKOffloadScheduler(compute, offload, "auk_engine")
    else:
        return scheduler(compute, device, max_batch_size, max_batch_wait_ms)


def create_decode_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_batch_size: int = 4,
    max_batch_wait_ms: int = 10,
    enable_serial_offload: bool = False,
) -> SimpleScheduler[StagePayload, StagePayload]:
    device = resolve_concrete_device(device, gpu_id)
    checkpoint = resolve_checkpoint(model_path)
    if enable_serial_offload:
        offload = get_coordinator(checkpoint, str(device))
        vae = offload.require_vae()
        return AuKOffloadScheduler(
            lambda payloads: decode_batch(payloads, vae, device, offload),
            offload,
            "decode",
        )
    else:
        vae = load_vae(checkpoint, str(device))
    return scheduler(
        lambda payloads: decode_batch(payloads, vae, device),
        device,
        max_batch_size,
        max_batch_wait_ms,
    )
