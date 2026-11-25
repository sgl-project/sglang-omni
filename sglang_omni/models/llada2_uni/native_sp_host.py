# SPDX-License-Identifier: Apache-2.0
"""One Omni host submits decoder requests to an engine-owned SP group."""

from __future__ import annotations

import importlib
import multiprocessing
import os
import secrets
from uuid import uuid4

import torch
from sglang.multimodal_gen.configs.sample.zimage import ZImageSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.platforms.plugins import apply_plugin_hooks
from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs

from sglang_omni.models.llada2_uni.components.decoder_model import (
    require_joint_sp_support,
)
from sglang_omni.models.llada2_uni.components.image_decoder import (
    DecoderMode,
    LLaDA2ImageDecoder,
    validate_checkpoint_sp_config,
)
from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.native_generation.stage import (
    NativeGenerationScheduler,
    native_server_kwargs,
)
from sglang_omni.proto import StagePayload
from sglang_omni.utils.device import resolve_concrete_device

EXTERNAL_MODEL_PACKAGE = "sglang_omni.models.llada2_uni.native_sp"
PRECISION_BY_DTYPE = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
}


class NativeSPDecoder:
    def __init__(
        self,
        model_path: str,
        gpu_id: int,
        devices: list[int],
        ring_degree: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        attention_backend: str = "fa",
        decode_mode: DecoderMode = "decoder-turbo",
        num_steps: int = 8,
        resolution_multiplier: int = 2,
    ) -> None:
        if dtype not in PRECISION_BY_DTYPE:
            raise ValueError(f"Unsupported native SP decoder dtype: {dtype}")
        else:
            pass
        self.decode_mode: DecoderMode = decode_mode
        self.num_steps = num_steps
        self.resolution_multiplier = resolution_multiplier
        kwargs = native_server_kwargs(
            model_path,
            gpu_id,
            {
                "pipeline_class_name": "LLaDASemanticDecoder",
                "sp_degree": len(devices),
                "ulysses_degree": len(devices) // ring_degree,
                "ring_degree": ring_degree,
                "attention_backend": attention_backend,
                "dit_precision": PRECISION_BY_DTYPE[dtype],
                "enable_torch_compile": False,
                "dit_cpu_offload": False,
            },
            devices,
        )
        previous_package = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
        if previous_package and previous_package != EXTERNAL_MODEL_PACKAGE:
            raise ValueError(
                "Native SP decoder requires SGLANG_EXTERNAL_MODEL_PACKAGE to be "
                f"unset or {EXTERNAL_MODEL_PACKAGE!r}, got {previous_package!r}"
            )
        else:
            pass
        os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = EXTERNAL_MODEL_PACKAGE
        try:
            importlib.import_module(EXTERNAL_MODEL_PACKAGE)
            apply_plugin_hooks()
            self.args = ServerArgs.from_kwargs(**kwargs)
            self.engine = DiffGenerator.from_server_args(self.args)
        finally:
            if previous_package is None:
                os.environ.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
            else:
                os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = previous_package

    def build_request(
        self,
        tokens: list[int],
        height: int,
        width: int,
        *,
        seed: int | None = None,
        decode_mode: DecoderMode | None = None,
        num_steps: int | None = None,
        resolution_multiplier: int | None = None,
    ) -> object:
        mode: DecoderMode = self.decode_mode if decode_mode is None else decode_mode
        steps = self.num_steps if num_steps is None else num_steps
        multiplier = (
            self.resolution_multiplier
            if resolution_multiplier is None
            else resolution_multiplier
        )
        LLaDA2ImageDecoder.validate_settings(mode, steps, multiplier)
        resolved_seed = secrets.randbits(63) if seed is None else int(seed)
        transport_seed = resolved_seed if resolved_seed >= 0 else 0
        params = ZImageSamplingParams(
            request_id=uuid4().hex,
            width=width * 16 * multiplier,
            height=height * 16 * multiplier,
            seed=transport_seed,
            num_inference_steps=steps,
            prompt="",
            save_output=False,
            return_file_paths_only=False,
        )
        request = prepare_request(self.args, params)
        request.extra["llada_vq"] = (
            tokens,
            height,
            width,
            resolved_seed,
            mode,
            steps,
            multiplier,
        )
        return request

    @staticmethod
    def extract_png(result: object) -> bytes:
        if result is None:
            raise RuntimeError("Native SP decoder returned no response")
        elif result.error:
            raise RuntimeError(result.error)
        elif not result.output_image_bytes:
            raise RuntimeError("Native SP decoder returned no PNG output")
        else:
            return result.output_image_bytes[0]

    def close(self) -> None:
        self.engine.shutdown()


class NativeSPDecoderScheduler(NativeGenerationScheduler):
    """Run LLaDA2-Uni semantic decoding on an SGL-Diffusion worker group."""

    def __init__(self, decoder: NativeSPDecoder) -> None:
        from sglang_omni.models.llada2_uni.stages import build_image_decode_step

        self.decoder = decoder
        self.decode_step = build_image_decode_step(self.decode_png)
        super().__init__(
            decoder.engine,
            output_dir="",
            request_fields=frozenset(),
            cancellation_mode=None,
            inline_media=True,
        )

    def decode_png(
        self,
        tokens: list[int],
        height: int,
        width: int,
        params: dict[str, int | DecoderMode],
    ) -> bytes:
        if self.closing:
            raise RuntimeError("Native SP decoder is shutting down")
        else:
            pass
        request = self.decoder.build_request(tokens, height, width, **params)
        result = self.run_native(
            {"batch": [request]}, target=sync_scheduler_client.forward
        )
        return self.decoder.extract_png(result)

    def generate(self, payload: StagePayload) -> StagePayload:
        return self.decode_step(payload)


def create_native_decoder(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    runtime_gpu_ids: list[int],
    dtype: str | torch.dtype | None = None,
    backend: str = "sglang",
    attention_backend: str = "fa",
    decode_mode: DecoderMode = "decoder-turbo",
    num_steps: int = 8,
    resolution_multiplier: int = 2,
    ring_degree: int = 1,
    enable_serial_offload: bool = False,
) -> NativeGenerationScheduler:
    if multiprocessing.current_process().daemon:
        raise RuntimeError("Native SP decoder requires allow_child_processes=true")
    else:
        pass
    concrete_device = resolve_concrete_device(device, gpu_id)
    if concrete_device.index is None:
        raise ValueError("Native SP decoder requires an indexed accelerator")
    else:
        pass
    resolved_dtype = resolve_dtype(dtype)
    if resolved_dtype not in PRECISION_BY_DTYPE:
        raise ValueError(f"Unsupported native SP decoder dtype: {resolved_dtype}")
    else:
        pass
    if enable_serial_offload:
        raise ValueError("Native SP does not support serial offload")
    elif backend != "sglang":
        raise ValueError("Native SP requires the sglang decoder backend")
    elif not attention_backend or attention_backend == "auto":
        raise ValueError("Native SP attention_backend must be explicit")
    elif not runtime_gpu_ids or ring_degree < 1 or len(runtime_gpu_ids) % ring_degree:
        raise ValueError("Ring degree must divide the number of decoder GPUs")
    else:
        pass
    LLaDA2ImageDecoder.validate_settings(decode_mode, num_steps, resolution_multiplier)
    ulysses_degree = len(runtime_gpu_ids) // ring_degree
    if len(runtime_gpu_ids) > 1:
        require_joint_sp_support()
    else:
        pass
    validate_checkpoint_sp_config(model_path, decode_mode, ulysses_degree)
    decoder = NativeSPDecoder(
        model_path,
        concrete_device.index,
        runtime_gpu_ids,
        ring_degree,
        dtype=resolved_dtype,
        attention_backend=attention_backend,
        decode_mode=decode_mode,
        num_steps=num_steps,
        resolution_multiplier=resolution_multiplier,
    )
    try:
        return NativeSPDecoderScheduler(decoder)
    except BaseException:
        decoder.close()
        raise
