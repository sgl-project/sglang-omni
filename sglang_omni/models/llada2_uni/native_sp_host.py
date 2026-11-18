# SPDX-License-Identifier: Apache-2.0
"""One Omni host submits decoder requests to an engine-owned SP group."""

import secrets
from typing import Literal
from uuid import uuid4

from sglang.multimodal_gen.configs.sample.zimage import ZImageSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs

from sglang_omni.models.llada2_uni.components.image_decoder import DecoderMode
from sglang_omni.models.llada2_uni.stages import build_image_decode_step
from sglang_omni.native_generation.stage import native_server_kwargs
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


class NativeSPDecoder:
    def __init__(self, model_path: str, gpu_id: int, devices: list[int]) -> None:
        kwargs = native_server_kwargs(
            model_path,
            gpu_id,
            {
                "pipeline_class_name": "LLaDASemanticDecoder",
                "sp_degree": 2,
                "ulysses_degree": 2,
                "ring_degree": 1,
                "attention_backend": "fa",
                "enable_torch_compile": False,
                "dit_cpu_offload": False,
            },
            devices,
        )
        self.args = ServerArgs.from_kwargs(**kwargs)
        self.engine = DiffGenerator.from_server_args(self.args)

    def decode_png(
        self,
        tokens: list[int],
        height: int,
        width: int,
        *,
        seed: int | None = None,
        decode_mode: Literal["normal", "decoder-turbo"] = "decoder-turbo",
        num_steps: int = 8,
    ) -> bytes:
        seed = secrets.randbits(63) if seed is None else seed
        params = ZImageSamplingParams(
            request_id=uuid4().hex,
            width=width * 32,
            height=height * 32,
            seed=seed,
            num_inference_steps=num_steps,
            prompt="",
            save_output=False,
            return_file_paths_only=False,
        )
        request = prepare_request(self.args, params)
        request.extra["llada_vq"] = (
            tokens,
            height,
            width,
            seed,
            decode_mode,
            num_steps,
        )
        result = sync_scheduler_client.forward([request])
        if result.error:
            raise RuntimeError(result.error)
        else:
            return result.output_image_bytes[0]

    def close(self) -> None:
        self.engine.shutdown()


def create_native_decoder(
    model_path: str,
    *,
    gpu_id: int,
    runtime_gpu_ids: list[int],
    backend: str,
    attention_backend: str,
    decode_mode: DecoderMode,
    num_steps: int,
    resolution_multiplier: int = 2,
) -> SimpleScheduler[StagePayload, StagePayload]:
    if (
        backend,
        attention_backend,
        decode_mode,
        num_steps,
        resolution_multiplier,
        len(runtime_gpu_ids),
    ) != ("sglang", "fa", "decoder-turbo", 8, 2, 2):
        raise ValueError("Native SP2 requires FA, turbo8 and resolution multiplier 2")
    else:
        pass
    decoder = NativeSPDecoder(model_path, gpu_id, runtime_gpu_ids)

    def decode_png(
        tokens: list[int], height: int, width: int, params: dict[str, int | DecoderMode]
    ) -> bytes:
        return decoder.decode_png(tokens, height, width, **params)

    return SimpleScheduler(
        build_image_decode_step(decode_png), shutdown_callback=decoder.close
    )
