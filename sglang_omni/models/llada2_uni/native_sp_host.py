"""One Omni host submits decoder requests to an engine-owned SP group."""

import io
import secrets
from uuid import uuid4

from PIL import Image
from sglang.multimodal_gen.configs.sample.zimage import ZImageSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs

from sglang_omni.native_generation.stage import native_server_kwargs


class NativeSPDecoder:
    def __init__(self, model_path: str, gpu_id: int, devices: list[int]) -> None:
        kwargs = native_server_kwargs(model_path, gpu_id, {
            "pipeline_class_name": "LLaDASemanticDecoder", "sp_degree": 2,
            "ulysses_degree": 2, "ring_degree": 1, "attention_backend": "fa",
            "enable_torch_compile": False, "dit_cpu_offload": False,
        }, devices)
        self.args = ServerArgs.from_kwargs(**kwargs)
        self.engine = DiffGenerator.from_server_args(self.args)

    def decode(self, tokens, height, width, *, seed=None, decode_mode="decoder-turbo", num_steps=8):
        seed = secrets.randbits(63) if seed is None else seed
        params = ZImageSamplingParams(request_id=uuid4().hex, width=width*32, height=height*32,
                                      seed=seed, num_inference_steps=num_steps, prompt="",
                                      save_output=False, return_file_paths_only=False)
        request = prepare_request(self.args, params)
        request.extra["llada_vq"] = (tokens, height, width, seed, decode_mode, num_steps)
        result = sync_scheduler_client.forward([request])
        if result.error:
            raise RuntimeError(result.error)
        return Image.open(io.BytesIO(result.output_image_bytes[0])).convert("RGB")

    def close(self) -> None:
        self.engine.shutdown()
