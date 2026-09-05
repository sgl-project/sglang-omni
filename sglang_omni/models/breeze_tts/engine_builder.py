# SPDX-License-Identifier: Apache-2.0
"""Build the Breeze SGLang engine without changing the downloaded checkpoint."""

import torch
from accelerate import init_empty_weights

from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

from .checkpoint import load_component, read_config
from .depth_decoder import BreezeDepthDecoder
from .hf_config import register_breeze_config


class BreezeEngineBuilder(TtsEngineBuilder):
    model_name = "Breeze-TTS-2"
    model_arch_override = "BreezeSGLangModel"
    context_length = 1024

    def pre_infra_setup(self, checkpoint_dir):
        read_config(checkpoint_dir)
        register_breeze_config()

    def generation_defaults(self, *, dtype):
        return {
            "max_running_requests": 2,
            "max_queued_requests": 64,
            "max_prefill_tokens": 2048,
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "chunked_prefill_size": 0,
            "mem_fraction_static": 0.60,
            "sampling_backend": "pytorch",
            "dtype": dtype,
        }

    def adjust_overrides(self, overrides):
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("Breeze-TTS-2 initial serving supports TP=1 only")
        if int(overrides["max_running_requests"]) != 2:
            raise ValueError(
                "Breeze-TTS-2 reserves exactly two AR rows for one CFG request"
            )
        for field, expected in (
            ("disable_cuda_graph", True),
            ("disable_overlap_schedule", True),
            ("disable_radix_cache", True),
            ("enable_torch_compile", False),
            ("chunked_prefill_size", 0),
        ):
            if overrides.get(field) != expected:
                raise ValueError(
                    f"Breeze-TTS-2 initial serving requires {field}={expected}"
                )
        if overrides["dtype"] != "bfloat16":
            raise ValueError("Breeze-TTS-2 initial serving requires bfloat16")
        if int(overrides["max_prefill_tokens"]) < 2 * self.context_length:
            raise ValueError("Breeze-TTS-2 requires max_prefill_tokens >= 2048")
        if overrides.get("quantization") is not None:
            raise ValueError("Breeze-TTS-2 quantization is not implemented")

    def setup_model(self, *, model_worker, checkpoint_dir, device, gpu_id, server_args):
        del gpu_id, server_args
        if torch.device(device).type != "cuda":
            raise ValueError("Breeze-TTS-2 serving currently requires CUDA")
        model = model_worker.model_runner.model
        raw = read_config(checkpoint_dir)
        with init_empty_weights(include_buffers=False):
            depth = BreezeDepthDecoder(raw["depth_decoder_config"])
        load_component(depth, checkpoint_dir, "depth_decoder.")
        model.depth_decoder = depth.to(device=device, dtype=torch.bfloat16).eval()
        model.lm_head.float()

    def make_model_runner(self, model_worker, output_proc):
        from .model_runner import BreezeModelRunner

        return BreezeModelRunner(model_worker, output_proc)

    def make_adapters(self, model):
        from .request_builders import apply_result, build_request

        return lambda payload: build_request(payload, model), apply_result

    def make_scheduler(self, **kwargs):
        from .request_builders import stream_output
        from .scheduler import BreezeScheduler

        return BreezeScheduler(
            tp_worker=kwargs.pop("model_worker"),
            **kwargs,
            stream_output_builder=stream_output,
            enable_async_decode=False,
        )
