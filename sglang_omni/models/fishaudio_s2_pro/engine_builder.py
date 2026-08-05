# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro SGLang engine builder."""

from __future__ import annotations

import importlib
import os
from typing import Any

from sglang_omni.models.fishaudio_s2_pro import request_builders
from sglang_omni.models.fishaudio_s2_pro import stages as fish_stages
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version
from sglang_omni.vendor.sglang.server_args import override_server_args
from sglang_omni.vendor.sglang.utils import is_flashinfer_available

_VALIDATED_AUTO_ATTENTION_BACKENDS = {
    89: "flashinfer",
    90: "fa3",
    100: "flashinfer",
    120: "flashinfer",
}


def _resolve_fast_ar_attention_backend(*, gpu_id: int) -> str:
    sm_version = get_visible_gpu_sm_version(gpu_id)
    if sm_version is None:
        raise RuntimeError(
            "FishAudio S2-Pro cannot validate Fast-AR attention because "
            f"CUDA compute capability for gpu_id={gpu_id} could not be detected."
        )

    backend = _VALIDATED_AUTO_ATTENTION_BACKENDS.get(sm_version)
    if backend is None:
        raise RuntimeError(
            f"FishAudio S2-Pro Fast-AR does not support SM{sm_version}; "
            "supported architectures are SM89, SM90, SM100, and SM120. "
            "A Slow-AR attention_backend override cannot bypass this requirement."
        )

    if backend == "flashinfer" and not is_flashinfer_available():
        raise RuntimeError(
            f"FishAudio S2-Pro Fast-AR requires FlashInfer on SM{sm_version}, but "
            "FlashInfer is unavailable. Install and enable FlashInfer and ensure "
            "SGLANG_IS_FLASHINFER_AVAILABLE is not false; a Slow-AR "
            "attention_backend override cannot bypass this requirement."
        )
    return backend


class FishS2ProEngineBuilder(TtsEngineBuilder):
    model_name = "FishAudio S2-Pro"
    context_length = 4096

    def __init__(
        self,
        *,
        max_new_tokens: int,
        ras_window: int,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.ras_window = ras_window
        self.adapter: Any | None = None
        self.tokenizer: Any | None = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        from sglang_omni.models.fishaudio_s2_pro import bootstrap as fish_bootstrap

        fish_bootstrap.patch_fish_config_for_sglang()

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        del dtype
        sm_version = get_visible_gpu_sm_version(self.gpu_id)
        return {
            "max_running_requests": 64,
            "disable_cuda_graph": False,
            "mem_fraction_static": 0.85,
            "chunked_prefill_size": 8192,
            "dtype": "bfloat16",
            # FlashInfer Fast-AR compile+graph replay is not yet validated with
            # trained S2-Pro weights. Keep those decoder layers uncompiled.
            "enable_torch_compile": sm_version == 90,
            "random_seed": int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF,
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        fast_ar_backend = _resolve_fast_ar_attention_backend(gpu_id=self.gpu_id)
        if overrides.get("attention_backend") is None:
            overrides["attention_backend"] = fast_ar_backend

    def customize_server_args(self, server_args: Any) -> None:
        updates: dict[str, Any] = {"disable_overlap_schedule": True}
        override_server_args(
            server_args,
            "sglang_omni.fishaudio_s2_pro.runtime_defaults",
            **updates,
        )

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del gpu_id
        from sglang_omni.models.fishaudio_s2_pro import bootstrap as fish_bootstrap
        from sglang_omni.models.fishaudio_s2_pro.tokenizer import S2ProTokenizerAdapter

        model = model_worker.model_runner.model
        fish_bootstrap.truncate_rope_to_bf16(model)
        audio_decoder, num_codebooks, codebook_size, tokenizer = (
            fish_bootstrap.load_audio_decoder(
                checkpoint_dir,
                device=device,
            )
        )
        self.tokenizer = tokenizer
        self.adapter = S2ProTokenizerAdapter(tokenizer)
        fish_bootstrap.bootstrap_text_model_for_decode(
            text_model=model,
            audio_decoder=audio_decoder,
            semantic_begin_id=self.adapter.semantic_begin_id,
            semantic_end_id=self.adapter.semantic_end_id,
            im_end_token_id=self.adapter.eos_token_ids[0],
            max_batch_size=server_args.max_running_requests,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            ras_window=self.ras_window,
        )

    def get_model_buffer_bs(self, model: Any) -> int | None:
        return fish_stages._resolve_s2pro_model_buffer_bs(model)

    def compile_model(self, model: Any, server_args: Any) -> None:
        if bool(server_args.enable_torch_compile):
            fish_stages._compile_s2pro_codebook_decoder(
                model,
                max_batch_size=server_args.torch_compile_max_bs,
            )
            override_server_args(
                server_args,
                "sglang_omni.fishaudio_s2_pro.compile_complete",
                enable_torch_compile=False,
            )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.fishaudio_s2_pro.model_runner"
        )

        return model_runner_mod.FishS2ProModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        request_builder, result_adapter, self._stream_output_builder = (
            request_builders.make_tts_scheduler_adapters(
                tokenizer=self.tokenizer,
                max_new_tokens_cap=self.max_new_tokens,
                context_length=self.context_length,
            )
        )
        return request_builder, result_adapter

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {"stream_output_builder": self._stream_output_builder}
