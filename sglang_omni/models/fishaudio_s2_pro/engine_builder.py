# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro SGLang engine builder."""

from __future__ import annotations

import importlib
import os
from typing import Any

from sglang_omni.models.fishaudio_s2_pro import request_builders
from sglang_omni.models.fishaudio_s2_pro import stages as fish_stages
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


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

    def resolve_checkpoint(self, model_path: str) -> str:
        return fish_stages._resolve_checkpoint(model_path)

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
        return {
            "max_running_requests": 64,
            "disable_cuda_graph": False,
            "mem_fraction_static": 0.85,
            "chunked_prefill_size": 8192,
            "dtype": "bfloat16",
            "enable_torch_compile": True,
            "random_seed": int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF,
        }

    def customize_server_args(self, server_args: Any) -> None:
        server_args.disable_overlap_schedule = True
        if getattr(server_args, "attention_backend", None) is None:
            server_args.attention_backend = "fa3"

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
        if bool(getattr(server_args, "enable_torch_compile", False)):
            fish_stages._compile_s2pro_codebook_decoder(
                model,
                max_batch_size=server_args.torch_compile_max_bs,
            )
            server_args.enable_torch_compile = False

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.fishaudio_s2_pro.model_runner"
        )

        return model_runner_mod.FishS2ProModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        return request_builders.make_tts_scheduler_adapters(tokenizer=self.tokenizer)

    def make_scheduler(
        self,
        *,
        model_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        prefill_manager: Any,
        decode_manager: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
    ) -> Any:
        del model_worker, model_config
        fish_scheduler_mod = importlib.import_module(
            "sglang_omni.models.fishaudio_s2_pro.fish_scheduler"
        )

        return fish_scheduler_mod.FishScheduler(
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            prefill_manager=prefill_manager,
            decode_manager=decode_manager,
            server_args=server_args,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            im_end_token_id=self.adapter.eos_token_ids[0],
            max_new_tokens=self.max_new_tokens,
        )
