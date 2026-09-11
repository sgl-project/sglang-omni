# SPDX-License-Identifier: Apache-2.0
"""PD-only engine composition for MOSS-Transcribe-Diarize."""

from __future__ import annotations

from typing import Any, Literal

from sglang.srt.managers.mm_utils import init_mm_embedding_cache

from sglang_omni.models.moss_transcribe_diarize.engine_builder import (
    MossTranscribeDiarizeEngineBuilder,
)
from sglang_omni.models.moss_transcribe_diarize.pd import (
    DECODE_STAGE,
    PREFILL_STAGE,
    request_builders,
)


class MossTranscribeDiarizePDEngineBuilder(MossTranscribeDiarizeEngineBuilder):
    """Select a PD scheduler without changing the standard MOSS-TD path."""

    def __init__(
        self,
        *,
        pd_role: Literal["prefill", "decode"],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pd_role = pd_role

    def setup_model_resources(
        self,
        model: Any,
        server_args: Any,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        if self.pd_role == "prefill":
            return super().setup_model_resources(
                model,
                server_args,
                generation_cuda_graph_enabled=generation_cuda_graph_enabled,
            )
        init_mm_embedding_cache(0)
        model.init_encoder_cache(0)

    def setup_runtime_resources(self, model: Any, server_args: Any) -> None:
        if self.pd_role == "prefill":
            super().setup_runtime_resources(model, server_args)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        return request_builders.make_scheduler_adapters(
            processor=self.processor,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            context_length=self.context_length,
            duration_scaled_default=self.requested_max_new_tokens is None,
            audio_encoder_service=self.audio_encoder_service,
        )

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        kwargs = super().extra_scheduler_kwargs()
        if self.pd_role == "prefill":
            kwargs.pop("stream_output_builder")
            kwargs["enable_async_decode"] = False
        return kwargs

    def _make_scheduler(
        self,
        *,
        model_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
        extra_scheduler_kwargs: dict[str, Any],
    ) -> Any:
        from sglang_omni.scheduling.pd_scheduler import (
            OmniDecodeScheduler,
            OmniPrefillScheduler,
        )

        scheduler_kwargs = {
            "tp_worker": model_worker,
            "tree_cache": tree_cache,
            "req_to_token_pool": req_to_token_pool,
            "token_to_kv_pool_allocator": token_to_kv_pool_allocator,
            "server_args": server_args,
            "model_config": model_config,
            "model_runner": model_runner,
            "request_builder": request_builder,
            "result_adapter": result_adapter,
            "abort_callback": self.make_abort_callback(),
            "request_finished_callback": self.make_request_finished_callback(),
        }
        scheduler_kwargs.update(self.extra_scheduler_callbacks())
        scheduler_kwargs.update(extra_scheduler_kwargs)
        state_builder, state_restorer = request_builders.make_state_adapters()
        if self.pd_role == "prefill":
            return OmniPrefillScheduler(
                **scheduler_kwargs,
                stage_name=PREFILL_STAGE,
                partner_stage=DECODE_STAGE,
                state_builder=state_builder,
            )
        return OmniDecodeScheduler(
            **scheduler_kwargs,
            stage_name=DECODE_STAGE,
            state_restorer=state_restorer,
            resume_schema=request_builders.MOSS_TD_PD_RESUME_SCHEMA,
        )


__all__ = ["MossTranscribeDiarizePDEngineBuilder"]
