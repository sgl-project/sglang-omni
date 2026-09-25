# SPDX-License-Identifier: Apache-2.0
"""Native linear pipeline; session lifecycle comes from the shared framework."""

from __future__ import annotations

import json
from typing import Protocol

import torch
from transformers import AutoTokenizer

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.nemotron_voicechat.code2wav_stream import DECODE_WINDOW_FRAMES
from sglang_omni.models.nemotron_voicechat.conformer import AudioPerception
from sglang_omni.models.nemotron_voicechat.duplex import CodecHooks, PerceptionHooks
from sglang_omni.models.nemotron_voicechat.duplex_ar import (
    DuplexTalkerRunner,
    DuplexThinkerRunner,
    TalkerAdapter,
    ThinkerAdapter,
)
from sglang_omni.models.nemotron_voicechat.engine_builder import (
    NemotronVoiceChatEngineBuilder,
    NemotronVoiceChatTalkerEngineBuilder,
)
from sglang_omni.models.nemotron_voicechat.stages import (
    PERCEPTION_PREFIX,
    create_code2wav_executor,
    perception_config,
)
from sglang_omni.models.weight_loader import load_module, resolve_dtype
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.session import SessionScheduler
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionAdapter
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.utils.device import resolve_concrete_device

EngineOverrideValue = str | int | float | bool | list[int] | list[str] | None


class OfflineRequestBuilder(Protocol):
    def __call__(self, payload: StagePayload, /) -> SGLangARRequestData: ...


class OfflineResultAdapter(Protocol):
    def __call__(self, request_data: SGLangARRequestData, /) -> StagePayload: ...


class DuplexSessionBuilderMixin:
    """Share session settings within the existing TtsEngineBuilder lifecycle."""

    adapter: ARSessionAdapter
    scheduler_class = OmniScheduler

    def generation_defaults(self, *, dtype: str) -> dict[str, str | int | bool]:
        return {
            **super().generation_defaults(dtype=dtype),
            "enable_streaming_session": True,
            # note (Codex): The session bridge reserves one slot for retained KV; allow a new unit too.
            "max_running_requests": 2,
            # note (Codex): Fixed KV budget in token positions, not a concurrent-session limit.
            "max_total_tokens": 16384,
        }

    def extra_scheduler_kwargs(self) -> dict[str, ARSessionAdapter | int]:
        # note (Codex): Adapter builds mutate per-session history and must stay ordered.
        return {"session_adapter": self.adapter, "request_build_max_workers": 1}


class ThinkerBuilder(DuplexSessionBuilderMixin, NemotronVoiceChatEngineBuilder):
    runner: DuplexThinkerRunner
    adapter: ThinkerAdapter

    def generation_defaults(self, *, dtype: str) -> dict[str, str | int | bool]:
        # Native continuations extend a single position. Avoid TRTLLM's
        # minimum 64-token prefill padding for each 80 ms audio frame.
        return {
            **super().generation_defaults(dtype=dtype),
            "attention_backend": "triton",
            "page_size": 1,
        }

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> DuplexThinkerRunner:
        self.runner = DuplexThinkerRunner(model_worker, output_proc)
        return self.runner

    def make_adapters(
        self, model: torch.nn.Module
    ) -> tuple[OfflineRequestBuilder, OfflineResultAdapter]:
        offline_adapters = super().make_adapters(model)
        prompt_token_ids, pad_token_id = self.prompt_tokens()
        stt = json.loads((self.source / "config.json").read_text())["model"]["stt"][
            "model"
        ]
        tokenizer = AutoTokenizer.from_pretrained(stt["pretrained_llm"])
        self.adapter = ThinkerAdapter(
            self.runner,
            prompt_token_ids=prompt_token_ids,
            pad_token_id=pad_token_id,
            tokenizer=tokenizer,
            context_length=self.context_length,
        )
        return offline_adapters


class TalkerBuilder(DuplexSessionBuilderMixin, NemotronVoiceChatTalkerEngineBuilder):
    runner: DuplexTalkerRunner
    adapter: TalkerAdapter

    def generation_defaults(self, *, dtype: str) -> dict[str, str | int | bool]:
        # EarTTS uses head_dim=72, unsupported by Blackwell's automatic
        # TRTLLM context kernel. Triton supports this dimension and page_size=1.
        return {
            **super().generation_defaults(dtype=dtype),
            "attention_backend": "triton",
            "page_size": 1,
        }

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> DuplexTalkerRunner:
        self.runner = DuplexTalkerRunner(model_worker, output_proc)
        return self.runner

    def make_adapters(
        self, model: torch.nn.Module
    ) -> tuple[OfflineRequestBuilder, OfflineResultAdapter]:
        offline_adapters = super().make_adapters(model)
        self.adapter = TalkerAdapter(self.runner, context_length=self.context_length)
        return offline_adapters


def create_thinker(
    model_path: str,
    *,
    dtype: str = "bfloat16",
    device: str | None = None,
    gpu_id: int | None = None,
    server_args_overrides: dict[str, EngineOverrideValue] | None = None,
    **overrides: EngineOverrideValue,
) -> OmniScheduler:
    return ThinkerBuilder().build(
        model_path,
        dtype=dtype,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides={**overrides, **(server_args_overrides or {})},
    )


def create_talker(
    model_path: str,
    *,
    dtype: str = "bfloat16",
    device: str | None = None,
    gpu_id: int | None = None,
    server_args_overrides: dict[str, EngineOverrideValue] | None = None,
    **overrides: EngineOverrideValue,
) -> OmniScheduler:
    return TalkerBuilder().build(
        model_path,
        dtype=dtype,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides={**overrides, **(server_args_overrides or {})},
    )


def create_perception(
    model_path: str,
    *,
    dtype: str = "float32",
    device: str | None = None,
    gpu_id: int | None = None,
) -> SessionScheduler:
    device = resolve_concrete_device(device, gpu_id)
    model = AudioPerception(perception_config(model_path))
    load_module(
        model,
        model_path,
        prefix=PERCEPTION_PREFIX,
        dtype=resolve_dtype(dtype),
        device=device,
        strict=True,
    )
    return SessionScheduler(
        PerceptionHooks(model.eval()), max_open_sessions=1, max_concurrency=1
    )


def create_codec(
    model_path: str,
    *,
    dtype: str = "float32",
    device: str | None = None,
    gpu_id: int | None = None,
) -> SessionScheduler:

    # Reuse checkpoint loading and marker validation from the offline factory.
    offline = create_code2wav_executor(
        model_path, dtype=dtype, device=device, gpu_id=gpu_id
    )
    hooks = CodecHooks(offline.decoder, offline.device)
    # Capture the steady-state codec before accepting a live microphone.
    if torch.device(offline.device).type == "cuda":
        hooks.decode(
            torch.zeros(
                DECODE_WINDOW_FRAMES,
                offline.decoder.silence_codes.numel(),
                dtype=torch.long,
                device=offline.device,
            )
        )
    else:
        pass
    return SessionScheduler(
        hooks,
        max_open_sessions=1,
        max_concurrency=1,
    )
