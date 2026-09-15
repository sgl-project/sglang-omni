# SPDX-License-Identifier: Apache-2.0
"""Native linear pipeline; session lifecycle comes from the shared framework."""

from __future__ import annotations

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
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.session import SessionScheduler


class SessionBuilder:
    scheduler_class = OmniScheduler

    def generation_defaults(self, *, dtype):
        return {
            **super().generation_defaults(dtype=dtype),
            "enable_streaming_session": True,
            "max_running_requests": 2,
            "max_total_tokens": 16384,
        }

    def extra_scheduler_kwargs(self):
        return {"session_adapter": self.adapter, "request_build_max_workers": 1}


class ThinkerBuilder(SessionBuilder, NemotronVoiceChatEngineBuilder):
    def make_model_runner(self, model_worker, output_proc):
        self.runner = DuplexThinkerRunner(model_worker, output_proc)
        return self.runner

    def make_adapters(self, model):
        import json

        from transformers import AutoTokenizer

        ordinary = super().make_adapters(model)
        prompt, pad = self._prompt_tokens()
        stt = json.loads((self._source / "config.json").read_text())["model"]["stt"][
            "model"
        ]
        tokenizer = AutoTokenizer.from_pretrained(stt["pretrained_llm"])
        self.adapter = ThinkerAdapter(
            self.runner,
            prompt_ids=prompt,
            pad_id=pad,
            tokenizer=tokenizer,
            context_length=self.context_length,
        )
        return ordinary


class TalkerBuilder(SessionBuilder, NemotronVoiceChatTalkerEngineBuilder):
    def generation_defaults(self, *, dtype):
        # EarTTS uses head_dim=72, unsupported by Blackwell's automatic
        # TRTLLM context kernel. Triton supports this dimension and page_size=1.
        return {
            **super().generation_defaults(dtype=dtype),
            "attention_backend": "triton",
            "page_size": 1,
        }

    def make_model_runner(self, model_worker, output_proc):
        self.runner = DuplexTalkerRunner(model_worker, output_proc)
        return self.runner

    def make_adapters(self, model):
        ordinary = super().make_adapters(model)
        self.adapter = TalkerAdapter(self.runner, context_length=self.context_length)
        return ordinary


def create_thinker(
    model_path,
    *,
    dtype="bfloat16",
    device=None,
    gpu_id=None,
    server_args_overrides=None,
    **overrides,
):
    return ThinkerBuilder().build(
        model_path,
        dtype=dtype,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides={**overrides, **(server_args_overrides or {})},
    )


def create_talker(
    model_path,
    *,
    dtype="bfloat16",
    device=None,
    gpu_id=None,
    server_args_overrides=None,
    **overrides,
):
    return TalkerBuilder().build(
        model_path,
        dtype=dtype,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides={**overrides, **(server_args_overrides or {})},
    )


def create_perception(model_path, *, dtype="float32", device=None, gpu_id=None):
    from sglang_omni.models.nemotron_voicechat.conformer import AudioPerception
    from sglang_omni.models.nemotron_voicechat.stages import (
        PERCEPTION_PREFIX,
        _perception_config,
    )
    from sglang_omni.models.weight_loader import load_module, resolve_dtype
    from sglang_omni.utils.device import resolve_concrete_device

    device = resolve_concrete_device(device, gpu_id)
    model = AudioPerception(_perception_config(model_path))
    load_module(
        model,
        model_path,
        prefix=PERCEPTION_PREFIX,
        dtype=resolve_dtype(dtype),
        device=device,
        strict=True,
    )
    return SessionScheduler(
        PerceptionHooks(model.eval()), max_sessions=1, max_concurrency=1
    )


def create_codec(model_path, *, dtype="float32", device=None, gpu_id=None):
    from sglang_omni.models.nemotron_voicechat.stages import create_code2wav_executor

    # Reuse checkpoint loading and marker validation from the offline factory.
    offline = create_code2wav_executor(
        model_path, dtype=dtype, device=device, gpu_id=gpu_id
    )
    return SessionScheduler(
        CodecHooks(offline._decoder, offline._device),
        max_sessions=1,
        max_concurrency=1,
    )
