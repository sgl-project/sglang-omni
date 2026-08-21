# SPDX-License-Identifier: Apache-2.0
"""In-process SGLang engine builders for Nemotron VoiceChat."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang_omni.scheduling.engine_factory import SGLangGenerationEngineBuilder

from .request_builders import TalkerAdapters, ThinkerAdapters


def resolve_stage_path(model_path: str, stage: str, explicit: str | None) -> str:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
    else:
        root = Path(model_path).expanduser().resolve()
        candidates = (root / "converted" / stage, root / stage)
        path = next(
            (candidate for candidate in candidates if candidate.is_dir()), candidates[0]
        )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"VoiceChat {stage} config not found at {path / 'config.json'}; "
            "model_path must point to the deployment root containing "
            "checkpoint/ and converted/{duplex,eartts}/"
        )
    return str(path)


class _VoiceChatEngineBuilder(SGLangGenerationEngineBuilder):
    context_length = 8192

    def __init__(
        self,
        *,
        context_length: int,
        max_sessions: int,
        total_gpu_memory_fraction: float | None,
    ) -> None:
        self.context_length = int(context_length)
        self.max_sessions = int(max_sessions)
        if self.max_sessions < 1:
            raise ValueError("VoiceChat max_sessions must be positive")
        self.total_gpu_memory_fraction = total_gpu_memory_fraction

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            # Every persistent streaming session retains one request-pool slot
            # for committed KV while its next turn needs a transient slot.
            "max_running_requests": 2 * self.max_sessions,
            "dtype": dtype,
            "disable_overlap_schedule": True,
            # Acoustic, function-token, and codec inputs change while token IDs
            # remain fixed, so replaying token-keyed graphs would be incorrect.
            "disable_cuda_graph": True,
            "enable_streaming_session": True,
            "max_prefill_tokens": self.context_length,
            "mem_fraction_static": self.total_gpu_memory_fraction,
            "trust_remote_code": False,
        }

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from .model_runner import VoiceChatModelRunner

        return VoiceChatModelRunner(model_worker, output_proc)

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        # Session admission mutates append-only state and must remain ordered.
        return {
            "request_build_max_workers": 1,
            "enable_streaming_sessions": True,
        }


class VoiceChatThinkerEngineBuilder(_VoiceChatEngineBuilder):
    model_name = "Nemotron VoiceChat thinker"

    def __init__(self, *, duplex_model_path: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.duplex_model_path = duplex_model_path
        self.config: Any = None
        self.tokenizer: Any = None

    def resolve_checkpoint(self, model_path: str) -> str:
        return resolve_stage_path(model_path, "duplex", self.duplex_model_path)

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from .registration import register_voicechat_models

        register_voicechat_models()
        self.config = AutoConfig.from_pretrained(
            checkpoint_dir, trust_remote_code=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=False
        )

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        adapters = ThinkerAdapters(
            config=self.config,
            tokenizer=self.tokenizer,
            context_length=self.context_length,
        )
        return adapters.request_builder, adapters.result_adapter


class VoiceChatTalkerEngineBuilder(_VoiceChatEngineBuilder):
    model_name = "Nemotron VoiceChat talker"

    def __init__(
        self,
        *,
        eartts_model_path: str | None = None,
        speaker_latent_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.eartts_model_path = eartts_model_path
        self.speaker_latent_path = speaker_latent_path
        self.config: Any = None
        self.speaker: torch.Tensor | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        return resolve_stage_path(model_path, "eartts", self.eartts_model_path)

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        if dtype != "float32":
            raise ValueError("VoiceChat EarTTS requires dtype='float32'")
        defaults = super().generation_defaults(dtype=dtype)
        # NVIDIA's reference enables TensorFloat32 for EarTTS. Preserve FP32
        # storage/range while allowing its large matmuls to use tensor cores.
        defaults["enable_tf32_matmul"] = True
        defaults["attention_backend"] = "torch_native"
        defaults["chunked_prefill_size"] = -1
        return defaults

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from .registration import register_voicechat_models

        register_voicechat_models()
        self.config = AutoConfig.from_pretrained(
            checkpoint_dir, trust_remote_code=False
        )
        speaker_path = self.speaker_latent_path
        if speaker_path is None:
            candidates = sorted((Path(checkpoint_dir) / "speaker_latents").glob("*.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"No VoiceChat speaker latent found under {checkpoint_dir}/speaker_latents"
                )
            speaker_path = str(candidates[0])
        speaker = torch.load(speaker_path, map_location="cpu", weights_only=True)
        if speaker.ndim == 3 and speaker.shape[0] == 1:
            speaker = speaker.squeeze(0)
        if speaker.ndim != 2:
            raise ValueError(
                f"VoiceChat speaker latent must be rank 2, got {tuple(speaker.shape)}"
            )
        self.speaker = speaker

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        assert self.speaker is not None
        adapters = TalkerAdapters(
            config=self.config,
            speaker=self.speaker,
            context_length=self.context_length,
        )
        return adapters.request_builder, adapters.result_adapter


__all__ = ["VoiceChatTalkerEngineBuilder", "VoiceChatThinkerEngineBuilder"]
