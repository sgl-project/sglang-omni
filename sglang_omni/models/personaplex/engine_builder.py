# SPDX-License-Identifier: Apache-2.0
"""Starts the SGLang engine for PersonaPlex from a Llama-shaped shim checkpoint."""

from __future__ import annotations

import atexit
import json
import shutil
import tempfile
from pathlib import Path

from sglang_omni.models.personaplex.architecture import MOSHI_WEIGHTS_NAME
from sglang_omni.models.personaplex.hf_config import (
    DEFAULT_CONTEXT_LENGTH,
    PERSONAPLEX_ARCH,
    build_backbone_config,
)
from sglang_omni.models.personaplex.model_runner import PersonaPlexModelRunner
from sglang_omni.models.personaplex.prompts import VoicePrompt
from sglang_omni.models.personaplex.request_builders import (
    apply_lm_result,
    build_lm_request,
    lm_stream_output_builder,
)
from sglang_omni.models.personaplex.session import PersonaPlexSessionAdapter
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


def shim_checkpoint_dir(source: Path, *, context_length: int) -> Path:
    """A directory SGLang can load: the LM weights and a Llama config.

    Only model.safetensors is linked. The Mimi weights sit next to it in
    the checkpoint and SGLang would otherwise sweep them into the LM.
    """
    source = source.resolve()
    weights = source / MOSHI_WEIGHTS_NAME
    if not weights.is_file():
        raise FileNotFoundError(f"PersonaPlex LM weights missing: {weights}")
    else:
        pass
    shim = Path(tempfile.mkdtemp(prefix="sglang-omni-personaplex-"))
    atexit.register(shutil.rmtree, shim, ignore_errors=True)
    (shim / MOSHI_WEIGHTS_NAME).symlink_to(weights)
    (shim / "config.json").write_text(
        json.dumps(build_backbone_config(context_length=context_length), indent=2)
    )
    return shim


class PersonaPlexEngineBuilder(TtsEngineBuilder):
    model_name = "personaplex"
    context_length = DEFAULT_CONTEXT_LENGTH
    model_arch_override = PERSONAPLEX_ARCH
    supports_context_length_override = True

    def resolve_checkpoint(self, model_path):
        source = Path(resolve_model_path(model_path))
        return str(shim_checkpoint_dir(source, context_length=self.context_length))

    def generation_defaults(self, *, dtype):
        return {
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            # Note (wilsonzheng0327): Offline, one conversation at a time; the
            # lm stage's engine.max_running_requests overrides it.
            "max_running_requests": 1,
            "chunked_prefill_size": -1,
            "dtype": dtype,
            "trust_remote_code": False,
            # Note (wilsonzheng0327): Seeded top-k sampling needs the torch sampler.
            "sampling_backend": "pytorch",
        }

    def setup_model(self, *, model_worker, checkpoint_dir, device, gpu_id, server_args):
        """Nothing beyond SGLang's own load: the model owns its buffers."""

    def make_model_runner(self, model_worker, output_proc):
        return PersonaPlexModelRunner(model_worker, output_proc)

    def make_adapters(self, model):
        vocab_size = int(model.config.vocab_size)
        voice_cache: dict[str, VoicePrompt] = {}

        def build(payload):
            return build_lm_request(
                payload,
                vocab_size=vocab_size,
                voice_cache=voice_cache,
                context_length=self.context_length,
            )

        return build, apply_lm_result

    def extra_scheduler_kwargs(self):
        return {"stream_output_builder": lm_stream_output_builder}


class PersonaPlexRealtimeEngineBuilder(PersonaPlexEngineBuilder):
    """The LM of a full-duplex deployment: one SGLang streaming session per call."""

    session_adapter: PersonaPlexSessionAdapter | None = None

    def generation_defaults(self, *, dtype):
        return {
            **super().generation_defaults(dtype=dtype),
            "enable_streaming_session": True,
        }

    def make_adapters(self, model):
        # Note (wilsonzheng0327): build_runtime asks for the adapters before the
        # scheduler kwargs, so the session adapter can take the model's vocabulary.
        self.session_adapter = PersonaPlexSessionAdapter(
            vocab_size=int(model.config.vocab_size),
            context_length=self.context_length,
        )
        return super().make_adapters(model)

    def extra_scheduler_kwargs(self):
        assert self.session_adapter is not None
        return {"session_adapter": self.session_adapter}
