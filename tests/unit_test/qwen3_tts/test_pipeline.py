# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from collections import deque
from queue import Empty, Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.config.runtime import resolve_stage_factory_kwargs
from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.qwen3_omni.pending_text_queue import PendingTextTensorQueue
from sglang_omni.models.qwen3_tts import request_builders as qwen3_request_builders
from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.models.qwen3_tts import streaming_vocoder as qwen3_streaming_vocoder
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalCodecState
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.qwen3_tts.request_builders import (
    Qwen3TTSPreparedRequest,
    Qwen3TTSSGLangRequestData,
    apply_sglang_qwen3_tts_result,
    build_embedding_cache_key_ids,
    build_qwen3_tts_state,
    build_sglang_qwen3_tts_request,
    derive_qwen3_tts_sampling_seeds,
)
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    Qwen3TTSStreamingVocoderScheduler,
    _Qwen3TTSDecodePlan,
    _Qwen3TTSInitialDecodeGraphs,
    _Qwen3TTSInvalidCodeRows,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.sampling import seed as sampling_seed
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.speaker_cache import (
    SpeakerCacheKey,
    get_speaker_artifact_cache,
)
from sglang_omni.scheduling.types import RequestOutput
from sglang_omni.utils import cuda_staging
from tests.unit_test.fakes import FakeExecutionBridge


def install_fake_sglang(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        import sglang.srt.managers.schedule_batch  # noqa: F401
        import sglang.srt.managers.scheduler  # noqa: F401
        import sglang.srt.sampling.sampling_params  # noqa: F401

        return
    except ImportError:
        pass

    class FakeReq:
        def __init__(
            self,
            *,
            rid,
            origin_input_text,
            origin_input_ids,
            sampling_params,
            eos_token_ids=None,
            vocab_size=None,
            extra_key=None,
            **kwargs,
        ) -> None:
            del kwargs
            self.rid = rid
            self.origin_input_text = origin_input_text
            self.origin_input_ids = origin_input_ids
            self.sampling_params = sampling_params
            self.eos_token_ids = eos_token_ids
            self.vocab_size = vocab_size
            self.extra_key = extra_key
            self.output_ids = []
            self.prefix_indices = []
            self.extend_range = SimpleNamespace(length=len(origin_input_ids))

        def reset_for_retract(self) -> None:
            self.prefix_indices = []
            self.extend_range = None

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)
            self.min_p = kwargs.get("min_p", 0.0)

        def normalize(self, tokenizer) -> None:
            del tokenizer

        def verify(self, vocab_size) -> None:
            self.vocab_size = vocab_size

    class FakeGenerationBatchResult:
        def __init__(self, *, logits_output=None, can_run_cuda_graph=False) -> None:
            self.logits_output = logits_output
            self.can_run_cuda_graph = can_run_cuda_graph
            self.next_token_ids = None

    class FakeLogitsProcessorOutput:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class FakeSamplingBatchInfo:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    def default_weight_loader(*args, **kwargs) -> None:
        del args, kwargs

    def add_prefix(name: str, prefix: str = "") -> str:
        return f"{prefix}.{name}" if prefix else name

    sampler_calls = []

    def multinomial_with_seed(inputs, seed, positions):
        sampler_calls.append(
            {
                "inputs": inputs.detach().clone(),
                "seed": seed.detach().clone(),
                "positions": positions.detach().clone(),
            }
        )
        return torch.zeros((inputs.shape[0], 1), device=inputs.device, dtype=torch.long)

    modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.managers": types.ModuleType("sglang.srt.managers"),
        "sglang.srt.managers.schedule_batch": types.ModuleType(
            "sglang.srt.managers.schedule_batch"
        ),
        "sglang.srt.managers.scheduler": types.ModuleType(
            "sglang.srt.managers.scheduler"
        ),
        "sglang.srt.layers": types.ModuleType("sglang.srt.layers"),
        "sglang.srt.layers.logits_processor": types.ModuleType(
            "sglang.srt.layers.logits_processor"
        ),
        "sglang.srt.layers.sampler": types.ModuleType("sglang.srt.layers.sampler"),
        "sglang.srt.model_executor": types.ModuleType("sglang.srt.model_executor"),
        "sglang.srt.model_executor.runner_backend_utils": types.ModuleType(
            "sglang.srt.model_executor.runner_backend_utils"
        ),
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph": (
            types.ModuleType(
                "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph"
            )
        ),
        "sglang.srt.model_loader": types.ModuleType("sglang.srt.model_loader"),
        "sglang.srt.model_loader.weight_utils": types.ModuleType(
            "sglang.srt.model_loader.weight_utils"
        ),
        "sglang.srt.sampling": types.ModuleType("sglang.srt.sampling"),
        "sglang.srt.sampling.sampling_batch_info": types.ModuleType(
            "sglang.srt.sampling.sampling_batch_info"
        ),
        "sglang.srt.sampling.sampling_params": types.ModuleType(
            "sglang.srt.sampling.sampling_params"
        ),
        "sglang.srt.utils": types.ModuleType("sglang.srt.utils"),
        "sgl_kernel": types.ModuleType("sgl_kernel"),
    }
    for package_name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.managers",
        "sglang.srt.layers",
        "sglang.srt.model_executor",
        "sglang.srt.model_executor.runner_backend_utils",
        "sglang.srt.model_loader",
        "sglang.srt.sampling",
    ):
        modules[package_name].__path__ = []
    modules["sglang"].srt = modules["sglang.srt"]
    modules["sglang.srt"].managers = modules["sglang.srt.managers"]
    modules["sglang.srt"].layers = modules["sglang.srt.layers"]
    modules["sglang.srt"].model_executor = modules["sglang.srt.model_executor"]
    modules["sglang.srt"].model_loader = modules["sglang.srt.model_loader"]
    modules["sglang.srt"].sampling = modules["sglang.srt.sampling"]
    modules["sglang.srt"].utils = modules["sglang.srt.utils"]
    modules["sglang.srt.managers"].schedule_batch = modules[
        "sglang.srt.managers.schedule_batch"
    ]
    modules["sglang.srt.managers"].scheduler = modules["sglang.srt.managers.scheduler"]
    modules["sglang.srt.layers"].logits_processor = modules[
        "sglang.srt.layers.logits_processor"
    ]
    modules["sglang.srt.layers"].sampler = modules["sglang.srt.layers.sampler"]
    modules["sglang.srt.model_executor"].runner_backend_utils = modules[
        "sglang.srt.model_executor.runner_backend_utils"
    ]
    modules["sglang.srt.model_executor.runner_backend_utils"].breakable_cuda_graph = (
        modules["sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph"]
    )
    modules["sglang.srt.model_loader"].weight_utils = modules[
        "sglang.srt.model_loader.weight_utils"
    ]
    modules["sglang.srt.sampling"].sampling_batch_info = modules[
        "sglang.srt.sampling.sampling_batch_info"
    ]
    modules["sglang.srt.sampling"].sampling_params = modules[
        "sglang.srt.sampling.sampling_params"
    ]
    modules["sgl_kernel"].fused_qk_norm_rope = lambda *args, **kwargs: None
    modules["sglang.srt.managers.schedule_batch"].Req = FakeReq
    modules["sglang.srt.managers.scheduler"].GenerationBatchResult = (
        FakeGenerationBatchResult
    )
    modules["sglang.srt.layers.logits_processor"].LogitsProcessorOutput = (
        FakeLogitsProcessorOutput
    )
    modules["sglang.srt.layers.sampler"].multinomial_with_seed = multinomial_with_seed
    modules["sglang.srt.layers.sampler"].sampler_calls = sampler_calls
    modules[
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph"
    ].eager_on_graph = lambda enabled: lambda function: function
    modules["sglang.srt.model_loader.weight_utils"].default_weight_loader = (
        default_weight_loader
    )
    modules["sglang.srt.sampling.sampling_batch_info"].SamplingBatchInfo = (
        FakeSamplingBatchInfo
    )
    modules["sglang.srt.sampling.sampling_params"].SamplingParams = FakeSamplingParams
    modules["sglang.srt.utils"].add_prefix = add_prefix
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def make_payload(
    *,
    inputs,
    params: dict | None = None,
    tts_params: dict | None = None,
) -> StagePayload:
    return StagePayload(
        request_id="req-qwen3-tts",
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def test_qwen3_tts_config_and_registry_contracts() -> None:
    config = Qwen3TTSPipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.stages[1].factory_path.endswith("create_sglang_tts_engine_executor")
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"tts_engine": 0, "vocoder": 0}
    assert config.stages[1].factory.device is None
    assert config.stages[2].factory.device is None
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert config.stages[1].stream_to == ["vocoder"]
    assert config.stages[2].can_accept_stream_before_payload is True
    assert Qwen3TTSPipelineConfig.stage_config_cls("tts_engine").engine_stage
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Qwen3TTSForConditionalGeneration")
        is Qwen3TTSPipelineConfig
    )


def test_qwen3_tts_deterministic_inference_configures_pipeline() -> None:
    """Propagate deterministic inference across the pipeline."""
    config = Qwen3TTSPipelineConfig(
        model_path="model",
        enable_deterministic_inference=True,
    )
    stages = {stage.name: stage for stage in config.stages}

    preprocessing = resolve_stage_factory_kwargs(stages["preprocessing"], config)
    tts_engine = resolve_stage_factory_kwargs(stages["tts_engine"], config)
    vocoder = resolve_stage_factory_kwargs(stages["vocoder"], config)

    assert preprocessing["max_concurrency"] == 1
    assert tts_engine["server_args_overrides"]["enable_deterministic_inference"]
    assert vocoder["enable_deterministic_inference"]
    assert vocoder["initial_cuda_graph"] is False
    assert vocoder["followup_cuda_graph"] is False


def test_qwen3_tts_breakable_prefill_is_compatible_but_not_default_enabled() -> None:
    """Compatibility permits explicit opt-in without changing shipped defaults."""
    from sglang_omni.models.qwen3_tts import CAPABILITIES
    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

    builder = Qwen3TtsEngineBuilder()
    defaults = builder.generation_defaults(dtype="bfloat16")

    assert CAPABILITIES.supports_breakable_prefill_cuda_graph is True
    assert (
        type(builder).supports_breakable_prefill_cuda_graph
        is CAPABILITIES.supports_breakable_prefill_cuda_graph
    )
    assert "cuda_graph_backend_prefill" not in defaults
    assert "cuda_graph_bs_prefill" not in defaults
    assert defaults["disable_cuda_graph"] is False


def test_qwen3_tts_breakable_prefill_breaks_around_qk_norm_rope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts import sglang_model

    events: list[tuple[str, object]] = []

    def fake_eager_on_graph(enabled: bool):
        events.append(("enabled", enabled))

        def decorate(function):
            def wrapped(*args, **kwargs):
                events.append(("wrapped", (args, kwargs)))
                return function(*args, **kwargs)

            return wrapped

        return decorate

    class FakeAttention(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()

        def apply_qk_norm_rope(self, qkv, positions, forward_batch):
            events.append(("inner", (qkv, positions, forward_batch)))
            return qkv, positions, forward_batch

    class FakeModule(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    monkeypatch.setattr(sglang_model, "eager_on_graph", fake_eager_on_graph)
    monkeypatch.setattr(
        sglang_model,
        "Qwen3OmniMoeThinkerTextAttention",
        FakeAttention,
    )
    monkeypatch.setattr(sglang_model, "Qwen3OmniMoeTalkerDenseMLP", FakeModule)
    monkeypatch.setattr(sglang_model, "RMSNorm", FakeModule)

    config = SimpleNamespace(
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        rope_theta=1_000_000,
        rope_scaling=None,
        max_position_embeddings=32_768,
        head_dim=128,
        rms_norm_eps=1e-6,
        attention_bias=False,
        intermediate_size=3072,
    )
    layer = sglang_model.Qwen3TTSTalkerDecoderLayer(config, layer_id=0)

    expected = (object(), object(), object())
    assert layer.self_attn.apply_qk_norm_rope(*expected) == expected
    assert events == [
        ("enabled", True),
        ("wrapped", (expected, {})),
        ("inner", expected),
    ]


@pytest.mark.parametrize(
    ("model_path", "expected"),
    [
        ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", True),
        ("Qwen/Qwen3-TTS-12Hz-1.7B-Base/", True),
        ("/models/Qwen3-TTS-12Hz-0.6B-Base/snapshots/abc123", True),
        ("/models/qwen3_tts_12hz_1_7b_base/checkpoint", True),
        ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", False),
        ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", False),
        ("/models/Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/abc123", False),
        ("/models/qwen3_tts_base/Qwen3-TTS-12Hz-0.6B-CustomVoice", False),
        ("/models/qwen3_tts_base/Qwen3-TTS-12Hz-1.7B-VoiceDesign", False),
        ("model", False),
    ],
)
def test_qwen3_tts_base_path_detection_for_uploaded_voice_requirement(
    model_path: str,
    expected: bool,
) -> None:
    config = Qwen3TTSPipelineConfig(model_path=model_path)

    assert config.requires_uploaded_voice_for_named_voice() is expected
    assert config.supports_uploaded_voice_references() is expected


def test_qwen3_tts_maps_references_and_keeps_upstream_sampling_defaults() -> None:
    payload = make_payload(
        inputs={
            "text": "target",
            "references": [{"audio_path": "voice.wav", "text": "reference"}],
        },
        params={
            "temperature": 0.8,
            "top_p": 0.8,
            "top_k": 30,
            "repetition_penalty": 1.1,
        },
    )

    state = build_qwen3_tts_state(payload)

    assert state.text == "target"
    assert state.task_type == "Base"
    assert state.language == "auto"
    assert state.ref_audio == "voice.wav"
    assert state.ref_text == "reference"
    assert state.x_vector_only_mode is False
    assert state.non_streaming_mode is False
    assert state.generation_kwargs == {"max_new_tokens": 2048}


def test_qwen3_tts_preserves_explicit_default_like_sampling_values() -> None:
    payload = make_payload(
        inputs={
            "text": "target",
            "references": [{"audio_path": "voice.wav", "text": "reference"}],
        },
        params={"temperature": 0.8, "top_k": 30},
        tts_params={"explicit_generation_params": ["temperature", "top_k"]},
    )

    state = build_qwen3_tts_state(payload)

    assert state.generation_kwargs == {
        "max_new_tokens": 2048,
        "temperature": 0.8,
        "top_k": 30,
    }


def test_qwen3_tts_ignores_client_sampling_defaults() -> None:
    payload = make_payload(
        inputs="target",
        params={
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
        },
        tts_params={"ref_audio": "voice.wav", "ref_text": "reference"},
    )

    state = build_qwen3_tts_state(payload)

    assert state.generation_kwargs == {"max_new_tokens": 2048}


def test_qwen3_tts_forwards_tts_engine_stage_sampling_params() -> None:
    sampled_payload = make_payload(
        inputs="sampled",
        params={
            "subtalker_dosample": False,
            "subtalker_top_k": 1,
            "stage_params": {
                "tts_engine": {
                    "subtalker_dosample": True,
                    "subtalker_temperature": 0.7,
                    "subtalker_top_p": 0.8,
                    "subtalker_top_k": 40,
                }
            },
        },
    )
    greedy_payload = make_payload(
        inputs="greedy",
        params={"stage_params": {"tts_engine": {"subtalker_dosample": False}}},
    )

    sampled_state = build_qwen3_tts_state(sampled_payload)
    greedy_state = build_qwen3_tts_state(greedy_payload)

    assert sampled_state.generation_kwargs == {
        "max_new_tokens": 2048,
        "subtalker_dosample": True,
        "subtalker_temperature": 0.7,
        "subtalker_top_p": 0.8,
        "subtalker_top_k": 40,
    }
    assert greedy_state.generation_kwargs == {
        "max_new_tokens": 2048,
        "subtalker_dosample": False,
    }


def test_qwen3_tts_embedding_cache_keys_are_stable_and_content_based() -> None:
    """Protects radix-cache keys for Qwen requests that prefill with embeddings."""
    embeds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    same = embeds.clone()
    different_same_length = torch.tensor([[1.0, 2.0], [3.0, 5.0]])

    assert build_embedding_cache_key_ids(embeds) == build_embedding_cache_key_ids(same)
    assert build_embedding_cache_key_ids(embeds) != build_embedding_cache_key_ids(
        different_same_length
    )


def test_qwen3_tts_maps_ref_audio_form_and_explicit_sampling() -> None:
    payload = make_payload(
        inputs="target",
        params={"temperature": 0.7, "top_k": 40, "max_new_tokens": 256},
        tts_params={
            "ref_audio": "voice.wav",
            "ref_text": "reference",
            "language": "en",
        },
    )

    state = build_qwen3_tts_state(payload)

    assert state.text == "target"
    assert state.language == "en"
    assert state.ref_audio == "voice.wav"
    assert state.generation_kwargs == {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_k": 40,
    }


def test_qwen3_tts_accepts_seed_as_request_metadata() -> None:
    payload = make_payload(
        inputs="target",
        tts_params={"ref_audio": "voice.wav", "ref_text": "reference", "seed": 123},
    )

    state = build_qwen3_tts_state(payload)

    assert state.seed == 123
    assert "seed" not in state.generation_kwargs


def test_qwen3_tts_rejects_invalid_seed() -> None:
    payload = make_payload(
        inputs="target",
        tts_params={"ref_audio": "voice.wav", "ref_text": "reference", "seed": True},
    )

    with pytest.raises(ValueError, match="seed must be an integer"):
        build_qwen3_tts_state(payload)


def test_qwen3_tts_preprocessing_does_not_mutate_global_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_payload(
        inputs="target",
        tts_params={"ref_audio": "voice.wav", "ref_text": "reference"},
    )

    class FakeSpeechTokenizer:
        def encode(self, waveforms, *, sr):
            assert sr == 24000
            return SimpleNamespace(
                audio_codes=[torch.ones((1, 2), dtype=torch.long) for _ in waveforms]
            )

    class FakeWrapper:
        def _normalize_audio_inputs(self, ref_audio):
            assert ref_audio == ["voice.wav"]
            return [(np.zeros(32, dtype=np.float32), 24000)]

        def _tokenize_texts(self, texts):
            return [[idx + 1 for idx, _ in enumerate(texts[0])]]

        def _build_assistant_text(self, text):
            return text

        def _build_ref_text(self, text):
            return text

        def _merge_generate_kwargs(self, **kwargs):
            return kwargs

    class FakeModel:
        device = torch.device("cpu")
        root_config = SimpleNamespace(tts_pad_token_id=0)
        model = SimpleNamespace(_feedback_buffer=torch.empty((1, 4)))
        speech_tokenizer = FakeSpeechTokenizer()
        speaker_encoder_sample_rate = 24000

        def extract_speaker_embedding(self, *, audio, sr):
            assert audio.shape == (32,)
            assert sr == 24000
            return torch.ones(4)

        def build_voice_clone_inputs(self, **kwargs):
            del kwargs
            return (
                torch.ones((1, 2, 4)),
                torch.ones((1, 2), dtype=torch.long),
                torch.ones((1, 1, 4)),
                None,
            )

        def get_text_embeddings(self):
            return lambda ids: torch.ones((*ids.shape, 4), device=ids.device)

        def text_projection(self, embeds):
            return embeds

    def fail_manual_seed(seed):
        raise AssertionError(f"global seed mutated: {seed}")

    monkeypatch.setattr(torch, "manual_seed", fail_manual_seed)

    prepared = qwen3_request_builders._prepare_qwen3_tts_request(
        payload,
        model=FakeModel(),
        wrapper=FakeWrapper(),
    )

    assert prepared.state.seed is None


def test_qwen3_tts_uploaded_voice_clone_prompt_uses_shared_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = get_speaker_artifact_cache()
    cache.clear()
    calls = 0

    class FakePrompt:
        ref_text = "reference"

    class FakeWrapper:
        def create_voice_clone_prompt(self, **kwargs):
            nonlocal calls
            calls += 1
            return [FakePrompt()]

        def _prompt_items_to_voice_clone_prompt(self, prompt_items):
            del prompt_items
            return {
                "ref_code": [torch.ones((1, 2), dtype=torch.long)],
                "ref_spk_embedding": [torch.ones(4)],
                "icl_mode": [True],
            }

        def _tokenize_texts(self, texts):
            return [torch.arange(len(texts[0]), dtype=torch.long).unsqueeze(0)]

        def _build_assistant_text(self, text):
            return text

        def _build_ref_text(self, text):
            return text

        def _merge_generate_kwargs(self, **kwargs):
            return kwargs

    class FakeModel:
        device = torch.device("cpu")
        root_config = SimpleNamespace(tts_pad_token_id=0)
        model = SimpleNamespace(_feedback_buffer=torch.empty((1, 4)))

        def build_voice_clone_inputs(self, **kwargs):
            assert kwargs["voice_clone_prompt"]["icl_mode"] == [True]
            return (
                torch.ones((1, 2, 4)),
                torch.ones((1, 2), dtype=torch.long),
                torch.ones((1, 1, 4)),
                None,
            )

        def get_text_embeddings(self):
            return lambda ids: torch.ones((*ids.shape, 4), device=ids.device)

        def text_projection(self, embeds):
            return embeds

    monkeypatch.setattr(
        qwen3_request_builders,
        "_build_qwen3_tts_pad_embed",
        lambda model: torch.zeros(4),
    )

    def make_uploaded_payload(created_at: int) -> StagePayload:
        return make_payload(
            inputs="target",
            tts_params={
                "ref_audio": "voice.wav",
                "ref_text": "reference",
                "uploaded_voice_name": "guide",
                "uploaded_voice_created_at": created_at,
            },
        )

    qwen3_request_builders._prepare_qwen3_tts_request(
        make_uploaded_payload(7),
        model=FakeModel(),
        wrapper=FakeWrapper(),
    )
    cached = cache.get(
        SpeakerCacheKey("qwen3_tts_icl", "guide", 7, "voice_clone_prompt")
    )
    assert isinstance(cached, dict)
    assert cached["artifact_type"] == "qwen3_tts_voice_clone_prompt"
    assert cached["ref_spk_embedding"][0].device.type == "cpu"
    assert cached["ref_code"][0].device.type == "cpu"

    qwen3_request_builders._prepare_qwen3_tts_request(
        make_uploaded_payload(7),
        model=FakeModel(),
        wrapper=FakeWrapper(),
    )
    qwen3_request_builders._prepare_qwen3_tts_request(
        make_uploaded_payload(8),
        model=FakeModel(),
        wrapper=FakeWrapper(),
    )
    cache.clear_voice("guide")
    qwen3_request_builders._prepare_qwen3_tts_request(
        make_uploaded_payload(8),
        model=FakeModel(),
        wrapper=FakeWrapper(),
    )

    assert calls == 3


def test_qwen3_tts_adhoc_voice_clone_prompt_uses_reference_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = get_speaker_artifact_cache()
    cache.clear()
    qwen3_request_builders.clear_qwen3_tts_preprocessing_context()
    calls = 0
    data_uri = "data:audio/wav;base64,AAAA"

    class FakeSpeechTokenizer:
        def encode(self, waveforms, *, sr):
            nonlocal calls
            calls += 1
            assert sr == 24000
            return SimpleNamespace(
                audio_codes=[torch.ones((1, 2), dtype=torch.long) for _ in waveforms]
            )

    class FakeWrapper:
        def _normalize_audio_inputs(self, ref_audio):
            assert ref_audio == [data_uri]
            return [(np.zeros(32, dtype=np.float32), 24000)]

        def _tokenize_texts(self, texts):
            return [torch.arange(len(texts[0]), dtype=torch.long).unsqueeze(0)]

        def _build_assistant_text(self, text):
            return text

        def _build_ref_text(self, text):
            return text

        def _merge_generate_kwargs(self, **kwargs):
            return kwargs

    class FakeModel:
        device = torch.device("cpu")
        root_config = SimpleNamespace(tts_pad_token_id=0)
        model = SimpleNamespace(_feedback_buffer=torch.empty((1, 4)))
        speech_tokenizer = FakeSpeechTokenizer()
        speaker_encoder_sample_rate = 24000

        def extract_speaker_embedding(self, *, audio, sr):
            assert audio.shape == (32,)
            assert sr == 24000
            return torch.ones(4)

        def build_voice_clone_inputs(self, **kwargs):
            assert kwargs["voice_clone_prompt"]["icl_mode"] in ([True], [False])
            return (
                torch.ones((1, 2, 4)),
                torch.ones((1, 2), dtype=torch.long),
                torch.ones((1, 1, 4)),
                None,
            )

        def get_text_embeddings(self):
            return lambda ids: torch.ones((*ids.shape, 4), device=ids.device)

        def text_projection(self, embeds):
            return embeds

    monkeypatch.setattr(
        qwen3_request_builders,
        "_build_qwen3_tts_pad_embed",
        lambda model: torch.zeros(4),
    )
    model = FakeModel()
    wrapper = FakeWrapper()

    def make_adhoc_payload(**tts_params) -> StagePayload:
        params = {
            "ref_audio": data_uri,
            "ref_text": "reference",
        }
        params.update(tts_params)
        return make_payload(inputs="target", tts_params=params)

    qwen3_request_builders._prepare_qwen3_tts_request(
        make_adhoc_payload(),
        model=model,
        wrapper=wrapper,
    )
    qwen3_request_builders._prepare_qwen3_tts_request(
        make_adhoc_payload(),
        model=model,
        wrapper=wrapper,
    )
    assert calls == 1
    assert cache.stats()["entries"] == 0

    qwen3_request_builders._prepare_qwen3_tts_request(
        make_adhoc_payload(ref_text="different"),
        model=model,
        wrapper=wrapper,
    )
    qwen3_request_builders._prepare_qwen3_tts_request(
        make_adhoc_payload(x_vector_only_mode=True),
        model=model,
        wrapper=wrapper,
    )
    assert calls == 3
    qwen3_request_builders.clear_qwen3_tts_preprocessing_context()


def test_qwen3_tts_reference_codes_batch_across_requests() -> None:
    batch_sizes: list[int] = []

    class FakeSpeechTokenizer:
        def encode(self, waveforms, *, sr):
            batch_sizes.append(len(waveforms))
            assert torch.is_inference_mode_enabled()
            assert sr == 24000
            return SimpleNamespace(
                audio_codes=[torch.tensor([index]) for index in range(len(waveforms))]
            )

    batcher = qwen3_request_builders._Qwen3TTSRefCodeBatcher(
        FakeSpeechTokenizer(),
        max_batch_size=2,
        max_batch_wait_ms=50.0,
    )
    barrier = threading.Barrier(3)
    results: list[torch.Tensor | None] = [None, None]

    def encode(index: int) -> None:
        barrier.wait()
        results[index] = batcher.encode(np.zeros(16, dtype=np.float32), 24000)

    threads = [threading.Thread(target=encode, args=(index,)) for index in range(2)]
    try:
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2.0)
    finally:
        batcher.close()

    assert all(not thread.is_alive() for thread in threads)
    assert batch_sizes == [2]
    assert sorted(int(result.item()) for result in results if result is not None) == [
        0,
        1,
    ]


def test_qwen3_tts_preprocessing_executor_admits_concurrent_requests() -> None:
    executor = qwen3_stages.create_preprocessing_executor("unused-model-path")
    assert executor._max_concurrency > 1


def test_qwen3_tts_preprocess_payload_batches_reference_codes_across_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = get_speaker_artifact_cache()
    cache.clear()
    qwen3_request_builders.clear_qwen3_tts_preprocessing_context()
    batch_sizes: list[int] = []
    both_normalizing = threading.Barrier(2)

    class PatientRefCodeBatcher(qwen3_request_builders._Qwen3TTSRefCodeBatcher):
        def __init__(self, speech_tokenizer, **kwargs):
            kwargs["max_batch_wait_ms"] = 500.0
            super().__init__(speech_tokenizer, **kwargs)

    monkeypatch.setattr(
        qwen3_request_builders,
        "_Qwen3TTSRefCodeBatcher",
        PatientRefCodeBatcher,
    )

    class FakeSpeechTokenizer:
        def encode(self, waveforms, *, sr):
            batch_sizes.append(len(waveforms))
            assert sr == 24000
            return SimpleNamespace(
                audio_codes=[
                    torch.full((1, 2), index, dtype=torch.long)
                    for index in range(len(waveforms))
                ]
            )

    class FakeWrapper:
        def _normalize_audio_inputs(self, ref_audio):
            both_normalizing.wait(timeout=5.0)
            return [(np.zeros(32, dtype=np.float32), 24000)]

        def _tokenize_texts(self, texts):
            return [torch.arange(len(texts[0]), dtype=torch.long).unsqueeze(0)]

        def _build_assistant_text(self, text):
            return text

        def _build_ref_text(self, text):
            return text

        def _merge_generate_kwargs(self, **kwargs):
            return kwargs

    class FakeModel:
        device = torch.device("cpu")
        root_config = SimpleNamespace(tts_pad_token_id=0)
        model = SimpleNamespace(_feedback_buffer=torch.empty((1, 4)))
        speech_tokenizer = FakeSpeechTokenizer()
        speaker_encoder_sample_rate = 24000

        def extract_speaker_embedding(self, *, audio, sr):
            return torch.ones(4)

        def build_voice_clone_inputs(self, **kwargs):
            del kwargs
            return (
                torch.ones((1, 2, 4)),
                torch.ones((1, 2), dtype=torch.long),
                torch.ones((1, 1, 4)),
                None,
            )

        def get_text_embeddings(self):
            return lambda ids: torch.ones((*ids.shape, 4), device=ids.device)

        def text_projection(self, embeds):
            return embeds

    monkeypatch.setattr(
        qwen3_request_builders,
        "_build_qwen3_tts_pad_embed",
        lambda model: torch.zeros(4),
    )
    qwen3_request_builders.set_qwen3_tts_preprocessing_context(
        model=FakeModel(),
        wrapper=FakeWrapper(),
    )

    def make_distinct_payload(index: int) -> StagePayload:
        return StagePayload(
            request_id=f"req-qwen3-tts-batch-{index}",
            request=OmniRequest(
                inputs=f"target-{index}",
                params={},
                metadata={
                    "tts_params": {
                        "ref_audio": f"data:audio/wav;base64,AAA{index}",
                        "ref_text": f"reference-{index}",
                    }
                },
            ),
            data={},
        )

    errors: list[Exception] = []

    def preprocess(index: int) -> None:
        try:
            qwen3_request_builders.preprocess_qwen3_tts_payload(
                make_distinct_payload(index)
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=preprocess, args=(index,)) for index in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
        assert not errors
        assert all(not thread.is_alive() for thread in threads)
        with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
            prepared_count = len(qwen3_request_builders._PREPARED_REQUESTS)
    finally:
        qwen3_request_builders.clear_qwen3_tts_preprocessing_context()

    assert prepared_count == 2
    assert batch_sizes == [2]


def test_qwen3_tts_reference_code_overlaps_speaker_embedding() -> None:
    tokenizer_started = threading.Event()
    speaker_started = threading.Event()

    class FakeSpeechTokenizer:
        def encode(self, waveforms, *, sr):
            tokenizer_started.set()
            assert speaker_started.wait(timeout=1.0)
            return SimpleNamespace(audio_codes=[torch.ones((1, 2), dtype=torch.long)])

    class FakeWrapper:
        def _normalize_audio_inputs(self, ref_audio):
            return [(np.zeros(32, dtype=np.float32), 24000)]

    class FakeModel:
        device = torch.device("cpu")
        speech_tokenizer = FakeSpeechTokenizer()
        speaker_encoder_sample_rate = 24000

        def extract_speaker_embedding(self, *, audio, sr):
            assert tokenizer_started.wait(timeout=1.0)
            speaker_started.set()
            return torch.ones(4)

    hook = qwen3_request_builders._Qwen3TTSAdhocReferenceHook(
        model=FakeModel(),
        wrapper=FakeWrapper(),
    )
    item = qwen3_request_builders._Qwen3TTSAdhocReferenceInput(
        ref_audio="data:audio/wav;base64,AAAA",
        ref_text="reference",
        x_vector_only_mode=False,
    )
    try:
        prompt, ref_text = hook.encode_one(item)
    finally:
        hook.close()

    assert ref_text == "reference"
    assert prompt["icl_mode"] == [True]


def test_qwen3_tts_reference_code_batcher_synchronizes_cuda_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    code = SimpleNamespace(is_cuda=True, device=torch.device("cuda"))

    class FakeCurrentStream:
        def synchronize(self):
            events.append("synchronize")

    class FakeSpeechTokenizer:
        def encode(self, waveforms, *, sr):
            assert len(waveforms) == 1
            assert sr == 24000
            return SimpleNamespace(audio_codes=[code])

    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: FakeCurrentStream(),
    )
    batcher = qwen3_request_builders._Qwen3TTSRefCodeBatcher(
        FakeSpeechTokenizer(),
        max_batch_wait_ms=0,
    )
    try:
        result = batcher.encode(np.zeros(16, dtype=np.float32), 24000)
    finally:
        batcher.close()

    assert result is code
    assert events == ["synchronize"]


def test_qwen3_tts_reference_code_batcher_has_no_stream_for_cpu_device() -> None:
    class FakeSpeechTokenizer:
        def encode(self, waveforms, *, sr):
            raise AssertionError("encode must not run in this test")

    batcher = qwen3_request_builders._Qwen3TTSRefCodeBatcher(
        FakeSpeechTokenizer(),
        device=torch.device("cpu"),
    )
    try:
        assert batcher._encode_stream is None
    finally:
        batcher.close()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen3_tts_reference_code_batcher_encodes_on_dedicated_cuda_stream() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    encode_streams: list[object] = []

    class FakeSpeechTokenizer:
        def encode(self, waveforms, *, sr):
            assert sr == 24000
            encode_streams.append(torch.cuda.current_stream(device))
            return SimpleNamespace(
                audio_codes=[
                    torch.full((1, 2), index, dtype=torch.long, device=device)
                    for index in range(len(waveforms))
                ]
            )

    batcher = qwen3_request_builders._Qwen3TTSRefCodeBatcher(
        FakeSpeechTokenizer(),
        device=device,
    )
    try:
        assert batcher._encode_stream is not None
        assert batcher._encode_stream != torch.cuda.default_stream(device)
        result = batcher.encode(np.zeros(16, dtype=np.float32), 24000)
    finally:
        batcher.close()

    assert encode_streams == [batcher._encode_stream]
    assert result.is_cuda
    assert torch.equal(result.cpu(), torch.zeros((1, 2), dtype=torch.long))


def test_qwen3_tts_uploaded_voice_x_vector_cache_omits_ref_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = get_speaker_artifact_cache()
    cache.clear()
    calls = 0

    class FakePrompt:
        ref_text = None

    class FakeWrapper:
        def create_voice_clone_prompt(self, **kwargs):
            nonlocal calls
            calls += 1
            assert kwargs["x_vector_only_mode"] is True
            return [FakePrompt()]

        def _prompt_items_to_voice_clone_prompt(self, prompt_items):
            del prompt_items
            return {
                "ref_code": [None],
                "ref_spk_embedding": [torch.ones(4)],
                "icl_mode": [False],
            }

        def _tokenize_texts(self, texts):
            return [torch.arange(len(texts[0]), dtype=torch.long).unsqueeze(0)]

        def _build_assistant_text(self, text):
            return text

        def _merge_generate_kwargs(self, **kwargs):
            return kwargs

    class FakeModel:
        device = torch.device("cpu")
        root_config = SimpleNamespace(tts_pad_token_id=0)
        model = SimpleNamespace(_feedback_buffer=torch.empty((1, 4)))

        def build_voice_clone_inputs(self, **kwargs):
            assert kwargs["voice_clone_prompt"]["icl_mode"] == [False]
            assert kwargs["voice_clone_prompt"].get("ref_code") in (None, [None])
            return (
                torch.ones((1, 2, 4)),
                torch.ones((1, 2), dtype=torch.long),
                torch.ones((1, 1, 4)),
                None,
            )

        def get_text_embeddings(self):
            return lambda ids: torch.ones((*ids.shape, 4), device=ids.device)

        def text_projection(self, embeds):
            return embeds

    monkeypatch.setattr(
        qwen3_request_builders,
        "_build_qwen3_tts_pad_embed",
        lambda model: torch.zeros(4),
    )

    payload = make_payload(
        inputs="target",
        tts_params={
            "ref_audio": "voice.wav",
            "uploaded_voice_name": "guide",
            "uploaded_voice_created_at": 9,
            "x_vector_only_mode": True,
        },
    )

    qwen3_request_builders._prepare_qwen3_tts_request(
        payload,
        model=FakeModel(),
        wrapper=FakeWrapper(),
    )
    cached = cache.get(
        SpeakerCacheKey("qwen3_tts_xvec", "guide", 9, "voice_clone_prompt")
    )
    assert isinstance(cached, dict)
    assert "ref_code" not in cached

    qwen3_request_builders._prepare_qwen3_tts_request(
        payload,
        model=FakeModel(),
        wrapper=FakeWrapper(),
    )

    assert calls == 1


def test_qwen3_tts_public_seed_derivation_is_stable() -> None:
    first = derive_qwen3_tts_sampling_seeds(123)
    second = derive_qwen3_tts_sampling_seeds(123)
    different = derive_qwen3_tts_sampling_seeds(124)

    assert first == second
    assert first != different
    assert first[0] != first[1]
    assert all(0 <= seed <= 0x7FFFFFFF for seed in first)
    assert derive_qwen3_tts_sampling_seeds(123456) == (709979716, 2088621061)


def test_qwen3_tts_text_only_defaults_to_custom_voice() -> None:
    payload = make_payload(inputs="target", tts_params={"voice": "default"})

    state = build_qwen3_tts_state(payload)

    assert state.task_type == "CustomVoice"
    assert state.task_type_explicit is False
    assert state.voice == "Vivian"
    assert state.ref_audio is None
    assert state.ref_text is None
    assert state.non_streaming_mode is True


def test_qwen3_tts_custom_voice_rejects_base_only_fields() -> None:
    payload = make_payload(
        inputs="target",
        tts_params={"task_type": "CustomVoice", "ref_text": "reference"},
    )

    with pytest.raises(ValueError, match="CustomVoice does not accept ref_text"):
        build_qwen3_tts_state(payload)


@pytest.mark.parametrize(
    ("task_type", "extra_tts_params", "match"),
    [
        ("CustomVoice", {}, "CustomVoice does not accept ref_audio"),
        (
            "VoiceDesign",
            {"instructions": "A warm adult voice."},
            "VoiceDesign does not accept ref_audio",
        ),
    ],
)
@pytest.mark.parametrize(
    ("inputs", "tts_params"),
    [
        ("target", {"ref_audio": "voice.wav"}),
        ({"text": "target", "references": [{"audio_path": "voice.wav"}]}, {}),
        ({"text": "target", "references": [{"ref_audio": "voice.wav"}]}, {}),
        ({"text": "target", "references": [{"audio": "voice.wav"}]}, {}),
    ],
)
def test_qwen3_tts_non_base_tasks_reject_audio_references(
    task_type: str,
    extra_tts_params: dict[str, str],
    match: str,
    inputs: object,
    tts_params: dict[str, str],
) -> None:
    payload = make_payload(
        inputs=inputs,
        tts_params={
            "task_type": task_type,
            **extra_tts_params,
            **tts_params,
        },
    )

    with pytest.raises(ValueError, match=match):
        build_qwen3_tts_state(payload)


def test_qwen3_tts_voice_design_requires_instructions() -> None:
    payload = make_payload(
        inputs="target",
        tts_params={"task_type": "VoiceDesign"},
    )

    with pytest.raises(ValueError, match="VoiceDesign requires instructions"):
        build_qwen3_tts_state(payload)


def test_qwen3_tts_voice_design_state_forces_non_streaming() -> None:
    payload = make_payload(
        inputs="target",
        tts_params={
            "task_type": "VoiceDesign",
            "instructions": "A warm adult voice.",
        },
    )

    state = build_qwen3_tts_state(payload)

    assert state.task_type == "VoiceDesign"
    assert state.instructions == "A warm adult voice."
    assert state.voice is None
    assert state.non_streaming_mode is True


def test_qwen3_tts_uses_x_vector_only_when_ref_text_is_missing() -> None:
    payload = make_payload(
        inputs={"text": "target", "references": [{"audio_path": "voice.wav"}]},
    )

    state = build_qwen3_tts_state(payload)

    assert state.ref_audio == "voice.wav"
    assert state.ref_text is None
    assert state.x_vector_only_mode is True


def test_qwen3_tts_rejects_missing_reference_audio() -> None:
    payload = make_payload(inputs="target", tts_params={"task_type": "Base"})

    with pytest.raises(ValueError, match="requires reference audio"):
        build_qwen3_tts_state(payload)


def test_qwen3_tts_predictor_codec_embeddings_use_talker_hidden_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protects 1.7B loading where talker and predictor hidden sizes differ."""
    install_fake_sglang(monkeypatch)
    from torch import nn

    from sglang_omni.models.qwen3_tts import sglang_model

    class FakeDecoderLayer(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeReplicatedLinear(nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            bias: bool = False,
            **kwargs,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        def forward(self, x):
            return self.linear(x), None

    monkeypatch.setattr(sglang_model, "Qwen3TTSTalkerDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(sglang_model, "ReplicatedLinear", FakeReplicatedLinear)
    monkeypatch.setattr(
        sglang_model,
        "RMSNorm",
        lambda hidden_size, eps=1e-6: nn.LayerNorm(hidden_size, eps=eps),
    )

    predictor_config = SimpleNamespace(
        vocab_size=2048,
        hidden_size=1024,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
    )
    talker_config = SimpleNamespace(
        hidden_size=2048,
        num_code_groups=16,
        code_predictor_config=predictor_config,
    )

    predictor = sglang_model.Qwen3TTSCodePredictor(talker_config)

    assert predictor.model.codec_embedding[0].weight.shape == (2048, 2048)
    assert predictor.small_to_mtp_projection.weight.shape == (1024, 2048)


def test_qwen3_tts_custom_voice_requires_speaker_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker.config = SimpleNamespace(spk_id={})

    with pytest.raises(ValueError, match="configured spk_id"):
        Qwen3TTSTalker.build_custom_voice_inputs(
            talker,
            input_id=torch.arange(8, dtype=torch.long).unsqueeze(0),
            voice="Vivian",
            language="auto",
            non_streaming_mode=True,
        )


def test_qwen3_tts_custom_voice_rejects_invalid_speaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker.config = SimpleNamespace(spk_id={"Vivian": 3065})

    with pytest.raises(ValueError, match="Unsupported Qwen3-TTS CustomVoice speaker"):
        Qwen3TTSTalker.build_custom_voice_inputs(
            talker,
            input_id=torch.arange(8, dtype=torch.long).unsqueeze(0),
            voice="Missing",
            language="auto",
            non_streaming_mode=True,
        )


def test_qwen3_tts_vocoder_batches_decode_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protects Qwen3-TTS vocoder throughput from regressing to serial decode."""
    from sglang_omni.models.qwen3_tts import stages

    decode_batch_sizes: list[int] = []

    class FakeTokenizer:
        model = SimpleNamespace(
            decoder=SimpleNamespace(total_upsample=4),
        )

        def get_output_sample_rate(self):
            return 24000

        def decode(self, encoded):
            decode_batch_sizes.append(len(encoded))
            return [
                torch.arange(6, dtype=torch.float32),
                torch.arange(8, dtype=torch.float32),
            ], 24000

    monkeypatch.setattr(
        stages,
        "_load_qwen3_tts_tokenizer",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    warmed_schedulers: list[Qwen3TTSStreamingVocoderScheduler] = []
    monkeypatch.setattr(
        Qwen3TTSStreamingVocoderScheduler,
        "warmup_now",
        lambda scheduler: warmed_schedulers.append(scheduler),
    )

    scheduler = stages.create_vocoder_executor(
        "model",
        device="cpu",
        max_batch_size=2,
        max_batch_wait_ms=3,
    )
    assert warmed_schedulers == [scheduler]
    assert scheduler.create_stream_state("request").initial_chunk_frames == 8
    assert scheduler._stream_left_context_frames == 16
    assert scheduler._stream_followup_stride == 8
    assert scheduler._followup_stride_ramp == (8,)
    assert scheduler._initial_max_batch_size == 32
    assert scheduler._initial_batch_wait_s == pytest.approx(0.002)
    assert scheduler._followup_max_batch_size == 8
    assert scheduler._followup_batch_wait_s == pytest.approx(0.001)
    first = make_payload(inputs="first")
    first.data = Qwen3TTSState(
        audio_codes=torch.tensor([[1, 2], [3, 4]]),
        ref_code_len=1,
    ).to_dict()
    second = make_payload(inputs="second")
    second.data = Qwen3TTSState(
        audio_codes=torch.tensor([[5, 6], [7, 8]]),
    ).to_dict()

    results = asyncio.run(scheduler._batch_fn([first, second]))

    assert scheduler._max_batch_size == 2
    assert scheduler._max_batch_wait_s == pytest.approx(0.003)
    assert decode_batch_sizes == [2]
    assert results[0].data["sample_rate"] == 24000
    first_audio = np.frombuffer(results[0].data["audio_waveform"], dtype=np.float32)
    assert first_audio.tolist() == [3.0, 4.0, 5.0]
    assert results[0].data["audio_waveform_shape"] == [3]
    assert results[0].data["audio_waveform_dtype"] == "float32"
    assert "audio_codes" not in results[0].data
    second_audio = np.frombuffer(results[1].data["audio_waveform"], dtype=np.float32)
    assert second_audio.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


class _FakeQwen3TTSDecoder:
    total_upsample = 4

    def __init__(self) -> None:
        self.decode_inputs: list[torch.Tensor] = []

    def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(codes.detach().clone())
        return (
            codes[:, :1]
            .to(torch.float32)
            .repeat_interleave(self.total_upsample, dim=-1)
        )

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        return self.chunked_decode(codes)


class _FakeQwen3TTSTokenizer:
    def __init__(self) -> None:
        self.model = SimpleNamespace(decoder=_FakeQwen3TTSDecoder())

    def get_output_sample_rate(self) -> int:
        return 24000

    def decode(self, encoded):
        waveforms = [
            item["audio_codes"][:, 0]
            .to(torch.float32)
            .repeat_interleave(self.model.decoder.total_upsample)
            .numpy()
            for item in encoded
        ]
        return waveforms, self.get_output_sample_rate()


class _FakeIncrementalQwen3TTSDecoder:
    def __init__(self, decoder, *, fail_on_call: int | None = None) -> None:
        self._decoder = decoder
        self._fail_on_call = fail_on_call
        self.decode_inputs: list[torch.Tensor] = []

    def decode(
        self,
        codes: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
    ) -> torch.Tensor:
        self.decode_inputs.append(codes.detach().clone())
        if len(self.decode_inputs) == self._fail_on_call:
            raise RuntimeError("injected incremental decode failure")
        state.frame_position += int(codes.shape[-1])
        return (
            codes[:, :1]
            .to(torch.float32)
            .repeat_interleave(self._decoder.total_upsample, dim=-1)
        )


def _stateful_qwen3_tts_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_on_call: int | None = None,
    stream_left_context_frames: int = 1,
) -> tuple[Qwen3TTSStreamingVocoderScheduler, _FakeIncrementalQwen3TTSDecoder]:
    created = []

    def make_incremental(decoder):
        incremental = _FakeIncrementalQwen3TTSDecoder(
            decoder, fail_on_call=fail_on_call
        )
        created.append(incremental)
        return incremental

    monkeypatch.setattr(
        qwen3_streaming_vocoder,
        "Qwen3TTSIncrementalDecoder",
        make_incremental,
    )
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        async_decode=True,
        initial_cuda_graph=True,
        stream_left_context_frames=stream_left_context_frames,
        enable_stateful_codec_decoder=True,
    )
    return scheduler, created[0]


def test_qwen3_tts_stateful_codec_uses_reference_once_then_fresh_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, incremental = _stateful_qwen3_tts_scheduler(monkeypatch)
    state = scheduler.create_stream_state("request")
    state.ref_frames = 2
    state.code_chunks.append(
        torch.tensor([[10, 1], [20, 2], [30, 3]], dtype=torch.long)
    )
    state.total_frames = 3

    first = scheduler.decode_delta("request", state, is_final=False)
    state.code_chunks.append(torch.tensor([[40, 4], [50, 5]], dtype=torch.long))
    state.total_frames = 5
    second = scheduler.decode_delta("request", state, is_final=False)

    assert scheduler._async_decode is False
    assert scheduler._initial_decode_graphs._enabled is False
    assert first is not None
    assert first.tolist() == [30.0] * 4
    assert second is not None
    assert second.tolist() == [40.0] * 4 + [50.0] * 4
    assert [tuple(item.shape) for item in incremental.decode_inputs] == [
        (1, 2, 3),
        (1, 2, 2),
    ]
    assert state.incremental_codec_state is not None
    assert state.incremental_codec_state.frame_position == 5
    assert state.emitted_generated_frames == 3
    assert state.pruned_frames == 3


def test_qwen3_tts_stateful_codec_failure_falls_back_without_committing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, incremental = _stateful_qwen3_tts_scheduler(monkeypatch, fail_on_call=2)
    state = scheduler.create_stream_state("request")
    state.code_chunks.append(torch.tensor([[10, 1]], dtype=torch.long))
    state.total_frames = 1
    first = scheduler.decode_delta("request", state, is_final=False)
    state.code_chunks.append(torch.tensor([[20, 2]], dtype=torch.long))
    state.total_frames = 2

    second = scheduler.decode_delta("request", state, is_final=False)

    assert first is not None
    assert first.tolist() == [10.0] * 4
    assert second is not None
    assert second.tolist() == [20.0] * 4
    assert state.incremental_codec_fallback is True
    assert state.incremental_codec_state is not None
    assert state.incremental_codec_state.frame_position == 1
    assert state.emitted_generated_frames == 2
    assert len(incremental.decode_inputs) == 2
    assert len(scheduler._decoder.decode_inputs) == 1

    state.code_chunks.append(torch.tensor([[30, 3]], dtype=torch.long))
    state.total_frames = 3
    third = scheduler.decode_delta("request", state, is_final=True)

    assert third is not None
    assert third.tolist() == [30.0] * 4
    assert len(incremental.decode_inputs) == 2


def test_qwen3_tts_stateful_codec_terminal_without_fresh_frames_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, incremental = _stateful_qwen3_tts_scheduler(monkeypatch)
    state = scheduler.create_stream_state("request")
    state.code_chunks.append(torch.tensor([[10, 1]], dtype=torch.long))
    state.total_frames = 1

    delta = scheduler.decode_delta("request", state, is_final=True)
    terminal = scheduler.decode_delta("request", state, is_final=True)

    assert delta is not None
    assert delta.numel() == scheduler._samples_per_frame
    assert terminal is None
    assert len(incremental.decode_inputs) == 1


class _FakeDecodeStream:
    """Stand-in for the decode CUDA stream; logs waits and syncs."""

    def __init__(
        self, events: list[str], *, sync_error: BaseException | None = None
    ) -> None:
        self._events = events
        self.sync_error = sync_error

    def wait_stream(self, stream) -> None:
        self._events.append("wait")

    def synchronize(self) -> None:
        self._events.append("stream_synchronize")
        if self.sync_error is not None:
            raise self.sync_error


class _FakeCudaEvent:
    """Stand-in for torch.cuda.Event with injectable record/sync failures."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.record_error: BaseException | None = None
        self.sync_error: BaseException | None = None

    def record(self, stream) -> None:
        self._events.append("record")
        if self.record_error is not None:
            raise self.record_error

    def synchronize(self) -> None:
        self._events.append("event_synchronize")
        if self.sync_error is not None:
            raise self.sync_error


class _FakeCudaGraph:
    """Stand-in for torch.cuda.CUDAGraph that counts replays."""

    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


def _fake_allocate_pinned(numel: int, dtype: torch.dtype) -> torch.Tensor:
    # Mirror the real allocator: an ordinary (non-inference) tensor even when
    # the slot grows under torch.inference_mode(), just not pinned.
    with torch.inference_mode(False):
        return torch.empty(numel, dtype=dtype)


def _force_pinned_cpu_decode(
    scheduler: Qwen3TTSStreamingVocoderScheduler,
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> list[_FakeCudaEvent]:
    """Route a CPU scheduler through the pinned async path with CUDA stand-ins.

    Returns the list of events created through ``torch.cuda.Event`` so tests
    can assert event reuse.
    """
    created: list[_FakeCudaEvent] = []

    class StreamContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    def make_event():
        event = _FakeCudaEvent(events)
        created.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: StreamContext())
    monkeypatch.setattr(torch.cuda, "Event", make_event)
    monkeypatch.setattr(cuda_staging, "_allocate_pinned", _fake_allocate_pinned)
    scheduler._pinned_staging_disabled = False
    return created


def _qwen3_tts_single_frame_plan(code: int) -> _Qwen3TTSDecodePlan:
    return _Qwen3TTSDecodePlan(
        decoder_input=torch.tensor([[[code]]], dtype=torch.long),
        absolute_emitted_frames=0,
        generated_frames=1,
        window_start=0,
        emitted_generated_frames=0,
    )


def _qwen3_tts_two_frame_plan(
    scheduler: Qwen3TTSStreamingVocoderScheduler,
) -> _Qwen3TTSDecodePlan:
    state = scheduler.create_stream_state("request")
    state.code_chunks.append(torch.ones((2, 2), dtype=torch.long))
    state.total_frames = 2
    plan = scheduler._build_decode_plan(state, is_final=True)
    assert plan is not None
    return plan


def test_qwen3_tts_initial_decode_graphs_noop_on_cpu() -> None:
    decoder = _FakeQwen3TTSDecoder()
    graphs = _Qwen3TTSInitialDecodeGraphs(
        decoder,
        device=torch.device("cpu"),
        num_quantizers=2,
        input_frames=17,
    )

    graphs.capture()

    assert graphs.decode(torch.zeros((1, 2, 17), dtype=torch.long)) is None
    assert decoder.decode_inputs == []


def test_qwen3_tts_decode_graphs_key_by_frames_and_batch_bucket() -> None:
    decoder = _FakeQwen3TTSDecoder()
    graphs = _Qwen3TTSInitialDecodeGraphs(
        decoder,
        device=torch.device("cpu"),
        num_quantizers=2,
        input_frames=(24, 32, 24),
        batch_sizes=(4, 1),
    )

    assert graphs._input_frames == (24, 32)
    assert graphs._batch_sizes == (1, 4)
    graphs.capture()
    assert graphs.decode(torch.zeros((1, 2, 24), dtype=torch.long)) is None


def test_qwen3_tts_decode_graphs_replay_pads_batch_and_slices_output() -> None:
    decoder = _FakeQwen3TTSDecoder()
    graphs = _Qwen3TTSInitialDecodeGraphs(
        decoder,
        device=torch.device("cpu"),
        num_quantizers=2,
        input_frames=(24, 32),
        batch_sizes=(1, 4),
    )
    for frames, batch in ((24, 1), (24, 4), (32, 1)):
        graphs._graphs[(frames, batch)] = _FakeCudaGraph()
        graphs._inputs[(frames, batch)] = torch.full(
            (batch, 2, frames), -1, dtype=torch.long
        )
        graphs._outputs[(frames, batch)] = torch.arange(
            batch * frames * 4, dtype=torch.float32
        ).view(batch, 1, frames * 4)

    codes = torch.arange(3 * 2 * 24, dtype=torch.long).view(3, 2, 24) + 1
    waveform = graphs.decode(codes)

    assert graphs._graphs[(24, 4)].replays == 1
    assert graphs._graphs[(24, 1)].replays == 0
    assert graphs._graphs[(32, 1)].replays == 0
    static_input = graphs._inputs[(24, 4)]
    assert torch.equal(static_input[:3], codes)
    assert torch.equal(static_input[3], torch.zeros((2, 24), dtype=torch.long))
    assert waveform is not None
    assert waveform.shape == (3, 1, 96)
    assert torch.equal(waveform, graphs._outputs[(24, 4)][:3])
    assert (
        waveform.untyped_storage().data_ptr()
        != graphs._outputs[(24, 4)].untyped_storage().data_ptr()
    )

    waveform = graphs.decode(torch.ones((1, 2, 32), dtype=torch.long))
    assert waveform is not None
    assert waveform.shape == (1, 1, 128)
    assert graphs._graphs[(32, 1)].replays == 1
    assert graphs._graphs[(24, 4)].replays == 1

    assert graphs.decode(torch.zeros((3, 2, 32), dtype=torch.long)) is None
    assert graphs.decode(torch.zeros((5, 2, 24), dtype=torch.long)) is None
    assert graphs.decode(torch.zeros((1, 2, 20), dtype=torch.long)) is None
    assert graphs.decode(torch.zeros((1, 3, 24), dtype=torch.long)) is None
    assert graphs.decode(torch.zeros((2, 24), dtype=torch.long)) is None
    assert graphs._graphs[(24, 4)].replays == 1
    assert graphs._graphs[(32, 1)].replays == 1
    assert decoder.decode_inputs == []


def test_qwen3_tts_streaming_vocoder_followup_graphs_can_be_disabled() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        followup_cuda_graph=False,
    )

    assert scheduler._followup_decode_graphs._enabled is False
    assert scheduler._initial_decode_graphs is not scheduler._followup_decode_graphs


def test_qwen3_tts_vocoder_warms_graphs_before_serving_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        async_decode=True,
    )
    captures: list[str] = []
    monkeypatch.setattr(
        scheduler._initial_decode_graphs,
        "capture",
        lambda: captures.append("initial"),
    )
    monkeypatch.setattr(
        scheduler._followup_decode_graphs,
        "capture",
        lambda: captures.append("followup"),
    )

    scheduler.warmup_now()
    scheduler.on_serving_start()
    try:
        assert captures == ["initial", "followup"]
    finally:
        scheduler.stop()


def test_qwen3_tts_deterministic_streaming_vocoder_decodes_each_plan_at_b1() -> None:
    """Match each streaming row to its independent batch-one decode."""
    tokenizer = _FakeQwen3TTSTokenizer()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cpu",
        enable_deterministic_inference=True,
    )
    plans = [
        _Qwen3TTSDecodePlan(
            decoder_input=torch.full((1, 2, 3), value, dtype=torch.long),
            absolute_emitted_frames=0,
            generated_frames=3,
            window_start=0,
            emitted_generated_frames=0,
        )
        for value in (1, 2, 3)
    ]

    deltas = scheduler._launch_decode_plans(plans, stream=None).resolve()

    assert [tuple(item.shape) for item in tokenizer.model.decoder.decode_inputs] == [
        (1, 2, 3),
        (1, 2, 3),
        (1, 2, 3),
    ]
    assert [item.tolist() for item in deltas] == [
        [float(value)] * 12 for value in (1, 2, 3)
    ]
    assert scheduler._initial_decode_graphs._batch_sizes == (1,)


def test_qwen3_tts_deterministic_vocoder_decodes_each_payload_at_b1() -> None:
    """Match each non-streaming row to its independent batch-one decode."""
    decode_batch_sizes = []

    class Tokenizer(_FakeQwen3TTSTokenizer):
        def decode(self, encoded):
            decode_batch_sizes.append(len(encoded))
            return super().decode(encoded)

    scheduler = Qwen3TTSStreamingVocoderScheduler(
        Tokenizer(),
        device="cpu",
        enable_deterministic_inference=True,
    )
    payloads = []
    for index in range(3):
        payload = make_payload(inputs=str(index))
        payload.data = Qwen3TTSState(
            audio_codes=torch.tensor([[index + 1, index + 2]]),
        ).to_dict()
        payloads.append(payload)

    results = asyncio.run(scheduler._vocode_payloads(payloads))

    assert decode_batch_sizes == [1, 1, 1]
    assert [
        np.frombuffer(result.data["audio_waveform"], dtype=np.float32).tolist()
        for result in results
    ] == [
        [1.0] * 4,
        [2.0] * 4,
        [3.0] * 4,
    ]


def test_qwen3_tts_streaming_vocoder_default_initial_chunk_is_continuity_safe() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )

    assert scheduler.create_stream_state("request").initial_chunk_frames == 8
    assert scheduler._initial_decode_graphs._input_frames == (
        scheduler._stream_left_context_frames + 8,
    )
    assert scheduler._followup_decode_graphs._input_frames == (
        scheduler._stream_left_context_frames + 8,
    )


def _qwen3_tts_stream_item(
    codes: torch.Tensor,
    *,
    chunk_id: int,
    ref_code_len: int | None = None,
) -> StreamItem:
    metadata = {
        "modality": "audio_codes",
        "stream": True,
        "num_quantizers": int(codes.shape[-1]),
    }
    if ref_code_len is not None:
        metadata["ref_code_len"] = ref_code_len
    return StreamItem(
        chunk_id=chunk_id,
        data=codes,
        from_stage="tts_engine",
        metadata=metadata,
    )


def test_qwen3_tts_initial_chunk_override_is_message_order_independent() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        stream_stride=16,
    )
    payload = make_payload(
        inputs="target",
        params={"stream": True, "initial_codec_chunk_frames": 32},
    )
    payload.request_id = "payload-first"
    scheduler._on_streaming_new_request(payload.request_id, payload)

    chunk = _qwen3_tts_stream_item(
        torch.ones((1, 2), dtype=torch.long),
        chunk_id=0,
        ref_code_len=0,
    )
    assert chunk.metadata is not None
    chunk.metadata["initial_codec_chunk_frames"] = 32
    scheduler._on_chunk("chunk-first", chunk)

    assert scheduler._stream_states["payload-first"].initial_chunk_frames == 16
    assert scheduler._stream_states["chunk-first"].initial_chunk_frames == 16


def test_qwen3_tts_streaming_vocoder_keeps_codec_chunks_on_source_device() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    state = scheduler.create_stream_state("request")
    state.num_quantizers = 2
    transfers: list[dict[str, object]] = []

    class DeviceTrackingCodes:
        def detach(self):
            return self

        def to(self, **kwargs):
            transfers.append(kwargs)
            return torch.ones((1, 2), dtype=torch.long)

    scheduler.validate_chunk("request", state, DeviceTrackingCodes())

    assert transfers == [{"dtype": torch.long}]


def test_qwen3_tts_streaming_vocoder_avoids_cuda_value_sync() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    state = scheduler.create_stream_state("request")
    state.num_quantizers = 2

    class CudaChunk:
        ndim = 2
        shape = (1, 2)
        is_cuda = True

        def __lt__(self, other):
            raise AssertionError("CUDA codec validation must not reduce on the host")

        def __ge__(self, other):
            raise AssertionError("CUDA codec validation must not reduce on the host")

    chunk = CudaChunk()

    class Codes:
        def detach(self):
            return self

        def to(self, **kwargs):
            return chunk

    assert scheduler.validate_chunk("request", state, Codes()) is chunk


def test_qwen3_tts_decode_stream_waits_for_input_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    state = scheduler.create_stream_state("request")
    state.code_chunks.append(torch.ones((1, 2), dtype=torch.long))
    state.total_frames = 1
    plan = scheduler._build_decode_plan(state, is_final=True)
    assert plan is not None

    producer_stream = object()
    events: list[object] = []

    class DecodeStream:
        def wait_stream(self, stream):
            events.append(("wait", stream))

        def synchronize(self):
            events.append("synchronize")

    decode_stream = DecodeStream()

    class StreamContext:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: producer_stream,
    )
    monkeypatch.setattr(
        torch.cuda,
        "stream",
        lambda stream: StreamContext(),
    )

    handle = scheduler._launch_decode_plans([plan], stream=decode_stream)
    handle.resolve()

    assert events[0] == ("wait", producer_stream)


def test_qwen3_tts_pageable_fallback_syncs_with_empty_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty .cpu() copies enqueue no D2H work, so fallback synchronizes explicitly."""

    class ShortDecoder(_FakeQwen3TTSDecoder):
        def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
            return torch.zeros((codes.shape[0], 1, 8), dtype=torch.float32)

    tokenizer = _FakeQwen3TTSTokenizer()
    tokenizer.model.decoder = ShortDecoder()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cpu",
    )
    state = scheduler.create_stream_state("request")
    state.num_quantizers = 2
    state.code_chunks.append(torch.ones((5, 2), dtype=torch.long))
    state.total_frames = 5
    state.emitted_generated_frames = 4
    state.next_decode_generated_frames = 5
    plan = scheduler._build_decode_plan(state, is_final=True)
    assert plan is not None

    events: list[str] = []

    class DecodeStream:
        def wait_stream(self, stream):
            events.append("wait")

        def synchronize(self):
            events.append("stream_synchronize")

    class StreamContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: StreamContext())

    handle = scheduler._launch_decode_plans([plan], stream=DecodeStream())

    assert (
        "stream_synchronize" in events
    ), "all-empty batch must wait for the decode stream"
    assert handle.slot is None
    delta = handle.resolve()[0]
    assert delta.numel() == 0
    with pytest.raises(RuntimeError, match="empty delta"):
        scheduler._commit_decode_plan(state, plan, delta)


def test_qwen3_tts_decode_launch_defers_resolve_to_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launch records the slot event; resolve() waits on it; the event is reused."""
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    state = scheduler.create_stream_state("request")
    state.code_chunks.append(torch.ones((2, 2), dtype=torch.long))
    state.total_frames = 2
    plan = scheduler._build_decode_plan(state, is_final=True)
    assert plan is not None

    events: list[str] = []
    created = _force_pinned_cpu_decode(scheduler, monkeypatch, events)
    stream = _FakeDecodeStream(events)
    slot = scheduler._thread_decode_slot()

    handle = scheduler._launch_decode_plans([plan], stream=stream)

    assert events == ["wait", "record"], events
    assert handle.slot is slot and slot.busy, "a pending handle owns the slot"
    assert (
        handle.decoder_input_keepalive is not None
    ), "handle must keep the decode input alive until resolve"

    deltas = handle.resolve()
    assert events[-1] == "event_synchronize"
    assert (
        handle.decoder_input_keepalive is None
    ), "resolve must release the decode input reference"
    assert handle.slot is None and not slot.busy, "resolve must release the slot"
    expected = torch.ones(2 * 4, dtype=torch.float32)
    assert torch.equal(deltas[0], expected)
    assert handle.resolve()[0] is deltas[0], "resolve must be idempotent"
    slot.output_transfer.view(8).zero_()
    assert torch.equal(deltas[0], expected), "resolved deltas must not alias the slot"

    second = scheduler._launch_decode_plans([plan], stream=stream)
    assert torch.equal(second.resolve()[0], expected)
    assert len(created) == 1, "the slot must reuse one event across launches"
    assert events.count("record") == 2 and events.count("event_synchronize") == 2


def test_qwen3_tts_decode_launch_syncs_when_event_record_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event-record failure synchronizes queued decode work and retires the slot."""
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    state = scheduler.create_stream_state("request")
    state.code_chunks.append(torch.ones((2, 2), dtype=torch.long))
    state.total_frames = 2
    plan = scheduler._build_decode_plan(state, is_final=True)
    assert plan is not None

    events: list[str] = []
    created = _force_pinned_cpu_decode(scheduler, monkeypatch, events)

    def make_exploding_event():
        event = _FakeCudaEvent(events)
        event.record_error = RuntimeError("event init failed")
        created.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", make_exploding_event)
    slot = scheduler._thread_decode_slot()
    stream = _FakeDecodeStream(events)

    with pytest.raises(RuntimeError, match="event init failed"):
        scheduler._launch_decode_plans([plan], stream=stream)

    assert (
        "stream_synchronize" in events
    ), "failed record must synchronize the decode stream"
    assert (
        slot.broken and not slot.busy
    ), "a slot whose event failed is released but never reused"
    assert not scheduler._cuda_decode_failed

    events.clear()
    handle = scheduler._launch_decode_plans([plan], stream=stream)
    assert (
        handle.slot is None and "record" not in events
    ), "later launches on this thread use pageable transfers"
    assert torch.equal(handle.resolve()[0], torch.ones(8))


def test_qwen3_tts_short_request_final_flush_decodes_synchronously() -> None:
    """A request that ends before the initial threshold flushes in stream-done."""
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        async_decode=True,
    )
    payload = make_payload(inputs="short", params={"stream": True})
    scheduler._on_streaming_new_request(payload.request_id, payload)
    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(
            torch.ones((2, 2), dtype=torch.long),
            chunk_id=0,
            ref_code_len=0,
        ),
    )
    assert scheduler.outbox.qsize() == 0, "below the threshold nothing is scheduled"

    scheduler._handle_stream_done(payload.request_id)

    chunk = scheduler.outbox.get_nowait()
    assert chunk.type == "stream"
    assert len(chunk.data["audio_waveform"]) == 2 * 4 * 4
    assert scheduler.outbox.get_nowait().type == "result"


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen3_tts_handle_retains_exact_decode_input_until_resolve() -> None:
    """The handle keeps the decoder's exact input alive until resolve()."""
    import gc
    import weakref

    class SlowEchoDecoder:
        # Reuse output storage to avoid allocation after delayed CUDA work
        # begins.
        total_upsample = 1

        def __init__(self) -> None:
            self.seen = None
            self.out = torch.empty((1, 1, 256), dtype=torch.float32, device="cuda")

        def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
            torch.cuda._sleep(300_000_000)
            self.seen = weakref.ref(codes)
            self.out.copy_(codes[:, :1])
            return self.out

    tokenizer = _FakeQwen3TTSTokenizer()
    decoder = SlowEchoDecoder()
    tokenizer.model.decoder = decoder
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cuda",
        initial_cuda_graph=False,
    )
    codes = torch.randint(0, 2048, (256, 2), dtype=torch.long, device="cuda")
    state = scheduler.create_stream_state("request")
    state.num_quantizers = 2
    scheduler.ingest("request", state, codes)
    plan = scheduler._build_decode_plan(state, is_final=True)
    assert plan is not None
    expected = plan.decoder_input[0, 0].to(torch.float32).cpu().clone()
    # Allocate buffers before launch so allocator synchronization cannot
    # affect the in-flight assertions.
    with torch.cuda.stream(scheduler._decode_stream):
        warm = torch.empty(1024, device="cuda")
    del warm
    slot = scheduler._thread_decode_slot()
    slot.input_codes.ensure_capacity(4096)
    slot.output_transfer.ensure_capacity(4096)
    torch.cuda.synchronize()

    handle = scheduler._launch_decode_plans([plan], stream=scheduler._decode_stream)

    assert handle.slot is not None
    assert handle.decoder_input_keepalive is not None
    assert decoder.seen is not None
    assert (
        decoder.seen() is handle.decoder_input_keepalive
    ), "the handle must retain the exact tensor consumed by the decoder"
    del plan
    state.code_chunks.clear()

    deltas = handle.resolve()
    assert handle.slot is None and handle.decoder_input_keepalive is None
    assert torch.equal(deltas[0], expected)
    del codes
    gc.collect()
    assert decoder.seen() is None, "resolve must release the decode input reference"


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen3_tts_pageable_fallback_synchronizes_stream_for_empty_delta_on_cuda() -> (
    None
):
    """The pageable fallback waits for delayed CUDA work when the delta is empty."""
    done_event = torch.cuda.Event()

    class SleepyShortDecoder:
        # Reuse output storage to avoid allocation after delayed CUDA work
        # begins.
        total_upsample = 4

        def __init__(self) -> None:
            self.out = torch.zeros((1, 1, 8), dtype=torch.float32, device="cuda")

        def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
            torch.cuda._sleep(300_000_000)
            done_event.record()
            return self.out

    tokenizer = _FakeQwen3TTSTokenizer()
    tokenizer.model.decoder = SleepyShortDecoder()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cuda",
        initial_cuda_graph=False,
    )
    scheduler._pinned_staging_disabled = True
    state = scheduler.create_stream_state("request")
    state.num_quantizers = 2
    state.code_chunks.append(torch.ones((5, 2), dtype=torch.long))
    state.total_frames = 5
    state.emitted_generated_frames = 4
    plan = scheduler._build_decode_plan(state, is_final=True)
    assert plan is not None
    # Prime the allocator before starting delayed work.
    warm = torch.empty(1024, device="cuda")
    del warm
    torch.cuda.synchronize()

    handle = scheduler._launch_decode_plans([plan], stream=scheduler._decode_stream)

    assert handle.slot is None
    assert done_event.query(), "fallback launch must wait for the decode stream"
    assert handle.resolve()[0].numel() == 0


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen3_tts_decode_input_stays_correct_under_allocator_pressure() -> None:
    """Caller-stream allocation pressure must not change an in-flight decode input."""

    class SlowEchoDecoder:
        total_upsample = 1

        def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
            torch.cuda._sleep(200_000_000)
            return codes[:, :1].to(torch.float32)

    tokenizer = _FakeQwen3TTSTokenizer()
    tokenizer.model.decoder = SlowEchoDecoder()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cuda",
        initial_cuda_graph=False,
    )
    frames = 4096
    codes = torch.randint(0, 2048, (frames, 2), dtype=torch.long, device="cuda")
    state = scheduler.create_stream_state("request")
    state.num_quantizers = 2
    scheduler.ingest("request", state, codes)
    plan = scheduler._build_decode_plan(state, is_final=True)
    assert plan is not None
    expected = plan.decoder_input[0, 0].to(torch.float32).cpu().clone()

    handle = scheduler._launch_decode_plans([plan], stream=scheduler._decode_stream)
    del plan
    state.code_chunks.clear()
    assert handle.slot is not None
    # Apply allocation pressure; this does not guarantee reuse of the input
    # block. Retaining the tensors forces fresh blocks instead of recycling
    # one spare.
    pressure = [
        torch.full((2, frames), 2047, dtype=torch.long, device="cuda")
        for _ in range(64)
    ]
    delta = handle.resolve()[0]
    del pressure
    assert torch.equal(delta, expected)


def test_qwen3_tts_decode_group_isolates_async_bad_rows_in_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async bad rows raise from resolve() after the slot is freed; survivors rerun."""
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    seen: list[torch.Tensor] = []

    def _decode(x):
        seen.append(x.clone())
        return torch.zeros(x.shape[0], 1, 16, dtype=torch.float32)

    scheduler._decoder = SimpleNamespace(chunked_decode=_decode)
    events: list[str] = []
    created = _force_pinned_cpu_decode(scheduler, monkeypatch, events)
    stream = _FakeDecodeStream(events)
    failures: list[tuple[str, BaseException]] = []
    monkeypatch.setattr(
        scheduler,
        "_fail_async_stream",
        lambda request_id, state, exc: failures.append((request_id, exc)),
    )
    group = [
        (
            "good",
            scheduler.create_stream_state("good"),
            _qwen3_tts_single_frame_plan(7),
        ),
        (
            "bad",
            scheduler.create_stream_state("bad"),
            _qwen3_tts_single_frame_plan(2150),
        ),
    ]

    decoded = scheduler._decode_group(group, stream=stream)

    assert decoded is not None
    survivors, deltas = decoded
    assert [entry[0] for entry in survivors] == ["good"]
    assert len(deltas) == 1
    assert [request_id for request_id, _ in failures] == ["bad"]
    assert isinstance(failures[0][1], _Qwen3TTSInvalidCodeRows)
    assert failures[0][1].indices == (1,)
    assert [int(x.shape[0]) for x in seen] == [
        2,
        1,
    ], "the group decodes once, then only the survivors rerun"
    assert int(seen[0].max()) == 2047, "bad rows are clamped before the decoder runs"
    assert events.count("record") == 2 and events.count("event_synchronize") == 2
    assert len(created) == 1, "the survivor rerun reuses the slot and its event"
    slot = scheduler._thread_decode_slot()
    assert not slot.busy and not slot.broken


def test_qwen3_tts_pageable_handle_raises_bad_rows_in_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete (pageable) handle still reports bad rows from resolve()."""
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    events: list[str] = []
    _force_pinned_cpu_decode(scheduler, monkeypatch, events)
    scheduler._pinned_staging_disabled = True
    stream = _FakeDecodeStream(events)

    handle = scheduler._launch_decode_plans(
        [_qwen3_tts_single_frame_plan(7), _qwen3_tts_single_frame_plan(2150)],
        stream=stream,
    )

    assert handle.slot is None and "stream_synchronize" in events
    with pytest.raises(_Qwen3TTSInvalidCodeRows) as excinfo:
        handle.resolve()
    assert excinfo.value.indices == (1,)
    assert handle.bad_rows is None and handle.deltas == []
    with pytest.raises(_Qwen3TTSInvalidCodeRows) as again:
        handle.resolve()
    assert again.value.indices == (1,)


def test_qwen3_tts_pinned_grow_failure_falls_back_to_pageable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed pinned allocation disables staging for good and decodes pageably."""
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    plan = _qwen3_tts_two_frame_plan(scheduler)
    events: list[str] = []
    created = _force_pinned_cpu_decode(scheduler, monkeypatch, events)

    def no_pinned_memory(numel, dtype):
        raise RuntimeError("pinned allocation failed")

    monkeypatch.setattr(cuda_staging, "_allocate_pinned", no_pinned_memory)
    stream = _FakeDecodeStream(events)
    slot = scheduler._thread_decode_slot()

    handle = scheduler._launch_decode_plans([plan], stream=stream)

    assert scheduler._pinned_staging_disabled is True
    assert handle.slot is None
    assert "record" not in events and "stream_synchronize" in events
    assert created == []
    assert not slot.busy and not slot.broken
    assert torch.equal(handle.resolve()[0], torch.ones(8))

    monkeypatch.setattr(cuda_staging, "_allocate_pinned", _fake_allocate_pinned)
    events.clear()
    second = scheduler._launch_decode_plans([plan], stream=stream)
    assert second.slot is None and "record" not in events, "fallback is sticky"


def test_qwen3_tts_launch_failure_with_proven_completion_breaks_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch failure whose stream drains releases the slot but never reuses it."""
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    plan = _qwen3_tts_two_frame_plan(scheduler)
    events: list[str] = []
    _force_pinned_cpu_decode(scheduler, monkeypatch, events)
    retained: list = []
    monkeypatch.setattr(qwen3_streaming_vocoder, "_CONTEXT_FATAL_RETAINED", retained)

    def _boom(x):
        raise RuntimeError("decoder exploded")

    working_decoder = scheduler._decoder
    scheduler._decoder = SimpleNamespace(chunked_decode=_boom)
    stream = _FakeDecodeStream(events)
    slot = scheduler._thread_decode_slot()

    with pytest.raises(RuntimeError, match="decoder exploded"):
        scheduler._launch_decode_plans([plan], stream=stream)

    assert "stream_synchronize" in events
    assert slot.broken and not slot.busy
    assert retained == [] and scheduler._cuda_decode_failed is False

    scheduler._decoder = working_decoder
    events.clear()
    handle = scheduler._launch_decode_plans([plan], stream=stream)
    assert handle.slot is None and "record" not in events
    assert "stream_synchronize" in events
    assert torch.equal(handle.resolve()[0], torch.ones(8))


def test_qwen3_tts_resolve_clone_failure_releases_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host-copy failure after the event completed releases the slot intact."""
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    plan = _qwen3_tts_two_frame_plan(scheduler)
    events: list[str] = []
    _force_pinned_cpu_decode(scheduler, monkeypatch, events)
    stream = _FakeDecodeStream(events)
    slot = scheduler._thread_decode_slot()

    class _BoomClone:
        def clone(self):
            raise RuntimeError("host copy failed")

    handle = scheduler._launch_decode_plans([plan], stream=stream)
    handle.deltas = [_BoomClone()]

    with pytest.raises(RuntimeError, match="host copy failed"):
        handle.resolve()

    assert events.count("event_synchronize") == 1
    assert handle.slot is None and not slot.busy and not slot.broken
    assert handle.decoder_input_keepalive is None and handle.deltas == []
    with pytest.raises(RuntimeError, match="previously failed"):
        handle.resolve()
    assert events.count("event_synchronize") == 1, "a terminal handle never waits again"


@pytest.mark.parametrize("failure_point", ["resolve", "launch"])
def test_qwen3_tts_unproven_completion_retains_resources_and_disables_cuda_decode(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    """Unproven completion: nothing is freed and CUDA decode is disabled."""
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    plan = _qwen3_tts_two_frame_plan(scheduler)
    events: list[str] = []
    created = _force_pinned_cpu_decode(scheduler, monkeypatch, events)
    retained: list = []
    monkeypatch.setattr(qwen3_streaming_vocoder, "_CONTEXT_FATAL_RETAINED", retained)
    stream = _FakeDecodeStream(events)
    slot = scheduler._thread_decode_slot()

    if failure_point == "launch":

        def make_exploding_event():
            event = _FakeCudaEvent(events)
            event.record_error = RuntimeError("record failed")
            created.append(event)
            return event

        monkeypatch.setattr(torch.cuda, "Event", make_exploding_event)
        stream.sync_error = RuntimeError("stream dead")
        with pytest.raises(RuntimeError, match="record failed"):
            scheduler._launch_decode_plans([plan], stream=stream)
    else:
        handle = scheduler._launch_decode_plans([plan], stream=stream)
        created[0].sync_error = RuntimeError("event dead")
        stream.sync_error = RuntimeError("stream dead")
        with pytest.raises(RuntimeError, match="event dead"):
            handle.resolve()
        assert handle.slot is slot, "an unproven handle keeps its slot"
        assert handle.decoder_input_keepalive is not None
        with pytest.raises(RuntimeError, match="previously failed"):
            handle.resolve()

    assert slot.busy and slot.broken
    assert scheduler._cuda_decode_failed is True
    assert len(retained) == 1
    bundle = retained[0]
    assert bundle.owner is scheduler and bundle.stream is stream
    assert bundle.slot is slot
    assert bundle.decoder_input is not None
    shapes = [tuple(item.shape) for item in bundle.keepalives]
    if failure_point == "launch":
        # decoder output, its delta, and the CPU source codes
        assert shapes == [(1, 1, 8), (8,), (1, 2, 2)], shapes
    else:
        # decoder output, its delta, and the pinned view still being written
        assert shapes == [(1, 1, 8), (8,), (8,)], shapes
        assert (
            bundle.keepalives[2].data_ptr() == slot.output_transfer.view(8).data_ptr()
        ), "the pinned output view must stay referenced"

    stream.sync_error = None
    with pytest.raises(RuntimeError, match="disabled after an unrecoverable"):
        scheduler._launch_decode_plans([plan], stream=stream)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen3_tts_decode_slot_reuses_event_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two real launches share one event; an empty batch still records and resolves."""

    class EchoOrShortDecoder:
        total_upsample = 4

        def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
            if codes.shape[-1] == 5:
                return torch.zeros(
                    (codes.shape[0], 1, 8), dtype=torch.float32, device=codes.device
                )
            return codes[:, :1].to(torch.float32).repeat_interleave(4, dim=-1)

    tokenizer = _FakeQwen3TTSTokenizer()
    tokenizer.model.decoder = EchoOrShortDecoder()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cuda",
        initial_cuda_graph=False,
    )
    real_event = torch.cuda.Event
    created: list = []

    def counting_event(*args, **kwargs):
        event = real_event(*args, **kwargs)
        created.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", counting_event)
    slot = scheduler._thread_decode_slot()

    first_plan = _qwen3_tts_two_frame_plan(scheduler)
    first = scheduler._launch_decode_plans(
        [first_plan], stream=scheduler._decode_stream
    )
    assert first.slot is slot and slot.busy
    assert torch.equal(first.resolve()[0], torch.ones(8))
    assert first.slot is None and first.decoder_input_keepalive is None

    state = scheduler.create_stream_state("short")
    state.num_quantizers = 2
    state.code_chunks.append(torch.ones((5, 2), dtype=torch.long))
    state.total_frames = 5
    state.emitted_generated_frames = 4
    second_plan = scheduler._build_decode_plan(state, is_final=True)
    assert second_plan is not None
    second = scheduler._launch_decode_plans(
        [second_plan], stream=scheduler._decode_stream
    )
    assert second.slot is slot, "an all-empty batch still goes through the slot"
    assert second.resolve()[0].numel() == 0
    assert second.slot is None and second.decoder_input_keepalive is None

    assert len(created) == 1, "both launches must record the same event"
    assert not slot.busy and not slot.broken


def test_qwen3_tts_streaming_vocoder_decodes_initial_chunk_early() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        stream_followup_stride=2,
    )
    payload = make_payload(inputs="target", params={"stream": True})
    scheduler._on_streaming_new_request(payload.request_id, payload)

    # note (akazaakane): derived from the shipped default instead of hardcoded.
    # This test asserted a 1-frame emit and broke silently when the default moved
    # to 8; the property under test is that the initial threshold stays below the
    # steady stride, not any particular frame count.
    initial_frames = scheduler._default_initial_chunk_frames
    assert initial_frames < scheduler._stream_stride

    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(
            torch.ones((initial_frames, 2), dtype=torch.long),
            chunk_id=0,
            ref_code_len=0,
        ),
    )
    assert scheduler.outbox.qsize() == 1

    first = scheduler.outbox.get_nowait()
    assert len(first.data["audio_waveform"]) == initial_frames * 4 * 4

    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(torch.ones((1, 2), dtype=torch.long), chunk_id=1),
    )
    assert scheduler.outbox.qsize() == 0
    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(torch.ones((1, 2), dtype=torch.long), chunk_id=2),
    )
    assert scheduler.outbox.qsize() == 1
    assert len(scheduler._decoder.decode_inputs) == 2


def test_qwen3_tts_streaming_vocoder_uses_steady_followup_stride() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    payload = make_payload(inputs="target", params={"stream": True})
    scheduler._on_streaming_new_request(payload.request_id, payload)

    initial_frames = scheduler._default_initial_chunk_frames
    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(
            torch.ones((initial_frames, 2), dtype=torch.long),
            chunk_id=0,
            ref_code_len=0,
        ),
    )
    state = scheduler._stream_states[payload.request_id]
    assert state.next_decode_generated_frames == initial_frames + 8

    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(torch.ones((8, 2), dtype=torch.long), chunk_id=1),
    )
    assert state.next_decode_generated_frames == initial_frames + 16
    assert len(scheduler._decoder.decode_inputs) == 2


def test_qwen3_tts_streaming_vocoder_chunk_ramp_schedules_early_chunks() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        stream_chunk_ramp=(2, 4, 8),
    )
    payload = make_payload(inputs="target", params={"stream": True})
    scheduler._on_streaming_new_request(payload.request_id, payload)

    emitted_frames: list[int] = []
    chunk_id = 0

    def feed(frames: int) -> None:
        nonlocal chunk_id
        scheduler._on_chunk(
            payload.request_id,
            _qwen3_tts_stream_item(
                torch.ones((frames, 2), dtype=torch.long),
                chunk_id=chunk_id,
                ref_code_len=0 if chunk_id == 0 else None,
            ),
        )
        chunk_id += 1
        while not scheduler.outbox.empty():
            message = scheduler.outbox.get_nowait()
            emitted_frames.append(len(message.data["audio_waveform"]) // (4 * 4))

    # note (Junnan Li): each schedule point is locked behaviorally by feeding
    # one frame short of it (must not emit) and then the last frame (must).
    feed(1)
    assert emitted_frames == []
    feed(1)
    assert emitted_frames == [2]
    feed(3)
    assert emitted_frames == [2]
    feed(1)
    assert emitted_frames == [2, 4]
    feed(7)
    assert emitted_frames == [2, 4]
    feed(1)
    assert emitted_frames == [2, 4, 8]
    feed(8)
    assert emitted_frames == [2, 4, 8, 8], "past the ramp the steady stride rules"


def test_qwen3_tts_streaming_vocoder_chunk_ramp_covers_graph_shapes() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        stream_chunk_ramp=(2, 4, 8),
    )
    left = scheduler._stream_left_context_frames
    assert scheduler._initial_decode_graphs._input_frames == (left + 2,)
    assert scheduler._followup_decode_graphs._input_frames == (left + 4, left + 8)

    payload = make_payload(inputs="target", params={"stream": True})
    scheduler._on_streaming_new_request(payload.request_id, payload)
    covered = (
        scheduler._initial_decode_graphs._input_frames
        + scheduler._followup_decode_graphs._input_frames
    )
    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(
            torch.ones((left + 2, 2), dtype=torch.long),
            chunk_id=0,
            ref_code_len=left,
        ),
    )
    for chunk_id, frames in enumerate((4, 8, 8), start=1):
        scheduler._on_chunk(
            payload.request_id,
            _qwen3_tts_stream_item(
                torch.ones((frames, 2), dtype=torch.long), chunk_id=chunk_id
            ),
        )
    decode_shapes = [int(codes.shape[-1]) for codes in scheduler._decoder.decode_inputs]
    assert len(decode_shapes) == 4
    assert all(shape in covered for shape in decode_shapes), decode_shapes


def test_qwen3_tts_streaming_vocoder_chunk_ramp_splits_backlogged_first_decode() -> (
    None
):
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        stream_chunk_ramp=(2, 4, 8),
    )
    payload = make_payload(inputs="target", params={"stream": True})
    scheduler._on_streaming_new_request(payload.request_id, payload)

    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(
            torch.ones((20, 2), dtype=torch.long),
            chunk_id=0,
            ref_code_len=0,
        ),
    )
    emitted_frames = []
    while not scheduler.outbox.empty():
        message = scheduler.outbox.get_nowait()
        emitted_frames.append(len(message.data["audio_waveform"]) // (4 * 4))
    assert emitted_frames == [2, 4, 8, 6]

    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(torch.ones((4, 2), dtype=torch.long), chunk_id=1),
    )
    assert scheduler.outbox.qsize() == 0
    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(torch.ones((4, 2), dtype=torch.long), chunk_id=2),
    )
    steady = scheduler.outbox.get_nowait()
    assert len(steady.data["audio_waveform"]) == 8 * 4 * 4


def test_qwen3_tts_streaming_vocoder_request_override_resizes_only_first_chunk() -> (
    None
):
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        stream_chunk_ramp=(2, 4, 8),
    )
    payload = make_payload(
        inputs="target",
        params={"stream": True, "initial_codec_chunk_frames": 1},
    )
    scheduler._on_streaming_new_request(payload.request_id, payload)
    emitted_frames: list[int] = []
    for chunk_id, frames in enumerate((1, 4, 8)):
        scheduler._on_chunk(
            payload.request_id,
            _qwen3_tts_stream_item(
                torch.ones((frames, 2), dtype=torch.long),
                chunk_id=chunk_id,
                ref_code_len=0 if chunk_id == 0 else None,
            ),
        )
        while not scheduler.outbox.empty():
            message = scheduler.outbox.get_nowait()
            emitted_frames.append(len(message.data["audio_waveform"]) // (4 * 4))
    assert emitted_frames == [1, 4, 8], "override resizes only the first chunk"


def test_qwen3_tts_streaming_vocoder_zero_override_with_ramp() -> None:
    # note (Junnan Li): a zero override disables the early chunk (legacy
    # semantics), so the first decode waits for the steady stride; the ramp
    # tail still schedules the following chunks.
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        stream_chunk_ramp=(2, 4, 8),
    )
    payload = make_payload(
        inputs="target",
        params={"stream": True, "initial_codec_chunk_frames": 0},
    )
    scheduler._on_streaming_new_request(payload.request_id, payload)
    emitted_frames: list[int] = []
    for chunk_id, frames in enumerate((16, 4, 8)):
        scheduler._on_chunk(
            payload.request_id,
            _qwen3_tts_stream_item(
                torch.ones((frames, 2), dtype=torch.long),
                chunk_id=chunk_id,
                ref_code_len=0 if chunk_id == 0 else None,
            ),
        )
        while not scheduler.outbox.empty():
            message = scheduler.outbox.get_nowait()
            emitted_frames.append(len(message.data["audio_waveform"]) // (4 * 4))
    assert emitted_frames == [16, 4, 8]


def test_qwen3_tts_streaming_vocoder_singleton_ramp_goes_straight_to_steady() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        stream_chunk_ramp=(2,),
    )
    payload = make_payload(inputs="target", params={"stream": True})
    scheduler._on_streaming_new_request(payload.request_id, payload)
    emitted_frames: list[int] = []
    for chunk_id, frames in enumerate((2, 8, 8)):
        scheduler._on_chunk(
            payload.request_id,
            _qwen3_tts_stream_item(
                torch.ones((frames, 2), dtype=torch.long),
                chunk_id=chunk_id,
                ref_code_len=0 if chunk_id == 0 else None,
            ),
        )
        while not scheduler.outbox.empty():
            message = scheduler.outbox.get_nowait()
            emitted_frames.append(len(message.data["audio_waveform"]) // (4 * 4))
    assert emitted_frames == [2, 8, 8]


def test_qwen3_tts_vocoder_factory_forwards_chunk_ramp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.qwen3_tts import stages

    monkeypatch.setattr(
        stages,
        "_load_qwen3_tts_tokenizer",
        lambda *args, **kwargs: _FakeQwen3TTSTokenizer(),
    )
    monkeypatch.setattr(
        Qwen3TTSStreamingVocoderScheduler, "warmup_now", lambda scheduler: None
    )

    scheduler = stages.create_vocoder_executor(
        "model",
        device="cpu",
        stream_chunk_ramp=[2, 4, 8],
    )
    payload = make_payload(inputs="target", params={"stream": True})
    scheduler._on_streaming_new_request(payload.request_id, payload)
    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(
            torch.ones((2, 2), dtype=torch.long), chunk_id=0, ref_code_len=0
        ),
    )
    first = scheduler.outbox.get_nowait()
    assert len(first.data["audio_waveform"]) == 2 * 4 * 4


def test_qwen3_tts_streaming_vocoder_chunk_ramp_rejects_mixed_config() -> None:
    with pytest.raises(ValueError, match="set only one form"):
        Qwen3TTSStreamingVocoderScheduler(
            _FakeQwen3TTSTokenizer(),
            device="cpu",
            stream_chunk_ramp=(2, 4, 8),
            stream_initial_followup_stride=4,
        )
    # note (Junnan Li): an explicit legacy value is refused even at its
    # default: None is the only "not set" spelling.
    for legacy_frames in (4, 8):
        with pytest.raises(ValueError, match="set only one form"):
            Qwen3TTSStreamingVocoderScheduler(
                _FakeQwen3TTSTokenizer(),
                device="cpu",
                stream_chunk_ramp=(2, 4, 8),
                initial_chunk_frames=legacy_frames,
            )
    with pytest.raises(ValueError, match="must be <= stream_stride"):
        Qwen3TTSStreamingVocoderScheduler(
            _FakeQwen3TTSTokenizer(),
            device="cpu",
            stream_stride=16,
            stream_chunk_ramp=(20, 4),
        )
    with pytest.raises(ValueError, match="at least one entry"):
        Qwen3TTSStreamingVocoderScheduler(
            _FakeQwen3TTSTokenizer(),
            device="cpu",
            stream_chunk_ramp=(),
        )
    for bad_ramp in ((0, 4), (2, -4)):
        with pytest.raises(ValueError, match="stream_chunk_ramp entries must be > 0"):
            Qwen3TTSStreamingVocoderScheduler(
                _FakeQwen3TTSTokenizer(),
                device="cpu",
                stream_chunk_ramp=bad_ramp,
            )
    for non_int_ramp in ("248", 8, (2, 4.0), (True, 4), (2, "4")):
        with pytest.raises(TypeError):
            Qwen3TTSStreamingVocoderScheduler(
                _FakeQwen3TTSTokenizer(),
                device="cpu",
                stream_chunk_ramp=non_int_ramp,
            )


def test_qwen3_tts_streaming_vocoder_zero_initial_chunk_uses_steady_stride() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    payload = make_payload(
        inputs="target",
        params={"stream": True, "initial_codec_chunk_frames": 0},
    )
    scheduler._on_streaming_new_request(payload.request_id, payload)

    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(
            torch.ones((15, 2), dtype=torch.long),
            chunk_id=0,
            ref_code_len=0,
        ),
    )
    assert scheduler.outbox.qsize() == 0
    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(torch.ones((1, 2), dtype=torch.long), chunk_id=1),
    )
    assert scheduler.outbox.qsize() == 1


def test_qwen3_tts_streaming_vocoder_short_utterance_flushes_complete_audio() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
    )
    generated_frames = scheduler._default_initial_chunk_frames - 1
    assert generated_frames > 0
    ref_frames = 2
    total_frames = ref_frames + generated_frames
    all_codes = torch.arange(1, total_frames * 2 + 1, dtype=torch.long).reshape(
        total_frames, 2
    )

    payload = make_payload(inputs="target", params={"stream": True})
    payload.data = Qwen3TTSState(
        audio_codes=all_codes,
        ref_code_len=ref_frames,
        prompt_tokens=2,
        completion_tokens=generated_frames,
    ).to_dict()

    scheduler._on_streaming_new_request(payload.request_id, payload)
    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(all_codes, chunk_id=0, ref_code_len=ref_frames),
    )
    # note (akazaakane): emitting nothing here is the contract, not a stall.
    # Below initial_chunk_frames the stream stays single-chunk until the final
    # flush, which is also why these requests are N/A for C50/C100/C200.
    assert scheduler.outbox.qsize() == 0

    scheduler._on_done(payload.request_id)
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())

    stream_messages = [message for message in messages if message.type == "stream"]
    assert len(stream_messages) == 1
    stream_audio = np.frombuffer(
        stream_messages[0].data["audio_waveform"],
        dtype=np.float32,
    )
    expected = all_codes[ref_frames:, 0].to(torch.float32).repeat_interleave(4).numpy()
    np.testing.assert_array_equal(stream_audio, expected)
    assert any(message.type == "result" for message in messages)
    assert payload.request_id not in scheduler._stream_states


def test_qwen3_tts_stream_output_prepends_reference_once() -> None:
    from sglang_omni.models.qwen3_tts.request_builders import (
        make_qwen3_tts_scheduler_adapters,
    )

    payload = make_payload(inputs="target", params={"stream": True})
    _, _, stream_output_builder = make_qwen3_tts_scheduler_adapters(
        model=None,
        wrapper=None,
    )
    data = Qwen3TTSSGLangRequestData(
        ref_code=torch.tensor([[10, 11], [12, 13]]),
        latest_stream_code_chunk=torch.tensor([1, 2]),
        stream_codec_output=True,
        stage_payload=payload,
    )

    first = stream_output_builder(payload.request_id, data, None)
    assert len(first) == 1
    assert first[0].data.tolist() == [[10, 11], [12, 13], [1, 2]]
    assert first[0].data.device.type == "cpu"
    assert first[0].metadata["ref_code_len"] == 2
    assert first[0].metadata["num_quantizers"] == 2

    data.latest_stream_code_chunk = torch.tensor([3, 4])
    second = stream_output_builder(payload.request_id, data, None)
    assert second[0].data.tolist() == [[3, 4]]
    assert "ref_code_len" not in second[0].metadata


def test_qwen3_tts_stream_output_skips_non_streaming_generation_modes() -> None:
    from sglang_omni.models.qwen3_tts.request_builders import (
        make_qwen3_tts_scheduler_adapters,
    )

    payload = make_payload(inputs="target", params={"stream": True})
    _, _, stream_output_builder = make_qwen3_tts_scheduler_adapters(
        model=None,
        wrapper=None,
    )
    data = Qwen3TTSSGLangRequestData(
        latest_stream_code_chunk=torch.tensor([1, 2]),
        stage_payload=payload,
    )
    data.stream_codec_output = False

    assert stream_output_builder(payload.request_id, data, None) == []


def test_qwen3_tts_streaming_vocoder_matches_full_decode() -> None:
    tokenizer = _FakeQwen3TTSTokenizer()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cpu",
        stream_stride=1,
        stream_followup_stride=2,
        initial_chunk_frames=1,
        stream_left_context_frames=2,
    )
    all_codes = torch.tensor(
        [[10, 11], [12, 13], [1, 2], [3, 4], [5, 6]],
        dtype=torch.long,
    )
    payload = make_payload(inputs="target", params={"stream": True})
    payload.data = Qwen3TTSState(
        audio_codes=all_codes,
        ref_code_len=2,
        prompt_tokens=2,
        completion_tokens=3,
    ).to_dict()

    scheduler._on_streaming_new_request(payload.request_id, payload)
    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(
            all_codes[:3],
            chunk_id=0,
            ref_code_len=2,
        ),
    )
    first_messages = []
    while not scheduler.outbox.empty():
        first_messages.append(scheduler.outbox.get_nowait())
    assert len(first_messages) == 1
    first_audio = np.frombuffer(
        first_messages[0].data["audio_waveform"],
        dtype=np.float32,
    )
    assert first_audio.size == 4

    scheduler._on_chunk(
        payload.request_id,
        _qwen3_tts_stream_item(all_codes[3:], chunk_id=1),
    )
    scheduler._on_done(payload.request_id)
    messages = first_messages
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())

    stream_audio = np.concatenate(
        [
            np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
            for message in messages
            if message.type == "stream"
        ]
    )
    expected = all_codes[2:, 0].to(torch.float32).repeat_interleave(4).numpy()
    np.testing.assert_array_equal(stream_audio, expected)
    result = next(message for message in messages if message.type == "result")
    assert result.data.data == {
        "modality": "audio",
        "sample_rate": 24000,
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
    }
    assert payload.request_id not in scheduler._stream_states


def test_qwen3_tts_streaming_fallback_matches_full_decode_reference_trim() -> None:
    class UnevenTokenizer(_FakeQwen3TTSTokenizer):
        def decode(self, encoded):
            return [np.arange(11, dtype=np.float32)], self.get_output_sample_rate()

    scheduler = Qwen3TTSStreamingVocoderScheduler(
        UnevenTokenizer(),
        device="cpu",
    )
    state = Qwen3TTSState(
        audio_codes=torch.ones((5, 2), dtype=torch.long),
        ref_code_len=2,
    )

    waveform = scheduler._decode_state_audio(state)

    assert waveform is not None
    np.testing.assert_array_equal(waveform.numpy(), np.arange(4, 11))


def test_qwen3_tts_async_followup_flushes_before_result() -> None:
    tokenizer = _FakeQwen3TTSTokenizer()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cpu",
        stream_stride=1,
        stream_followup_stride=2,
        initial_chunk_frames=1,
        stream_left_context_frames=2,
        async_decode=True,
    )
    all_codes = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long)
    payload = make_payload(inputs="target", params={"stream": True})
    payload.data = Qwen3TTSState(
        audio_codes=all_codes,
        completion_tokens=3,
    ).to_dict()

    scheduler.on_serving_start()
    try:
        scheduler._on_streaming_new_request(payload.request_id, payload)
        scheduler._on_chunk(
            payload.request_id,
            _qwen3_tts_stream_item(
                all_codes[:1],
                chunk_id=0,
                ref_code_len=0,
            ),
        )
        first = scheduler.outbox.get(timeout=1)
        scheduler._on_chunk(
            payload.request_id,
            _qwen3_tts_stream_item(all_codes[1:], chunk_id=1),
        )
        scheduler._on_done(payload.request_id)

        followup = scheduler.outbox.get(timeout=1)
        result = scheduler.outbox.get(timeout=1)
    finally:
        scheduler.stop()

    assert first.type == "stream"
    assert followup.type == "stream"
    assert result.type == "result"
    streamed = np.concatenate(
        [
            np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
            for message in (first, followup)
        ]
    )
    expected = all_codes[:, 0].to(torch.float32).repeat_interleave(4).numpy()
    np.testing.assert_array_equal(streamed, expected)
    assert payload.request_id not in scheduler._stream_states


def test_qwen3_tts_async_initial_batches_ready_requests() -> None:
    tokenizer = _FakeQwen3TTSTokenizer()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cpu",
        stream_stride=1,
        initial_chunk_frames=1,
        async_decode=True,
        initial_batch_wait_ms=20,
    )
    payloads = [
        make_payload(inputs="first", params={"stream": True}),
        make_payload(inputs="second", params={"stream": True}),
    ]
    payloads[0].request_id = "req-first"
    payloads[1].request_id = "req-second"

    scheduler.on_serving_start()
    try:
        for payload in payloads:
            scheduler._on_streaming_new_request(payload.request_id, payload)
            scheduler._on_chunk(
                payload.request_id,
                _qwen3_tts_stream_item(
                    torch.ones((1, 2), dtype=torch.long),
                    chunk_id=0,
                    ref_code_len=0,
                ),
            )

        assert scheduler.outbox.get(timeout=1).type == "stream"
        assert scheduler.outbox.get(timeout=1).type == "stream"
    finally:
        for payload in payloads:
            scheduler.abort(payload.request_id)
        scheduler.stop()

    assert [int(codes.shape[0]) for codes in tokenizer.model.decoder.decode_inputs] == [
        2
    ]


def test_qwen3_tts_async_initial_flushes_before_result() -> None:
    tokenizer = _FakeQwen3TTSTokenizer()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cpu",
        stream_stride=1,
        initial_chunk_frames=1,
        async_decode=True,
    )
    payload = make_payload(inputs="target", params={"stream": True})
    payload.data = Qwen3TTSState(
        audio_codes=torch.ones((1, 2), dtype=torch.long),
        completion_tokens=1,
    ).to_dict()

    scheduler.on_serving_start()
    try:
        scheduler._on_streaming_new_request(payload.request_id, payload)
        scheduler._on_chunk(
            payload.request_id,
            _qwen3_tts_stream_item(
                torch.ones((1, 2), dtype=torch.long),
                chunk_id=0,
                ref_code_len=0,
            ),
        )
        scheduler._on_done(payload.request_id)
        stream = scheduler.outbox.get(timeout=1)
        result = scheduler.outbox.get(timeout=1)
    finally:
        scheduler.stop()

    assert stream.type == "stream"
    assert result.type == "result"
    assert payload.request_id not in scheduler._stream_states


def test_qwen3_tts_async_followup_round_robins_backlog() -> None:
    tokenizer = _FakeQwen3TTSTokenizer()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cpu",
        stream_stride=1,
        stream_followup_stride=2,
        initial_chunk_frames=1,
        stream_left_context_frames=2,
        async_decode=True,
    )
    all_codes = torch.arange(1, 15, dtype=torch.long).reshape(7, 2)
    payload = make_payload(inputs="target", params={"stream": True})
    payload.data = Qwen3TTSState(
        audio_codes=all_codes,
        completion_tokens=7,
    ).to_dict()

    scheduler.on_serving_start()
    try:
        scheduler._on_streaming_new_request(payload.request_id, payload)
        scheduler._on_chunk(
            payload.request_id,
            _qwen3_tts_stream_item(
                all_codes[:1],
                chunk_id=0,
                ref_code_len=0,
            ),
        )
        messages = [scheduler.outbox.get(timeout=1)]
        scheduler._on_chunk(
            payload.request_id,
            _qwen3_tts_stream_item(all_codes[1:], chunk_id=1),
        )
        scheduler._on_done(payload.request_id)
        messages.extend(scheduler.outbox.get(timeout=1) for _ in range(4))
    finally:
        scheduler.stop()

    assert [message.type for message in messages] == [
        "stream",
        "stream",
        "stream",
        "stream",
        "result",
    ]
    assert [
        int(codes.shape[-1]) for codes in tokenizer.model.decoder.decode_inputs
    ] == [1, 3, 4, 4]
    streamed = np.concatenate(
        [
            np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
            for message in messages[:-1]
        ]
    )
    expected = all_codes[:, 0].to(torch.float32).repeat_interleave(4).numpy()
    np.testing.assert_array_equal(streamed, expected)


def test_qwen3_tts_async_followup_batches_ready_requests() -> None:
    tokenizer = _FakeQwen3TTSTokenizer()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cpu",
        stream_stride=1,
        stream_followup_stride=2,
        initial_chunk_frames=1,
        async_decode=True,
        followup_batch_wait_ms=20,
    )
    payloads = [
        make_payload(inputs="first", params={"stream": True}),
        make_payload(inputs="second", params={"stream": True}),
    ]
    payloads[0].request_id = "req-first"
    payloads[1].request_id = "req-second"

    scheduler.on_serving_start()
    try:
        for payload in payloads:
            scheduler._on_streaming_new_request(payload.request_id, payload)
            scheduler._on_chunk(
                payload.request_id,
                _qwen3_tts_stream_item(
                    torch.ones((1, 2), dtype=torch.long),
                    chunk_id=0,
                    ref_code_len=0,
                ),
            )
            assert scheduler.outbox.get(timeout=1).type == "stream"

        for payload in payloads:
            scheduler._on_chunk(
                payload.request_id,
                _qwen3_tts_stream_item(
                    torch.ones((2, 2), dtype=torch.long),
                    chunk_id=1,
                ),
            )

        assert scheduler.outbox.get(timeout=1).type == "stream"
        assert scheduler.outbox.get(timeout=1).type == "stream"
    finally:
        for payload in payloads:
            scheduler.abort(payload.request_id)
        scheduler.stop()

    batch_sizes = [
        int(codes.shape[0]) for codes in tokenizer.model.decoder.decode_inputs
    ]
    assert batch_sizes == [1, 1, 2]


def test_qwen3_tts_followup_queue_prioritizes_playback_deadline() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        async_decode=True,
        followup_batch_wait_ms=0,
    )
    later = scheduler.create_stream_state("later")
    later.playback_deadline_s = 20.0
    earlier = scheduler.create_stream_state("earlier")
    earlier.playback_deadline_s = 10.0
    scheduler._enqueue_followup("later", later)
    scheduler._enqueue_followup("earlier", earlier)

    assert scheduler._collect_followup_batch() == [("earlier", earlier)]


@pytest.mark.parametrize("worker", ["initial", "followup"])
def test_qwen3_tts_async_worker_propagates_process_exit(
    monkeypatch: pytest.MonkeyPatch,
    worker: str,
) -> None:
    # note (akazaakane): initial_chunk_frames is pinned alongside the strides so
    # this stays an error-propagation test. On the shipped default of 8 the
    # single frame below never reaches the decode threshold, so no plan is built
    # and the interrupt never fires.
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        stream_stride=1,
        stream_followup_stride=1,
        initial_chunk_frames=1,
    )
    state = scheduler.create_stream_state("request")
    state.num_quantizers = 2
    state.code_chunks.append(torch.ones((1, 2), dtype=torch.long))
    state.total_frames = 1
    if worker == "followup":
        state.decoded_chunks = 1
    scheduler._stream_states["request"] = state

    def interrupt(*args, **kwargs):
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(scheduler, "_launch_decode_plans", interrupt)

    with pytest.raises(KeyboardInterrupt):
        if worker == "initial":
            scheduler._run_initial_batch([("request", state)])
        else:
            scheduler._run_followup_batch([("request", state)])


@pytest.mark.parametrize("commit", ["initial", "followup"])
def test_qwen3_tts_async_commit_propagates_process_exit(
    monkeypatch: pytest.MonkeyPatch,
    commit: str,
) -> None:
    # note (akazaakane): initial_chunk_frames is pinned alongside the stride for
    # the same reason as the worker test above. On the shipped default of 8 the
    # single frame below never reaches the decode threshold, so
    # _build_decode_plan returns None and the assertion under it cannot hold.
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        stream_stride=1,
        initial_chunk_frames=1,
    )
    state = scheduler.create_stream_state("request")
    state.num_quantizers = 2
    state.code_chunks.append(torch.ones((1, 2), dtype=torch.long))
    state.total_frames = 1
    scheduler._stream_states["request"] = state
    plan = scheduler._build_decode_plan(state, is_final=False)
    assert plan is not None

    def interrupt(*args, **kwargs):
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(scheduler, "_commit_decode_plan", interrupt)

    with pytest.raises(KeyboardInterrupt):
        if commit == "initial":
            scheduler._commit_initial("request", state, plan, torch.ones(4))
        else:
            scheduler._commit_followup("request", state, plan, torch.ones(4))


def test_qwen3_tts_async_followup_drops_late_audio_after_abort() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingDecoder(_FakeQwen3TTSDecoder):
        def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
            if self.decode_inputs:
                entered.set()
                assert release.wait(timeout=2)
            return super().chunked_decode(codes)

    tokenizer = _FakeQwen3TTSTokenizer()
    tokenizer.model.decoder = BlockingDecoder()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cpu",
        stream_stride=1,
        stream_followup_stride=2,
        initial_chunk_frames=1,
        async_decode=True,
    )
    payload = make_payload(inputs="target", params={"stream": True})

    scheduler.on_serving_start()
    try:
        scheduler._on_streaming_new_request(payload.request_id, payload)
        scheduler._on_chunk(
            payload.request_id,
            _qwen3_tts_stream_item(
                torch.ones((1, 2), dtype=torch.long),
                chunk_id=0,
                ref_code_len=0,
            ),
        )
        scheduler.outbox.get(timeout=1)
        scheduler._on_chunk(
            payload.request_id,
            _qwen3_tts_stream_item(
                torch.ones((2, 2), dtype=torch.long),
                chunk_id=1,
            ),
        )
        assert entered.wait(timeout=1)
        scheduler.abort(payload.request_id)
        release.set()
        with pytest.raises(Empty):
            scheduler.outbox.get(timeout=0.1)
    finally:
        release.set()
        scheduler.stop()

    assert payload.request_id not in scheduler._stream_states


def test_qwen3_tts_async_initial_drops_late_audio_after_abort() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingDecoder(_FakeQwen3TTSDecoder):
        def chunked_decode(self, codes: torch.Tensor) -> torch.Tensor:
            entered.set()
            assert release.wait(timeout=2)
            return super().chunked_decode(codes)

    tokenizer = _FakeQwen3TTSTokenizer()
    tokenizer.model.decoder = BlockingDecoder()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device="cpu",
        stream_stride=1,
        initial_chunk_frames=1,
        async_decode=True,
    )
    payload = make_payload(inputs="target", params={"stream": True})

    scheduler.on_serving_start()
    try:
        scheduler._on_streaming_new_request(payload.request_id, payload)
        scheduler._on_chunk(
            payload.request_id,
            _qwen3_tts_stream_item(
                torch.ones((1, 2), dtype=torch.long),
                chunk_id=0,
                ref_code_len=0,
            ),
        )
        assert entered.wait(timeout=1)
        scheduler.abort(payload.request_id)
        release.set()
        with pytest.raises(Empty):
            scheduler.outbox.get(timeout=0.1)
    finally:
        release.set()
        scheduler.stop()

    assert payload.request_id not in scheduler._stream_states


def test_qwen3_tts_result_adapter_keeps_code_handoff_tensor_native() -> None:
    """Avoids list serialization between the AR stage and vocoder stage."""
    payload = make_payload(inputs="target")
    data = Qwen3TTSSGLangRequestData(
        req=SimpleNamespace(output_ids=[]),
        output_codes=[torch.tensor([1, 2]), torch.tensor([3, 4])],
        ref_code=torch.tensor([[9, 9]]),
        ref_code_len=1,
        stage_payload=payload,
    )

    result = apply_sglang_qwen3_tts_result(payload, data)

    assert isinstance(result.data["audio_codes"], torch.Tensor)
    assert result.data["audio_codes"].tolist() == [[9, 9], [1, 2], [3, 4]]
    assert result.data["completion_tokens"] == 2
    assert result.data["finish_reason"] == "stop"


def test_qwen3_tts_result_adapter_preserves_length_finish_reason() -> None:
    """A length-capped generation must be distinguishable from natural EOS."""
    payload = make_payload(inputs="target")
    data = Qwen3TTSSGLangRequestData(
        req=SimpleNamespace(output_ids=[]),
        output_codes=[torch.tensor([1, 2])],
        stage_payload=payload,
        finish_reason="length",
    )

    result = apply_sglang_qwen3_tts_result(payload, data)

    assert result.data["finish_reason"] == "length"


def test_qwen3_tts_result_adapter_normalizes_scheduler_stop_reason() -> None:
    payload = make_payload(inputs="target")
    data = Qwen3TTSSGLangRequestData(
        req=SimpleNamespace(
            output_ids=[],
            finished_reason=SimpleNamespace(
                to_json=lambda: {"type": "stop", "matched": 2150}
            ),
        ),
        output_codes=[torch.tensor([1, 2])],
        stage_payload=payload,
    )

    result = apply_sglang_qwen3_tts_result(payload, data)

    assert result.data["finish_reason"] == "stop"


def test_qwen3_tts_result_adapter_infers_length_at_generation_budget() -> None:
    """Without a scheduler reason, reaching the budget is still a length stop."""
    payload = make_payload(inputs="target")
    data = Qwen3TTSSGLangRequestData(
        req=SimpleNamespace(output_ids=[]),
        output_codes=[torch.tensor([1, 2]), torch.tensor([3, 4])],
        stage_payload=payload,
        max_new_tokens=2,
    )

    result = apply_sglang_qwen3_tts_result(payload, data)

    assert result.data["finish_reason"] == "length"


def test_qwen3_tts_state_round_trips_finish_reason() -> None:
    """The reason must survive the stage-payload state round trip."""
    state = Qwen3TTSState.from_dict({"finish_reason": "length"})

    assert state.finish_reason == "length"
    assert state.to_dict()["finish_reason"] == "length"


def test_speech_batch_result_exposes_finish_reason() -> None:
    """Batch JSON results carry the reason the single-item header carries."""
    from sglang_omni.serve.protocol import SpeechBatchResult

    result = SpeechBatchResult(index=0, status="success", finish_reason="length")

    assert result.model_dump(exclude_none=True)["finish_reason"] == "length"


def test_qwen3_tts_request_data_keeps_decode_tensors_on_prepared_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    dtype = torch.float64
    payload = make_payload(inputs="target")
    payload.data = {
        qwen3_request_builders._QWEN3_TTS_PREPARED_MARKER: payload.request_id
    }
    prepared = Qwen3TTSPreparedRequest(
        state=Qwen3TTSState(),
        input_ids_list=[11, 12, 13],
        input_ids=torch.tensor([11, 12, 13], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        trailing_text_hidden=torch.randn(2, 4, dtype=dtype),
        ref_code=torch.tensor([[9, 9]], dtype=torch.long),
        prompt_input_embeds=torch.randn(3, 4, dtype=dtype),
        tts_pad_embed=torch.randn(4, dtype=dtype),
        gen_kwargs={
            "max_new_tokens": 16,
            "temperature": 0.8,
            "top_k": 30,
            "repetition_penalty": 1.1,
        },
    )
    with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
        qwen3_request_builders._PREPARED_REQUESTS[payload.request_id] = prepared

    data = build_sglang_qwen3_tts_request(
        payload,
        model=SimpleNamespace(
            config=SimpleNamespace(codec_eos_token_id=42, vocab_size=1200)
        ),
        wrapper=object(),
    )

    assert data.prompt_input_embeds is prepared.prompt_input_embeds
    assert data.prefill_input_embeds is prepared.prompt_input_embeds
    assert data.ref_code is prepared.ref_code
    assert data.tts_pad_embed is prepared.tts_pad_embed
    assert data.stream_codec_output is True
    assert isinstance(data.pending_text_queue, PendingTextTensorQueue)
    assert data.pending_text_queue.rows is not None
    assert data.pending_text_queue.rows.device == prepared.trailing_text_hidden.device
    assert data.pending_text_queue.rows.dtype == prepared.trailing_text_hidden.dtype
    assert isinstance(data.semantic_sampling_seed, int)
    assert 0 <= data.semantic_sampling_seed <= 0x7FFFFFFF
    assert data.req.sampling_params.sampling_seed == data.semantic_sampling_seed
    assert data.req.sampling_params.repetition_penalty == 1.1
    assert isinstance(data.subtalker_sampling_seed, int)
    assert 0 <= data.subtalker_sampling_seed <= 0x7FFFFFFF


def test_qwen3_tts_request_data_uses_private_sampling_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    urandom_values = iter([b"\x39\x30\x00\x00", b"\x32\x09\x01\x00"])
    monkeypatch.setattr(
        sampling_seed.os,
        "urandom",
        lambda size: next(urandom_values) if size == 4 else b"\x00" * size,
    )
    payload = make_payload(inputs="target")
    payload.data = {
        qwen3_request_builders._QWEN3_TTS_PREPARED_MARKER: payload.request_id
    }
    prepared = Qwen3TTSPreparedRequest(
        state=Qwen3TTSState(),
        input_ids_list=[11, 12],
        input_ids=torch.tensor([11, 12], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        trailing_text_hidden=torch.randn(1, 4),
        ref_code=None,
        prompt_input_embeds=torch.randn(2, 4),
        tts_pad_embed=torch.randn(4),
        gen_kwargs={"max_new_tokens": 16},
    )
    with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
        qwen3_request_builders._PREPARED_REQUESTS[payload.request_id] = prepared

    data = build_sglang_qwen3_tts_request(
        payload,
        model=SimpleNamespace(
            config=SimpleNamespace(codec_eos_token_id=42, vocab_size=1200)
        ),
        wrapper=object(),
    )

    assert data.semantic_sampling_seed == 12345
    assert data.subtalker_sampling_seed == 67890
    assert data.req.sampling_params.sampling_seed == data.semantic_sampling_seed


def test_qwen3_tts_request_data_uses_public_seed_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    payload = make_payload(inputs="target")
    payload.data = {
        qwen3_request_builders._QWEN3_TTS_PREPARED_MARKER: payload.request_id
    }
    prepared = Qwen3TTSPreparedRequest(
        state=Qwen3TTSState(seed=123),
        input_ids_list=[11, 12],
        input_ids=torch.tensor([11, 12], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        trailing_text_hidden=torch.randn(1, 4),
        ref_code=None,
        prompt_input_embeds=torch.randn(2, 4),
        tts_pad_embed=torch.randn(4),
        gen_kwargs={"max_new_tokens": 16},
    )
    with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
        qwen3_request_builders._PREPARED_REQUESTS[payload.request_id] = prepared

    data = build_sglang_qwen3_tts_request(
        payload,
        model=SimpleNamespace(
            config=SimpleNamespace(codec_eos_token_id=42, vocab_size=1200)
        ),
        wrapper=object(),
    )
    expected_semantic_seed, expected_subtalker_seed = derive_qwen3_tts_sampling_seeds(
        123
    )

    assert data.semantic_sampling_seed == expected_semantic_seed
    assert data.subtalker_sampling_seed == expected_subtalker_seed
    assert data.req.sampling_params.sampling_seed == expected_semantic_seed


def _stage_qwen3_tts_prepared(payload: StagePayload) -> None:
    prepared = Qwen3TTSPreparedRequest(
        state=Qwen3TTSState(),
        input_ids_list=[11, 12, 13],
        input_ids=torch.tensor([11, 12, 13], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        trailing_text_hidden=torch.randn(1, 4),
        ref_code=None,
        prompt_input_embeds=torch.randn(3, 4),
        tts_pad_embed=torch.randn(4),
        gen_kwargs={"max_new_tokens": 16},
    )
    payload.data = {
        qwen3_request_builders._QWEN3_TTS_PREPARED_MARKER: payload.request_id
    }
    with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
        qwen3_request_builders._PREPARED_REQUESTS[payload.request_id] = prepared


def _build_qwen3_tts_sglang_request(monkeypatch: pytest.MonkeyPatch):
    install_fake_sglang(monkeypatch)
    payload = make_payload(inputs="target")
    _stage_qwen3_tts_prepared(payload)
    return build_sglang_qwen3_tts_request(
        payload,
        model=SimpleNamespace(
            config=SimpleNamespace(codec_eos_token_id=42, vocab_size=1200)
        ),
        wrapper=object(),
    )


def test_qwen3_tts_prompt_key_and_tail_guard_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.scheduling import omni_scheduler as scheduler_module

    first = _build_qwen3_tts_sglang_request(monkeypatch)
    second = _build_qwen3_tts_sglang_request(monkeypatch)

    assert first.req.rid == second.req.rid
    assert list(first.req.origin_input_ids) == list(second.req.origin_input_ids)
    assert first.req.extra_key == second.req.extra_key == "qwen3_tts:prompt:v1"
    assert first.req._omni_prompt_cache_key == second.req._omni_prompt_cache_key

    seen = []
    emit = iter([False, True, True])

    def process_result(_, batch, __):
        seen.append(
            [getattr(req, "skip_radix_cache_insert", False) for req in batch.reqs]
        )
        if next(emit):
            for req in batch.reqs:
                req.output_ids.append(7)

    monkeypatch.setattr(
        scheduler_module._Upstream, "process_batch_result", process_result
    )
    scheduler = object.__new__(OmniScheduler)
    plain = SimpleNamespace(output_ids=[])
    batch = SimpleNamespace(reqs=[first.req, second.req, plain])
    scheduler.process_batch_result(batch, None)
    assert seen == [[False, False, False]]
    assert not any(getattr(req, "skip_radix_cache_insert", False) for req in batch.reqs)
    scheduler.process_batch_result(batch, None)
    assert [getattr(req, "skip_radix_cache_insert", False) for req in batch.reqs] == [
        True,
        True,
        False,
    ]
    scheduler.process_batch_result(batch, None)

    assert seen == [
        [False, False, False],
        [False, False, False],
        [True, True, False],
    ]

    first.req.reset_for_retract()
    assert first.req.extra_key == "qwen3_tts:prompt:v1"
    assert first.req.skip_radix_cache_insert


def test_qwen3_tts_prepared_payload_missing_state_fails_without_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    payload = make_payload(inputs="target")
    payload.data = {qwen3_request_builders._QWEN3_TTS_PREPARED_MARKER: "missing"}

    with pytest.raises(RuntimeError, match="must not rebuild"):
        build_sglang_qwen3_tts_request(
            payload,
            model=SimpleNamespace(
                config=SimpleNamespace(codec_eos_token_id=42, vocab_size=1200)
            ),
            wrapper=object(),
        )


def test_qwen3_tts_prepare_custom_voice_uses_speaker_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeWrapper:
        def _build_assistant_text(self, text):
            return f"assistant:{text}"

        def _build_instruct_text(self, text):
            return f"instruct:{text}"

        def _tokenize_texts(self, texts):
            return [torch.arange(8, dtype=torch.long).unsqueeze(0) for _ in texts]

        def _merge_generate_kwargs(self, **kwargs):
            return kwargs

        def create_voice_clone_prompt(self, **kwargs):
            calls.append(("base", kwargs))
            return []

    class FakeModel:
        tts_model_type = "custom_voice"
        model = SimpleNamespace(_feedback_buffer=torch.zeros(4, 4))

        def build_custom_voice_inputs(self, **kwargs):
            calls.append(("custom", kwargs))
            return (
                torch.ones(1, 3, 4),
                torch.ones(1, 3, dtype=torch.long),
                torch.ones(1, 1, 4),
                None,
            )

    monkeypatch.setattr(
        qwen3_request_builders,
        "_build_qwen3_tts_pad_embed",
        lambda model: torch.zeros(4),
    )

    prepared = qwen3_request_builders._prepare_qwen3_tts_request(
        make_payload(
            inputs="target",
            tts_params={
                "task_type": "CustomVoice",
                "voice": "Ryan",
                "instructions": "calm",
            },
        ),
        model=FakeModel(),
        wrapper=FakeWrapper(),
    )

    assert prepared.state.task_type == "CustomVoice"
    assert prepared.state.voice == "Ryan"
    assert [name for name, _ in calls] == ["custom"]
    kwargs = calls[0][1]
    assert kwargs["voice"] == "Ryan"
    assert kwargs["instruct_id"] is not None


def test_qwen3_tts_prepare_voice_design_uses_instruction_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class FakeWrapper:
        def _build_assistant_text(self, text):
            return f"assistant:{text}"

        def _build_instruct_text(self, text):
            return f"instruct:{text}"

        def _tokenize_texts(self, texts):
            return [torch.arange(8, dtype=torch.long).unsqueeze(0) for _ in texts]

        def _merge_generate_kwargs(self, **kwargs):
            return kwargs

    class FakeModel:
        tts_model_type = "voice_design"
        model = SimpleNamespace(_feedback_buffer=torch.zeros(4, 4))

        def build_voice_design_inputs(self, **kwargs):
            calls.append(kwargs)
            return (
                torch.ones(1, 3, 4),
                torch.ones(1, 3, dtype=torch.long),
                torch.ones(1, 1, 4),
                None,
            )

    monkeypatch.setattr(
        qwen3_request_builders,
        "_build_qwen3_tts_pad_embed",
        lambda model: torch.zeros(4),
    )

    prepared = qwen3_request_builders._prepare_qwen3_tts_request(
        make_payload(
            inputs="target",
            tts_params={
                "task_type": "VoiceDesign",
                "instructions": "A warm adult voice.",
            },
        ),
        model=FakeModel(),
        wrapper=FakeWrapper(),
    )

    assert prepared.state.task_type == "VoiceDesign"
    assert prepared.state.instructions == "A warm adult voice."
    assert len(calls) == 1
    assert calls[0]["instruct_id"] is not None


def test_qwen3_tts_base_checkpoint_text_only_rejects_custom_voice_default() -> None:
    class FakeWrapper:
        def _merge_generate_kwargs(self, **kwargs):
            return kwargs

    model = SimpleNamespace(tts_model_type="base")

    with pytest.raises(
        ValueError, match="Base requires ref_audio or speaker_embedding"
    ):
        qwen3_request_builders._prepare_qwen3_tts_request(
            make_payload(inputs="target"),
            model=model,
            wrapper=FakeWrapper(),
        )


def test_qwen3_tts_preprocessing_abort_cleans_prepared_state() -> None:
    """Aborting after preprocessing stored tensors must release the handoff."""
    from sglang_omni.models.qwen3_tts import stages

    request_id = "req-prepared-abort"
    try:
        with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
            qwen3_request_builders._PREPARED_REQUESTS[request_id] = object()

        scheduler = stages.create_preprocessing_executor("model")
        scheduler.abort(request_id)

        with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
            assert request_id not in qwen3_request_builders._PREPARED_REQUESTS
    finally:
        qwen3_request_builders.cleanup_prepared_qwen3_tts_request(request_id)


def test_qwen3_tts_preprocessing_abort_race_cleans_late_prepared_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If preprocessing finishes after abort, its late prepared tensors are dropped."""
    from sglang_omni.models.qwen3_tts import stages

    request_id = "req-preprocess-race"
    started = threading.Event()
    release = threading.Event()

    def fake_preprocess(payload: StagePayload) -> StagePayload:
        started.set()
        assert release.wait(timeout=2.0)
        with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
            qwen3_request_builders._PREPARED_REQUESTS[payload.request_id] = object()
        return payload

    monkeypatch.setattr(stages, "preprocess_qwen3_tts_payload", fake_preprocess)
    scheduler = stages.create_preprocessing_executor("model")
    payload = make_payload(inputs="target")
    payload.request_id = request_id

    thread = threading.Thread(target=scheduler.start, daemon=True)
    try:
        thread.start()
        scheduler.inbox.put(
            IncomingMessage(
                request_id=request_id,
                type="new_request",
                data=payload,
            )
        )
        assert started.wait(timeout=2.0)

        scheduler.abort(request_id)
        release.set()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
                if request_id not in qwen3_request_builders._PREPARED_REQUESTS:
                    break
            time.sleep(0.01)

        assert scheduler.outbox.empty()
        with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
            assert request_id not in qwen3_request_builders._PREPARED_REQUESTS
    finally:
        release.set()
        scheduler.stop()
        thread.join(timeout=2.0)
        qwen3_request_builders.cleanup_prepared_qwen3_tts_request(request_id)


def test_qwen3_tts_ar_scheduler_abort_cleans_prepared_state() -> None:
    """The AR scheduler abort path also owns the prepared handoff cleanup."""
    request_id = "req-ar-abort"
    try:
        with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
            qwen3_request_builders._PREPARED_REQUESTS[request_id] = object()

        scheduler = object.__new__(OmniScheduler)
        scheduler._abort_callback = (
            qwen3_request_builders.cleanup_prepared_qwen3_tts_request
        )
        scheduler._aborted_request_ids = set()
        scheduler._aborted_request_id_order = deque()
        scheduler._pending_stream_ingress = {}
        scheduler._deferred_request_payloads = {}
        scheduler._dirty_deferred_request_ids = set()
        scheduler._first_emit_done = set()
        scheduler._prefill_start_done = set()
        scheduler._prefill_end_done = set()
        scheduler.waiting_queue = []
        scheduler._request_admission_lock = threading.RLock()
        scheduler._request_build_executor = None
        scheduler.request_build_max_pending = 0
        scheduler._pending_request_builds = {}
        scheduler._pending_request_admissions = {}
        scheduler._backlogged_request_build_payloads = []
        scheduler._request_build_max_pending_observed = 0
        scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
        scheduler.cur_batch = None
        scheduler.last_batch = None
        scheduler._async_pending = None
        scheduler.inbox = Queue()

        scheduler.abort(request_id)

        with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
            assert request_id not in qwen3_request_builders._PREPARED_REQUESTS
    finally:
        qwen3_request_builders.cleanup_prepared_qwen3_tts_request(request_id)


def test_qwen3_tts_prefill_attaches_runner_composed_embeddings_to_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefill conditioning stays outside upstream graph-admission fields."""
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner

    calls: list[str] = []
    input_embeds = torch.ones(3, 2)
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.model = SimpleNamespace(
        prepare_decode_buffers=lambda requests: calls.append("prepare")
    )
    runner._build_prefill_input_embeds = (
        lambda forward_batch, requests: calls.append("embeds") or input_embeds
    )
    mm_inputs = [object()]
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(3, dtype=torch.long),
        input_embeds=None,
        replace_embeds=None,
        mm_inputs=mm_inputs,
    )

    runner.before_prefill(forward_batch, object(), [object()])

    sidecar = get_omni_prefill_inputs(forward_batch)
    assert sidecar is not None
    assert sidecar.input_embeds is input_embeds
    assert forward_batch.input_embeds is None
    assert forward_batch.mm_inputs is mm_inputs
    assert calls == ["prepare", "embeds"]


def test_qwen3_tts_prefill_uses_shared_late_bound_forward_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standard generation path late-binds Qwen conditioning and request ids."""
    install_fake_sglang(monkeypatch)
    from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
    from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner

    input_embeds = torch.arange(6, dtype=torch.float32).view(3, 2)
    received: dict[str, object] = {}

    class FakeTalker:
        def prepare_decode_buffers(self, requests) -> None:
            received["prepared_requests"] = requests

        def __call__(self, **kwargs):
            received.update(kwargs)
            return "logits"

    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.model = FakeTalker()
    runner._build_prefill_input_embeds = lambda _batch, _requests: input_embeds

    shared_runner = SGLModelRunner.__new__(SGLModelRunner)
    shared_runner.support_pp = False
    shared_runner.is_generation = True
    shared_runner.dtype = torch.float32

    def forward_batch_generation(forward_batch):
        kwargs = shared_runner._extend_forward_kwargs(forward_batch, object())
        logits_output = runner.model(
            input_ids=forward_batch.input_ids,
            positions=forward_batch.positions,
            forward_batch=forward_batch,
            **kwargs,
        )
        return SimpleNamespace(logits_output=logits_output, next_token_ids=None)

    runner.tp_worker = SimpleNamespace(
        forward_batch_generation=forward_batch_generation
    )
    requests = [SimpleNamespace(request_id="request-a")]
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(3, dtype=torch.long),
        input_embeds=None,
        replace_embeds=None,
        replace_positions=None,
        mm_inputs=[None],
        batch_size=1,
        rids=["request-a"],
        positions=torch.arange(3),
    )
    schedule_batch = SimpleNamespace(is_prefill_only=True)

    result = runner._prepare_and_forward(
        forward_batch,
        schedule_batch,
        requests,
        is_prefill=True,
    )

    assert result.logits_output == "logits"
    assert received["input_embeds"] is input_embeds
    assert received["omni_prefill_rids"] is forward_batch.rids
    assert forward_batch.input_embeds is None
    assert get_omni_prefill_inputs(forward_batch) is None


def test_qwen3_tts_sampling_installs_semantic_seed_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner

    sample_calls: list[list[int]] = []

    def sample(logits_output, forward_batch):
        del logits_output
        sample_calls.append(forward_batch.sampling_info.sampling_seed.tolist())
        return torch.tensor([2, 3], dtype=torch.long)

    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.model = SimpleNamespace(
        _semantic_sampling_seed_tensor=torch.tensor([101, 202], dtype=torch.long),
        config=SimpleNamespace(vocab_size=1200, codec_eos_token_id=1100),
    )
    runner.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(sample=sample))
    forward_batch = SimpleNamespace(
        sampling_info=SimpleNamespace(
            sampling_seed=None,
            need_min_p_sampling=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
        )
    )
    logits_output = SimpleNamespace(next_token_logits=torch.zeros((2, 4)))
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    sampling_params=SimpleNamespace(repetition_penalty=1.0),
                    output_ids=[],
                ),
                return_logprob=False,
            )
        ),
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    sampling_params=SimpleNamespace(repetition_penalty=1.0),
                    output_ids=[],
                ),
                return_logprob=False,
            )
        ),
    ]

    token_ids = runner._sample_next_token_ids(
        logits_output,
        forward_batch,
        object(),
        requests,
    )

    assert token_ids.tolist() == [2, 3]
    assert sample_calls == [[101, 202]]
    assert forward_batch.sampling_info.sampling_seed.tolist() == [101, 202]


def test_qwen3_tts_collect_codes_excludes_semantic_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner

    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner._has_pending_code_step = False

    def code_predictor_forward(layer0_codes, hidden, semantic_positions=None):
        assert layer0_codes.tolist() == [[7], [42]]
        assert hidden.shape == (2, 1, 4)
        assert semantic_positions.tolist() == [3, 3]

    runner.model = SimpleNamespace(
        config=SimpleNamespace(codec_eos_token_id=42),
        code_predictor_forward=code_predictor_forward,
        _output_codes=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        _output_embeds=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    )
    result = SimpleNamespace(
        next_token_ids=torch.tensor([7, 42], dtype=torch.long),
        logits_output=SimpleNamespace(hidden_states=torch.ones((2, 4))),
    )
    schedule_batch = SimpleNamespace(output_ids=None)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        positions=torch.tensor([3, 3], dtype=torch.long),
    )
    requests = [
        SimpleNamespace(request_id="active", data=Qwen3TTSSGLangRequestData()),
        SimpleNamespace(request_id="eos", data=Qwen3TTSSGLangRequestData()),
    ]

    runner._collect_codes(result, forward_batch, schedule_batch, requests)

    assert requests[0].data.output_codes == []
    assert requests[1].data.output_codes == []

    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=requests),
        {
            "active": RequestOutput("active", data=7),
            "eos": RequestOutput("eos", data=42),
        },
    )

    assert [chunk.tolist() for chunk in requests[0].data.output_codes] == [[1, 2]]
    assert requests[0].data.latest_stream_code_chunk.tolist() == [1, 2]
    assert len(requests[0].data.pending_feedback_queue) == 1
    assert requests[1].data.output_codes == []
    assert len(requests[1].data.pending_feedback_queue) == 0

    runner.post_process_outputs(result, SimpleNamespace(requests=requests), {})
    assert len(requests[0].data.output_codes) == 1


def test_qwen3_tts_steady_decode_reports_cuda_graph_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode should use SGLang's graph-capable forward result."""
    install_fake_sglang(monkeypatch)
    from sglang.srt.model_executor import forward_batch_info

    from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
    from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner

    fake_forward_batch = SimpleNamespace(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        mrope_positions=None,
    )
    monkeypatch.setattr(
        forward_batch_info.ForwardBatch,
        "init_new",
        staticmethod(
            lambda model_worker_batch, model_runner, *, capture_hidden_mode=None, return_hidden_states_before_norm: fake_forward_batch
        ),
    )
    monkeypatch.setattr(
        QwenTalkerModelRunner,
        "_take_next_decode_input_embed",
        staticmethod(
            lambda *, sched_req, device, dtype: torch.ones(
                4, device=device, dtype=dtype
            )
        ),
    )

    class FakeQwenModel:
        config = SimpleNamespace(codec_eos_token_id=-1)

        def __init__(self) -> None:
            self._feedback_buffer = torch.zeros(1, 4)
            self._feedback_mask = torch.zeros(1, dtype=torch.bool)
            self._decode_feedback_embedding = torch.nn.Embedding(1, 4)
            self._output_codes = torch.ones(1, 2)
            self._output_embeds = torch.ones(1, 4)
            self.prepare_calls = 0

        def prepare_decode_buffers(self, requests) -> None:
            del requests
            self.prepare_calls += 1

        def code_predictor_forward(
            self,
            layer0_codes,
            hidden,
            semantic_positions=None,
        ) -> None:
            del layer0_codes, hidden, semantic_positions

    class FakeTPWorker:
        gpu_id = 0
        model_runner = SimpleNamespace(model=FakeQwenModel())

        def forward_batch_generation(self, forward_batch):
            del forward_batch
            return SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=torch.ones(1, 4)),
                next_token_ids=torch.tensor([7]),
                can_run_cuda_graph=True,
            )

    class FakeOutputProcessor:
        _capture_hidden = False

        def process(self, model_output, scheduler_output, host_token_ids=None):
            del model_output, host_token_ids
            return {
                req.request_id: RequestOutput(req.request_id, data=7)
                for req in scheduler_output.requests
            }

    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.tp_worker = FakeTPWorker()
    runner.output_processor = FakeOutputProcessor()
    runner.device = torch.device("cpu")
    runner.model = runner.tp_worker.model_runner.model
    runner.bind_execution_bridge(FakeExecutionBridge())

    data = SimpleNamespace(
        req=SimpleNamespace(sampling_params=SimpleNamespace(repetition_penalty=1.0)),
        output_codes=[],
        pending_feedback_queue=[],
        generation_steps=0,
        extra_model_outputs={},
    )
    request = SimpleNamespace(request_id="req", data=data)
    schedule_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False),
        is_prefill_only=False,
    )

    output = runner.execute(
        SimpleNamespace(requests=[request], batch_data=schedule_batch)
    )

    assert output.can_run_cuda_graph is True
    assert runner.model.prepare_calls == 1
    assert fake_forward_batch.input_ids.tolist() == [0]


def test_qwen3_tts_decode_feedback_empty_batch_noops() -> None:
    from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner

    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.model = SimpleNamespace(
        _decode_feedback_embedding=torch.nn.Embedding(1, 4),
    )
    forward_batch = SimpleNamespace(input_ids=torch.empty(0, dtype=torch.long))

    runner._write_feedback_buffers(forward_batch, [])

    assert forward_batch.input_ids.numel() == 0


def test_qwen3_tts_decode_forward_does_not_clear_feedback_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalkerTextModel

    class IdentityNorm(torch.nn.Module):
        def forward(self, hidden_states, residual=None):
            if residual is None:
                return hidden_states
            return hidden_states, residual

    model = Qwen3TTSTalkerTextModel.__new__(Qwen3TTSTalkerTextModel)
    torch.nn.Module.__init__(model)
    model.codec_embedding = torch.nn.Embedding(8, 4)
    model.layers = torch.nn.ModuleList([])
    model.start_layer = 0
    model.end_layer = 0
    model.norm = IdentityNorm()
    model._feedback_buffer = torch.full((1, 4), 5.0)
    model._feedback_mask = torch.tensor([True])
    model._decode_feedback_embedding = torch.nn.Embedding(1, 4)
    model._decode_feedback_embedding.weight.requires_grad_(False)
    with torch.no_grad():
        model._decode_feedback_embedding.weight[0].fill_(7.0)

    output = model.forward(
        input_ids=torch.tensor([0]),
        positions=torch.tensor([0]),
        forward_batch=SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
        ),
    )

    assert torch.equal(output, model._decode_feedback_embedding.weight[:1])
    assert bool(model._feedback_mask[0]) is True


def test_qwen3_tts_decode_forward_rejects_invalid_feedback_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalkerTextModel

    model = Qwen3TTSTalkerTextModel.__new__(Qwen3TTSTalkerTextModel)
    torch.nn.Module.__init__(model)
    model.codec_embedding = torch.nn.Embedding(8, 4)
    model.layers = torch.nn.ModuleList([])
    model.start_layer = 0
    model.end_layer = 0
    model.norm = torch.nn.Identity()
    model._decode_feedback_embedding = torch.nn.Embedding(1, 4)

    with pytest.raises(IndexError):
        model.forward(
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
            forward_batch=SimpleNamespace(
                forward_mode=SimpleNamespace(is_decode=lambda: True),
            ),
        )


def test_qwen3_tts_prepare_decode_buffers_collects_private_subtalker_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker.config = SimpleNamespace(
        code_predictor_config=SimpleNamespace(vocab_size=2048)
    )
    talker.model = SimpleNamespace(
        codec_embedding=SimpleNamespace(weight=torch.empty(1, device="cpu"))
    )
    talker._sub_temperature_tensor = torch.empty(2, dtype=torch.float32)
    talker._sub_top_p_tensor = torch.empty(2, dtype=torch.float32)
    talker._sub_top_k_tensor = torch.empty(2, dtype=torch.long)
    talker._semantic_sampling_seed_tensor = torch.empty(2, dtype=torch.long)
    talker._sub_sampling_seed_tensor = torch.empty(2, dtype=torch.long)
    talker._sub_do_sample_tensor = torch.empty(2, dtype=torch.bool)
    talker._sub_identity_row_indices_tensor = torch.arange(2, dtype=torch.long)
    talker._sub_sample_row_indices_tensor = torch.empty(2, dtype=torch.long)
    requests = [
        SimpleNamespace(
            data=Qwen3TTSSGLangRequestData(
                semantic_sampling_seed=5,
                subtalker_dosample=True,
                subtalker_temperature=0.8,
                subtalker_top_p=0.9,
                subtalker_top_k=40,
                subtalker_sampling_seed=7,
            )
        ),
        SimpleNamespace(
            data=Qwen3TTSSGLangRequestData(
                semantic_sampling_seed=9,
                subtalker_dosample=False,
                subtalker_temperature=1.0,
                subtalker_top_p=1.0,
                subtalker_top_k=-1,
                subtalker_sampling_seed=11,
            )
        ),
    ]

    Qwen3TTSTalker.prepare_decode_buffers(talker, requests)

    assert talker._sub_batch_size == 2
    assert talker._semantic_sampling_seed_tensor[:2].tolist() == [5, 9]
    assert talker._sub_sampling_seed_tensor[:2].tolist() == [7, 11]
    assert talker._sub_temperature_tensor[:2].tolist() == pytest.approx([0.8, 1.0])
    assert talker._sub_top_k_tensor[:2].tolist() == [40, 1]
    assert talker._sub_do_sample_tensor[:2].tolist() == [True, False]
    assert talker._sub_sample_count == 1
    assert talker._sub_identity_row_indices_tensor.tolist() == [0, 1]
    assert talker._sub_has_sampled_rows is True
    assert talker._sub_sampled_has_top_p is True
    # top_k=40 ladder-quantizes to 50 (shared predictor-graph key width).
    assert talker._sub_sampled_max_top_k == 50
    assert talker._sub_sampled_has_unbounded_top_k is False


def test_qwen3_tts_prepare_decode_buffers_requires_owned_request_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker._sub_temperature_tensor = torch.empty(1, dtype=torch.float32)
    requests = [SimpleNamespace(data=SimpleNamespace())]

    with pytest.raises(TypeError, match="request data with"):
        Qwen3TTSTalker.prepare_decode_buffers(talker, requests)


def test_qwen3_tts_subtalker_sampling_batches_argmax_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker._sub_batch_size = 2
    talker._sub_has_sampled_rows = False

    tokens = Qwen3TTSTalker._sample_subtalker_token(
        talker,
        torch.tensor([[0.1, 0.9], [0.7, 0.2]]),
        0,
    )

    assert tokens.tolist() == [1, 0]


def test_qwen3_tts_subtalker_sampling_batches_sampled_path_without_global_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts import sglang_model
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker.config = SimpleNamespace(num_code_groups=4)
    talker._sub_batch_size = 2
    talker._sub_temperature_tensor = torch.tensor([1.0, 1.0])
    talker._sub_top_p_tensor = torch.tensor([1.0, 1.0])
    talker._sub_top_k_tensor = torch.tensor([-1, -1])
    talker._sub_sampling_seed_tensor = torch.tensor([17, 23])
    talker._sub_do_sample_tensor = torch.tensor([True, True])
    talker._sub_identity_row_indices_tensor = torch.tensor([0, 1])
    talker._sub_sample_row_indices_tensor = torch.tensor([0, 1])
    talker._sub_sample_max_row_index = 1
    talker._sub_sample_count = 2
    talker._sub_has_sampled_rows = True
    talker._sub_sampled_has_top_p = False
    talker._sub_sampled_max_top_k = 0
    talker._sub_sampled_has_unbounded_top_k = True

    sampler_calls = []

    def fake_multinomial_with_seed(logprobs, seed, positions):
        assert torch.all(logprobs <= 0)
        assert torch.allclose(logprobs.exp().sum(dim=1), torch.ones(logprobs.shape[0]))
        sampler_calls.append(
            {
                "logprobs": logprobs.detach().clone(),
                "seed": seed.detach().clone(),
                "positions": positions.detach().clone(),
            }
        )
        return torch.zeros(
            (logprobs.shape[0], 1), device=logprobs.device, dtype=torch.long
        )

    monkeypatch.setattr(
        sglang_model, "multinomial_with_seed", fake_multinomial_with_seed
    )

    def fail_multinomial(*args, **kwargs):
        del args, kwargs
        raise AssertionError("sampled subtalker path must not use global RNG")

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)

    def fail_argmax(*args, **kwargs):
        del args, kwargs
        raise AssertionError("all-sampled subtalker path must not compute argmax")

    monkeypatch.setattr(torch, "argmax", fail_argmax)

    tokens = Qwen3TTSTalker._sample_subtalker_token(
        talker,
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        0,
        semantic_positions=torch.tensor([3, 3]),
    )

    assert tokens.shape == (2,)
    assert tokens.dtype == torch.long
    assert set(tokens.tolist()) <= {0, 1}
    assert sampler_calls[0]["seed"].tolist() == [17, 23]
    assert sampler_calls[0]["positions"].tolist() == [10, 10]

    Qwen3TTSTalker._sample_subtalker_token(
        talker,
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        1,
        semantic_positions=torch.tensor([3, 3]),
    )

    assert sampler_calls[1]["positions"].tolist() == [11, 11]


def test_qwen3_tts_subtalker_top_p_keeps_threshold_crossing_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts import sglang_model
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker.config = SimpleNamespace(num_code_groups=4)
    talker._sub_temperature_tensor = torch.tensor([1.0])
    talker._sub_top_p_tensor = torch.tensor([0.5])
    talker._sub_top_k_tensor = torch.tensor([-1])
    talker._sub_sampling_seed_tensor = torch.tensor([17])
    talker._sub_sampled_has_top_p = True
    talker._sub_sampled_max_top_k = 0
    talker._sub_sampled_has_unbounded_top_k = True
    sampler_calls = []

    def fake_multinomial_with_seed(logprobs, seed, positions):
        del seed, positions
        sampler_calls.append(logprobs.detach().clone())
        return torch.ones((1, 1), dtype=torch.long)

    monkeypatch.setattr(
        sglang_model, "multinomial_with_seed", fake_multinomial_with_seed
    )

    token = Qwen3TTSTalker._sample_subtalker_token_seeded(
        talker,
        torch.log(torch.tensor([[0.4, 0.35, 0.25]])),
        0,
        row_indices=torch.tensor([0]),
        semantic_positions=torch.tensor([0]),
    )

    assert torch.isfinite(sampler_calls[0]).tolist() == [[True, True, False]]
    assert torch.allclose(sampler_calls[0][0, :2].exp(), torch.tensor([0.4, 0.35]))
    assert token.item() == 1


def test_qwen3_tts_sampled_subtalker_requires_semantic_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker.config = SimpleNamespace(num_code_groups=4)
    talker._sub_batch_size = 1
    talker._sub_temperature_tensor = torch.tensor([1.0])
    talker._sub_top_p_tensor = torch.tensor([1.0])
    talker._sub_top_k_tensor = torch.tensor([-1])
    talker._sub_sampling_seed_tensor = torch.tensor([17])
    talker._sub_do_sample_tensor = torch.tensor([True])
    talker._sub_identity_row_indices_tensor = torch.tensor([0])
    talker._sub_sample_row_indices_tensor = torch.tensor([0])
    talker._sub_sample_max_row_index = 0
    talker._sub_sample_count = 1
    talker._sub_has_sampled_rows = True
    talker._sub_sampled_has_top_p = False
    talker._sub_sampled_max_top_k = 0
    talker._sub_sampled_has_unbounded_top_k = True

    with pytest.raises(RuntimeError, match="require positions"):
        Qwen3TTSTalker._sample_subtalker_token(
            talker,
            torch.tensor([[0.0, 0.0]]),
            0,
        )


def test_qwen3_tts_engine_disables_torch_compile_by_default() -> None:
    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

    defaults = Qwen3TtsEngineBuilder().generation_defaults(dtype="bfloat16")

    assert defaults["enable_torch_compile"] is False


@pytest.mark.parametrize("value", [True, 1, "1", "true", " yes ", "on"])
def test_qwen3_tts_engine_rejects_torch_compile(value) -> None:
    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

    with pytest.raises(ValueError, match="Qwen3-TTS torch.compile is not supported"):
        Qwen3TtsEngineBuilder().adjust_overrides({"enable_torch_compile": value})


@pytest.mark.parametrize("value", [False, 0, "false", "no", "", None])
def test_qwen3_tts_engine_accepts_disabled_torch_compile(value) -> None:
    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

    Qwen3TtsEngineBuilder().adjust_overrides({"enable_torch_compile": value})


def test_qwen3_tts_engine_accepts_64_batch_policy_and_enables_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from transformers import AutoProcessor
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.utils import generic

    from sglang_omni.models.qwen3_tts import model_runner as model_runner_mod
    from sglang_omni.models.qwen3_tts import request_builders as request_builders_mod
    from sglang_omni.models.qwen3_tts import stages
    from sglang_omni.models.qwen3_tts.request_builders import (
        clear_qwen3_tts_preprocessing_context,
    )
    from sglang_omni.scheduling import bootstrap as bootstrap_mod
    from sglang_omni.scheduling import omni_scheduler as scheduler_mod
    from sglang_omni.scheduling import sglang_backend

    check_model_inputs_calls = []
    expected_cuda_graph_bs = [
        1,
        2,
        4,
        8,
        12,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
    ]

    def transformers_56_check_model_inputs(func):
        check_model_inputs_calls.append(func)
        return f"wrapped:{func.__name__}"

    monkeypatch.setattr(
        generic, "check_model_inputs", transformers_56_check_model_inputs
    )
    monkeypatch.delitem(ROPE_INIT_FUNCTIONS, "default", raising=False)

    build_kwargs: dict = {}
    infrastructure_saw_deferred_capture: list[bool] = []
    init_graph_calls: list[bool] = []

    class FakeModel:
        def load_speech_tokenizer(self, tokenizer) -> None:
            self.speech_tokenizer = tokenizer

    class FakeSGLangRunner:
        def __init__(self, server_args) -> None:
            self.server_args = server_args
            self.model = FakeModel()

        def init_cuda_graphs(self) -> None:
            assert self.server_args.enable_torch_compile is False
            assert self.server_args.torch_compile_max_bs == 64
            init_graph_calls.append(True)

    class FakeWorker:
        def __init__(self, server_args) -> None:
            self.model_runner = FakeSGLangRunner(server_args)
            self.enable_prefill_input_embeds = False

    class FakeQwen3TTSModel:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    qwen_tts_module = types.ModuleType("qwen_tts")
    qwen_tts_module.Qwen3TTSModel = FakeQwen3TTSModel
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_tts_module)

    from sglang_omni.scheduling import engine_factory
    from sglang_omni.scheduling.generation_batch_policy import (
        validate_generation_batch_policy as validate_generation_batch_policy_impl,
    )

    monkeypatch.setattr(stages, "_register_qwen3_tts_hf_config", lambda: None)
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(
        engine_factory, "_resolve_checkpoint", lambda model_path: model_path
    )

    validation_state: dict[str, object] = {}

    def record_generation_batch_validation(
        *, model_name, server_args, model_buffer_bs=None
    ):
        decode_config = server_args.cuda_graph_config.decode
        validation_state.update(
            {
                "model_name": model_name,
                "max_running_requests": server_args.max_running_requests,
                "cuda_graph_max_bs": decode_config.max_bs,
                "cuda_graph_bs": list(decode_config.bs),
                "torch_compile_max_bs": server_args.torch_compile_max_bs,
                "enable_torch_compile": server_args.enable_torch_compile,
            }
        )
        return validate_generation_batch_policy_impl(
            model_name=model_name,
            server_args=server_args,
            model_buffer_bs=model_buffer_bs,
        )

    monkeypatch.setattr(
        engine_factory,
        "validate_generation_batch_policy",
        record_generation_batch_validation,
    )
    monkeypatch.setattr(
        stages,
        "_load_qwen3_tts_tokenizer",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        AutoProcessor,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: object()),
    )
    monkeypatch.setattr(
        request_builders_mod,
        "make_qwen3_tts_scheduler_adapters",
        lambda **kwargs: (
            lambda payload: payload,
            lambda data: data,
            lambda request_id, data, output: [],
        ),
    )

    def fake_build_sglang_server_args(model_path, context_length, **kwargs):
        del model_path, context_length
        build_kwargs.update(kwargs)
        return SimpleNamespace(
            cuda_graph_bs=kwargs["cuda_graph_bs"],
            cuda_graph_max_bs=kwargs["cuda_graph_max_bs"],
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(
                    max_bs=kwargs["cuda_graph_max_bs"],
                    bs=kwargs["cuda_graph_bs"],
                ),
                prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
            ),
            disable_cuda_graph=kwargs["disable_cuda_graph"],
            disable_overlap_schedule=kwargs["disable_overlap_schedule"],
            enable_deterministic_inference=kwargs.get(
                "enable_deterministic_inference", False
            ),
            enable_torch_compile=kwargs["enable_torch_compile"],
            page_size=1,
            chunked_prefill_size=0,
            max_prefill_tokens=kwargs["max_prefill_tokens"],
            max_running_requests=kwargs["max_running_requests"],
            torch_compile_max_bs=kwargs["torch_compile_max_bs"],
        )

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        del gpu_id
        infrastructure_saw_deferred_capture.append(
            bool(kwargs.get("defer_cuda_graph_capture"))
        )
        return (
            FakeWorker(server_args),
            object(),
            object(),
            object(),
            SimpleNamespace(),
        )

    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(
        bootstrap_mod,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        model_runner_mod,
        "Qwen3TTSModelRunner",
        lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
    )
    monkeypatch.setattr(
        scheduler_mod,
        "OmniScheduler",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    scheduler = stages.create_sglang_tts_engine_executor(
        "model",
        device=None,
        server_args_overrides={
            "cuda_graph_max_bs": 64,
            "torch_compile_max_bs": 64,
            "mem_fraction_static": 0.7,
            "max_running_requests": 64,
        },
    )

    assert build_kwargs["disable_cuda_graph"] is False
    assert build_kwargs["cuda_graph_bs"] == expected_cuda_graph_bs
    assert build_kwargs["cuda_graph_max_bs"] == 64
    assert build_kwargs["enable_torch_compile"] is False
    assert build_kwargs["sampling_backend"] == "pytorch"
    assert build_kwargs["mem_fraction_static"] == 0.7
    assert build_kwargs["max_running_requests"] == 64
    assert build_kwargs["torch_compile_max_bs"] == 64
    assert validation_state == {
        "model_name": "Qwen3-TTS",
        "max_running_requests": 64,
        "cuda_graph_max_bs": 64,
        "cuda_graph_bs": expected_cuda_graph_bs,
        "torch_compile_max_bs": 64,
        "enable_torch_compile": False,
    }

    def target():
        return None

    decorator = generic.check_model_inputs()
    assert decorator(target) == "wrapped:target"
    assert generic.check_model_inputs(target) == "wrapped:target"
    assert check_model_inputs_calls == [target, target]

    inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS["default"](
        SimpleNamespace(
            rope_theta=10000.0,
            hidden_size=8,
            num_attention_heads=2,
        ),
        None,
    )
    assert attention_scaling == 1.0
    torch.testing.assert_close(
        inv_freq,
        torch.tensor([1.0, 0.01], dtype=torch.float32),
    )

    assert infrastructure_saw_deferred_capture == [True]
    assert init_graph_calls == [True]
    assert scheduler.server_args.cuda_graph_bs == expected_cuda_graph_bs
    assert scheduler.server_args.cuda_graph_max_bs == 64
    assert scheduler.server_args.disable_cuda_graph is False
    assert scheduler.server_args.enable_torch_compile is False
    assert scheduler.server_args.torch_compile_max_bs == 64
    clear_qwen3_tts_preprocessing_context()


def test_qwen3_tts_engine_probes_runtime_before_checkpoint_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.qwen3_tts import engine_builder as engine_builder_mod
    from sglang_omni.scheduling import engine_factory

    checkpoint_resolutions: list[str] = []

    def fake_resolve_checkpoint(model_path: str) -> str:
        checkpoint_resolutions.append(model_path)
        raise AssertionError("_resolve_checkpoint should not run before qwen_tts probe")

    original_import_module = engine_builder_mod.importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "qwen_tts":
            raise ImportError("missing qwen_tts")
        return original_import_module(name, package)

    monkeypatch.setattr(engine_factory, "_resolve_checkpoint", fake_resolve_checkpoint)
    monkeypatch.setattr(
        engine_builder_mod.importlib, "import_module", fake_import_module
    )

    with pytest.raises(ImportError, match="missing qwen_tts"):
        engine_builder_mod.Qwen3TtsEngineBuilder().resolve_checkpoint(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        )

    assert checkpoint_resolutions == []


def test_qwen3_tts_cli_mem_fraction_static_pins_tts_engine() -> None:
    from sglang_omni.cli.serve import patches_from_broadcast_flags
    from sglang_omni.config.resolver import ConfigResolver

    config = Qwen3TTSPipelineConfig(model_path="fake-model")

    resolved = (
        ConfigResolver(config)
        .resolve(
            patches_from_broadcast_flags(
                config,
                mem_fraction_static=0.27,
            )
        )
        .config
    )

    tts_engine = resolved.stage_named("tts_engine")
    assert tts_engine.engine.mem_fraction_static == 0.27
    assert all(
        s.engine is None or s.engine.mem_fraction_static is None
        for s in resolved.stages
        if s.name != "tts_engine"
    )


def test_qwen3_tts_dotted_mem_fraction_wins_over_the_broadcast() -> None:
    from sglang_omni.cli.serve import patches_from_broadcast_flags
    from sglang_omni.config.manager import ConfigManager

    config = Qwen3TTSPipelineConfig(model_path="fake-model")
    patches = patches_from_broadcast_flags(
        config,
        mem_fraction_static=0.27,
    )
    merged = ConfigManager(config).merge_config(
        [("tts_engine.engine.mem_fraction_static", "0.3")],
        extra_patches=patches,
    )

    assert merged.stage_named("tts_engine").engine.mem_fraction_static == 0.3


def test_qwen3_tts_cli_rejects_out_of_range_mem_fraction() -> None:
    from sglang_omni.cli.serve import patches_from_broadcast_flags
    from sglang_omni.config.manager import ConfigManager

    config = Qwen3TTSPipelineConfig(model_path="fake-model")

    # Range is the schema's rule: the flag builds patches, resolution refuses.
    patches = patches_from_broadcast_flags(config, mem_fraction_static=1.5)
    with pytest.raises(ValueError, match="mem_fraction_static"):
        ConfigManager(config).merge_config([], extra_patches=patches)


def test_qwen3_tts_talker_forward_accepts_shared_prefill_request_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared request-id transport does not alter Talker prefill results."""
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    input_embeds = torch.arange(6, dtype=torch.float32).view(3, 2)
    received: dict[str, object] = {}

    def backbone(**kwargs):
        received.update(kwargs)
        return kwargs["input_embeds"]

    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker.model = backbone
    talker.codec_head = lambda hidden_states: (hidden_states + 1, None)
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(3, dtype=torch.long),
        mrope_positions=None,
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        extend_seq_lens=torch.tensor([3]),
    )
    request_ids = ["request-a"]

    result = Qwen3TTSTalker.forward(
        talker,
        input_ids=forward_batch.input_ids,
        positions=torch.arange(3),
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        omni_prefill_rids=request_ids,
    )

    assert received["input_embeds"] is input_embeds
    assert Qwen3TTSTalker.is_mrope_enabled is True
    torch.testing.assert_close(
        result.next_token_logits,
        input_embeds[-1:] + 1,
    )


def _make_prep_talker(monkeypatch):
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker.config = SimpleNamespace(
        code_predictor_config=SimpleNamespace(vocab_size=2048)
    )
    talker._sub_temperature_tensor = torch.empty(2, dtype=torch.float32)
    talker._sub_top_p_tensor = torch.empty(2, dtype=torch.float32)
    talker._sub_top_k_tensor = torch.empty(2, dtype=torch.long)
    talker._semantic_sampling_seed_tensor = torch.empty(2, dtype=torch.long)
    talker._sub_sampling_seed_tensor = torch.empty(2, dtype=torch.long)
    talker._sub_do_sample_tensor = torch.empty(2, dtype=torch.bool)
    talker._sub_identity_row_indices_tensor = torch.arange(2, dtype=torch.long)
    talker._sub_sample_row_indices_tensor = torch.empty(2, dtype=torch.long)
    return Qwen3TTSTalker, talker


def _prep_request(request_id, temperature):
    return SimpleNamespace(
        request_id=request_id,
        data=Qwen3TTSSGLangRequestData(
            semantic_sampling_seed=5,
            subtalker_dosample=True,
            subtalker_temperature=temperature,
            subtalker_top_p=0.9,
            subtalker_top_k=40,
            subtalker_sampling_seed=7,
        ),
    )


def test_qwen3_tts_prepare_decode_buffers_reuses_unchanged_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    talker_cls, talker = _make_prep_talker(monkeypatch)
    requests = [_prep_request("req-a", 0.8)]
    talker_cls.prepare_decode_buffers(talker, requests)
    assert talker._sub_temperature_tensor[:1].tolist() == pytest.approx([0.8])

    # Unchanged batch: staging is skipped, so a manual poke survives.
    talker._sub_temperature_tensor[0] = 0.123
    talker_cls.prepare_decode_buffers(talker, requests)
    assert talker._sub_temperature_tensor[:1].tolist() == pytest.approx([0.123])


def test_qwen3_tts_prepare_decode_buffers_restages_on_request_id_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed request's id may be legally reused by a new request."""
    talker_cls, talker = _make_prep_talker(monkeypatch)
    talker_cls.prepare_decode_buffers(talker, [_prep_request("req-a", 0.8)])
    assert talker._sub_temperature_tensor[:1].tolist() == pytest.approx([0.8])

    # Same request id, brand-new request data: must restage, not reuse.
    talker_cls.prepare_decode_buffers(talker, [_prep_request("req-a", 0.4)])
    assert talker._sub_temperature_tensor[:1].tolist() == pytest.approx([0.4])


def test_qwen3_tts_stream_prune_matches_full_history_windows() -> None:
    """Pruned decode windows must be byte-identical to full-history slicing."""
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        stream_left_context_frames=6,
        initial_chunk_frames=4,
        stream_stride=4,
        stream_followup_stride=3,
    )
    state = scheduler.create_stream_state("request")
    full_history: list[torch.Tensor] = []
    frame = 0
    for step in range(30):
        chunk = torch.arange(frame, frame + 2, dtype=torch.long).reshape(2, 1)
        frame += 2
        full_history.append(chunk.clone())
        scheduler.ingest("request", state, chunk)
        plan = scheduler._build_decode_plan(state, is_final=False)
        if plan is None:
            continue
        codes_full = torch.cat(full_history, dim=0)
        window_end = state.ref_frames + plan.generated_frames
        expected = (
            codes_full[plan.window_start : window_end].transpose(0, 1).unsqueeze(0)
        )
        assert torch.equal(plan.decoder_input, expected), step
        # commit bookkeeping only (no real decode on the fake tokenizer path)
        state.emitted_generated_frames = plan.generated_frames
        state.decoded_chunks += 1
        state.next_decode_generated_frames = plan.generated_frames + 3

    assert state.pruned_frames > 0, "long stream should have pruned dead chunks"
    assert len(state.code_chunks) < len(full_history)


@pytest.mark.parametrize("deterministic", [False, True])
def test_qwen3_tts_decode_isolates_rows_with_out_of_range_codes(
    deterministic: bool,
) -> None:
    """A bad row fails alone and the decoder only ever sees in-range ids."""
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        _FakeQwen3TTSTokenizer(),
        device="cpu",
        enable_deterministic_inference=deterministic,
    )
    seen: list[torch.Tensor] = []

    def _decode(x):
        seen.append(x.clone())
        return torch.zeros(x.shape[0], 1, 16, dtype=torch.float32)

    scheduler._decoder = SimpleNamespace(chunked_decode=_decode)

    def _plan(code):
        return _Qwen3TTSDecodePlan(
            decoder_input=torch.tensor([[[code]]], dtype=torch.long),
            absolute_emitted_frames=0,
            generated_frames=1,
            window_start=0,
            emitted_generated_frames=0,
        )

    with pytest.raises(ValueError) as excinfo:
        scheduler._launch_decode_plans([_plan(7), _plan(2150)], stream=None)
    assert excinfo.value.indices == (1,)
    assert seen == [], "decoder must not run while a row is out of range"

    scheduler._launch_decode_plans([_plan(7), _plan(8)], stream=None).resolve()
    assert [int(item.max()) for item in seen] == ([7, 8] if deterministic else [8])
