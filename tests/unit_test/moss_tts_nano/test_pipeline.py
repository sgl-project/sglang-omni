# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
import types
from contextlib import contextmanager

import pytest
import torch

from sglang_omni.models.moss_tts_local import config as local_config
from sglang_omni.models.moss_tts_local.radix_hash import gpu_radix_row_hash
from sglang_omni.models.moss_tts_nano.config import MossTTSNanoPipelineConfig
from sglang_omni.models.moss_tts_nano.payload_types import MossTTSNanoState
from sglang_omni.models.moss_tts_nano.prompting import (
    ASSISTANT_ROLE_PREFIX,
    ASSISTANT_TURN_PREFIX,
    USER_ROLE_PREFIX,
    USER_TEMPLATE_AFTER_REFERENCE,
    USER_TEMPLATE_REFERENCE_PREFIX,
    USER_TEMPLATE_SUFFIX,
    build_prompt_rows,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload

N_VQ = 16
TEXT_VOCAB_SIZE = 16384
AUDIO_PAD_TOKEN_ID = 1024


class _FakeTokenizer:
    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [100 + ord(character) for character in text]


MODEL_CONFIG = types.SimpleNamespace(
    n_vq=N_VQ,
    audio_pad_token_id=AUDIO_PAD_TOKEN_ID,
    pad_token_id=3,
    im_start_token_id=4,
    im_end_token_id=5,
    audio_start_token_id=6,
    audio_end_token_id=7,
    audio_user_slot_token_id=8,
    audio_assistant_slot_token_id=9,
)


def _encode(text: str) -> list[int]:
    return _FakeTokenizer.encode(text)


def _expected_prompt_text_ids(text: str) -> list[int]:
    return (
        [MODEL_CONFIG.im_start_token_id]
        + _encode(USER_ROLE_PREFIX)
        + _encode(USER_TEMPLATE_REFERENCE_PREFIX)
        + _encode("None")
        + _encode(USER_TEMPLATE_AFTER_REFERENCE)
        + _encode(text)
        + _encode(USER_TEMPLATE_SUFFIX)
        + [MODEL_CONFIG.im_end_token_id]
        + _encode(ASSISTANT_TURN_PREFIX)
        + [MODEL_CONFIG.im_start_token_id]
        + _encode(ASSISTANT_ROLE_PREFIX)
        + [MODEL_CONFIG.audio_start_token_id]
    )


def _install_stub_package(name: str) -> None:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module


def _sglang_is_installed() -> bool:
    try:
        return importlib.util.find_spec("sglang") is not None
    except (ImportError, ValueError):
        return False


@contextmanager
def _nano_request_builders_module():
    modules_before = set(sys.modules)
    using_stubs = not _sglang_is_installed()
    if using_stubs:
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.managers",
            "sglang.srt.sampling",
        ):
            _install_stub_package(name)
        schedule_batch = types.ModuleType("sglang.srt.managers.schedule_batch")

        class _FakeReq:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)
                self.output_ids = []

        schedule_batch.Req = _FakeReq
        sys.modules[schedule_batch.__name__] = schedule_batch
        sampling_params = types.ModuleType("sglang.srt.sampling.sampling_params")

        class _FakeSamplingParams:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

            @staticmethod
            def normalize(tokenizer) -> None:
                del tokenizer

            def verify(self, vocab_size) -> None:
                self.vocab_size = vocab_size

        sampling_params.SamplingParams = _FakeSamplingParams
        sys.modules[sampling_params.__name__] = sampling_params
    try:
        yield importlib.import_module(
            "sglang_omni.models.moss_tts_nano.request_builders"
        )
    finally:
        if using_stubs:
            for name in set(sys.modules) - modules_before:
                if name == "sglang" or name.startswith("sglang."):
                    sys.modules.pop(name, None)
                elif name in {
                    "sglang_omni.models.moss_tts.request_builders",
                    "sglang_omni.models.moss_tts_local.request_builders",
                    "sglang_omni.models.moss_tts_nano.request_builders",
                }:
                    sys.modules.pop(name, None)


@contextmanager
def _nano_stages_module():
    modules_before = set(sys.modules)
    using_stubs = not _sglang_is_installed()
    with _nano_request_builders_module():
        if using_stubs:
            for name in (
                "sglang.kernels",
                "sglang.kernels.ops",
                "sglang.kernels.ops.attention",
            ):
                _install_stub_package(name)
            flash_attention = types.ModuleType(
                "sglang.kernels.ops.attention.flash_attention"
            )
            flash_attention.flash_attn_varlen_func = lambda *args, **kwargs: None
            sys.modules[flash_attention.__name__] = flash_attention
            flash_attention_v3 = types.ModuleType(
                "sglang.kernels.ops.attention.flash_attention_v3"
            )
            flash_attention_v3._is_fa3_supported = lambda: False
            sys.modules[flash_attention_v3.__name__] = flash_attention_v3
        try:
            yield importlib.import_module("sglang_omni.models.moss_tts_nano.stages")
        finally:
            for name in set(sys.modules) - modules_before:
                if name.startswith("sglang_omni.models.moss_tts_nano.stages"):
                    sys.modules.pop(name, None)
                elif name.startswith("sglang_omni.models.moss_tts_local.stages"):
                    sys.modules.pop(name, None)
                elif name.startswith(
                    "sglang_omni.models.moss_tts_local.streaming_vocoder"
                ):
                    sys.modules.pop(name, None)


# Registry / pipeline configuration


def test_registry_and_pipeline_stage_wiring() -> None:
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config("MossTTSNanoForCausalLM")
    assert config_cls is MossTTSNanoPipelineConfig

    config = config_cls(model_path="OpenMOSS-Team/MOSS-TTS-Nano")
    assert [(stage.name, stage.next, stage.terminal) for stage in config.stages] == [
        ("preprocessing", "tts_engine", False),
        ("tts_engine", "vocoder", False),
        ("vocoder", None, True),
    ]
    assert [stage.factory_path for stage in config.stages] == [
        "sglang_omni.models.moss_tts_nano.stages.create_preprocessing_executor",
        "sglang_omni.models.moss_tts_nano.stages.create_tts_engine_executor",
        "sglang_omni.models.moss_tts_nano.stages.create_vocoder_executor",
    ]
    assert config.process_local_edges() == frozenset({("preprocessing", "tts_engine")})
    assert config.supports_uploaded_voice_references() is True
    assert config.stage_named("preprocessing").factory.compute_dtype == "float32"
    assert config.stage_named("vocoder").factory.compute_dtype == "float32"


def test_pipeline_factory_kwargs_receive_resolved_values(monkeypatch) -> None:
    monkeypatch.setattr(local_config, "_uses_rocm_wsl_dxg", lambda: True)
    config = MossTTSNanoPipelineConfig(
        model_path="OpenMOSS-Team/MOSS-TTS-Nano",
        cuda_graph=None,
        cuda_graph_frames=[25, 10, 5],
        cuda_graph_min_free_gb=1.5,
        ref_audio_cache=False,
        ref_audio_cache_max_items=17,
        ref_audio_cache_max_bytes=4096,
    )

    assert config.cuda_graph is None
    assert config.stage_factory_kwargs("preprocessing") == {
        "ref_audio_cache": False,
        "ref_audio_cache_max_items": 17,
        "ref_audio_cache_max_bytes": 4096,
    }
    assert config.stage_factory_kwargs("vocoder") == {
        "cuda_graph": False,
        "cuda_graph_frames": [25, 10, 5],
        "cuda_graph_min_free_gb": 1.5,
    }


def test_pipeline_rejects_unsafe_explicit_dxg_graph_enable(monkeypatch) -> None:
    monkeypatch.setattr(local_config, "_uses_rocm_wsl_dxg", lambda: True)

    with pytest.raises(
        ValueError,
        match="MOSS-TTS-Nano vocoder CUDA graphs cannot be enabled",
    ):
        MossTTSNanoPipelineConfig(
            model_path="OpenMOSS-Team/MOSS-TTS-Nano",
            cuda_graph=True,
        )


def test_codec_factories_default_to_official_fp32_compute() -> None:
    with _nano_stages_module() as stages:
        assert (
            stages.create_preprocessing_executor.__kwdefaults__["compute_dtype"]
            == "float32"
        )
        assert (
            stages.create_vocoder_executor.__kwdefaults__["compute_dtype"] == "float32"
        )


# Prompt construction


def test_prompt_without_reference_has_17_padded_channels() -> None:
    text = "Hello, Nano"
    rows = build_prompt_rows(
        tokenizer=_FakeTokenizer(),
        config=MODEL_CONFIG,
        text=text,
        reference_codes=None,
    )

    expected_text_ids = _expected_prompt_text_ids(text)
    assert tuple(rows.shape) == (len(expected_text_ids), N_VQ + 1)
    assert rows[:, 0].tolist() == expected_text_ids
    assert torch.all(rows[:, 1:] == AUDIO_PAD_TOKEN_ID)


def test_prompt_with_reference_places_16_codebooks_in_user_audio_rows() -> None:
    reference_codes = torch.arange(3 * N_VQ, dtype=torch.long).reshape(3, N_VQ)
    text = "clone me"
    rows = build_prompt_rows(
        tokenizer=_FakeTokenizer(),
        config=MODEL_CONFIG,
        text=text,
        reference_codes=reference_codes,
    )

    prefix = (
        [MODEL_CONFIG.im_start_token_id]
        + _encode(USER_ROLE_PREFIX)
        + _encode(USER_TEMPLATE_REFERENCE_PREFIX)
    )
    audio_start_index = len(prefix)
    audio_rows = rows[audio_start_index + 1 : audio_start_index + 4]
    audio_end_index = audio_start_index + 4

    assert rows.shape[1] == N_VQ + 1
    assert rows[:audio_start_index, 0].tolist() == prefix
    assert int(rows[audio_start_index, 0]) == MODEL_CONFIG.audio_start_token_id
    assert torch.all(rows[audio_start_index, 1:] == AUDIO_PAD_TOKEN_ID)
    assert torch.all(audio_rows[:, 0] == MODEL_CONFIG.audio_user_slot_token_id)
    torch.testing.assert_close(audio_rows[:, 1:], reference_codes)
    assert int(rows[audio_end_index, 0]) == MODEL_CONFIG.audio_end_token_id
    assert torch.all(rows[audio_end_index, 1:] == AUDIO_PAD_TOKEN_ID)
    assert int(rows[-1, 0]) == MODEL_CONFIG.audio_start_token_id
    assert torch.all(rows[-1, 1:] == AUDIO_PAD_TOKEN_ID)


# Sampling defaults / overrides


def test_generation_kwargs_match_official_nano_defaults() -> None:
    with _nano_request_builders_module() as request_builders:
        kwargs = request_builders.build_generation_kwargs(
            {
                # Generic API defaults are intentionally ignored unless the request
                # records that the user supplied them explicitly.
                "temperature": 0.25,
                "top_p": 0.5,
            },
            tts_params={},
        )

    assert kwargs == {
        "max_new_tokens": 375,
        "text_temperature": 1.0,
        "text_top_p": 1.0,
        "text_top_k": 50,
        "audio_temperature": 0.8,
        "audio_top_p": 0.95,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.2,
    }


def test_generation_kwargs_apply_only_explicit_or_nano_specific_overrides() -> None:
    with _nano_request_builders_module() as request_builders:
        kwargs = request_builders.build_generation_kwargs(
            {
                "max_new_tokens": 41,
                "temperature": 0.65,
                "top_p": 0.75,
                "top_k": 19,
                "repetition_penalty": 1.1,
                "audio_temperature": 0.9,
            },
            tts_params={
                "explicit_generation_params": [
                    "temperature",
                    "top_p",
                    "top_k",
                    "repetition_penalty",
                ],
                "audio_top_k": 23,
                "seed": 1234,
            },
        )

    assert kwargs == {
        "max_new_tokens": 41,
        "text_temperature": 0.65,
        "text_top_p": 0.75,
        "text_top_k": 19,
        "audio_temperature": 0.9,
        "audio_top_p": 0.75,
        "audio_top_k": 23,
        "audio_repetition_penalty": 1.1,
        "seed": 1234,
    }


def test_sglang_request_uses_prompt_only_radix_namespace(monkeypatch) -> None:
    payload = _payload()
    generation_kwargs = {
        "max_new_tokens": 12,
        "text_temperature": 1.0,
        "text_top_p": 1.0,
        "text_top_k": 50,
        "audio_temperature": 0.8,
        "audio_top_p": 0.95,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.2,
    }
    prompt_rows = torch.full((3, N_VQ + 1), AUDIO_PAD_TOKEN_ID, dtype=torch.long)

    with _nano_request_builders_module() as request_builders:
        prepared = request_builders.MossTTSNanoPreparedRequest(
            state=MossTTSNanoState(
                text="hello",
                generation_kwargs=generation_kwargs,
            ),
            input_ids_list=[101, 102, 103],
            input_ids=torch.tensor([101, 102, 103]),
            prompt_rows=prompt_rows,
            gen_kwargs=generation_kwargs,
        )
        monkeypatch.setattr(
            request_builders,
            "pop_prepared_moss_tts_nano_request",
            lambda _payload: prepared,
        )
        data = request_builders.build_sglang_moss_tts_nano_request(
            payload,
            model=types.SimpleNamespace(
                config=types.SimpleNamespace(
                    audio_end_token_id=MODEL_CONFIG.audio_end_token_id,
                    vocab_size_list=[TEXT_VOCAB_SIZE],
                )
            ),
        )

    assert data.req.extra_key == "moss_tts_nano:prompt:v1"
    assert data.req._omni_prompt_cache_key == "moss_tts_nano:prompt:v1"
    assert data.req._omni_prompt_only_radix is True


# Nano radix domain


def test_nano_radix_keys_avoid_special_tokens_and_stay_in_text_vocab() -> None:
    torch.manual_seed(7)
    rows = torch.randint(0, 1024, (256, N_VQ + 1), dtype=torch.long)
    rows[:, 0] = MODEL_CONFIG.audio_assistant_slot_token_id
    next_text = torch.full(
        (rows.shape[0],),
        MODEL_CONFIG.audio_assistant_slot_token_id,
        dtype=torch.long,
    )
    next_text[::31] = MODEL_CONFIG.audio_end_token_id

    keys = gpu_radix_row_hash(
        rows,
        next_text,
        MODEL_CONFIG.audio_end_token_id,
        hash_space=TEXT_VOCAB_SIZE,
        hash_offset=10,
    )

    eos_mask = next_text == MODEL_CONFIG.audio_end_token_id
    assert torch.all(keys[eos_mask] == MODEL_CONFIG.audio_end_token_id)
    continuing = keys[~eos_mask]
    assert int(continuing.min()) >= 10
    assert int(continuing.max()) < TEXT_VOCAB_SIZE


# Audio preparation / result state


class _FakeEncodedAudio:
    def __init__(self, audio_codes: torch.Tensor, audio_codes_lengths: torch.Tensor):
        self.audio_codes = audio_codes
        self.audio_codes_lengths = audio_codes_lengths


class _FakeAudioTokenizerModel:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(sampling_rate=48000, number_channels=2)
        self.prepared_wavs: list[torch.Tensor] = []

    def batch_encode(
        self,
        wavs: list[torch.Tensor],
        *,
        num_quantizers: int,
    ) -> _FakeEncodedAudio:
        self.prepared_wavs = [wav.detach().clone() for wav in wavs]
        frame_count = int(wavs[0].shape[-1])
        return _FakeEncodedAudio(
            torch.zeros(num_quantizers, len(wavs), frame_count, dtype=torch.long),
            torch.full((len(wavs),), frame_count, dtype=torch.long),
        )


@contextmanager
def _nano_audio_tokenizer_class():
    modules_before = set(sys.modules)
    if not _sglang_is_installed():
        for name in (
            "sglang",
            "sglang.kernels",
            "sglang.kernels.ops",
            "sglang.kernels.ops.attention",
        ):
            _install_stub_package(name)
        flash_attention = types.ModuleType(
            "sglang.kernels.ops.attention.flash_attention"
        )
        flash_attention.flash_attn_varlen_func = lambda *args, **kwargs: None
        sys.modules[flash_attention.__name__] = flash_attention
        flash_attention_v3 = types.ModuleType(
            "sglang.kernels.ops.attention.flash_attention_v3"
        )
        flash_attention_v3._is_fa3_supported = lambda: False
        sys.modules[flash_attention_v3.__name__] = flash_attention_v3
    try:
        module = importlib.import_module(
            "sglang_omni.models.moss_tts_nano.audio_tokenizer"
        )
        yield module.MossTTSNanoAudioTokenizer
    finally:
        for name in set(sys.modules) - modules_before:
            if name == "sglang" or name.startswith("sglang."):
                sys.modules.pop(name, None)
            elif name in {
                "sglang_omni.models.moss_tts.attention",
                "sglang_omni.models.moss_tts.audio_tokenizer",
                "sglang_omni.models.moss_tts.vocoder_kernels",
                "sglang_omni.models.moss_tts_local.audio_tokenizer",
                "sglang_omni.models.moss_tts_nano.audio_tokenizer",
            }:
                sys.modules.pop(name, None)


def test_audio_tokenizer_preserves_amplitude_without_loudness_normalization() -> None:
    model = _FakeAudioTokenizerModel()
    mono = torch.full((1, 8), 0.5)

    with _nano_audio_tokenizer_class() as tokenizer_cls:
        tokenizer = tokenizer_cls(model, device="cpu")
        encoded = tokenizer.encode_wavs([mono], 48000, num_quantizers=N_VQ)

    assert tuple(encoded[0].shape) == (8, N_VQ)
    torch.testing.assert_close(model.prepared_wavs[0], mono.repeat(2, 1))


def _payload() -> StagePayload:
    return StagePayload(
        request_id="nano-1",
        request=OmniRequest(inputs={"text": "hello"}, params={}, metadata={}),
        data={},
    )


def test_result_adapter_persists_nano_state_and_16_codebooks() -> None:
    payload = _payload()
    state = MossTTSNanoState(
        text="hello",
        generation_kwargs={"audio_temperature": 0.8},
    )
    prompt_rows = torch.full((6, N_VQ + 1), AUDIO_PAD_TOKEN_ID, dtype=torch.long)
    with _nano_request_builders_module() as request_builders:
        data = request_builders.MossTTSNanoSGLangRequestData(
            input_ids=torch.arange(6, dtype=torch.long),
            max_new_tokens=12,
            temperature=0.0,
            output_ids=[],
            state=state,
            prompt_rows=prompt_rows,
            stage_payload=payload,
            engine_start_s=time.perf_counter(),
        )
        data.output_rows = [
            torch.cat(
                [
                    torch.tensor([MODEL_CONFIG.audio_assistant_slot_token_id]),
                    torch.arange(N_VQ, dtype=torch.long) + frame,
                ]
            )
            for frame in range(3)
        ]
        result = request_builders.apply_sglang_moss_tts_nano_result(payload, data)
    restored = MossTTSNanoState.from_dict(result.data)

    assert restored.sample_rate == 48000
    assert restored.text == "hello"
    assert restored.ref_text is None
    assert restored.generation_kwargs == {"audio_temperature": 0.8}
    assert restored.prompt_tokens == 6
    assert restored.completion_tokens == 3
    assert restored.engine_time_s >= 0
    assert isinstance(restored.audio_codes, torch.Tensor)
    assert tuple(restored.audio_codes.shape) == (3, N_VQ)
    torch.testing.assert_close(
        restored.audio_codes,
        torch.stack([torch.arange(N_VQ) + frame for frame in range(3)]),
    )


def test_result_adapter_emits_empty_16_codebook_tensor() -> None:
    payload = _payload()
    with _nano_request_builders_module() as request_builders:
        data = request_builders.MossTTSNanoSGLangRequestData(
            input_ids=torch.arange(4, dtype=torch.long),
            max_new_tokens=12,
            temperature=0.0,
            output_ids=[],
            prompt_rows=torch.full(
                (4, N_VQ + 1),
                AUDIO_PAD_TOKEN_ID,
                dtype=torch.long,
            ),
            stage_payload=payload,
            engine_start_s=time.perf_counter(),
        )
        result = request_builders.apply_sglang_moss_tts_nano_result(payload, data)

    assert tuple(torch.as_tensor(result.data["audio_codes"]).shape) == (0, N_VQ)


def test_state_rejects_reference_transcript_for_voice_cloning() -> None:
    payload = StagePayload(
        request_id="nano-ref-text",
        request=OmniRequest(
            inputs={
                "text": "hello",
                "references": [{"audio_path": "reference.wav", "text": "spoken words"}],
            },
            params={},
            metadata={},
        ),
        data={},
    )

    with _nano_request_builders_module() as request_builders:
        with pytest.raises(
            ValueError,
            match="does not accept a reference transcript",
        ):
            request_builders.build_moss_tts_nano_state(payload)
