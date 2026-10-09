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
from sglang_omni.models.moss_tts_local.radix_hash import (
    build_rows_and_radix_token_ids,
    gpu_radix_row_hash,
)
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


class FakeTokenizer:
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


def encode(text: str) -> list[int]:
    return FakeTokenizer.encode(text)


def expected_prompt_text_ids(text: str) -> list[int]:
    return (
        [MODEL_CONFIG.im_start_token_id]
        + encode(USER_ROLE_PREFIX)
        + encode(USER_TEMPLATE_REFERENCE_PREFIX)
        + encode("None")
        + encode(USER_TEMPLATE_AFTER_REFERENCE)
        + encode(text)
        + encode(USER_TEMPLATE_SUFFIX)
        + [MODEL_CONFIG.im_end_token_id]
        + encode(ASSISTANT_TURN_PREFIX)
        + [MODEL_CONFIG.im_start_token_id]
        + encode(ASSISTANT_ROLE_PREFIX)
        + [MODEL_CONFIG.audio_start_token_id]
    )


def install_stub_package(name: str) -> None:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module


def sglang_is_installed() -> bool:
    try:
        return importlib.util.find_spec("sglang") is not None
    except (ImportError, ValueError):
        return False


@contextmanager
def nano_request_builders_module():
    modules_before = set(sys.modules)
    using_stubs = not sglang_is_installed()
    if using_stubs:
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.managers",
            "sglang.srt.sampling",
        ):
            install_stub_package(name)
        schedule_batch = types.ModuleType("sglang.srt.managers.schedule_batch")

        class FakeReq:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)
                self.output_ids = []

        schedule_batch.Req = FakeReq
        sys.modules[schedule_batch.__name__] = schedule_batch
        sampling_params = types.ModuleType("sglang.srt.sampling.sampling_params")

        class FakeSamplingParams:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

            @staticmethod
            def normalize(tokenizer) -> None:
                del tokenizer

            def verify(self, vocab_size) -> None:
                self.vocab_size = vocab_size

        sampling_params.SamplingParams = FakeSamplingParams
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
def nano_stages_module():
    modules_before = set(sys.modules)
    using_stubs = not sglang_is_installed()
    with nano_request_builders_module():
        if using_stubs:
            for name in (
                "sglang.kernels",
                "sglang.kernels.ops",
                "sglang.kernels.ops.attention",
            ):
                install_stub_package(name)
            flash_attention = types.ModuleType(
                "sglang.kernels.ops.attention.flash_attention"
            )
            flash_attention.flash_attn_varlen_func = lambda *args, **kwargs: None
            sys.modules[flash_attention.__name__] = flash_attention
            flash_attention_v3 = types.ModuleType(
                "sglang.kernels.ops.attention.flash_attention_v3"
            )
            flash_attention_v3._is_fa3_supported = (
                lambda: False
            )  # noqa: leading-underscore  # SGLang capability probe.
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
    monkeypatch.setattr(local_config, "uses_rocm_wsl_dxg", lambda: True)
    config = MossTTSNanoPipelineConfig(
        model_path="OpenMOSS-Team/MOSS-TTS-Nano",
        vocoder_cuda_graph=None,
        vocoder_cuda_graph_frames=[25, 10, 5],
        vocoder_cuda_graph_min_free_gb=1.5,
        ref_audio_cache=False,
        ref_audio_cache_max_items=17,
        ref_audio_cache_max_bytes=4096,
    )

    assert config.vocoder_cuda_graph is None
    assert config.stage_factory_kwargs("preprocessing") == {
        "ref_audio_cache": False,
        "ref_audio_cache_max_items": 17,
        "ref_audio_cache_max_bytes": 4096,
    }
    assert config.stage_factory_kwargs("vocoder") == {
        "vocoder_cuda_graph": False,
        "vocoder_cuda_graph_frames": [25, 10, 5],
        "vocoder_cuda_graph_min_free_gb": 1.5,
    }


def test_pipeline_rejects_unsafe_explicit_dxg_graph_enable(monkeypatch) -> None:
    monkeypatch.setattr(local_config, "uses_rocm_wsl_dxg", lambda: True)

    with pytest.raises(
        ValueError,
        match="MOSS-TTS-Nano vocoder CUDA graphs cannot be enabled",
    ):
        MossTTSNanoPipelineConfig(
            model_path="OpenMOSS-Team/MOSS-TTS-Nano",
            vocoder_cuda_graph=True,
        )


def test_codec_factories_default_to_official_fp32_compute() -> None:
    with nano_stages_module() as stages:
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
        tokenizer=FakeTokenizer(),
        config=MODEL_CONFIG,
        text=text,
        reference_codes=None,
    )

    expected_text_ids = expected_prompt_text_ids(text)
    assert tuple(rows.shape) == (len(expected_text_ids), N_VQ + 1)
    assert rows[:, 0].tolist() == expected_text_ids
    assert torch.all(rows[:, 1:] == AUDIO_PAD_TOKEN_ID)


def test_prompt_with_reference_places_16_codebooks_in_user_audio_rows() -> None:
    reference_codes = torch.arange(3 * N_VQ, dtype=torch.long).reshape(3, N_VQ)
    text = "clone me"
    rows = build_prompt_rows(
        tokenizer=FakeTokenizer(),
        config=MODEL_CONFIG,
        text=text,
        reference_codes=reference_codes,
    )

    prefix = (
        [MODEL_CONFIG.im_start_token_id]
        + encode(USER_ROLE_PREFIX)
        + encode(USER_TEMPLATE_REFERENCE_PREFIX)
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
    with nano_request_builders_module() as request_builders:
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
    with nano_request_builders_module() as request_builders:
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
    payload = make_payload()
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

    with nano_request_builders_module() as request_builders:
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
            lambda payload: prepared,
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
    assert (
        data.req._omni_prompt_cache_key == "moss_tts_nano:prompt:v1"
    )  # noqa: leading-underscore  # SGLang request metadata.
    assert (
        data.req._omni_prompt_only_radix is True
    )  # noqa: leading-underscore  # SGLang request metadata.


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


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA required"
                ),
            ],
        ),
    ],
)
def test_nano_fused_row_builder_preserves_codes_and_special_tokens(device):
    codes = torch.arange(3 * N_VQ, device=device).reshape(3, N_VQ)
    stop = torch.tensor([0, 1, 0], device=device)
    rows, keys = build_rows_and_radix_token_ids(
        stop,
        codes,
        MODEL_CONFIG.audio_assistant_slot_token_id,
        MODEL_CONFIG.audio_end_token_id,
        hash_space=TEXT_VOCAB_SIZE,
        hash_offset=10,
    )
    assert torch.equal(rows[:, 1:], codes)
    assert rows[:, 0].tolist() == [9, 7, 9]
    assert keys[1].item() == MODEL_CONFIG.audio_end_token_id
    assert all(10 <= value < TEXT_VOCAB_SIZE for value in keys[[0, 2]].tolist())
    expected = gpu_radix_row_hash(
        rows.cpu(),
        rows[:, 0].cpu(),
        MODEL_CONFIG.audio_end_token_id,
        hash_space=TEXT_VOCAB_SIZE,
        hash_offset=10,
    )
    assert torch.equal(keys.cpu(), expected)


@pytest.mark.parametrize("stop_choice", [0, 1])
@pytest.mark.parametrize("async_decode", [False, True])
def test_nano_runner_preserves_frame_token_domain(
    monkeypatch, stop_choice, async_decode
):
    pytest.importorskip("sglang")
    from sglang_omni.model_runner import base
    from sglang_omni.models.moss_tts_local.state_pool import MossTTSLocalDecodeStatePool
    from sglang_omni.models.moss_tts_nano.model_runner import MossTTSNanoModelRunner
    from sglang_omni.models.moss_tts_nano.request_builders import (
        MossTTSNanoSGLangRequestData,
    )

    monkeypatch.setattr(
        base,
        "current_platform",
        types.SimpleNamespace(get_device=lambda index: torch.device("cpu")),
    )
    codes = torch.arange(N_VQ).reshape(1, N_VQ)
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(**vars(MODEL_CONFIG), vocab_size=TEXT_VOCAB_SIZE),
        decode_input_embedding=types.SimpleNamespace(weight=torch.zeros(1, 4)),
        device=torch.device("cpu"),
        frame_graph_max_bs=0,
        decode_frame=lambda hidden, **kwargs: (torch.tensor([stop_choice]), codes),
        prepare_multi_modal_inputs=lambda rows: torch.ones(1, 4),
    )
    model.state_pool = MossTTSLocalDecodeStatePool(model)
    model.acquire_row = model.state_pool.acquire_row
    worker = types.SimpleNamespace(
        gpu_id=0, model_runner=types.SimpleNamespace(model=model)
    )
    runner = MossTTSNanoModelRunner(worker, None)
    runner.async_enabled = async_decode
    data = MossTTSNanoSGLangRequestData(audio_repetition_penalty=1.0, sampling_seed=7)
    request = types.SimpleNamespace(request_id="nano", data=data)
    result = types.SimpleNamespace(
        logits_output=types.SimpleNamespace(hidden_states=torch.zeros(1, 4))
    )

    rows, end_id, ids = runner.run_frame_decode(result, None, [request])

    assert end_id == MODEL_CONFIG.audio_end_token_id
    assert torch.equal(rows[:, 1:], codes)
    if stop_choice:
        assert ids.item() == end_id
    else:
        assert 10 <= ids.item() < TEXT_VOCAB_SIZE
        expected = gpu_radix_row_hash(
            rows, rows[:, 0], end_id, hash_space=TEXT_VOCAB_SIZE, hash_offset=10
        )
        assert torch.equal(ids, expected)


# Audio preparation / result state


class FakeEncodedAudio:
    def __init__(self, audio_codes: torch.Tensor, audio_codes_lengths: torch.Tensor):
        self.audio_codes = audio_codes
        self.audio_codes_lengths = audio_codes_lengths


class FakeAudioTokenizerModel:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(sampling_rate=48000, number_channels=2)
        self.prepared_wavs: list[torch.Tensor] = []

    def batch_encode(
        self,
        wavs: list[torch.Tensor],
        *,
        num_quantizers: int,
    ) -> FakeEncodedAudio:
        self.prepared_wavs = [wav.detach().clone() for wav in wavs]
        frame_count = int(wavs[0].shape[-1])
        return FakeEncodedAudio(
            torch.zeros(num_quantizers, len(wavs), frame_count, dtype=torch.long),
            torch.full((len(wavs),), frame_count, dtype=torch.long),
        )


@contextmanager
def nano_audio_tokenizer_class():
    modules_before = set(sys.modules)
    if not sglang_is_installed():
        for name in (
            "sglang",
            "sglang.kernels",
            "sglang.kernels.ops",
            "sglang.kernels.ops.attention",
        ):
            install_stub_package(name)
        flash_attention = types.ModuleType(
            "sglang.kernels.ops.attention.flash_attention"
        )
        flash_attention.flash_attn_varlen_func = lambda *args, **kwargs: None
        sys.modules[flash_attention.__name__] = flash_attention
        flash_attention_v3 = types.ModuleType(
            "sglang.kernels.ops.attention.flash_attention_v3"
        )
        flash_attention_v3._is_fa3_supported = (
            lambda: False
        )  # noqa: leading-underscore  # SGLang capability probe.
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
                "sglang_omni.models.moss_tts_nano.audio_tokenizer",
            }:
                sys.modules.pop(name, None)


def test_audio_tokenizer_preserves_amplitude_without_loudness_normalization() -> None:
    model = FakeAudioTokenizerModel()
    mono = torch.full((1, 8), 0.5)

    with nano_audio_tokenizer_class() as tokenizer_cls:
        tokenizer = tokenizer_cls(model, device="cpu")
        encoded = tokenizer.encode_wavs([mono], 48000, num_quantizers=N_VQ)

    assert tuple(encoded[0].shape) == (8, N_VQ)
    torch.testing.assert_close(model.prepared_wavs[0], mono.repeat(2, 1))


def test_audio_tokenizer_load_paths_falls_back_without_torchcodec(
    monkeypatch, tmp_path
) -> None:
    sf = pytest.importorskip("soundfile")
    samples = torch.stack(
        [
            torch.linspace(-0.5, 0.5, 16),
            torch.linspace(0.25, -0.25, 16),
        ],
        dim=1,
    ).numpy()
    path = tmp_path / "reference.wav"
    sf.write(path, samples, 48000, subtype="FLOAT")

    def missing_torchcodec(_path):
        raise ImportError("TorchCodec is required for load_with_torchcodec")

    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        types.SimpleNamespace(load=missing_torchcodec),
    )

    with nano_audio_tokenizer_class() as tokenizer_cls:
        tokenizer = tokenizer_cls(FakeAudioTokenizerModel(), device="cpu")
        loaded = tokenizer.load_paths([str(path)])

    assert len(loaded) == 1
    waveform, sample_rate = loaded[0]
    assert sample_rate == 48000
    assert tuple(waveform.shape) == (2, 16)
    torch.testing.assert_close(
        waveform,
        torch.from_numpy(samples.transpose().copy()),
    )


def make_payload() -> StagePayload:
    return StagePayload(
        request_id="nano-1",
        request=OmniRequest(inputs={"text": "hello"}, params={}, metadata={}),
        data={},
    )


def test_result_adapter_persists_nano_state_and_16_codebooks() -> None:
    payload = make_payload()
    state = MossTTSNanoState(
        text="hello",
        generation_kwargs={"audio_temperature": 0.8},
    )
    prompt_rows = torch.full((6, N_VQ + 1), AUDIO_PAD_TOKEN_ID, dtype=torch.long)
    with nano_request_builders_module() as request_builders:
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
    payload = make_payload()
    with nano_request_builders_module() as request_builders:
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

    with nano_request_builders_module() as request_builders:
        with pytest.raises(
            ValueError,
            match="does not accept a reference transcript",
        ):
            request_builders.build_moss_tts_nano_state(payload)
