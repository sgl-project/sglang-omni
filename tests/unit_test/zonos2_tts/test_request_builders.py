# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.zonos2_tts import request_builders as rb
from sglang_omni.models.zonos2_tts import text_normalization as tn
from sglang_omni.models.zonos2_tts.payload_types import Zonos2TTSState
from sglang_omni.proto import OmniRequest, StagePayload


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(
        n_codebooks=9,
        codebook_size=1024,
        eoa_id=1024,
        audio_pad_id=1025,
        text_vocab=519,
        speaker_enabled=True,
        speaker_embedding_dim=2048,
        speaking_rate_num_buckets=8,
        speaking_rate_buckets=("0-8", "8-12", "12-16", "16-20", "20-24", "24-28", "28-32", "32+"),
        quality_num_buckets=60,
        quality_features=(
            "lufs",
            "estimated_snr",
            "max_pause",
            "estimated_bandlimit_hz",
            "leading_silence_s",
            "trailing_silence_s",
        ),
        quality_buckets={
            "lufs": tuple(str(i) for i in range(12)),
            "estimated_snr": tuple(str(i) for i in range(12)),
            "max_pause": tuple(str(i) for i in range(12)),
            "estimated_bandlimit_hz": tuple(str(i) for i in range(8)),
            "leading_silence_s": tuple(str(i) for i in range(8)),
            "trailing_silence_s": tuple(str(i) for i in range(8)),
        },
        speaker_background_token_enabled=True,
        accurate_mode_token_enabled=True,
    )


def _payload(
    *,
    inputs="hello",
    params: dict | None = None,
    tts_params: dict | None = None,
    request_id: str = "req-zonos2",
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def test_conditioning_token_layout_matches_reference_offsets() -> None:
    counts = (12, 12, 12, 8, 8, 8)

    assert rb.speaking_rate_token_id(519, 8, 3, counts, 2, 1) == 451
    assert rb.quality_token_id(519, 8, counts, 5, 3, 2, 1) == 511
    assert rb.speaker_background_token_id(519, 8, counts, True, 2, 1) == 516
    assert rb.speaker_background_token_id(519, 8, counts, False, 2, 1) == 517
    assert rb.accurate_mode_token_id(519, 8, counts, 2, 1) == 518


def test_default_text_vocab_matches_zonos2_checkpoint_pad_id() -> None:
    assert Zonos2TTSState().text_vocab == 519
    assert rb.build_silence_prefix()[0, -1].item() == 519


def test_generation_defaults_ignore_openai_s2_sampling_defaults() -> None:
    gen = rb.build_generation_kwargs(
        {
            "temperature": 0.8,
            "top_p": 0.8,
            "top_k": 30,
            "min_p": 0.0,
            "repetition_penalty": 1.1,
        },
        tts_params={"explicit_generation_params": []},
    )

    assert gen["temperature"] == rb.ZONOS2_DEFAULT_TEMPERATURE
    assert gen["top_p"] == rb.ZONOS2_DEFAULT_TOP_P
    assert gen["top_k"] == rb.ZONOS2_DEFAULT_TOP_K
    assert gen["min_p"] == rb.ZONOS2_DEFAULT_MIN_P
    assert gen["repetition_penalty"] == rb.ZONOS2_DEFAULT_REPETITION_PENALTY


def test_generation_explicit_sampling_overrides_defaults() -> None:
    gen = rb.build_generation_kwargs(
        {"temperature": 0.7, "top_k": 25, "seed": 123},
        tts_params={"explicit_generation_params": ["temperature", "top_k", "seed"]},
    )

    assert gen["temperature"] == 0.7
    assert gen["top_k"] == 25
    assert gen["seed"] == 123


def test_generation_defaults_leave_max_tokens_unset_until_prompt_budget() -> None:
    gen = rb.build_generation_kwargs({}, tts_params={})

    assert "max_new_tokens" not in gen


def test_generation_budget_defaults_to_remaining_context() -> None:
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=6144))

    resolved = rb.resolve_zonos2_max_new_tokens(
        model=model,
        prompt_len=37,
        requested=None,
    )

    assert resolved == 6107


def test_duration_safety_limit_bounds_omitted_generation_budget() -> None:
    resolved = rb.apply_zonos2_duration_safety_limit(
        6107,
        text="Vernon, Signal Engineer.",
        requested=None,
    )

    assert resolved == rb.estimate_zonos2_duration_safety_frames(
        "Vernon, Signal Engineer."
    )
    assert resolved < 6107


def test_duration_safety_limit_respects_explicit_generation_budget() -> None:
    resolved = rb.apply_zonos2_duration_safety_limit(
        4096,
        text="short",
        requested=4096,
    )

    assert resolved == 4096


def test_generation_budget_respects_explicit_user_cap() -> None:
    gen = rb.build_generation_kwargs(
        {"max_new_tokens": 1024},
        tts_params={"explicit_generation_params": ["max_new_tokens"]},
    )
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=6144))

    resolved = rb.resolve_zonos2_max_new_tokens(
        model=model,
        prompt_len=37,
        requested=gen["max_new_tokens"],
    )

    assert resolved == 1024


def test_generation_budget_clamps_explicit_cap_to_context() -> None:
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048))

    resolved = rb.resolve_zonos2_max_new_tokens(
        model=model,
        prompt_len=100,
        requested=4096,
    )

    assert resolved == 1948


def test_generation_budget_rejects_non_positive_user_cap() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        rb.build_generation_kwargs(
            {"max_new_tokens": 0},
            tts_params={"explicit_generation_params": ["max_new_tokens"]},
        )


def test_preprocessing_uses_prepared_store_without_tensor_payload() -> None:
    rb.set_zonos2_preprocessing_context(model_config=_model_config())
    preprocessed = rb.preprocess_zonos2_tts_payload(_payload())

    assert preprocessed.data[rb._ZONOS2_PREPARED_MARKER] == "req-zonos2"
    assert "_prompt_rows" not in preprocessed.data
    assert "speaker_embedding" not in preprocessed.data

    prepared = rb.pop_prepared_zonos2_request(preprocessed)
    assert prepared.speaker_embedding is None
    assert prepared.speaker_token_position == -1
    assert prepared.prompt_rows[0, -1].item() == 511
    assert prepared.state.prompt_tokens == prepared.prompt_rows.shape[0]
    rb.cleanup_prepared_zonos2_request("req-zonos2")


def test_preprocessing_normalizes_language_alias_and_text(monkeypatch) -> None:
    rb.set_zonos2_preprocessing_context(model_config=_model_config())

    def fake_normalize(text: str, *, language, enabled: bool = True) -> str:
        assert text == "Dr. Smith has 2 cats."
        assert language == "en_us"
        assert enabled is True
        return "doctor smith has two cats."

    monkeypatch.setattr(rb, "normalize_zonos2_text", fake_normalize)
    preprocessed = rb.preprocess_zonos2_tts_payload(
        _payload(
            inputs="Dr. Smith has 2 cats.",
            tts_params={"language": "en"},
        )
    )
    prepared = rb.pop_prepared_zonos2_request(preprocessed)

    assert prepared.state.language == "en_us"
    assert prepared.state.text == "doctor smith has two cats."
    assert prepared.prompt_rows[2, -1].item() == ord("d") + rb.ZONOS2_LEGACY_SYMBOL_VOCAB


def test_preprocessing_can_disable_text_normalization(monkeypatch) -> None:
    rb.set_zonos2_preprocessing_context(model_config=_model_config())
    calls = []

    def fake_normalize(text: str, *, language, enabled: bool = True) -> str:
        calls.append((text, language, enabled))
        return text

    monkeypatch.setattr(rb, "normalize_zonos2_text", fake_normalize)
    preprocessed = rb.preprocess_zonos2_tts_payload(
        _payload(tts_params={"language": "zh", "text_normalization": False})
    )
    prepared = rb.pop_prepared_zonos2_request(preprocessed)

    assert prepared.state.language == "cmn"
    assert prepared.state.text_normalization is False
    assert calls == [("hello", "cmn", False)]


def test_text_normalizer_fails_open_when_optional_dependency_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        tn.Zonos2TextNormalizer,
        "_build",
        lambda self, lang: (_ for _ in ()).throw(RuntimeError("missing")),
    )
    normalizer = tn.Zonos2TextNormalizer()

    assert normalizer.normalize("Room 101.", "en_us") == "Room 101."


def test_direct_speaker_embedding_adds_slot_and_cache_namespace() -> None:
    rb.set_zonos2_preprocessing_context(model_config=_model_config())
    speaker_embedding = torch.ones(2048)
    preprocessed = rb.preprocess_zonos2_tts_payload(
        _payload(tts_params={"speaker_embedding": speaker_embedding})
    )

    prepared = rb.pop_prepared_zonos2_request(preprocessed)
    assert prepared.speaker_token_position == -1
    assert prepared.prompt_rows[0, -1].item() == 511
    rb.cleanup_prepared_zonos2_request("req-zonos2")

    preprocessed = rb.preprocess_zonos2_tts_payload(
        _payload(tts_params={"speaker_embedding": speaker_embedding})
    )
    speaker_encoded = rb.encode_zonos2_speaker_payload(preprocessed)
    prepared = rb.pop_prepared_zonos2_request(speaker_encoded)

    assert prepared.speaker_token_position == 0
    assert prepared.speaker_cache_key is not None
    assert prepared.speaker_cache_key.startswith("speaker:")
    assert prepared.prompt_rows[0, -1].item() == 519
    assert prepared.prompt_rows[1, -1].item() == 517
    assert prepared.prompt_rows[2, -1].item() == 518
    assert prepared.prompt_rows[3, -1].item() == 511


def _install_fake_sglang() -> None:
    class FakeReq:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)
            self.output_ids = []
            self.prefix_indices = []
            self.extend_input_len = len(kwargs.get("origin_input_ids") or [])

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

        def normalize(self, tokenizer) -> None:
            del tokenizer

        def verify(self, vocab_size) -> None:
            self.vocab_size = vocab_size

    modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.managers": types.ModuleType("sglang.srt.managers"),
        "sglang.srt.managers.schedule_batch": types.ModuleType(
            "sglang.srt.managers.schedule_batch"
        ),
        "sglang.srt.sampling": types.ModuleType("sglang.srt.sampling"),
        "sglang.srt.sampling.sampling_params": types.ModuleType(
            "sglang.srt.sampling.sampling_params"
        ),
    }
    for package_name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.managers",
        "sglang.srt.sampling",
    ):
        modules[package_name].__path__ = []
    modules["sglang"].srt = modules["sglang.srt"]
    modules["sglang.srt"].managers = modules["sglang.srt.managers"]
    modules["sglang.srt"].sampling = modules["sglang.srt.sampling"]
    modules["sglang.srt.managers"].schedule_batch = modules[
        "sglang.srt.managers.schedule_batch"
    ]
    modules["sglang.srt.sampling"].sampling_params = modules[
        "sglang.srt.sampling.sampling_params"
    ]
    modules["sglang.srt.managers.schedule_batch"].Req = FakeReq
    modules["sglang.srt.sampling.sampling_params"].SamplingParams = FakeSamplingParams
    sys.modules.update(modules)


def test_scheduler_request_extra_key_namespaces_speaker_cache() -> None:
    _install_fake_sglang()
    rb.set_zonos2_preprocessing_context(model_config=_model_config())
    preprocessed = rb.preprocess_zonos2_tts_payload(
        _payload(tts_params={"speaker_embedding": torch.ones(2048)})
    )
    speaker_encoded = rb.encode_zonos2_speaker_payload(preprocessed)

    request_builder, _ = rb.make_zonos2_scheduler_adapters(
        model=SimpleNamespace(n_codebooks=9)
    )
    req_data = request_builder(speaker_encoded)

    assert req_data.speaker_embedding is not None
    assert req_data.req.extra_key is not None
    assert req_data.req.extra_key.startswith("speaker:")
    assert req_data.req.origin_input_ids == req_data.input_ids.tolist()
