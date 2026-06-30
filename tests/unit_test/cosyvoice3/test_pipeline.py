# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.cosyvoice3 import request_builders as cv3_request_builders
from sglang_omni.models.cosyvoice3.config import CosyVoice3PipelineConfig
from sglang_omni.models.cosyvoice3.payload_types import CosyVoice3State
from sglang_omni.models.cosyvoice3.request_builders import (
    COSYVOICE3_DEFAULT_TEMPERATURE,
    COSYVOICE3_DEFAULT_TOP_K,
    COSYVOICE3_DEFAULT_TOP_P,
    CV3_REPETITION_PENALTY,
    EOS_ID,
    SPEECH_TOKEN_SIZE,
    SPEECH_VOCAB_SIZE,
    STOP_TOKEN_IDS,
    CosyVoice3PreparedRequest,
    CosyVoice3SGLangRequestData,
    apply_sglang_cosyvoice3_result,
    build_embedding_cache_key_ids,
    build_sglang_cosyvoice3_request,
    cleanup_prepared_cosyvoice3_request,
    clear_cosyvoice3_preprocessing_context,
    cosyvoice3_suppress_tokens,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.sampling.seed import derive_sampling_seed


def install_fake_sglang(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a CPU-only fake ``sglang.srt.*`` tree used by the AR request builder.

    CosyVoice3's ``build_sglang_cosyvoice3_request`` only imports ``Req`` and
    ``SamplingParams``; faking just those keeps the request/result path off the
    GPU and free of a real sglang engine (mirrors ``moss_tts``/``qwen3_tts``).
    """

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
            **kwargs,
        ) -> None:
            del kwargs
            self.rid = rid
            self.origin_input_text = origin_input_text
            self.origin_input_ids = origin_input_ids
            self.sampling_params = sampling_params
            self.eos_token_ids = eos_token_ids
            self.vocab_size = vocab_size
            self.output_ids = []
            self.prefix_indices = []
            self.extend_input_len = len(origin_input_ids)

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
    for name in ("sglang", "sglang.srt", "sglang.srt.managers", "sglang.srt.sampling"):
        modules[name].__path__ = []
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
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def make_payload(
    *,
    inputs,
    params: dict | None = None,
    tts_params: dict | None = None,
    request_id: str = "req-cosyvoice3",
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


class _FakeCosyVoice3Model(torch.nn.Module):
    """Minimal stand-in for the AR LM used by ``preprocess_cosyvoice3_payload``.

    ``_build_cosyvoice3_prompt_embeds`` only touches ``speech_embedding`` (weight +
    call), ``text_embedding`` (call), ``hidden_size`` and ``parameters()`` for dtype.
    """

    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden
        self.speech_embedding = torch.nn.Embedding(SPEECH_VOCAB_SIZE, hidden)
        self.text_embedding = torch.nn.Embedding(151936, hidden)


class _RecordingFrontend:
    """Captures the ``prompt_text`` the preprocessor hands to the CosyVoice frontend."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def frontend_zero_shot(self, text, prompt_text, ref_audio, sample_rate, spk_id):
        del text, ref_audio, sample_rate, spk_id
        self.calls.append(prompt_text)
        return {
            # ``full_text`` (prompt_text ++ text) must carry <|endofprompt|> (151646).
            "text": torch.tensor([[151646, 10, 11]], dtype=torch.long),
            "text_len": torch.tensor([3]),
            "prompt_text": torch.tensor([[151646]], dtype=torch.long),
            "llm_prompt_speech_token": torch.tensor([[1, 2]], dtype=torch.long),
            "flow_prompt_speech_token": torch.tensor([[1, 2]], dtype=torch.long),
            "prompt_speech_feat": torch.zeros(1, 4, 80),
            "flow_embedding": torch.zeros(1, 192),
        }


def _weight_load_standin() -> SimpleNamespace:
    """A tiny ``self`` for ``CosyVoice3LM.load_weights`` exercising the real method.

    Constructing the real 24-layer TP model needs a process group; the fused-shard
    coverage logic only reads ``named_parameters()`` and each param's ``weight_loader``,
    so a stand-in with just the two fused params is enough to drive the real code path.
    """
    params = {
        "layers.0.self_attn.qkv_proj.weight": SimpleNamespace(
            weight_loader=lambda param, weight, shard_id: None
        ),
        "layers.0.gate_up_proj.weight": SimpleNamespace(
            weight_loader=lambda param, weight, shard_id: None
        ),
    }
    return SimpleNamespace(named_parameters=lambda: list(params.items()))


def test_cosyvoice3_config_and_registry_contracts() -> None:
    config = CosyVoice3PipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.stages[1].factory.endswith("create_sglang_tts_engine_executor")
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"preprocessing": 0, "tts_engine": 0, "vocoder": 0}
    assert "device" not in config.stages[1].factory_args
    assert "device" not in config.stages[2].factory_args
    # gpu_id is owned by placement (stage.gpu) and injected into the factory at
    # startup; setting it via factory_args is rejected by the runtime guard.
    assert "gpu_id" not in config.stages[1].factory_args
    assert "gpu_id" not in config.stages[2].factory_args
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert config.supports_uploaded_voice_references() is True
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("CosyVoice3ForCausalLM")
        is CosyVoice3PipelineConfig
    )


def test_cosyvoice3_state_round_trip_preserves_request_fields() -> None:
    prompt_speech_token = torch.tensor([1, 2, 3], dtype=torch.long)
    prompt_feat = torch.arange(160, dtype=torch.float32).reshape(1, 2, 80)
    flow_embedding = torch.ones((1, 192), dtype=torch.float32)
    state = CosyVoice3State(
        prepared=True,
        prompt_speech_token=prompt_speech_token,
        prompt_feat=prompt_feat,
        flow_embedding=flow_embedding,
        min_len=4,
        max_len=200,
        top_k=25,
        top_p=0.8,
        temperature=0.7,
        repetition_penalty=1.25,
        seed=99,
        speech_tokens=[5, 6, 7],
        prompt_tokens=11,
        completion_tokens=3,
        engine_time_s=1.5,
        finish_reason="stop",
        sample_rate=24000,
    )

    restored = CosyVoice3State.from_dict(state.to_dict())

    assert restored.prepared is True
    assert torch.equal(restored.prompt_speech_token, prompt_speech_token)
    assert torch.equal(restored.prompt_feat, prompt_feat)
    assert torch.equal(restored.flow_embedding, flow_embedding)
    assert restored.min_len == 4
    assert restored.max_len == 200
    assert restored.top_k == 25
    assert restored.top_p == 0.8
    # Per-request sampling knobs added alongside seeded/deterministic generation.
    assert restored.temperature == 0.7
    assert restored.repetition_penalty == 1.25
    assert restored.seed == 99
    assert restored.speech_tokens == [5, 6, 7]
    assert restored.prompt_tokens == 11
    assert restored.completion_tokens == 3
    assert restored.engine_time_s == 1.5
    assert restored.finish_reason == "stop"
    assert restored.sample_rate == 24000
    # ``audio_samples`` was removed from the state contract.
    assert not hasattr(restored, "audio_samples")
    assert "audio_samples" not in state.to_dict()


def test_cosyvoice3_embedding_cache_keys_are_stable_and_content_based() -> None:
    """Protects radix-cache keys for CosyVoice3 requests that prefill with embeddings."""
    embeds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    same = embeds.clone()
    different_same_length = torch.tensor([[1.0, 2.0], [3.0, 5.0]])

    assert build_embedding_cache_key_ids(embeds) == build_embedding_cache_key_ids(same)
    assert build_embedding_cache_key_ids(embeds) != build_embedding_cache_key_ids(
        different_same_length
    )


def test_cosyvoice3_speech_token_vocab_constants() -> None:
    # No suppression: CosyVoice3 samples the full speech vocab and stops on ANY
    # special token (see ``cosyvoice3_suppress_tokens`` docstring).
    assert cosyvoice3_suppress_tokens() == []
    assert STOP_TOKEN_IDS == list(range(6561, 6761))
    assert len(STOP_TOKEN_IDS) == 200
    assert EOS_ID == 6562
    assert SPEECH_TOKEN_SIZE == 6561
    assert SPEECH_VOCAB_SIZE == 6761
    assert CV3_REPETITION_PENALTY == 1.5


def test_cosyvoice3_min_new_tokens_tokenizer_shim_exposes_eos() -> None:
    shim = cv3_request_builders._min_new_tokens_tokenizer_shim()

    assert shim.additional_stop_token_ids is None
    assert shim.eos_token_id == EOS_ID
    assert shim.eos_token_id == 6562


def test_cosyvoice3_request_handoff_sets_min_new_tokens_and_stop_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    payload = make_payload(inputs="target")
    payload.data = {
        cv3_request_builders._COSYVOICE3_PREPARED_MARKER: payload.request_id
    }
    prepared = CosyVoice3PreparedRequest(
        state=CosyVoice3State(top_k=25, top_p=0.8),
        prompt_input_embeds=torch.randn(5, 8),
        min_len=4,
        max_len=120,
    )
    with cv3_request_builders._PREPARED_REQUESTS_LOCK:
        cv3_request_builders._PREPARED_REQUESTS[payload.request_id] = prepared

    try:
        data = build_sglang_cosyvoice3_request(payload, model=SimpleNamespace())
    finally:
        clear_cosyvoice3_preprocessing_context()

    sampling_params = data.req.sampling_params
    assert sampling_params.min_new_tokens == 4
    assert sampling_params.min_new_tokens > 0
    assert sampling_params.max_new_tokens == 120
    assert sampling_params.stop_token_ids == list(STOP_TOKEN_IDS)
    assert sampling_params.repetition_penalty == CV3_REPETITION_PENALTY
    assert sampling_params.top_k == 25
    assert sampling_params.top_p == 0.8
    assert sampling_params.vocab_size == SPEECH_VOCAB_SIZE

    # ``min_new_tokens`` needs a tokenizer shim (qwen3/moss set it to None): the
    # penalty reads eos/additional stop ids off it.
    assert data.req.tokenizer is not None
    assert data.req.tokenizer.eos_token_id == EOS_ID
    assert data.req.tokenizer.additional_stop_token_ids is None
    assert data.req.eos_token_ids == set(STOP_TOKEN_IDS)

    assert data.suppress_tokens == []
    assert data.input_embeds_are_projected is True
    assert data.req._input_embeds_are_projected is True
    assert data.prompt_input_embeds is prepared.prompt_input_embeds
    assert data.state is prepared.state
    assert data.stage_payload is payload
    assert data.input_ids.shape[0] == 5

    # The handoff is consumed exactly once.
    with cv3_request_builders._PREPARED_REQUESTS_LOCK:
        assert payload.request_id not in cv3_request_builders._PREPARED_REQUESTS


def test_cosyvoice3_result_filters_speech_tokens_to_real_range() -> None:
    payload = make_payload(inputs="target")
    # Real speech tokens live in 0..6560; specials/stops (sos/eos/...) are 6561..6760
    # and must be dropped from the vocoder handoff.
    output_ids = [0, 5, SPEECH_TOKEN_SIZE - 1, EOS_ID, 6561, 6760, 100]
    data = CosyVoice3SGLangRequestData(
        input_ids=torch.zeros(9, dtype=torch.long),
        output_ids=output_ids,
        state=CosyVoice3State(),
        engine_start_s=0.0,
    )

    result = apply_sglang_cosyvoice3_result(payload, data)

    assert result.data["speech_tokens"] == [0, 5, SPEECH_TOKEN_SIZE - 1, 100]
    # completion_tokens counts every generated token (incl. the stop), not just
    # the kept speech tokens.
    assert result.data["completion_tokens"] == len(output_ids)
    assert result.data["prompt_tokens"] == 9


def test_cosyvoice3_preprocessing_abort_cleans_prepared_state() -> None:
    """Aborting after preprocessing stored tensors must release the handoff."""
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    request_id = "req-cv3-prepared-abort"
    try:
        clear_cosyvoice3_preprocessing_context()
        with cv3_request_builders._PREPARED_REQUESTS_LOCK:
            cv3_request_builders._PREPARED_REQUESTS[request_id] = object()

        # ``create_preprocessing_executor`` wires this abort_callback; the heavy
        # frontend it builds needs a real checkpoint, so we exercise the same
        # SimpleScheduler abort contract directly.
        scheduler = SimpleScheduler(
            lambda payload: payload,
            abort_callback=cleanup_prepared_cosyvoice3_request,
        )
        scheduler.abort(request_id)

        with cv3_request_builders._PREPARED_REQUESTS_LOCK:
            assert request_id not in cv3_request_builders._PREPARED_REQUESTS
    finally:
        cleanup_prepared_cosyvoice3_request(request_id)
        clear_cosyvoice3_preprocessing_context()


# ---------------------------------------------------------------------------
# _resolve_cosyvoice3_gen_params — per-request generation params
# ---------------------------------------------------------------------------


def test_cosyvoice3_resolve_gen_params_defaults_when_unset() -> None:
    gen = cv3_request_builders._resolve_cosyvoice3_gen_params(make_payload(inputs="x"))

    assert gen == {
        "temperature": COSYVOICE3_DEFAULT_TEMPERATURE,
        "top_k": COSYVOICE3_DEFAULT_TOP_K,
        "top_p": COSYVOICE3_DEFAULT_TOP_P,
        "repetition_penalty": CV3_REPETITION_PENALTY,
        "max_new_tokens": None,
        "seed": None,
    }


def test_cosyvoice3_resolve_gen_params_reads_from_params() -> None:
    # Non-default values (none coincide with the service's generic sampling defaults) are
    # honored as-is.
    gen = cv3_request_builders._resolve_cosyvoice3_gen_params(
        make_payload(
            inputs="x",
            params={
                "temperature": 0.5,
                "top_k": 10,
                "top_p": 0.3,
                "repetition_penalty": 1.3,
                "max_new_tokens": 50,
                "seed": 7,
            },
        )
    )

    assert gen == {
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.3,
        "repetition_penalty": 1.3,
        "max_new_tokens": 50,
        "seed": 7,
    }


def test_cosyvoice3_resolve_gen_params_ignores_unmarked_service_defaults() -> None:
    # The speech service materializes generic defaults (temperature 0.8, top_k 30, top_p 0.8,
    # repetition_penalty 1.1) for EVERY request. Without an explicit_generation_params marker
    # these must NOT override CosyVoice3's own defaults (1.0 / 25 / 0.8 / 1.5).
    service_defaults = {
        "temperature": 0.8,
        "top_k": 30,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
    }
    gen = cv3_request_builders._resolve_cosyvoice3_gen_params(
        make_payload(inputs="x", params=dict(service_defaults))
    )
    assert gen["temperature"] == 1.0
    assert gen["top_k"] == 25
    assert gen["repetition_penalty"] == 1.5

    # But when the caller explicitly set them, they ARE honored.
    gen2 = cv3_request_builders._resolve_cosyvoice3_gen_params(
        make_payload(
            inputs="x",
            params=dict(service_defaults),
            tts_params={
                "explicit_generation_params": ["temperature", "repetition_penalty"]
            },
        )
    )
    assert gen2["temperature"] == 0.8
    assert gen2["repetition_penalty"] == 1.1


def test_cosyvoice3_resolve_gen_params_tts_params_take_precedence() -> None:
    payload = make_payload(
        inputs="x",
        params={"temperature": 0.5, "top_k": 10, "seed": 1},
        tts_params={"temperature": 0.9, "top_p": 0.2, "max_new_tokens": 33, "seed": 2},
    )

    gen = cv3_request_builders._resolve_cosyvoice3_gen_params(payload)

    assert gen["temperature"] == 0.9  # tts_params overrides params
    assert gen["seed"] == 2  # tts_params overrides params
    assert gen["top_k"] == 10  # only present in params
    assert gen["top_p"] == 0.2  # only present in tts_params
    assert gen["max_new_tokens"] == 33  # only present in tts_params
    assert gen["repetition_penalty"] == CV3_REPETITION_PENALTY  # unset -> default


def test_cosyvoice3_resolve_gen_params_tts_params_default_valued_is_honored() -> None:
    # A tts_params value that coincides with a generic service default (temperature 0.8) is
    # caller-authored on the direct channel, so it must be honored WITHOUT needing an
    # explicit_generation_params marker (unlike the same value arriving via `params`).
    gen = cv3_request_builders._resolve_cosyvoice3_gen_params(
        make_payload(inputs="x", tts_params={"temperature": 0.8, "top_k": 30})
    )
    assert gen["temperature"] == 0.8
    assert gen["top_k"] == 30


@pytest.mark.parametrize("bad_seed", [1.5, "abc", True])
def test_cosyvoice3_resolve_gen_params_rejects_non_int_seed(bad_seed) -> None:
    with pytest.raises(ValueError, match="seed"):
        cv3_request_builders._resolve_cosyvoice3_gen_params(
            make_payload(inputs="x", params={"seed": bad_seed})
        )


@pytest.mark.parametrize("bad_max", [0, -5, True, 1.5])
def test_cosyvoice3_resolve_gen_params_rejects_bad_max_new_tokens(bad_max) -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        cv3_request_builders._resolve_cosyvoice3_gen_params(
            make_payload(inputs="x", params={"max_new_tokens": bad_max})
        )


def test_cosyvoice3_prompt_embeds_handoff_requests_cpu(monkeypatch, tmp_path) -> None:
    """Record the ``.to(...)`` call on the built prompt embeds.

    A plain ``device.type == "cpu"`` check on the prepared tensor passes trivially
    on a CPU-only host (the fake model already lives on CPU), so it cannot catch a
    regression that drops ``device="cpu"`` from the handoff. Recording the call
    arguments guards the contract on any host.
    """
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"RIFF0000WAVEfmt ")
    to_calls: list[dict] = []
    built = torch.zeros(5, 4)

    class _RecordingEmbeds:
        def to(self, *args, **kwargs):
            to_calls.append(dict(kwargs))
            return built.to(*args, **kwargs)

    monkeypatch.setattr(
        cv3_request_builders,
        "_build_cosyvoice3_prompt_embeds",
        lambda *args, **kwargs: _RecordingEmbeds(),
    )
    cv3_request_builders.set_cosyvoice3_preprocessing_context(
        model=_FakeCosyVoice3Model()
    )
    try:
        payload = make_payload(
            inputs={
                "text": "hello",
                "references": [{"text": "plain", "audio_path": str(ref_audio)}],
            }
        )
        result = cv3_request_builders.preprocess_cosyvoice3_payload(
            payload, frontend=_RecordingFrontend()
        )
    finally:
        clear_cosyvoice3_preprocessing_context()

    assert any(call.get("device") == "cpu" for call in to_calls), (
        "prompt embeds must be handed off CPU-side (waiting requests would "
        f"otherwise pile up GPU memory); recorded .to() calls: {to_calls}"
    )
    del result


def test_cosyvoice3_frontend_pins_ort_provider_to_placed_device(monkeypatch) -> None:
    """The vendored frontend must pass the placed device to torch AND the ORT
    speech-tokenizer session (``{"device_id": N}``), not silently default to 0."""
    from sglang_omni.models.cosyvoice3.cosyvoice.cli import frontend as frontend_mod

    captured_providers: list = []

    class _FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            del path, sess_options
            captured_providers.append(providers)

    monkeypatch.setattr(frontend_mod.onnxruntime, "InferenceSession", _FakeSession)

    fe = frontend_mod.CosyVoiceFrontEnd(
        lambda: None,
        None,
        "campplus.onnx",
        "speech_tokenizer.onnx",
        "",
        "all",
        device="cuda:1",
    )
    assert fe.device == torch.device("cuda:1")
    # campplus always stays on CPU; the speech tokenizer pins the placed device id.
    assert captured_providers[0] == ["CPUExecutionProvider"]
    assert captured_providers[1] == [("CUDAExecutionProvider", {"device_id": 1})]

    captured_providers.clear()
    fe_cpu = frontend_mod.CosyVoiceFrontEnd(
        lambda: None,
        None,
        "campplus.onnx",
        "speech_tokenizer.onnx",
        "",
        "all",
        device="cpu",
    )
    assert fe_cpu.device == torch.device("cpu")
    assert captured_providers[1] == ["CPUExecutionProvider"]


def test_cosyvoice3_resolve_gen_params_leaves_speed_to_the_service() -> None:
    # OpenAI `speed` is applied generically by the serving layer on the encoded
    # waveform (openai_api `apply_speed`); the adapter must NOT consume it too or
    # the effective factor becomes speed**2.
    gen = cv3_request_builders._resolve_cosyvoice3_gen_params(
        make_payload(inputs="x", tts_params={"speed": 2.0})
    )
    assert "speed" not in gen


def test_cosyvoice3_state_round_trip_omits_none_seed() -> None:
    state = CosyVoice3State()
    data = state.to_dict()

    # An unset seed is not serialized; temperature/repetition_penalty always are.
    assert "seed" not in data
    assert data["temperature"] == 1.0
    assert data["repetition_penalty"] == 1.5

    restored = CosyVoice3State.from_dict(data)
    assert restored.seed is None
    assert restored.temperature == 1.0
    assert restored.repetition_penalty == 1.5


# ---------------------------------------------------------------------------
# Request handoff — generation params + seed flow into SamplingParams
# ---------------------------------------------------------------------------


def test_cosyvoice3_request_handoff_propagates_gen_params_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    payload = make_payload(inputs="target")
    payload.data = {
        cv3_request_builders._COSYVOICE3_PREPARED_MARKER: payload.request_id
    }
    prepared = CosyVoice3PreparedRequest(
        state=CosyVoice3State(
            top_k=13,
            top_p=0.5,
            temperature=0.7,
            repetition_penalty=1.2,
            seed=123,
        ),
        prompt_input_embeds=torch.randn(5, 8),
        min_len=4,
        max_len=120,
    )
    with cv3_request_builders._PREPARED_REQUESTS_LOCK:
        cv3_request_builders._PREPARED_REQUESTS[payload.request_id] = prepared

    try:
        data = build_sglang_cosyvoice3_request(payload, model=SimpleNamespace())
    finally:
        clear_cosyvoice3_preprocessing_context()

    sampling_params = data.req.sampling_params
    assert sampling_params.temperature == 0.7
    assert sampling_params.top_k == 13
    assert sampling_params.top_p == 0.5
    assert sampling_params.repetition_penalty == 1.2
    # A request seed routes to sglang's seeded sampler via a derived positive int32.
    assert sampling_params.sampling_seed is not None
    assert sampling_params.sampling_seed == derive_sampling_seed("cosyvoice3", 123)
    # The same knobs also surface on the scheduler request data.
    assert data.temperature == 0.7
    assert data.top_k == 13
    assert data.top_p == 0.5


def test_cosyvoice3_request_handoff_without_seed_leaves_sampling_seed_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    payload = make_payload(inputs="target")
    payload.data = {
        cv3_request_builders._COSYVOICE3_PREPARED_MARKER: payload.request_id
    }
    prepared = CosyVoice3PreparedRequest(
        state=CosyVoice3State(),  # seed defaults to None
        prompt_input_embeds=torch.randn(3, 8),
        min_len=2,
        max_len=60,
    )
    with cv3_request_builders._PREPARED_REQUESTS_LOCK:
        cv3_request_builders._PREPARED_REQUESTS[payload.request_id] = prepared

    try:
        data = build_sglang_cosyvoice3_request(payload, model=SimpleNamespace())
    finally:
        clear_cosyvoice3_preprocessing_context()

    # No request seed -> the base runner derives a fallback later; SamplingParams stays None.
    assert data.req.sampling_params.sampling_seed is None


# ---------------------------------------------------------------------------
# _slice_flow_hift_yaml — keep flow/hift, drop unrelated !new: blocks, rewrite prefix
# ---------------------------------------------------------------------------


def test_cosyvoice3_slice_flow_hift_yaml_keeps_flow_hift_and_rewrites_prefix() -> None:
    from sglang_omni.models.cosyvoice3.stages import _slice_flow_hift_yaml

    yaml_text = "\n".join(
        [
            "sample_rate: 24000",
            "__set_seed: !apply:random.seed [1986]",
            "llm: !new:cosyvoice.llm.llm.Qwen2LM",
            "    llm_input_size: 896",
            "    speech_token_size: 6561",
            "flow: !new:cosyvoice.flow.flow.CausalMaskedDiffWithXvec",
            "    input_size: 512",
            "    output_size: 80",
            "hift: !new:matcha.hifigan.generator.HiFTGenerator",
            "    in_channels: 80",
            "    nb_harmonics: 8",
            "hifigan: !new:cosyvoice.hifigan.hifigan.HiFiGan",
            "    generator: 1",
        ]
    )

    sliced = _slice_flow_hift_yaml(yaml_text)

    # Scalar params + the flow & hift blocks are kept.
    assert "sample_rate: 24000" in sliced
    assert "flow:" in sliced
    assert "hift:" in sliced
    assert "input_size: 512" in sliced
    assert "in_channels: 80" in sliced

    # The unrelated top-level ``llm:`` !new: block is dropped (block header + its body).
    assert "Qwen2LM" not in sliced
    assert "llm_input_size" not in sliced
    # The explicitly drop-listed ``hifigan`` block is dropped too.
    assert "HiFiGan" not in sliced

    # The module-prefix rewrite happened: bare cosyvoice./matcha. -> vendored package path.
    assert (
        "!new:sglang_omni.models.cosyvoice3.cosyvoice.flow.flow.CausalMaskedDiffWithXvec"
        in sliced
    )
    assert (
        "!new:sglang_omni.models.cosyvoice3.matcha.hifigan.generator.HiFTGenerator"
        in sliced
    )
    assert "!new:cosyvoice." not in sliced
    assert "!new:matcha." not in sliced


# ---------------------------------------------------------------------------
# load_weights — fused (qkv / gate_up) shard-completeness coverage
# ---------------------------------------------------------------------------


def test_cosyvoice3_load_weights_raises_on_missing_qkv_shard() -> None:
    from sglang_omni.models.cosyvoice3.sglang_model import CosyVoice3LM

    standin = _weight_load_standin()
    # Feed only q_proj + v_proj (no k_proj): the fused qkv_proj is left a shard short.
    weights = [
        ("llm.model.model.layers.0.self_attn.q_proj.weight", torch.zeros(1)),
        ("llm.model.model.layers.0.self_attn.v_proj.weight", torch.zeros(1)),
    ]

    with pytest.raises(RuntimeError) as exc:
        CosyVoice3LM.load_weights(standin, weights)

    message = str(exc.value)
    assert "layers.0.self_attn.qkv_proj.weight" in message
    assert "missing shards" in message
    assert "'k'" in message


def test_cosyvoice3_load_weights_raises_on_missing_gate_up_shard() -> None:
    from sglang_omni.models.cosyvoice3.sglang_model import CosyVoice3LM

    standin = _weight_load_standin()
    # Feed only gate_proj (no up_proj): the fused gate_up_proj is left a shard short.
    weights = [
        ("llm.model.model.layers.0.mlp.gate_proj.weight", torch.zeros(1)),
    ]

    with pytest.raises(RuntimeError) as exc:
        CosyVoice3LM.load_weights(standin, weights)

    message = str(exc.value)
    assert "layers.0.gate_up_proj.weight" in message
    assert "missing shards" in message
    assert "'1'" in message


def test_cosyvoice3_load_weights_accepts_complete_qkv_shards() -> None:
    from sglang_omni.models.cosyvoice3.sglang_model import CosyVoice3LM

    standin = _weight_load_standin()
    weights = [
        ("llm.model.model.layers.0.self_attn.q_proj.weight", torch.zeros(1)),
        ("llm.model.model.layers.0.self_attn.k_proj.weight", torch.zeros(1)),
        ("llm.model.model.layers.0.self_attn.v_proj.weight", torch.zeros(1)),
    ]

    loaded = CosyVoice3LM.load_weights(standin, weights)

    assert "layers.0.self_attn.qkv_proj.weight" in loaded


# ---------------------------------------------------------------------------
# endofprompt dedup — the frontend prompt marker must not be doubled
# ---------------------------------------------------------------------------


def test_cosyvoice3_preprocess_does_not_double_endofprompt(tmp_path) -> None:
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"RIFF0000WAVEfmt ")
    frontend = _RecordingFrontend()

    cv3_request_builders.set_cosyvoice3_preprocessing_context(
        model=_FakeCosyVoice3Model()
    )
    try:
        # A well-formed preformatted prompt (instruction<|endofprompt|>transcript) is used
        # verbatim and NOT re-wrapped / doubled.
        payload = make_payload(
            inputs={
                "text": "hello",
                "references": [
                    {
                        "text": "custom instruction<|endofprompt|>already done",
                        "audio_path": str(ref_audio),
                    }
                ],
            }
        )
        result = cv3_request_builders.preprocess_cosyvoice3_payload(
            payload, frontend=frontend
        )
        assert frontend.calls[-1] == "custom instruction<|endofprompt|>already done"
        assert frontend.calls[-1].count("<|endofprompt|>") == 1
        # The prepared prompt embeds are handed off CPU-side (moved to the engine
        # device only at prefill) so waiting requests cannot pile up GPU memory.
        prepared = cv3_request_builders._PREPARED_REQUESTS[result.request_id]
        assert prepared.prompt_input_embeds.device.type == "cpu"

        # A ref transcript without the marker is formatted as
        # `<instruction><|endofprompt|><transcript>` (the reference zero-shot layout).
        payload2 = make_payload(
            inputs={
                "text": "hello",
                "references": [{"text": "plain", "audio_path": str(ref_audio)}],
            }
        )
        cv3_request_builders.preprocess_cosyvoice3_payload(payload2, frontend=frontend)
        assert frontend.calls[-1] == (
            f"{cv3_request_builders.CV3_ZERO_SHOT_INSTRUCTION}<|endofprompt|>plain"
        )
        assert frontend.calls[-1].count("<|endofprompt|>") == 1
    finally:
        clear_cosyvoice3_preprocessing_context()


def test_cosyvoice3_preprocess_rejects_malformed_preformatted_prompt(tmp_path) -> None:
    # A marker with an empty side ("transcript<|endofprompt|>") would silently mis-condition
    # the clone; the preprocessor must reject it rather than pass it through.
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"RIFF0000WAVEfmt ")
    cv3_request_builders.set_cosyvoice3_preprocessing_context(
        model=_FakeCosyVoice3Model()
    )
    try:
        payload = make_payload(
            inputs={
                "text": "hello",
                "references": [
                    {"text": "trailing<|endofprompt|>", "audio_path": str(ref_audio)}
                ],
            }
        )
        with pytest.raises(ValueError, match="endofprompt"):
            cv3_request_builders.preprocess_cosyvoice3_payload(
                payload, frontend=_RecordingFrontend()
            )
    finally:
        clear_cosyvoice3_preprocessing_context()


def test_cosyvoice3_preprocess_rejects_max_new_tokens_below_min_len(tmp_path) -> None:
    # The fake frontend reports text_len=3 -> min_len=6; a smaller max_new_tokens cannot yield
    # valid audio, so the request is rejected instead of silently generating min_len tokens.
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"RIFF0000WAVEfmt ")
    cv3_request_builders.set_cosyvoice3_preprocessing_context(
        model=_FakeCosyVoice3Model()
    )
    try:
        payload = make_payload(
            inputs={
                "text": "hello",
                "references": [{"text": "plain", "audio_path": str(ref_audio)}],
            },
            params={"max_new_tokens": 2},
        )
        with pytest.raises(ValueError, match="max_new_tokens"):
            cv3_request_builders.preprocess_cosyvoice3_payload(
                payload, frontend=_RecordingFrontend()
            )
    finally:
        clear_cosyvoice3_preprocessing_context()
