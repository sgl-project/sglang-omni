# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.client import Client
from sglang_omni.config import (
    build_process_topology_plan,
    build_stage_placement_plan,
    resolve_stage_factory_args,
)
from sglang_omni.models.zonos2 import callbacks
from sglang_omni.models.zonos2 import engine_builder as eb
from sglang_omni.models.zonos2.components import text_frontend
from sglang_omni.models.zonos2.config import (
    Zonos2MultiGPUPipelineConfig,
    Zonos2PipelineConfig,
)
from sglang_omni.models.zonos2.engine_builder import Zonos2EngineBuilder
from sglang_omni.models.zonos2.request_builders import (
    build_zonos2_state,
    build_zonos2_stream_metadata,
)
from sglang_omni.models.zonos2.streaming_contract import (
    DEFAULT_ZONOS2_PRODUCER_FIRST_FLUSH_ROWS,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.streaming_vocoder import INITIAL_CODEC_CHUNK_FRAMES_PARAM
from sglang_omni.serve.speech_service import SpeechRequestValidator
from tests.unit_test.fakes import FakeServerArgs


def test_zonos2_decode_buffers_pad_async_lookahead_rows() -> None:
    feedback = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    class _Pool:
        feedback_embeds = feedback

        def release_inactive(self, request_ids: set[str]) -> None:
            assert request_ids == {"r0", "r1"}

        def prepare_active_rows(self, requests: list) -> torch.Tensor:
            assert [request.request_id for request in requests] == ["r0", "r1"]
            return torch.tensor([1, 4])

    weight = torch.full((4, 2), -1.0)
    runner = SimpleNamespace(
        model=SimpleNamespace(
            _decode_input_embedding=SimpleNamespace(weight=weight),
            _decode_state_pool=_Pool(),
        )
    )
    forward_batch = SimpleNamespace(batch_size=4, input_ids=None, input_embeds=object())
    requests = [
        SimpleNamespace(request_id="r0"),
        SimpleNamespace(request_id="r1"),
    ]

    callbacks.write_zonos2_buffers(runner, forward_batch, None, requests)

    assert torch.equal(weight[:2], feedback[torch.tensor([1, 4])])
    assert torch.count_nonzero(weight[2:]) == 0
    assert torch.equal(forward_batch.input_ids, torch.arange(4))
    assert forward_batch.input_embeds is None


def test_zonos2_streaming_pipeline_routes_chunks_to_vocoder() -> None:
    config = Zonos2PipelineConfig(model_path="fake-model")
    stages_by_name = {stage.name: stage for stage in config.stages}

    assert stages_by_name["tts_engine"].stream_to == ["vocoder"]
    assert stages_by_name["vocoder"].can_accept_stream_before_payload is True
    assert (
        stages_by_name["tts_engine"].factory_args["stream_emit_first_chunk_frames"]
        == DEFAULT_ZONOS2_PRODUCER_FIRST_FLUSH_ROWS
        == 58
    )


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"stream": True}, None),
        ({"stream": True, INITIAL_CODEC_CHUNK_FRAMES_PARAM: 0}, 0),
        ({"stream": True, INITIAL_CODEC_CHUNK_FRAMES_PARAM: 5}, 5),
    ],
)
def test_zonos2_stream_metadata_preserves_request_override_provenance(
    params: dict, expected: int | None
) -> None:
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(inputs="", params=params),
        data={},
    )

    metadata = build_zonos2_stream_metadata(payload, n_codebooks=9)

    if expected is None:
        assert INITIAL_CODEC_CHUNK_FRAMES_PARAM not in metadata
    else:
        assert metadata[INITIAL_CODEC_CHUNK_FRAMES_PARAM] == expected


def test_zonos2_multi_gpu_uses_typed_gpu_one_process() -> None:
    config = Zonos2MultiGPUPipelineConfig(model_path="fake-model")
    stages_by_name = {stage.name: stage for stage in config.stages}
    placement = build_stage_placement_plan(config)
    topology = build_process_topology_plan(config, placement)

    for stage_name in ("speaker_encode", "vocoder"):
        stage = stages_by_name[stage_name]
        assert stage.gpu == 1
        assert stage.process == "auxiliary"
        assert topology.stage_to_process[stage_name] == "auxiliary"
        args = resolve_stage_factory_args(stage, config)
        assert args["gpu_id"] == 1
        assert "device" not in args

    assert stages_by_name["preprocessing"].gpu == 0
    assert stages_by_name["tts_engine"].gpu == 0
    assert topology.stage_to_process["preprocessing"] == "pipeline"
    assert topology.stage_to_process["tts_engine"] == "pipeline"


def _speech_payload(payload: dict) -> StagePayload:
    validator = SpeechRequestValidator(default_model="Zyphra/zonos2")
    prepared = validator.parse_generation_request(payload)
    generation_request = validator.build_generate_request(
        prepared.request,
        validate=False,
        reference_descriptors=prepared.reference_descriptors,
    )
    return StagePayload(
        request_id="request",
        request=Client._build_omni_request(generation_request),
        data={},
    )


@pytest.mark.parametrize(
    ("language", "nemo_language"),
    [("english", "en"), ("chinese", "zh")],
)
def test_speech_language_reaches_prompt_normalization(
    monkeypatch, language: str, nemo_language: str
) -> None:
    calls: list[str] = []
    normalizer = text_frontend.TTSTextNormalizer()

    class _FakeNemoNormalizer:
        def __init__(self, lang: str) -> None:
            self.lang = lang

        def normalize(self, text: str, *, punct_post_process: bool) -> str:
            return f"{self.lang}:{text}"

    def _get(lang: str):
        calls.append(lang)
        return _FakeNemoNormalizer(lang)

    monkeypatch.setattr(normalizer, "get", _get)
    monkeypatch.setattr(text_frontend, "_NORMALIZER", normalizer)
    state = build_zonos2_state(
        _speech_payload({"input": f"{language} prompt", "language": language})
    )
    rows = text_frontend.build_prompt_rows(state.text, language=state.language)
    expected = text_frontend.text_to_byte_ids(f"{nemo_language}:{language} prompt")

    assert state.language == language.title()
    assert calls == [nemo_language]
    assert rows[: len(expected), -1].tolist() == expected


@pytest.mark.parametrize("language", ["auto", "russian"])
def test_auto_and_unsupported_normalization_keep_raw_prompt(
    monkeypatch, language: str
) -> None:
    class _FailingNormalizer:
        def normalize(self, text: str, language: str) -> str:
            raise AssertionError("normalizer should not be called")

    monkeypatch.setattr(text_frontend, "_NORMALIZER", _FailingNormalizer())
    text = f"{language} raw prompt"
    state = build_zonos2_state(_speech_payload({"input": text, "language": language}))
    rows = text_frontend.build_prompt_rows(state.text, language=state.language)
    expected = text_frontend.text_to_byte_ids(text)

    assert rows[: len(expected), -1].tolist() == expected


def test_speech_seed_is_rejected_until_request_rng_is_supported() -> None:
    with pytest.raises(ValueError, match="does not support seed"):
        build_zonos2_state(_speech_payload({"input": "seeded prompt", "seed": 17}))


def test_zonos2_engine_builder_disables_chunked_prefill() -> None:
    """The per-frame feedback/EOS state machine has no rollback, so the builder
    must disable chunked prefill regardless of the ServerArgs default."""
    server_args = FakeServerArgs(chunked_prefill_size=8192)
    Zonos2EngineBuilder().customize_server_args(server_args)
    assert server_args.chunked_prefill_size == 0


def test_zonos2_engine_builder_declares_model_arch_override() -> None:
    assert Zonos2EngineBuilder.model_arch_override == "Zonos2SGLangModel"


def test_zonos2_engine_builder_resolves_context_length(monkeypatch) -> None:
    monkeypatch.setattr(eb, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        eb,
        "load_zonos2_pretrained_config",
        lambda path: SimpleNamespace(max_seqlen=6144),
    )
    monkeypatch.setattr(eb, "_build_config_shim", lambda path, cfg: "/tmp/shim")

    builder = Zonos2EngineBuilder()
    assert builder.resolve_checkpoint("fake-zonos2") == "/tmp/shim"
    assert builder.context_length == 6144


def test_zonos2_engine_builder_keeps_power_of_two_cuda_graph_buckets() -> None:
    overrides = {"cuda_graph_max_bs": 16}
    Zonos2EngineBuilder(cuda_graph_max_bs=16).adjust_overrides(overrides)
    assert overrides["cuda_graph_bs"] == [1, 2, 4, 8, 16]
