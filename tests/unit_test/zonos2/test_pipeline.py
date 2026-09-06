# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang_omni.client import Client
from sglang_omni.config import resolve_stage_factory_args
from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
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
from tests.unit_test.pipeline.helpers import build_compiled_process_topology


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


def test_zonos2_prefill_attaches_embeddings_to_sidecar() -> None:
    expected = torch.randn(4, 4)
    runner = SimpleNamespace(_build_prefill_embeds=lambda *_: expected)
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.long),
        input_embeds=None,
        replace_embeds=None,
    )

    callbacks.zonos2_prefill_forward(runner, forward_batch, None, [])

    payload = get_omni_prefill_inputs(forward_batch)
    assert payload is not None
    assert payload.input_embeds is expected
    assert forward_batch.input_embeds is None


def test_zonos2_streaming_pipeline_routes_chunks_to_vocoder() -> None:
    config = Zonos2PipelineConfig(model_path="fake-model")
    stages_by_name = {stage.name: stage for stage in config.stages}

    assert stages_by_name["tts_engine"].stream_to == ["vocoder"]
    assert stages_by_name["vocoder"].can_accept_stream_before_payload is True
    # The first-flush row count is the factory signature's default now; the
    # config leaves it unset.
    assert stages_by_name["tts_engine"].factory.model_extra in (None, {})
    assert DEFAULT_ZONOS2_PRODUCER_FIRST_FLUSH_ROWS == 58


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
    topology = build_compiled_process_topology(config)

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
    server_args = SimpleNamespace(chunked_prefill_size=8192)
    Zonos2EngineBuilder().customize_server_args(server_args)
    assert server_args.chunked_prefill_size == 0


def test_zonos2_builder_supports_breakable_prefill() -> None:
    assert Zonos2EngineBuilder.supports_breakable_prefill_cuda_graph is True


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


def _zonos2_sglang_model_module():
    return pytest.importorskip(
        "sglang_omni.models.zonos2.sglang_model",
        reason="SGLang runtime dependencies are not installed",
    )


class _PassthroughLayer(nn.Module):
    def forward(self, x, residual, router_states, positions, forward_batch):
        del residual, router_states, positions, forward_batch
        return x, torch.zeros_like(x), None


def _minimal_body(module, hidden_size: int = 4):
    body = module.Zonos2TransformerBody.__new__(module.Zonos2TransformerBody)
    nn.Module.__init__(body)
    body.emb_norm_eps = 1e-6
    body.layers = nn.ModuleList([_PassthroughLayer()])
    body.out_norm = nn.Parameter(torch.ones(hidden_size))
    return body


def _minimal_outer(module, body):
    outer = module.Zonos2SGLangModel.__new__(module.Zonos2SGLangModel)
    nn.Module.__init__(outer)
    outer.model = body
    return outer


def test_zonos2_outer_model_uses_resolved_transformer_body() -> None:
    module = _zonos2_sglang_model_module()
    resolver = pytest.importorskip(
        "sglang.srt.model_loader.utils"
    ).resolve_language_model
    body = _minimal_body(module)
    outer = _minimal_outer(module, body)
    input_ids = torch.zeros(2, dtype=torch.long)
    positions = torch.arange(2, dtype=torch.long)
    forward_batch = SimpleNamespace()
    input_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    resolved = resolver(outer)

    assert resolved is body
    hidden = resolved(input_ids, positions, forward_batch, input_embeds)
    assert torch.is_tensor(hidden)

    result = outer.forward(
        input_ids,
        positions,
        forward_batch,
        input_embeds=input_embeds,
    )

    torch.testing.assert_close(result.hidden_states, hidden)


def test_zonos2_loader_maps_checkpoint_weights_into_transformer_body() -> None:
    module = _zonos2_sglang_model_module()

    def parameter(shape):
        return nn.Parameter(torch.empty(shape))

    attention = SimpleNamespace(
        wq=parameter((4, 4)),
        wkv=parameter((2, 4, 4)),
        wo=parameter((4, 4)),
        gater=parameter((1, 4)),
        temp=parameter((1, 1, 1)),
    )
    layer = SimpleNamespace(
        attention=attention,
        attention_norm=parameter((4,)),
        ffn_norm=parameter((4,)),
        is_moe=False,
        feed_forward=SimpleNamespace(
            w_in=parameter((2, 3, 4)),
            w_out=parameter((4, 3)),
        ),
    )
    loader = SimpleNamespace(
        n_codebooks=0,
        embedders=[SimpleNamespace(weight=parameter((3, 4)))],
        model=SimpleNamespace(
            out_norm=parameter((4,)),
            layers=[layer],
        ),
        multi_output=parameter((2, 4)),
        speaker_lda_projection=SimpleNamespace(
            weight=parameter((2, 3)),
            bias=parameter((2,)),
        ),
        speaker_projection=SimpleNamespace(
            weight=parameter((4, 2)),
            bias=parameter((4,)),
        ),
    )
    checkpoint = {
        "multi_embedder.embedders.0.weight": torch.full((3, 4), 1.0),
        "out_norm.weight": torch.full((4,), 2.0),
        "multi_output.weight": torch.full((2, 4), 3.0),
        "speaker_lda_projection.weight": torch.full((2, 3), 4.0),
        "speaker_projection.weight": torch.full((4, 2), 5.0),
        "speaker_projection.bias": torch.full((4,), 6.0),
        "layers.0.attention.wq.weight": torch.full((4, 4), 7.0),
        "layers.0.attention.wkv.weight": torch.full((2, 4, 4), 8.0),
        "layers.0.attention.wo.weight": torch.full((4, 4), 9.0),
        "layers.0.attention.gater.weight": torch.full((1, 4), 10.0),
        "layers.0.attention.temp": torch.full((1, 1, 1), 11.0),
        "layers.0.attention_norm.weight": torch.full((4,), 12.0),
        "layers.0.ffn_norm.weight": torch.full((4,), 13.0),
        "layers.0.feed_forward.w_in.weight": torch.full((2, 3, 4), 14.0),
        "layers.0.feed_forward.w_out.weight": torch.full((4, 3), 15.0),
    }

    module.Zonos2SGLangModel.load_weights(loader, checkpoint.items())

    torch.testing.assert_close(loader.model.out_norm, checkpoint["out_norm.weight"])
    torch.testing.assert_close(
        loader.model.layers[0].attention.wq,
        checkpoint["layers.0.attention.wq.weight"],
    )
