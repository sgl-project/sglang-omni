# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from sglang_omni.models.zonos2 import engine_builder as eb
from sglang_omni.models.zonos2.config import Zonos2PipelineConfig
from sglang_omni.models.zonos2.engine_builder import Zonos2EngineBuilder


def test_zonos2_streaming_pipeline_routes_chunks_to_vocoder() -> None:
    config = Zonos2PipelineConfig(model_path="fake-model")
    stages_by_name = {stage.name: stage for stage in config.stages}

    assert stages_by_name["tts_engine"].stream_to == ["vocoder"]
    assert stages_by_name["vocoder"].can_accept_stream_before_payload is True


def test_zonos2_engine_builder_disables_chunked_prefill() -> None:
    """The per-frame feedback/EOS state machine has no rollback, so the builder
    must disable chunked prefill regardless of the ServerArgs default."""
    server_args = SimpleNamespace(chunked_prefill_size=8192)
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
