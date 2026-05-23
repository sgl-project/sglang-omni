# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from sglang_omni.models.higgs_tts import stages


def test_higgs_tts_engine_enables_cuda_graph_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_sglang_server_args(checkpoint_dir, context_length, **overrides):
        server_args = SimpleNamespace(
            disable_cuda_graph=overrides["disable_cuda_graph"],
            disable_overlap_schedule=False,
        )
        captured["checkpoint_dir"] = checkpoint_dir
        captured["context_length"] = context_length
        captured["overrides"] = overrides
        captured["server_args"] = server_args
        return server_args

    def fake_create_sglang_infrastructure(server_args, gpu_id):
        captured["gpu_id"] = gpu_id
        return (
            SimpleNamespace(model_runner=SimpleNamespace(model=object())),
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
        )

    class FakeOutputProcessor:
        def __init__(self, **kwargs) -> None:
            captured["output_processor_kwargs"] = kwargs

    class FakeModelRunner:
        def __init__(self, model_worker, output_proc) -> None:
            captured["model_runner_args"] = (model_worker, output_proc)

    class FakeScheduler:
        def __init__(self, **kwargs) -> None:
            captured["scheduler_kwargs"] = kwargs

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(
        stages, "build_sglang_server_args", fake_build_sglang_server_args
    )
    monkeypatch.setattr(
        stages, "create_sglang_infrastructure", fake_create_sglang_infrastructure
    )
    monkeypatch.setattr(stages, "truncate_rope_to_bf16", lambda model: None)
    monkeypatch.setattr(stages, "SGLangOutputProcessor", FakeOutputProcessor)
    monkeypatch.setattr(stages, "HiggsTTSModelRunner", FakeModelRunner)
    monkeypatch.setattr(stages, "make_higgs_scheduler_adapters", lambda: (None, None))
    monkeypatch.setattr(stages, "HiggsScheduler", FakeScheduler)

    stages.create_sglang_tts_engine_executor("boson-sglang/higgs-audio-v3-tts-4b-base")

    assert captured["checkpoint_dir"] == "boson-sglang/higgs-audio-v3-tts-4b-base"
    assert captured["context_length"] == 4096
    assert captured["gpu_id"] == 0
    assert captured["overrides"]["disable_cuda_graph"] is False
    assert captured["overrides"]["cuda_graph_max_bs"] == 32
    assert captured["server_args"].disable_overlap_schedule is True
