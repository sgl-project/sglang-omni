# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import typer

from sglang_omni.cli.serve import apply_torch_compile_cli_overrides
from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.models.fishaudio_s2_pro.request_builders import (
    S2ProSGLangRequestData,
    apply_tts_result,
    build_sglang_tts_request,
    make_tts_scheduler_adapters,
)
from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
    Reference,
    S2ProTokenizerAdapter,
)
from tests.unit_test.fixtures.fish_fakes import (
    FakeFishTokenizer,
    make_s2pro_payload,
    make_s2pro_state,
)


@pytest.fixture(autouse=True)
def fast_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )


def test_fish_config_state_and_tokenizer_prompt_contracts() -> None:
    """Preserves S2-Pro topology, state tensor round-trip, and prompt VQ layout."""
    config = S2ProPipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"tts_engine": 0, "vocoder": 0}
    assert config.supports_uploaded_voice_references() is True

    state = S2ProState(
        input_ids=torch.tensor([1, 2, 3]),
        vq_mask_tokens=torch.tensor([False, True, False]),
        vq_parts=[torch.tensor([[10, 11], [20, 21]])],
        output_codes=torch.tensor([[100, 101], [1, 2], [3, 4]]),
    )
    restored = S2ProState.from_dict(state.to_dict())
    assert restored.input_ids == [1, 2, 3]
    assert torch.equal(restored.vq_parts[0], torch.tensor([[10, 11], [20, 21]]))
    assert torch.equal(
        restored.output_codes, torch.tensor([[100, 101], [1, 2], [3, 4]])
    )

    tokenizer = FakeFishTokenizer()
    adapter = S2ProTokenizerAdapter(tokenizer)
    prompt = adapter.build_prompt(
        "target",
        references=[
            Reference(
                audio_bytes=b"",
                text="ref",
                vq_codes=torch.tensor([[0, 1], [10, 11]], dtype=torch.long),
            )
        ],
        num_codebooks=2,
        speaker="alice",
    )
    assert adapter.eos_token_ids == [99]
    assert prompt["vq_mask_tokens"].dtype == torch.bool
    assert prompt["vq_mask_tokens"].sum().item() == 2
    assert torch.equal(prompt["vq_parts"][0], torch.tensor([[0, 1], [10, 11]]))
    assert any("<|speaker:alice|>target" in text for text in tokenizer.encoded_texts)


def test_fish_tts_request_and_result_adapters_preserve_tensor_contracts() -> None:
    """Preserves TTS request tensor fields and result adapter output-code shape."""
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(
        input_ids=[10, 11, 12],
        vq_mask_tokens=[False, True, True],
        vq_parts=[[[1, 2], [3, 4]]],
        max_new_tokens=6,
        temperature=0.6,
    )

    req_data = build_sglang_tts_request(state, tokenizer, request_id="req-1")
    assert torch.equal(req_data.input_ids, torch.tensor([10, 11, 12]))
    assert req_data.vq_mask_tokens.dtype == torch.bool
    assert torch.equal(req_data.vq_parts[0], torch.tensor([[1, 2], [3, 4]]))
    assert req_data.req.eos_token_ids == {99}

    req_data.output_codes = [
        torch.tensor([[100], [1], [2]], dtype=torch.long),
        torch.tensor([[101], [3], [4]], dtype=torch.long),
    ]
    apply_tts_result(state, req_data)
    assert torch.equal(
        state.output_codes,
        torch.tensor([[100, 101], [1, 3], [2, 4]], dtype=torch.long),
    )
    assert state.prompt_tokens == 3
    assert state.completion_tokens == 2

    payload = make_s2pro_payload(request_id="req-2")
    request_builder, result_adapter = make_tts_scheduler_adapters(tokenizer=tokenizer)
    adapted = request_builder(payload)
    adapted.output_codes = [torch.tensor([[100], [1], [2]], dtype=torch.long)]
    result_payload = result_adapter(adapted)
    assert adapted.stage_payload is payload
    assert result_payload.request is payload.request
    assert result_payload.data["output_codes"] == [[100], [1], [2]]


@pytest.mark.parametrize("top_k", [0, 31])
def test_fish_tts_rejects_top_k_outside_graph_width(top_k: int) -> None:
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(top_k=top_k)

    with pytest.raises(ValueError, match="S2-Pro top_k must be -1 or between 1 and 30"):
        build_sglang_tts_request(state, tokenizer, request_id="bad-top-k")

    with pytest.raises(ValueError, match="S2-Pro top_k must be -1 or between 1 and 30"):
        S2ProSGLangRequestData(
            input_ids=torch.tensor([], dtype=torch.long),
            req=object(),
            top_k=top_k,
        )


def test_fish_tts_accepts_graph_top_k_width() -> None:
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(top_k=30)

    req_data = build_sglang_tts_request(state, tokenizer, request_id="top-k-30")

    assert req_data.top_k == 30


def test_fish_tts_accepts_default_top_k_sentinel() -> None:
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(top_k=-1)

    req_data = build_sglang_tts_request(state, tokenizer, request_id="top-k-default")

    assert req_data.top_k == -1


def _server_args_overrides(config: S2ProPipelineConfig, name: str) -> dict[str, object]:
    stage = next(stage for stage in config.stages if stage.name == name)
    return dict(stage.factory_args.get("server_args_overrides") or {})


@pytest.mark.parametrize(
    "talker_mode,talker_max_bs,expected",
    [
        ("on", None, {"enable_torch_compile": True}),
        ("off", None, {"enable_torch_compile": False}),
        ("default", 2, {"torch_compile_max_bs": 2}),
        ("on", 4, {"enable_torch_compile": True, "torch_compile_max_bs": 4}),
    ],
)
def test_s2pro_cli_talker_torch_compile_targets_tts_engine(
    talker_mode: str,
    talker_max_bs: int | None,
    expected: dict[str, object],
) -> None:
    config = S2ProPipelineConfig(model_path="model")

    apply_torch_compile_cli_overrides(
        config,
        thinker_torch_compile="default",
        talker_torch_compile=talker_mode,
        thinker_torch_compile_max_bs=None,
        talker_torch_compile_max_bs=talker_max_bs,
    )

    assert _server_args_overrides(config, "tts_engine") == expected
    assert _server_args_overrides(config, "vocoder") == {}


def test_s2pro_cli_talker_torch_compile_default_is_noop() -> None:
    config = S2ProPipelineConfig(model_path="model")

    apply_torch_compile_cli_overrides(
        config,
        thinker_torch_compile="default",
        talker_torch_compile="default",
        thinker_torch_compile_max_bs=None,
        talker_torch_compile_max_bs=None,
    )

    assert _server_args_overrides(config, "tts_engine") == {}


def test_s2pro_cli_talker_torch_compile_max_bs_rejects_non_positive() -> None:
    config = S2ProPipelineConfig(model_path="model")

    with pytest.raises(
        typer.BadParameter,
        match="torch compile max batch size must be >= 1",
    ):
        apply_torch_compile_cli_overrides(
            config,
            thinker_torch_compile="default",
            talker_torch_compile="default",
            thinker_torch_compile_max_bs=None,
            talker_torch_compile_max_bs=0,
        )


def test_s2pro_compile_helper_targets_forward_kvcached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", "/tmp")
    stages = importlib.import_module("sglang_omni.models.fishaudio_s2_pro.stages")

    fake_runner = ModuleType("sglang.srt.model_executor.cuda_graph_runner")
    fake_runner.set_torch_compile_config = lambda: None
    monkeypatch.setitem(
        sys.modules, "sglang.srt.model_executor.cuda_graph_runner", fake_runner
    )

    compile_calls: list[tuple[object, str | None, dict[str, object]]] = []

    def fake_compile(
        target: object, *, mode: str | None = None, **kwargs: object
    ) -> object:
        compile_calls.append((target, mode, kwargs))
        return f"compiled-{len(compile_calls)}"

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setenv("SGLANG_TORCH_COMPILE_MODE", "reduce-overhead")

    class _Layer:
        def forward_kvcached(
            self, x: torch.Tensor, freqs_cis: torch.Tensor, cache_seqlens: torch.Tensor
        ) -> torch.Tensor:
            del freqs_cis, cache_seqlens
            return x

    class _AudioDecoder:
        def __init__(self) -> None:
            self.layers = [_Layer()]

        def set_compiled_forward_kvcached_layers(
            self,
            forward_kvcached_layers: list[object],
            *,
            max_batch_size: int,
        ) -> None:
            self._compiled_forward_kvcached_layers = forward_kvcached_layers
            self._compiled_forward_kvcached_max_bs = max_batch_size

    audio_decoder = _AudioDecoder()
    model = SimpleNamespace(_audio_decoder=audio_decoder)

    stages._compile_s2pro_codebook_decoder(model, max_batch_size=2)

    assert len(compile_calls) == 1
    target, mode, kwargs = compile_calls[0]
    assert getattr(target, "__self__", None) is audio_decoder.layers[0]
    assert getattr(target, "__name__", "") == "forward_kvcached"
    assert mode == "reduce-overhead"
    assert kwargs == {}
    assert audio_decoder._compiled_forward_kvcached_layers == ["compiled-1"]
    assert audio_decoder._compiled_forward_kvcached_max_bs == 2


def _run_s2pro_engine_with_fake_buffers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    text_buffer_bs: int = 64,
    audio_buffer_bs: int = 64,
) -> SimpleNamespace:
    stages = importlib.import_module("sglang_omni.models.fishaudio_s2_pro.stages")
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)

    build_kwargs: dict[str, object] = {}
    infrastructure_saw_graph_disabled: list[bool] = []
    compile_calls: list[tuple[object, int]] = []
    init_graph_calls: list[bool] = []

    class _FakeSGLangRunner:
        def __init__(self, server_args: SimpleNamespace) -> None:
            self.server_args = server_args
            self.model = SimpleNamespace()

        def init_device_graphs(self) -> None:
            assert self.server_args.enable_torch_compile is False
            assert self.server_args.torch_compile_max_bs == 64
            init_graph_calls.append(True)

    class _FakeWorker:
        def __init__(self, server_args: SimpleNamespace) -> None:
            self.model_runner = _FakeSGLangRunner(server_args)

    fake_bootstrap = ModuleType("sglang_omni.models.fishaudio_s2_pro.bootstrap")
    fake_bootstrap.patch_fish_config_for_sglang = lambda: None
    fake_bootstrap.truncate_rope_to_bf16 = lambda model: None
    fake_bootstrap.load_audio_decoder = lambda checkpoint_dir, device: (
        SimpleNamespace(kv_cache_max_batch_size=-1),
        10,
        4096,
        FakeFishTokenizer(),
    )

    def fake_bootstrap_text_model_for_decode(**kwargs: object) -> None:
        text_model = kwargs["text_model"]
        audio_decoder = kwargs["audio_decoder"]
        text_model.vq_decode_max_batch_size = text_buffer_bs
        text_model._audio_decoder = audio_decoder
        audio_decoder.kv_cache_max_batch_size = audio_buffer_bs

    fake_bootstrap.bootstrap_text_model_for_decode = (
        fake_bootstrap_text_model_for_decode
    )

    def fake_build_sglang_server_args(
        model_path: str,
        context_length: int,
        **kwargs: object,
    ) -> SimpleNamespace:
        del model_path, context_length
        build_kwargs.update(kwargs)
        return SimpleNamespace(
            disable_cuda_graph=kwargs["disable_cuda_graph"],
            enable_torch_compile=kwargs["enable_torch_compile"],
            torch_compile_max_bs=kwargs["torch_compile_max_bs"],
            max_running_requests=kwargs["max_running_requests"],
            cuda_graph_max_bs=kwargs["cuda_graph_max_bs"],
            cuda_graph_bs=kwargs["cuda_graph_bs"],
            page_size=1,
            chunked_prefill_size=kwargs["chunked_prefill_size"],
            max_prefill_tokens=16384,
            attention_backend=None,
        )

    def fake_create_sglang_infrastructure(
        server_args: SimpleNamespace,
        gpu_id: int,
    ) -> tuple[object, object, object, object, object, object, object]:
        assert gpu_id == 0
        infrastructure_saw_graph_disabled.append(bool(server_args.disable_cuda_graph))
        return (
            _FakeWorker(server_args),
            object(),
            object(),
            object(),
            object(),
            object(),
            SimpleNamespace(),
        )

    fake_scheduler_bootstrap = ModuleType("sglang_omni.scheduling.bootstrap")
    fake_scheduler_bootstrap.create_sglang_infrastructure = (
        fake_create_sglang_infrastructure
    )

    fake_sglang_backend = ModuleType("sglang_omni.scheduling.sglang_backend")
    fake_sglang_backend.build_sglang_server_args = fake_build_sglang_server_args
    fake_sglang_backend.SGLangOutputProcessor = lambda **kwargs: SimpleNamespace(
        **kwargs
    )

    fake_fish_scheduler = ModuleType(
        "sglang_omni.models.fishaudio_s2_pro.fish_scheduler"
    )

    class _FakeFishScheduler:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    fake_fish_scheduler.FishScheduler = _FakeFishScheduler

    fake_model_runner = ModuleType("sglang_omni.models.fishaudio_s2_pro.model_runner")
    fake_model_runner.FishS2ProModelRunner = lambda *args, **kwargs: SimpleNamespace(
        args=args, kwargs=kwargs
    )

    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.fishaudio_s2_pro.bootstrap",
        fake_bootstrap,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.scheduling.bootstrap",
        fake_scheduler_bootstrap,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.scheduling.sglang_backend",
        fake_sglang_backend,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.fishaudio_s2_pro.fish_scheduler",
        fake_fish_scheduler,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.fishaudio_s2_pro.model_runner",
        fake_model_runner,
    )

    def fake_compile(model: object, *, max_batch_size: int) -> None:
        compile_calls.append((model, max_batch_size))

    monkeypatch.setattr(stages, "_compile_s2pro_codebook_decoder", fake_compile)

    scheduler = stages.create_sglang_tts_engine_executor("model", device="cuda:0")
    return SimpleNamespace(
        scheduler=scheduler,
        build_kwargs=build_kwargs,
        infrastructure_saw_graph_disabled=infrastructure_saw_graph_disabled,
        compile_calls=compile_calls,
        init_graph_calls=init_graph_calls,
    )


def test_s2pro_engine_disables_generic_compile_after_local_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_s2pro_engine_with_fake_buffers(monkeypatch)
    scheduler = result.scheduler
    build_kwargs = result.build_kwargs

    assert build_kwargs["enable_torch_compile"] is True
    assert build_kwargs["max_running_requests"] == 64
    assert build_kwargs["cuda_graph_max_bs"] == 64
    assert build_kwargs["cuda_graph_bs"] == [
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
    assert build_kwargs["torch_compile_max_bs"] == 64
    assert result.infrastructure_saw_graph_disabled == [True]
    assert result.compile_calls == [
        (scheduler.model_runner.args[0].model_runner.model, 64)
    ]
    assert result.init_graph_calls == [True]
    assert scheduler.server_args.disable_cuda_graph is False
    assert scheduler.server_args.enable_torch_compile is False
    assert scheduler.server_args.cuda_graph_max_bs == 64
    assert scheduler.server_args.cuda_graph_bs == [
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
    assert scheduler.server_args.torch_compile_max_bs == 64


@pytest.mark.parametrize(
    "text_buffer_bs,audio_buffer_bs",
    [
        (32, 64),
        (64, 32),
    ],
)
def test_s2pro_engine_validates_allocated_decode_buffers(
    monkeypatch: pytest.MonkeyPatch,
    text_buffer_bs: int,
    audio_buffer_bs: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="model_buffer_bs must cover max_running_requests",
    ):
        _run_s2pro_engine_with_fake_buffers(
            monkeypatch,
            text_buffer_bs=text_buffer_bs,
            audio_buffer_bs=audio_buffer_bs,
        )


def test_decoder_forward_kvcached_obeys_compiled_batch_size_cap() -> None:
    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling import (
        FishQwen3AudioDecoder,
    )

    class _EagerLayer:
        def forward_kvcached(
            self, x: torch.Tensor, freqs_cis: torch.Tensor, cache_seqlens: torch.Tensor
        ) -> torch.Tensor:
            del freqs_cis, cache_seqlens
            seen_calls.append("eager")
            return x + 10

    decoder = object.__new__(FishQwen3AudioDecoder)
    decoder.input_pos = torch.zeros(1, dtype=torch.long)
    decoder.freqs_cis = torch.zeros(8, 1, 1, dtype=torch.float32)
    decoder.layers = [_EagerLayer()]
    decoder._eager_forward_kvcached_layers = [
        layer.forward_kvcached for layer in decoder.layers
    ]
    decoder.norm = lambda x: x
    decoder.output = lambda x: x

    seen_calls: list[str] = []

    def compiled(
        x: torch.Tensor, freqs_cis: torch.Tensor, cache_seqlens: torch.Tensor
    ) -> torch.Tensor:
        del freqs_cis, cache_seqlens
        seen_calls.append("compiled")
        return x + 1

    decoder._compiled_forward_kvcached_layers = [compiled]
    decoder._compiled_forward_kvcached_max_bs = 2

    x = torch.zeros((2, 1, 4), dtype=torch.float32)
    out = FishQwen3AudioDecoder.forward_kvcached(decoder, x=x, codebook_idx=2)

    assert torch.equal(out, torch.ones_like(x))
    assert seen_calls == ["compiled"]

    seen_calls.clear()
    x = torch.zeros((3, 1, 4), dtype=torch.float32)
    out = FishQwen3AudioDecoder.forward_kvcached(decoder, x=x, codebook_idx=2)

    assert torch.equal(out, torch.full_like(x, 10.0))
    assert seen_calls == ["eager"]
