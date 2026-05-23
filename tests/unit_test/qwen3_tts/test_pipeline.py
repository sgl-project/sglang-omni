# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
import threading
import types
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni.pending_text_queue import PendingTextTensorQueue
from sglang_omni.models.qwen3_tts import request_builders as qwen3_request_builders
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.qwen3_tts.request_builders import (
    Qwen3TTSPreparedRequest,
    Qwen3TTSSGLangRequestData,
    apply_sglang_qwen3_tts_result,
    build_embedding_cache_key_ids,
    build_qwen3_tts_state,
    build_sglang_qwen3_tts_request,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


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
        "sglang.srt.model_loader",
        "sglang.srt.sampling",
    ):
        modules[package_name].__path__ = []
    modules["sglang"].srt = modules["sglang.srt"]
    modules["sglang.srt"].managers = modules["sglang.srt.managers"]
    modules["sglang.srt"].layers = modules["sglang.srt.layers"]
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
    assert config.stages[1].factory.endswith("create_sglang_tts_engine_executor")
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"tts_engine": 0, "vocoder": 0}
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Qwen3TTSForConditionalGeneration")
        is Qwen3TTSPipelineConfig
    )


def test_qwen3_tts_state_round_trip_preserves_request_fields() -> None:
    state = Qwen3TTSState(
        text="hello",
        language="en",
        ref_audio="voice.wav",
        ref_text="reference",
        generation_kwargs={"max_new_tokens": 128, "temperature": 0.7},
        seed=123,
        audio_codes=[[1, 2], [3, 4]],
        ref_code_len=1,
        audio_samples=[0.0, 0.1],
        sample_rate=24000,
    )
    restored = Qwen3TTSState.from_dict(state.to_dict())
    assert restored.text == "hello"
    assert restored.language == "en"
    assert restored.ref_audio == "voice.wav"
    assert restored.ref_text == "reference"
    assert restored.generation_kwargs["max_new_tokens"] == 128
    assert restored.audio_codes == [[1, 2], [3, 4]]
    assert restored.ref_code_len == 1
    assert restored.audio_samples == [0.0, 0.1]


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
    assert state.language == "auto"
    assert state.ref_audio == "voice.wav"
    assert state.ref_text == "reference"
    assert state.x_vector_only_mode is False
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


def test_qwen3_tts_uses_x_vector_only_when_ref_text_is_missing() -> None:
    payload = make_payload(
        inputs={"text": "target", "references": [{"audio_path": "voice.wav"}]},
    )

    state = build_qwen3_tts_state(payload)

    assert state.ref_audio == "voice.wav"
    assert state.ref_text is None
    assert state.x_vector_only_mode is True


def test_qwen3_tts_rejects_missing_reference_audio() -> None:
    payload = make_payload(inputs="target")

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


def test_qwen3_tts_vocoder_batches_decode_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protects Qwen3-TTS vocoder throughput from regressing to serial decode."""
    from sglang_omni.models.qwen3_tts import stages

    decode_batch_sizes: list[int] = []

    class FakeTokenizer:
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

    scheduler = stages.create_vocoder_executor(
        "model",
        max_batch_size=2,
        max_batch_wait_ms=3,
    )
    first = make_payload(inputs="first")
    first.data = Qwen3TTSState(
        audio_codes=torch.tensor([[1, 2], [3, 4]]),
        ref_code_len=1,
    ).to_dict()
    second = make_payload(inputs="second")
    second.data = Qwen3TTSState(
        audio_codes=torch.tensor([[5, 6], [7, 8]]),
    ).to_dict()

    results = scheduler._batch_fn([first, second])

    assert scheduler._max_batch_size == 2
    assert scheduler._max_batch_wait_s == pytest.approx(0.003)
    assert decode_batch_sizes == [2]
    assert results[0].data["sample_rate"] == 24000
    assert results[0].data["audio_data"] == [3.0, 4.0, 5.0]
    assert "audio_codes" not in results[0].data
    assert results[1].data["audio_data"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


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
        state=Qwen3TTSState(seed=123),
        input_ids_list=[11, 12, 13],
        input_ids=torch.tensor([11, 12, 13], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        trailing_text_hidden=torch.randn(2, 4, dtype=dtype),
        ref_code=torch.tensor([[9, 9]], dtype=torch.long),
        prompt_input_embeds=torch.randn(3, 4, dtype=dtype),
        tts_pad_embed=torch.randn(4, dtype=dtype),
        gen_kwargs={"max_new_tokens": 16, "temperature": 0.8, "top_k": 30},
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
    assert data.ref_code is prepared.ref_code
    assert data.tts_pad_embed is prepared.tts_pad_embed
    assert isinstance(data.pending_text_queue, PendingTextTensorQueue)
    assert data.pending_text_queue.rows is not None
    assert data.pending_text_queue.rows.device == prepared.trailing_text_hidden.device
    assert data.pending_text_queue.rows.dtype == prepared.trailing_text_hidden.dtype


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
    loop = asyncio.new_event_loop()
    errors: list[BaseException] = []

    def run_compute() -> None:
        try:
            scheduler._run_single(
                IncomingMessage(
                    request_id=request_id,
                    type="new_request",
                    data=payload,
                ),
                loop,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_compute)
    try:
        thread.start()
        assert started.wait(timeout=2.0)

        scheduler.abort(request_id)
        release.set()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert errors == []
        assert scheduler.outbox.empty()
        with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
            assert request_id not in qwen3_request_builders._PREPARED_REQUESTS
    finally:
        release.set()
        thread.join(timeout=2.0)
        loop.close()
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
        scheduler._pending_stream_chunks = {}
        scheduler._pending_stream_done = set()
        scheduler._deferred_request_payloads = {}
        scheduler._first_emit_done = set()
        scheduler._prefill_start_done = set()
        scheduler.waiting_queue = []
        scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
        scheduler.cur_batch = None
        scheduler.last_batch = None
        scheduler.inbox = Queue()

        scheduler.abort(request_id)

        with qwen3_request_builders._PREPARED_REQUESTS_LOCK:
            assert request_id not in qwen3_request_builders._PREPARED_REQUESTS
    finally:
        qwen3_request_builders.cleanup_prepared_qwen3_tts_request(request_id)


def test_qwen3_tts_prefill_prepares_subtalker_buffers_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner

    calls: list[str] = []
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.model = SimpleNamespace(
        prepare_decode_buffers=lambda requests: calls.append("prepare")
    )
    runner._build_prefill_input_embeds = (
        lambda forward_batch, requests: calls.append("embeds") or object()
    )
    runner._forward_with_input_embeds = (
        lambda forward_batch, input_embeds: calls.append("forward") or "result"
    )

    assert runner.prepare_prefill(object(), object(), [object()]) == "result"
    assert calls == ["prepare", "embeds", "forward"]


def test_qwen3_tts_subtalker_sampling_reuses_request_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker.model = SimpleNamespace(
        codec_embedding=SimpleNamespace(weight=torch.empty(1))
    )
    data = SimpleNamespace(
        req=SimpleNamespace(
            sampling_params=SimpleNamespace(sampling_seed=7),
        )
    )

    Qwen3TTSTalker.prepare_decode_buffers(talker, [SimpleNamespace(data=data)])
    generator = data._subtalker_generator
    Qwen3TTSTalker.prepare_decode_buffers(talker, [SimpleNamespace(data=data)])

    assert data._subtalker_generator is generator
    assert talker._sub_generators == [generator]


def test_qwen3_tts_subtalker_sampling_advances_request_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang(monkeypatch)
    from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(11)
    before = generator.get_state().clone()
    talker._sub_dosample = [True]
    talker._sub_temperature = [1.0]
    talker._sub_top_p = [1.0]
    talker._sub_top_k = [-1]
    talker._sub_generators = [generator]

    Qwen3TTSTalker._sample_subtalker_token(talker, torch.tensor([[0.2, 0.8]]), 0)

    assert not torch.equal(generator.get_state(), before)


def test_qwen3_tts_engine_keeps_cuda_graph_disabled_after_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qwen3-TTS CUDA graph support remains disabled until a follow-up PR."""
    install_fake_sglang(monkeypatch)
    from transformers import AutoProcessor

    from sglang_omni.models.qwen3_tts import model_runner as model_runner_mod
    from sglang_omni.models.qwen3_tts import stages
    from sglang_omni.models.qwen3_tts.request_builders import (
        clear_qwen3_tts_preprocessing_context,
    )
    from sglang_omni.scheduling import bootstrap as bootstrap_mod
    from sglang_omni.scheduling import omni_scheduler as scheduler_mod
    from sglang_omni.scheduling import sglang_backend

    build_kwargs: dict = {}
    infrastructure_saw_graph_disabled: list[bool] = []
    init_graph_calls: list[bool] = []

    class FakeModel:
        def load_speech_tokenizer(self, tokenizer) -> None:
            self.speech_tokenizer = tokenizer

    class FakeSGLangRunner:
        def __init__(self, server_args) -> None:
            self.server_args = server_args
            self.model = FakeModel()

        def init_device_graphs(self) -> None:
            init_graph_calls.append(True)

    class FakeWorker:
        def __init__(self, server_args) -> None:
            self.model_runner = FakeSGLangRunner(server_args)

    class FakeQwen3TTSModel:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    qwen_tts_module = types.ModuleType("qwen_tts")
    qwen_tts_module.Qwen3TTSModel = FakeQwen3TTSModel
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_tts_module)

    monkeypatch.setattr(stages, "_register_qwen3_tts_hf_config", lambda: None)
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
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
        stages,
        "make_qwen3_tts_scheduler_adapters",
        lambda **kwargs: (lambda payload: payload, lambda data: data),
    )

    def fake_build_sglang_server_args(model_path, context_length, **kwargs):
        del model_path, context_length
        build_kwargs.update(kwargs)
        return SimpleNamespace(
            disable_cuda_graph=kwargs["disable_cuda_graph"],
            disable_overlap_schedule=kwargs["disable_overlap_schedule"],
            page_size=1,
            chunked_prefill_size=0,
            max_prefill_tokens=kwargs["max_prefill_tokens"],
            max_running_requests=kwargs["max_running_requests"],
        )

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        del gpu_id, kwargs
        infrastructure_saw_graph_disabled.append(bool(server_args.disable_cuda_graph))
        return (
            FakeWorker(server_args),
            object(),
            object(),
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

    scheduler = stages.create_sglang_tts_engine_executor("model", device="cuda:0")

    assert build_kwargs["disable_cuda_graph"] is True
    assert build_kwargs["sampling_backend"] == "pytorch"
    assert infrastructure_saw_graph_disabled == [True]
    assert init_graph_calls == []
    assert scheduler.server_args.disable_cuda_graph is True
    clear_qwen3_tts_preprocessing_context()
