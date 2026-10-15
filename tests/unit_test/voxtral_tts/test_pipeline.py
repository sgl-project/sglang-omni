# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import collections
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.voxtral_tts.acoustic_transformer import (
    FlowMatchingAudioTransformer,
)
from sglang_omni.models.voxtral_tts.audio_tokenizer import VoxtralTTSAudioTokenizer
from sglang_omni.models.voxtral_tts.config import VoxtralTTSPipelineConfig
from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.models.voxtral_tts.pipeline import stages
from sglang_omni.models.voxtral_tts.request_builders import build_sglang_voxtral_request
from sglang_omni.models.voxtral_tts.sglang_model import VoxtralSGLangTTSModel
from sglang_omni.models.voxtral_tts.vocoder import (
    VoxtralStreamingVocoder,
    VoxtralStreamWindow,
    VoxtralTTSVocoder,
    ensure_non_empty_audio_codes,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import RequestOutput
from sglang_omni.utils.audio_payload import audio_waveform_payload
from tests.unit_test.fakes import FakeExecutionBridge
from tests.unit_test.pipeline.helpers import build_compiled_process_topology


def test_voxtral_tts_config_uses_current_stage_schema() -> None:
    config = VoxtralTTSPipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_generation",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"tts_generation": 0, "vocoder": 0}
    assert config.stages[1].factory.device is None
    assert config.stages[2].factory.device is None
    assert [stage.process for stage in config.stages] == [
        "pipeline",
        "pipeline",
        "pipeline",
    ]
    assert [
        stage.gpu_memory_fraction for stage in config.stages if stage.gpu is not None
    ] == [None, None]
    build_compiled_process_topology(config)
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("VoxtralTTSForConditionalGeneration")
        is VoxtralTTSPipelineConfig
    )


def test_voxtral_stage_factories_preserve_generation_placement_and_resolve_vocoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.voxtral_tts.pipeline import engine_builder

    captured: dict[str, object] = {}

    class FakeBuilder:
        def build(self, model_path, **kwargs):
            captured["build"] = (model_path, kwargs)
            return "generation"

    monkeypatch.setattr(engine_builder, "VoxtralTtsEngineBuilder", FakeBuilder)
    generation = stages.create_generation_executor("model", gpu_id=3)

    assert generation == "generation"
    assert captured["build"] == (
        "model",
        {
            "device": None,
            "gpu_id": 3,
            "server_args_overrides": None,
        },
    )

    stages.create_generation_executor("model", device="cuda:2", gpu_id=5)
    assert captured["build"] == (
        "model",
        {
            "device": "cuda:2",
            "gpu_id": 5,
            "server_args_overrides": None,
        },
    )

    seen_devices: list[str] = []
    resolved_devices: list[tuple[str | None, int | None]] = []
    monkeypatch.setattr(
        stages,
        "resolve_concrete_device",
        lambda device, gpu_id: (
            resolved_devices.append((device, gpu_id)) or f"npu:{gpu_id}"
        ),
    )
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(
        stages,
        "load_audio_tokenizer",
        lambda _checkpoint, config, device: (
            seen_devices.append(device) or SimpleNamespace(sampling_rate=24000)
        ),
    )

    stages.create_vocoder_executor("model", gpu_id=4)
    assert resolved_devices == [(None, 4)]
    assert seen_devices == ["npu:4"]


def test_voxtral_radix_cache_is_namespaced_by_voice() -> None:
    """Different voice embeddings must not share a placeholder-token cache prefix."""
    model = SimpleNamespace(
        audio_token_id=24,
        voxtral_config=SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=32000),
        ),
    )
    voice_embeddings = {
        "cheerful_female": torch.ones(4, 8),
        "neutral_female": torch.ones(4, 8),
    }

    def make_payload(request_id: str, voice: str) -> StagePayload:
        state = VoxtralTTSState(
            input_ids=[1, 25, 24, 24, 24, 36, 100, 25],
            voice=voice,
        )
        return StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="", params={}),
            data=state.to_dict(),
        )

    cheerful = build_sglang_voxtral_request(
        make_payload("r1", "cheerful_female"),
        model=model,
        voice_embeddings=voice_embeddings,
    )
    neutral = build_sglang_voxtral_request(
        make_payload("r2", "neutral_female"),
        model=model,
        voice_embeddings=voice_embeddings,
    )

    assert cheerful.req.origin_input_ids == neutral.req.origin_input_ids
    assert cheerful.req.extra_key != neutral.req.extra_key
    assert cheerful.req.extra_key.startswith("voxtral_voice:")
    assert cheerful.voice_embedding is voice_embeddings["cheerful_female"]


def test_voxtral_speech_validation_accepts_supported_fields() -> None:
    stages.validate_voxtral_speech_params(
        inputs="hello",
        params={"max_new_tokens": 128, "stream": True},
        tts_params={
            "voice": "cheerful_female",
            "response_format": "wav",
            "speed": 1.0,
        },
    )


@pytest.mark.parametrize(
    ("params", "tts_params", "inputs", "field"),
    [
        ({"temperature": 0.2}, {}, "hello", "temperature"),
        ({"seed": 7}, {"seed": 7}, "hello", "seed"),
        ({}, {"language": "en"}, "hello", "language"),
        ({}, {"ref_audio": "ref.wav"}, "hello", "ref_audio"),
        (
            {},
            {},
            {"text": "hello", "references": [{"audio_path": "ref.wav"}]},
            "references",
        ),
        ({"stage_params": {"tts_generation": {"x": 1}}}, {}, "hello", "stage_params"),
    ],
)
def test_voxtral_speech_validation_rejects_ignored_fields(
    params: dict,
    tts_params: dict,
    inputs,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        stages.validate_voxtral_speech_params(
            inputs=inputs,
            params=params,
            tts_params=tts_params,
        )


@pytest.mark.parametrize("audio_codes", [None, torch.empty((0, 0), dtype=torch.long)])
def test_voxtral_vocoder_rejects_empty_audio_codes(audio_codes) -> None:
    with pytest.raises(ValueError, match="generated no audio codes"):
        ensure_non_empty_audio_codes(audio_codes)


def test_voxtral_audio_waveform_payload_is_compact() -> None:
    payload = audio_waveform_payload(
        torch.tensor([0.0, 0.5, -0.5]),
        source_hint="Voxtral TTS",
    )

    audio = np.frombuffer(payload["audio_waveform"], dtype=np.float32)
    assert audio.tolist() == [0.0, 0.5, -0.5]
    assert payload["audio_waveform_shape"] == [3]
    assert payload["audio_waveform_dtype"] == "float32"


def test_voxtral_audio_codes_payload_is_compact() -> None:
    state = VoxtralTTSState(audio_codes=torch.tensor([[1, 2], [3, 4]]))

    data = state.to_dict()
    restored = VoxtralTTSState.from_dict(data)

    assert "audio_codes_bytes" in data
    assert "audio_codes" not in data
    assert restored.audio_codes.tolist() == [[1, 2], [3, 4]]


def test_voxtral_audio_attention_accepts_non_contiguous_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.voxtral_tts import audio_tokenizer

    monkeypatch.setattr(audio_tokenizer, "HAS_FLASH_ATTN", False)
    attention = audio_tokenizer.Attention(
        SimpleNamespace(
            n_heads=2,
            n_kv_heads=2,
            attn_sliding_window_size=8,
            dim=4,
            head_dim=2,
            use_biases=False,
            qk_norm=False,
            causal=True,
        ),
        layer_id=0,
    )

    output = attention(torch.randn(3, 4))

    assert output.shape == (3, 4)


def test_voxtral_vocoder_preserves_warmup_trim_and_fade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_codes: list[torch.Tensor] = []

    class FakeAudioTokenizer:
        sampling_rate = 1000
        downsample_factor = 3

        def decode_helper_batch_async(self, codes_list):
            seen_codes.extend(codes.clone() for codes in codes_list)
            return [torch.arange(20, dtype=torch.float32)]

    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(
        stages,
        "load_audio_tokenizer",
        lambda *args, **kwargs: FakeAudioTokenizer(),
    )

    scheduler = stages.create_vocoder_executor("model", device="cpu")
    payload = StagePayload(
        request_id="voxtral-vocoder",
        request=OmniRequest(inputs="hello", params={}),
        data=VoxtralTTSState(
            audio_codes=torch.tensor([[9, 10], [11, 12]], dtype=torch.long),
            prompt_tokens=2,
            completion_tokens=4,
        ).to_dict(),
    )

    result = asyncio.run(scheduler.compute_fn(payload))

    assert scheduler.max_batch_size == 1
    assert scheduler.max_batch_wait_s == 0
    assert [codes.tolist() for codes in seen_codes] == [
        [[9, 10], [9, 10], [9, 10], [11, 12]]
    ]
    audio = np.frombuffer(result.data["audio_waveform"], dtype=np.float32)
    expected = torch.arange(6, 20, dtype=torch.float32)
    expected[:10] *= torch.linspace(0, 1, 10)
    assert audio.tolist() == pytest.approx(expected.tolist())
    assert result.data["sample_rate"] == 1000
    assert result.data["modality"] == "audio"
    assert result.data["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 4,
        "total_tokens": 6,
    }


class CausalFakeAudioTokenizer:
    """Each frame decodes to 3 samples depending on itself and the previous frame."""

    sampling_rate = 1000
    downsample_factor = 3
    num_codebooks = 2

    def decode_helper_batch_async(self, codes_list):
        outputs = []
        for codes in codes_list:
            values = codes[:, 0].to(torch.float32)
            previous = torch.cat([values[:1], values[:-1]])
            outputs.append((values * 10 + previous).repeat_interleave(3))
        return outputs


def test_voxtral_streamed_chunks_match_full_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(
        stages,
        "load_audio_tokenizer",
        lambda *args, **kwargs: CausalFakeAudioTokenizer(),
    )
    scheduler = stages.create_vocoder_executor("model", device="cpu")
    scheduler.initial_chunk_frames = 2
    scheduler.chunk_frames = 3
    scheduler.left_context_frames = 2
    codes = torch.stack([torch.arange(3, 13), torch.zeros(10, dtype=torch.long)], 1)
    payload = StagePayload(
        request_id="voxtral-stream",
        request=OmniRequest(inputs="hello", params={}),
        data=VoxtralTTSState(audio_codes=codes).to_dict(),
    )
    full = np.frombuffer(
        asyncio.run(scheduler.compute_fn(payload)).data["audio_waveform"],
        dtype=np.float32,
    )

    state = scheduler.create_stream_state("voxtral-stream")
    end_audio = torch.tensor([[1, 0]], dtype=torch.long)
    chunks = []
    for rows in (codes[:1], codes[1:4], codes[4:9], torch.cat([codes[9:], end_audio])):
        scheduler.ingest("voxtral-stream", state, rows)
        chunk = scheduler.decode_delta("voxtral-stream", state, is_final=False)
        if chunk is not None:
            chunks.append(chunk)
        else:
            pass
    scheduler.ingest("voxtral-stream", state, codes[:1])
    chunks.append(scheduler.decode_delta("voxtral-stream", state, is_final=True))

    assert [len(chunk) for chunk in chunks] == [12, 15, 3]
    assert torch.cat(chunks).tolist() == pytest.approx(full.tolist())
    scheduler.chunk_frames = 2
    whole = scheduler.decode_utterance(list(codes.unbind(0)))
    assert whole.tolist() == pytest.approx(full.tolist())
    scheduler.chunk_frames = 3

    for request_id, rows in (("a", codes[:6]), ("b", codes[:4]), ("c", codes[:1])):
        scheduler.ingest(
            request_id, scheduler.get_or_create_stream_state(request_id), rows
        )
    participants = scheduler.select_step_participants()
    assert [request_id for request_id, _ in participants] == ["a", "b"]
    coalesced = scheduler.run_step(
        participants, scheduler.build_step_plan(participants)
    )
    assert coalesced["a"].tolist() == pytest.approx(full[:18].tolist())
    assert coalesced["b"].tolist() == pytest.approx(full[:12].tolist())


def test_voxtral_stream_backlog_decodes_in_bounded_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(
        stages,
        "load_audio_tokenizer",
        lambda *args, **kwargs: CausalFakeAudioTokenizer(),
    )
    scheduler = stages.create_vocoder_executor("model", device="cpu")
    scheduler.initial_chunk_frames = 2
    scheduler.chunk_frames = 3
    scheduler.left_context_frames = 2
    codes = torch.stack([torch.arange(3, 23), torch.zeros(20, dtype=torch.long)], 1)
    full = scheduler.decode_utterance(list(codes.unbind(0)))
    window_frames = []
    decode_windows = scheduler.decode_windows

    def record_windows(windows):
        window_frames.extend(window.codes.shape[0] for window in windows)
        return decode_windows(windows)

    scheduler.decode_windows = record_windows
    state = scheduler.get_or_create_stream_state("backlog")
    scheduler.ingest("backlog", state, codes[:10])
    chunks = []
    while participants := scheduler.select_step_participants():
        plan = scheduler.build_step_plan(participants)
        chunks.append(scheduler.run_step(participants, plan)["backlog"])
    scheduler.ingest("backlog", state, codes[10:])
    chunks.append(scheduler.decode_delta("backlog", state, is_final=True))

    assert window_frames == [8, 6, 8, 6]
    assert len(chunks) == 3
    assert torch.cat(chunks).tolist() == pytest.approx(full.tolist())


def test_voxtral_stream_output_batches_rows_and_flushes_tail() -> None:
    from sglang_omni.models.voxtral_tts.request_builders import VoxtralStreamOutput

    builder = VoxtralStreamOutput()
    data = SimpleNamespace(stream_pending_rows=[torch.tensor([i, 0]) for i in range(4)])
    assert list(builder("req", data)) == []
    data.stream_pending_rows.append(torch.tensor([4, 0]))
    (message,) = builder("req", data)
    assert message.target == "vocoder"
    assert message.metadata == {"modality": "audio_codes", "stream": True}
    assert message.data[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert data.stream_pending_rows == []
    data.stream_pending_rows.extend([torch.tensor([5, 0]), torch.tensor([6, 0])])
    (tail,) = builder.flush("req", data)
    assert tail.data[:, 0].tolist() == [5, 6]
    assert list(builder.flush("req", data)) == []


def test_voxtral_collect_audio_step_reuses_output_tokens_for_eos_filter() -> None:
    from sglang_omni.models.voxtral_tts.acoustic_transformer import AudioSpecialTokens
    from sglang_omni.models.voxtral_tts.model_runner import VoxtralTTSModelRunner

    eos_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
    runner = VoxtralTTSModelRunner.__new__(VoxtralTTSModelRunner)
    runner.pending_audio_codes = None
    runner.pending_audio_embeds = None
    runner.model = SimpleNamespace(
        run_acoustic_transformer=lambda hidden: torch.tensor(
            [[11, 12, 13], [eos_id, 21, 22]], dtype=torch.long
        ),
        audio_token_embedding=lambda codes: codes.to(torch.float32).unsqueeze(-1),
    )
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.ones((2, 4))),
        next_token_ids=None,
    )
    schedule_batch = SimpleNamespace(output_ids=None)
    requests = [
        SimpleNamespace(
            request_id="active",
            data=SimpleNamespace(
                output_codes=[],
                pending_feedback_queue=[],
                stream=True,
                stream_pending_rows=[],
            ),
        ),
        SimpleNamespace(
            request_id="eos",
            data=SimpleNamespace(
                output_codes=[],
                pending_feedback_queue=[],
                stream=False,
                stream_pending_rows=[],
            ),
        ),
    ]

    runner.collect_audio_step(result, schedule_batch, requests)

    assert result.next_token_ids.tolist() == [11, eos_id]
    assert requests[0].data.output_codes == []
    assert requests[1].data.output_codes == []

    runner.post_process_outputs(
        result,
        SimpleNamespace(requests=requests),
        {
            "active": RequestOutput("active", data=11),
            "eos": RequestOutput("eos", data=eos_id),
        },
    )

    assert [chunk.tolist() for chunk in requests[0].data.output_codes] == [[11, 12, 13]]
    assert [row.tolist() for row in requests[0].data.stream_pending_rows] == [
        [11, 12, 13]
    ]
    assert requests[1].data.stream_pending_rows == []
    assert len(requests[0].data.pending_feedback_queue) == 1
    assert requests[1].data.output_codes == []
    assert requests[1].data.pending_feedback_queue == []

    runner.post_process_outputs(result, SimpleNamespace(requests=requests), {})
    assert len(requests[0].data.output_codes) == 1


def test_voxtral_decode_writes_feedback_buffer_for_standard_forward() -> None:
    from sglang_omni.models.voxtral_tts.model_runner import VoxtralTTSModelRunner

    runner = VoxtralTTSModelRunner.__new__(VoxtralTTSModelRunner)
    runner.model = SimpleNamespace(
        hidden_size=3,
        decode_input_embed_buffer=torch.zeros(2, 3, dtype=torch.float16),
    )
    first = SimpleNamespace(
        data=SimpleNamespace(
            pending_feedback_queue=collections.deque([torch.tensor([1.0, 2.0, 3.0])])
        )
    )
    second = SimpleNamespace(
        data=SimpleNamespace(pending_feedback_queue=collections.deque())
    )

    result = runner.before_decode(object(), object(), [first, second])

    assert result is None
    assert not first.data.pending_feedback_queue
    assert torch.equal(
        runner.model.decode_input_embed_buffer,
        torch.tensor(
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
            dtype=torch.float16,
        ),
    )


def test_voxtral_decode_empty_batch_keeps_feedback_buffer() -> None:
    from sglang_omni.models.voxtral_tts.model_runner import VoxtralTTSModelRunner

    runner = VoxtralTTSModelRunner.__new__(VoxtralTTSModelRunner)
    runner.model = SimpleNamespace(
        hidden_size=3,
        decode_input_embed_buffer=torch.ones(1, 3, dtype=torch.float16),
    )

    result = runner.before_decode(object(), object(), [])

    assert result is None
    assert torch.equal(
        runner.model.decode_input_embed_buffer,
        torch.ones(1, 3, dtype=torch.float16),
    )


def test_voxtral_steady_decode_reports_cuda_graph_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode should use SGLang's graph-capable forward result."""
    from sglang.srt.model_executor import forward_batch_info

    from sglang_omni.models.voxtral_tts.model_runner import VoxtralTTSModelRunner

    fake_forward_batch = SimpleNamespace(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        mrope_positions=None,
    )
    monkeypatch.setattr(
        forward_batch_info.ForwardBatch,
        "init_new",
        staticmethod(
            lambda model_worker_batch, model_runner, *, capture_hidden_mode=None, return_hidden_states_before_norm: fake_forward_batch
        ),
    )

    class FakeVoxtralModel:
        hidden_size = 3

        def __init__(self) -> None:
            self.decode_input_embed_buffer = torch.zeros(1, 3)

        def run_acoustic_transformer(self, hidden):
            assert hidden.shape == (1, 3)
            return torch.tensor([[5]])

        def audio_token_embedding(self, codes):
            del codes
            return torch.ones(1, 1, 3)

    class FakeTPWorker:
        gpu_id = 0
        model_runner = SimpleNamespace(model=FakeVoxtralModel())

        def forward_batch_generation(self, forward_batch):
            del forward_batch
            return SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=torch.ones(1, 3)),
                next_token_ids=None,
                can_run_cuda_graph=True,
            )

    class FakeOutputProcessor:
        capture_hidden = False

        def process(self, model_output, scheduler_output):
            del model_output
            return {
                req.request_id: RequestOutput(req.request_id, data=5)
                for req in scheduler_output.requests
            }

    runner = VoxtralTTSModelRunner.__new__(VoxtralTTSModelRunner)
    runner.tp_worker = FakeTPWorker()
    runner.output_processor = FakeOutputProcessor()
    runner.device = torch.device("cpu")
    runner.model = runner.tp_worker.model_runner.model
    runner.bind_execution_bridge(FakeExecutionBridge())

    data = SimpleNamespace(
        pending_feedback_queue=collections.deque([torch.tensor([1.0, 2.0, 3.0])]),
        output_codes=[],
        stream=False,
        generation_steps=0,
        extra_model_outputs={},
    )
    request = SimpleNamespace(request_id="req", data=data)
    schedule_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False),
        is_prefill_only=False,
    )

    output = runner.execute(
        SimpleNamespace(requests=[request], batch_data=schedule_batch)
    )

    assert output.can_run_cuda_graph is True
    assert torch.equal(
        runner.model.decode_input_embed_buffer,
        torch.tensor([[1.0, 2.0, 3.0]]),
    )


def test_voxtral_forward_returns_graph_compatible_logits() -> None:
    from sglang_omni.models.voxtral_tts.sglang_model import VoxtralSGLangTTSModel

    model = VoxtralSGLangTTSModel.__new__(VoxtralSGLangTTSModel)
    model.decode_input_embed_buffer = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    def fake_language_model(input_ids, positions, forward_batch, input_embeds=None):
        del input_ids, positions, forward_batch
        assert input_embeds is not None
        return input_embeds + 1

    model.language_model = fake_language_model
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: True,
            is_extend=lambda: False,
        )
    )

    output = model.forward(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        forward_batch,
    )

    assert output.hidden_states.shape == (2, 3)
    assert output.next_token_logits.shape == (2, 1)
    assert output.next_token_logits.dtype == output.hidden_states.dtype
    assert output.next_token_logits.device == output.hidden_states.device


def test_voxtral_generation_reenables_cuda_graph_after_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.voxtral_tts import model_runner as model_runner_mod
    from sglang_omni.models.voxtral_tts import request_builders
    from sglang_omni.models.voxtral_tts.pipeline import stages
    from sglang_omni.scheduling import bootstrap as bootstrap_mod
    from sglang_omni.scheduling import engine_factory
    from sglang_omni.scheduling import omni_scheduler as scheduler_mod
    from sglang_omni.scheduling import sglang_backend

    build_kwargs: dict = {}
    infrastructure_saw_deferred_capture: list[bool] = []
    init_graph_calls: list[bool] = []

    acoustic_graph_sizes: list[list[int]] = []

    class FakeModel:
        def init_acoustic_graphs(self, batch_sizes) -> None:
            acoustic_graph_sizes.append(batch_sizes)

    class FakeSGLangRunner:
        def __init__(self, server_args) -> None:
            self.server_args = server_args
            self.model = FakeModel()

        def init_cuda_graphs(self) -> None:
            assert self.server_args.enable_torch_compile is True
            assert self.server_args.torch_compile_max_bs == 16
            init_graph_calls.append(True)

    class FakeWorker:
        def __init__(self, server_args) -> None:
            self.model_runner = FakeSGLangRunner(server_args)
            self.enable_prefill_input_embeds = False

    monkeypatch.setattr(
        engine_factory, "_resolve_checkpoint", lambda model_path: model_path
    )
    monkeypatch.setattr(
        stages,
        "write_voxtral_sglang_config",
        lambda checkpoint_dir: f"{checkpoint_dir}/config.json",
    )
    monkeypatch.setattr(
        stages,
        "load_voxtral_voice_embeddings",
        lambda checkpoint_dir, device: {},
    )
    monkeypatch.setattr(
        request_builders,
        "make_voxtral_scheduler_adapters",
        lambda **kwargs: (lambda payload: payload, lambda data: data),
    )

    def fake_build_sglang_server_args(model_path, context_length, **kwargs):
        del model_path, context_length
        build_kwargs.update(kwargs)
        return SimpleNamespace(
            cuda_graph_bs=kwargs["cuda_graph_bs"],
            cuda_graph_max_bs=kwargs["cuda_graph_max_bs"],
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(
                    max_bs=kwargs["cuda_graph_max_bs"],
                    bs=kwargs["cuda_graph_bs"],
                ),
                prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
            ),
            disable_cuda_graph=kwargs["disable_cuda_graph"],
            disable_overlap_schedule=kwargs["disable_overlap_schedule"],
            enable_torch_compile=kwargs["enable_torch_compile"],
            page_size=1,
            chunked_prefill_size=0,
            max_prefill_tokens=kwargs["max_prefill_tokens"],
            max_running_requests=kwargs["max_running_requests"],
            torch_compile_max_bs=kwargs["torch_compile_max_bs"],
        )

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        del gpu_id
        infrastructure_saw_deferred_capture.append(
            bool(kwargs.get("defer_cuda_graph_capture"))
        )
        return (
            FakeWorker(server_args),
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
        "VoxtralTTSModelRunner",
        lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
    )
    monkeypatch.setattr(
        scheduler_mod,
        "OmniScheduler",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    from sglang_omni.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cuda", raising=False)
    scheduler = stages.create_generation_executor("model", device="cuda", gpu_id=0)

    assert build_kwargs["disable_cuda_graph"] is False
    assert build_kwargs["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16]
    assert build_kwargs["cuda_graph_max_bs"] == 16
    assert build_kwargs["enable_torch_compile"] is True
    assert build_kwargs["sampling_backend"] == "pytorch"
    assert build_kwargs["torch_compile_max_bs"] == 16
    assert infrastructure_saw_deferred_capture == [True]
    assert init_graph_calls == [True]
    assert acoustic_graph_sizes == [[1, 2, 4, 8, 12, 16]]
    assert scheduler.server_args.cuda_graph_bs == [1, 2, 4, 8, 12, 16]
    assert scheduler.server_args.cuda_graph_max_bs == 16
    assert scheduler.server_args.disable_cuda_graph is False
    assert scheduler.server_args.enable_torch_compile is True
    assert scheduler.server_args.torch_compile_max_bs == 16


def test_enable_inductor_gemm_autotune_sets_per_shape_autotuning() -> None:
    from torch._inductor import config as inductor_config

    saved_gemm = inductor_config.max_autotune_gemm
    saved_backends = inductor_config.max_autotune_gemm_backends
    try:
        inductor_config.max_autotune_gemm = False
        inductor_config.max_autotune_gemm_backends = "ATEN"
        stages.enable_inductor_gemm_autotune()
        assert inductor_config.max_autotune_gemm is True
        assert inductor_config.max_autotune_gemm_backends == "TRITON,ATEN"
    finally:
        inductor_config.max_autotune_gemm = saved_gemm
        inductor_config.max_autotune_gemm_backends = saved_backends


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA graphs")
def test_voxtral_codec_graphs_refresh_mixed_windows_and_keep_delivered_audio() -> None:
    """Graph padding and overflow must preserve every window and earlier CPU audio."""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(2801)
        tokenizer = VoxtralTTSAudioTokenizer(
            {
                "semantic_codebook_size": 16,
                "semantic_dim": 8,
                "acoustic_dim": 4,
                "dim": 64,
                "hidden_dim": 128,
                "n_heads": 2,
                "n_kv_heads": 2,
                "head_dim": 32,
                "pretransform_patch_size": 8,
            },
            {"audio_model_args": {}},
        ).to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            tokenizer.quantizer.semantic_codebook.embedding_sum.normal_()
        tokenizer.eval()
        scheduler = VoxtralStreamingVocoder(VoxtralTTSVocoder(tokenizer), device="cuda")
        windows = []
        for index in range(35):
            frame_count = (1, 5, 24, 25, 50, 74, 75)[index % 7]
            codes = torch.randint(2, 23, (frame_count, tokenizer.num_codebooks))
            codes[:, 0] = torch.randint(2, 18, (frame_count,))
            windows.append(codes)

        retained_audio = []
        retained_snapshots = []
        for group_start, window_count in ((0, 16), (16, 3), (0, 35), (19, 3)):
            group_windows = windows[group_start : group_start + window_count]
            actual = scheduler.decode_windows(
                [
                    VoxtralStreamWindow(
                        codes=codes, offset_frames=0, new_frames=len(codes)
                    )
                    for codes in group_windows
                ]
            )
            expected_audio = []
            with torch.no_grad(), torch.cuda.stream(scheduler.decode_stream):
                for start in range(0, window_count, 16):
                    replay_windows = group_windows[start : start + 16]
                    bucket = next(
                        size for size in (1, 2, 4, 8, 16) if size >= len(replay_windows)
                    )
                    padded = torch.zeros(
                        bucket,
                        tokenizer.num_codebooks,
                        75,
                        device="cuda",
                        dtype=torch.long,
                    )
                    for index, codes in enumerate(replay_windows):
                        padded[index, :, : len(codes)] = (codes.t() - 2).to("cuda")
                    decoded = (
                        tokenizer.decode(padded, dtype=torch.bfloat16).float().cpu()
                    )
                    expected_audio.extend(
                        decoded[index, 0, : len(codes) * tokenizer.downsample_factor]
                        for index, codes in enumerate(replay_windows)
                    )
            assert len(actual) == len(expected_audio) == window_count
            for waveform, expected in zip(actual, expected_audio, strict=True):
                torch.testing.assert_close(waveform, expected, rtol=0, atol=0)
            for waveform, snapshot in zip(
                retained_audio, retained_snapshots, strict=True
            ):
                torch.testing.assert_close(waveform, snapshot, rtol=0, atol=0)
            retained_audio.extend(actual)
            retained_snapshots.extend(waveform.clone() for waveform in actual)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA graphs")
def test_voxtral_acoustic_graphs_refresh_inputs_and_advance_rng() -> None:
    """Successive padded replays must match the same seeded eager draw sequence."""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(2801)
        acoustic = (
            FlowMatchingAudioTransformer(
                {
                    "semantic_codebook_size": 16,
                    "acoustic_codebook_size": 21,
                    "n_acoustic_codebook": 4,
                    "acoustic_transformer_args": {
                        "input_dim": 16,
                        "dim": 16,
                        "n_layers": 1,
                        "head_dim": 8,
                        "hidden_dim": 32,
                        "n_heads": 2,
                        "n_kv_heads": 1,
                    },
                }
            )
            .to(device="cuda", dtype=torch.bfloat16)
            .eval()
        )
        with torch.no_grad():
            acoustic.semantic_codebook_output.weight.zero_()
            acoustic.semantic_codebook_output.weight[2].fill_(1)
            acoustic.semantic_codebook_output.weight[3].fill_(-1)
            acoustic.acoustic_codebook_output.weight.zero_()
        model = VoxtralSGLangTTSModel.__new__(VoxtralSGLangTTSModel)
        torch.nn.Module.__init__(model)
        model.acoustic_transformer = acoustic
        model.hidden_size = 16
        model.decode_input_embed_buffer = torch.zeros(
            4, 16, device="cuda", dtype=torch.bfloat16
        )
        model.acoustic_graphs = {}
        model.init_acoustic_graphs([1, 4])
        first_hidden = torch.ones(3, 16, device="cuda", dtype=torch.bfloat16)
        second_hidden = -first_hidden
        first_padded = torch.zeros(4, 16, device="cuda", dtype=torch.bfloat16)
        first_padded[:3] = first_hidden
        second_padded = -first_padded
        with torch.no_grad():
            torch.manual_seed(2888)
            expected_first = acoustic(first_padded)[:3]
            expected_second = acoustic(second_padded)[:3]
            expected_overflow = acoustic(first_padded.repeat(2, 1)[:5])
            torch.manual_seed(2888)
            actual_first = model.run_acoustic_transformer(first_hidden)
            actual_second = model.run_acoustic_transformer(second_hidden)
            actual_overflow = model.run_acoustic_transformer(
                first_padded.repeat(2, 1)[:5]
            )
        torch.testing.assert_close(actual_first, expected_first, rtol=0, atol=0)
        torch.testing.assert_close(actual_second, expected_second, rtol=0, atol=0)
        torch.testing.assert_close(actual_overflow, expected_overflow, rtol=0, atol=0)
