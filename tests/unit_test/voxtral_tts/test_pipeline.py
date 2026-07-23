# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import collections
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.voxtral_tts.config import VoxtralTTSPipelineConfig
from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.models.voxtral_tts.pipeline import stages
from sglang_omni.models.voxtral_tts.request_builders import build_sglang_voxtral_request
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import RequestOutput
from sglang_omni.utils.audio_payload import audio_waveform_payload


def test_voxtral_tts_config_uses_current_stage_schema() -> None:
    config = VoxtralTTSPipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_generation",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"tts_generation": 0, "vocoder": 0}
    assert "device" not in config.stages[1].factory_args
    assert "device" not in config.stages[2].factory_args
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("VoxtralTTSForConditionalGeneration")
        is VoxtralTTSPipelineConfig
    )


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
    stages._validate_voxtral_speech_params(
        inputs="hello",
        params={
            "max_new_tokens": 128,
            "temperature": 0.8,
            "top_p": 0.8,
            "top_k": 30,
            "repetition_penalty": 1.1,
            "stream": True,
        },
        tts_params={
            "voice": "cheerful_female",
            "response_format": "wav",
            "speed": 1.0,
            "explicit_generation_params": ["max_new_tokens"],
        },
    )


@pytest.mark.parametrize(
    ("params", "tts_params", "inputs", "field"),
    [
        (
            {"temperature": 0.2},
            {"explicit_generation_params": ["temperature"]},
            "hello",
            "temperature",
        ),
        ({}, {"explicit_generation_params": ["seed"], "seed": 7}, "hello", "seed"),
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
        stages._validate_voxtral_speech_params(
            inputs=inputs,
            params=params,
            tts_params=tts_params,
        )


@pytest.mark.parametrize("audio_codes", [None, torch.empty((0, 0), dtype=torch.long)])
def test_voxtral_vocoder_rejects_empty_audio_codes(audio_codes) -> None:
    with pytest.raises(ValueError, match="generated no audio codes"):
        stages._ensure_non_empty_audio_codes(audio_codes)


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
        "_load_audio_tokenizer",
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

    result = asyncio.run(scheduler._fn(payload))

    assert scheduler._max_batch_size == 1
    assert scheduler._max_batch_wait_s == 0
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


def test_voxtral_collect_audio_step_reuses_output_tokens_for_eos_filter() -> None:
    from sglang_omni.models.voxtral_tts.acoustic_transformer import AudioSpecialTokens
    from sglang_omni.models.voxtral_tts.model_runner import VoxtralTTSModelRunner

    eos_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
    runner = VoxtralTTSModelRunner.__new__(VoxtralTTSModelRunner)
    runner._pending_audio_codes = None
    runner._pending_audio_embeds = None
    runner.model = SimpleNamespace(
        acoustic_transformer=lambda hidden: torch.tensor(
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
            ),
        ),
        SimpleNamespace(
            request_id="eos",
            data=SimpleNamespace(
                output_codes=[],
                pending_feedback_queue=[],
            ),
        ),
    ]

    runner._collect_audio_step(result, schedule_batch, requests)

    assert result.next_token_ids.tolist() == [11, eos_id]
    assert schedule_batch.output_ids.tolist() == [11, eos_id]
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
        _decode_input_embed_buffer=torch.zeros(2, 3, dtype=torch.float16),
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
        runner.model._decode_input_embed_buffer,
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
        _decode_input_embed_buffer=torch.ones(1, 3, dtype=torch.float16),
    )

    result = runner.before_decode(object(), object(), [])

    assert result is None
    assert torch.equal(
        runner.model._decode_input_embed_buffer,
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
        staticmethod(lambda model_worker_batch, model_runner: fake_forward_batch),
    )

    class FakeVoxtralModel:
        hidden_size = 3

        def __init__(self) -> None:
            self._decode_input_embed_buffer = torch.zeros(1, 3)

        def acoustic_transformer(self, hidden):
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
        _capture_hidden = False

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

    data = SimpleNamespace(
        pending_feedback_queue=collections.deque([torch.tensor([1.0, 2.0, 3.0])]),
        output_codes=[],
        generation_steps=0,
        extra_model_outputs={},
    )
    request = SimpleNamespace(request_id="req", data=data)
    schedule_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False),
        is_prefill_only=False,
        output_ids=None,
        get_model_worker_batch=lambda: SimpleNamespace(),
    )

    output = runner.execute(
        SimpleNamespace(requests=[request], batch_data=schedule_batch)
    )

    assert output.can_run_cuda_graph is True
    assert schedule_batch.output_ids.tolist() == [5]
    assert torch.equal(
        runner.model._decode_input_embed_buffer,
        torch.tensor([[1.0, 2.0, 3.0]]),
    )


def test_voxtral_forward_returns_graph_compatible_logits() -> None:
    from sglang_omni.models.voxtral_tts.sglang_model import VoxtralSGLangTTSModel

    model = VoxtralSGLangTTSModel.__new__(VoxtralSGLangTTSModel)
    model._decode_input_embed_buffer = torch.arange(6, dtype=torch.float32).reshape(
        2, 3
    )

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
    infrastructure_saw_graph_disabled: list[bool] = []
    init_graph_calls: list[bool] = []

    class FakeModel:
        pass

    class FakeSGLangRunner:
        def __init__(self, server_args) -> None:
            self.server_args = server_args
            self.model = FakeModel()

        def init_device_graphs(self) -> None:
            assert self.server_args.enable_torch_compile is True
            assert self.server_args.torch_compile_max_bs == 16
            init_graph_calls.append(True)

    class FakeWorker:
        def __init__(self, server_args) -> None:
            self.model_runner = FakeSGLangRunner(server_args)

    monkeypatch.setattr(
        engine_factory, "_resolve_checkpoint", lambda model_path: model_path
    )
    monkeypatch.setattr(
        stages,
        "_write_voxtral_sglang_config",
        lambda checkpoint_dir: f"{checkpoint_dir}/config.json",
    )
    monkeypatch.setattr(
        stages,
        "_load_voxtral_voice_embeddings",
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
        "VoxtralTTSModelRunner",
        lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
    )
    monkeypatch.setattr(
        scheduler_mod,
        "OmniScheduler",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    scheduler = stages.create_generation_executor("model", device="cuda:0")

    assert build_kwargs["disable_cuda_graph"] is False
    assert build_kwargs["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16]
    assert build_kwargs["cuda_graph_max_bs"] == 16
    assert build_kwargs["enable_torch_compile"] is True
    assert build_kwargs["sampling_backend"] == "pytorch"
    assert build_kwargs["torch_compile_max_bs"] == 16
    assert infrastructure_saw_graph_disabled == [True]
    assert init_graph_calls == [True]
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
        stages._enable_inductor_gemm_autotune()
        assert inductor_config.max_autotune_gemm is True
        assert inductor_config.max_autotune_gemm_backends == "TRITON,ATEN"
    finally:
        inductor_config.max_autotune_gemm = saved_gemm
        inductor_config.max_autotune_gemm_backends = saved_backends
