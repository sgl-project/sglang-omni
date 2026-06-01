# SPDX-License-Identifier: Apache-2.0

import base64
import queue
from types import SimpleNamespace

import numpy as np
import torch

from sglang_omni.models.higgs_tts import stages
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.utils import EOC_ID, apply_delay_pattern
from sglang_omni.models.higgs_tts.vocoder_scheduler import (
    HiggsStreamingVocoderScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload


def test_higgs_streaming_pipeline_routes_chunks_to_vocoder() -> None:
    config = HiggsTtsPipelineConfig(model_path="fake-model")
    stages_by_name = {stage.name: stage for stage in config.stages}

    assert stages_by_name["tts_engine"].stream_to == ["vocoder"]
    assert stages_by_name["vocoder"].can_accept_stream_before_payload is True


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
        model = SimpleNamespace(reset_request=lambda _request_id: None)
        return (
            SimpleNamespace(model_runner=SimpleNamespace(model=model)),
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

        def set_stream_outbox(self, outbox) -> None:
            self._outbox = outbox
            captured["stream_outbox"] = outbox

    class FakeScheduler:
        def __init__(self, **kwargs) -> None:
            captured["scheduler_kwargs"] = kwargs
            self.outbox = object()

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

    def fake_make_adapters(model, **kwargs):
        captured["adapter_kwargs"] = kwargs
        return None, None

    monkeypatch.setattr(stages, "make_higgs_scheduler_adapters", fake_make_adapters)
    monkeypatch.setattr(stages, "OmniScheduler", FakeScheduler)

    stages.create_sglang_tts_engine_executor(
        "boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999"
    )

    assert (
        captured["checkpoint_dir"]
        == "boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999"
    )
    assert captured["context_length"] == 4096
    assert captured["gpu_id"] == 0
    assert captured["overrides"]["disable_cuda_graph"] is False
    assert captured["overrides"]["cuda_graph_max_bs"] == 32
    assert captured["server_args"].disable_overlap_schedule is True
    assert captured["adapter_kwargs"] == {"max_new_tokens_cap": 2048}
    assert (
        captured["stream_outbox"]
        is captured["scheduler_kwargs"]["model_runner"]._outbox
    )


def test_higgs_reference_cache_key_round_trip() -> None:
    state = HiggsTtsState(reference_cache_key="path:/tmp/ref.wav")

    restored = HiggsTtsState.from_dict(state.to_dict())

    assert restored.reference_cache_key == "path:/tmp/ref.wav"


def test_higgs_reference_cache_key_tracks_file_content(tmp_path) -> None:
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"a")
    first_key = stages._reference_audio_cache_key(ref_audio)

    # Same content -> stable key (so repeat requests hit the cache).
    assert first_key == stages._reference_audio_cache_key(ref_audio)

    ref_audio.write_bytes(b"longer")
    second_key = stages._reference_audio_cache_key(ref_audio)

    # Different content -> different key (so a replaced file is not stale-served).
    assert first_key is not None and first_key.startswith("file:")
    assert second_key is not None and second_key.startswith("file:")
    assert first_key != second_key


def test_higgs_reference_cache_key_same_size_edit_and_urls(tmp_path) -> None:
    # Same path, same size, same head/tail, different middle must not stale-hit.
    head, tail = b"H" * 8192, b"T" * 8192
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(head + b"a" * 4096 + tail)
    key_a = stages._reference_audio_cache_key(ref_audio)
    ref_audio.write_bytes(head + b"b" * 4096 + tail)  # same size, middle differs
    assert key_a is not None and key_a != stages._reference_audio_cache_key(ref_audio)

    # URLs and missing files are not cached.
    assert stages._reference_audio_cache_key("https://example.com/ref.wav") is None
    assert stages._reference_audio_cache_key(str(tmp_path / "missing.wav")) is None


def test_higgs_reference_cache_key_memoizes_stable_file_hash(
    monkeypatch, tmp_path
) -> None:
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake wav bytes")
    stages._REF_PATH_HASH_MEMO.clear()
    read_calls = 0
    original_read_bytes = stages.Path.read_bytes

    def counting_read_bytes(path):
        nonlocal read_calls
        if path == ref_audio:
            read_calls += 1
        return original_read_bytes(path)

    monkeypatch.setattr(stages.Path, "read_bytes", counting_read_bytes)

    first_key = stages._reference_audio_cache_key(ref_audio)
    second_key = stages._reference_audio_cache_key(ref_audio)

    assert first_key == second_key
    assert read_calls == 1


def test_higgs_reference_cache_key_ignores_media_type() -> None:
    raw = b"\x01\x02\x03fake-audio-bytes"
    encoded = base64.b64encode(raw).decode()
    key_wav = stages._reference_audio_cache_key(
        {"base64": encoded, "media_type": "audio/wav"}
    )
    key_mp3 = stages._reference_audio_cache_key(
        {"base64": encoded, "media_type": "audio/mpeg"}
    )

    # media_type is a decode hint the codec ignores, so it must not split keys.
    assert key_wav is not None
    assert key_wav == key_mp3
    # Raw bytes and equivalent base64 resolve to the same content key.
    assert stages._reference_audio_cache_key({"bytes": raw}) == key_wav


def test_higgs_audio_encoder_uses_reference_code_cache(monkeypatch) -> None:
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(
        stages.Tokenizer,
        "from_file",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        stages,
        "PreTrainedTokenizerFast",
        lambda tokenizer_object: object(),
    )

    class FakeAdapter:
        def __init__(self, _tokenizer) -> None:
            pass

        def build_prompt(
            self, text: str, *, num_ref_tokens: int, reference_text: str | None
        ) -> list[int]:
            return [len(text), num_ref_tokens, len(reference_text or "")]

    class FakeCodec:
        SAMPLE_RATE = 24000

        def __init__(self) -> None:
            self.calls = 0
            self.model = SimpleNamespace(acoustic_encoder=torch.nn.Identity())

        def encode_reference(self, waveform, sample_rate: int) -> torch.Tensor:
            self.calls += 1
            return torch.tensor([[11, 12], [21, 22]], dtype=torch.long)

    fake_codec = FakeCodec()
    monkeypatch.setattr(stages, "HiggsTokenizerAdapter", FakeAdapter)
    monkeypatch.setattr(stages, "get_or_load_codec", lambda *args, **kwargs: fake_codec)

    scheduler = stages.create_audio_encoder_executor(
        "ckpt",
        device="cuda:0",
        num_codebooks=2,
    )
    # Ignore the construction-time codec warm-up call added by #612.
    fake_codec.calls = 0
    encode = scheduler._fn

    def make_payload(request_id: str) -> StagePayload:
        state = HiggsTtsState(
            reference_waveform=torch.zeros(1, 1, 16),
            reference_cache_key="path:/tmp/ref.wav",
            target_text="hello",
            reference_text="speaker",
            num_codebooks=2,
        )
        return StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs={}),
            data=state.to_dict(),
        )

    first = encode(make_payload("first"))
    second = encode(make_payload("second"))

    assert fake_codec.calls == 1
    assert (
        first.data["reference_codes_delayed"] == second.data["reference_codes_delayed"]
    )
    assert first.data["prompt_token_ids"] == [5, 3, 7]
    assert second.data["prompt_token_ids"] == [5, 3, 7]
    assert "reference_waveform" not in second.data
    assert "reference_cache_key" not in second.data


def test_higgs_preprocessing_uses_waveform_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(stages.Tokenizer, "from_file", lambda _path: object())
    monkeypatch.setattr(
        stages,
        "PreTrainedTokenizerFast",
        lambda tokenizer_object: object(),
    )
    monkeypatch.setattr(stages, "HiggsTokenizerAdapter", lambda _tokenizer: object())

    load_calls = 0

    def fake_load_audio_to_24k(reference_audio):
        nonlocal load_calls
        load_calls += 1
        return np.zeros(16, dtype=np.float32), 24000

    monkeypatch.setattr(stages, "load_audio_to_24k", fake_load_audio_to_24k)

    scheduler = stages.create_preprocessing_executor("ckpt", num_codebooks=2)
    preprocess = scheduler._fn
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake wav bytes")

    def make_payload(request_id: str) -> StagePayload:
        return StagePayload(
            request_id=request_id,
            request=OmniRequest(
                inputs={
                    "text": "hello",
                    "references": [{"audio_path": str(ref_audio), "text": "speaker"}],
                },
                params={},
            ),
            data={},
        )

    first = preprocess(make_payload("first"))
    second = preprocess(make_payload("second"))
    first_state = HiggsTtsState.from_dict(first.data)
    second_state = HiggsTtsState.from_dict(second.data)

    assert load_calls == 1
    assert first_state.reference_cache_key == second_state.reference_cache_key
    assert torch.equal(first_state.reference_waveform, second_state.reference_waveform)
    assert (
        first_state.reference_waveform.data_ptr()
        != second_state.reference_waveform.data_ptr()
    )


def test_higgs_model_runner_marks_sampler_finish() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    runner._outbox = None
    runner._vocoder_target = "vocoder"
    runner.model = SimpleNamespace(
        _rid_to_row={"req": 0},
        _output_codes={"req": [torch.tensor([EOC_ID, 1, 2])]},
        _sampler_pool=SimpleNamespace(generation_done=torch.tensor([True])),
    )
    req = SimpleNamespace(
        is_chunked=0,
        finished_reason=None,
        finished=lambda: False,
    )
    data = SimpleNamespace(req=req, output_codes=[], generation_done=False)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(1, 4))
    )

    runner._collect_step_outputs(
        result,
        [SimpleNamespace(request_id="req", data=data)],
    )

    assert data.generation_done is True
    assert req.finished_reason.to_json() == {"type": "stop", "matched": EOC_ID}
    assert len(data.output_codes) == 1


def test_higgs_model_runner_emits_latched_stream_metadata() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    runner._outbox = queue.Queue()
    runner._vocoder_target = "vocoder"
    runner.model = SimpleNamespace(
        _rid_to_row={"req": 0},
        _output_codes={"req": [torch.tensor([EOC_ID, 1, 2])]},
        _sampler_pool=SimpleNamespace(generation_done=torch.tensor([True])),
    )
    req = SimpleNamespace(
        is_chunked=0,
        finished_reason=None,
        finished=lambda: False,
    )
    data = SimpleNamespace(
        req=req,
        output_codes=[],
        generation_done=False,
        stream_metadata={
            "modality": "audio_codes",
            "stream": True,
            "num_codebooks": 3,
            "codebook_size": 17,
        },
    )
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(1, 4))
    )

    runner._collect_step_outputs(
        result,
        [SimpleNamespace(request_id="req", data=data)],
    )

    out = runner._outbox.get_nowait()
    assert out.type == "stream"
    assert out.target == "vocoder"
    assert out.data.tolist() == [EOC_ID, 1, 2]
    assert out.metadata == {
        "modality": "audio_codes",
        "stream": True,
        "num_codebooks": 3,
        "codebook_size": 17,
    }


def test_higgs_model_runner_marks_sampler_finish_cg() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    runner._outbox = None
    runner._vocoder_target = "vocoder"
    runner.model = SimpleNamespace(
        _cg_row_indices=torch.tensor([0]),
        _cg_active_delay_count=torch.tensor([8], dtype=torch.int32),
        _cg_active_eoc_countdown=torch.tensor([0], dtype=torch.int32),
        _cg_active_generation_done=torch.tensor([True]),
        _cg_active_last_codes=torch.tensor([[1, 2, 3]]),
        _cg_was_done=torch.tensor([False]),
        _cg_codes_BN=torch.tensor([[EOC_ID, 1, 2]]),
        _cg_collect_staging=torch.zeros((1, 3 + 2), dtype=torch.long),
        _sampler_pool=SimpleNamespace(
            delay_count=torch.zeros(1, dtype=torch.int32),
            eoc_countdown=torch.zeros(1, dtype=torch.int32),
            generation_done=torch.zeros(1, dtype=torch.bool),
            last_codes=torch.zeros((1, 3), dtype=torch.long),
        ),
    )
    req = SimpleNamespace(is_chunked=0, finished_reason=None, finished=lambda: False)
    data = SimpleNamespace(req=req, output_codes=[], generation_done=False)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(1, 4))
    )
    forward_batch = SimpleNamespace(batch_size=1)

    runner._collect_step_outputs_cg(
        result,
        forward_batch,
        [SimpleNamespace(request_id="req", data=data)],
    )

    assert data.generation_done is True
    assert req.finished_reason.to_json() == {"type": "stop", "matched": EOC_ID}
    assert len(data.output_codes) == 1


def test_higgs_model_runner_collect_cg_mixed_batch() -> None:
    """A 4-row batch covering chunked / was-done / active rows verifies the
    batched single-D2H packing preserves per-row semantics, including the
    bool->long->bool round-trip for generation_done.
    """
    n, k = 4, 3
    runner = object.__new__(HiggsTTSModelRunner)
    runner._outbox = None
    runner._vocoder_target = "vocoder"
    runner.model = SimpleNamespace(
        _cg_row_indices=torch.arange(n),
        _cg_active_delay_count=torch.zeros(n, dtype=torch.int32),
        _cg_active_eoc_countdown=torch.zeros(n, dtype=torch.int32),
        # row1's True must NOT leak into the was-done (skipped) request.
        _cg_active_generation_done=torch.tensor([False, True, False, True]),
        _cg_active_last_codes=torch.zeros((n, k), dtype=torch.long),
        _cg_was_done=torch.tensor([False, True, False, False]),
        _cg_codes_BN=torch.tensor([[1, 1, 1], [7, 8, 9], [20, 1, 2], [EOC_ID, 3, 4]]),
        _cg_collect_staging=torch.zeros((n, k + 2), dtype=torch.long),
        _sampler_pool=SimpleNamespace(
            delay_count=torch.zeros(n, dtype=torch.int32),
            eoc_countdown=torch.zeros(n, dtype=torch.int32),
            generation_done=torch.zeros(n, dtype=torch.bool),
            last_codes=torch.zeros((n, k), dtype=torch.long),
        ),
    )
    # row0 chunked, row1 was-done, row2 active (not done), row3 active (EOC done).
    reqs = [
        SimpleNamespace(is_chunked=1, finished_reason=None, finished=lambda: False),
        SimpleNamespace(is_chunked=0, finished_reason=None, finished=lambda: False),
        SimpleNamespace(is_chunked=0, finished_reason=None, finished=lambda: False),
        SimpleNamespace(is_chunked=0, finished_reason=None, finished=lambda: False),
    ]
    datas = [
        SimpleNamespace(req=r, output_codes=[], generation_done=False) for r in reqs
    ]
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(n, 4))
    )
    forward_batch = SimpleNamespace(batch_size=n)

    runner._collect_step_outputs_cg(
        result,
        forward_batch,
        [SimpleNamespace(request_id=f"req{i}", data=d) for i, d in enumerate(datas)],
    )

    assert [len(d.output_codes) for d in datas] == [0, 0, 1, 1]
    # Direct bool-list equality locks the bool->long->bool round-trip; the
    # was-done row stays False despite _cg_active_generation_done[1] being True.
    assert [d.generation_done for d in datas] == [False, False, False, True]
    assert result.next_token_ids.tolist() == [0, 0, 20, EOC_ID]
    assert datas[2].output_codes[0].tolist() == [20, 1, 2]
    assert datas[3].output_codes[0].tolist() == [EOC_ID, 3, 4]
    assert reqs[3].finished_reason.to_json() == {"type": "stop", "matched": EOC_ID}
    assert all(reqs[i].finished_reason is None for i in (0, 1, 2))


def test_higgs_model_runner_skips_already_finished_eager_request() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    runner._outbox = None
    runner._vocoder_target = "vocoder"
    runner.model = SimpleNamespace(
        _rid_to_row={"req": 0},
        _output_codes={"req": [torch.tensor([EOC_ID, 1, 2])]},
        _sampler_pool=SimpleNamespace(generation_done=torch.tensor([True])),
    )
    req = SimpleNamespace(
        is_chunked=0,
        finished_reason=object(),
        finished=lambda: True,
    )
    data = SimpleNamespace(req=req, output_codes=[], generation_done=True)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(1, 4))
    )

    runner._collect_step_outputs(
        result,
        [SimpleNamespace(request_id="req", data=data)],
    )

    assert data.output_codes == []
    assert result.next_token_ids.tolist() == [0]


def _make_payload(request_id: str, state: HiggsTtsState) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=""),
        data=state.to_dict(),
    )


def _fake_codec_fixtures(monkeypatch):
    """Patch codec loading; return list that records decode_batch call sizes."""
    decode_batch_sizes: list[int] = []

    class FakeCodec:
        SAMPLE_RATE = 24_000

        def decode(self, codes_TN):
            return torch.zeros(codes_TN.shape[0], dtype=torch.float32)

        def decode_batch(self, codes_list):
            decode_batch_sizes.append(len(codes_list))
            return [torch.arange(c.shape[0], dtype=torch.float32) for c in codes_list]

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda p: p)
    monkeypatch.setattr(stages, "get_or_load_codec", lambda *a, **kw: FakeCodec())
    return decode_batch_sizes


def test_higgs_tts_vocoder_batches_decode_requests(
    monkeypatch,
) -> None:
    """Protects Higgs TTS vocoder throughput from regressing to serial decode."""
    decode_batch_sizes = _fake_codec_fixtures(monkeypatch)

    scheduler = stages.create_vocoder_executor(
        "fake-model", max_batch_size=4, max_batch_wait_ms=2
    )

    p1 = _make_payload(
        "r1",
        HiggsTtsState(
            output_codes_delayed=[[i % 100] * 8 for i in range(10)],
            prompt_tokens=5,
            completion_tokens=10,
            engine_time_s=0.5,
        ),
    )
    p2 = _make_payload(
        "r2",
        HiggsTtsState(
            output_codes_delayed=[[i % 100] * 8 for i in range(12)],
        ),
    )

    results = scheduler._batch_fn([p1, p2])

    assert decode_batch_sizes == [2], "should call decode_batch once with 2 items"
    assert len(results) == 2
    assert len(results[0].data["audio_data"]) > 0
    assert results[0].data["usage"]["prompt_tokens"] == 5


def test_higgs_tts_vocoder_batch_handles_empty_items(
    monkeypatch,
) -> None:
    """Items with empty/too-short codes get empty audio_data, not a crash."""
    decode_batch_sizes = _fake_codec_fixtures(monkeypatch)

    scheduler = stages.create_vocoder_executor("fake-model", max_batch_size=4)

    payloads = [
        _make_payload("r-empty", HiggsTtsState(output_codes_delayed=None)),
        _make_payload(
            "r-short",
            HiggsTtsState(output_codes_delayed=[[0] * 8 for _ in range(3)]),
        ),
        _make_payload(
            "r-valid",
            HiggsTtsState(output_codes_delayed=[[i % 100] * 8 for i in range(10)]),
        ),
    ]

    results = scheduler._batch_fn(payloads)

    assert decode_batch_sizes == [1], "only the valid item should be batched"
    assert results[0].data["audio_data"] == []
    assert results[1].data["audio_data"] == []
    assert len(results[2].data["audio_data"]) > 0


class _FakeHiggsStreamingCodec:
    def __init__(self, samples_per_frame: int = 4) -> None:
        self.samples_per_frame = samples_per_frame
        self.decode_inputs: list[torch.Tensor] = []
        self.decode_batch_sizes: list[int] = []

    def decode(self, codes_TN: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(codes_TN.detach().clone())
        frames = int(codes_TN.shape[0])
        return torch.arange(frames * self.samples_per_frame, dtype=torch.float32)

    def decode_batch(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        self.decode_batch_sizes.append(len(codes_list))
        return [self.decode(codes) for codes in codes_list]


class _FakeUnevenHiggsStreamingCodec:
    class _Model:
        class config:
            hop_length = 5

    def __init__(self, tail_samples: int = 3) -> None:
        self.model = self._Model()
        self.tail_samples = tail_samples

    def decode(self, codes_TN: torch.Tensor) -> torch.Tensor:
        frames = []
        offsets = torch.arange(
            self.model.config.hop_length,
            dtype=torch.float32,
        )
        for row in codes_TN:
            frames.append(row.sum().float().repeat(self.model.config.hop_length))
            frames[-1] = frames[-1] + offsets / 10.0
        body = torch.cat(frames) if frames else torch.empty(0, dtype=torch.float32)
        tail = torch.arange(self.tail_samples, dtype=torch.float32) + 10_000
        return torch.cat([body, tail])

    def decode_batch(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.decode(codes) for codes in codes_list]


def _higgs_stream_payload(
    request_id: str,
    *,
    stream: bool,
    delayed_rows: list[list[int]],
    num_codebooks: int = 3,
    codebook_size: int = 20,
) -> StagePayload:
    state = HiggsTtsState(
        output_codes_delayed=delayed_rows,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        prompt_tokens=2,
        completion_tokens=len(delayed_rows),
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="", params={"stream": stream}),
        data=state.to_dict(),
    )


def _higgs_stream_item(
    row: torch.Tensor,
    *,
    num_codebooks: int = 3,
    codebook_size: int = 20,
) -> StreamItem:
    return StreamItem(
        chunk_id=0,
        data=row,
        from_stage="tts_engine",
        metadata={
            "modality": "audio_codes",
            "stream": True,
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
        },
    )


def test_higgs_streaming_vocoder_emits_compact_chunks_and_slim_final() -> None:
    raw_codes = torch.tensor(
        [
            [1, 8, 9],
            [2, 3, 10],
            [4, 11, 12],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )
    delayed = apply_delay_pattern(raw_codes)
    codec = _FakeHiggsStreamingCodec()
    scheduler = HiggsStreamingVocoderScheduler(
        codec,
        stream_stride=3,
        stream_followup_stride=2,
        stream_holdback_tokens=0,
    )
    payload = _higgs_stream_payload(
        "req",
        stream=True,
        delayed_rows=delayed.tolist(),
        codebook_size=7,
    )

    scheduler._on_streaming_new_request("req", payload)
    for idx, row in enumerate(delayed):
        item = _higgs_stream_item(row, codebook_size=7)
        item.chunk_id = idx
        scheduler._on_chunk("req", item)
    scheduler._on_done("req")

    messages = _drain_higgs_outbox(scheduler)
    stream_messages = [msg for msg in messages if msg.type == "stream"]
    result_messages = [msg for msg in messages if msg.type == "result"]

    assert stream_messages
    first_chunk = stream_messages[0].data
    assert "audio_waveform" in first_chunk
    assert "audio_data" not in first_chunk
    audio = np.frombuffer(first_chunk["audio_waveform"], dtype=np.float32)
    assert audio.shape == tuple(first_chunk["audio_waveform_shape"])
    assert first_chunk["audio_waveform_dtype"] == "float32"
    assert first_chunk["sample_rate"] == 24000
    assert codec.decode_inputs
    assert all(int(codes.max().item()) < 5 for codes in codec.decode_inputs)

    assert len(result_messages) == 1
    final_data = result_messages[0].data.data
    assert final_data == {
        "modality": "audio",
        "sample_rate": 24000,
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": len(delayed),
            "total_tokens": 2 + len(delayed),
        },
    }
    assert "req" not in scheduler._pending_done
    assert "req" not in scheduler._stream_states


def test_higgs_streaming_vocoder_matches_full_decode_with_codec_tail() -> None:
    raw_codes = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
        ],
        dtype=torch.long,
    )
    delayed = apply_delay_pattern(raw_codes)
    codec = _FakeUnevenHiggsStreamingCodec()
    scheduler = HiggsStreamingVocoderScheduler(
        codec,
        stream_stride=3,
        stream_followup_stride=2,
        stream_overlap_tokens=1,
        stream_holdback_tokens=0,
    )
    payload = _higgs_stream_payload(
        "req",
        stream=True,
        delayed_rows=delayed.tolist(),
        codebook_size=64,
    )

    full = scheduler._decode_state_to_audio(HiggsTtsState.from_dict(payload.data))
    assert full is not None

    scheduler._on_streaming_new_request("req", payload)
    for idx, row in enumerate(delayed):
        item = _higgs_stream_item(row, codebook_size=64)
        item.chunk_id = idx
        scheduler._on_chunk("req", item)
    scheduler._on_done("req")

    stream_chunks = [
        np.frombuffer(msg.data["audio_waveform"], dtype=np.float32).copy()
        for msg in _drain_higgs_outbox(scheduler)
        if msg.type == "stream"
    ]
    streamed = np.concatenate(stream_chunks)
    np.testing.assert_array_equal(streamed, full.numpy())


def _drain_higgs_outbox(
    scheduler: HiggsStreamingVocoderScheduler,
) -> list:
    messages = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def _make_fake_codec(call_log: list[tuple[int, int]]):
    """Build a HiggsAudioCodec wrapping a deterministic FakeModel that logs (B, T)."""
    from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec

    class FakeModel:
        class config:
            hop_length = 320

        def decode(self, codes_BNT):
            B, N, T = codes_BNT.shape
            call_log.append((B, T))
            L = 320 * T + 64
            audio = torch.zeros(B, 1, L)
            for b in range(B):
                audio[b, 0, :] = codes_BNT[b].float().sum(dim=0).repeat(L // T + 1)[:L]
            return SimpleNamespace(audio_values=audio)

    codec = object.__new__(HiggsAudioCodec)
    codec.model = FakeModel()
    codec.device = torch.device("cpu")
    codec._dtype = torch.float32
    return codec


def test_decode_batch_buckets_by_length() -> None:
    """Same-T items batch into one call; mixed-T items get separate calls."""
    call_log: list[tuple[int, int]] = []
    codec = _make_fake_codec(call_log)

    # Same length → single batched forward pass
    same = [torch.randint(0, 100, (10, 8)) for _ in range(3)]
    results = codec.decode_batch(same)
    assert call_log == [(3, 10)], "single batched call with B=3"
    assert all(r.shape == (320 * 10 + 64,) for r in results)

    # Mixed lengths → per-bucket calls
    call_log.clear()
    mixed = [
        torch.randint(0, 100, (10, 8)),
        torch.randint(0, 100, (10, 8)),
        torch.randint(0, 100, (15, 8)),
    ]
    results = codec.decode_batch(mixed)
    assert sorted(call_log) == [(1, 15), (2, 10)]
    assert results[2].shape == (320 * 15 + 64,)


def test_decode_batch_bit_exact_with_single_decode() -> None:
    """Batched decode must produce identical output to individual decode."""
    call_log: list[tuple[int, int]] = []
    codec = _make_fake_codec(call_log)

    codes_a = torch.randint(0, 100, (10, 8))
    codes_b = torch.randint(0, 100, (10, 8))

    single_a = codec.decode(codes_a)
    single_b = codec.decode(codes_b)
    call_log.clear()

    batch_results = codec.decode_batch([codes_a, codes_b])

    assert call_log == [(2, 10)]
    assert torch.equal(single_a, batch_results[0])
    assert torch.equal(single_b, batch_results[1])


def _make_fake_encoder_codec(encode_calls: list):
    """Build a HiggsAudioCodec wrapping a FakeModel that logs encode calls.

    Each item in the batch gets codes filled with ``int(batch[i,0,0]*100)``,
    making outputs distinguishable by the waveform's first sample value.
    """
    from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec

    N = 8  # num_codebooks

    class FakeModel:
        def encode(self, batch: torch.Tensor):
            B, _, L = batch.shape
            T = max(L // 320, 1)
            encode_calls.append(tuple(batch.shape))
            codes = torch.zeros(B, N, T, dtype=torch.long)
            for i in range(B):
                codes[i] = int(round(batch[i, 0, 0].item() * 100))
            return SimpleNamespace(audio_codes=codes)

        def parameters(self):
            return iter([torch.zeros(1)])

    codec = object.__new__(HiggsAudioCodec)
    codec.model = FakeModel()
    codec.device = torch.device("cpu")
    codec._dtype = torch.float32
    return codec


def test_higgs_audio_codec_encode_batch_empty() -> None:
    codec = _make_fake_encoder_codec([])
    assert codec.encode_batch([]) == []


def test_higgs_audio_codec_encode_batch_same_length_batched() -> None:
    calls: list = []
    codec = _make_fake_encoder_codec(calls)
    wav1 = torch.zeros(1, 1, 24000)
    wav2 = torch.ones(1, 1, 24000) * 0.5
    results = codec.encode_batch([wav1, wav2])
    assert len(calls) == 1, "same-length waveforms must batch into one forward pass"
    assert calls[0] == (2, 1, 24000)
    assert len(results) == 2
    assert all(r.shape == (75, 8) for r in results)


def test_higgs_audio_codec_encode_batch_short_waveforms_padded_and_batched() -> None:
    calls: list = []
    codec = _make_fake_encoder_codec(calls)
    wav1 = torch.zeros(1, 1, 8000)
    wav2 = torch.zeros(1, 1, 12000)
    results = codec.encode_batch([wav1, wav2])
    assert len(calls) == 1, "short waveforms must be padded to same length and batched"
    assert calls[0] == (2, 1, 24000)
    assert len(results) == 2


def test_higgs_audio_codec_encode_batch_different_lengths_separate_calls() -> None:
    calls: list = []
    codec = _make_fake_encoder_codec(calls)
    wav1 = torch.zeros(1, 1, 24000)
    wav2 = torch.zeros(1, 1, 48000)
    results = codec.encode_batch([wav1, wav2])
    assert len(calls) == 2, "different-length waveforms must use separate passes"
    assert len(results) == 2
    assert results[0].shape[0] == 75
    assert results[1].shape[0] == 150


def test_higgs_audio_codec_encode_batch_order_preserved() -> None:
    calls: list = []
    codec = _make_fake_encoder_codec(calls)
    wavs = [
        torch.full((1, 1, 48000), 0.3),
        torch.full((1, 1, 24000), 0.5),
        torch.full((1, 1, 48000), 0.7),
    ]
    ref0 = codec.encode_reference(wavs[0])
    ref1 = codec.encode_reference(wavs[1])
    ref2 = codec.encode_reference(wavs[2])
    calls.clear()

    results = codec.encode_batch(wavs)

    assert len(results) == 3
    assert torch.equal(results[0], ref0)
    assert torch.equal(results[1], ref1)
    assert torch.equal(results[2], ref2)


def test_higgs_audio_codec_encode_batch_matches_encode_reference() -> None:
    """encode_batch must produce bit-exact codes vs per-item encode_reference."""
    calls: list = []
    codec = _make_fake_encoder_codec(calls)
    wav1 = torch.full((1, 1, 24000), 0.3)
    wav2 = torch.full((1, 1, 24000), 0.7)
    wav3 = torch.full((1, 1, 48000), 0.5)

    ref1 = codec.encode_reference(wav1)
    ref2 = codec.encode_reference(wav2)
    ref3 = codec.encode_reference(wav3)
    calls.clear()

    batch_results = codec.encode_batch([wav1, wav2, wav3])

    assert torch.equal(batch_results[0], ref1)
    assert torch.equal(batch_results[1], ref2)
    assert torch.equal(batch_results[2], ref3)


def test_higgs_audio_codec_encode_batch_input_normalisation() -> None:
    """1-D tensor, 2-D tensor, and np.ndarray inputs all normalise to the same codes."""
    calls: list = []
    codec = _make_fake_encoder_codec(calls)
    amp = 0.4

    wav_3d = torch.full((1, 1, 24000), amp)
    wav_2d = torch.full((1, 24000), amp)
    wav_1d = torch.full((24000,), amp)
    wav_np = np.full(24000, amp, dtype=np.float32)

    ref = codec.encode_reference(wav_3d)
    calls.clear()

    results = codec.encode_batch([wav_3d, wav_2d, wav_1d, wav_np])

    for i, r in enumerate(results):
        assert torch.equal(
            r, ref
        ), f"input format {i} produced different codes than encode_reference"
