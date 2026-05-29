# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from sglang_omni.models.higgs_tts import stages
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.utils import EOC_ID
from sglang_omni.proto import OmniRequest, StagePayload


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

    def fake_make_adapters(model, **kwargs):
        captured["adapter_kwargs"] = kwargs
        return None, None

    monkeypatch.setattr(stages, "make_higgs_scheduler_adapters", fake_make_adapters)
    monkeypatch.setattr(stages, "OmniScheduler", FakeScheduler)

    stages.create_sglang_tts_engine_executor("boson-sglang/higgs-audio-v3-tts-4b-base")

    assert captured["checkpoint_dir"] == "boson-sglang/higgs-audio-v3-tts-4b-base"
    assert captured["context_length"] == 4096
    assert captured["gpu_id"] == 0
    assert captured["overrides"]["disable_cuda_graph"] is False
    assert captured["overrides"]["cuda_graph_max_bs"] == 32
    assert captured["server_args"].disable_overlap_schedule is True
    assert captured["adapter_kwargs"] == {"max_new_tokens_cap": 2048}


def test_higgs_model_runner_marks_sampler_finish() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
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


def test_higgs_model_runner_marks_sampler_finish_cg() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
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
