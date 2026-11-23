# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.scheduling.omni_scheduler as omni_scheduler_module
from sglang_omni.models.fishaudio_s2_pro.model_runner import (
    FishS2ProModelRunner,
    collect_s2pro_step_outputs,
)
from sglang_omni.models.fishaudio_s2_pro.request_builders import (
    S2ProSGLangRequestData,
    make_tts_scheduler_adapters,
)
from sglang_omni.models.fishaudio_s2_pro.sglang_model import S2ProSGLangTextModel
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import SchedulerRequest
from tests.unit_test.fixtures.fish_fakes import (
    FakeFishModel,
    FakeFishReq,
    FakeFishTokenizer,
    make_s2pro_payload,
    make_s2pro_state,
)

IM_END_TOKEN_ID = 151645
SEMANTIC_TOKEN_ID = 151678


def test_fish_model_runner_vq_injection_and_code_collection_contracts() -> None:
    """Preserves VQ prompt embedding injection and semantic code collection."""
    runner = object.__new__(FishS2ProModelRunner)
    runner.model = FakeFishModel()
    runner.semantic_begin_id = 200
    runner.semantic_end_id = 295
    runner.im_end_token_id = 99
    prefill_request = SchedulerRequest(
        request_id="prefill",
        data=SimpleNamespace(
            req=FakeFishReq(extend_len=3),
            vq_mask_tokens=torch.tensor([True, False, True]),
            vq_parts=[torch.tensor([[7, 8], [9, 10]], dtype=torch.long)],
        ),
    )
    embeds = runner.build_prefill_input_embeds(
        SimpleNamespace(input_ids=torch.tensor([10, 11, 12])),
        [prefill_request],
    )
    assert torch.equal(embeds[0], torch.tensor([1007.0, 1009.0]))
    assert torch.equal(embeds[1], torch.tensor([11.0, 11.0]))

    active = SchedulerRequest(
        request_id="active",
        data=SimpleNamespace(
            req=FakeFishReq(inflight_middle_chunks=0),
            output_codes=[],
            previous_semantic_tokens=[],
            semantic_history_tokens=None,
            semantic_history_count=0,
            last_codebook_values=None,
            latest_stream_code_chunk=None,
        ),
    )
    runner.collect_step_outputs(SimpleNamespace(next_token_ids=None), [active])
    assert len(active.data.output_codes) == 1
    assert torch.equal(active.data.last_codebook_values, torch.tensor([1, 2]))
    assert torch.equal(
        active.data.latest_stream_code_chunk,
        active.data.output_codes[0],
    )
    assert active.data.previous_semantic_tokens == [201]
    assert active.data.semantic_history_count == 1
    assert torch.equal(
        active.data.semantic_history_tokens,
        torch.tensor([201, 0, 0, 0], dtype=torch.long),
    )


def make_s2pro_request(
    request_id: str, *, inflight_middle_chunks: int = 0
) -> SchedulerRequest:
    req = FakeFishReq(inflight_middle_chunks=inflight_middle_chunks)
    return SchedulerRequest(
        request_id=request_id,
        data=SimpleNamespace(
            req=req,
            output_codes=[],
            previous_semantic_tokens=[],
            semantic_history_tokens=None,
            semantic_history_count=0,
            last_codebook_values=None,
            latest_stream_code_chunk=None,
            max_new_tokens=2048,
            temperature=0.8,
            top_p=0.8,
            top_k=30,
            repetition_penalty=1.1,
            ras_temperature=1.0,
            ras_top_p=0.9,
        ),
    )


def collect_s2pro_step(
    requests: list[SchedulerRequest],
    code_rows: list[list[int]],
    *,
    rep_history_len: int | None = None,
) -> SimpleNamespace:
    result = SimpleNamespace(next_token_ids=None)
    output_codes = torch.tensor(code_rows, dtype=torch.long)
    collect_s2pro_step_outputs(
        result,
        requests,
        output_codes=output_codes,
        output_semantic_ids=output_codes[:, 0].clone(),
        im_end_token_id=IM_END_TOKEN_ID,
        rep_history_len=rep_history_len,
    )
    return result


def test_fish_s2pro_audio_timestep_updates_audio_and_stream_state() -> None:
    request = make_s2pro_request("req-audio")
    data = request.data

    result = collect_s2pro_step(
        [request],
        [[SEMANTIC_TOKEN_ID, 11, 22]],
        rep_history_len=4,
    )

    assert int(result.next_token_ids[0].item()) == SEMANTIC_TOKEN_ID
    assert len(data.output_codes) == 1
    assert torch.equal(
        data.output_codes[0],
        torch.tensor([[SEMANTIC_TOKEN_ID], [11], [22]], dtype=torch.long),
    )
    assert torch.equal(data.latest_stream_code_chunk, data.output_codes[0])
    assert data.previous_semantic_tokens == [SEMANTIC_TOKEN_ID]
    assert data.semantic_history_count == 1
    assert torch.equal(
        data.semantic_history_tokens,
        torch.tensor([SEMANTIC_TOKEN_ID, 0, 0, 0], dtype=torch.long),
    )
    assert torch.equal(data.last_codebook_values, torch.tensor([11, 22]))


def test_fish_s2pro_before_decode_uses_gpu_history_buffer() -> None:
    req = FakeFishReq()
    data = S2ProSGLangRequestData(input_ids=torch.tensor([], dtype=torch.long), req=req)
    data.previous_semantic_tokens = [9999]
    data.semantic_history_tokens = torch.tensor(
        [SEMANTIC_TOKEN_ID + 1, SEMANTIC_TOKEN_ID + 2, 0, 0],
        dtype=torch.long,
    )
    data.semantic_history_count = 2
    data.last_codebook_values = torch.tensor([11, 22], dtype=torch.long)
    data.temperature = 0.6
    data.top_p = 0.7
    data.top_k = 8
    data.repetition_penalty = 1.3
    data.ras_temperature = 0.4
    data.ras_top_p = 0.5
    request = SchedulerRequest(request_id="req-history", data=data)

    runner = object.__new__(FishS2ProModelRunner)
    runner.semantic_begin_id = SEMANTIC_TOKEN_ID
    runner.semantic_end_id = SEMANTIC_TOKEN_ID + 10
    runner.model = SimpleNamespace(
        rep_history_len=4,
        vq_mask=torch.zeros(1, dtype=torch.bool),
        sampling_temperature=torch.zeros(1),
        sampling_top_p=torch.zeros(1),
        sampling_top_k=torch.zeros(1, dtype=torch.long),
        sampling_rep_penalty=torch.zeros(1),
        sampling_seeds=torch.full((1,), -1, dtype=torch.long),
        step_count=torch.zeros(1, dtype=torch.long),
        ras_temperature=torch.zeros(1),
        ras_top_p=torch.zeros(1),
        prev_tokens=torch.zeros(1, 4, dtype=torch.long),
        prev_token_count=torch.zeros(1, dtype=torch.long),
        vq_codes=torch.zeros(1, 2, dtype=torch.long),
    )
    forward_batch = SimpleNamespace(input_ids=torch.tensor([SEMANTIC_TOKEN_ID]))

    runner.before_decode(forward_batch, None, [request])

    assert torch.equal(
        runner.model.prev_tokens[0],
        torch.tensor([SEMANTIC_TOKEN_ID + 1, SEMANTIC_TOKEN_ID + 2, 0, 0]),
    )
    assert int(runner.model.prev_token_count[0].item()) == 2
    assert torch.equal(runner.model.vq_codes[0], torch.tensor([11, 22]))
    assert bool(runner.model.vq_mask[0])
    assert torch.allclose(runner.model.sampling_temperature, torch.tensor([0.6]))
    assert torch.allclose(runner.model.sampling_top_p, torch.tensor([0.7]))
    assert int(runner.model.sampling_top_k[0].item()) == 8
    assert torch.allclose(runner.model.sampling_rep_penalty, torch.tensor([1.3]))
    assert torch.allclose(runner.model.ras_temperature, torch.tensor([0.4]))
    assert torch.allclose(runner.model.ras_top_p, torch.tensor([0.5]))


def test_fish_s2pro_before_prefill_syncs_decode_state() -> None:
    first = S2ProSGLangRequestData(
        input_ids=torch.tensor([], dtype=torch.long),
        req=FakeFishReq(extend_len=1),
    )
    first.temperature = 0.55
    first.top_p = 0.65
    first.top_k = 6
    first.repetition_penalty = 1.25
    first.ras_temperature = 0.35
    first.ras_top_p = 0.45

    second = S2ProSGLangRequestData(
        input_ids=torch.tensor([], dtype=torch.long),
        req=FakeFishReq(extend_len=1),
    )
    second.temperature = 0.75
    second.top_p = 0.85
    second.top_k = 12
    second.repetition_penalty = 1.05
    second.ras_temperature = 0.25
    second.ras_top_p = 0.95
    second.semantic_history_tokens = torch.tensor(
        [SEMANTIC_TOKEN_ID + 1, SEMANTIC_TOKEN_ID + 2, 0, 0],
        dtype=torch.long,
    )
    second.semantic_history_count = 2

    runner = object.__new__(FishS2ProModelRunner)

    def embed(input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 2)

    runner.model = SimpleNamespace(
        get_embed_tokens=lambda: embed,
        audio_decoder=SimpleNamespace(
            embed_text_dim=lambda embeds, parts, mask: embeds
        ),
        rep_history_len=4,
        sampling_temperature=torch.zeros(2),
        sampling_top_p=torch.zeros(2),
        sampling_top_k=torch.zeros(2, dtype=torch.long),
        sampling_rep_penalty=torch.zeros(2),
        sampling_seeds=torch.full((2,), -1, dtype=torch.long),
        step_count=torch.zeros(2, dtype=torch.long),
        ras_temperature=torch.zeros(2),
        ras_top_p=torch.zeros(2),
        prev_tokens=torch.full((2, 4), 999, dtype=torch.long),
        prev_token_count=torch.full((2,), 99, dtype=torch.long),
    )
    forward_batch = SimpleNamespace(input_ids=torch.tensor([10, 11]))

    runner.before_prefill(
        forward_batch,
        None,
        [
            SchedulerRequest(request_id="req-first", data=first),
            SchedulerRequest(request_id="req-second", data=second),
        ],
    )

    assert hasattr(forward_batch, "input_embeds")
    assert torch.equal(runner.model.prev_tokens[0], torch.zeros(4, dtype=torch.long))
    assert int(runner.model.prev_token_count[0].item()) == 0
    assert torch.equal(
        runner.model.prev_tokens[1],
        torch.tensor([SEMANTIC_TOKEN_ID + 1, SEMANTIC_TOKEN_ID + 2, 0, 0]),
    )
    assert int(runner.model.prev_token_count[1].item()) == 2
    assert runner.model.sampling_top_k.tolist() == [6, 12]
    assert torch.allclose(
        runner.model.sampling_temperature,
        torch.tensor([0.55, 0.75]),
    )
    assert torch.allclose(runner.model.sampling_top_p, torch.tensor([0.65, 0.85]))
    assert torch.allclose(
        runner.model.sampling_rep_penalty,
        torch.tensor([1.25, 1.05]),
    )
    assert torch.allclose(runner.model.ras_temperature, torch.tensor([0.35, 0.25]))
    assert torch.allclose(runner.model.ras_top_p, torch.tensor([0.45, 0.95]))


def test_fish_s2pro_setup_vq_decode_allocates_sampling_state() -> None:
    model = SimpleNamespace(
        vocab_size=80,
        embed_tokens=SimpleNamespace(weight=torch.empty(1, device="cpu")),
    )
    audio_decoder = SimpleNamespace(
        codebook_embeddings=torch.nn.Embedding(16, 4),
        codebook_offsets=torch.tensor([0, 8], dtype=torch.long),
    )

    S2ProSGLangTextModel.setup_vq_decode(
        model,
        audio_decoder,
        num_codebooks=2,
        codebook_size=8,
        semantic_begin_id=10,
        semantic_end_id=20,
        im_end_token_id=30,
        max_batch_size=3,
        rep_history_len=5,
    )

    assert model.rep_history_len == 5
    assert model.prev_tokens.shape == (3, 5)
    assert model.prev_token_count.shape == (3,)
    assert model.sampling_temperature.shape == (3,)
    assert model.sampling_top_p.shape == (3,)
    assert model.sampling_top_k.tolist() == [30, 30, 30]
    assert model.sampling_rep_penalty.shape == (3,)
    assert model.ras_temperature.shape == (3,)
    assert model.ras_top_p.shape == (3,)
    assert model.rep_positions.tolist() == [0, 1, 2, 3, 4]
    assert model.top_k_positions.shape == (30,)
    assert model.vq_ready


def test_fish_s2pro_decode_codebooks_keeps_eos_out_of_audio_embedding(
    monkeypatch,
) -> None:
    # multinomial_with_seed is a GPU-only Triton kernel; this CPU test only
    # exercises EOS handling, and all rows are unseeded, so stub it out.
    def fake_multinomial_with_seed(logprobs, seeds, pos):
        del seeds, pos
        assert torch.all(logprobs <= 0)
        assert torch.isneginf(logprobs).any()
        return logprobs.argmax(-1, keepdim=True)

    decode_codebooks = S2ProSGLangTextModel.decode_codebooks.__wrapped__
    monkeypatch.setitem(
        decode_codebooks.__globals__,
        "multinomial_with_seed",
        fake_multinomial_with_seed,
    )

    class AudioDecoder:
        def __init__(self) -> None:
            self.seen_embedding_ids: list[torch.Tensor] = []

        def reset_caches(self) -> None:
            pass

        def project_in(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

        def forward_kvcached(
            self,
            hidden_states: torch.Tensor,
            *,
            codebook_idx: int,
        ) -> torch.Tensor:
            del hidden_states, codebook_idx
            return torch.zeros(1, 1, 8)

        def embeddings(self, ids: torch.Tensor) -> torch.Tensor:
            assert int(ids.max().item()) < 8
            self.seen_embedding_ids.append(ids.detach().clone())
            return torch.zeros(ids.shape[0], 4)

    audio_decoder = AudioDecoder()
    model = SimpleNamespace(
        semantic_bias=torch.full((40,), -float("inf")),
        prev_token_count=torch.zeros(1, dtype=torch.long),
        ras_range=torch.arange(4, 0, -1),
        prev_tokens=torch.zeros(1, 4, dtype=torch.long),
        ras_temperature=torch.ones(1),
        sampling_temperature=torch.ones(1),
        ras_top_p=torch.ones(1),
        sampling_top_p=torch.ones(1),
        sampling_rep_penalty=torch.ones(1),
        sampling_seeds=torch.full((1,), -1, dtype=torch.long),
        step_count=torch.zeros(1, dtype=torch.long),
        rep_positions=torch.arange(4),
        graph_top_k=30,
        sampling_top_k=torch.full((1,), 30, dtype=torch.long),
        top_k_positions=torch.arange(30),
        audio_decoder=audio_decoder,
        semantic_begin_id=10,
        im_end_token_id=30,
        codebook_size=8,
        num_codebooks=2,
        output_codes=torch.zeros(1, 3, dtype=torch.long),
        output_semantic_ids=torch.zeros(1, dtype=torch.long),
    )
    model.semantic_bias[10:18] = 0.0
    model.semantic_bias[30] = 0.0
    logits = torch.full((1, 40), -1_000_000.0)
    logits[0, 30] = 1_000_000.0

    S2ProSGLangTextModel.decode_codebooks(
        model,
        logits,
        torch.zeros(1, 4),
    )

    assert int(model.output_semantic_ids[0].item()) == 30
    assert int(model.output_codes[0, 0].item()) == 30
    assert int(audio_decoder.seen_embedding_ids[0][0].item()) == 0


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="multinomial_with_seed needs CUDA"
)
def test_fish_s2pro_seeded_sampler_preserves_probability_distribution() -> None:
    batch = 20_000
    device = torch.device("cuda")
    semantic_begin_id = 10
    im_end_token_id = 39
    vocab_size = 40

    class AudioDecoder:
        def reset_caches(self) -> None:
            pass

        def project_in(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

        def forward_kvcached(
            self,
            hidden_states: torch.Tensor,
            *,
            codebook_idx: int,
        ) -> torch.Tensor:
            del codebook_idx
            return torch.zeros(
                hidden_states.shape[0], 1, 8, device=hidden_states.device
            )

        def embeddings(self, ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros(ids.shape[0], 4, device=ids.device)

    semantic_bias = torch.full((vocab_size,), -float("inf"), device=device)
    semantic_bias[semantic_begin_id : semantic_begin_id + 2] = 0.0
    semantic_bias[im_end_token_id] = 0.0
    logits = torch.full((batch, vocab_size), -float("inf"), device=device)
    logits[:, semantic_begin_id] = torch.log(torch.tensor(0.9, device=device))
    logits[:, semantic_begin_id + 1] = torch.log(torch.tensor(0.1, device=device))

    model = SimpleNamespace(
        semantic_bias=semantic_bias,
        prev_token_count=torch.zeros(batch, dtype=torch.long, device=device),
        ras_range=torch.arange(4, 0, -1, device=device),
        prev_tokens=torch.zeros(batch, 4, dtype=torch.long, device=device),
        ras_temperature=torch.ones(batch, device=device),
        sampling_temperature=torch.ones(batch, device=device),
        ras_top_p=torch.ones(batch, device=device),
        sampling_top_p=torch.ones(batch, device=device),
        sampling_rep_penalty=torch.ones(batch, device=device),
        sampling_seeds=torch.arange(1, batch + 1, dtype=torch.long, device=device),
        step_count=torch.zeros(batch, dtype=torch.long, device=device),
        rep_positions=torch.arange(4, device=device),
        graph_top_k=30,
        sampling_top_k=torch.full((batch,), 2, dtype=torch.long, device=device),
        top_k_positions=torch.arange(30, device=device),
        audio_decoder=AudioDecoder(),
        semantic_begin_id=semantic_begin_id,
        im_end_token_id=im_end_token_id,
        codebook_size=8,
        num_codebooks=1,
        output_codes=torch.zeros(batch, 2, dtype=torch.long, device=device),
        output_semantic_ids=torch.zeros(batch, dtype=torch.long, device=device),
    )

    S2ProSGLangTextModel.decode_codebooks(
        model,
        logits,
        torch.zeros(batch, 4, device=device),
    )

    token0_rate = (model.output_semantic_ids == semantic_begin_id).float().mean().item()
    assert 0.87 < token0_rate < 0.93


def test_fish_s2pro_terminal_im_end_is_not_audio_codebook_frame() -> None:
    request = make_s2pro_request("req-terminal")

    collect_s2pro_step([request], [[SEMANTIC_TOKEN_ID, 11, 22]])
    request.data.latest_stream_code_chunk = None

    result = collect_s2pro_step([request], [[IM_END_TOKEN_ID, 33, 44]])

    assert int(result.next_token_ids[0].item()) == IM_END_TOKEN_ID
    assert len(request.data.output_codes) == 1
    assert torch.equal(
        request.data.output_codes[0],
        torch.tensor([[SEMANTIC_TOKEN_ID], [11], [22]], dtype=torch.long),
    )
    assert request.data.latest_stream_code_chunk is None
    assert request.data.previous_semantic_tokens == [SEMANTIC_TOKEN_ID]
    assert torch.equal(request.data.last_codebook_values, torch.tensor([11, 22]))


def test_fish_s2pro_mixed_batch_keeps_terminal_and_audio_state_separate() -> None:
    audio_request = make_s2pro_request("req-audio")
    terminal_request = make_s2pro_request("req-terminal")

    collect_s2pro_step(
        [audio_request, terminal_request],
        [
            [SEMANTIC_TOKEN_ID, 11, 22],
            [IM_END_TOKEN_ID, 33, 44],
        ],
    )

    assert len(audio_request.data.output_codes) == 1
    assert terminal_request.data.output_codes == []
    assert torch.equal(
        audio_request.data.latest_stream_code_chunk,
        audio_request.data.output_codes[0],
    )
    assert terminal_request.data.latest_stream_code_chunk is None
    assert torch.equal(
        audio_request.data.output_codes[0],
        torch.tensor([[SEMANTIC_TOKEN_ID], [11], [22]], dtype=torch.long),
    )
    assert audio_request.data.previous_semantic_tokens == [SEMANTIC_TOKEN_ID]
    assert terminal_request.data.previous_semantic_tokens == []
    assert torch.equal(audio_request.data.last_codebook_values, torch.tensor([11, 22]))
    assert terminal_request.data.last_codebook_values is None


def test_fish_s2pro_chunked_step_does_not_mutate_decode_state() -> None:
    request = make_s2pro_request("req-chunked", inflight_middle_chunks=1)

    collect_s2pro_step([request], [[SEMANTIC_TOKEN_ID, 11, 22]])

    assert request.data.output_codes == []
    assert request.data.latest_stream_code_chunk is None
    assert request.data.previous_semantic_tokens == []
    assert request.data.last_codebook_values is None


def test_fish_tts_request_builder_maps_finish_contract_onto_req() -> None:
    request_builder, _, _ = make_tts_scheduler_adapters(
        tokenizer=FakeFishTokenizer(),
        max_new_tokens_cap=4,
    )
    payload = make_s2pro_payload(
        make_s2pro_state(max_new_tokens=6), request_id="req-contract"
    )

    data = request_builder(payload)

    assert data.req.rid == "req-contract"
    assert data.req.sampling_params.repetition_penalty == 1.0
    assert data.repetition_penalty == 1.05
    assert data.req.sampling_params.stop_token_ids == {99}
    assert data.req.eos_token_ids == {99}
    assert data.req.sampling_params.max_new_tokens == 4
    assert data.max_new_tokens == 4
    assert data.req.vocab_size == 640
    assert data.engine_start_s > 0.0
    assert data.stage_payload is payload

    same_ref = request_builder(
        make_s2pro_payload(make_s2pro_state(max_new_tokens=6), request_id="req-again")
    )
    assert data.req.extra_key == same_ref.req.extra_key is not None

    other_cb1 = request_builder(
        make_s2pro_payload(
            make_s2pro_state(vq_parts=[torch.tensor([[1], [3]], dtype=torch.long)]),
            request_id="req-other-ref",
        )
    )
    assert other_cb1.req.extra_key != data.req.extra_key

    no_ref = request_builder(
        make_s2pro_payload(make_s2pro_state(vq_parts=None), request_id="req-zero-shot")
    )
    assert no_ref.req.extra_key is None


def test_fish_tts_request_builder_clamps_budget_to_context() -> None:
    request_builder, _, _ = make_tts_scheduler_adapters(
        tokenizer=FakeFishTokenizer(), context_length=8
    )
    data = request_builder(
        make_s2pro_payload(make_s2pro_state(max_new_tokens=6), request_id="req-clamp")
    )
    # note (Gaokai): 8 (context) - 1 - 3 (prompt ids)
    assert data.req.sampling_params.max_new_tokens == 4


def test_fish_tts_result_adapter_maps_finish_reason_and_engine_time() -> None:
    request_builder, result_adapter, _ = make_tts_scheduler_adapters(
        tokenizer=FakeFishTokenizer()
    )
    data = request_builder(make_s2pro_payload(request_id="req-result"))
    data.output_codes = [torch.tensor([[100], [1], [2]], dtype=torch.long)]
    data.finish_reason = "length"

    payload = result_adapter(data)

    assert payload.data["finish_reason"] == "length"
    assert payload.data["engine_time_s"] > 0.0
    assert payload.data["completion_tokens"] == 1
    assert payload.data["prompt_tokens"] == 3

    data.finish_reason = None
    assert result_adapter(data).data["finish_reason"] == "stop"


def test_fish_req_hits_max_new_tokens_and_scheduler_reports_length(
    monkeypatch,
) -> None:
    """Budget exhaustion runs the upstream length path end-to-end: the Req
    finishes with FINISH_LENGTH and the scheduler maps it onto the terminal
    Fish payload."""
    request_builder, result_adapter, _ = make_tts_scheduler_adapters(
        tokenizer=FakeFishTokenizer()
    )
    data = request_builder(
        make_s2pro_payload(make_s2pro_state(max_new_tokens=2), request_id="req-length")
    )
    req = data.req
    req.omni_data = data
    req._omni_terminal_claimed = False  # noqa: leading-underscore  # production name

    for value in (200, 201):
        assert not req.finished()
        req.output_ids.append(value)
        data.output_codes.append(torch.tensor([[value], [5]], dtype=torch.long))
        req.update_finish_state()
    assert req.finished()

    scheduler = object.__new__(OmniScheduler)
    scheduler.request_admission_lock = threading.RLock()
    scheduler.outbox = Queue()
    scheduler.aborted_request_ids = set()
    scheduler.completed_request_ids = {}
    scheduler.pending_stream_ingress = {}
    scheduler.request_finished_callback = None
    scheduler.first_emit_done = set()
    scheduler.prefill_start_done = set()
    scheduler.prefill_end_done = set()
    scheduler.result_adapter = result_adapter
    scheduler.model_runner = None
    scheduler.stream_output_builder = None
    monkeypatch.setattr(
        omni_scheduler_module,
        "get_serving",
        lambda: SimpleNamespace(weight_version=None),
    )

    scheduler.stream_output([req])

    message = scheduler.outbox.get_nowait()
    assert message.type == "result"
    assert message.data.data["finish_reason"] == "length"
    assert message.data.data["completion_tokens"] == 2


def test_fish_tts_result_adapter_raises_for_empty_codes() -> None:
    request_builder, result_adapter, _ = make_tts_scheduler_adapters(
        tokenizer=FakeFishTokenizer()
    )
    data = request_builder(make_s2pro_payload(request_id="req-empty"))

    with pytest.raises(ValueError, match="S2-Pro generated no audio codec tokens"):
        result_adapter(data)


def test_fish_tts_stream_output_builder_gates_and_clears_chunks() -> None:
    _, _, stream_output_builder = make_tts_scheduler_adapters(
        tokenizer=FakeFishTokenizer()
    )
    codes = torch.full((11, 1), 7, dtype=torch.long)

    stream_data = SimpleNamespace(
        stage_payload=make_s2pro_payload(request_id="stream", params={"stream": True}),
        latest_stream_code_chunk=codes,
    )
    messages = stream_output_builder("stream", stream_data, None)
    assert len(messages) == 1
    assert messages[0].type == "stream"
    assert messages[0].target == "vocoder"
    assert messages[0].metadata == {"modality": "audio_codes"}
    assert messages[0].data is codes
    assert stream_data.latest_stream_code_chunk is None
    assert stream_output_builder("stream", stream_data, None) == []

    non_stream_data = SimpleNamespace(
        stage_payload=make_s2pro_payload(
            request_id="non-stream", params={"stream": False}
        ),
        latest_stream_code_chunk=torch.full((11, 1), 8, dtype=torch.long),
    )
    assert stream_output_builder("non-stream", non_stream_data, None) == []


def test_fish_lookahead_is_disabled_by_default() -> None:
    """History-free generic sampling must not bypass the Fish opt-in flag."""
    runner = object.__new__(FishS2ProModelRunner)
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            min_new_tokens=0,
        ),
        custom_logit_processor=None,
    )
    assert runner.lookahead_eligible(SimpleNamespace(reqs=[req])) is False


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
            ],
        ),
    ],
)
def test_fish_async_snapshots_survive_next_launch_and_skip_inactive_rows(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """The next launch must not overwrite an unresolved frame or append overruns."""
    requests = [make_s2pro_request(str(index)) for index in range(5)]
    for index, request in enumerate(requests):
        request.data.req.finished = lambda index=index: index == 3
        request.data.req.is_retracted = index == 4
    requests[2].data.req.inflight_middle_chunks = 1
    code_rows = torch.arange(55, device=device).reshape(5, 11)
    code_rows[:, 0] = SEMANTIC_TOKEN_ID
    code_rows[1, 0] = IM_END_TOKEN_ID
    expected_codes = code_rows.cpu().clone()
    runner = object.__new__(FishS2ProModelRunner)
    runner.model = SimpleNamespace(output_codes=code_rows, rep_history_len=4)
    runner.im_end_token_id = IM_END_TOKEN_ID
    runner.staging_slot = 0
    runner.host_staging_buffers = []
    if device == "cpu":
        monkeypatch.setattr(
            runner,
            "next_host_staging",
            lambda shape, dtype: torch.empty(shape, dtype=dtype),
        )
    else:
        pass
    first_result = SimpleNamespace(next_token_ids=None)
    first_snapshot = runner.post_decode_launch(first_result, None, requests)
    runner.model.output_codes[0, 0].add_(1)
    runner.model.output_codes[:, 1:].add_(100)
    second_result = SimpleNamespace(next_token_ids=None)
    second_snapshot = runner.post_decode_launch(second_result, None, requests)
    if device == "cuda":
        torch.cuda.synchronize()
    else:
        pass

    runner.post_decode_resolve(first_snapshot, first_result, None, None, requests)

    assert first_result.next_token_ids.tolist() == expected_codes[:, 0].tolist()
    torch.testing.assert_close(
        requests[0].data.output_codes[0].cpu(), expected_codes[0].unsqueeze(1)
    )
    assert requests[0].data.semantic_history_count == 1
    for request in requests[1:]:
        assert request.data.output_codes == []
        assert request.data.latest_stream_code_chunk is None
        assert request.data.semantic_history_count == 0
        assert request.data.last_codebook_values is None

    runner.post_decode_resolve(second_snapshot, second_result, None, None, requests)
    assert len(requests[0].data.output_codes) == 2
    assert requests[0].data.semantic_history_count == 2
    assert requests[0].data.previous_semantic_tokens == [
        SEMANTIC_TOKEN_ID,
        SEMANTIC_TOKEN_ID + 1,
    ]
    torch.testing.assert_close(
        requests[0].data.last_codebook_values.cpu(), expected_codes[0, 1:] + 100
    )


class FishAsyncAudioDecoder:
    """Small deterministic fast head for testing the real decode state lifecycle."""

    def __init__(self, device: str) -> None:
        self.codebook_embeddings = torch.nn.Embedding(80, 4, device=device)
        self.codebook_offsets = torch.arange(10, device=device) * 8

    def reset_caches(self) -> None:
        pass

    def project_in(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.unsqueeze(1).expand(-1, 4).float()

    def forward_kvcached(
        self, hidden_states: torch.Tensor, *, codebook_idx: int
    ) -> torch.Tensor:
        token_ids = torch.arange(8, device=hidden_states.device)
        return (
            -(token_ids - codebook_idx % 8)
            .float()
            .square()
            .expand(hidden_states.shape[0], 1, -1)
        )


def make_fish_async_runner(
    device: str, *, enabled: bool
) -> tuple[FishS2ProModelRunner, list[SchedulerRequest]]:
    runner = object.__new__(FishS2ProModelRunner)
    runner.model = SimpleNamespace(
        vocab_size=640,
        embed_tokens=SimpleNamespace(weight=torch.empty(1, device=device)),
    )
    S2ProSGLangTextModel.setup_vq_decode(
        runner.model,
        FishAsyncAudioDecoder(device),
        num_codebooks=10,
        codebook_size=8,
        semantic_begin_id=200,
        semantic_end_id=207,
        im_end_token_id=99,
        max_batch_size=3,
        rep_history_len=4,
    )
    runner.model.async_decode_state_enabled = enabled
    runner.async_enabled = enabled
    runner.semantic_begin_id = 200
    runner.semantic_end_id = 207
    runner.im_end_token_id = 99
    runner.last_decode_request_ids = None
    request_builder, _, _ = make_tts_scheduler_adapters(tokenizer=FakeFishTokenizer())
    requests = [
        SchedulerRequest(
            request_id=f"request-{index}",
            data=request_builder(
                make_s2pro_payload(
                    make_s2pro_state(vq_parts=None, max_new_tokens=32, seed=42),
                    request_id=f"request-{index}",
                )
            ),
        )
        for index in range(2)
    ]
    return runner, requests


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
            ],
        ),
    ],
)
def test_fish_async_device_state_matches_sync_after_history_wrap(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """Decode without host collect must retain sync history, steps, codes and EOS."""
    if device == "cpu":
        monkeypatch.setitem(
            S2ProSGLangTextModel.decode_codebooks.__wrapped__.__globals__,
            "multinomial_with_seed",
            lambda logprobs, seeds, pos: logprobs.argmax(-1, keepdim=True),
        )
    else:
        pass
    synchronous, requests = make_fish_async_runner(device, enabled=False)
    asynchronous, _ = make_fish_async_runner(device, enabled=True)
    requests[1].data.semantic_history_tokens = torch.tensor(
        [205, 206, 0, 0], device=device
    )
    requests[1].data.semantic_history_count = 2
    logits = torch.full((2, 640), -float("inf"), device=device)
    logits[0, 200:208] = torch.arange(8, device=device).float() / 8
    logits[1, 99] = 1.0
    hidden_states = torch.zeros(2, 4, device=device)
    asynchronous.sync_decode_state(requests)
    if device == "cuda":
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            S2ProSGLangTextModel.decode_codebooks(
                asynchronous.model, logits, hidden_states
            )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            S2ProSGLangTextModel.decode_codebooks(
                asynchronous.model, logits, hidden_states
            )
        asynchronous.sync_decode_state(requests)
        asynchronous.model.vq_codes.zero_()
    else:
        pass

    for step in range(8):
        synchronous.sync_decode_state(requests)
        S2ProSGLangTextModel.decode_codebooks(synchronous.model, logits, hidden_states)
        synchronous.collect_step_outputs(SimpleNamespace(next_token_ids=None), requests)
        if device == "cuda":
            graph.replay()
        else:
            S2ProSGLangTextModel.decode_codebooks(
                asynchronous.model, logits, hidden_states
            )
        torch.testing.assert_close(
            asynchronous.model.output_codes, synchronous.model.output_codes
        )
        torch.testing.assert_close(
            asynchronous.model.vq_codes[:2], synchronous.model.output_codes[:2, 1:]
        )
        torch.testing.assert_close(
            asynchronous.model.prev_tokens[0], requests[0].data.semantic_history_tokens
        )
        assert asynchronous.model.prev_token_count.tolist() == [min(step + 1, 4), 2, 0]
        assert asynchronous.model.step_count.tolist() == [step + 1, 2, 0]
        torch.testing.assert_close(
            asynchronous.model.prev_tokens[1], requests[1].data.semantic_history_tokens
        )
        assert asynchronous.model.prev_tokens[2].count_nonzero().item() == 0
        assert asynchronous.model.vq_codes[2].count_nonzero().item() == 0


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fish_async_stable_rows_keep_device_state_and_reordered_rows_resync() -> None:
    """Stable rows must not reload stale host state; reordered rows must reload it."""
    runner, requests = make_fish_async_runner("cuda", enabled=True)
    forward_batch = SimpleNamespace(input_ids=torch.tensor([200, 200], device="cuda"))
    batch = SimpleNamespace(reqs=[request.data.req for request in requests])
    assert not runner.lookahead_eligible(batch)
    runner.before_decode(forward_batch, batch, requests)
    assert runner.lookahead_eligible(batch)
    runner.model.vq_codes[:2].fill_(7)
    runner.model.prev_tokens[:2].fill_(201)
    runner.model.step_count[:2].fill_(9)

    runner.before_decode(forward_batch, batch, requests, is_lookahead=True)

    assert runner.model.vq_codes[:2].eq(7).all().item()
    assert runner.model.prev_tokens[:2].eq(201).all().item()
    assert runner.model.step_count[:2].tolist() == [9, 9]
    for index, request in enumerate(requests):
        request.data.last_codebook_values = torch.full((10,), index + 1)
        request.data.semantic_history_tokens = torch.full((4,), 202 + index)
        request.data.semantic_history_count = 6 + index
    requests.reverse()
    batch.reqs.reverse()
    assert not runner.lookahead_eligible(batch)

    runner.before_decode(forward_batch, batch, requests)

    assert runner.model.vq_codes[:2].tolist() == [[2] * 10, [1] * 10]
    assert runner.model.prev_tokens[:2].tolist() == [[203] * 4, [202] * 4]
    assert runner.model.prev_token_count[:2].tolist() == [4, 4]
    assert runner.model.step_count[:2].tolist() == [7, 6]
    assert runner.lookahead_eligible(batch)
