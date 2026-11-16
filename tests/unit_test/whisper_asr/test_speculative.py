# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferIndicesUpdaterDecode,
    FlashInferIndicesUpdaterPrefill,
)
from sglang.srt.mem_cache.kv_loc_plan import IdSpaceKind
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from torch import nn
from transformers import WhisperConfig

from sglang_omni.models.whisper_asr.sglang_model import WhisperEncoder

speculative = pytest.importorskip("sglang_omni.models.whisper_asr.speculative")


def tiny_config(**overrides) -> WhisperConfig:
    return WhisperConfig(
        d_model=8,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        vocab_size=32,
        max_target_positions=16,
        max_source_positions=8,
        pad_token_id=0,
        **overrides,
    )


def test_materialized_draft_names_the_decoder_only_class(tmp_path) -> None:
    source = tmp_path / "draft"
    source.mkdir()
    tiny_config().to_json_file(source / "config.json")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer.json").write_text("{}")
    (source / "preprocessor_config.json").write_text("{}")

    checkpoint = speculative.materialize_draft_checkpoint(str(source), None)
    with open(os.path.join(checkpoint.name, "config.json")) as config_file:
        config = json.load(config_file)

    # note (wilsonzheng0327): the second name keeps SGLang's encoder-decoder
    # handling for the draft.
    assert config["architectures"] == [
        speculative.DRAFT_ARCHITECTURE,
        "WhisperForConditionalGeneration",
    ]
    assert config["encoder_layers"] == 0
    assert sorted(os.listdir(checkpoint.name)) == [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    ]
    checkpoint.cleanup()


def test_materialize_rejects_a_draft_that_is_not_whisper(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    with pytest.raises(ValueError, match="not a Whisper checkpoint"):
        speculative.materialize_draft_checkpoint(str(tmp_path), None)


def test_draft_loads_decoder_weights_and_skips_the_encoder(monkeypatch) -> None:
    monkeypatch.setattr(speculative, "LogitsProcessor", lambda config: None)
    config = tiny_config()
    draft = speculative.WhisperDraftForCausalLM(config)
    assert not hasattr(draft.model, "encoder")
    weight = torch.full((config.d_model, config.d_model), 0.5)
    draft.load_weights(
        [
            ("model.encoder.layers.0.self_attn.q_proj.weight", torch.zeros(1)),
            ("model.decoder.layers.0.self_attn.k_proj.weight", weight),
            ("model.decoder.layers.0.encoder_attn.v_proj.weight", weight),
        ]
    )
    layer = draft.model.decoder.layers[0]
    d_model = config.d_model
    assert torch.equal(layer.self_attn.qkv_proj.weight[d_model : 2 * d_model], weight)
    assert torch.equal(layer.encoder_attn.kv_proj.weight[d_model:], weight)
    with pytest.raises(RuntimeError, match="no encoder of its own"):
        draft.run_encoder(torch.zeros(1))


class ReqToTokenTranslator:
    """Packs each row's KV slots from a req_to_token table, as the backend's translator does on the GPU."""

    def __init__(self, req_to_token: torch.Tensor) -> None:
        self.req_to_token = req_to_token

    def pack_read_stream(
        self,
        plan: None,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        indptr: torch.Tensor,
        out: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kind: IdSpaceKind,
    ) -> None:
        for row, pool_index in enumerate(req_pool_indices.tolist()):
            start_column = int(kv_start_idx[row])
            out[indptr[row] : indptr[row + 1]] = self.req_to_token[
                pool_index, start_column : start_column + int(seq_lens[row])
            ]


def planned_kv_slots(
    plan_wrapper: SimpleNamespace, *, kv_indptr_position: int = 1
) -> list[list[int]]:
    """The KV slots of each row in a wrapper's last plan; a prefill plan passes its query indptr first, a decode plan does not."""
    kv_indptr, kv_indices = plan_wrapper.begin_forward.call_args.args[
        kv_indptr_position : kv_indptr_position + 2
    ]
    return [
        kv_indices[kv_indptr[row] : kv_indptr[row + 1]].tolist()
        for row in range(kv_indptr.numel() - 1)
    ]


@pytest.mark.parametrize("padding_rows", [0, 1], ids=["eager", "replay_9_in_10"])
def test_verify_reads_the_encoder_slots_in_cross_attention_and_the_chain_in_self_attention(
    padding_rows: int,
) -> None:
    """Graph replay padded 9 requests to 10 rows, and cross-attention then read past each request's encoder slots into its decoder slots."""
    encoder_len, draft_token_num, max_decoder_len = 4, 3, 9
    req_to_token = torch.arange(10 * 16).view(10, 16)
    decoder_lens = list(range(1, max_decoder_len + 1))
    # note (wilsonzheng0327): the decode graph runner pads with request slot 0,
    # sequence length 1 and encoder length max_source_positions.
    req_pool_indices = torch.tensor(list(range(1, 10)) + [0] * padding_rows)
    seq_lens = torch.tensor(decoder_lens + [1] * padding_rows, dtype=torch.int32)
    encoder_lens = torch.full((9 + padding_rows,), encoder_len, dtype=torch.int32)
    updater = object.__new__(FlashInferIndicesUpdaterPrefill)
    updater.attn_backend = SimpleNamespace(
        kv_index_translator=ReqToTokenTranslator(req_to_token)
    )
    updater.prefill_wrapper_ragged = None
    updater.kv_indptr = updater.qo_indptr = [torch.zeros(11, dtype=torch.int32)] * 2
    updater.kv_last_page_len = torch.ones(10, dtype=torch.int32)
    updater.num_qo_heads = updater.num_kv_heads = updater.head_dim = 1
    updater.q_data_type = updater.data_type = torch.float16
    self_attention, cross_attention = (
        SimpleNamespace(
            begin_forward=Mock(),
            is_cuda_graph_enabled=False,
            _custom_mask_buf=None,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
        for _ in range(2)
    )

    updater.update_cross_attention(
        req_pool_indices,
        seq_lens,
        seq_lens,
        int(seq_lens.sum()),
        None,
        [self_attention, cross_attention],
        False,
        encoder_lens,
        speculative.WhisperVerifyInput(
            draft_token=None,
            positions=None,
            draft_token_num=draft_token_num,
            custom_mask=None,
        ),
        plan=None,
    )

    for row, decoder_len in enumerate(decoder_lens):
        slots = req_to_token[row + 1]
        assert planned_kv_slots(cross_attention)[row] == slots[:encoder_len].tolist()
        assert (
            planned_kv_slots(self_attention)[row]
            == slots[encoder_len : encoder_len + decoder_len + draft_token_num].tolist()
        )


def graph_decode_updater(
    req_to_token: torch.Tensor,
) -> tuple[FlashInferIndicesUpdaterDecode, list[SimpleNamespace]]:
    """A decode updater over graph-mode self-attention and cross-attention wrappers, whose indices buffers the replayed kernels read."""
    updater = object.__new__(FlashInferIndicesUpdaterDecode)
    updater.attn_backend = SimpleNamespace(
        kv_index_translator=ReqToTokenTranslator(req_to_token)
    )
    updater.kv_indptr = [torch.zeros(11, dtype=torch.int32) for _ in range(2)]
    updater.kv_last_page_len = torch.ones(10, dtype=torch.int32)
    updater.num_qo_heads = updater.num_kv_heads = updater.head_dim = 1
    updater.q_data_type = updater.data_type = torch.float16
    updater.update = updater.update_cross_attention
    wrappers = [
        SimpleNamespace(
            begin_forward=Mock(spec=[]),
            is_cuda_graph_enabled=True,
            _paged_kv_indices_buf=torch.zeros(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                64, dtype=torch.int32
            ),
        )
        for _ in range(2)
    ]
    return updater, wrappers


def test_later_draft_steps_keep_the_cross_attention_inputs_a_full_plan_would_write() -> (
    None
):
    """A later step plans self-attention as a full update does, and the first step's cross-attention inputs equal what a full update at the later step writes."""
    req_to_token = torch.arange(10 * 16).view(10, 16)
    req_pool_indices = torch.tensor([3, 1, 4])
    encoder_lens = torch.tensor([4, 2, 5], dtype=torch.int32)
    first_step_lens = torch.tensor([2, 5, 3], dtype=torch.int32)
    later_step_lens = first_step_lens + 1
    updater, wrappers = graph_decode_updater(req_to_token)
    reference_updater, reference_wrappers = graph_decode_updater(req_to_token)

    updater.update_cross_attention(
        first_step_lens,
        first_step_lens,
        int(first_step_lens.sum()),
        wrappers,
        encoder_lens,
        None,
        plan=None,
        req_pool_indices=req_pool_indices,
    )
    speculative.plan_decode_self_attention(
        updater,
        later_step_lens,
        later_step_lens,
        int(later_step_lens.sum()),
        wrappers,
        encoder_lens,
        None,
        plan=None,
        req_pool_indices=req_pool_indices,
    )
    reference_updater.update_cross_attention(
        later_step_lens,
        later_step_lens,
        int(later_step_lens.sum()),
        reference_wrappers,
        encoder_lens,
        None,
        plan=None,
        req_pool_indices=req_pool_indices,
    )

    self_attention, cross_attention = wrappers
    reference_self_attention, reference_cross_attention = reference_wrappers
    assert planned_kv_slots(self_attention, kv_indptr_position=0) == planned_kv_slots(
        reference_self_attention, kv_indptr_position=0
    )
    assert cross_attention.begin_forward.call_count == 1
    assert torch.equal(updater.kv_indptr[1], reference_updater.kv_indptr[1])
    assert torch.equal(
        cross_attention._paged_kv_indices_buf,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        reference_cross_attention._paged_kv_indices_buf,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    )


@pytest.mark.parametrize("should_plan_cross_attention", [True, False])
def test_a_draft_step_plans_cross_attention_only_when_asked_and_puts_the_full_update_back(
    monkeypatch, should_plan_cross_attention: bool
) -> None:
    """A round's first step must plan cross-attention, and a later step must not leave the next round without it."""
    req_pool_indices = torch.tensor([2])
    seq_lens = torch.tensor([5], dtype=torch.int32)
    updater, wrappers = graph_decode_updater(torch.arange(10 * 16).view(10, 16))
    full_update = updater.update
    attention_backend = object.__new__(FlashInferAttnBackend)
    attention_backend.indices_updater_decode = updater
    logits = torch.zeros(1, 32)

    def forward(forward_batch: SimpleNamespace) -> SimpleNamespace:
        updater.update(
            seq_lens,
            seq_lens,
            int(seq_lens.sum()),
            wrappers,
            torch.tensor([4], dtype=torch.int32),
            None,
            plan=None,
            req_pool_indices=req_pool_indices,
        )
        return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=logits))

    monkeypatch.setattr(
        speculative.ForwardBatch, "init_new", lambda step, runner, **options: step
    )
    worker = object.__new__(speculative.WhisperSpeculativeWorker)
    batch = SimpleNamespace(
        reqs=[None],
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
    )

    output = worker.decode_forward(
        batch,
        SimpleNamespace(attn_backend=attention_backend, forward=forward),
        token_ids=torch.tensor([7]),
        offset=3,
        slots=torch.tensor([20]),
        should_plan_cross_attention=should_plan_cross_attention,
    )

    self_attention, cross_attention = wrappers
    assert output is logits
    assert self_attention.begin_forward.call_count == 1
    assert cross_attention.begin_forward.call_count == int(should_plan_cross_attention)
    assert updater.update == full_update


def tiny_target() -> nn.Module:
    target = nn.Module()
    target.model = nn.Module()
    target.model.encoder = WhisperEncoder(tiny_config())
    return target


def test_shared_encoder_accepts_the_same_weights_in_another_half_precision(
    tmp_path,
) -> None:
    target = tiny_target()
    save_file(
        {
            name: dict(target.named_parameters())[name].detach().to(torch.bfloat16)
            for name in speculative.SHARED_ENCODER_TENSOR_NAMES
        },
        str(tmp_path / "model.safetensors"),
    )

    speculative.check_shared_encoder(target, str(tmp_path))


@pytest.mark.parametrize(
    ("draft_names", "shift", "message"),
    [
        (speculative.SHARED_ENCODER_TENSOR_NAMES, 0.5, "differs from the target's"),
        (speculative.SHARED_ENCODER_TENSOR_NAMES[:1], 0.0, "cannot be checked"),
    ],
)
def test_shared_encoder_refuses_a_different_or_unknown_encoder(
    tmp_path, draft_names: tuple[str, ...], shift: float, message: str
) -> None:
    """A draft paired with another encoder reads states it was never trained on."""
    target = tiny_target()
    save_file(
        {
            name: dict(target.named_parameters())[name].detach() + shift
            for name in draft_names
        },
        str(tmp_path / "model.safetensors"),
    )

    with pytest.raises(ValueError, match=message):
        speculative.check_shared_encoder(target, str(tmp_path))


def test_draft_decode_graphs_are_captured_without_torch_compile(monkeypatch) -> None:
    """A compiled draft decode copies its layer's whole K and V buffer on every KV store."""
    flags = SimpleNamespace(capture=SimpleNamespace(enable_torch_compile=True))
    monkeypatch.setattr(speculative, "get_flags", lambda: flags)
    seen = []
    worker = object.__new__(speculative.WhisperDraftWorker)
    worker.draft_worker = SimpleNamespace(
        init_cuda_graphs=lambda *, capture_decode_cuda_graph: seen.append(
            (capture_decode_cuda_graph, flags.capture.enable_torch_compile)
        )
    )

    worker.init_cuda_graphs()

    assert seen == [(True, False)]
    assert flags.capture.enable_torch_compile is True


def chain_batch(
    num_requests: int,
    *,
    greedy: list[bool] | None = None,
    temperature: float = 1.0,
    custom_params: list[dict | None] | None = None,
    output_ids: list[list[int]] | None = None,
) -> SimpleNamespace:
    greedy = greedy if greedy is not None else [False] * num_requests
    custom_params = custom_params or [None] * num_requests
    output_ids = output_ids or [[5]] * num_requests
    return SimpleNamespace(
        sampling_info=SimpleNamespace(
            logit_bias=None,
            is_all_greedy=all(greedy),
            top_ks=torch.tensor([1 if row else TOP_K_ALL for row in greedy]),
            temperatures=torch.tensor(
                [[1.0 if row else temperature] for row in greedy]
            ),
        ),
        reqs=[
            SimpleNamespace(
                sampling_params=SimpleNamespace(custom_params=params), output_ids=ids
            )
            for params, ids in zip(custom_params, output_ids, strict=True)
        ],
    )


def test_sampled_chain_commits_tokens_from_the_target_distribution() -> None:
    """A sampled draft that disagrees with the target must not pull the committed tokens toward it."""
    torch.manual_seed(0)
    num_requests = 200_000
    draft_logits = torch.tensor([2.0, 0.0, -1.0, 1.0])
    target_logits = torch.tensor([[0.0, 1.5, 0.5, -0.5], [1.0, -1.0, 0.0, 2.0]])
    sampler = speculative.ChainSampler(chain_batch(num_requests, temperature=0.8))

    drafts = sampler.draw(draft_logits.expand(num_requests, -1))[:, None]
    num_correct, committed = sampler.verify(
        target_logits.expand(num_requests, -1, -1), drafts
    )

    first = torch.bincount(committed[:, 0], minlength=4).float() / num_requests
    expected_first = torch.softmax(target_logits[0] / 0.8, dim=-1)
    torch.testing.assert_close(first, expected_first, atol=5e-3, rtol=0)
    accepted = num_correct == 1
    bonus = torch.bincount(committed[accepted, 1], minlength=4).float()
    expected_bonus = torch.softmax(target_logits[1] / 0.8, dim=-1)
    torch.testing.assert_close(bonus / bonus.sum(), expected_bonus, atol=5e-3, rtol=0)


def test_greedy_rows_of_a_mixed_chain_keep_argmax_matching() -> None:
    torch.manual_seed(0)
    sampler = speculative.ChainSampler(chain_batch(2, greedy=[True, False]))
    draft_logits = torch.tensor([[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    target_logits = torch.tensor(
        [
            [[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [5.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0]] * 3,
        ]
    )

    drafts = torch.stack([sampler.draw(draft_logits), sampler.draw(draft_logits)], 1)
    num_correct, committed = sampler.verify(target_logits, drafts)

    assert drafts[0].tolist() == [1, 1]
    assert num_correct[0].item() == 1
    assert committed[0, :2].tolist() == [1, 2]


def test_all_greedy_chain_commits_the_target_argmax_of_every_row() -> None:
    sampler = speculative.ChainSampler(chain_batch(1, greedy=[True]))
    draft_logits = torch.tensor([[0.0, 3.0, 0.0, 0.0]])
    target_logits = torch.tensor(
        [[[0.0, 3.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]]]
    )

    drafts = torch.stack([sampler.draw(draft_logits), sampler.draw(draft_logits)], 1)
    num_correct, committed = sampler.verify(target_logits, drafts)

    assert num_correct.tolist() == [2]
    assert committed.tolist() == [[1, 1, 3]]


def test_timestamp_rows_see_the_drafts_ahead_of_them() -> None:
    """A verify row predicts the token after every draft before it, so the timestamp rules must count them."""
    from sglang_omni.models.whisper_asr.timestamp_logit_processor import (
        apply_timestamp_rules,
    )

    torch.manual_seed(0)
    vocab, eos, no_timestamps, timestamp_begin = 40, 20, 29, 30
    params = {
        "segment_timestamps": True,
        "timestamp_begin_id": timestamp_begin,
        "no_timestamps_token_id": no_timestamps,
        "eos_token_id": eos,
        "max_initial_timestamp_index": None,
    }
    histories = [[5, 31], [31, 33], [7], [32, 8, 9]]
    num_requests, draft_steps = len(histories), 4
    draft_logits = torch.randn(draft_steps, num_requests, vocab) * 4
    target_logits = torch.randn(num_requests, draft_steps + 1, vocab) * 4
    sampler = speculative.ChainSampler(
        chain_batch(
            num_requests,
            greedy=[True] * num_requests,
            custom_params=[params] * num_requests,
            output_ids=histories,
        )
    )

    drafts = torch.stack([sampler.draw(step) for step in draft_logits], dim=1)
    num_correct, committed = sampler.verify(target_logits, drafts)

    for row, history in enumerate(histories):
        expected_drafts = []
        for step in range(draft_steps):
            scores = draft_logits[step, row].clone()
            apply_timestamp_rules(scores, history + expected_drafts, params)
            expected_drafts.append(int(scores.argmax()))
        assert drafts[row].tolist() == expected_drafts
        expected_targets = []
        for position in range(draft_steps + 1):
            scores = target_logits[row, position].clone()
            apply_timestamp_rules(scores, history + expected_drafts[:position], params)
            expected_targets.append(int(scores.argmax()))
        assert committed[row].tolist() == expected_targets
