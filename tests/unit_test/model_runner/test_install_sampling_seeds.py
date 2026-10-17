# SPDX-License-Identifier: Apache-2.0
"""Base-runner _install_sampling_seeds: wire a request seed to the sampler."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_model_runner import (
    register_flashinfer_seeded_sampler,
)
from sglang_omni.sampling.seed import derive_sampling_seed

VOCAB_SIZE = 3072
TOP_K = 50


def req(seed, request_id="req"):
    sp = SimpleNamespace(sampling_seed=seed)
    return SimpleNamespace(
        request_id=request_id,
        data=SimpleNamespace(req=SimpleNamespace(sampling_params=sp)),
    )


def make_fb(sampling_seed=None, *, top_p=False, top_k=False, min_p=False, device="cpu"):
    return SimpleNamespace(
        sampling_info=SimpleNamespace(
            device=device,
            sampling_seed=sampling_seed,
            need_top_p_sampling=top_p,
            need_top_k_sampling=top_k,
            need_min_p_sampling=min_p,
        )
    )


def test_installs_per_row_seeds_and_noops_without_one():
    runner = object.__new__(ModelRunner)
    # seeded rows -> per-row int64 seed tensor; mixed unseeded rows get a
    # rank-shared fallback derived from the request id.
    fb = make_fb()
    requests = [req(42, "seeded"), req(None, "unseeded")]
    runner.install_sampling_seeds(fb, requests)
    ss = fb.sampling_info.sampling_seed
    assert isinstance(ss, torch.Tensor) and ss.dtype == torch.long
    assert int(ss[0]) == 42 and ss.shape == (2,)
    assert int(ss[1]) == derive_sampling_seed("sglang-omni-unseeded-row", "unseeded")
    assert requests[1].data.req.sampling_params.sampling_seed is None
    # no seed anywhere -> left unseeded (preserves random sampling)
    fb2 = make_fb()
    runner.install_sampling_seeds(fb2, [req(None), req(None)])
    assert fb2.sampling_info.sampling_seed is None


def test_does_not_clobber_subclass_installed_seed():
    runner = object.__new__(ModelRunner)
    preset = torch.tensor([1, 2, 3])
    fb = make_fb(sampling_seed=preset)
    runner.install_sampling_seeds(fb, [req(42), req(42), req(42)])
    assert fb.sampling_info.sampling_seed is preset


def test_preinstalled_seed_requires_sampling_mode_contract():
    runner = object.__new__(ModelRunner)
    preset = torch.tensor([1, 2])
    fb = make_fb(sampling_seed=preset, min_p=True)
    with pytest.raises(ValueError, match="min_p"):
        runner.install_sampling_seeds(fb, [req(42), req(42)])


def test_rejects_seeded_min_p_before_upstream_sampler():
    runner = object.__new__(ModelRunner)
    with pytest.raises(ValueError, match="min_p"):
        runner.install_sampling_seeds(make_fb(min_p=True), [req(42)])


def test_installs_seeded_top_k_and_top_p_on_flashinfer() -> None:
    runner = object.__new__(ModelRunner)
    fb = make_fb(top_p=True, top_k=True)
    with get_context().override_server_args(sampling_backend="flashinfer"):
        runner.install_sampling_seeds(fb, [req(42)])
    assert int(fb.sampling_info.sampling_seed[0]) == 42


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_installs_cuda_seeds_without_a_host_wait() -> None:
    runner = object.__new__(ModelRunner)
    fb = make_fb(device="cuda")
    torch.cuda._sleep(1_000_000_000)  # noqa: leading-underscore  # upstream name
    runner.install_sampling_seeds(fb, [req(42, "seeded"), req(None, "unseeded")])
    assert not torch.cuda.current_stream().query()
    assert fb.sampling_info.sampling_seed.tolist() == [
        42,
        derive_sampling_seed("sglang-omni-unseeded-row", "unseeded"),
    ]


def sampling_info_for(
    rows: int,
    sampling_seeds: list[int] | None,
    *,
    top_k: int = TOP_K,
    sampling_mask_rows: list[int] | None = None,
) -> SamplingBatchInfo:
    return SamplingBatchInfo(
        temperatures=torch.full((rows, 1), 0.9, device="cuda"),
        top_ps=torch.ones(rows, device="cuda"),
        top_ks=torch.full((rows,), top_k, dtype=torch.int32, device="cuda"),
        min_ps=torch.zeros(rows, device="cuda"),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=True,
        need_min_p_sampling=False,
        vocab_size=VOCAB_SIZE,
        sampling_seed=(
            None
            if sampling_seeds is None
            else torch.tensor(sampling_seeds, dtype=torch.long, device="cuda")
        ),
        sampling_mask_batch_indices=(
            None
            if sampling_mask_rows is None
            else torch.tensor(sampling_mask_rows, dtype=torch.long, device="cuda")
        ),
        device="cuda",
    )


def draw(
    sampling_backend: str,
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    positions: torch.Tensor,
) -> tuple[list[int], LogitsProcessorOutput]:
    rows = logits.shape[0]
    logits_output = LogitsProcessorOutput(next_token_logits=logits.clone())
    with (
        get_context().override_server_args(sampling_backend=sampling_backend),
        get_parallel().override(tp_group=SimpleNamespace(device_group=None)),
    ):
        token_ids = create_sampler()(
            logits_output,
            sampling_info,
            False,
            [0] * rows,
            [[] for _ in range(rows)],
            positions,
        )
    return token_ids.view(-1).tolist(), logits_output


def served_logits(rows: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(0)
    return (torch.randn(rows, VOCAB_SIZE, generator=generator) * 4).to(
        "cuda", torch.bfloat16
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_rows_draw_the_pytorch_backend_tokens_in_any_order() -> None:
    register_flashinfer_seeded_sampler()
    logits = served_logits(4)
    seeds = [11, 22, 33, 44]
    positions = torch.tensor([3, 17, 5, 40], device="cuda")
    reversed_rows = torch.tensor([3, 2, 1, 0], device="cuda")

    pytorch_tokens, _ = draw("pytorch", logits, sampling_info_for(4, seeds), positions)
    flashinfer_tokens, _ = draw(
        "flashinfer", logits, sampling_info_for(4, seeds), positions
    )
    reversed_tokens, _ = draw(
        "flashinfer",
        logits[reversed_rows],
        sampling_info_for(4, seeds[::-1]),
        positions[reversed_rows],
    )

    assert flashinfer_tokens == pytorch_tokens
    assert reversed_tokens == pytorch_tokens[::-1]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_unseeded_rows_draw_with_flashinfer_keeping_top_k_ties() -> None:
    register_flashinfer_seeded_sampler()
    rows = 64
    tied_tokens = {7, 300, 2048}
    logits = torch.full((rows, VOCAB_SIZE), -20.0, device="cuda")
    logits[:, sorted(tied_tokens)] = 10.0
    positions = torch.arange(rows, device="cuda")

    unseeded_tokens, _ = draw(
        "flashinfer", logits, sampling_info_for(rows, None, top_k=1), positions
    )
    seeded_tokens, _ = draw(
        "flashinfer",
        logits,
        sampling_info_for(rows, list(range(1, rows + 1)), top_k=1),
        positions,
    )

    assert set(unseeded_tokens) == tied_tokens
    assert len(set(seeded_tokens)) == 1
    assert set(seeded_tokens) <= tied_tokens


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_rows_return_the_pytorch_backend_sampling_mask() -> None:
    """A seeded FlashInfer batch that asks for the sampling mask gets the PyTorch
    backend's mask."""
    register_flashinfer_seeded_sampler()
    logits = served_logits(3)
    seeds = [11, 22, 33]
    positions = torch.tensor([3, 17, 5], device="cuda")

    pytorch_tokens, pytorch_output = draw(
        "pytorch",
        logits,
        sampling_info_for(3, seeds, sampling_mask_rows=[0, 2]),
        positions,
    )
    flashinfer_tokens, flashinfer_output = draw(
        "flashinfer",
        logits,
        sampling_info_for(3, seeds, sampling_mask_rows=[0, 2]),
        positions,
    )
    pytorch_mask = pytorch_output.sampling_mask_output
    flashinfer_mask = flashinfer_output.sampling_mask_output

    assert flashinfer_tokens == pytorch_tokens
    assert torch.equal(flashinfer_mask.token_ids, pytorch_mask.token_ids)
    assert torch.equal(flashinfer_mask.lengths, pytorch_mask.lengths)
    assert torch.equal(
        flashinfer_mask.selected_logprobs, pytorch_mask.selected_logprobs
    )
    assert torch.equal(flashinfer_mask.statuses, pytorch_mask.statuses)
