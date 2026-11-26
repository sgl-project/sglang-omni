# SPDX-License-Identifier: Apache-2.0
"""The DSpark worker routes seeded rounds without changing ordinary DSpark execution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2

from sglang_omni.speculative import seeded_dspark
from sglang_omni.speculative.seeded_dspark import (
    SeededDraftProposer,
    SeededDSparkWorker,
    SeededVerifyExecutor,
)
from sglang_omni.speculative.target_worker import SpeculativeTargetWorker


@pytest.mark.parametrize(
    "tensor_parallel_size,vanilla_markov,verify_mode,greedy_draft,simulated_length,supported",
    [
        (1, True, "static", False, -1.0, True),
        (1, True, "static", False, 0.0, True),
        (2, True, "static", False, 0.0, False),
        (1, False, "static", False, 0.0, False),
        (1, True, "compact", False, 0.0, False),
        (1, True, "static", True, 0.0, False),
        (1, True, "static", False, 2.0, False),
    ],
)
def test_worker_seed_capability_tracks_supported_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tensor_parallel_size: int,
    vanilla_markov: bool,
    verify_mode: str,
    greedy_draft: bool,
    simulated_length: float,
    supported: bool,
) -> None:
    def initialize(
        self: DSparkWorkerV2,
        *,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: SpeculativeTargetWorker,
    ) -> None:
        self._hosts_draft = True  # noqa: leading-underscore  # upstream spelling
        self._target_worker = (
            target_worker  # noqa: leading-underscore  # upstream spelling
        )
        self.draft_model_runner = self.model_runner = Mock()
        self.gamma, self.verify_num_draft_tokens = 2, 3
        self._mask_token_id = 0  # noqa: leading-underscore  # upstream spelling
        self._draft_block_spec_info = self._tp_sync = self._kv_injector = (
            Mock()
        )  # noqa: leading-underscore  # upstream spelling
        self.draft_model = SimpleNamespace(
            markov_head=(
                VanillaMarkov(vocab_size=4, markov_rank=2) if vanilla_markov else None
            ),
        )
        self._verify_planner = SimpleNamespace(
            mode_value=verify_mode
        )  # noqa: leading-underscore  # upstream spelling
        self._greedy_draft = (
            greedy_draft  # noqa: leading-underscore  # upstream spelling
        )
        self._simulate_acc_len = (
            simulated_length  # noqa: leading-underscore  # upstream spelling
        )

    monkeypatch.setattr(DSparkWorkerV2, "__init__", initialize)
    monkeypatch.setattr(
        seeded_dspark,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=tensor_parallel_size),
    )
    proposer, verifier = Mock(), Mock()
    monkeypatch.setattr(seeded_dspark, "SeededDraftProposer", proposer)
    monkeypatch.setattr(seeded_dspark, "SeededVerifyExecutor", verifier)
    worker = SeededDSparkWorker(
        server_args=None, gpu_id=0, nccl_port=0, target_worker=Mock(random_seed=42)
    )
    assert worker.supports_request_seed is supported
    assert proposer.call_count == verifier.call_count == int(supported)
    if supported:
        assert worker.seeded_proposer is proposer.return_value
        assert worker.seeded_verifier is verifier.return_value


def test_rank_without_draft_initializes_only_native_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def initialize(
        self: DSparkWorkerV2,
        *,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: SpeculativeTargetWorker,
    ) -> None:
        self._hosts_draft = False  # noqa: leading-underscore  # upstream spelling
        self._target_worker = (
            SimpleNamespace()
        )  # noqa: leading-underscore  # upstream spelling

    monkeypatch.setattr(DSparkWorkerV2, "__init__", initialize)
    monkeypatch.setattr(
        seeded_dspark, "get_parallel", lambda: SimpleNamespace(tp_size=1)
    )
    native_graphs = Mock()
    monkeypatch.setattr(DSparkWorkerV2, "init_cuda_graphs", native_graphs)
    worker = SeededDSparkWorker(
        server_args=None, gpu_id=0, nccl_port=0, target_worker=None
    )
    assert not worker.supports_request_seed
    worker.init_cuda_graphs()
    native_graphs.assert_called_once()


def worker_and_batch(
    seed: int | None, min_p: float = 0.0, top_k: int = -1
) -> tuple[SeededDSparkWorker, SimpleNamespace]:
    worker = object.__new__(SeededDSparkWorker)
    worker.supports_request_seed = True
    worker._proposer = object.__new__(
        DraftBlockProposer
    )  # noqa: leading-underscore  # upstream spelling
    worker._verify_executor = object.__new__(
        TargetVerifyExecutor
    )  # noqa: leading-underscore  # upstream spelling
    worker._draft_sampler = None  # noqa: leading-underscore  # upstream spelling
    worker._proposer._staged_params_key = (
        None  # noqa: leading-underscore  # upstream spelling
    )
    worker._proposer.deferred_commit = (
        None  # noqa: leading-underscore  # upstream spelling
    )
    worker.seeded_proposer = object.__new__(SeededDraftProposer)
    worker.seeded_verifier = object.__new__(SeededVerifyExecutor)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
        sampling_info=SimpleNamespace(is_all_greedy=False),
        seq_lens=torch.tensor([17, 29], dtype=torch.int32),
        reqs=[
            SimpleNamespace(
                rid="seeded",
                sampling_params=SimpleNamespace(
                    sampling_seed=seed, min_p=min_p, top_k=top_k
                ),
            ),
            SimpleNamespace(
                rid="unseeded",
                sampling_params=SimpleNamespace(
                    sampling_seed=None, min_p=0.0, top_k=-1
                ),
            ),
        ],
    )
    return worker, batch


@pytest.mark.parametrize(
    "seed,forward_mode,greedy,supported",
    [
        (None, ForwardMode.DECODE, False, True),
        (42, ForwardMode.EXTEND, False, True),
        (42, ForwardMode.IDLE, False, True),
        (42, ForwardMode.DECODE, True, True),
        (42, ForwardMode.DECODE, False, False),
    ],
)
def test_native_round_preserves_components_and_callback(
    monkeypatch: pytest.MonkeyPatch,
    seed: int | None,
    forward_mode: ForwardMode,
    greedy: bool,
    supported: bool,
) -> None:
    worker, batch = worker_and_batch(seed)
    batch.forward_mode = forward_mode
    batch.sampling_info.is_all_greedy = greedy
    worker.supports_request_seed = supported
    original_proposer = (
        worker._proposer
    )  # noqa: leading-underscore  # upstream spelling
    original_verifier = (
        worker._verify_executor
    )  # noqa: leading-underscore  # upstream spelling
    result = GenerationBatchResult()
    publications: list[torch.Tensor] = []

    def forbidden_seed_staging(
        request_ids: list[str], request_seeds: list[int | None], positions: torch.Tensor
    ) -> None:
        pytest.fail("A native round allocated seed state")

    def forward(
        self: DSparkWorkerV2,
        current: SimpleNamespace,
        on_publish: seeded_dspark.PublishLengths | None = None,
        grammar_barrier: seeded_dspark.GrammarBarrier | None = None,
        *,
        pp_proxy_tensors: None = None,
    ) -> GenerationBatchResult:
        assert current is batch
        assert (
            self._proposer is original_proposer
        )  # noqa: leading-underscore  # upstream spelling
        assert (
            self._verify_executor is original_verifier
        )  # noqa: leading-underscore  # upstream spelling
        assert on_publish is not None
        on_publish(current.seq_lens)
        return result

    monkeypatch.setattr(seeded_dspark, "prepare_seeded_batch", forbidden_seed_staging)
    monkeypatch.setattr(DSparkWorkerV2, "forward_batch_generation", forward)
    assert worker.forward_batch_generation(batch, publications.append) is result
    assert publications == [batch.seq_lens]


@pytest.mark.parametrize("raises", [False, True])
def test_seeded_round_restores_components_and_drops_temporary_state(
    monkeypatch: pytest.MonkeyPatch, raises: bool
) -> None:
    worker, batch = worker_and_batch(0)
    original_proposer = (
        worker._proposer
    )  # noqa: leading-underscore  # upstream spelling
    original_verifier = (
        worker._verify_executor
    )  # noqa: leading-underscore  # upstream spelling
    result = GenerationBatchResult()

    def forward(
        self: DSparkWorkerV2,
        current: SimpleNamespace,
        on_publish: seeded_dspark.PublishLengths | None = None,
        grammar_barrier: seeded_dspark.GrammarBarrier | None = None,
        *,
        pp_proxy_tensors: None = None,
    ) -> GenerationBatchResult:
        assert (
            self._proposer is worker.seeded_proposer
        )  # noqa: leading-underscore  # upstream spelling
        assert (
            self._verify_executor is worker.seeded_verifier
        )  # noqa: leading-underscore  # upstream spelling
        state = worker.seeded_proposer.step_state
        assert state is not None and state is worker.seeded_verifier.step_state
        assert state[0][0].item() == 0
        current.seq_lens.add_(3)
        assert state[1].tolist() == [17, 29]
        if raises:
            raise RuntimeError("model forward failed")
        else:
            return result

    monkeypatch.setattr(DSparkWorkerV2, "forward_batch_generation", forward)
    if raises:
        with pytest.raises(RuntimeError, match="model forward failed"):
            worker.forward_batch_generation(batch)
    else:
        assert worker.forward_batch_generation(batch) is result
    assert (
        worker._proposer is original_proposer
    )  # noqa: leading-underscore  # upstream spelling
    assert (
        worker._verify_executor is original_verifier
    )  # noqa: leading-underscore  # upstream spelling
    assert worker.seeded_proposer.step_state is None
    assert worker.seeded_verifier.step_state is None


def test_switching_rounds_keeps_graph_sampling_parameters_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A seeded round must not leave the native draft using stale temperatures."""
    worker, batch = worker_and_batch(42)
    native_proposer = worker._proposer  # noqa: leading-underscore  # upstream spelling
    sampler = SimpleNamespace(stage_sampling_params=Mock())
    commit = Mock()

    def initialize_graphs(self: DSparkWorkerV2) -> None:
        self._draft_sampler = sampler  # noqa: leading-underscore  # upstream spelling
        native_proposer.attach_draft_sampler(sampler)
        native_proposer.deferred_commit = commit

    monkeypatch.setattr(DSparkWorkerV2, "init_cuda_graphs", initialize_graphs)
    worker.init_cuda_graphs()
    assert worker.seeded_proposer.deferred_commit is commit
    assert (
        worker.seeded_proposer._draft_sampler is sampler
    )  # noqa: leading-underscore  # upstream spelling
    batch.sampling_info.temperatures = torch.tensor([[0.7], [0.8]])
    batch.sampling_info.top_ks = torch.tensor([10, 20])
    native_info = batch.sampling_info
    native_proposer._stage_sampling_params(
        sampler, bs=2, sampling_info=native_info
    )  # noqa: leading-underscore  # upstream spelling
    batch.sampling_info = SimpleNamespace(
        is_all_greedy=False,
        temperatures=torch.tensor([[0.9], [1.1]]),
        top_ks=torch.tensor([30, 40]),
    )
    seeded_info = batch.sampling_info

    def forward(
        self: DSparkWorkerV2,
        current: SimpleNamespace,
        on_publish: seeded_dspark.PublishLengths | None = None,
        grammar_barrier: seeded_dspark.GrammarBarrier | None = None,
        *,
        pp_proxy_tensors: None = None,
    ) -> GenerationBatchResult:
        self._proposer._stage_sampling_params(
            self._proposer._draft_sampler, bs=2, sampling_info=current.sampling_info
        )  # noqa: leading-underscore  # upstream spelling
        return GenerationBatchResult()

    monkeypatch.setattr(DSparkWorkerV2, "forward_batch_generation", forward)
    worker.forward_batch_generation(batch)
    batch.reqs[0].sampling_params.sampling_seed = None
    batch.sampling_info = native_info
    worker.forward_batch_generation(batch)
    assert sampler.stage_sampling_params.call_count == 3
    assert (
        sampler.stage_sampling_params.call_args.kwargs["sampling_info"] is native_info
    )
    batch.reqs[0].sampling_params.sampling_seed = 42
    batch.sampling_info = seeded_info
    worker.forward_batch_generation(batch)
    assert sampler.stage_sampling_params.call_count == 4
    assert (
        sampler.stage_sampling_params.call_args.kwargs["sampling_info"] is seeded_info
    )


def test_an_unseeded_min_p_round_takes_the_seeded_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the seeded verify filters by min_p, so a sampled min_p batch takes it without a seed."""
    worker, batch = worker_and_batch(None, min_p=0.1)
    staged: list[list[int | None]] = []

    def stage(
        request_ids: list[str], request_seeds: list[int | None], positions: torch.Tensor
    ) -> None:
        staged.append(request_seeds)
        raise RuntimeError("seeded round")

    monkeypatch.setattr(seeded_dspark, "prepare_seeded_batch", stage)
    with pytest.raises(RuntimeError, match="seeded round"):
        worker.forward_batch_generation(batch)
    assert staged == [[None, None]]


def test_a_greedy_min_p_row_keeps_its_sampled_neighbor_on_the_native_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """min_p never changes a greedy row, so it alone does not take the seeded verify."""
    worker, batch = worker_and_batch(None, min_p=0.1, top_k=1)
    result = GenerationBatchResult()

    def forbidden_seed_staging(
        request_ids: list[str], request_seeds: list[int | None], positions: torch.Tensor
    ) -> None:
        pytest.fail("A greedy min_p row took the seeded round")

    def forward(
        self: DSparkWorkerV2,
        current: SimpleNamespace,
        on_publish: seeded_dspark.PublishLengths | None = None,
        grammar_barrier: seeded_dspark.GrammarBarrier | None = None,
        *,
        pp_proxy_tensors: None = None,
    ) -> GenerationBatchResult:
        return result

    monkeypatch.setattr(seeded_dspark, "prepare_seeded_batch", forbidden_seed_staging)
    monkeypatch.setattr(DSparkWorkerV2, "forward_batch_generation", forward)
    assert worker.forward_batch_generation(batch) is result
