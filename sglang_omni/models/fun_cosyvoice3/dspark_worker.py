# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 DSpark worker whose verification keeps the stage's sampling exact.

The DSpark verify samples from temperature and top-k/top-p only. The stage also
applies a repetition penalty over the output history, suppresses stop ids until the
minimum length, and redraws a speech token that repeats inside the recent window, so
this worker feeds every verify row's history to dspark_sampling.exact_accept.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import torch
from sglang.kernels.ops.speculative.dspark.dspark_accept import FinalizeAcceptLens
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import BuildOutTokens
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.models.dspark import BaseDSparkMarkovHead
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components import dspark_draft as sglang_dspark_draft
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    DraftNoise,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.srt.utils import DynamicGradMode

from sglang_omni.models.fun_cosyvoice3.dspark_draft import (
    FunCosyVoice3DSparkDraftModel,
    MirroredMarkovHead,
)
from sglang_omni.models.fun_cosyvoice3.dspark_sampling import (
    EMPTY_HISTORY_TOKEN,
    DraftSamplingState,
    ExactAccept,
    exact_accept,
    sample_draft_block,
)
from sglang_omni.models.fun_cosyvoice3.model_runner import COSYVOICE3_RAS_WINDOW_SIZE
from sglang_omni.models.fun_cosyvoice3.sglang_model import TOTAL_VOCAB_SIZE
from sglang_omni.speculative.target_worker import SpeculativeTargetWorker


@dataclass(kw_only=True)
class StepSamplingState:
    """Sampling inputs of the running requests for one speculative step."""

    draft: DraftSamplingState
    recent_tokens: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor


class RequestHistories:
    """Which tokens each request slot has committed, kept on the device.

    A step uploads only the tokens committed since that slot's last step; a slot is
    cleared when a different request takes it over.
    """

    def __init__(self, *, slot_count: int, device: torch.device) -> None:
        self.presence_by_slot: torch.Tensor = torch.zeros(
            (slot_count, TOTAL_VOCAB_SIZE), dtype=torch.bool, device=device
        )
        self.slot_owners: list[str | None] = [None] * slot_count
        self.slot_lengths: list[int] = [0] * slot_count

    def presence_mask(self, batch: ScheduleBatch) -> torch.Tensor:
        cleared_slots: list[int] = []
        update_slots: list[int] = []
        update_tokens: list[int] = []
        for request in batch.reqs:
            slot = request.kv.req_pool_idx
            output_ids = request.output_ids
            # Every step grows a request's outputs, so a length that did not grow
            # means a new request reused the id.
            if self.slot_owners[slot] != request.rid or self.slot_lengths[slot] >= len(
                output_ids
            ):
                cleared_slots.append(slot)
                self.slot_owners[slot] = request.rid
                synced_length = 0
            else:
                synced_length = self.slot_lengths[slot]
            update_tokens.extend(output_ids[synced_length:])
            update_slots.extend([slot] * (len(output_ids) - synced_length))
            self.slot_lengths[slot] = len(output_ids)
        device = self.presence_by_slot.device
        if cleared_slots:
            self.presence_by_slot[torch.tensor(cleared_slots, device=device)] = False
        else:
            pass
        if update_tokens:
            self.presence_by_slot[
                torch.tensor(update_slots, device=device),
                torch.tensor(update_tokens, dtype=torch.long, device=device),
            ] = True
        else:
            pass
        return self.presence_by_slot[batch.req_pool_indices]


def build_step_sampling_state(
    batch: ScheduleBatch,
    sampling_info: SamplingBatchInfo,
    histories: RequestHistories,
) -> StepSamplingState:
    """Collect each running request's committed history and sampling parameters."""
    device = sampling_info.temperatures.device
    recent_tokens = torch.tensor(
        [
            [EMPTY_HISTORY_TOKEN]
            * (
                COSYVOICE3_RAS_WINDOW_SIZE
                - len(request.output_ids[-COSYVOICE3_RAS_WINDOW_SIZE:])
            )
            + list(request.output_ids[-COSYVOICE3_RAS_WINDOW_SIZE:])
            for request in batch.reqs
        ],
        dtype=torch.long,
    )
    # One host-to-device copy carries the three per-request values.
    request_values = torch.tensor(
        [
            [
                request.sampling_params.repetition_penalty,
                len(request.output_ids),
                request.sampling_params.min_new_tokens,
            ]
            for request in batch.reqs
        ],
        dtype=torch.float64,
    ).to(device)
    top_ks = sampling_info.top_ks.view(-1)
    top_ps = sampling_info.top_ps.view(-1).float()
    if not (sampling_info.need_top_p_sampling or sampling_info.need_top_k_sampling):
        # The stage samples the full softmax when no request filters tokens.
        top_ps = torch.full_like(top_ps, float("inf"))
    else:
        pass
    return StepSamplingState(
        draft=DraftSamplingState(
            presence_mask=histories.presence_mask(batch),
            repetition_penalties=request_values[:, 0].float(),
            output_lengths=request_values[:, 1].long(),
            min_new_tokens=request_values[:, 2].long(),
            temperatures=sampling_info.temperatures.view(-1).float(),
            greedy=top_ks == 1,
        ),
        recent_tokens=recent_tokens.to(device),
        top_ks=top_ks,
        top_ps=top_ps,
    )


class ExactVerifyExecutor(TargetVerifyExecutor):
    """Accept drafts against the stage's exact per-position distribution."""

    step_state: StepSamplingState | None = None
    accept_function: ExactAccept = staticmethod(exact_accept)

    def accept_and_finalize(
        self,
        *,
        folded_accept: bool,
        bs: int,
        verify_ids_2d: torch.Tensor,
        target_logits: torch.Tensor | None,
        draft_block: DraftBlockResult,
        sampling_info: SamplingBatchInfo,
        draft_input: DFlashDraftInputV2,
        layout: None,
        prefix_lens: torch.Tensor,
        draft_tokens: torch.Tensor,
        simulate_bonus_sampling_info: None = None,
    ) -> AcceptOuts:
        state = self.step_state
        assert not folded_accept and layout is None and target_logits is not None
        assert state is not None, "a verify step needs its sampling state"
        draft_state = state.draft
        draft_tokens = draft_tokens.long()
        accepted_count, next_tokens = self.accept_function(
            target_logits,
            draft_block.corrected_logits,
            draft_tokens,
            draft_state.presence_mask,
            draft_state.repetition_penalties,
            draft_state.output_lengths,
            draft_state.min_new_tokens,
            draft_state.temperatures,
            draft_state.greedy,
            state.top_ks,
            state.top_ps,
            state.recent_tokens,
            torch.rand(draft_tokens.shape, device=draft_tokens.device),
            torch.empty(
                (bs, TOTAL_VOCAB_SIZE), device=draft_tokens.device
            ).exponential_(),
        )
        finalized = FinalizeAcceptLens.execute(
            correct_len=accepted_count,
            cap_trim_lens=torch.zeros_like(accepted_count),
            prefix_lens=prefix_lens,
        )
        out_tokens = BuildOutTokens.execute(
            draft_tokens=draft_tokens,
            correct_len=accepted_count,
            bonus=next_tokens,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            gamma=self.gamma,
        )
        return AcceptOuts(
            correct_len=accepted_count,
            bonus=next_tokens,
            cap_trim_lens=finalized.cap_trim_lens,
            commit_lens=finalized.commit_lens,
            new_seq_lens=finalized.new_seq_lens,
            out_tokens=out_tokens,
        )


MIRRORED_DRAFT_SAMPLING_LOCK = Lock()
mirrored_draft_sampling_installed = False


def install_mirrored_draft_sampling() -> None:
    """Let a MirroredMarkovHead draw its own DSpark draft block."""
    global mirrored_draft_sampling_installed
    with MIRRORED_DRAFT_SAMPLING_LOCK:
        if mirrored_draft_sampling_installed:
            return
        else:
            pass
        sglang_sample_draft_block = sglang_dspark_draft.sample_draft_block

        def sample_draft_block(
            *,
            base_logits: torch.Tensor,
            anchor_tokens: torch.Tensor,
            draft_hidden: torch.Tensor | None,
            sampling_info: SamplingBatchInfo | None,
            markov_head: BaseDSparkMarkovHead,
            noise: DraftNoise,
            device: torch.device,
            tp_sync: SpecTpSync,
            greedy_draft: bool,
        ) -> DraftBlockResult:
            if not isinstance(markov_head, MirroredMarkovHead):
                return sglang_sample_draft_block(
                    base_logits=base_logits,
                    anchor_tokens=anchor_tokens,
                    draft_hidden=draft_hidden,
                    sampling_info=sampling_info,
                    markov_head=markov_head,
                    noise=noise,
                    device=device,
                    tp_sync=tp_sync,
                    greedy_draft=greedy_draft,
                )
            else:
                # note (wilsonzheng0327): SGLang drafts through per-step additive
                # biases, which cannot carry the stage's penalty over the drafts
                # before each step, so the mirrored head draws the block itself.
                params = sglang_dspark_draft.resolve_draft_params(
                    sampling_info=sampling_info, bs=base_logits.shape[0], device=device
                )
                draft_tokens, shaped_logits = markov_head.draw_block(
                    base_logits, anchor_tokens
                )
                return DraftBlockResult(
                    draft_tokens=draft_tokens,
                    corrected_logits=shaped_logits,
                    greedy_mask=params.greedy_mask,
                    temperatures=params.temperatures,
                    greedy_draft=greedy_draft,
                )

        sglang_dspark_draft.sample_draft_block = sample_draft_block
        mirrored_draft_sampling_installed = True


class FunCosyVoice3DSparkWorker(DSparkWorkerV2):
    """DSpark worker that mirrors the stage's sampling in drafting and verification."""

    request_histories: RequestHistories

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: SpeculativeTargetWorker,
    ) -> None:
        install_mirrored_draft_sampling()
        # note (wilsonzheng0327): the draft draw and the acceptance noise are local,
        # so tensor-parallel ranks would disagree on the committed tokens.
        if get_parallel().tp_size != 1:
            raise ValueError("Fun-CosyVoice3 speculative decoding runs on one GPU")
        elif (
            read_ragged_verify_mode() != RaggedVerifyMode.STATIC
            or envs.SGLANG_SIMULATE_ACC_LEN.get() > 0
            or envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.get()
        ):
            raise ValueError(
                "Fun-CosyVoice3 speculative decoding verifies whole sampled blocks; "
                "unset SGLANG_RAGGED_VERIFY_MODE, SGLANG_SIMULATE_ACC_LEN and "
                "SGLANG_DSPARK_USE_GREEDY_DRAFT"
            )
        else:
            pass
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        # The engine builder always materializes the draft under this class.
        assert isinstance(self.draft_model, FunCosyVoice3DSparkDraftModel)
        self.exact_verify_executor = ExactVerifyExecutor(
            target_worker=self.target_worker,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            tp_sync=self._tp_sync,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
        self._verify_executor = (
            self.exact_verify_executor
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        # note (wilsonzheng0327): eager, the draft block and the exact verify are about a
        # hundred small kernels per step; compiled they fuse into a few.
        self.exact_verify_executor.accept_function = torch.compile(
            exact_accept, dynamic=True
        )
        self.draft_model.markov_head.block_sampler = torch.compile(
            sample_draft_block, dynamic=True
        )
        self.warm_up_compiled_sampling()

    def alloc_memory_pool(
        self,
        memory_pool_config: MemoryPoolConfig | None = None,
        req_to_token_pool: ReqToTokenPool | None = None,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator | None = None,
    ) -> None:
        super().alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        # note (wilsonzheng0327): the draft loads before the target sizes its pool, so
        # the per-slot histories are built once the slots exist.
        self.request_histories = RequestHistories(
            slot_count=self.model_runner.req_to_token_pool.req_to_token.shape[0],
            device=self.device,
        )

    # note (wilsonzheng0327): compiled graphs guard on grad mode, so the warm-up runs
    # under the scheduler loop's mode.
    @DynamicGradMode()
    def warm_up_compiled_sampling(self) -> None:
        """Compile the draft block and the exact verify before the first request.

        Dynamic shapes still specialize a single running request, so both one and two
        requests are traced here instead of on the serving thread.
        """
        markov_head = self.draft_model.markov_head
        draft_count = self.gamma
        for batch_size in (1, 2):
            zero_per_request = torch.zeros(
                batch_size, dtype=torch.long, device=self.device
            )
            one_per_request = torch.ones(
                batch_size, dtype=torch.float32, device=self.device
            )
            presence_mask = torch.zeros(
                (batch_size, TOTAL_VOCAB_SIZE), dtype=torch.bool, device=self.device
            )
            greedy = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            base_logits = torch.zeros(
                (batch_size, draft_count, TOTAL_VOCAB_SIZE), device=self.device
            )
            draft_tokens, shaped_logits = markov_head.block_sampler(
                base_logits,
                zero_per_request,
                torch.ones_like(base_logits),
                markov_head.markov_w1.weight,
                markov_head.markov_w2.weight,
                presence_mask,
                one_per_request,
                zero_per_request,
                zero_per_request,
                one_per_request,
                greedy,
            )
            self.exact_verify_executor.accept_function(
                torch.zeros(
                    (batch_size * (draft_count + 1), TOTAL_VOCAB_SIZE),
                    device=self.device,
                ),
                shaped_logits,
                draft_tokens,
                presence_mask,
                one_per_request,
                zero_per_request,
                zero_per_request,
                one_per_request,
                greedy,
                torch.full(
                    (batch_size,),
                    TOTAL_VOCAB_SIZE,
                    dtype=torch.int32,
                    device=self.device,
                ),
                one_per_request,
                torch.full(
                    (batch_size, COSYVOICE3_RAS_WINDOW_SIZE),
                    EMPTY_HISTORY_TOKEN,
                    dtype=torch.long,
                    device=self.device,
                ),
                torch.zeros((batch_size, draft_count), device=self.device),
                torch.ones((batch_size, TOTAL_VOCAB_SIZE), device=self.device),
            )

    def _maybe_build_draft_sampler(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self,
    ) -> None:
        """The folded in-graph sampler would skip the mirrored Markov head."""
        return None

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish: None = None,
        grammar_barrier: None = None,
        *,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ) -> GenerationBatchResult:
        is_decode = not (
            batch.forward_mode.is_extend()
            or batch.is_extend_in_batch
            or batch.forward_mode.is_idle()
        )
        if is_decode:
            sampling_info = batch.sampling_info
            # Forward isolation hands over a copy without the penalizer, whose
            # step-start penalties would be wrong for every verify row after the first.
            assert sampling_info.penalizer_orchestrator is None
            # The stage's request builder sets none of these, and the exact verify has no
            # term for them.
            assert not (
                sampling_info.need_min_p_sampling
                or sampling_info.logit_bias is not None
                or sampling_info.has_custom_logit_processor
                or batch.return_logprob
            )
            state = build_step_sampling_state(
                batch, sampling_info, self.request_histories
            )
            self.draft_model.markov_head.sampling_state = state.draft
            self.exact_verify_executor.step_state = state
        else:
            pass
        return super().forward_batch_generation(
            batch,
            on_publish,
            grammar_barrier,
            pp_proxy_tensors=pp_proxy_tensors,
        )
