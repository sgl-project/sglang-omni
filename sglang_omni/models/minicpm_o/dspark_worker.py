# SPDX-License-Identifier: Apache-2.0
"""Request-seeded DSpark rounds for the MiniCPM-o thinker."""

from __future__ import annotations

from typing import Protocol

import torch
from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    AcceptGreedy,
    SoftmaxTemp,
)
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import build_speculative_verify_target_probs
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DraftBlockResult,
    DraftProposal,
    resolve_draft_params,
)
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    TargetVerifyExecutor,
    build_accept_outs,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite

from sglang_omni.models.minicpm_o.components.sglang_thinker import (
    MiniCPMOThinkerForCausalLM,
)
from sglang_omni.models.minicpm_o.speculative_sampling import (
    prepare_seeded_batch,
    sample_draft_block,
    verify_draft_block,
)
from sglang_omni.sampling.speculative_noise import (
    ACCEPTANCE_NOISE_SEED_SALT,
    DRAFT_NOISE_SEED_SALT,
    LARGEST_FLOAT32_BELOW_ONE,
    RESIDUAL_NOISE_SEED_SALT,
    seeded_exponential_noise,
    seeded_uniform_noise,
)
from sglang_omni.speculative.target_worker import SpeculativeTargetWorker


class PublishLengths(Protocol):
    def __call__(
        self, sequence_lengths: torch.Tensor, *, confidence: torch.Tensor | None = None
    ) -> None: ...


class GrammarBarrier(Protocol):
    def __call__(self) -> None: ...


class SeededDraftProposer(DraftBlockProposer):
    step_state: tuple[torch.Tensor, torch.Tensor] | None = None

    def propose(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: VerifyWindow,
        bs: int,
        device: str,
        target_model: MiniCPMOThinkerForCausalLM,
        sampling_info: SamplingBatchInfo,
        anchor_prepared: bool = False,
    ) -> DraftProposal:
        assert self.step_state is not None
        seeds, positions = self.step_state
        embeddings = unwrap_lora_layer(
            target_model.get_input_embeddings()
            if self.sample_from_anchor
            else self.draft_model.embed_tokens
        )

        forward = self._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device=device,
            embed_module=embeddings,
            draft_sampler=self._draft_sampler,
            sampling_info=sampling_info,
            anchor_prepared=anchor_prepared,
        )  # noqa: leading-underscore  # upstream spelling
        with (
            self._base_logits_context()
        ):  # noqa: leading-underscore  # upstream spelling
            logits, features = self.draft_model.compute_base_logits(forward.raw_hidden)
        logits = logits.view(bs, self.gamma, -1)
        params = resolve_draft_params(
            sampling_info=sampling_info, bs=bs, device=logits.device
        )
        noise = seeded_exponential_noise(
            seeds,
            positions,
            salt=DRAFT_NOISE_SEED_SALT,
            column_count=self.gamma * logits.shape[-1],
        ).view_as(logits)
        head = self.draft_model.markov_head
        tokens, corrected = sample_draft_block(
            logits,
            forward.draft_block_ids[:, 0],
            head.markov_w1.weight,
            head.markov_w2.weight,
            params.step_temperatures,
            noise,
        )
        confidence = confidence_raw = None
        if self.draft_model.confidence_head is not None:
            confidence, confidence_raw = self.draft_model.compute_confidence(
                anchor_tokens=forward.draft_block_ids[:, 0],
                draft_tokens=tokens,
                confidence_features=features,
            )
        else:
            pass
        return DraftProposal(
            draft_block_ids=(
                forward.draft_block_ids
                if self.sample_from_anchor
                else forward.draft_block_ids[:, : self.gamma].contiguous()
            ),
            draft_block=DraftBlockResult(
                draft_tokens=tokens,
                corrected_logits=corrected,
                greedy_mask=params.greedy_mask,
                temperatures=params.temperatures,
                greedy_draft=False,
            ),
            confidence=confidence,
            confidence_raw=confidence_raw,
            folded=False,
        )


class SeededVerifyExecutor(TargetVerifyExecutor):
    step_state: tuple[torch.Tensor, torch.Tensor] | None = None

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
        layout: RaggedVerifyLayout | None,
        prefix_lens: torch.Tensor,
        draft_tokens: torch.Tensor,
        simulate_bonus_sampling_info: SamplingBatchInfo | None = None,
    ) -> AcceptOuts:
        assert self.step_state is not None
        assert not folded_accept and layout is None and target_logits is not None
        assert draft_block.corrected_logits is not None
        assert simulate_bonus_sampling_info is None
        seeds, positions = self.step_state

        if sampling_info.need_top_k_sampling or sampling_info.need_top_p_sampling:
            probabilities = build_speculative_verify_target_probs(
                next_token_logits=target_logits,
                sampling_info=sampling_info,
                draft_token_num=self.verify_num_draft_tokens,
                bs=bs,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )
        else:
            probabilities = SoftmaxTemp.execute(
                logits=target_logits,
                temperatures=sampling_info.temperatures,
                rows_per_request=self.verify_num_draft_tokens,
            ).view(bs, self.verify_num_draft_tokens, -1)
        uniforms = (
            seeded_uniform_noise(
                seeds,
                positions,
                salt=ACCEPTANCE_NOISE_SEED_SALT,
                column_count=self.gamma,
            )
            .float()
            .clamp_max(LARGEST_FLOAT32_BELOW_ONE)
        )
        noise = seeded_exponential_noise(
            seeds,
            positions,
            salt=RESIDUAL_NOISE_SEED_SALT,
            column_count=probabilities.shape[-1],
        )
        temperatures = torch.where(
            draft_block.greedy_mask, 0.0, draft_block.temperatures
        )
        accepted, bonus = verify_draft_block(
            probabilities,
            draft_block.corrected_logits,
            draft_tokens,
            temperatures,
            uniforms,
            noise,
        )
        if sampling_info.is_any_greedy:
            # note (IcyDesert): top-k can break logit ties differently from the greedy argmax.
            greedy_accepted, greedy_bonus, _ = AcceptGreedy.execute(
                candidates=verify_ids_2d,
                target_logits=target_logits,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
            )
            accepted = torch.where(draft_block.greedy_mask, greedy_accepted, accepted)
            bonus = torch.where(draft_block.greedy_mask, greedy_bonus, bonus)
        else:
            pass
        return build_accept_outs(
            correct_len=accepted,
            bonus=bonus,
            cap_trim_lens=torch.zeros_like(accepted),
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_sync=self._tp_sync,
            sync_site=SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE,
        )  # noqa: leading-underscore  # upstream spelling


class MiniCPMDSparkWorker(DSparkWorkerV2):
    supports_request_seed: bool = False
    seeded_proposer: SeededDraftProposer
    seeded_verifier: SeededVerifyExecutor

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: SpeculativeTargetWorker,
    ) -> None:
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        self.supports_request_seed = (
            # note (IcyDesert): the seeded proposer samples the full vocabulary locally without synchronizing token choices across TP ranks.
            self._hosts_draft
            and get_parallel().tp_size == 1
            and type(self.draft_model.markov_head) is VanillaMarkov
            and self._verify_planner.mode_value == "static"
            and not self._greedy_draft
            and self._simulate_acc_len <= 0
        )  # noqa: leading-underscore  # upstream spelling

        if self.supports_request_seed:
            self.seeded_proposer = SeededDraftProposer(
                draft_model=self.draft_model,
                draft_model_runner=self.draft_model_runner,
                gamma=self.gamma,
                mask_token_id=self._mask_token_id,
                draft_block_spec_info=self._draft_block_spec_info,
                tp_sync=self._tp_sync,
                random_seed=self.target_worker.random_seed,
                greedy_draft=False,
            )  # noqa: leading-underscore  # upstream spelling
            self.seeded_verifier = SeededVerifyExecutor(
                target_worker=self.target_worker,
                gamma=self.gamma,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                model_runner=self.model_runner,
                kv_injector=self._kv_injector,
                tp_sync=self._tp_sync,
            )  # noqa: leading-underscore  # upstream spelling
        else:
            pass

    def init_cuda_graphs(self) -> None:
        super().init_cuda_graphs()
        if self.supports_request_seed:
            self.seeded_proposer.attach_draft_sampler(
                self._draft_sampler
            )  # noqa: leading-underscore  # upstream spelling
            self.seeded_proposer.deferred_commit = (
                self._proposer.deferred_commit
            )  # noqa: leading-underscore  # upstream spelling
        else:
            pass

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish: PublishLengths | None = None,
        grammar_barrier: GrammarBarrier | None = None,
        *,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ) -> GenerationBatchResult:
        if (
            not self.supports_request_seed
            or batch.forward_mode.is_extend()
            or batch.is_extend_in_batch
            or batch.forward_mode.is_idle()
            or batch.sampling_info.is_all_greedy
            or all(
                request.sampling_params.sampling_seed is None for request in batch.reqs
            )
        ):
            return super().forward_batch_generation(
                batch, on_publish, grammar_barrier, pp_proxy_tensors=pp_proxy_tensors
            )
        else:
            pass

        state = prepare_seeded_batch(
            [request.rid for request in batch.reqs],
            [request.sampling_params.sampling_seed for request in batch.reqs],
            batch.seq_lens,
        )  # (seeds, positions)

        self.seeded_proposer.step_state = self.seeded_verifier.step_state = state
        native_proposer, native_verifier = (
            self._proposer,
            self._verify_executor,
        )  # noqa: leading-underscore  # upstream spelling
        # note (IcyDesert): both proposers stage temperatures into the same sampler buffers.
        self.seeded_proposer._staged_params_key = (
            native_proposer._staged_params_key
        )  # noqa: leading-underscore  # upstream spelling
        self._proposer, self._verify_executor = (
            self.seeded_proposer,
            self.seeded_verifier,
        )  # noqa: leading-underscore  # upstream spelling

        try:
            return super().forward_batch_generation(
                batch, on_publish, grammar_barrier, pp_proxy_tensors=pp_proxy_tensors
            )
        finally:
            native_proposer._staged_params_key = (
                self.seeded_proposer._staged_params_key
            )  # noqa: leading-underscore  # upstream spelling
            self._proposer, self._verify_executor = (
                native_proposer,
                native_verifier,
            )  # noqa: leading-underscore  # upstream spelling
            self.seeded_proposer.step_state = self.seeded_verifier.step_state = None
