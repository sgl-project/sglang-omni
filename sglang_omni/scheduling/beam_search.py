# SPDX-License-Identifier: Apache-2.0
"""Beam search with N-best output on SGLang's leader-plus-member-row beam groups."""

from __future__ import annotations

import math

import torch
from sglang.srt.beam_search import batch_tail
from sglang.srt.beam_search.beam_group import BeamGroup, BeamGroupState, CompletedBeam
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.beam_search.fork import MEMBER_LENGTH_MARGIN
from sglang.srt.beam_search.history import BeamNode
from sglang.srt.beam_search.joint_select import FinalSelect, SelectResult
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers import schedule_batch as schedule_batch_module
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.runtime_context import get_memory, get_parallel, get_schedule
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.scheduling.types import (
    MAX_BEAM_WIDTH,
    BeamHypothesis,
    BeamLogitRules,
    BeamSearchSettings,
)


def compact_by_slot(
    values: torch.Tensor, slots: torch.Tensor, size: int
) -> torch.Tensor:
    """Fixed-shape compaction: values[i] lands at slots[i]; slot size collects the dropped values."""
    compacted = values.new_zeros(size + 1)
    compacted.scatter_(0, slots, values)
    return compacted[:size]


def rank_beam_candidates(
    cumulative_logprobs: torch.Tensor,
    candidate_logprobs: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    beam_width: int,
    diversity_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank the frontier's candidates and keep the best 2 * beam_width.

    candidate_logprobs and candidate_token_ids hold each frontier row's top 2 * beam_width tokens, best first. The ranking adds diversity_rate * (rank within parent % beam_width); the returned cumulative log-probs do not.

    Returns:
        Token ids, parent rows, and raw cumulative log-probs, in ranking order.
    """
    candidates_per_row = candidate_logprobs.shape[1]
    rank_within_parent = torch.arange(
        candidates_per_row, device=candidate_logprobs.device
    )
    cumulative = cumulative_logprobs[:, None] + candidate_logprobs
    ranking_scores = cumulative + diversity_rate * (rank_within_parent % beam_width)
    order = ranking_scores.reshape(-1).topk(candidates_per_row).indices
    return (
        candidate_token_ids.reshape(-1)[order],
        order // candidates_per_row,
        cumulative.reshape(-1)[order],
    )


def select_beam_candidates(
    cumulative_logprobs: torch.Tensor,
    candidate_logprobs: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    stop_token_ids: torch.Tensor,
    beam_width: int,
    diversity_rate: float,
) -> SelectResult:
    """One beam step with Hugging Face generate's rules: the first beam_width non-stop candidates survive, and a stop candidate finishes only within the top beam_width ranks."""
    token_ids, parent_rows, cumulative = rank_beam_candidates(
        cumulative_logprobs,
        candidate_logprobs,
        candidate_token_ids,
        beam_width,
        diversity_rate,
    )
    is_stop = torch.isin(token_ids, stop_token_ids)
    is_live = ~is_stop
    live_rank = is_live.long().cumsum(0)
    is_survivor = is_live & (live_rank <= beam_width)
    is_finished = is_stop & (
        torch.arange(token_ids.shape[0], device=token_ids.device) < beam_width
    )
    finished_rank = is_finished.long().cumsum(0)
    survivor_slots = torch.where(is_survivor, live_rank - 1, beam_width)
    finished_slots = torch.where(is_finished, finished_rank - 1, beam_width)
    return SelectResult(
        next_tokens=compact_by_slot(token_ids, survivor_slots, beam_width),
        parent_idx=compact_by_slot(parent_rows, survivor_slots, beam_width),
        new_cum_logprobs=compact_by_slot(cumulative, survivor_slots, beam_width),
        num_survivors=is_survivor.long().sum(),
        fin_tokens=compact_by_slot(token_ids, finished_slots, beam_width),
        fin_parent_idx=compact_by_slot(parent_rows, finished_slots, beam_width),
        fin_cum_logprobs=compact_by_slot(cumulative, finished_slots, beam_width),
        num_finished=is_finished.long().sum(),
    )


def select_final_beam_candidates(
    cumulative_logprobs: torch.Tensor,
    candidate_logprobs: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    beam_width: int,
    diversity_rate: float,
) -> FinalSelect:
    """The step that reaches max_new_tokens: the top beam_width candidates all finish."""
    token_ids, parent_rows, cumulative = rank_beam_candidates(
        cumulative_logprobs,
        candidate_logprobs,
        candidate_token_ids,
        beam_width,
        diversity_rate,
    )
    return FinalSelect(
        tokens=token_ids[:beam_width],
        parent_idx=parent_rows[:beam_width],
        cum_logprobs=cumulative[:beam_width],
    )


class OmniBeamGroup(BeamGroup):
    """A beam group with Hugging Face generate's finish rules: the best beam_width finished hypotheses, with early stopping."""

    def __init__(
        self,
        *,
        beam_width: int,
        stop_token_ids: list[int],
        max_new_tokens: int,
        diversity_rate: float,
        logit_rules: BeamLogitRules | None,
        device: torch.device,
    ) -> None:
        super().__init__(
            beam_width=beam_width,
            stop_token_ids=stop_token_ids,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        self.diversity_rate = diversity_rate
        self.logit_rules = logit_rules
        self.stop_token_id_set = frozenset(stop_token_ids)

    def keep_best_completed(self) -> None:
        self.completed = sorted(
            self.completed,
            key=lambda beam: self.beam_score(beam.cum_logprob, beam.num_tokens),
            reverse=True,
        )[: self.beam_width]

    def _commit_step(  # noqa: leading-underscore  # SGLang BeamGroup commit hook
        self, sel: SelectResult
    ) -> bool:
        num_survivors = int(sel.num_survivors)
        num_finished = int(sel.num_finished)
        generated_length = self.num_committed + 1
        for token_id, parent_row, cumulative_logprob in zip(
            sel.fin_tokens[:num_finished].tolist(),
            sel.fin_parent_idx[:num_finished].tolist(),
            sel.fin_cum_logprobs[:num_finished].tolist(),
        ):
            self.completed.append(
                CompletedBeam(
                    BeamNode(token_id, self.leaves[parent_row]),
                    cumulative_logprob,
                    generated_length,
                    matched_token=token_id,
                )
            )
        self.keep_best_completed()
        survivor_cumulative_logprobs = sel.new_cum_logprobs[:num_survivors].tolist()
        self.leaves = [
            BeamNode(token_id, self.leaves[parent_row])
            for token_id, parent_row in zip(
                sel.next_tokens[:num_survivors].tolist(),
                sel.parent_idx[:num_survivors].tolist(),
            )
        ]
        self.num_committed = generated_length
        if num_survivors < self.beam_width:
            for leaf, cumulative_logprob in zip(
                self.leaves, survivor_cumulative_logprobs
            ):
                self.completed.append(
                    CompletedBeam(
                        leaf, cumulative_logprob, generated_length, matched_token=None
                    )
                )
            self.keep_best_completed()
            self.leaves = []
            self.state = BeamGroupState.FINISHED
            return True
        elif len(self.completed) == self.beam_width and self.beam_score(
            max(survivor_cumulative_logprobs), generated_length
        ) <= self.beam_score(
            self.completed[-1].cum_logprob, self.completed[-1].num_tokens
        ):
            # note (zhumengzhiren): Hugging Face's early_stopping=False stops once the best running beam, scored at its current length, cannot beat the worst kept hypothesis.
            self.leaves = []
            self.state = BeamGroupState.FINISHED
            return True
        else:
            return False

    def _commit_final(  # noqa: leading-underscore  # SGLang BeamGroup commit hook
        self, sel: FinalSelect
    ) -> bool:
        generated_length = self.num_committed + 1
        parent_rows = sel.parent_idx.tolist()
        assert not parent_rows or max(parent_rows) < len(
            self.leaves
        ), "beam commit consumed an unsynced or misordered step"
        for token_id, parent_row, cumulative_logprob in zip(
            sel.tokens.tolist(), parent_rows, sel.cum_logprobs.tolist()
        ):
            self.completed.append(
                CompletedBeam(
                    BeamNode(token_id, self.leaves[parent_row]),
                    cumulative_logprob,
                    generated_length,
                    matched_token=(
                        token_id if token_id in self.stop_token_id_set else None
                    ),
                )
            )
        self.keep_best_completed()
        self.leaves = []
        self.num_committed = generated_length
        self.state = BeamGroupState.FINISHED
        return True


def beam_hypotheses(group: BeamGroup) -> list[BeamHypothesis]:
    """The finished group's N-best, best first."""
    return [
        BeamHypothesis(
            token_ids=list(result.tokens),
            cumulative_logprob=result.cum_logprob,
            score=result.beam_score,
            finish_reason="stop" if result.matched_token is not None else "length",
        )
        for result in group.final_results
    ]


class OmniBeamCoordinator(BeamCoordinator):
    """Beam groups for omni runners, which sample and relay next tokens themselves."""

    @property
    def has_live_groups(self) -> bool:
        return (
            self._num_live_groups > 0
        )  # noqa: leading-underscore  # SGLang BeamCoordinator counter

    def beam_request_error(self, req: Req, settings: BeamSearchSettings) -> str | None:
        """Why req cannot run as a beam request on this server, or None."""
        sampling_params = req.sampling_params
        beam_width = settings.beam_width
        max_new_tokens = self.beam_max_new_tokens(req)
        feature_conflicts = (
            ("speculative decoding", not self.spec_algorithm.is_none()),
            ("pipeline parallelism", get_parallel().pp_size > 1),
            ("dp attention", is_dp_attention_enabled()),
            ("hierarchical cache", bool(get_memory().enable_hierarchical_cache)),
            ("hisparse", bool(get_memory().enable_hisparse)),
            (
                "SWA or mamba caches",
                self.tree_cache.supports_swa() or self.tree_cache.supports_mamba(),
            ),
            ("diffusion decoding", self.dllm_enabled),
            ("sessions", req.session_id is not None),
            ("LoRA", req.lora_id is not None),
            ("logprobs", req.return_logprob),
            ("return_hidden_states", req.return_hidden_states),
            (
                "constrained decoding",
                any(
                    grammar is not None
                    for grammar in (
                        sampling_params.json_schema,
                        sampling_params.regex,
                        sampling_params.ebnf,
                        sampling_params.structural_tag,
                    )
                ),
            ),
            (
                "stop strings",
                bool(sampling_params.stop_strs or sampling_params.stop_regex_strs),
            ),
            ("min_new_tokens", sampling_params.min_new_tokens > 0),
        )
        conflicts = [name for name, is_set in feature_conflicts if is_set]
        if beam_width > MAX_BEAM_WIDTH:
            return f"beam_width must be at most {MAX_BEAM_WIDTH}, got {beam_width}."
        elif not math.isfinite(settings.diversity_rate):
            return (
                "beam_width > 1 needs a finite beam_search_diversity_rate, "
                f"got {settings.diversity_rate}."
            )
        elif conflicts:
            return f"beam_width > 1 does not support {', '.join(conflicts)}."
        elif get_schedule().page_size > 1:
            return "beam_width > 1 requires page_size 1."
        elif 2 * beam_width > self.model_config.vocab_size:
            return f"beam_width {beam_width} is too large for the vocabulary."
        elif beam_width > self.req_to_token_pool.size:
            return (
                f"beam_width {beam_width} needs {beam_width} request slots but the "
                f"server holds {self.req_to_token_pool.size}; raise max_running_requests."
            )
        elif max_new_tokens < 1:
            return (
                "beam_width > 1 needs at least one generated token within the "
                f"context budget (prompt_len={len(req.origin_input_ids)}, "
                f"max_req_len={self.max_req_len})."
            )
        else:
            return None

    def beam_max_new_tokens(self, req: Req) -> int:
        """The generation budget that keeps member rows inside the request's row width."""
        requested = req.sampling_params.max_new_tokens
        return min(
            requested if requested is not None else self.max_req_len,
            self.max_req_len - len(req.origin_input_ids) - 1 - MEMBER_LENGTH_MARGIN,
        )

    def attach_group(self, req: Req, settings: BeamSearchSettings) -> None:
        """Turn req into a beam leader; call only after beam_request_error returned None."""
        user_params = req.sampling_params
        max_new_tokens = self.beam_max_new_tokens(req)
        group = OmniBeamGroup(
            beam_width=settings.beam_width,
            stop_token_ids=self._collect_stop_token_ids(  # noqa: leading-underscore  # SGLang BeamCoordinator helper
                req, user_params
            ),
            max_new_tokens=max_new_tokens,
            diversity_rate=settings.diversity_rate,
            logit_rules=settings.logit_rules,
            device=self.req_to_token_pool.device,
        )
        group.leader = req
        group.prompt_len = len(req.origin_input_ids)
        # note (zhumengzhiren): the leader row's own sample is discarded; greedy params keep a greedy batch on the argmax sampler.
        leader_params = SamplingParams(
            max_new_tokens=max_new_tokens + MEMBER_LENGTH_MARGIN,
            temperature=0.0,
            ignore_eos=True,
            skip_special_tokens=user_params.skip_special_tokens,
            spaces_between_special_tokens=user_params.spaces_between_special_tokens,
            no_stop_trim=user_params.no_stop_trim,
        )
        leader_params.normalize(tokenizer=None)
        req.sampling_params = leader_params
        req.custom_logit_processor = None
        req.beam_group = group
        # The leader's decode suffix is a beam path, never a radix tree entry.
        req.skip_radix_cache_insert = True
        self._num_live_groups += (  # noqa: leading-underscore  # SGLang BeamCoordinator counter
            1
        )

    @staticmethod
    def candidate_logprobs(
        group: OmniBeamGroup, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each frontier row's top 2 * beam_width tokens by log_softmax(logits), after the group's masks; like Hugging Face, masking does not renormalize."""
        scores = logits.to(torch.float32, copy=True)
        log_normalizers = torch.logsumexp(scores, dim=-1, keepdim=True)
        if group.logit_rules is not None:
            group.logit_rules.mask_scores(scores, group.num_generated)
        else:
            pass
        candidates = (scores - log_normalizers).topk(group.num_candidates, dim=-1)
        return candidates.values, candidates.indices

    def select_group_step(
        self, group: OmniBeamGroup, logits: torch.Tensor, tick: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Launch half of one step: select on device and stage the result for commit.

        Returns:
            The next token of every frontier row, and each row's parent, which is None on the final step.
        """
        candidate_logprobs, candidate_token_ids = self.candidate_logprobs(group, logits)
        if group.next_step_is_final():
            final_selection = select_final_beam_candidates(
                group.frontier_cum_logprobs,
                candidate_logprobs,
                candidate_token_ids,
                group.beam_width,
                group.diversity_rate,
            )
            group.advance_final_frontier(final_selection, tick)
            return final_selection.tokens, None
        else:
            selection = select_beam_candidates(
                group.frontier_cum_logprobs,
                candidate_logprobs,
                candidate_token_ids,
                group.stop_token_ids,
                group.beam_width,
                group.diversity_rate,
            )
            group.advance_frontier(selection, tick)
            if group.logit_rules is not None:
                group.logit_rules.advance(selection.parent_idx, selection.next_tokens)
            else:
                pass
            return selection.next_tokens, selection.parent_idx

    def select_leader_prefill(
        self,
        req: Req,
        pos: int,
        logits_output: LogitsProcessorOutput,
        tick: int = 0,
    ) -> None:
        group = req.beam_group
        is_final = group.next_step_is_final()
        next_token_ids, _ = self.select_group_step(
            group, logits_output.beam.leader_logits[pos : pos + 1], tick
        )
        if is_final:
            # A one-token budget spawns no members; the relay slot still needs a token.
            self._stash_next_tokens(  # noqa: leading-underscore  # SGLang BeamCoordinator helper
                [req.kv.req_pool_idx], next_token_ids[:1]
            )
        else:
            self._spawn_member_rows(  # noqa: leading-underscore  # SGLang BeamCoordinator helper
                group, req
            )
            req.output_ids.append(0)
            self._stash_next_tokens(  # noqa: leading-underscore  # SGLang BeamCoordinator helper
                group.all_rows, next_token_ids
            )

    def select_and_relay_decode(
        self, batch: ScheduleBatch, logits_output: LogitsProcessorOutput
    ) -> None:
        tail = batch.beam_tail
        if tail is None:
            return
        else:
            pass
        capture = logits_output.beam
        for entry_index, entry in enumerate(tail.entries):
            group = entry.group
            if group.retired or group.state != BeamGroupState.DECODING:
                continue
            else:
                pass
            logits = torch.cat(
                [
                    capture.leader_logits[entry_index : entry_index + 1],
                    capture.tail_logits[entry.start : entry.end],
                ]
            )
            next_token_ids, parent_indices = self.select_group_step(
                group, logits, batch.forward_iter
            )
            self._apply_survivors(  # noqa: leading-underscore  # SGLang BeamCoordinator helper
                group, next_token_ids, parent_indices, batch.forward_iter
            )


def append_encoder_beam_tail(batch: ScheduleBatch) -> None:
    """Append the member rows; each also takes its leader's encoder length."""
    batch_tail.append_beam_tail(batch)
    tail = batch.beam_tail
    if tail is None or not batch.model_config.is_encoder_decoder:
        return
    else:
        pass
    leader_rows = [
        entry.leader_idx
        for entry in tail.entries
        for _ in range(entry.end - entry.start)
    ]
    leader_rows_device = torch.tensor(leader_rows, dtype=torch.int64).to(
        batch.encoder_lens.device, non_blocking=True
    )
    batch.encoder_lens = torch.cat(
        [batch.encoder_lens, batch.encoder_lens[leader_rows_device]]
    )
    batch.encoder_lens_cpu = batch.encoder_lens_cpu + [
        batch.encoder_lens_cpu[row] for row in leader_rows
    ]


def strip_encoder_beam_tail(batch: ScheduleBatch) -> None:
    """Restore the one-row-per-request layout, encoder lengths included."""
    tail = batch.beam_tail
    if tail is not None and batch.model_config.is_encoder_decoder:
        batch.encoder_lens = batch.encoder_lens[: tail.num_base_rows]
        batch.encoder_lens_cpu = batch.encoder_lens_cpu[: tail.num_base_rows]
    else:
        pass
    batch_tail.strip_beam_tail(batch)


def install_encoder_beam_tail() -> None:
    # note (zhumengzhiren): ScheduleBatch calls these module functions directly, and decode allocation reads encoder_lens for every row.
    schedule_batch_module.append_beam_tail = append_encoder_beam_tail
    schedule_batch_module.strip_beam_tail = strip_encoder_beam_tail


__all__ = [
    "OmniBeamCoordinator",
    "OmniBeamGroup",
    "beam_hypotheses",
    "install_encoder_beam_tail",
    "select_beam_candidates",
    "select_final_beam_candidates",
]
