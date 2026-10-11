# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o talker runner: condition-embeds prefill + windowed rep penalty."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

from sglang_omni.model_runner.base import (
    ModelRunner,
    current_sglang_sampling_backend,
    rank_shared_unseeded_sampling_seed,
)
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.minicpm_o.talker_session import TalkerUnitRequestData
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK, resolve_row_seed
from sglang_omni.scheduling.sglang_backend.request_data import (
    SGLangARRequestData,
    session_prefill_rows,
)
from sglang_omni.scheduling.types import SchedulerRequest

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
else:
    pass

logger = logging.getLogger(__name__)

# note (MayDomine): the checkpoint penalizes only the most recent 16 codec tokens.
REP_PENALTY_WINDOW = 16


@dataclass(kw_only=True)
class TalkerSlotState:
    """GPU-resident per-slot state for async decode.

    Token i of a request sits at windows[slot, i % REP_PENALTY_WINDOW]; the spare slot
    takes the padded rows of a sample graph.
    """

    windows: torch.Tensor
    generated: torch.Tensor
    min_new_tokens: torch.Tensor
    penalties: torch.Tensor
    seeds: torch.Tensor
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor
    suppress: torch.Tensor
    vocab: int
    eos_id: int
    spare: int

    @classmethod
    def allocate(
        cls, slots: int, vocab: int, eos_id: int, device: torch.device
    ) -> TalkerSlotState:
        rows = slots + 1
        return cls(
            windows=torch.full(
                (rows, REP_PENALTY_WINDOW), vocab, dtype=torch.long, device=device
            ),
            generated=torch.zeros(rows, dtype=torch.long, device=device),
            min_new_tokens=torch.zeros(rows, dtype=torch.long, device=device),
            penalties=torch.ones(rows, dtype=torch.float32, device=device),
            seeds=torch.zeros(rows, dtype=torch.long, device=device),
            temperatures=torch.ones(rows, 1, dtype=torch.float32, device=device),
            top_ps=torch.ones(rows, dtype=torch.float32, device=device),
            top_ks=torch.full((rows,), vocab, dtype=torch.int32, device=device),
            min_ps=torch.zeros(rows, dtype=torch.float32, device=device),
            suppress=torch.zeros(rows, vocab, dtype=torch.bool, device=device),
            vocab=vocab,
            eos_id=eos_id,
            spare=slots,
        )

    def reset(
        self,
        rows: torch.Tensor,
        requests: list[SchedulerRequest],
        sampling_info: SamplingBatchInfo,
        suppress: torch.Tensor,
    ) -> None:
        count = len(requests)
        windows = torch.full((count, REP_PENALTY_WINDOW), self.vocab, dtype=torch.long)
        generated = torch.zeros(count, dtype=torch.long)
        min_new_tokens = torch.zeros(count, dtype=torch.long)
        penalties = torch.ones(count, dtype=torch.float32)
        seeds = torch.zeros(count, dtype=torch.long)
        for row, sched_req in enumerate(requests):
            req = sched_req.data.req
            length = len(req.output_ids)
            for index in range(max(0, length - REP_PENALTY_WINDOW), length):
                windows[row, index % REP_PENALTY_WINDOW] = int(req.output_ids[index])
            generated[row] = length
            inputs = sched_req.data.talker_model_inputs
            min_new_tokens[row] = int(inputs.get("min_new_tokens", 0))
            penalties[row] = float(inputs.get("rep_penalty", 1.0))
            seed = req.sampling_params.sampling_seed
            if seed is None:
                seed = rank_shared_unseeded_sampling_seed(sched_req, row)
            elif not (0 <= seed <= SAMPLING_SEED_MASK):
                seed = resolve_row_seed(seed)
                req.sampling_params.sampling_seed = seed
            else:
                pass
            seeds[row] = seed
        device = rows.device
        self.windows[rows] = windows.to(device)
        self.generated[rows] = generated.to(device)
        self.min_new_tokens[rows] = min_new_tokens.to(device)
        self.penalties[rows] = penalties.to(device)
        if sampling_info.sampling_seed is None:
            self.seeds[rows] = seeds.to(device)
        else:
            # note (0xtoward): deterministic inference has already seeded every row.
            self.seeds[rows] = sampling_info.sampling_seed[:count]
        self.temperatures[rows] = sampling_info.temperatures[:count].view(count, 1)
        self.top_ps[rows] = sampling_info.top_ps[:count].to(torch.float32)
        self.top_ks[rows] = sampling_info.top_ks[:count].to(torch.int32)
        self.min_ps[rows] = sampling_info.min_ps[:count].to(torch.float32)
        self.suppress[rows] = suppress

    def sampling_info(self, rows: torch.Tensor) -> SamplingBatchInfo:
        return SamplingBatchInfo(
            temperatures=self.temperatures[rows],
            top_ps=self.top_ps[rows],
            top_ks=self.top_ks[rows],
            min_ps=self.min_ps[rows],
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=True,
            need_top_k_sampling=True,
            need_min_p_sampling=False,
            vocab_size=self.vocab,
            grammars=[],
            penalizer_orchestrator=None,
            has_custom_logit_processor=False,
            custom_params=None,
            custom_logit_processor=None,
            sampling_seed=self.seeds[rows],
            device=self.seeds.device.type,
            logit_bias=None,
        )

    def apply(self, logits: torch.Tensor, rows: torch.Tensor) -> None:
        logits.masked_fill_(self.suppress[rows], float("-inf"))
        windows = self.windows[rows]
        counts = torch.zeros(
            len(rows), self.vocab + 1, dtype=torch.float32, device=logits.device
        )
        counts.scatter_add_(1, windows, torch.ones_like(windows, dtype=torch.float32))
        counts = counts[:, : self.vocab]
        factors = self.penalties[rows].unsqueeze(1) ** counts
        scores = logits.to(torch.float32)
        penalized = torch.where(scores < 0, scores * factors, scores / factors)
        logits.copy_(torch.where(counts > 0, penalized, scores).to(logits.dtype))
        eos_logits = logits[:, self.eos_id]
        logits[:, self.eos_id] = torch.where(
            self.generated[rows] < self.min_new_tokens[rows],
            torch.full_like(eos_logits, float("-inf")),
            eos_logits,
        )

    def append(self, rows: torch.Tensor, next_token_ids: torch.Tensor) -> None:
        generated = self.generated[rows]
        self.windows[rows, generated % REP_PENALTY_WINDOW] = next_token_ids.long()
        self.generated[rows] = generated + 1


class TalkerSampleGraphs:
    """CUDA graphs of the device-side sampling step, one per decode graph batch size.

    Every size is captured at construction, before serving; the padded rows of a
    replay sample on the spare slot.
    """

    def __init__(
        self,
        state: TalkerSlotState,
        sampler: Sampler,
        batch_sizes: list[int],
        logits_dtype: torch.dtype,
        backend: DeviceGraphBackend,
    ) -> None:
        self.state = state
        self.sampler = sampler
        self.batch_sizes = sorted(set(batch_sizes))
        largest_batch_size = self.batch_sizes[-1]
        device = state.seeds.device
        self.logits = torch.zeros(
            largest_batch_size, state.vocab, dtype=logits_dtype, device=device
        )
        self.rows = torch.full(
            (largest_batch_size,), state.spare, dtype=torch.long, device=device
        )
        self.positions = torch.zeros(
            largest_batch_size, dtype=torch.long, device=device
        )
        self.next_token_ids = torch.zeros(
            largest_batch_size, dtype=torch.int32, device=device
        )
        self.graphs: dict[int, ReplayableGraph] = {}
        device_module = torch.get_device_module(device)
        pool = backend.graph_pool_handle()
        stream = device_module.Stream(device=device)
        # note (0xtoward): largest first on one stream, so smaller graphs reuse its pool.
        for batch_size in reversed(self.batch_sizes):
            logits = self.logits[:batch_size]
            rows = self.rows[:batch_size]
            positions = self.positions[:batch_size]
            stream.wait_stream(device_module.current_stream(device))
            with device_module.stream(stream):
                # note (0xtoward): warm-ups on the spare slot settle lazy sampler state.
                for _ in range(2):
                    self.step(logits, rows, positions)
            # note (0xtoward): code2wav runs on other threads of this process, so a
            # capture failure there must not abort this one.
            with backend.capture(
                pool=pool, stream=stream, thread_local_errors=True
            ) as graph:
                self.next_token_ids[:batch_size].copy_(
                    self.step(logits, rows, positions)
                )
            device_module.current_stream(device).wait_stream(stream)
            self.graphs[batch_size] = graph

    def step(
        self, logits: torch.Tensor, rows: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        self.state.apply(logits, rows)
        next_token_ids = self.sampler(
            LogitsProcessorOutput(next_token_logits=logits, hidden_states=None),
            self.state.sampling_info(rows),
            False,
            [0] * len(rows),
            [[] for _ in range(len(rows))],
            positions,
        )
        self.state.append(rows, next_token_ids)
        return next_token_ids

    def fits(self, batch_size: int) -> bool:
        return batch_size <= self.batch_sizes[-1]

    def sample(
        self, logits: torch.Tensor, rows: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        count = len(rows)
        batch_size = next(size for size in self.batch_sizes if size >= count)
        self.logits[:count].copy_(logits)
        self.rows[:count].copy_(rows)
        self.rows[count:batch_size].fill_(self.state.spare)
        self.positions[:count].copy_(positions)
        self.graphs[batch_size].replay()
        return self.next_token_ids[:count].clone()


class MiniCPMOTalkerModelRunner(ModelRunner):
    """Prefill codec conditions and apply a frequency penalty over recent tokens."""

    slot_state: TalkerSlotState | None = None
    sample_graphs: TalkerSampleGraphs | None = None

    @torch.no_grad()
    def enable_device_sampling(self) -> None:
        """Keep the decode step on the GPU, with sampling graphs where they are supported.

        Each capability falls back on its own: without the graphs the decode step still
        samples on the device, and without the slot state the runner samples on the host.
        """
        self.slot_state = TalkerSlotState.allocate(
            self.tp_worker.model_runner.req_to_token_pool.req_to_token.shape[0],
            self.model.num_audio_tokens,
            self.model.codec_eos_id,
            self.device,
        )
        self.sample_graphs = self.capture_sample_graphs()

    def capture_sample_graphs(self) -> TalkerSampleGraphs | None:
        """The sampling graphs of this deployment, or None to sample eagerly."""
        decode_graphs = self.tp_worker.model_runner.decode_cuda_graph_runner
        backend = current_platform.get_device_graph_backend(self.device)
        if backend is None:
            reason = f"{self.device.type} records no model-owned graph"
        elif current_sglang_sampling_backend() != "pytorch":
            reason = "the sampling graph needs the pytorch sampling backend"
        elif decode_graphs is None:
            reason = (
                "the decode CUDA graphs this deployment would share were not captured"
            )
        else:
            reason = None
        if reason is not None:
            logger.info(f"MiniCPM-o talker samples eagerly: {reason}")
            return None
        else:
            pass
        try:
            graphs = TalkerSampleGraphs(
                self.slot_state,
                self.tp_worker.model_runner.sampler,
                [int(batch_size) for batch_size in decode_graphs.capture_bs],
                self.model.head_code.weight.dtype,
                backend,
            )
        except Exception:
            logger.exception(
                "MiniCPM-o talker sampling graph capture failed; sampling eagerly"
            )
            return None
        logger.info(
            f"MiniCPM-o talker captured sampling graphs for batch sizes {graphs.batch_sizes}"
        )
        return graphs

    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Prepare request embeddings; schedule_batch follows the runner interface."""
        parts: list[torch.Tensor] = []
        for sched_req in requests:
            data = sched_req.data
            if isinstance(data, TalkerUnitRequestData):
                parts.append(
                    session_prefill_rows(
                        data, self.model.emb_code, self.model.emb_code.weight.device
                    )
                )
                continue
            else:
                pass
            tensor = data.prefill_input_embeds
            if tensor is None:
                raise RuntimeError(
                    "MiniCPM-o talker prefill requires condition embeddings"
                )
            else:
                pass
            req = data.req
            prefix_len = len(req.prefix_indices)
            end = prefix_len + int(req.extend_range.length)
            prompt_len = int(tensor.shape[0])
            if prefix_len < prompt_len:
                parts.append(tensor[prefix_len : min(end, prompt_len)])
            else:
                pass
            if end > prompt_len:
                # note (MayDomine): retracted requests replay already-generated tokens.
                fill_ids = req.get_fill_ids()
                generated = torch.tensor(
                    fill_ids[max(prefix_len, prompt_len) : end],
                    dtype=torch.long,
                    device=self.model.emb_code.weight.device,
                )
                parts.append(self.model.emb_code(generated))
            else:
                pass
        input_embeds = torch.cat(parts, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=self.model.emb_code.weight.dtype,
        )
        expected_rows = int(forward_batch.input_ids.shape[0])
        if input_embeds.shape[0] != expected_rows:
            raise RuntimeError(
                "Talker prefill embeds must align with forward input_ids: "
                f"got {input_embeds.shape[0]} rows for {expected_rows} input ids"
            )
        else:
            pass
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=input_embeds,
                input_embeds_are_projected=True,
            ),
        )

    def sample_next_token_ids(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        if self.slot_state is None:
            return super().sample_next_token_ids(
                logits_output, forward_batch, schedule_batch, requests
            )
        else:
            pass
        if any(sched_req.data.return_logprob for sched_req in requests):
            raise ValueError("the talker's device sampler does not return logprobs")
        else:
            pass
        logits = logits_output.next_token_logits[: len(requests)]
        rows = forward_batch.req_pool_indices[: len(requests)]
        sampling_info = forward_batch.sampling_info
        if forward_batch.forward_mode.is_extend():
            self.slot_state.reset(
                rows, requests, sampling_info, self.suppress_mask(requests, logits)
            )
        elif self.samples_in_graph(len(requests), sampling_info):
            return self.sample_graphs.sample(
                logits, rows, forward_batch.positions[: len(requests)]
            )
        else:
            pass
        self.slot_state.apply(logits, rows)
        if any(
            sched_req.data.req.sampling_params.sampling_seed is not None
            for sched_req in requests
        ):
            self.validate_seeded_sampling_supported(sampling_info)
            sampling_info.sampling_seed = self.slot_state.seeds[rows]
        else:
            pass
        next_token_ids = self.tp_worker.model_runner.sample(
            logits_output, forward_batch
        )
        self.slot_state.append(rows, next_token_ids[: len(requests)])
        return next_token_ids

    def suppress_mask(
        self, requests: list[SchedulerRequest], logits: torch.Tensor
    ) -> torch.Tensor:
        probe = LogitsProcessorOutput(
            next_token_logits=torch.zeros_like(logits, dtype=torch.float32),
            hidden_states=None,
        )
        self.apply_codec_suppress_tokens(probe, requests)
        return torch.isinf(probe.next_token_logits)

    def samples_in_graph(
        self, batch_size: int, sampling_info: SamplingBatchInfo
    ) -> bool:
        # note (0xtoward): the graph replays SGLang's sorted top-k/top-p draw; greedy and
        # unfiltered batches take another draw there, so they sample eagerly.
        return (
            self.sample_graphs is not None
            and self.sample_graphs.fits(batch_size)
            and not sampling_info.is_all_greedy
            and (sampling_info.need_top_k_sampling or sampling_info.need_top_p_sampling)
            and not (
                sampling_info.need_min_p_sampling
                or sampling_info.grammars
                or sampling_info.has_custom_logit_processor
                or sampling_info.logit_bias is not None
            )
        )

    def process_sampling_logits(
        self, logits_output: LogitsProcessorOutput, requests: list[SchedulerRequest]
    ) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        else:
            pass
        vocab = logits.shape[1]
        device = logits.device
        penalized_rows: list[int] = []
        penalties: list[float] = []
        windows: list[list[int]] = []
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            penalty = float(data.talker_model_inputs.get("rep_penalty", 1.0))
            if penalty == 1.0:
                continue
            else:
                pass
            window = [
                tok
                for tok in map(int, data.req.output_ids[-REP_PENALTY_WINDOW:])
                if 0 <= tok < vocab
            ]
            if not window:
                continue
            else:
                pass
            penalized_rows.append(row_idx)
            penalties.append(penalty)
            windows.append(window)
        if not penalized_rows:
            return
        else:
            pass
        # note (MayDomine): a dummy vocabulary bin excludes ragged-window padding.
        num = len(windows)
        window_ids = torch.full((num, REP_PENALTY_WINDOW), vocab, dtype=torch.long)
        for i, window in enumerate(windows):
            window_ids[i, : len(window)] = torch.tensor(window, dtype=torch.long)
        window_ids = window_ids.to(device)
        counts = torch.zeros(num, vocab + 1, dtype=torch.float32, device=device)
        counts.scatter_add_(
            1, window_ids, torch.ones_like(window_ids, dtype=torch.float32)
        )
        counts = counts[:, :vocab]
        alphas = (
            torch.tensor(penalties, dtype=torch.float32, device=device).unsqueeze(1)
            ** counts
        )
        rows_t = torch.tensor(penalized_rows, dtype=torch.long, device=device)
        orig_dtype = logits.dtype
        scores = logits[rows_t].to(torch.float32)
        penalized = torch.where(scores < 0, scores * alphas, scores / alphas)
        scores = torch.where(counts > 0, penalized, scores)
        logits[rows_t] = scores.to(orig_dtype)

    def on_request_finished(self, request_id: str, data: SGLangARRequestData) -> None:
        if isinstance(data, TalkerUnitRequestData):
            # note (Junnan Li): The last sample has no KV and is not committed by chunk TTS.
            data.req.output_ids = data.req.output_ids[:-1]
            data.req.finished_len = len(data.req.output_ids)
        else:
            pass
        super().on_request_finished(request_id, data)
