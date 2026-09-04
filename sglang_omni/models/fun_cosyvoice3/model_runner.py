# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.mlx_model_worker import MlxSchedulerModelRunner
from sglang_omni.model_runner.sglang_execution import attn_forward_context
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK

from .sglang_model import VOCAB_SIZE


class FunCosyVoice3ModelRunner(ModelRunner):
    """Runs Fun-CosyVoice3 AR steps and collects generated speech tokens."""

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        return self._forward_with_input_embeds(forward_batch, input_embeds)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_tokens(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_tokens(result, forward_batch, schedule_batch, requests)

    def sample_before_post_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> bool:
        """Sample the first speech token before collecting prefill output."""
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> bool:
        """Sample each speech token before collecting decode output."""
        del forward_batch, schedule_batch, requests
        return True

    def _sample_next_token_ids(
        self,
        logits_output: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> Any:
        if logits_output.next_token_logits.device.type != "mps":
            return super()._sample_next_token_ids(
                logits_output,
                forward_batch,
                schedule_batch,
                requests,
            )
        if len(requests) != 1:
            raise RuntimeError(
                "Fun-CosyVoice3 Torch MPS currently requires " "max_running_requests=1"
            )

        self._apply_repetition_penalty(logits_output, requests)
        self._apply_codec_suppress_tokens(logits_output, requests)
        self._install_sampling_seeds(forward_batch, requests)
        sampling_info = forward_batch.sampling_info
        installed_seeds = sampling_info.sampling_seed
        rng_context = nullcontext()
        if installed_seeds is not None:
            # SGLang's seeded PyTorch sampler converts filtered probabilities
            # to float64, which MPS cannot represent. Keep the same top-k,
            # top-p, temperature, stop, and repetition processing, but drive
            # MPS multinomial from a stable per-request/per-step RNG seed.
            sampling_params = requests[0].data.req.sampling_params
            row_seed = (
                int(sampling_params.sampling_seed)
                if sampling_params.sampling_seed is not None
                else 42
            )
            req = requests[0].data.req
            absolute_position = len(req.origin_input_ids) + len(req.output_ids) - 1
            step_seed = (row_seed + absolute_position * 0x9E3779B1) & SAMPLING_SEED_MASK
            device_index = logits_output.next_token_logits.device.index or 0
            rng_context = torch.random.fork_rng(
                devices=[device_index],
                device_type="mps",
            )
            sampling_info.sampling_seed = None

        wants_rollout_logprob = any(sr.data.return_logprob for sr in requests)
        if wants_rollout_logprob:
            self._enable_sampler_logprobs(forward_batch, len(requests))
        try:
            with rng_context:
                if installed_seeds is not None:
                    torch.manual_seed(step_seed)
                # SGLang decorates its repetition-penalty helper with
                # torch.compile independently of enable_torch_compile.
                # Scope the public eager stance to sampling so an opt-in
                # compile in another pipeline stage remains unaffected.
                with torch.compiler.set_stance("force_eager"):
                    next_token_ids = self.tp_worker.model_runner.sample(
                        logits_output,
                        forward_batch,
                    )
        finally:
            sampling_info.sampling_seed = installed_seeds
        if wants_rollout_logprob:
            next_token_logprobs = logits_output.next_token_logprobs
            if next_token_logprobs is None:
                raise RuntimeError(
                    "Sampler did not populate next_token_logprobs when "
                    "return_logprob is enabled"
                )
            self._record_rollout_logprobs(
                next_token_logprobs,
                next_token_ids,
                requests,
            )
        return next_token_ids

    def _collect_tokens(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if result.next_token_ids is None:
            return
        token_ids = result.next_token_ids
        if token_ids.ndim != 1:
            token_ids = token_ids.reshape(-1)
        # note: copy the whole batch to host once instead of calling
        # ``.item()`` per request — each ``.item()`` forces its own
        # host/GPU synchronization, which is a per-decode-step cost that
        # scales with batch size.
        token_ids_cpu = token_ids.tolist()
        for idx, sched_req in enumerate(requests):
            token_id = int(token_ids_cpu[idx])
            if token_id >= VOCAB_SIZE:
                continue
            sched_req.data.output_codes.append(
                torch.tensor([token_id], dtype=torch.long)
            )

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
            prompt_embeds = data.prompt_input_embeds
            if prompt_embeds is None:
                raise RuntimeError(
                    "Fun-CosyVoice3 prefill requires prompt_input_embeds"
                )
            pieces.append(prompt_embeds[prefix_len : prefix_len + req_len])
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        input_embeds: torch.Tensor,
    ) -> GenerationBatchResult:
        model_runner = self.tp_worker.model_runner
        model_dtype = next(self.model.parameters()).dtype
        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=model_dtype,
        )
        with attn_forward_context(model_runner.attn_backend):
            logits_output = self.model(
                input_ids=forward_batch.input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )


class FunCosyVoice3MlxSchedulerModelRunner(MlxSchedulerModelRunner):
    """MLX scheduler bridge that records generated speech-code tokens.

    ``MlxSchedulerModelRunner`` finalizes lazy launches directly through its
    shared ``_finalize`` path, so the Torch runner's phase hooks are not used.
    ``post_process_outputs`` is the common point for both sync and lookahead
    MLX execution and runs after the worker has materialized the sampled ids.
    """

    def lookahead_eligible(self, batch: Any) -> bool:
        if len(batch.reqs) != 1 or batch.has_grammar:
            return False
        previous = self._last_mlx_pending
        if previous is not None:
            previous_ids = [req.rid for req in previous.reqs]
            current_ids = [req.rid for req in batch.reqs]
            if previous.launch.mode != "decode" or previous_ids != current_ids:
                return False
        req = batch.reqs[0]
        sampling_params = req.sampling_params
        return (
            sampling_params.frequency_penalty == 0.0
            and sampling_params.presence_penalty == 0.0
            and req.custom_logit_processor is None
        )

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, Any],
    ) -> None:
        del outputs
        token_ids = result.next_token_ids
        if token_ids is None:
            return
        token_ids = token_ids.reshape(-1).tolist()
        requests = scheduler_output.requests
        if len(token_ids) != len(requests):
            raise RuntimeError(
                "Fun-CosyVoice3 MLX sampled-token row count does not match "
                f"the scheduler batch ({len(token_ids)} != {len(requests)})"
            )
        for sched_req, token_id in zip(requests, token_ids, strict=True):
            if sched_req.request_id in self._resolve_skip_rids:
                continue
            token_id = int(token_id)
            if 0 <= token_id < VOCAB_SIZE:
                sched_req.data.output_codes.append(
                    torch.tensor([token_id], dtype=torch.long)
                )
