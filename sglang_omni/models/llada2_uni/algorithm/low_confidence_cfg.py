# SPDX-License-Identifier: Apache-2.0
"""Low-confidence decoding with optional grouped classifier-free guidance."""

from __future__ import annotations

from typing import Any, List, Mapping

import torch
from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class LowConfidenceCFG(DllmAlgorithm):
    """Advance ordinary or atomically grouped CFG dLLM blocks in one step.

    The inherited ``run`` method owns both synchronous and first-done-first-out
    execution. This class only performs the CUDA-graph-safe in-place step.
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def run(
        self,
        model_runner: Any,
        forward_batch: ForwardBatch,
        algo_states: List[Any] | None = None,
    ):
        """Run through SGLang's current sync/FDFO implementation.

        CFG metadata is made available to the custom attention backend for
        CUDA-graph replay. A left pad that reaches the active query block uses
        eager attention for this block only, then restores the graph runner.
        """
        if getattr(forward_batch, "omni_dllm_group", None) is None:
            return super().run(model_runner, forward_batch, algo_states)

        graph_runner = getattr(model_runner, "decode_cuda_graph_runner", None)
        candidate_backends = (
            getattr(model_runner, "attn_backend", None),
            getattr(model_runner, "decode_attn_backend", None),
            getattr(graph_runner, "attn_backend", None),
        )
        backends = []
        seen_backend_ids = set()
        for backend in candidate_backends:
            if backend is None or id(backend) in seen_backend_ids:
                continue
            if hasattr(backend, "set_cfg_runtime_forward_batch"):
                backends.append(backend)
                seen_backend_ids.add(id(backend))

        for backend in backends:
            backend.set_cfg_runtime_forward_batch(forward_batch)

        group_is_prefill = bool(
            getattr(forward_batch, "omni_dllm_group_is_prefill", False)
        )
        use_eager = group_is_prefill or self._has_in_query_left_pad(forward_batch)
        if use_eager:
            model_runner.decode_cuda_graph_runner = None
        try:
            if group_is_prefill:
                output = model_runner.forward(forward_batch, pp_proxy_tensors=None)
                return (
                    output.logits_output,
                    [[] for _ in range(int(forward_batch.batch_size))],
                    None,
                    None,
                    output.can_run_graph,
                )
            return super().run(model_runner, forward_batch, algo_states)
        finally:
            if use_eager:
                model_runner.decode_cuda_graph_runner = graph_runner
            for backend in backends:
                backend.set_cfg_runtime_forward_batch(None)

    @staticmethod
    def _has_in_query_left_pad(forward_batch: ForwardBatch) -> bool:
        left_pad_lengths = torch.as_tensor(
            forward_batch.dllm_left_pad_lens_cpu,
            dtype=torch.int64,
            device="cpu",
        )
        prefix_lengths = torch.as_tensor(
            forward_batch.extend_prefix_lens_cpu,
            dtype=torch.int64,
            device="cpu",
        )
        if left_pad_lengths.shape != prefix_lengths.shape:
            raise RuntimeError("CFG padding metadata must match prefix metadata")
        return bool(torch.any(left_pad_lengths > prefix_lengths))

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        del states
        group = getattr(forward_batch, "omni_dllm_group", None)
        if group is None:
            return self._step_independent(forward_batch, full_logits)
        return self._step_cfg(forward_batch, full_logits, group)

    def _step_independent(
        self, forward_batch: ForwardBatch, full_logits: torch.Tensor
    ) -> List[bool]:
        batch_size = forward_batch.batch_size
        vocab_size = full_logits.shape[-1]
        logits = full_logits.view(batch_size, self.block_size, vocab_size)
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        block_mask = input_ids == self.mask_id
        done = block_mask.sum(dim=1) == 0

        image_token_offsets = getattr(
            forward_batch, "omni_dllm_image_token_offsets", None
        )
        if image_token_offsets is not None:
            if image_token_offsets.numel() != batch_size:
                raise RuntimeError(
                    "native image vocabulary metadata must match the dLLM batch"
                )
            vocabulary = torch.arange(vocab_size, device=logits.device)
            valid_vocabulary = vocabulary.view(1, 1, -1) >= (
                image_token_offsets.to(device=logits.device).view(batch_size, 1, 1)
            )
            logits = logits.masked_fill(~valid_vocabulary, -float("inf"))

        predictions, confidence = self._predictions_and_confidence(logits)
        confidence = confidence.masked_fill(~block_mask, -float("inf"))
        transfer = self._transfer_mask(confidence)
        predictions = torch.where(block_mask, predictions, input_ids)
        new_input_ids = torch.where(transfer, predictions, input_ids)
        forward_batch.input_ids.copy_(new_input_ids.reshape(-1))
        return done.tolist()

    def _step_cfg(
        self, forward_batch: ForwardBatch, full_logits: torch.Tensor, group: Any
    ) -> List[bool]:
        batch_size = forward_batch.batch_size
        roles = tuple(group.roles)
        expected_roles = {
            2: ("conditional", "unconditional"),
            3: ("conditional", "unconditional", "no_image"),
        }.get(batch_size)
        if roles != expected_roles:
            raise RuntimeError(
                "grouped CFG roles must be ordered as conditional/unconditional"
                "[/no_image]"
            )

        vocab_size = full_logits.shape[-1]
        logits = full_logits.view(batch_size, self.block_size, vocab_size)
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        block_mask = input_ids[0] == self.mask_id
        done = block_mask.sum() == 0

        args: Mapping[str, Any] = group.algorithm_args
        force_image_only = bool(args.get("force_image_only", False))
        image_token_offset = 0
        if force_image_only:
            configured_offset = args.get("image_token_offset")
            if (
                not isinstance(configured_offset, int)
                or isinstance(configured_offset, bool)
                or not 0 < configured_offset < vocab_size
            ):
                raise ValueError(
                    "image_token_offset must be a checkpoint-provided vocabulary offset"
                )
            image_token_offset = configured_offset

        guided_logits = self._guided_logits(logits, args)
        if force_image_only:
            vocabulary = torch.arange(vocab_size, device=guided_logits.device)
            guided_logits = guided_logits.masked_fill(
                vocabulary.unsqueeze(0) < image_token_offset,
                -float("inf"),
            )
        predictions, confidence = self._predictions_and_confidence(guided_logits)

        confidence = confidence.masked_fill(~block_mask, -float("inf"))
        transfer = self._transfer_mask(confidence.unsqueeze(0)).squeeze(0)
        conditional_ids = input_ids[0]
        predictions = torch.where(block_mask, predictions, conditional_ids)
        updated = torch.where(transfer, predictions, conditional_ids)

        # Every CFG row advances atomically and receives the same sampled block.
        # copy_ preserves the captured input tensor identity for CUDA graphs.
        forward_batch.input_ids.copy_(
            updated.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        )
        return torch.as_tensor(done).expand(batch_size).tolist()

    @staticmethod
    def _guided_logits(logits: torch.Tensor, args: Mapping[str, Any]) -> torch.Tensor:
        cfg_scale = float(args.get("cfg_scale", 1.0))
        conditional = logits[0]
        unconditional = logits[1]
        if logits.shape[0] == 2:
            guided = unconditional + cfg_scale * (conditional - unconditional)
        else:
            no_image = logits[2]
            cfg_image_scale = float(args.get("cfg_image_scale", cfg_scale))
            guided = (
                unconditional
                + cfg_scale * (conditional - unconditional)
                + cfg_image_scale * (unconditional - no_image)
            )

        cfg_rescale = float(args.get("cfg_rescale", 0.0))
        if cfg_rescale:
            dims = tuple(range(1, guided.ndim))
            guided_std = guided.std(dim=dims, keepdim=True).clamp_min(1e-6)
            conditional_std = conditional.std(dim=dims, keepdim=True)
            rescaled = guided * (conditional_std / guided_std)
            guided = cfg_rescale * rescaled + (1.0 - cfg_rescale) * guided
        return guided

    @staticmethod
    def _predictions_and_confidence(
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = torch.argmax(logits, dim=-1)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence = torch.gather(
            probabilities, dim=-1, index=predictions.unsqueeze(-1)
        ).squeeze(-1)
        return predictions, confidence

    def _transfer_mask(self, confidence: torch.Tensor) -> torch.Tensor:
        transfer = confidence > self.threshold
        has_transfer = transfer.any(dim=1)
        top1_indices = torch.argmax(confidence, dim=1)
        fallback = torch.zeros_like(transfer, dtype=torch.bool)
        fallback.scatter_(1, top1_indices.unsqueeze(1), True)
        return torch.where(has_transfer.unsqueeze(1), transfer, fallback)


Algorithm = LowConfidenceCFG
