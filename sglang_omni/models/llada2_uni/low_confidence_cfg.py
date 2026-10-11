# SPDX-License-Identifier: Apache-2.0
"""Classifier-free guidance for LLaDA2 mask diffusion."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from sglang.srt.dllm.algorithm.base import DllmAlgorithm, DllmRunOutput
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidenceCFG(DllmAlgorithm):
    """LowConfidence unmasking with per-step Classifier-Free Guidance."""

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        if self.fdfo:
            raise ValueError("LowConfidenceCFG requires synchronous DLLM, not FDFO")
        else:
            pass
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.image_token_offset = config.algorithm_config.get(
            "image_token_offset", 157184
        )

    def run_block(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        *,
        cond_idx: int = 0,
        no_text_idx: int | None = None,
        no_img_idx: int | None = None,
        cfg_text_scale: float = 1.0,
        cfg_image_scale: float = 0.0,
        cfg_rescale: float = 0.0,
    ) -> tuple[LogitsProcessorOutput | torch.Tensor, list[torch.Tensor], bool]:
        reqs = forward_batch.reqs
        is_cfg = no_text_idx is not None
        if reqs and all(req.is_dllm_prefill() for req in reqs):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph
        else:
            pass

        ids = forward_batch.input_ids.view(forward_batch.batch_size, self.block_size)
        starts = self._block_start_list(
            forward_batch
        )  # noqa: leading-underscore  # SGLang API
        if all(start == self.block_size for start in starts):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph
        else:
            pass
        if is_cfg:
            # note (Anmuliar): companion mask padding is outside the generated suffix.
            starts = [starts[cond_idx]] * forward_batch.batch_size
        else:
            pass

        req = reqs[cond_idx]
        steps = (
            req._dllm_steps or self.block_size
        )  # noqa: leading-underscore  # DLLM protocol
        steps = min(max(steps, 1), self.block_size)
        base, remainder = divmod(self.block_size, steps)
        force_image_only = (
            req._task_kind in ("t2i", "edit") and not req._is_thinking_phase1
        )  # noqa: leading-underscore  # DLLM protocol
        active_ids = ids[cond_idx : cond_idx + 1] if is_cfg else ids

        for step in range(steps):
            mask = active_ids == self.mask_id
            if not mask.any().item():
                break
            else:
                pass
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits = out.logits_output.full_logits.view(
                forward_batch.batch_size, self.block_size, -1
            )
            if is_cfg:
                cond_logits = logits[cond_idx]
                no_text_logits = logits[no_text_idx]
                guided = no_text_logits + cfg_text_scale * (
                    cond_logits - no_text_logits
                )
                if no_img_idx is not None:
                    guided = guided + cfg_image_scale * (
                        no_text_logits - logits[no_img_idx]
                    )
                else:
                    pass
                if cfg_rescale > 0:
                    rescaled = guided * (
                        cond_logits.std(dim=-1, keepdim=True)
                        / (guided.std(dim=-1, keepdim=True) + 1e-6)
                    )
                    guided = cfg_rescale * rescaled + (1.0 - cfg_rescale) * guided
                else:
                    pass
                logits = guided.unsqueeze(0)
            else:
                pass

            num_to_transfer = base + (step < remainder)
            for row, row_logits, row_mask in zip(active_ids, logits, mask):
                if force_image_only:
                    row_logits[:, : self.image_token_offset] = -torch.inf
                else:
                    pass
                predicted_ids = row_logits.argmax(dim=-1)
                confidence = (
                    F.softmax(row_logits, dim=-1)
                    .gather(-1, predicted_ids.unsqueeze(-1))
                    .squeeze(-1)
                )
                confidence = confidence.masked_fill(~row_mask, -torch.inf)
                top_indices = confidence.topk(num_to_transfer).indices
                keep = (confidence > self.threshold).scatter(0, top_indices, True)
                # note (Anmuliar): the quota may exceed the remaining masked suffix.
                keep &= row_mask
                row.copy_(torch.where(keep, predicted_ids, row))
                if is_cfg:
                    for branch in range(forward_batch.batch_size):
                        if branch != cond_idx:
                            ids[branch].copy_(
                                torch.where(keep, predicted_ids, ids[branch])
                            )
                        else:
                            pass
                else:
                    pass

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        return (
            out.logits_output,
            [row[start:] for row, start in zip(ids, starts)],
            out.can_run_graph,
        )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: None = None,
    ) -> DllmRunOutput:
        """Run synchronous CFG generation through SGLang's DLLM contract."""
        if algo_states is not None:
            raise ValueError("LowConfidenceCFG does not accept carried FDFO state")
        else:
            pass
        reqs = forward_batch.reqs
        batch_size = forward_batch.batch_size

        cfg_marked = bool(reqs) and any(  # noqa: leading-underscore  # DLLM protocol
            req._is_uncond or req._cfg_group_rid is not None for req in reqs
        )
        if cfg_marked:
            cond_indices = [  # noqa: leading-underscore  # DLLM protocol
                i for i, req in enumerate(reqs) if not req._is_uncond
            ]
            uncond_text_indices = [  # noqa: leading-underscore  # DLLM protocol
                i
                for i, req in enumerate(reqs)
                if req._is_uncond and not req._is_uncond_img
            ]
            uncond_img_indices = [  # noqa: leading-underscore  # DLLM protocol
                i for i, req in enumerate(reqs) if req._is_uncond_img
            ]
            valid_roles = (
                len(reqs) == batch_size
                and len(cond_indices) == 1
                and len(uncond_text_indices) == 1
                and (
                    (batch_size == 2 and not uncond_img_indices)
                    or (batch_size == 3 and len(uncond_img_indices) == 1)
                )
            )
            cond_idx = cond_indices[0] if len(cond_indices) == 1 else None
            cond_rid = reqs[cond_idx].rid if cond_idx is not None else None
            valid_group = (
                cond_rid is not None
                and all(  # noqa: leading-underscore  # DLLM protocol
                    req._cfg_group_rid == cond_rid for req in reqs
                )
            )
            if not valid_roles or not valid_group:
                raise RuntimeError(
                    "Malformed CFG batch: expected one conditional request, "
                    "one no-text companion, and for batch=3 one no-image "
                    "companion from the same request group"
                )
            else:
                pass

            cond_req = reqs[cond_idx]
            uncond_text_idx = uncond_text_indices[0]
            cfg_text_scale = float(
                cond_req._cfg_scale
            )  # noqa: leading-underscore  # DLLM protocol
            cfg_rescale = float(
                cond_req._cfg_rescale
            )  # noqa: leading-underscore  # DLLM protocol
            uncond_img_idx = uncond_img_indices[0] if uncond_img_indices else None
            cfg_image_scale = (
                float(
                    cond_req._cfg_image_scale
                )  # noqa: leading-underscore  # DLLM protocol
                if uncond_img_idx is not None
                else 0.0
            )
            for name, value in (
                ("_cfg_scale", cfg_text_scale),
                ("_cfg_rescale", cfg_rescale),
                ("_cfg_image_scale", cfg_image_scale),
            ):
                if not math.isfinite(value):
                    raise ValueError(f"{name} must be finite")
                else:
                    pass
            result = self.run_block(
                model_runner,
                forward_batch,
                cond_idx=cond_idx,
                no_text_idx=uncond_text_idx,
                no_img_idx=uncond_img_idx,
                cfg_text_scale=cfg_text_scale,
                cfg_image_scale=cfg_image_scale,
                cfg_rescale=cfg_rescale,
            )
        else:
            result = self.run_block(model_runner, forward_batch)
        logits, token_ids, can_run_graph = result
        return logits, token_ids, None, None, can_run_graph


Algorithm = LowConfidenceCFG
