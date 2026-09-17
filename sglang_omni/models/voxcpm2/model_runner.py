# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 AR runner: drives the diffusion half of every decode step."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.voxcpm2.request_builders import VoxCPM2SGLangRequestData


class VoxCPM2ModelRunner(ModelRunner):
    """Runs the local DiT after each AR forward and feeds the result back."""

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        if self.model.graph_feedback_buffer is not None:
            return CaptureHiddenMode.FULL
        return CaptureHiddenMode.LAST

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        """Turn each request's prefill layout into the embedding sequence.

        The AR stacks read continuous embeddings, not token ids: text positions
        come from the token table and audio positions from the local encoder,
        so nothing upstream of the model can produce this tensor.
        """
        if not requests:
            return
        embeds = []
        masks = []
        for request in requests:
            prefill = request.data.prefill
            # note (Xinhao Tan): embeddings and masks cover the full prompt,
            # but a radix hit would schedule only its uncached suffix.
            reused = len(request.data.req.prefix_indices)
            if reused:
                raise RuntimeError(
                    "VoxCPM2 does not support radix prefix reuse: "
                    f"{reused} positions were reused"
                )
            prompt = self.model.build_input_embeds(
                prefill.text_token,
                prefill.audio_feat,
                prefill.text_mask,
                prefill.audio_mask,
            )
            mask = prefill.audio_mask
            generated = int(request.data.req.extend_range.length) - len(prompt)
            history = request.data.decode_input_embeds
            if generated < 0 or len(history) != generated:
                raise RuntimeError(
                    "VoxCPM2 prefill audio history mismatch: "
                    f"have {len(history)} rows, need {generated}"
                )
            if generated:
                # note (Xinhao Tan): retraction frees KV but keeps output IDs.
                # Rebuild those audio positions from their original embeddings
                # without sampling old patches again or advancing their RNG.
                prompt = torch.cat((prompt, torch.stack(history)), dim=0)
                mask = torch.cat((mask, mask.new_ones(generated)), dim=0)
            embeds.append(prompt)
            masks.append(mask)
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=torch.cat(embeds, dim=0),
                audio_mask=torch.cat(masks, dim=0),
            ),
        )

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        """Hand the step its input: the embedding the previous step produced.

        A captured graph reads the model's own buffer, but the eager path reads
        forward_batch, so the embedding has to reach both.
        """
        embeds = [request.data.next_embed for request in requests]
        if not embeds or any(embed is None for embed in embeds):
            return
        stacked = torch.stack(embeds, dim=0)
        if self.model.graph_feedback_buffer is not None:
            self.model.write_feedback(stacked)
        else:
            forward_batch.input_embeds = stacked

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        if bool(getattr(schedule_batch, "is_prefill_only", False)) or not requests:
            return
        self.model.set_hidden_states(result.logits_output.hidden_states)
        self.advance(requests, rows=self.prefill_rows(requests), is_prefill=True)

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        if requests:
            self.model.set_hidden_states(result.logits_output.hidden_states)
            self.advance(requests, rows=None, is_prefill=False)

    @staticmethod
    def prefill_rows(requests: list) -> torch.Tensor:
        """Index of each request's final prompt position in the packed batch."""
        indices: list[int] = []
        offset = 0
        for request in requests:
            offset += int(request.data.req.extend_range.length)
            indices.append(offset - 1)
        return torch.tensor(indices, dtype=torch.long)

    def advance(
        self, requests: list, *, rows: torch.Tensor | None, is_prefill: bool
    ) -> None:
        """Sample one latent patch per request and stage the next step's input."""
        rows_data = [request.data for request in requests]
        if rows is None:
            rows = torch.arange(len(rows_data), dtype=torch.long)

        patches: list[Any] = [None] * len(rows_data)
        embeddings: list[Any] = [None] * len(rows_data)
        for group in recipe_groups(rows_data):
            group_data = [rows_data[index] for index in group]
            group_patches, group_embeddings = self.model.decode_patch(
                self.batch_cond(group_data),
                inference_timesteps=int(group_data[0].state.inference_timesteps),
                cfg_value=float(group_data[0].state.cfg_value),
                rows=rows[torch.tensor(group, dtype=torch.long)],
                noise=self.batch_noise(group_data),
            )
            for slot, index in enumerate(group):
                patches[index] = group_patches[slot : slot + 1]
                embeddings[index] = group_embeddings[slot]
        stop_flags = self.model.stop_flags(rows)

        for index, data in enumerate(rows_data):
            patch = patches[index]
            data.cond = patch
            data.next_embed = embeddings[index]
            # note (Xinhao Tan): retain the newest feedback too: a request can
            # be retracted before that embedding is consumed by decode.
            data.decode_input_embeds.append(data.next_embed.detach().clone())
            data.latent_patches.append(patch.squeeze(0).detach().cpu())
            state = data.state
            steps = len(data.latent_patches)
            # note (Xinhao Tan): upstream compares a zero-based patch index;
            # comparing the patch count would allow stopping one patch early.
            if steps - 1 > state.min_len and bool(stop_flags[index]):
                data.finish_reason = "stop"
            elif steps >= state.max_len:
                data.finish_reason = "length"
            # note (Xinhao Tan): prefill skips already-finished requests before
            # releasing their KV slots. Its token limit must finish the first
            # patch through the upstream result processor instead.
            if data.finish_reason is not None and not is_prefill:
                data.req.finished_reason = FINISH_MATCHED_TOKEN(0)

    def batch_noise(
        self, rows_data: list[VoxCPM2SGLangRequestData]
    ) -> torch.Tensor | None:
        """Draw this step's flow-matching noise per request, when a seed asks.

        None hands the draw back to the sampler, which is what an unseeded
        request wants: one fused draw for the batch, exactly as upstream. Only
        a seeded request pays for the per-row draw its generator requires.
        """
        if all(data.noise_generator is None for data in rows_data):
            return None
        parameter = next(self.model.parameters())
        shape = (1, self.model.feat_dim, self.model.patch_size)
        rows = [
            torch.randn(shape, generator=data.noise_generator) for data in rows_data
        ]
        return torch.cat(rows, dim=0).to(device=parameter.device, dtype=parameter.dtype)

    def batch_cond(self, rows_data: list[VoxCPM2SGLangRequestData]) -> torch.Tensor:
        """Stack each request's previous patch into the DiT's condition batch."""
        parameter = next(self.model.parameters())
        zeros = torch.zeros(
            (1, self.model.patch_size, self.model.feat_dim),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return torch.cat(
            [
                (
                    zeros
                    if data.cond is None
                    else data.cond.to(device=parameter.device, dtype=parameter.dtype)
                )
                for data in rows_data
            ],
            dim=0,
        )


def recipe_groups(rows_data: list[VoxCPM2SGLangRequestData]) -> list[list[int]]:
    """Split a batch into the runs of requests that share one sampling recipe.

    The DiT samples its batch in lockstep, so one call cannot honor a differing
    step count or guidance scale. Grouping lets requests that differ still
    share the AR forward and split only the diffusion step; refusing the batch
    instead would let one caller's parameters fail everyone beside them.
    """
    groups: dict[tuple[int, float], list[int]] = {}
    for index, data in enumerate(rows_data):
        key = (int(data.state.inference_timesteps), float(data.state.cfg_value))
        groups.setdefault(key, []).append(index)
    return list(groups.values())


__all__ = ["VoxCPM2ModelRunner"]
