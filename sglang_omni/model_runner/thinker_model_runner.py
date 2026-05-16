# SPDX-License-Identifier: Apache-2.0
"""Thinker model runner — injects multimodal embeddings before forward.

Handles image/video/audio token → embedding replacement and deepstack
visual embeddings for Qwen3-Omni's thinker stage.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner

logger = logging.getLogger(__name__)


class ThinkerModelRunner(ModelRunner):
    """Thinker: injects multimodal embeddings in the prefill phase."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)

        model = self.model
        self._outer_model = model.thinker
        self._text_model = self._outer_model.model
        self._embed_tokens = self._text_model.embed_tokens

        thinker_cfg = tp_worker.model_runner.model_config.hf_config.thinker_config
        self._image_token_id = thinker_cfg.image_token_id
        self._video_token_id = thinker_cfg.video_token_id
        self._audio_token_id = thinker_cfg.audio_token_id

    def prepare_prefill(self, forward_batch, schedule_batch, requests):
        """Inject multimodal embeddings. Returns batch_result if custom forward needed."""
        if not schedule_batch.forward_mode.is_extend():
            return None

        omni_result = self._inject_multimodal_embeds(forward_batch, schedule_batch)
        if omni_result is not None and omni_result[0] is not None:
            input_embeds, ds_embeds, vis_masks = omni_result
            return self._forward_with_omni_embeds(
                forward_batch, input_embeds, ds_embeds, vis_masks
            )
        return None

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ):
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        # Hidden capture for thinker streaming comes from our local forward hooks,
        # not from SGLang's logits-output hidden-state path. Requesting LAST here
        # causes CUDA-graph mode mismatches and can silently disable replay.
        return CaptureHiddenMode.NULL

    def requested_capture_hidden_mode_decode(self, schedule_batch: Any, requests: list):
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        # Hidden capture for thinker streaming comes from our local forward hooks,
        # not from SGLang's logits-output hidden-state path. Requesting LAST here
        # causes CUDA-graph mode mismatches and can silently disable replay.
        return CaptureHiddenMode.NULL

    # ------------------------------------------------------------------
    # Multimodal embedding injection (~160 lines, from SGLangModelRunner)
    # ------------------------------------------------------------------

    def _inject_multimodal_embeds(
        self, forward_batch: Any, schedule_batch: Any
    ) -> tuple[torch.Tensor | None, list | None, torch.Tensor | None] | None:
        if not any(req.omni_model_inputs is not None for req in schedule_batch.reqs):
            return None

        device = forward_batch.input_ids.device
        image_token_id = self._image_token_id
        video_token_id = self._video_token_id
        audio_token_id = self._audio_token_id

        embed_input_ids = forward_batch.input_ids.clamp(
            0, self._embed_tokens.num_embeddings - 1
        )
        input_embeds = self._embed_tokens(embed_input_ids)

        extend_lens = forward_batch.extend_seq_lens_cpu
        offsets = []
        pos = 0
        for length in extend_lens:
            offsets.append(pos)
            pos += length

        deepstack_visual_embeds_list = []
        visual_pos_masks_list = []
        has_deepstack = False

        for i, req in enumerate(schedule_batch.reqs):
            omni_inputs = req.omni_model_inputs
            if omni_inputs is None:
                continue

            start = offsets[i]
            end = start + extend_lens[i]
            req_input_ids = forward_batch.input_ids[start:end]
            consumed = req._omni_consumed or {}
            chunk_offsets: dict[str, tuple[int, int]] = {}
            pad_values = omni_inputs.get("pad_values", {})

            for modality, token_id in [
                ("image", image_token_id),
                ("video", video_token_id),
                ("audio", audio_token_id),
            ]:
                embeds = omni_inputs.get(f"{modality}_embeds")
                if embeds is None:
                    continue
                match_id = pad_values.get(modality, token_id)
                mask = req_input_ids == match_id
                if not mask.any():
                    continue
                n_tokens = int(mask.sum().item())
                offset = consumed.get(modality, 0)
                chunk_offsets[modality] = (offset, n_tokens)
                chunk_embeds = embeds[offset : offset + n_tokens].to(
                    device=device, dtype=input_embeds.dtype
                )
                input_embeds[torch.where(mask)[0] + start] = chunk_embeds
                consumed[modality] = offset + n_tokens

            req._omni_consumed = consumed

            ds_embeds = omni_inputs.get("deepstack_visual_embeds")
            image_ds = omni_inputs.get("image_deepstack_visual_embeds")
            video_ds = omni_inputs.get("video_deepstack_visual_embeds")

            if ds_embeds is not None or image_ds is not None or video_ds is not None:
                has_deepstack = True
                img_match_id = pad_values.get("image", image_token_id)
                vid_match_id = pad_values.get("video", video_token_id)
                img_mask = req_input_ids == img_match_id
                vid_mask = req_input_ids == vid_match_id
                visual_mask = img_mask | vid_mask

                if ds_embeds is None:
                    if image_ds and video_ds:
                        image_offset, image_count = chunk_offsets.get("image", (0, 0))
                        video_offset, video_count = chunk_offsets.get("video", (0, 0))
                        merged = []
                        for img_e, vid_e in zip(image_ds, video_ds):
                            img_e = img_e[image_offset : image_offset + image_count]
                            vid_e = vid_e[video_offset : video_offset + video_count]
                            num_visual = int(visual_mask.sum().item())
                            joint = img_e.new_zeros(num_visual, img_e.shape[-1])
                            img_in_visual = img_mask[visual_mask]
                            vid_in_visual = vid_mask[visual_mask]
                            if img_in_visual.any():
                                joint[img_in_visual] = img_e.to(device=device)
                            if vid_in_visual.any():
                                joint[vid_in_visual] = vid_e.to(device=device)
                            merged.append(joint)
                        ds_embeds = merged
                    elif image_ds:
                        image_offset, image_count = chunk_offsets.get("image", (0, 0))
                        ds_embeds = [
                            layer[image_offset : image_offset + image_count]
                            for layer in image_ds
                        ]
                    elif video_ds:
                        video_offset, video_count = chunk_offsets.get("video", (0, 0))
                        ds_embeds = [
                            layer[video_offset : video_offset + video_count]
                            for layer in video_ds
                        ]
                elif visual_mask.any():
                    visual_count = int(visual_mask.sum().item())
                    if vid_mask.any() and not img_mask.any():
                        visual_offset = chunk_offsets.get("video", (0, 0))[0]
                    elif img_mask.any() and not vid_mask.any():
                        visual_offset = chunk_offsets.get("image", (0, 0))[0]
                    else:
                        visual_offset = consumed.get("_visual", 0)
                    ds_embeds = [
                        layer[visual_offset : visual_offset + visual_count]
                        for layer in ds_embeds
                    ]
                    consumed["_visual"] = visual_offset + visual_count
                else:
                    ds_embeds = None

                if ds_embeds is not None:
                    global_mask = torch.zeros(
                        len(forward_batch.input_ids),
                        dtype=torch.bool,
                        device=device,
                    )
                    global_mask[start:end] = visual_mask
                    deepstack_visual_embeds_list.append(ds_embeds)
                    visual_pos_masks_list.append(global_mask)

            if req.is_chunked == 0:
                req.omni_model_inputs = None
                req._omni_consumed = None

        ds_embeds_out = None
        visual_masks_out = None
        if has_deepstack and deepstack_visual_embeds_list:
            if len(deepstack_visual_embeds_list) == 1:
                ds_embeds_out = deepstack_visual_embeds_list[0]
                visual_masks_out = visual_pos_masks_list[0]
            else:
                combined_mask = torch.zeros(
                    len(forward_batch.input_ids), dtype=torch.bool, device=device
                )
                for m in visual_pos_masks_list:
                    combined_mask |= m
                num_layers = len(deepstack_visual_embeds_list[0])
                merged_ds = []
                for layer_idx in range(num_layers):
                    parts = [
                        req_ds[layer_idx].to(device=device, dtype=input_embeds.dtype)
                        for req_ds in deepstack_visual_embeds_list
                    ]
                    merged_ds.append(torch.cat(parts, dim=0))
                ds_embeds_out = merged_ds
                visual_masks_out = combined_mask

        return input_embeds, ds_embeds_out, visual_masks_out

    # ------------------------------------------------------------------
    # Custom forward with multimodal embeddings + deepstack
    # ------------------------------------------------------------------

    def _forward_with_omni_embeds(
        self,
        forward_batch,
        input_embeds,
        deepstack_visual_embeds=None,
        visual_pos_masks=None,
    ):
        model_runner = self.tp_worker.model_runner
        outer = self._outer_model

        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        ds_input = None
        if deepstack_visual_embeds is not None and visual_pos_masks is not None:
            device = input_embeds.device
            dtype = input_embeds.dtype
            layer_tensors = [
                t.to(device=device, dtype=dtype) for t in deepstack_visual_embeds
            ]
            ds_input = torch.cat(layer_tensors, dim=-1)
            full_ds = torch.zeros(
                input_embeds.shape[0], ds_input.shape[-1], device=device, dtype=dtype
            )
            full_ds[visual_pos_masks] = ds_input
            ds_input = full_ds

        hidden_states = outer.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            input_deepstack_embeds=ds_input,
        )

        logits_output = outer.logits_processor(
            forward_batch.input_ids,
            hidden_states,
            outer.lm_head,
            forward_batch,
        )

        return GenerationBatchResult(
            logits_output=logits_output, can_run_cuda_graph=False
        )
