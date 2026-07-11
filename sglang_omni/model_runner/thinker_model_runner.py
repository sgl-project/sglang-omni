# SPDX-License-Identifier: Apache-2.0
"""Thinker model runner — injects multimodal embeddings before forward.

Handles image/video/audio token → embedding replacement and deepstack
visual embeddings for Qwen3-Omni's thinker stage.
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner

logger = logging.getLogger(__name__)


class ThinkerModelRunner(ModelRunner):
    """Thinker: injects multimodal embeddings in the prefill phase."""

    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        *,
        should_capture_hidden: Callable[[Any], bool] | None = None,
    ):
        super().__init__(tp_worker, output_processor)
        self._should_capture_hidden = should_capture_hidden

        model = self.model
        self._outer_model = model.thinker
        self._text_model = self._outer_model.model
        self._embed_tokens = self._text_model.embed_tokens
        self._th_host_bufs = None
        self._th_slot = 0

        thinker_cfg = tp_worker.model_runner.model_config.hf_config.thinker_config
        self._image_token_id = thinker_cfg.image_token_id
        self._video_token_id = thinker_cfg.video_token_id
        self._audio_token_id = thinker_cfg.audio_token_id

    @contextlib.contextmanager
    def _text_only_capture_guard(self, requests: list[Any]):
        # note (jiaxin deng): drop hidden-capture for an all-text batch, shared by
        # sync execute() and async execute_launch so both take the same path.
        capture_layers = self._text_model.layers_to_capture
        if not (capture_layers and not self._batch_should_capture_hidden(requests)):
            yield
            return
        saved_capture_layers = list(capture_layers)
        self._text_model.layers_to_capture = []
        try:
            yield
        finally:
            self._text_model.layers_to_capture = saved_capture_layers

    def execute(self, scheduler_output: Any):
        with self._text_only_capture_guard(scheduler_output.requests):
            return super().execute(scheduler_output)

    def execute_launch(self, scheduler_output: Any):
        with self._text_only_capture_guard(scheduler_output.requests):
            return super().execute_launch(scheduler_output)

    def _batch_should_capture_hidden(self, requests: list[Any]) -> bool:
        if self._should_capture_hidden is None:
            return True
        for request in requests:
            if self._should_capture_hidden(request):
                return True
        return False

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        """Run custom prefill when multimodal embeddings must be injected."""
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

    def lookahead_eligible(self, batch: Any) -> bool:
        """Route to sync where the one-step lag would diverge from sync. A request
        that emits audio captures hidden states for the talker; the per-forward
        _captured_aux_hidden_states side channel would be overwritten by a lookahead
        launch(N) before resolve(N-1) collects it, so those requests route to sync
        per batch. Sampling that reads the lagged output history (repetition /
        presence / frequency penalty, min_new_tokens), a fixed seed, or
        return_logprob (the lookahead sampler skips the base logprob path) also
        diverges; logit_bias / custom_params are routed conservatively.
        """
        from sglang_omni.models.qwen3_omni.request_builders import (
            should_generate_audio_output,
        )

        for req in batch.reqs:
            # note (jiaxin deng): fail closed if the request data is missing or None
            # so a hidden-capture batch can never slip onto the async path.
            try:
                data = req._omni_data
            except AttributeError:
                data = None
            if data is None or should_generate_audio_output(data.stage_payload):
                return False
            try:
                needs_logprob = data.return_logprob
            except AttributeError:
                needs_logprob = False
            if needs_logprob:
                return False
            sp = req.sampling_params
            if (
                sp.repetition_penalty != 1.0
                or sp.presence_penalty != 0.0
                or sp.frequency_penalty != 0.0
                or sp.min_new_tokens > 0
                or sp.sampling_seed is not None
                or sp.logit_bias is not None
                or sp.custom_params
            ):
                return False
        return True

    def _async_host_buf(self, like: torch.Tensor, n: int) -> torch.Tensor:
        # note (jiaxin deng): two pinned buffers ping-ponged so resolve(N) reads
        # one while launch(N+1) writes the other.
        if self._th_host_bufs is None or self._th_host_bufs[0].shape[0] < n:
            self._th_host_bufs = [
                torch.empty(n, dtype=like.dtype, device="cpu", pin_memory=True)
                for _ in range(2)
            ]
            self._th_slot = 0
        buf = self._th_host_bufs[self._th_slot]
        self._th_slot ^= 1
        return buf

    def _sample_lookahead(self, logits_output, forward_batch, requests):
        # note (jiaxin deng): penalties never reach here (lookahead_eligible routes
        # those batches to sync); only static suppress tokens are lag-safe.
        self._apply_codec_suppress_tokens(logits_output, requests)
        return self.tp_worker.model_runner.sample(logits_output, forward_batch)

    def post_decode_launch(self, result, forward_batch, requests):
        n = len(requests)
        if n == 0:
            return None
        # note (jiaxin deng): the decode forward leaves next_token_ids None (sync
        # samples in _finalize); set it here for the next-step input chain.
        if result.next_token_ids is None:
            result.next_token_ids = self._sample_lookahead(
                result.logits_output, forward_batch, requests
            )
        nt = result.next_token_ids
        host_buf = self._async_host_buf(nt, n)
        host_buf[:n].copy_(nt[:n], non_blocking=True)
        return host_buf

    def post_decode_resolve(
        self, launch_buf, result, forward_batch, schedule_batch, requests
    ):
        del forward_batch, schedule_batch
        if len(requests) == 0 or launch_buf is None:
            return
        n = len(requests)
        result.next_token_ids = launch_buf[:n].to(torch.long).clone()
