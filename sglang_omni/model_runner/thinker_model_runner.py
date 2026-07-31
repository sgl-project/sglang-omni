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

from sglang_omni.model_runner._hidden_capture import unpack_packed_hidden_capture
from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_execution import attn_forward_context

logger = logging.getLogger(__name__)


class ThinkerModelRunner(ModelRunner):
    """Thinker: injects multimodal embeddings in the prefill phase."""

    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        *,
        should_capture_hidden: Callable[[Any], bool] | None = None,
        capture_hidden_layers: list[int] | None = None,
        capture_hidden_width: int | None = None,
    ):
        super().__init__(tp_worker, output_processor)
        self._should_capture_hidden = should_capture_hidden
        # Capture configuration is frozen at construction. A text-only
        # deployment installs no capture layers, so every batch there must stay
        # on the NULL capture path no matter what per-request metadata says.
        self._capture_hidden_layers = (
            list(capture_hidden_layers) if capture_hidden_layers else None
        )
        self._capture_hidden_width = capture_hidden_width

        model = self.model
        self._outer_model = model.thinker
        self._text_model = self._outer_model.model
        self._embed_tokens = self._text_model.embed_tokens
        self._th_host_bufs = None
        self._th_slot = 0
        self._th_hidden_bufs: list[list[torch.Tensor]] | None = None
        self._th_hidden_slot = 0

        thinker_cfg = tp_worker.model_runner.model_config.hf_config.thinker_config
        self._image_token_id = thinker_cfg.image_token_id
        self._video_token_id = thinker_cfg.video_token_id
        self._audio_token_id = thinker_cfg.audio_token_id

    @contextlib.contextmanager
    def _text_only_capture_guard(self, requests: list[Any]):
        # note (jiaxin deng): drop hidden-capture for an all-text batch, shared by
        # sync execute() and async execute_launch so both take the same path.
        # These thinker layers feed Qwen3-Omni's talker. This toggle affects eager
        # forwards only; graph replay still runs the layer capture recorded at graph build.
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
        if self._capture_hidden_layers is None:
            return False
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
            # Publish ordinary multimodal embeddings through SGLang's
            # ForwardBatch contract so its runner remains the sole owner of
            # attention metadata and eager/CUDA-graph dispatch. Visual
            # deepstack still needs the model-specific forward below because
            # ForwardBatch has no field for those residual embeddings.
            if ds_embeds is None:
                forward_batch.input_embeds = input_embeds
                return None
            return self._forward_with_omni_embeds(
                forward_batch, input_embeds, ds_embeds, vis_masks
            )
        return None

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ):
        del schedule_batch
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return (
            CaptureHiddenMode.LAST
            if self._batch_should_capture_hidden(requests)
            else CaptureHiddenMode.NULL
        )

    def requested_capture_hidden_mode_decode(self, schedule_batch: Any, requests: list):
        del schedule_batch
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        # Speech CUDA graphs are captured with CaptureHiddenMode.FULL.
        # Decode must use the same mode; LAST would prevent graph replay.
        return (
            CaptureHiddenMode.FULL
            if self._batch_should_capture_hidden(requests)
            else CaptureHiddenMode.NULL
        )

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

            if req.inflight_middle_chunks == 0:
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

        with attn_forward_context(model_runner.attn_backend):
            hidden_states = outer.model(
                input_ids=None,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                input_deepstack_embeds=ds_input,
            )

            logits_output = outer.process_hidden_states(
                input_ids=forward_batch.input_ids,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        return GenerationBatchResult(
            logits_output=logits_output, can_run_cuda_graph=False
        )

    def lookahead_eligible(self, batch: Any) -> bool:
        """Route to sync where the one-step lag would diverge from sync.

        Speech hidden states are snapshotted into ping-pong buffers at launch, so
        audio-output requests are safe here. Sampling that reads the lagged output
        history (repetition / presence / frequency penalty, min_new_tokens), a
        fixed seed, or return_logprob (the lookahead sampler skips the base
        logprob path) still diverges; logit_bias / custom_params are routed
        conservatively.
        """
        for req in batch.reqs:
            if req._omni_data.return_logprob:
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

    @staticmethod
    def _hidden_buf_fits(buf: torch.Tensor, source: torch.Tensor) -> bool:
        return (
            buf.dtype == source.dtype
            and buf.device == source.device
            and buf.shape[0] >= source.shape[0]
            and buf.shape[1:] == source.shape[1:]
        )

    def _async_hidden_bufs(
        self, sources: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        """Copy one step's captured hidden tensors into a private launch slot.

        CUDA-graph replay reuses its output storage every step. Two device-side
        slots let launch(N+1) publish new hidden states while resolve(N) still
        owns the previous step. Smaller batches reuse a leading slice; growth
        or a layout change replaces both slots (a resolve still holding the old
        buffers keeps them alive through its own reference).
        """
        need_alloc = (
            self._th_hidden_bufs is None
            or len(self._th_hidden_bufs[0]) != len(sources)
            or any(
                not self._hidden_buf_fits(buf, source)
                for buf, source in zip(self._th_hidden_bufs[0], sources)
            )
        )
        if need_alloc:
            self._th_hidden_bufs = [
                [torch.empty_like(source) for source in sources] for _ in range(2)
            ]
            self._th_hidden_slot = 0

        assert self._th_hidden_bufs is not None
        slot_bufs = self._th_hidden_bufs[self._th_hidden_slot]
        self._th_hidden_slot ^= 1
        snapshots: list[torch.Tensor] = []
        for buf, source in zip(slot_bufs, sources):
            view = buf[: source.shape[0]]
            view.copy_(source, non_blocking=True)
            snapshots.append(view)
        return tuple(snapshots)

    def _stage_async_hidden_capture(self, result: Any) -> None:
        """Snapshot graph-owned hidden output into this lookahead launch."""
        logits_output = result.logits_output
        packed_hidden = logits_output.hidden_states
        if packed_hidden is None:
            raise RuntimeError(
                "Speech lookahead requested hidden capture, but the model "
                "produced no hidden states"
            )
        captured_aux = unpack_packed_hidden_capture(
            packed_hidden,
            capture_layer_count=len(self._capture_hidden_layers),
            hidden_size=self._capture_hidden_width,
        )
        result._captured_aux_hidden_states = self._async_hidden_bufs(list(captured_aux))

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
        if self._batch_should_capture_hidden(requests):
            self._stage_async_hidden_capture(result)
        return host_buf

    def post_decode_resolve(
        self, launch_buf, result, forward_batch, schedule_batch, requests
    ):
        del forward_batch, schedule_batch
        if len(requests) == 0 or launch_buf is None:
            return
        n = len(requests)
        result.next_token_ids = launch_buf[:n].to(torch.long).clone()
