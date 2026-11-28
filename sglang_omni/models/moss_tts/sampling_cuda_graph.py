# SPDX-License-Identifier: Apache-2.0
"""Bucket-aligned CUDA graphs for MOSS-TTS Delay sampling and its FSM."""

from __future__ import annotations

import gc
import logging
from bisect import bisect_left
from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang_omni.models.moss_tts.sampler import DelayGraphBatch, DelaySamplingOutput
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphPool, ReplayableGraph

if TYPE_CHECKING:
    from sglang_omni.models.moss_tts.sglang_model import MossTTSDelaySGLangModel
else:
    pass

logger = logging.getLogger(__name__)

_MIN_CAPTURE_FREE_BYTES = 512 * 1024**2


class StaticSamplingInputs(NamedTuple):
    control_logits: torch.Tensor
    audio_logits: torch.Tensor
    delay_state: torch.Tensor
    seeds: torch.Tensor
    generation_steps: torch.Tensor
    audio_sample_output: torch.Tensor
    # Row indices the full-vocabulary text draw reads; only the mixed runner
    # compacts it, but the buffer is shared so capture can stage any width.
    text_rows: torch.Tensor


class CapturedSamplingGraph(NamedTuple):
    graph: ReplayableGraph
    rows: torch.Tensor
    next_delay_state: torch.Tensor


class MossTTSDelaySamplingCudaGraphRunner:
    """Sampling/FSM graphs aligned to the backbone CUDA-graph buckets.

    LM heads and feedback embedding deliberately remain eager. Every graph uses
    all 32 codebook rows and applies the Delay mask only to its fixed-shape
    output, so replay topology does not depend on the number of active channels.
    """

    def __init__(
        self,
        *,
        model: MossTTSDelaySGLangModel,
        capture_bs: tuple[int, ...],
        disable_padding: bool = False,
    ) -> None:
        if not capture_bs:
            raise ValueError("MOSS-TTS Delay sampling CUDA graph bs must be non-empty")
        else:
            pass
        if any(batch_size < 1 for batch_size in capture_bs):
            raise ValueError("MOSS-TTS Delay sampling CUDA graph bs must be >= 1")
        else:
            pass
        if tuple(sorted(set(capture_bs))) != tuple(capture_bs):
            raise ValueError(
                "MOSS-TTS Delay sampling CUDA graph bs must be strictly increasing"
            )
        else:
            pass
        self.model = model
        self.device_module = torch.npu if model.device.type == "npu" else torch.cuda
        self.graph_backend = current_platform.get_device_graph_backend(model.device)
        self.capture_bs = tuple(int(batch_size) for batch_size in capture_bs)
        self.disable_padding = bool(disable_padding)
        self.graphs: dict[int, CapturedSamplingGraph] = {}
        self.compacted_graphs: dict[int, dict[int, CapturedSamplingGraph]] = {}
        self.inputs: StaticSamplingInputs | None = None
        self.graph_pool: DeviceGraphPool | None = None
        self.warmup_stream: torch.Stream | None = None
        self.capture_stream: torch.Stream | None = None

    @classmethod
    def capture(
        cls,
        *,
        model: MossTTSDelaySGLangModel,
        capture_bs: tuple[int, ...],
        disable_padding: bool = False,
    ) -> "MossTTSDelaySamplingCudaGraphRunner":
        runner = cls(
            model=model,
            capture_bs=capture_bs,
            disable_padding=disable_padding,
        )
        capture_failed = False
        try:
            with runner.device_module.device(model.device):
                runner.device_module.synchronize()
                runner.capture_all()
        except Exception as exc:
            capture_failed = True
            failure = "OOM" if runner.is_cuda_oom(exc) else "failure"
            logger.exception(
                f"MOSS-TTS Delay sampling graph initialization {failure} "
                f"on {model.device}; discarding partial graphs"
            )
        # note (Zhang Yiyang): Clear outside the active exception scope so its
        # traceback cannot retain failed-capture tensors while empty_cache runs.
        if capture_failed:
            try:
                runner.clear()
            except Exception:
                logger.exception(
                    f"Failed to release MOSS-TTS Delay sampling graph resources "
                    f"on {model.device} after capture failure"
                )
        else:
            pass
        return runner

    def capture_all(self) -> None:
        buckets = self.capture_buckets
        if not buckets:
            return
        else:
            pass
        self.inputs = self.make_static_inputs(buckets[-1])

        for bucket in reversed(buckets):
            free_bytes, _ = self.device_module.mem_get_info(self.model.device)
            if free_bytes < _MIN_CAPTURE_FREE_BYTES:
                logger.warning(
                    f"MOSS-TTS Delay sampling graph capture stopped on {self.model.device}: "
                    f"free device memory {free_bytes / 1024**3:.2f} GiB is below "
                    f"{_MIN_CAPTURE_FREE_BYTES / 1024**3:.2f} GiB headroom; "
                    "discarding partial graphs"
                )
                self.clear()
                return
            else:
                pass

            try:
                self.graphs[bucket] = self.capture_bucket(bucket)
                logger.info(
                    f"Captured MOSS-TTS Delay sampling graph device={self.model.device} "
                    f"backbone_bucket={bucket}"
                )
            except Exception as exc:
                self.graphs.pop(bucket, None)
                if self.is_cuda_oom(exc):
                    logger.exception(
                        f"MOSS-TTS Delay sampling graph OOM on {self.model.device} "
                        f"for backbone_bucket={bucket}; discarding partial graphs"
                    )
                    self.clear()
                    return
                else:
                    pass
                logger.exception(
                    f"Failed to capture MOSS-TTS Delay sampling graph "
                    f"on {self.model.device} for backbone_bucket={bucket}"
                )

    @staticmethod
    def is_cuda_oom(exc: BaseException) -> bool:
        return (
            isinstance(exc, torch.OutOfMemoryError)
            or "out of memory" in str(exc).lower()
        )

    def clear(self) -> None:
        self.graphs.clear()
        self.compacted_graphs.clear()
        self.inputs = None
        self.graph_pool = None
        self.warmup_stream = None
        self.capture_stream = None
        gc.collect()
        if self.model.device.type in ("cuda", "musa", "npu"):
            device_module = torch.get_device_module(self.model.device)
            if device_module.is_available():
                with device_module.device(self.model.device):
                    device_module.empty_cache()
            else:
                pass
        else:
            pass

    def control_logit_columns(self) -> int:
        """Text logit width: the two audio control tokens."""
        return 2

    def make_static_inputs(self, max_bs: int) -> StaticSamplingInputs:
        device = self.model.device
        n_vq = int(self.model.config.n_vq)
        audio_vocab = int(self.model.config.vocab_size_list[1])
        return StaticSamplingInputs(
            control_logits=torch.zeros(
                max_bs,
                self.control_logit_columns(),
                device=device,
                dtype=torch.float32,
            ),
            audio_logits=torch.zeros(
                max_bs,
                n_vq,
                audio_vocab,
                device=device,
                dtype=torch.float32,
            ),
            delay_state=torch.zeros(
                max_bs,
                3,
                device=device,
                dtype=torch.long,
            ),
            seeds=torch.zeros(max_bs, device=device, dtype=torch.long),
            generation_steps=torch.zeros(max_bs, device=device, dtype=torch.long),
            audio_sample_output=torch.empty(
                max_bs * n_vq,
                device=device,
                dtype=torch.long,
            ),
            text_rows=torch.zeros(max_bs, device=device, dtype=torch.long),
        )

    def require_inputs(self) -> StaticSamplingInputs:
        if self.inputs is None:
            raise RuntimeError("MOSS-TTS Delay sampling CUDA graph is not captured")
        else:
            pass
        return self.inputs

    def run_static(
        self,
        bucket: int,
    ) -> DelaySamplingOutput:
        inputs = self.require_inputs()
        batch = DelayGraphBatch(
            delay_state=inputs.delay_state[:bucket],
            seeds=inputs.seeds[:bucket],
            generation_steps=inputs.generation_steps[:bucket],
        )
        return self.model.sample_delay_fixed_shape(
            inputs.control_logits[:bucket],
            inputs.audio_logits[:bucket],
            batch,
            audio_sample_output=inputs.audio_sample_output,
        )

    def capture_bucket(
        self,
        bucket: int,
    ) -> CapturedSamplingGraph:
        return self.capture_callable(lambda: self.run_static(bucket))

    def capture_callable(
        self,
        step,
    ) -> CapturedSamplingGraph:
        device = self.model.device
        graph_backend = self.graph_backend
        assert graph_backend is not None
        if self.warmup_stream is None:
            self.warmup_stream = self.device_module.Stream(device=device)
        else:
            pass
        warmup_stream = self.warmup_stream
        warmup_stream.wait_stream(self.device_module.current_stream(device))
        with self.device_module.stream(warmup_stream):
            for _ in range(2):
                step()
        self.device_module.current_stream(device).wait_stream(warmup_stream)
        self.device_module.synchronize(device)

        if self.graph_pool is None:
            self.graph_pool = graph_backend.graph_pool_handle()
        else:
            pass
        if self.capture_stream is None:
            self.capture_stream = self.device_module.Stream(device=device)
        else:
            pass
        capture_stream = self.capture_stream
        capture_stream.wait_stream(self.device_module.current_stream(device))
        with self.device_module.stream(capture_stream):
            with graph_backend.capture(
                pool=self.graph_pool,
                stream=capture_stream,
                thread_local_errors=True,
            ) as graph:
                output = step()
        self.device_module.current_stream(device).wait_stream(capture_stream)
        return CapturedSamplingGraph(
            graph=graph,
            rows=output.rows,
            next_delay_state=output.next_delay_state,
        )

    def free_memory_bytes(self) -> int:
        free_bytes, _ = self.device_module.mem_get_info(self.model.device)
        return int(free_bytes)

    def canonical_bucket(self, batch_size: int) -> int | None:
        index = bisect_left(self.capture_bs, int(batch_size))
        if index == len(self.capture_bs):
            return None
        else:
            pass
        return self.capture_bs[index]

    @property
    def capture_buckets(self) -> tuple[int, ...]:
        return self.capture_bs

    def graph_key(self, batch_size: int) -> int | None:
        bucket = self.canonical_bucket(batch_size)
        if bucket is None or (self.disable_padding and bucket != int(batch_size)):
            return None
        else:
            pass
        return bucket

    def can_replay(self, batch_size: int) -> bool:
        key = self.graph_key(batch_size)
        return key is not None and key in self.graphs

    def require_full_graph(self, batch_size: int) -> tuple[int, int]:
        bucket = self.canonical_bucket(batch_size)
        key = self.graph_key(batch_size)
        if key is None or key not in self.graphs:
            raise RuntimeError(
                "MOSS-TTS Delay sampling CUDA graph is unavailable for "
                f"raw_bs={batch_size} backbone_bucket={bucket}"
            )
        else:
            pass
        bucket = int(bucket)
        return int(key), bucket

    @staticmethod
    def validate_audio_rows(
        inputs: StaticSamplingInputs,
        audio_logits: torch.Tensor,
        batch: DelayGraphBatch,
        batch_size: int,
    ) -> None:
        expected_audio_shape = (
            batch_size,
            int(inputs.audio_logits.shape[1]),
            int(inputs.audio_logits.shape[2]),
        )
        if tuple(audio_logits.shape) != expected_audio_shape:
            raise RuntimeError(
                "MOSS-TTS Delay sampling graph audio-logits shape mismatch: "
                f"got {tuple(audio_logits.shape)}, expected {expected_audio_shape}"
            )
        else:
            pass
        if tuple(batch.delay_state.shape) != (batch_size, 3):
            raise RuntimeError("MOSS-TTS Delay sampling graph state shape mismatch")
        else:
            pass

    @staticmethod
    def stage_audio_rows(
        inputs: StaticSamplingInputs,
        audio_logits: torch.Tensor,
        batch: DelayGraphBatch,
        batch_size: int,
        bucket: int,
    ) -> None:
        inputs.audio_logits[:batch_size].copy_(audio_logits)
        inputs.delay_state[:batch_size].copy_(batch.delay_state)
        inputs.seeds[:batch_size].copy_(batch.seeds)
        inputs.generation_steps[:batch_size].copy_(batch.generation_steps)
        if batch_size < bucket:
            inputs.audio_logits[batch_size:bucket].zero_()
            inputs.delay_state[batch_size:bucket].zero_()
            inputs.seeds[batch_size:bucket].zero_()
            inputs.generation_steps[batch_size:bucket].zero_()
        else:
            pass

    @torch.no_grad()
    def replay(
        self,
        control_logits: torch.Tensor,
        audio_logits: torch.Tensor,
        batch: DelayGraphBatch,
    ) -> DelaySamplingOutput:
        batch_size = int(control_logits.shape[0])
        key, bucket = self.require_full_graph(batch_size)
        inputs = self.require_inputs()
        expected_control_shape = (batch_size, int(inputs.control_logits.shape[1]))
        if tuple(control_logits.shape) != expected_control_shape:
            raise RuntimeError(
                "MOSS-TTS Delay sampling graph control-logits shape mismatch: "
                f"got {tuple(control_logits.shape)}, expected {expected_control_shape}"
            )
        else:
            pass
        self.validate_audio_rows(inputs, audio_logits, batch, batch_size)
        inputs.control_logits[:batch_size].copy_(control_logits)
        if batch_size < bucket:
            inputs.control_logits[batch_size:bucket].zero_()
        else:
            pass
        self.stage_audio_rows(inputs, audio_logits, batch, batch_size, bucket)

        captured = self.graphs[key]
        captured.graph.replay()
        return DelaySamplingOutput(
            rows=captured.rows[:batch_size],
            next_delay_state=captured.next_delay_state[:batch_size],
        )


class MossTTSDelayMixedSamplingCudaGraphRunner(MossTTSDelaySamplingCudaGraphRunner):
    """Sampling graphs for batches that sample the full text vocabulary.

    Besides the full-width graph per backbone bucket, this runner captures
    narrower text-draw graphs so a step that only has a few live text rows does
    not pay for the whole bucket's ``[rows, text_vocab]`` Hash32/FP64 draw. Every
    width samples exactly the rows the caller names in ``text_rows``; rows
    outside that set are FSM-fixed by the Delay countdown and never read the
    drawn token.
    """

    TEXT_ROW_BUCKETS = (1, 2, 4, 8, 16)

    def control_logit_columns(self) -> int:
        return int(self.model.config.vocab_size_list[0])

    def compacted_text_buckets(self, bucket: int) -> tuple[int, ...]:
        return tuple(size for size in self.TEXT_ROW_BUCKETS if size < bucket)

    def run_static(
        self,
        bucket: int,
        text_bucket: int | None = None,
    ) -> DelaySamplingOutput:
        inputs = self.require_inputs()
        width = bucket if text_bucket is None else int(text_bucket)
        batch = DelayGraphBatch(
            delay_state=inputs.delay_state[:bucket],
            seeds=inputs.seeds[:bucket],
            generation_steps=inputs.generation_steps[:bucket],
        )
        text_rows = None if text_bucket is None else inputs.text_rows[:width]
        return self.model.sample_mixed_fixed_shape(
            inputs.control_logits[:width],
            inputs.audio_logits[:bucket],
            batch,
            text_rows=text_rows,
        )

    def capture_all(self) -> None:
        super().capture_all()
        if not self.graphs:
            return
        else:
            pass
        for bucket in self.capture_buckets:
            if bucket not in self.graphs:
                continue
            else:
                pass
            widths = self.compacted_text_buckets(bucket)
            if not widths:
                continue
            else:
                pass
            if self.free_memory_bytes() < _MIN_CAPTURE_FREE_BYTES:
                logger.warning(
                    "MOSS-TTS compacted text-draw graph capture skipped on "
                    f"{self.model.device}: free device memory is below the "
                    f"{_MIN_CAPTURE_FREE_BYTES / 1024**3:.2f} GiB headroom; "
                    "falling back to the full-width graph for this bucket"
                )
                break
            else:
                pass
            captured: dict[int, CapturedSamplingGraph] = {}
            for width in widths:
                try:
                    captured[width] = self.capture_callable(
                        lambda b=bucket, w=width: self.run_static(b, w)
                    )
                except Exception as exc:
                    if self.is_cuda_oom(exc):
                        logger.exception(
                            "MOSS-TTS compacted text-draw CUDA graph OOM for "
                            "backbone_bucket=%s text_bucket=%s; falling back to "
                            "the full-width graph for this bucket",
                            bucket,
                            width,
                        )
                        captured = {}
                        break
                    else:
                        pass
                    logger.exception(
                        "Failed to capture MOSS-TTS compacted text-draw graph "
                        "on %s for backbone_bucket=%s text_bucket=%s",
                        self.model.device,
                        bucket,
                        width,
                    )
            if captured:
                self.compacted_graphs[bucket] = captured
                logger.info(
                    "Captured MOSS-TTS compacted text-draw graphs "
                    f"device={self.model.device} backbone_bucket={bucket} "
                    f"text_buckets={sorted(captured)}"
                )
            else:
                pass

    def compacted_text_width(self, bucket: int, text_count: int) -> int | None:
        captured = self.compacted_graphs.get(bucket)
        if not captured:
            return None
        else:
            pass
        for width in sorted(captured):
            if width >= int(text_count):
                return width
            else:
                pass
        return None

    @torch.no_grad()
    def replay(
        self,
        control_logits: torch.Tensor,
        audio_logits: torch.Tensor,
        batch: DelayGraphBatch,
        *,
        text_rows: torch.Tensor | None = None,
    ) -> DelaySamplingOutput:
        batch_size = int(audio_logits.shape[0])
        key, bucket = self.require_full_graph(batch_size)
        inputs = self.require_inputs()
        expected_columns = int(inputs.control_logits.shape[1])
        if int(control_logits.shape[0]) != batch_size:
            raise RuntimeError(
                "MOSS-TTS mixed sampling graph expects full text logits; "
                f"got {int(control_logits.shape[0])} rows for batch {batch_size}"
            )
        else:
            pass
        if int(control_logits.shape[1]) != expected_columns:
            raise RuntimeError(
                "MOSS-TTS mixed sampling graph control-logits width mismatch: "
                f"got {int(control_logits.shape[1])}, expected {expected_columns}"
            )
        else:
            pass
        self.validate_audio_rows(inputs, audio_logits, batch, batch_size)

        if text_rows is not None:
            text_rows = text_rows.to(torch.long).reshape(-1)
            text_count = int(text_rows.numel())
            if text_count < 1 or text_count > batch_size:
                raise RuntimeError(
                    "MOSS-TTS mixed sampling graph text_rows must cover "
                    f"1..{batch_size} rows, got {text_count}"
                )
            else:
                pass
            width = self.compacted_text_width(bucket, text_count)
        else:
            width = None

        self.stage_audio_rows(inputs, audio_logits, batch, batch_size, bucket)
        if width is None:
            # Full-width graph: the sampler draws text for every bucket row, so
            # ``text_rows`` is ignored and only the control logits are staged.
            inputs.control_logits[:batch_size].copy_(control_logits)
            if batch_size < bucket:
                inputs.control_logits[batch_size:bucket].zero_()
            else:
                pass
            captured = self.graphs[key]
        else:
            # Compacted graph: gather the live text rows, then repeat the first
            # one into the padding slots (same state/seed/logits, so the same
            # token lands on a duplicate index that ``index_copy_`` overwrites).
            text_count = int(text_rows.numel())
            torch.index_select(
                control_logits,
                0,
                text_rows,
                out=inputs.control_logits[:text_count],
            )
            inputs.text_rows[:text_count].copy_(text_rows)
            if text_count < width:
                pad = width - text_count
                inputs.control_logits[text_count:width].copy_(
                    inputs.control_logits[:1].expand(pad, -1)
                )
                inputs.text_rows[text_count:width].fill_(int(text_rows[0]))
            else:
                pass
            captured = self.compacted_graphs[bucket][width]

        captured.graph.replay()
        return DelaySamplingOutput(
            rows=captured.rows[:batch_size],
            next_delay_state=captured.next_delay_state[:batch_size],
        )
