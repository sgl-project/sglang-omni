# SPDX-License-Identifier: Apache-2.0
"""Bucket-aligned CUDA graphs for MOSS-TTS Delay sampling and its FSM."""

from __future__ import annotations

import gc
import logging
from bisect import bisect_left
from typing import Any, NamedTuple

import torch

from sglang_omni.models.moss_tts.sampler import DelayGraphBatch, DelaySamplingOutput

logger = logging.getLogger(__name__)

_MIN_CAPTURE_FREE_BYTES = 512 * 1024**2


class _StaticSamplingInputs(NamedTuple):
    control_logits: torch.Tensor
    audio_logits: torch.Tensor
    delay_state: torch.Tensor
    seeds: torch.Tensor
    generation_steps: torch.Tensor
    audio_sample_output: torch.Tensor


class _CapturedSamplingGraph(NamedTuple):
    graph: torch.cuda.CUDAGraph
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
        model: Any,
        capture_bs: tuple[int, ...],
        disable_padding: bool = False,
    ) -> None:
        if not capture_bs:
            raise ValueError("MOSS-TTS Delay sampling CUDA graph bs must be non-empty")
        if any(batch_size < 1 for batch_size in capture_bs):
            raise ValueError("MOSS-TTS Delay sampling CUDA graph bs must be >= 1")
        if tuple(sorted(set(capture_bs))) != tuple(capture_bs):
            raise ValueError(
                "MOSS-TTS Delay sampling CUDA graph bs must be strictly increasing"
            )
        self.model = model
        self.capture_bs = tuple(int(batch_size) for batch_size in capture_bs)
        self.disable_padding = bool(disable_padding)
        self.graphs: dict[int, _CapturedSamplingGraph] = {}
        self._inputs: _StaticSamplingInputs | None = None
        self._graph_pool = None
        self._warmup_stream: torch.cuda.Stream | None = None
        self._capture_stream: torch.cuda.Stream | None = None

    @classmethod
    def capture(
        cls,
        *,
        model: Any,
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
            with torch.cuda.device(model.device):
                torch.cuda.synchronize()
                runner._capture_all()
        except Exception as exc:
            capture_failed = True
            failure = "OOM" if runner._is_cuda_oom(exc) else "failure"
            logger.exception(
                "MOSS-TTS Delay sampling CUDA graph initialization %s; "
                "discarding partial graphs and using eager sampling",
                failure,
            )
        # note (Zhang Yiyang): Clear outside the active exception scope so its
        # traceback cannot retain failed-capture tensors while empty_cache runs.
        if capture_failed:
            try:
                runner.clear()
            except Exception:
                logger.exception(
                    "Failed to release MOSS-TTS Delay sampling CUDA graph "
                    "resources after capture failure"
                )
        return runner

    def _capture_all(self) -> None:
        buckets = self.capture_buckets
        if not buckets:
            return
        self._inputs = self._make_static_inputs(buckets[-1])

        for bucket in reversed(buckets):
            free_bytes, _ = torch.cuda.mem_get_info(self.model.device)
            if free_bytes < _MIN_CAPTURE_FREE_BYTES:
                logger.warning(
                    "MOSS-TTS Delay sampling CUDA graph capture stopped: free "
                    "VRAM %.2f GiB is below %.2f GiB headroom; discarding "
                    "partial graphs and using eager sampling",
                    free_bytes / 1024**3,
                    _MIN_CAPTURE_FREE_BYTES / 1024**3,
                )
                self.clear()
                return

            try:
                self.graphs[bucket] = self._capture_bucket(bucket)
                logger.info(
                    "Captured MOSS-TTS Delay sampling CUDA graph backbone_bucket=%s",
                    bucket,
                )
            except Exception as exc:
                self.graphs.pop(bucket, None)
                if self._is_cuda_oom(exc):
                    logger.exception(
                        "MOSS-TTS Delay sampling CUDA graph OOM for "
                        "backbone_bucket=%s; discarding partial graphs",
                        bucket,
                    )
                    self.clear()
                    return
                logger.exception(
                    "Failed to capture MOSS-TTS Delay sampling CUDA graph "
                    "backbone_bucket=%s; that bucket will use eager",
                    bucket,
                )

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        return (
            isinstance(exc, torch.OutOfMemoryError)
            or "out of memory" in str(exc).lower()
        )

    def clear(self) -> None:
        self.graphs.clear()
        self._inputs = None
        self._graph_pool = None
        self._warmup_stream = None
        self._capture_stream = None
        gc.collect()
        if self.model.device.type in ("cuda", "musa"):
            device_module = torch.get_device_module(self.model.device)
            if device_module.is_available():
                with device_module.device(self.model.device):
                    device_module.empty_cache()

    def _make_static_inputs(self, max_bs: int) -> _StaticSamplingInputs:
        device = self.model.device
        n_vq = int(self.model.config.n_vq)
        audio_vocab = int(self.model.config.vocab_size_list[1])
        return _StaticSamplingInputs(
            control_logits=torch.zeros(
                max_bs,
                2,
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
        )

    def _require_inputs(self) -> _StaticSamplingInputs:
        if self._inputs is None:
            raise RuntimeError("MOSS-TTS Delay sampling CUDA graph is not captured")
        return self._inputs

    def _run_static(
        self,
        bucket: int,
    ) -> DelaySamplingOutput:
        inputs = self._require_inputs()
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

    def _capture_bucket(
        self,
        bucket: int,
    ) -> _CapturedSamplingGraph:
        device = self.model.device
        if self._warmup_stream is None:
            self._warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream = self._warmup_stream
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                self._run_static(bucket)
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)

        if self._graph_pool is None:
            self._graph_pool = torch.cuda.graph_pool_handle()
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(device=device)
        capture_stream = self._capture_stream
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            with torch.cuda.graph(
                graph,
                pool=self._graph_pool,
                stream=capture_stream,
                capture_error_mode="thread_local",
            ):
                output = self._run_static(bucket)
        torch.cuda.current_stream(device).wait_stream(capture_stream)
        return _CapturedSamplingGraph(
            graph=graph,
            rows=output.rows,
            next_delay_state=output.next_delay_state,
        )

    def canonical_bucket(self, batch_size: int) -> int | None:
        index = bisect_left(self.capture_bs, int(batch_size))
        if index == len(self.capture_bs):
            return None
        return self.capture_bs[index]

    @property
    def capture_buckets(self) -> tuple[int, ...]:
        return self.capture_bs

    def _graph_key(self, batch_size: int) -> int | None:
        bucket = self.canonical_bucket(batch_size)
        if bucket is None or (self.disable_padding and bucket != int(batch_size)):
            return None
        return bucket

    def can_replay(self, batch_size: int) -> bool:
        key = self._graph_key(batch_size)
        return key is not None and key in self.graphs

    @torch.no_grad()
    def replay(
        self,
        control_logits: torch.Tensor,
        audio_logits: torch.Tensor,
        batch: DelayGraphBatch,
    ) -> DelaySamplingOutput:
        batch_size = int(control_logits.shape[0])
        bucket = self.canonical_bucket(batch_size)
        key = self._graph_key(batch_size)
        if key is None or key not in self.graphs:
            raise RuntimeError(
                "MOSS-TTS Delay sampling CUDA graph is unavailable for "
                f"raw_bs={batch_size} backbone_bucket={bucket}"
            )

        inputs = self._require_inputs()
        expected_audio_shape = (
            batch_size,
            int(inputs.audio_logits.shape[1]),
            int(inputs.audio_logits.shape[2]),
        )
        if tuple(control_logits.shape) != (batch_size, 2):
            raise RuntimeError(
                "MOSS-TTS Delay sampling graph control-logits shape mismatch: "
                f"got {tuple(control_logits.shape)}, expected {(batch_size, 2)}"
            )
        if tuple(audio_logits.shape) != expected_audio_shape:
            raise RuntimeError(
                "MOSS-TTS Delay sampling graph audio-logits shape mismatch: "
                f"got {tuple(audio_logits.shape)}, expected {expected_audio_shape}"
            )
        if tuple(batch.delay_state.shape) != (batch_size, 3):
            raise RuntimeError("MOSS-TTS Delay sampling graph state shape mismatch")

        inputs.control_logits[:batch_size].copy_(control_logits)
        inputs.audio_logits[:batch_size].copy_(audio_logits)
        inputs.delay_state[:batch_size].copy_(batch.delay_state)
        inputs.seeds[:batch_size].copy_(batch.seeds)
        inputs.generation_steps[:batch_size].copy_(batch.generation_steps)
        assert bucket is not None
        if batch_size < bucket:
            inputs.control_logits[batch_size:bucket].zero_()
            inputs.audio_logits[batch_size:bucket].zero_()
            inputs.delay_state[batch_size:bucket].zero_()
            inputs.seeds[batch_size:bucket].zero_()
            inputs.generation_steps[batch_size:bucket].zero_()

        captured = self.graphs[key]
        captured.graph.replay()
        return DelaySamplingOutput(
            rows=captured.rows[:batch_size],
            next_delay_state=captured.next_delay_state[:batch_size],
        )
