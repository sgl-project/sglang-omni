from __future__ import annotations

import logging
import time

import torch

from sglang_omni.models.nemotron_voicechat.codec import RVQVAEDecoder
from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph

logger = logging.getLogger(__name__)

GRAPH_WARMUP_STEPS = 3


class CodecDecodeGraphRunner:
    """Replays the codec decode for every window of 1 to max_window_frames frames.

    A returned waveform belongs to the graph; the next call overwrites it.
    """

    def __init__(
        self,
        decoder: RVQVAEDecoder,
        backend: DeviceGraphBackend,
        device: torch.device,
        max_window_frames: int,
    ) -> None:
        self.samples_per_frame: int = decoder.samples_per_frame
        self.num_quantizers: int = decoder.num_quantizers
        self.staging_codes_TQ: torch.Tensor = torch.zeros(
            max_window_frames, decoder.num_quantizers, dtype=torch.long, device=device
        )
        self.graph_by_window_frames: dict[int, ReplayableGraph] = {}
        self.waveform_by_window_frames: dict[int, torch.Tensor] = {}
        device_module = torch.get_device_module(device)
        # note (Xinhao Tan): one pool for every window size. Replays never
        # overlap, and the caller copies each waveform off before the next call.
        pool = device_module.graph_pool_handle()
        started = time.perf_counter()
        reserved_before = device_module.memory_reserved(device)
        with torch.inference_mode():
            for window_frames in range(1, max_window_frames + 1):
                codes_TQ = self.staging_codes_TQ[:window_frames]
                for _ in range(GRAPH_WARMUP_STEPS):
                    decoder(codes_TQ)
                device_module.synchronize(device)
                with backend.capture(pool=pool, thread_local_errors=True) as graph:
                    waveform = decoder(codes_TQ)
                self.graph_by_window_frames[window_frames] = graph
                self.waveform_by_window_frames[window_frames] = waveform
        device_module.synchronize(device)
        reserved_mib = (device_module.memory_reserved(device) - reserved_before) / 2**20
        logger.info(
            f"Nemotron codec decode graphs captured for 1-{max_window_frames} frames "
            f"in {time.perf_counter() - started:.1f}s, reserved {reserved_mib:.0f} MiB"
        )

    def __call__(self, codes_TQ: torch.Tensor) -> torch.Tensor:
        window_frames = codes_TQ.shape[0]
        self.staging_codes_TQ[:window_frames].copy_(codes_TQ)
        self.graph_by_window_frames[window_frames].replay()
        return self.waveform_by_window_frames[window_frames]
