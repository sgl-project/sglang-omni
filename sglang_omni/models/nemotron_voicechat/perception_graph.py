"""CUDA graph replay once the causal Perception window reaches its fixed size."""

import torch

from sglang_omni.models.nemotron_voicechat.conformer import (
    SAMPLES_PER_FRAME,
    AudioPerception,
    StreamingPerception,
)

GRAPH_WARMUP_STEPS = 3


class GraphPerception(StreamingPerception):
    def __init__(self, perception: AudioPerception) -> None:
        super().__init__(perception)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.graph_input: torch.Tensor | None = None
        self.graph_output: torch.Tensor | None = None

    @torch.inference_mode()
    def push(self, samples: torch.Tensor) -> torch.Tensor:
        assert not self.flushed
        assert samples.shape == (SAMPLES_PER_FRAME,)
        if self.device.type != "cuda" or self.cached_frame_count < self.max_keys:
            return super().push(samples)
        else:
            pass
        if self.graph is None:
            buffers = self.state_buffers()
            saved_values = [buffer.clone() for buffer in buffers]
            self.graph_input = samples.to(device=self.device, dtype=self.dtype).clone()
            capture_stream = torch.cuda.Stream(device=self.device)
            current_stream = torch.cuda.current_stream(self.device)
            capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(capture_stream):
                for iteration in range(GRAPH_WARMUP_STEPS):
                    super().push(self.graph_input)
            current_stream.wait_stream(capture_stream)
            # note (Codex): Warmup must not consume real causal history.
            for buffer, saved_value in zip(buffers, saved_values, strict=True):
                buffer.copy_(saved_value)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=capture_stream):
                self.graph_output = super().push(self.graph_input)
        else:
            pass
        assert self.graph_input is not None and self.graph_output is not None
        self.graph_input.copy_(samples)
        self.graph.replay()
        # note (Codex): Offline callers retain every row after the next replay.
        return self.graph_output.clone()
