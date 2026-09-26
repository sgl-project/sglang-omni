# SPDX-License-Identifier: Apache-2.0
"""CUDA graph replay preserves inputs, causal state, and random sampling."""

from unittest.mock import Mock

import pytest
import torch

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.nemotron_voicechat.conformer import (
    AudioPerception,
    StreamingPerception,
)
from sglang_omni.models.nemotron_voicechat.duplex import CodecHooks, GraphPerception
from sglang_omni.models.nemotron_voicechat.duplex_ar import DuplexTalkerRunner
from sglang_omni.models.nemotron_voicechat.talker_model_runner import (
    NemotronVoiceChatTalkerModelRunner,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph requires GPU"
)


@torch.inference_mode()
def test_sampler_replay_uses_new_hidden_states_and_randomness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sample_codes(
        hidden: torch.Tensor,
        head: torch.nn.Module,
        *,
        num_iter: int,
        exponent: float,
        top_p: float,
        noise_scale: float,
        assignment_counts: tuple[int, ...],
    ) -> torch.Tensor:
        return hidden + torch.rand_like(hidden)

    def initialize_runner(
        runner: NemotronVoiceChatTalkerModelRunner,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
    ) -> None:
        runner.model = Mock(
            hidden_out=torch.zeros(1, 16, device="cuda"),
            talker=Mock(num_quantizers=8, generate_codes=sample_codes),
            mog_head=Mock(),
        )
        runner.exponent = 1.0
        runner.top_p = 0.9
        runner.noise_scale = 1.0

    monkeypatch.setattr(
        NemotronVoiceChatTalkerModelRunner, "__init__", initialize_runner
    )
    runner = DuplexTalkerRunner(
        Mock(spec=ModelWorker), Mock(spec=SGLangOutputProcessor)
    )
    first_codes = runner.generate_codes(0)
    saved_codes = first_codes.clone()
    second_codes = runner.generate_codes(0)
    assert not torch.equal(first_codes, second_codes)
    runner.model.hidden_out.fill_(10)
    changed_codes = runner.generate_codes(0)
    assert torch.all(changed_codes >= 10)
    assert torch.all(changed_codes < 11)
    torch.testing.assert_close(first_codes, saved_codes, rtol=0, atol=0)


@torch.inference_mode()
def test_codec_replay_matches_eager_for_changing_codes() -> None:
    def decode_codes(codes: torch.Tensor) -> torch.Tensor:
        return codes.float().sin().sum(-1).repeat_interleave(4)

    hooks = CodecHooks(decode_codes, "cuda")
    for frame_count in [1, 15, 16, 16, 16]:
        codes = torch.randint(0, 100, (frame_count, 8), device="cuda")
        torch.testing.assert_close(
            hooks.decode(codes), decode_codes(codes), rtol=0, atol=0
        )


@torch.inference_mode()
def test_perception_capture_and_replay_match_causal_eager_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def initialize_stream(
        stream: StreamingPerception, perception: AudioPerception
    ) -> None:
        stream.device = torch.device("cuda")
        stream.dtype = torch.float32
        stream.max_keys = 2
        stream.sample_buffer = torch.zeros(1, device="cuda")
        stream.preemphasis_carry = torch.zeros(1, device="cuda")
        stream.sub_caches = [torch.zeros(2, device="cuda")]
        stream.key_caches = [torch.zeros(2, device="cuda")]
        stream.value_caches = [torch.zeros(2, device="cuda")]
        stream.conv_caches = [torch.zeros(2, device="cuda")]

    def push_samples(
        stream: StreamingPerception, samples: torch.Tensor
    ) -> torch.Tensor:
        stream.preemphasis_carry = stream.preemphasis_carry + samples[:1]
        stream.sample_buffer = stream.sample_buffer + samples[:1]
        stream.sub_caches = [stream.sub_caches[0] + samples[:1]]
        stream.key_caches = [stream.key_caches[0] + samples[:1]]
        stream.value_caches = [stream.value_caches[0] + samples[:1]]
        stream.conv_caches = [stream.conv_caches[0] + samples[:1]]
        return (
            stream.preemphasis_carry
            + stream.sample_buffer
            + sum(
                cache.sum()
                for cache in (
                    stream.sub_caches[0],
                    stream.key_caches[0],
                    stream.value_caches[0],
                    stream.conv_caches[0],
                )
            )
        )

    monkeypatch.setattr(StreamingPerception, "__init__", initialize_stream)
    monkeypatch.setattr(StreamingPerception, "push", push_samples)
    eager_stream = StreamingPerception(Mock(spec=AudioPerception))
    graph_stream = GraphPerception(Mock(spec=AudioPerception))
    for frame_index in range(1, 6):
        samples = torch.full((1280,), float(frame_index), device="cuda")
        torch.testing.assert_close(
            graph_stream.push(samples), eager_stream.push(samples), rtol=0, atol=0
        )
