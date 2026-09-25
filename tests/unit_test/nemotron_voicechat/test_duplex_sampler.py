# SPDX-License-Identifier: Apache-2.0
"""GPU sampler replay must consume new input/RNG and return owned outputs."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.duplex_ar import DuplexTalkerRunner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires GPU")
@torch.inference_mode()
def test_sampler_replay_updates_input_and_randomness():
    seen = []

    def sample(hidden, head, **kwargs):
        counts = kwargs["assignment_counts"]
        assert sum(counts) == 8
        seen.append(counts)
        return hidden + torch.rand_like(hidden)

    runner = object.__new__(DuplexTalkerRunner)
    runner.model = SimpleNamespace(
        hidden_out=torch.zeros(1, 16, device="cuda"),
        talker=SimpleNamespace(num_quantizers=8, generate_codes=sample),
        mog_head=None,
    )
    runner.exponent, runner.top_p, runner.noise_scale = 1.0, 0.9, 1.0
    first = runner.generate_codes(0)
    saved = first.clone()
    second = runner.generate_codes(0)
    assert not torch.equal(first, second)
    runner.model.hidden_out.fill_(10)
    third = runner.generate_codes(0)
    assert torch.all(third >= 10)
    assert torch.all(third < 11)
    assert torch.equal(first, saved)
    assert all(counts == seen[0] for counts in seen)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires GPU")
@torch.inference_mode()
def test_codec_replay_matches_eager_for_changing_codes():
    from sglang_omni.models.nemotron_voicechat.duplex import CodecHooks

    def decoder(codes):
        return codes.float().sin().sum(-1).repeat_interleave(4)

    hooks = CodecHooks(decoder, "cuda")
    for frames in [1, 15, 16, 16, 16]:
        codes = torch.randint(0, 100, (frames, 8), device="cuda")
        torch.testing.assert_close(hooks.decode(codes), decoder(codes))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires GPU")
@torch.inference_mode()
def test_perception_capture_does_not_advance_history(monkeypatch):
    from sglang_omni.models.nemotron_voicechat.conformer import StreamingPerception
    from sglang_omni.models.nemotron_voicechat.duplex import GraphPerception

    def push(self, samples):
        output = self.preemphasis_carry + samples[:1]
        self.preemphasis_carry = output
        self.sample_buffer = self.sample_buffer + samples[:1]
        for name in self.LIST_BUFFERS:
            setattr(
                self, name, [tensor + samples[:1] for tensor in getattr(self, name)]
            )
        return output

    monkeypatch.setattr(StreamingPerception, "push", push)
    stream = object.__new__(GraphPerception)
    stream.device, stream.dtype, stream.max_keys = (
        torch.device("cuda"),
        torch.float32,
        2,
    )
    for name in stream.SINGLE_BUFFERS:
        setattr(stream, name, torch.zeros(1, device="cuda"))
    for name in stream.LIST_BUFFERS:
        setattr(stream, name, [torch.zeros(2, device="cuda")])
    for value in range(1, 5):
        result = stream.push(torch.ones(1280, device="cuda"))
        torch.testing.assert_close(result, torch.tensor([float(value)], device="cuda"))
        for buffer in stream.state_buffers():
            torch.testing.assert_close(buffer, torch.full_like(buffer, value))
