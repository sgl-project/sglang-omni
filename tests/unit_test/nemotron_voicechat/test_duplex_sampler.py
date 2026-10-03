# SPDX-License-Identifier: Apache-2.0
"""CUDA graph replay preserves inputs, causal state, and random sampling."""

from unittest.mock import Mock

import pytest
import torch

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.nemotron_voicechat.duplex import CodecHooks
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
