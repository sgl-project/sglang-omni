# SPDX-License-Identifier: Apache-2.0
"""Intel XPU adaptation tests for Qwen3-ASR.

Both cases capture the audio encoder graph on an Intel XPU with a seeded random
tower, so they need no checkpoint, and skip elsewhere.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.distributed.parallel_state import (
    get_default_distributed_backend,
    model_parallel_is_initialized,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from sglang.srt.runtime_context import get_context, get_parallel

from sglang_omni.models.qwen3_asr.audio_lengths import qwen3_asr_num_audio_tokens
from sglang_omni.models.qwen3_asr.encoder_cuda_graph import (
    Qwen3ASREncoderLayerStackGraphRunner,
)
from sglang_omni.models.qwen3_asr.sglang_model import Qwen3ASRForConditionalGeneration
from sglang_omni.platforms.device_graph import XpuDeviceGraphBackend
from tests.unit_test.fixtures.accelerator import require_device_streams

MEL_BINS = 128
# Encoder tokens for the longest clip the buckets must cover: one attention
# window, which is what this tower's n_window and n_window_infer pair admits.
MAX_TOKENS_PER_CLIP = 104
MAX_BATCH_SIZE = 2
# Replay matched the eager tower exactly on Arc Pro B60. The budget is a
# fraction of the eager output's own peak because this tower's output is small:
# an absolute budget would also admit a replay that read none of the input.
MAX_REPLAY_DIFFERENCE_FRACTION = 0.02


@pytest.fixture
def xpu_device() -> torch.device:
    device = require_device_streams()
    if device.type != "xpu":
        pytest.skip(f"needs an Intel XPU, found {device.type}")
    else:
        pass
    return device


@pytest.fixture
def graphed_model(
    xpu_device: torch.device,
) -> Iterator[Qwen3ASRForConditionalGeneration]:
    with (
        get_context().override_server_args(
            model_path="Qwen/Qwen3-ASR-1.7B", mm_attention_backend="triton_attn"
        ),
        get_parallel().override(tp_rank=0, attn_tp_rank=0),
    ):
        if not model_parallel_is_initialized():
            init_distributed_environment(
                backend=get_default_distributed_backend(xpu_device.type),
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method="tcp://127.0.0.1:29607",
            )
            initialize_model_parallel()
        else:
            pass
        model = Qwen3ASRForConditionalGeneration.__new__(
            Qwen3ASRForConditionalGeneration
        )
        torch.nn.Module.__init__(model)
        model.audio_tower = seeded_audio_tower(xpu_device)
        model.encoder_graph_runner = None
        model.init_encoder_graphs(
            max_batch_size=MAX_BATCH_SIZE, max_tokens_per_clip=MAX_TOKENS_PER_CLIP
        )
        yield model


def seeded_audio_tower(device: torch.device) -> Qwen3OmniMoeAudioEncoder:
    config = Qwen3OmniMoeAudioEncoderConfig(
        d_model=64,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        output_dim=32,
        num_mel_bins=MEL_BINS,
        n_window=50,
        n_window_infer=800,
    )
    torch.manual_seed(0)
    tower = Qwen3OmniMoeAudioEncoder(config).to(device).eval().to(torch.bfloat16)
    with torch.no_grad():
        for name, parameter in tower.named_parameters():
            if parameter.dim() >= 2:
                parameter.normal_(0.0, 0.02)
            elif "weight" in name:
                parameter.fill_(1.0)
            else:
                parameter.zero_()
    return tower


@pytest.mark.accelerator
def test_init_encoder_graphs_captures_every_bucket_on_an_xpu_tower(
    graphed_model: Qwen3ASRForConditionalGeneration,
) -> None:
    """The entry point the engine calls must reach the XPU graph backend.

    A device gate here leaves the encoder eager on XPU without failing, which
    is how the capture was lost before the platform chose the backend.
    """
    runner = graphed_model.encoder_graph_runner

    assert isinstance(runner, Qwen3ASREncoderLayerStackGraphRunner)
    assert isinstance(runner.graph_backend, XpuDeviceGraphBackend)
    assert set(runner.graphs) == set(runner.buckets)
    assert not runner.failed


@pytest.mark.accelerator
def test_planned_audio_feature_replays_what_the_eager_tower_computes(
    graphed_model: Qwen3ASRForConditionalGeneration, xpu_device: torch.device
) -> None:
    """The host-planned encode is the path a served request takes on XPU.

    Its mel chunks and valid-row gather are copied to the device without any
    read back, so a replay reading a buffer the copies have not reached yet
    would return finite values that are not the tower's.
    """
    frame_counts = [300, 500]
    token_counts = [qwen3_asr_num_audio_tokens(frames) for frames in frame_counts]
    features = [
        (torch.randn(1, MEL_BINS, frames) * 0.05).to(torch.bfloat16)
        for frames in frame_counts
    ]
    items = [
        MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=feature,
            model_specific_data={
                "feature_attention_mask": torch.ones(
                    1, feature.shape[-1], dtype=torch.long
                ),
                "num_audio_tokens": tokens,
            },
        )
        for feature, tokens in zip(features, token_counts)
    ]
    packed_features = torch.cat([feature[0] for feature in features], dim=1).to(
        xpu_device
    )
    with torch.no_grad():
        expected = graphed_model.audio_tower(
            packed_features,
            feature_lens=torch.tensor(frame_counts, device=xpu_device),
        ).last_hidden_state

    planned = graphed_model.planned_audio_feature(items, xpu_device)

    assert planned is not None
    assert planned.device.type == "xpu"
    assert planned.dtype == torch.bfloat16
    assert planned.shape == (1, sum(token_counts), expected.shape[-1])
    assert torch.isfinite(planned).all()
    peak = expected.float().abs().max().item()
    difference = (planned.float() - expected.float()).abs().max().item()
    assert (
        difference <= MAX_REPLAY_DIFFERENCE_FRACTION * peak
    ), f"max|difference|={difference} against eager peak={peak}"
