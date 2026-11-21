# SPDX-License-Identifier: Apache-2.0
"""Intel XPU adaptation tests for Qwen3-ASR.

Every case replays the audio encoder graph on an Intel XPU with a seeded random
tower captured once per module, so they need no checkpoint, and skip elsewhere.
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
MAX_TOKENS_PER_CLIP = 104
MAX_BATCH_SIZE = 2
# Replay matched eager exactly on Arc Pro B60; a fraction of the eager peak,
# because an absolute budget would admit a replay that read none of the input.
MAX_REPLAY_DIFFERENCE_FRACTION = 0.02
# Two clips have to land further apart than that budget or a stale buffer reads
# as a pass: at 0.05 this tower separates them by 0.7% of peak, at 1.0 by 11.6%.
FEATURE_SCALE = 1.0


@pytest.fixture(scope="module")
def xpu_device() -> torch.device:
    device = require_device_streams()
    if device.type != "xpu":
        pytest.skip(f"needs an Intel XPU, found {device.type}")
    else:
        pass
    return device


@pytest.fixture(scope="module")
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


def encoder_token_count(frame_counts: list[int]) -> int:
    return sum(qwen3_asr_num_audio_tokens(frames) for frames in frame_counts)


def planned_and_eager(
    model: Qwen3ASRForConditionalGeneration,
    device: torch.device,
    frame_counts: list[int],
    *,
    seed: int,
    feature_dtype: torch.dtype = torch.bfloat16,
    padding_frames: int = 0,
    padding_value: float = 0.0,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    torch.manual_seed(seed)
    clips = [
        (torch.randn(1, MEL_BINS, frames) * FEATURE_SCALE).to(feature_dtype)
        for frames in frame_counts
    ]
    items: list[MultimodalDataItem] = []
    for clip, frames in zip(clips, frame_counts):
        feature = torch.full(
            (1, MEL_BINS, frames + padding_frames), padding_value, dtype=feature_dtype
        )
        feature[:, :, :frames] = clip
        mask = torch.zeros(1, frames + padding_frames, dtype=torch.long)
        mask[0, :frames] = 1
        items.append(
            MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=feature,
                model_specific_data={
                    "feature_attention_mask": mask,
                    "num_audio_tokens": qwen3_asr_num_audio_tokens(frames),
                },
            )
        )
    packed_features = (
        torch.cat([clip[0] for clip in clips], dim=1).to(torch.bfloat16).to(device)
    )
    with torch.no_grad():
        expected = model.audio_tower(
            packed_features,
            feature_lens=torch.tensor(frame_counts, device=device),
        ).last_hidden_state
    return model.planned_audio_feature(items, device), expected


def assert_matches_eager(
    planned: torch.Tensor | None, expected: torch.Tensor, token_count: int
) -> None:
    assert planned is not None
    assert planned.device.type == "xpu"
    assert planned.dtype == torch.bfloat16
    assert planned.shape == expected.shape
    assert planned.shape[:2] == (1, token_count)
    assert torch.isfinite(planned).all()
    peak = expected.float().abs().max().item()
    difference = (planned.float() - expected.float()).abs().max().item()
    assert (
        difference <= MAX_REPLAY_DIFFERENCE_FRACTION * peak
    ), f"max|difference|={difference} against eager peak={peak}"


@pytest.mark.accelerator
def test_init_encoder_graphs_captures_every_bucket_on_an_xpu_tower(
    graphed_model: Qwen3ASRForConditionalGeneration,
) -> None:
    """A device gate here leaves the encoder eager on XPU without failing."""
    runner = graphed_model.encoder_graph_runner

    assert isinstance(runner, Qwen3ASREncoderLayerStackGraphRunner)
    assert isinstance(runner.graph_backend, XpuDeviceGraphBackend)
    assert set(runner.graphs) == set(runner.buckets)
    assert not runner.failed


@pytest.mark.accelerator
def test_planned_audio_feature_replays_what_the_eager_tower_computes(
    graphed_model: Qwen3ASRForConditionalGeneration, xpu_device: torch.device
) -> None:
    """The host-planned encode is the path a served request takes on XPU."""
    frame_counts = [300, 500]

    planned, expected = planned_and_eager(
        graphed_model, xpu_device, frame_counts, seed=0
    )

    assert_matches_eager(planned, expected, encoder_token_count(frame_counts))


@pytest.mark.accelerator
def test_one_captured_bucket_serves_repeated_inputs_without_leaking(
    graphed_model: Qwen3ASRForConditionalGeneration, xpu_device: torch.device
) -> None:
    frame_counts = [300, 500]
    token_count = encoder_token_count(frame_counts)
    runner = graphed_model.encoder_graph_runner
    graphs_before = dict(runner.graphs)

    first_planned, first_expected = planned_and_eager(
        graphed_model, xpu_device, frame_counts, seed=11
    )
    assert_matches_eager(first_planned, first_expected, token_count)
    first_snapshot = first_planned.clone()
    for seed in (12, 13):
        planned, expected = planned_and_eager(
            graphed_model, xpu_device, frame_counts, seed=seed
        )
        assert_matches_eager(planned, expected, token_count)
        assert not torch.equal(planned, first_snapshot)

    assert torch.equal(first_planned, first_snapshot)
    assert set(runner.graphs) == set(graphs_before)
    assert all(
        runner.graphs[bucket] is entry for bucket, entry in graphs_before.items()
    )


@pytest.mark.accelerator
@pytest.mark.parametrize("frames", [99, 101, 507, 799, 800, 801])
def test_a_clip_that_fills_no_whole_chunk_or_window_matches_eager(
    graphed_model: Qwen3ASRForConditionalGeneration,
    xpu_device: torch.device,
    frames: int,
) -> None:
    planned, expected = planned_and_eager(
        graphed_model, xpu_device, [frames], seed=frames
    )

    assert_matches_eager(planned, expected, encoder_token_count([frames]))


@pytest.mark.accelerator
def test_the_same_token_count_split_at_other_clip_boundaries_matches_eager(
    graphed_model: Qwen3ASRForConditionalGeneration, xpu_device: torch.device
) -> None:
    first_frames = [300, 500]
    second_frames = [400, 400]
    token_count = encoder_token_count(first_frames)
    assert encoder_token_count(second_frames) == token_count

    first_planned, first_expected = planned_and_eager(
        graphed_model, xpu_device, first_frames, seed=21
    )
    assert_matches_eager(first_planned, first_expected, token_count)
    first_snapshot = first_planned.clone()

    second_planned, second_expected = planned_and_eager(
        graphed_model, xpu_device, second_frames, seed=22
    )

    assert_matches_eager(second_planned, second_expected, token_count)
    assert torch.equal(first_planned, first_snapshot)


@pytest.mark.accelerator
def test_frames_the_mask_excludes_cannot_reach_the_output(
    graphed_model: Qwen3ASRForConditionalGeneration, xpu_device: torch.device
) -> None:
    frame_counts = [507]
    quiet_padding, expected = planned_and_eager(
        graphed_model, xpu_device, frame_counts, seed=31, padding_frames=93
    )
    loud_padding, _ = planned_and_eager(
        graphed_model,
        xpu_device,
        frame_counts,
        seed=31,
        padding_frames=93,
        padding_value=1e4,
    )

    assert_matches_eager(quiet_padding, expected, encoder_token_count(frame_counts))
    assert torch.equal(quiet_padding, loud_padding)


@pytest.mark.accelerator
def test_float32_features_reach_the_bfloat16_tower(
    graphed_model: Qwen3ASRForConditionalGeneration, xpu_device: torch.device
) -> None:
    frame_counts = [301]

    planned, expected = planned_and_eager(
        graphed_model,
        xpu_device,
        frame_counts,
        seed=41,
        feature_dtype=torch.float32,
    )

    assert_matches_eager(planned, expected, encoder_token_count(frame_counts))


@pytest.mark.accelerator
def test_replaying_another_bucket_leaves_the_first_bucket_result_intact(
    graphed_model: Qwen3ASRForConditionalGeneration, xpu_device: torch.device
) -> None:
    runner = graphed_model.encoder_graph_runner
    small_frames = [300, 500]
    large_frames = [799, 799]
    small_tokens = encoder_token_count(small_frames)
    large_tokens = encoder_token_count(large_frames)
    assert (
        runner.plan(small_tokens, len(small_frames))[0]
        != runner.plan(large_tokens, len(large_frames))[0]
    )

    small_planned, small_expected = planned_and_eager(
        graphed_model, xpu_device, small_frames, seed=51
    )
    assert_matches_eager(small_planned, small_expected, small_tokens)
    small_snapshot = small_planned.clone()

    large_planned, large_expected = planned_and_eager(
        graphed_model, xpu_device, large_frames, seed=52
    )
    assert_matches_eager(large_planned, large_expected, large_tokens)
    large_snapshot = large_planned.clone()

    again_planned, again_expected = planned_and_eager(
        graphed_model, xpu_device, small_frames, seed=53
    )

    assert_matches_eager(again_planned, again_expected, small_tokens)
    assert torch.equal(small_planned, small_snapshot)
    assert torch.equal(large_planned, large_snapshot)
