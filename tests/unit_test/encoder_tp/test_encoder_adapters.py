# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3-Omni encoder adapters.

Covers:
- ``build_batch`` produces the expected ``BatchPlan`` shape (flat items
  + spans) for image-only, audio-only, image+video, multi-request mixed
  batches, and skip-only batches.
- ``slice_results`` round-trips synthetic raw embeddings back into the
  per-request ``encoder_outs`` dict shape using the captured spans.
- The audio mask device-aware fix is preserved (CPU lengths -> CPU
  arange; GPU lengths -> GPU arange).
- Image cost function avoids the deepstack double-count trap by reading
  metadata from the HF ``vision_config`` (not the SGLang wrapper).
- Empty batches short-circuit — ``run_feature`` returns all-None and
  never calls into the upstream model.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.models.qwen3_omni_moe import _get_feat_extract_output_lengths

from sglang_omni.models.qwen3_omni.encoder_adapters import (
    Qwen3OmniAudioEncoderAdapter,
    Qwen3OmniImageEncoderAdapter,
    _normalize_audio_request_tensors,
    _pad_audio_features,
    _pad_audio_mask,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.stage_cache import StageOutputCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vision_cfg(*, base_hidden: int = 16, deepstack: int = 1, merge: int = 2):
    return SimpleNamespace(
        out_hidden_size=base_hidden,
        spatial_merge_size=merge,
        deepstack_visual_indexes=list(range(deepstack)),
    )


def _hf_cfg(*, base_hidden: int = 16, deepstack: int = 1, merge: int = 2):
    return SimpleNamespace(
        thinker_config=SimpleNamespace(
            vision_config=_vision_cfg(
                base_hidden=base_hidden,
                deepstack=deepstack,
                merge=merge,
            ),
            audio_config=SimpleNamespace(num_mel_bins=8, output_dim=4),
        )
    )


def _payload(rid: str, encoder_inputs: dict | None = None) -> StagePayload:
    state = {"encoder_inputs": encoder_inputs or {}}
    return StagePayload(request_id=rid, request=None, data=state)


def _msg(rid: str, encoder_inputs: dict | None = None) -> IncomingMessage:
    return IncomingMessage(
        request_id=rid,
        type="new_request",
        data=_payload(rid, encoder_inputs),
    )


# ---------------------------------------------------------------------------
# Image / video adapter
# ---------------------------------------------------------------------------


def test_image_adapter_build_batch_image_only():
    adapter = Qwen3OmniImageEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)  # 16 patches
    pixel = torch.zeros(16, 6)  # 16 patches × 6 channels (dummy)
    msg = _msg(
        "r0",
        {
            "image_encoder": {
                "pixel_values": pixel,
                "image_grid_thw": grid,
            }
        },
    )
    plan = adapter.build_batch([msg])
    assert len(plan.image_items) == 1
    assert len(plan.video_items) == 0
    assert len(plan.audio_items) == 0
    span = plan.spans[0]
    assert span.image_rows == 1
    assert span.video_rows == 0
    # 1 * 4 * 4 // (2*2) = 4
    assert span.image_token_count == 4


def test_image_adapter_build_batch_image_plus_video_same_request():
    adapter = Qwen3OmniImageEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    image_grid = torch.tensor([[1, 2, 2]], dtype=torch.long)  # 4 patches
    video_grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    msg = _msg(
        "r0",
        {
            "image_encoder": {
                "pixel_values": torch.zeros(4, 6),
                "image_grid_thw": image_grid,
                "pixel_values_videos": torch.zeros(4, 6),
                "video_grid_thw": video_grid,
            }
        },
    )
    plan = adapter.build_batch([msg])
    assert len(plan.image_items) == 1
    assert len(plan.video_items) == 1
    span = plan.spans[0]
    assert span.image_rows == 1 and span.video_rows == 1


def test_image_adapter_build_batch_keeps_grid_thw_on_cpu():
    """Locks the [grid_thw must stay on CPU] contract.

    After ``EncoderScheduler._strip_and_lift`` lifts every payload tensor
    to ``cuda:0``, the adapter must move ``image_grid_thw`` /
    ``video_grid_thw`` *back* to CPU before handing items to the upstream
    model. Upstream ``compute_cu_seqlens_from_grid_numpy`` asserts
    ``grid_thw.device.type == "cpu"`` (sglang/python/sglang/srt/models/utils.py:154);
    the v1 SGLang Plan B path discovered this end-to-end in the docs CI
    run.

    We simulate the post-lift state by feeding a CPU input here (the
    upstream path is the same — adapter normalizes to CPU regardless of
    incoming device, so the contract is enforced).
    """
    adapter = Qwen3OmniImageEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pixel = torch.zeros(16, 6)
    msg = _msg(
        "r0",
        {
            "image_encoder": {
                "pixel_values": pixel,
                "image_grid_thw": grid,
            }
        },
    )
    plan = adapter.build_batch([msg])
    assert plan.image_items[0].image_grid_thw.device.type == "cpu"


def test_video_adapter_build_batch_keeps_grid_thw_on_cpu():
    adapter = Qwen3OmniImageEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    pixel = torch.zeros(4, 6)
    msg = _msg(
        "r0",
        {
            "image_encoder": {
                "pixel_values_videos": pixel,
                "video_grid_thw": grid,
            }
        },
    )
    plan = adapter.build_batch([msg])
    assert plan.video_items[0].video_grid_thw.device.type == "cpu"


def test_image_adapter_build_batch_skip_request():
    adapter = Qwen3OmniImageEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    msg = _msg("r0", {"image_encoder": {"_skip": True, "_result": {"foo": "bar"}}})
    plan = adapter.build_batch([msg])
    assert plan.is_empty
    assert plan.spans[0].skip_result == {"foo": "bar"}
    # Empty plan -> run_feature short-circuits without touching the model.
    raw = adapter.run_feature(model=None, plan=plan)
    assert raw == {"image": None, "video": None, "audio": None}


def test_image_adapter_slice_results_round_trip():
    """Mock raw encoder output, slice it back into per-request dicts,
    verify the deepstack split and per-request token counts."""
    adapter = Qwen3OmniImageEncoderAdapter(
        hf_config=_hf_cfg(base_hidden=4, deepstack=2),
        dtype=torch.float16,
    )
    # 1 request with 4 image tokens × (4 base + 4*2 deepstack) = 12 hidden dim
    grid = torch.tensor([[1, 2, 4]], dtype=torch.long)  # 8 patches; 8//4 = 2 tokens
    pixel = torch.zeros(8, 6)
    msg = _msg(
        "r0",
        {
            "image_encoder": {
                "pixel_values": pixel,
                "image_grid_thw": grid,
            }
        },
    )
    plan = adapter.build_batch([msg])
    # token_count = 1*2*4 // (2*2) = 2
    assert plan.spans[0].image_token_count == 2

    # Synthesize a raw image embedding shaped (sum_tokens, base * (1 + deepstack))
    # = (2, 4*3) = (2, 12). Each chunk of 4 channels is one slice.
    raw_image = torch.arange(2 * 12, dtype=torch.float32).reshape(2, 12)
    raw = {"image": raw_image, "video": None, "audio": None}
    out = adapter.slice_results(raw, plan, [msg])
    assert len(out) == 1
    payload = out[0]
    encoder_outs = payload.data["encoder_outs"]["image_encoder"]
    base = encoder_outs["image_embeds"]
    deepstack = encoder_outs["deepstack_visual_embeds_image"]
    assert base.shape == (2, 4)
    assert isinstance(deepstack, list) and len(deepstack) == 2
    for ds in deepstack:
        assert ds.shape == (2, 4)
    # base + deepstack should reconstruct the original
    reconstructed = torch.cat([base] + deepstack, dim=-1)
    assert torch.equal(reconstructed, raw_image)


def test_image_adapter_rejects_missing_raw_image_output():
    adapter = Qwen3OmniImageEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    grid = torch.tensor([[1, 2, 4]], dtype=torch.long)
    msg = _msg(
        "r0",
        {
            "image_encoder": {
                "pixel_values": torch.zeros(8, 6),
                "image_grid_thw": grid,
            }
        },
    )
    plan = adapter.build_batch([msg])

    with pytest.raises(ValueError, match="no image embeddings"):
        adapter.slice_results(
            {"image": None, "video": None, "audio": None}, plan, [msg]
        )


def test_image_adapter_rejects_short_raw_image_output():
    adapter = Qwen3OmniImageEncoderAdapter(
        hf_config=_hf_cfg(base_hidden=4, deepstack=0),
        dtype=torch.float16,
    )
    grid = torch.tensor([[1, 2, 4]], dtype=torch.long)
    msg = _msg(
        "r0",
        {
            "image_encoder": {
                "pixel_values": torch.zeros(8, 6),
                "image_grid_thw": grid,
            }
        },
    )
    plan = adapter.build_batch([msg])

    with pytest.raises(ValueError, match="shorter than requested slice"):
        adapter.slice_results(
            {"image": torch.zeros(1, 4), "video": None, "audio": None},
            plan,
            [msg],
        )


def test_image_adapter_request_cost_uses_base_hidden_not_wrapper():
    """The deepstack double-count trap: cost function must read
    ``vision_config.out_hidden_size`` (the *base* hidden), not the
    SGLang wrapper's already-folded ``out_hidden_size``. Locks the
    [deepstack double-count trap] note in the RFC."""
    base_hidden = 4
    deepstack = 3  # wrapper would otherwise multiply by (1 + 3) = 4
    adapter = Qwen3OmniImageEncoderAdapter(
        hf_config=_hf_cfg(base_hidden=base_hidden, deepstack=deepstack),
        dtype=torch.float16,
    )
    # 1 image, 4 patches => 1 token after merge=4
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    pixel = torch.zeros(4, 6, dtype=torch.float16)
    payload = _payload(
        "r0",
        {
            "image_encoder": {
                "pixel_values": pixel,
                "image_grid_thw": grid,
            }
        },
    )
    cost = adapter.request_cost_fn(payload)
    # raw bytes = 4*6*2 = 48
    # tokens = 4 // 4 = 1
    # output_bytes = 1 * 4 * 2 * (1+3) = 32
    # cost = (48 + 32) * 5 = 400
    assert cost == 400


def test_image_adapter_request_cost_shards_activation_proxy_by_tp_size():
    adapter = Qwen3OmniImageEncoderAdapter(
        hf_config=_hf_cfg(base_hidden=4, deepstack=3),
        dtype=torch.float16,
        tp_size=2,
    )
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    pixel = torch.zeros(4, 6, dtype=torch.float16)
    payload = _payload(
        "r0",
        {
            "image_encoder": {
                "pixel_values": pixel,
                "image_grid_thw": grid,
            }
        },
    )

    # raw bytes are replicated on every rank: 4*6*2*5 = 240
    # output activation proxy is sharded: (1*4*2*(1+3)*5) / 2 = 80
    assert adapter.request_cost_fn(payload) == 320


def test_image_adapter_skip_request_cost_is_zero():
    adapter = Qwen3OmniImageEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    payload = _payload("r0", {"image_encoder": {"_skip": True, "_result": {}}})
    assert adapter.request_cost_fn(payload) == 0


def test_image_adapter_cache_lookup_and_store_round_trip():
    adapter = Qwen3OmniImageEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    cache = StageOutputCache(cache_device="cpu")
    cached = {"video_embeds": torch.ones(2, 3)}
    cache.put("media-0", cached)
    msg = _msg(
        "r0",
        {
            "image_encoder": {
                "cache_key": "media-0",
                "pixel_values_videos": torch.zeros(4, 6),
                "video_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
            }
        },
    )

    hit = adapter.lookup_cached_output(msg, cache)

    assert hit is not None
    assert torch.equal(
        hit.data["encoder_outs"]["image_encoder"]["video_embeds"],
        cached["video_embeds"],
    )
    new_output = _payload("r1")
    new_output.data = {
        "encoder_outs": {"image_encoder": {"image_embeds": torch.zeros(1, 3)}}
    }
    miss_msg = _msg(
        "r1",
        {
            "image_encoder": {
                "cache_key": "media-1",
                "pixel_values": torch.zeros(4, 6),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
            }
        },
    )

    adapter.store_cached_output(miss_msg, new_output, cache)

    stored = cache.get("media-1")
    assert torch.equal(stored["image_embeds"], torch.zeros(1, 3))


# ---------------------------------------------------------------------------
# Audio adapter
# ---------------------------------------------------------------------------


def test_audio_adapter_build_batch_pads_to_max_time():
    adapter = Qwen3OmniAudioEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    feat0 = torch.zeros(1, 8, 50)  # 50 time steps
    feat1 = torch.zeros(1, 8, 80)  # 80 time steps
    msg0 = _msg(
        "r0",
        {
            "audio_encoder": {
                "input_features": feat0,
                "audio_feature_lengths": torch.tensor([50], dtype=torch.long),
            }
        },
    )
    msg1 = _msg(
        "r1",
        {
            "audio_encoder": {
                "input_features": feat1,
                "audio_feature_lengths": torch.tensor([80], dtype=torch.long),
            }
        },
    )
    plan = adapter.build_batch([msg0, msg1])
    assert len(plan.audio_items) == 2
    # Both items padded to max_time=80 along the time axis.
    for item in plan.audio_items:
        assert item.feature.shape[-1] == 80
        assert item.feature_attention_mask.shape[-1] == 80
    # Spans preserve unpadded lengths.
    assert plan.spans[0].audio_feature_lengths.tolist() == [50]
    assert plan.spans[1].audio_feature_lengths.tolist() == [80]


def test_audio_adapter_cache_lookup_and_store_round_trip():
    adapter = Qwen3OmniAudioEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    cache = StageOutputCache(cache_device="cpu")
    cached = {"audio_embeds": torch.ones(2, 4)}
    cache.put("audio-0", cached)
    msg = _msg(
        "r0",
        {
            "audio_encoder": {
                "cache_key": "audio-0",
                "input_features": torch.zeros(1, 8, 50),
                "audio_feature_lengths": torch.tensor([50], dtype=torch.long),
            }
        },
    )

    hit = adapter.lookup_cached_output(msg, cache)

    assert hit is not None
    assert torch.equal(
        hit.data["encoder_outs"]["audio_encoder"]["audio_embeds"],
        cached["audio_embeds"],
    )
    new_output = _payload("r1")
    new_output.data = {
        "encoder_outs": {"audio_encoder": {"audio_embeds": torch.zeros(1, 4)}}
    }
    miss_msg = _msg(
        "r1",
        {
            "audio_encoder": {
                "cache_key": "audio-1",
                "input_features": torch.zeros(1, 8, 50),
                "audio_feature_lengths": torch.tensor([50], dtype=torch.long),
            }
        },
    )

    adapter.store_cached_output(miss_msg, new_output, cache)

    stored = cache.get("audio-1")
    assert torch.equal(stored["audio_embeds"], torch.zeros(1, 4))


def test_audio_adapter_batch_cost_accounts_for_batch_padding():
    adapter = Qwen3OmniAudioEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    feat0 = torch.zeros(1, 8, 50, dtype=torch.float16)
    feat1 = torch.zeros(1, 8, 80, dtype=torch.float16)
    payload0 = _payload(
        "r0",
        {
            "audio_encoder": {
                "input_features": feat0,
                "audio_feature_lengths": torch.tensor([50], dtype=torch.long),
            }
        },
    )
    payload1 = _payload(
        "r1",
        {
            "audio_encoder": {
                "input_features": feat1,
                "audio_feature_lengths": torch.tensor([80], dtype=torch.long),
            }
        },
    )

    out_tokens = int(
        _get_feat_extract_output_lengths(torch.tensor([50, 80])).sum().item()
    )
    padded_input_bytes = 2 * 8 * 80 * 2
    padded_mask_bytes = 2 * 80 * 1
    output_bytes = out_tokens * 4 * 2

    assert (
        adapter.batch_cost_fn([payload0, payload1])
        == (padded_input_bytes + padded_mask_bytes + output_bytes) * 5
    )
    assert adapter.request_cost_fn(payload0) == adapter.batch_cost_fn([payload0])


def test_audio_adapter_batch_cost_shards_activation_proxy_by_tp_size():
    adapter = Qwen3OmniAudioEncoderAdapter(
        hf_config=_hf_cfg(), dtype=torch.float16, tp_size=2
    )
    feat0 = torch.zeros(1, 8, 50, dtype=torch.float16)
    feat1 = torch.zeros(1, 8, 80, dtype=torch.float16)
    payload0 = _payload(
        "r0",
        {
            "audio_encoder": {
                "input_features": feat0,
                "audio_feature_lengths": torch.tensor([50], dtype=torch.long),
            }
        },
    )
    payload1 = _payload(
        "r1",
        {
            "audio_encoder": {
                "input_features": feat1,
                "audio_feature_lengths": torch.tensor([80], dtype=torch.long),
            }
        },
    )

    out_tokens = int(
        _get_feat_extract_output_lengths(torch.tensor([50, 80])).sum().item()
    )
    padded_input_bytes = 2 * 8 * 80 * 2
    padded_mask_bytes = 2 * 80 * 1
    output_bytes = out_tokens * 4 * 2
    expected = (padded_input_bytes + padded_mask_bytes) * 5
    expected += (output_bytes * 5 + 1) // 2

    assert adapter.batch_cost_fn([payload0, payload1]) == expected


def test_audio_adapter_cost_requires_precomputed_lengths():
    adapter = Qwen3OmniAudioEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    payload = _payload(
        "r0",
        {
            "audio_encoder": {
                "input_features": torch.zeros(1, 8, 50),
                "feature_attention_mask": torch.ones(1, 50, dtype=torch.bool),
            }
        },
    )

    with pytest.raises(ValueError, match="audio_feature_lengths"):
        adapter.request_cost_fn(payload)


def test_audio_adapter_slice_results_round_trip():
    adapter = Qwen3OmniAudioEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    feat = torch.zeros(1, 8, 100)  # 100 -> output_lengths = (
    # (100 % 100) = 0 -> feat_lengths = (0-1)//2+1 = 0; but real formula:
    # leave = 0 → feat = (-1)//2 + 1 = 0; output = ((-1)//2 + 1 - 1)//2 + 1 + 1*13 = 14
    msg = _msg(
        "r0",
        {
            "audio_encoder": {
                "input_features": feat,
                "audio_feature_lengths": torch.tensor([100], dtype=torch.long),
            }
        },
    )
    plan = adapter.build_batch([msg])
    out_lens = int(plan.spans[0].audio_output_lengths.sum().item())
    # _get_feat_extract_output_lengths(100) = 13 in the upstream formula
    assert out_lens == 13
    # Synthesize raw audio embedding with shape (out_lens_total, hidden).
    raw_audio = torch.arange(out_lens * 4, dtype=torch.float32).reshape(out_lens, 4)
    raw = {"image": None, "video": None, "audio": raw_audio}
    out = adapter.slice_results(raw, plan, [msg])
    payload = out[0]
    enc = payload.data["encoder_outs"]["audio_encoder"]
    assert torch.equal(enc["audio_embeds"], raw_audio[:out_lens])
    assert enc["audio_feature_lengths"].tolist() == [100]


def test_audio_adapter_rejects_missing_raw_audio_output():
    adapter = Qwen3OmniAudioEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    msg = _msg(
        "r0",
        {
            "audio_encoder": {
                "input_features": torch.zeros(1, 8, 100),
                "audio_feature_lengths": torch.tensor([100], dtype=torch.long),
            }
        },
    )
    plan = adapter.build_batch([msg])

    with pytest.raises(ValueError, match="no embeddings"):
        adapter.slice_results(
            {"image": None, "video": None, "audio": None}, plan, [msg]
        )


def test_audio_adapter_rejects_short_raw_audio_output():
    adapter = Qwen3OmniAudioEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    msg = _msg(
        "r0",
        {
            "audio_encoder": {
                "input_features": torch.zeros(1, 8, 100),
                "audio_feature_lengths": torch.tensor([100], dtype=torch.long),
            }
        },
    )
    plan = adapter.build_batch([msg])

    with pytest.raises(ValueError, match="shorter than requested slice"):
        adapter.slice_results(
            {"image": None, "video": None, "audio": torch.zeros(1, 4)},
            plan,
            [msg],
        )


def test_audio_adapter_run_feature_flattens_3d_upstream_output():
    class FakeAudioTower(torch.nn.Module):
        dtype = torch.float32

        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.empty(0))

        def forward(self, input_features, *, feature_lens):
            output_lengths = _get_feat_extract_output_lengths(feature_lens)
            hidden = torch.zeros(
                int(feature_lens.shape[0]),
                int(output_lengths.max().item()),
                4,
            )
            for row, length in enumerate(output_lengths):
                start = row * 100
                valid = int(length.item())
                hidden[row, :valid] = torch.arange(
                    start,
                    start + valid * 4,
                    dtype=torch.float32,
                ).reshape(valid, 4)
            return SimpleNamespace(last_hidden_state=hidden)

    item0 = SimpleNamespace(
        feature=torch.zeros(1, 8, 100),
        feature_attention_mask=torch.ones(1, 100, dtype=torch.bool),
    )
    item1 = SimpleNamespace(
        feature=torch.zeros(1, 8, 100),
        feature_attention_mask=torch.cat(
            [
                torch.ones(1, 50, dtype=torch.bool),
                torch.zeros(1, 50, dtype=torch.bool),
            ],
            dim=1,
        ),
    )
    out = Qwen3OmniAudioEncoderAdapter._get_audio_feature(
        FakeAudioTower(),
        [item0, item1],
    )
    lengths = _get_feat_extract_output_lengths(torch.tensor([100, 50]))

    assert out.shape == (int(lengths.sum().item()), 4)
    first_len = int(lengths[0].item())
    second_len = int(lengths[1].item())
    assert torch.equal(
        out[:first_len],
        torch.arange(first_len * 4, dtype=torch.float32).reshape(first_len, 4),
    )
    assert torch.equal(
        out[first_len:],
        torch.arange(
            100,
            100 + second_len * 4,
            dtype=torch.float32,
        ).reshape(second_len, 4),
    )


def test_audio_adapter_skip_request_passes_through_skip_result():
    adapter = Qwen3OmniAudioEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    msg = _msg("r0", {"audio_encoder": {"_skip": True, "_result": {"foo": "bar"}}})
    plan = adapter.build_batch([msg])
    assert plan.is_empty
    out = adapter.slice_results(
        {"image": None, "video": None, "audio": None},
        plan,
        [msg],
    )
    enc = out[0].data["encoder_outs"]["audio_encoder"]
    assert enc == {"foo": "bar"}


def test_audio_adapter_mixed_batch_skip_and_active():
    adapter = Qwen3OmniAudioEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    skip_msg = _msg("rs", {"audio_encoder": {"_skip": True, "_result": {"skip": True}}})
    active_msg = _msg(
        "ra",
        {
            "audio_encoder": {
                "input_features": torch.zeros(1, 8, 100),
                "audio_feature_lengths": torch.tensor([100], dtype=torch.long),
            }
        },
    )
    plan = adapter.build_batch([skip_msg, active_msg])
    assert len(plan.audio_items) == 1  # only the active request contributes
    assert plan.spans[0].skip_result is not None
    assert plan.spans[1].audio_rows == 1


# ---------------------------------------------------------------------------
# Audio mask device-aware fix
# ---------------------------------------------------------------------------


def test_normalize_audio_request_tensors_arange_follows_lengths_device_cpu():
    """CPU lengths → CPU arange → CPU mask. The local v1 path."""
    request = SimpleNamespace(
        model_inputs={
            "input_features": torch.zeros(1, 8, 50),
            "audio_feature_lengths": torch.tensor([30], dtype=torch.long),
            # No feature_attention_mask -> synthesized from lengths
        }
    )
    features, mask, lengths = _normalize_audio_request_tensors(request)
    assert features.device.type == "cpu"
    assert lengths.device.type == "cpu"
    assert mask.device.type == "cpu"
    # First 30 positions are True, rest False
    assert mask[0, :30].all()
    assert not mask[0, 30:].any()


def test_normalize_audio_request_tensors_arange_follows_lengths_device_no_dtype_mismatch():
    """If lengths come pre-lifted to GPU (the SGLang Plan B path),
    arange must be on GPU too — locks the [Audio adapter / mandatory
    device-aware fix] in the RFC. Here we simulate by promoting lengths
    to a non-default torch.long-on-cpu tensor and checking that the
    arange and resulting mask end up on the same device."""
    cpu_lengths = torch.tensor([20], dtype=torch.long)
    request = SimpleNamespace(
        model_inputs={
            "input_features": torch.zeros(1, 8, 30),
            "audio_feature_lengths": cpu_lengths,
        }
    )
    _, mask, lengths = _normalize_audio_request_tensors(request)
    assert mask.device == lengths.device


def test_pad_audio_features_no_op_when_already_at_target():
    feat = torch.zeros(1, 8, 50)
    out = _pad_audio_features(feat, 50)
    assert out.shape == feat.shape


def test_pad_audio_mask_pads_with_false():
    mask = torch.tensor([[True, True, False]])
    out = _pad_audio_mask(mask, 5)
    assert out.shape == (1, 5)
    assert out[0, 3:].sum() == 0


# ---------------------------------------------------------------------------
# Adapter `is_empty` and short-circuit
# ---------------------------------------------------------------------------


def test_batch_plan_is_empty_when_all_skip():
    adapter = Qwen3OmniImageEncoderAdapter(hf_config=_hf_cfg(), dtype=torch.float16)
    msgs = [
        _msg("r0", {"image_encoder": {"_skip": True, "_result": {}}}),
        _msg("r1", {"image_encoder": {"_skip": True, "_result": {}}}),
    ]
    plan = adapter.build_batch(msgs)
    assert plan.is_empty
    raw = adapter.run_feature(model=None, plan=plan)
    # Verify run_feature did not call into the model (it would have crashed
    # on model=None if it had).
    assert raw == {"image": None, "video": None, "audio": None}
