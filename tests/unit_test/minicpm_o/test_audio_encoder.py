# SPDX-License-Identifier: Apache-2.0
"""Tests for the native MiniCPM-o audio encoder.

The golden-parity test compares the native encoder against the checkpoint's
remote-code MiniCPMWhisperEncoder on shared random weights, so it needs a
checkpoint directory with the remote modeling files (weights not required).
Set MINICPMO_CHECKPOINT or place MiniCPM-o-4_6 / MiniCPM-o-4_5 in the repo
root; the test skips otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import sglang_omni.models.minicpm_o.audio_encoder_batching as audio_encoder_batching
from sglang_omni.models.minicpm_o.audio_encoder_batching import (
    batch_audio_encoder_payloads,
    encode_audio_payload,
)
from sglang_omni.models.minicpm_o.components.audio_encoder import (
    MiniCPMOAudioEncoder,
    MiniCPMWhisperEncoder,
    MultiModalProjector,
    chunked_causal_mask,
    feature_lens_after_pooling,
    fuse_qkv,
    min_mel_frames,
)
from sglang_omni.models.minicpm_o.config import audio_encoder_stage
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.stage_cache import StageOutputCache

REPO_ROOT = Path(__file__).resolve().parents[3]


def _checkpoint_dir() -> Path | None:
    env = os.environ.get("MINICPMO_CHECKPOINT")
    candidates = [Path(env)] if env else []
    candidates += [REPO_ROOT / "MiniCPM-o-4_6", REPO_ROOT / "MiniCPM-o-4_5"]
    for path in candidates:
        if (path / "modeling_minicpmo.py").exists():
            return path
    return None


def _small_whisper_config():
    from transformers import WhisperConfig

    return WhisperConfig(
        num_mel_bins=80,
        d_model=64,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=256,
        max_source_positions=1500,
        activation_function="gelu",
    )


def _native_state_from_hf(encoder: torch.nn.Module) -> dict[str, torch.Tensor]:
    return fuse_qkv(dict(encoder.state_dict()))


def _build_remote_encoder(checkpoint: Path, config):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    remote_cls = get_class_from_dynamic_module(
        "modeling_minicpmo.MiniCPMWhisperEncoder", str(checkpoint)
    )
    config._attn_implementation = "sdpa"
    remote = remote_cls(config).eval()
    # transformers v5 attention returns (out, weights); the remote layer
    # unpacks a v4-era 3-tuple. Pad the return for the golden run.
    for layer in remote.layers:
        attn = layer.self_attn
        orig_forward = attn.forward

        def _forward(*args, _orig=orig_forward, **kwargs):
            out = _orig(*args, **kwargs)
            if isinstance(out, tuple) and len(out) == 2:
                return (*out, None)
            return out

        attn.forward = _forward
    return remote


@pytest.mark.parametrize("lens", [[3000, 2000, 137], [700]])
def test_golden_parity_vs_remote_code(lens: list[int]) -> None:
    checkpoint = _checkpoint_dir()
    if checkpoint is None:
        pytest.skip("no MiniCPM-o checkpoint with remote modeling files")

    torch.manual_seed(0)
    config = _small_whisper_config()
    remote = _build_remote_encoder(checkpoint, config)

    native = MiniCPMWhisperEncoder(config).eval()
    native.load_state_dict(_native_state_from_hf(remote), strict=True)

    batch = len(lens)
    max_mel = max(lens)
    mel = torch.randn(batch, config.num_mel_bins, max_mel)
    for i, length in enumerate(lens):
        mel[i, :, length:] = 0.0
    feat_lens = torch.tensor(lens)

    max_seq_len = (max_mel - 1) // 2 + 1
    # chunk spanning frame boundaries: audio_chunk_length=1s → 50 frames
    chunk = 50
    seq_range = torch.arange(max_seq_len)
    valid = seq_range[None, :] < ((feat_lens - 1) // 2 + 1)[:, None]
    allowed = (
        chunked_causal_mask(max_seq_len, chunk, torch.device("cpu"))[None]
        & valid[:, None, :]
    )

    # Remote path: additive -inf mask, output_hidden_states, last hidden state.
    remote_mask = torch.zeros(batch, 1, max_seq_len, max_seq_len)
    remote_mask[~allowed.unsqueeze(1).expand(batch, 1, max_seq_len, max_seq_len)] = (
        float("-inf")
    )
    with torch.no_grad():
        golden = remote(
            mel, attention_mask=remote_mask, output_hidden_states=True
        ).hidden_states[-1]

    native_mask = torch.where(allowed, 0.0, -1e9).unsqueeze(1)
    with torch.no_grad():
        got = native(mel, native_mask)

    for i, length in enumerate(lens):
        valid_frames = (length - 1) // 2 + 1
        torch.testing.assert_close(
            got[i, :valid_frames], golden[i, :valid_frames], rtol=1e-4, atol=1e-4
        )


def _tiny_audio_encoder(pool_step: int = 2) -> MiniCPMOAudioEncoder:
    """A MiniCPMOAudioEncoder with random weights and no checkpoint I/O."""
    torch.manual_seed(0)
    config = _small_whisper_config()
    encoder = object.__new__(MiniCPMOAudioEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32
    encoder.apm = MiniCPMWhisperEncoder(config)
    encoder.audio_projection_layer = MultiModalProjector(
        in_dim=config.d_model, out_dim=16
    )
    encoder.audio_pool_step = pool_step
    encoder.audio_avg_pooler = torch.nn.AvgPool1d(pool_step, stride=pool_step)
    encoder.chunk_num_frame = 50
    encoder.chunk_mask_cache = None
    return encoder


@pytest.mark.parametrize("padding", ["random", "nan"])
def test_padding_content_does_not_change_valid_output(padding: str) -> None:
    """Padding must affect neither valid embeddings nor the caller's input."""
    encoder = _tiny_audio_encoder()
    short_len, long_len = 137, 3000

    torch.manual_seed(1)
    short_mel = torch.randn(80, short_len)
    long_mel = torch.randn(80, long_len)

    padded_mel = torch.zeros(2, 80, long_len)
    padded_mel[0, :, :long_len] = long_mel
    padded_mel[1, :, :short_len] = short_mel
    lens = torch.tensor([long_len, short_len])

    with torch.no_grad():
        clean = encoder(audio_features=padded_mel, audio_feature_lens=lens)

        torch.manual_seed(2)
        polluted_mel = padded_mel.clone()
        polluted_mel[1, :, short_len:] = (
            torch.randn(80, long_len - short_len) if padding == "random" else torch.nan
        )
        original_mel = polluted_mel.clone()
        polluted = encoder(audio_features=polluted_mel, audio_feature_lens=lens)

    torch.testing.assert_close(
        polluted_mel, original_mel, rtol=0, atol=0, equal_nan=True
    )
    pooled_short = int(
        feature_lens_after_pooling(torch.tensor([short_len]), encoder.audio_pool_step)
    )
    # Rows are emitted in chunk order with each chunk trimmed to its pooled
    # length, so the short row sits at the end of the flattened output.
    torch.testing.assert_close(
        polluted["audio_embeds"][-pooled_short:],
        clean["audio_embeds"][-pooled_short:],
        rtol=0,
        atol=0,
    )


def test_short_audio_is_rejected_before_pooling() -> None:
    """A clip too short for one pooling window must raise a typed error."""
    pool_step = 5
    encoder = _tiny_audio_encoder(pool_step=pool_step)
    too_short = min_mel_frames(pool_step) - 1
    mel = torch.randn(1, 80, too_short).to(encoder.dtype)
    lens = torch.tensor([too_short])

    with pytest.raises(ValueError, match="accepts audio up to"):
        with torch.no_grad():
            encoder(audio_features=mel, audio_feature_lens=lens)


def test_sub_pooling_tail_keeps_long_audio_embeddings() -> None:
    """A partial final segment contributes no tokens without rejecting its clip."""
    encoder = _tiny_audio_encoder(pool_step=5)
    mel = torch.randn(2, 80, 3000)
    mel[1, :, 5:] = 0

    expected = encoder(audio_features=mel[:1], audio_feature_lens=torch.tensor([3000]))[
        "audio_embeds"
    ]
    actual = encoder(audio_features=mel, audio_feature_lens=torch.tensor([3000, 5]))[
        "audio_embeds"
    ]

    assert actual.shape == (300, 16)
    torch.testing.assert_close(actual, expected)


def test_minimum_length_audio_still_encodes() -> None:
    """The shortest accepted clip yields exactly one pooled frame."""
    pool_step = 5
    encoder = _tiny_audio_encoder(pool_step=pool_step)
    shortest = min_mel_frames(pool_step)
    mel = torch.randn(1, 80, shortest).to(encoder.dtype)
    lens = torch.tensor([shortest])

    with torch.no_grad():
        out = encoder(audio_features=mel, audio_feature_lens=lens)

    assert out["audio_embeds"].shape[0] == 1


def _audio_payload(
    request_id: str,
    audio_features: torch.Tensor,
    audio_feature_lens: torch.Tensor,
    *,
    cache_key: str,
) -> StagePayload:
    state = MiniCPMOPipelineState(
        encoder_inputs={
            "audio_encoder": {
                "audio_features": audio_features,
                "audio_feature_lens": audio_feature_lens,
                "cache_key": cache_key,
            }
        }
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )


def test_cross_request_batch_matches_serial_outputs() -> None:
    encoder = _tiny_audio_encoder()
    short = torch.randn(1, 80, 137)
    long = torch.randn(2, 80, 301)
    long[1, :, 201:] = 0
    payloads = [
        _audio_payload("short", short, torch.tensor([137]), cache_key="short"),
        _audio_payload("long", long, torch.tensor([301, 201]), cache_key="long"),
    ]
    serial_cache = StageOutputCache(cache_device="cpu")
    expected = [
        encode_audio_payload(
            _audio_payload(
                payload.request_id,
                payload.data["encoder_inputs"]["audio_encoder"]["audio_features"],
                payload.data["encoder_inputs"]["audio_encoder"]["audio_feature_lens"],
                cache_key=f"serial-{payload.request_id}",
            ),
            encoder=encoder,
            cache=serial_cache,
        )
        for payload in payloads
    ]

    actual = batch_audio_encoder_payloads(
        payloads,
        encoder=encoder,
        cache=StageOutputCache(cache_device="cpu"),
    )

    for actual_payload, expected_payload in zip(actual, expected):
        actual_embeds = actual_payload.data["encoder_outs"]["audio_encoder"][
            "audio_embeds"
        ]
        expected_embeds = expected_payload.data["encoder_outs"]["audio_encoder"][
            "audio_embeds"
        ]
        torch.testing.assert_close(actual_embeds, expected_embeds, rtol=1e-4, atol=1e-4)


def test_single_audio_encoder_emits_cache_and_forward_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        audio_encoder_batching,
        "emit_audio_encoder_event",
        lambda payload, event_name: events.append(event_name),
    )
    encode_audio_payload(
        _audio_payload(
            "single",
            torch.randn(1, 80, 137),
            torch.tensor([137]),
            cache_key="single",
        ),
        encoder=_tiny_audio_encoder(),
        cache=StageOutputCache(cache_device="cpu"),
    )

    assert events == [
        "encoder_cache_miss",
        "encoder_forward_start",
        "encoder_forward_end",
        "encoder_cache_put_end",
    ]


def test_cross_request_batching_is_enabled_by_default() -> None:
    stage = audio_encoder_stage(gpu=0, process="pipeline")
    assert stage.factory.max_batch_size == 8
    assert stage.factory.max_batch_wait_ms == 0


def test_cross_request_batch_reuses_cache_and_deduplicates_same_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _tiny_audio_encoder()
    cache = StageOutputCache(cache_device="cpu")
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        audio_encoder_batching,
        "emit_audio_encoder_event",
        lambda payload, event_name: events.append((payload.request_id, event_name)),
    )
    features = torch.randn(1, 80, 137)
    payloads = [
        _audio_payload("leader", features, torch.tensor([137]), cache_key="shared"),
        _audio_payload(
            "follower", torch.randn(1, 80, 201), torch.tensor([201]), cache_key="shared"
        ),
    ]

    first = batch_audio_encoder_payloads(payloads, encoder=encoder, cache=cache)
    expected = first[0].data["encoder_outs"]["audio_encoder"]["audio_embeds"]
    torch.testing.assert_close(
        first[1].data["encoder_outs"]["audio_encoder"]["audio_embeds"], expected
    )
    assert events == [
        ("leader", "encoder_cache_miss"),
        ("leader", "encoder_forward_start"),
        ("leader", "encoder_forward_end"),
        ("leader", "encoder_cache_put_end"),
    ]

    def fail_forward(**_: object) -> dict[str, torch.Tensor]:
        raise AssertionError("cache hit unexpectedly called the encoder")

    encoder.forward = fail_forward
    cached = batch_audio_encoder_payloads(
        [
            _audio_payload(
                "cached",
                torch.randn(1, 80, 301),
                torch.tensor([301]),
                cache_key="shared",
            )
        ],
        encoder=encoder,
        cache=cache,
    )
    torch.testing.assert_close(
        cached[0].data["encoder_outs"]["audio_encoder"]["audio_embeds"],
        expected,
    )
    assert events[-1] == ("cached", "encoder_cache_hit")
