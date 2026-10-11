# SPDX-License-Identifier: Apache-2.0
"""Tests for the native MiniCPM-o audio encoder.

The golden-parity test compares the native encoder against the checkpoint's
remote-code MiniCPMWhisperEncoder on shared random weights, so it needs a
checkpoint directory with the remote modeling files (weights not required).
Set MINICPMO_CHECKPOINT or place MiniCPM-o-4_6 / MiniCPM-o-4_5 in the repo
root; the test skips otherwise.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from sglang_omni.models.minicpm_o.audio_encoder_batching import (
    batch_audio_encoder_payloads,
)
from sglang_omni.models.minicpm_o.components.audio_encoder import (
    MiniCPMOAudioEncoder,
    MultiModalProjector,
    StreamingAudioChunk,
    chunked_causal_mask,
    feature_lens_after_pooling,
    fuse_qkv,
    min_mel_frames,
)
from sglang_omni.models.minicpm_o.components.whisper_encoder import (
    AudioEncoderState,
    MiniCPMWhisperEncoder,
)
from sglang_omni.models.minicpm_o.config import audio_encoder_stage
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.stage_cache import StageOutputCache

REPO_ROOT = Path(__file__).resolve().parents[3]


def checkpoint_dir() -> Path | None:
    env = os.environ.get("MINICPMO_CHECKPOINT")
    candidates = [Path(env)] if env else []
    candidates += [REPO_ROOT / "MiniCPM-o-4_6", REPO_ROOT / "MiniCPM-o-4_5"]
    for path in candidates:
        if (path / "modeling_minicpmo.py").exists():
            return path
    return None


def small_whisper_config():
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


def native_state_from_hf(encoder: torch.nn.Module) -> dict[str, torch.Tensor]:
    return fuse_qkv(dict(encoder.state_dict()))


def build_remote_encoder(checkpoint: Path, config):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    remote_cls = get_class_from_dynamic_module(
        "modeling_minicpmo.MiniCPMWhisperEncoder", str(checkpoint)
    )
    config._attn_implementation = "sdpa"  # noqa: leading-underscore  # production name
    remote = remote_cls(config).eval()
    # transformers v5 attention returns (out, weights); the remote layer
    # unpacks a v4-era 3-tuple. Pad the return for the golden run.
    for layer in remote.layers:
        attn = layer.self_attn
        orig_forward = attn.forward

        def forward(*args, _orig=orig_forward, **kwargs):
            out = _orig(*args, **kwargs)
            if isinstance(out, tuple) and len(out) == 2:
                return (*out, None)
            return out

        attn.forward = forward
    return remote


@pytest.mark.parametrize("lens", [[3000, 2000, 137], [700]])
def test_golden_parity_vs_remote_code(lens: list[int]) -> None:
    checkpoint = checkpoint_dir()
    if checkpoint is None:
        pytest.skip("no MiniCPM-o checkpoint with remote modeling files")

    torch.manual_seed(0)
    config = small_whisper_config()
    remote = build_remote_encoder(checkpoint, config)

    native = MiniCPMWhisperEncoder(config).eval()
    native.load_state_dict(native_state_from_hf(remote), strict=True)

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


def tiny_audio_encoder(pool_step: int = 2) -> MiniCPMOAudioEncoder:
    """A MiniCPMOAudioEncoder with random weights and no checkpoint I/O."""
    torch.manual_seed(0)
    config = small_whisper_config()
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
    encoder = tiny_audio_encoder()
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
    encoder = tiny_audio_encoder(pool_step=pool_step)
    too_short = min_mel_frames(pool_step) - 1
    mel = torch.randn(1, 80, too_short).to(encoder.dtype)
    lens = torch.tensor([too_short])

    with pytest.raises(ValueError, match="accepts audio up to"):
        with torch.no_grad():
            encoder(audio_features=mel, audio_feature_lens=lens)


def test_sub_pooling_tail_keeps_long_audio_embeddings() -> None:
    """A partial final segment contributes no tokens without rejecting its clip."""
    encoder = tiny_audio_encoder(pool_step=5)
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
    encoder = tiny_audio_encoder(pool_step=pool_step)
    shortest = min_mel_frames(pool_step)
    mel = torch.randn(1, 80, shortest).to(encoder.dtype)
    lens = torch.tensor([shortest])

    with torch.no_grad():
        out = encoder(audio_features=mel, audio_feature_lens=lens)

    assert out["audio_embeds"].shape[0] == 1


STREAMING_MEL_FRAMES = 104


def random_history(length: int, generator: torch.Generator) -> AudioEncoderState | None:
    """A session's attention history of the given length, or none for a new session."""
    config = small_whisper_config()
    head_dim = config.d_model // config.encoder_attention_heads
    if length == 0:
        return None
    else:
        return AudioEncoderState(
            key_value_states=torch.randn(
                (
                    config.encoder_layers,
                    2,
                    1,
                    config.encoder_attention_heads,
                    length,
                    head_dim,
                ),
                generator=generator,
            )
        )


@pytest.mark.parametrize(
    "history_lengths",
    [(300,), (0, 50, 200), (0, 0, 0), (1480, 100, 0)],
    ids=["single", "staggered", "all_empty", "reset_in_batch"],
)
def test_batched_streaming_matches_each_session_alone(
    history_lengths: tuple[int, ...],
) -> None:
    """Each session in a batch gets its own result and owns exactly its own history."""
    encoder = tiny_audio_encoder()
    generator = torch.Generator().manual_seed(1)
    chunks = [
        StreamingAudioChunk(
            audio_features=torch.randn(
                1, 80, STREAMING_MEL_FRAMES, generator=generator
            ),
            state=random_history(length, generator),
            prefix_extra_frames=2,
            suffix_extra_frames=2,
        )
        for length in history_lengths
    ]
    batch_chunks = [dataclasses.replace(chunk) for chunk in chunks]
    batched = encoder.forward_streaming_batch(batch_chunks)
    assert all(chunk.state is None for chunk in batch_chunks)
    for chunk, (embeds, state) in zip(chunks, batched, strict=True):
        [(alone_embeds, alone_state)] = encoder.forward_streaming_batch([chunk])
        torch.testing.assert_close(embeds, alone_embeds, rtol=1e-5, atol=1e-5)
        assert state.past_length == alone_state.past_length
        torch.testing.assert_close(
            state.key_value_states,
            alone_state.key_value_states,
            rtol=1e-5,
            atol=1e-5,
        )
        assert (
            state.key_value_states.untyped_storage().nbytes()
            == state.key_value_states.nbytes
        )


def audio_payload(
    request_id: str,
    audio_features: torch.Tensor,
    audio_feature_lens: torch.Tensor,
    *,
    cache_key: str | None,
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


@pytest.mark.parametrize("short_width,request_count", [(139, 1), (139, 2), (140, 2)])
def test_cross_request_batch_matches_serial_outputs(
    short_width: int, request_count: int
) -> None:
    encoder = tiny_audio_encoder()
    short_features = torch.randn(1, 80, short_width)
    long_features = torch.randn(3, 80, 301)
    short_lengths = torch.tensor([short_width])
    long_lengths = torch.tensor([301, 201, 1])
    expected = [
        encoder(audio_features=features, audio_feature_lens=lengths)["audio_embeds"]
        for features, lengths in (
            (short_features, short_lengths),
            (long_features, long_lengths),
        )
    ]
    payloads = [
        audio_payload("short", short_features, short_lengths, cache_key="short"),
        audio_payload("long", long_features, long_lengths, cache_key="long"),
    ]

    actual = batch_audio_encoder_payloads(
        payloads[:request_count],
        encoder=encoder,
        cache=StageOutputCache(cache_device="cpu"),
    )

    assert [payload.request_id for payload in actual] == ["short", "long"][
        :request_count
    ]
    for actual_payload, expected_embeddings in zip(
        actual, expected[:request_count], strict=True
    ):
        actual_embeddings = actual_payload.data["encoder_outs"]["audio_encoder"][
            "audio_embeds"
        ]
        torch.testing.assert_close(
            actual_embeddings, expected_embeddings, rtol=1e-4, atol=1e-4
        )


def test_cross_request_batching_is_enabled_by_default() -> None:
    stage = audio_encoder_stage(gpu=0, process="pipeline")
    assert stage.factory.max_batch_size == 8
    assert stage.factory.max_batch_wait_ms == 0


def test_cross_request_batch_reuses_cache_and_preserves_request_order() -> None:
    encoder = tiny_audio_encoder()
    cache = StageOutputCache(max_size=1, cache_device="cpu")
    features = torch.randn(1, 80, 137)
    lengths = torch.tensor([137])
    expected = encoder(audio_features=features, audio_feature_lens=lengths)
    other_features = -features
    other_expected = encoder(audio_features=other_features, audio_feature_lens=lengths)
    cache.put("cached", expected)
    payloads = [
        audio_payload("leader", features, lengths, cache_key="shared"),
        audio_payload("cached", features, lengths, cache_key="cached"),
        StagePayload(request_id="skipped", request=OmniRequest(inputs={}), data={}),
        audio_payload("uncached-1", features, lengths, cache_key=None),
        audio_payload("evicts-leader", features, lengths, cache_key="last"),
        audio_payload("follower", features, lengths, cache_key="shared"),
        audio_payload("uncached-2", other_features, lengths, cache_key=None),
    ]

    with patch.object(encoder, "forward", wraps=encoder.forward) as forward:
        outputs = batch_audio_encoder_payloads(payloads, encoder=encoder, cache=cache)
        assert forward.call_count == 1
        assert forward.call_args.kwargs["audio_features"].shape[0] == 4

    assert [payload.request_id for payload in outputs] == [
        "leader",
        "cached",
        "skipped",
        "uncached-1",
        "evicts-leader",
        "follower",
        "uncached-2",
    ]
    for payload in outputs:
        result = payload.data["encoder_outs"]["audio_encoder"]
        if payload.request_id == "skipped":
            assert result == {}
        else:
            expected_output = (
                other_expected if payload.request_id == "uncached-2" else expected
            )
            torch.testing.assert_close(
                result["audio_embeds"], expected_output["audio_embeds"]
            )

    with patch.object(encoder, "forward", wraps=encoder.forward) as forward:
        cached = batch_audio_encoder_payloads(
            [audio_payload("cached-again", features, lengths, cache_key="last")],
            encoder=encoder,
            cache=cache,
        )
        forward.assert_not_called()
    torch.testing.assert_close(
        cached[0].data["encoder_outs"]["audio_encoder"]["audio_embeds"],
        expected["audio_embeds"],
    )
