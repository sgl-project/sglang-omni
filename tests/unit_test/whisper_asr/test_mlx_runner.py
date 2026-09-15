# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.cache import ArraysCache, CacheList, KVCache  # noqa: E402

from sglang_omni.models.whisper_asr.mlx.config import ModelConfig  # noqa: E402
from sglang_omni.models.whisper_asr.mlx.model import WhisperMlxModel  # noqa: E402
from sglang_omni.models.whisper_asr.mlx.runner import (  # noqa: E402
    WhisperMlxModelRunner,
    make_whisper_mlx_runner_class,
)

DECODER_LAYERS = 2
ENCODER_TOKENS = 20
PAD_TOKEN_ID = 50257


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        d_model=64,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        num_mel_bins=8,
        max_source_positions=ENCODER_TOKENS,
        decoder_layers=DECODER_LAYERS,
        decoder_attention_heads=4,
        decoder_ffn_dim=128,
        max_target_positions=16,
        vocab_size=64,
        pad_token_id=PAD_TOKEN_ID,
    )


def _runner() -> WhisperMlxModelRunner:
    model = WhisperMlxModel(_tiny_config())
    runner_class = make_whisper_mlx_runner_class()

    class StubbedRunner(runner_class):
        def _load_model(self):
            self.model = model

    return StubbedRunner(model_path="unused", disable_radix_cache=True)


def _request(num_audio_tokens: int = ENCODER_TOKENS) -> SimpleNamespace:
    # Mirrors what the shared request builder attaches: the encoder token count
    # rides in model_specific_data, not as a field on the item.
    item = SimpleNamespace(
        feature=None,
        model_specific_data={"num_audio_tokens": num_audio_tokens},
    )
    return SimpleNamespace(
        multimodal_inputs=SimpleNamespace(
            mm_items=[item], num_image_tokens=num_audio_tokens
        ),
    )


def test_runner_constructs_despite_missing_rope() -> None:
    """SGLang's discovery rejects Whisper; the declared layout must replace it.

    Without it, MlxModelRunner.__init__ raises "MLX model has no supported
    attention layers" because no Whisper attention module exposes ``rope``.
    """
    runner = _runner()

    layout = runner._cache_layout
    assert layout.num_layers == DECODER_LAYERS
    assert layout.num_attention_layers == DECODER_LAYERS
    assert layout.auxiliary_layer_indices == ()


def test_declared_layout_does_not_leak_past_construction() -> None:
    """The discovery patch is scoped to __init__, not installed globally."""
    _runner()

    from sglang.srt.hardware_backend.mlx import model_runner as runner_module

    assert not hasattr(runner_module.find_attention_layers, "side_effect")


def test_cache_has_both_lifetimes_per_layer() -> None:
    """The base runner would install one pooled KV stream per layer."""
    runner = _runner()

    cache = runner._acquire_cache()

    assert len(cache) == DECODER_LAYERS
    for entry in cache:
        assert isinstance(entry, CacheList)
        assert isinstance(entry[0], KVCache)
        assert isinstance(entry[1], ArraysCache)


def test_decoder_prompt_drops_the_encoder_placeholders() -> None:
    """The shared request builder prefixes pad ids for the CUDA KV reservation.

    Decoding those placeholders would emit tokens from meaningless positions,
    since this path holds the encoder projection in its own cross cache.
    """
    runner = _runner()
    prompt = [50258, 50259, 50360]
    token_ids = [PAD_TOKEN_ID] * ENCODER_TOKENS + prompt

    assert runner._decoder_prompt_ids(_request(), token_ids) == prompt


def test_decoder_prompt_passes_through_when_already_stripped() -> None:
    """Whether the prefix reaches the runner depends on how the batch was built."""
    runner = _runner()
    prompt = [50258, 50259, 50360]

    assert runner._decoder_prompt_ids(_request(), prompt) == prompt


def test_decoder_prompt_rejects_an_empty_prompt() -> None:
    runner = _runner()

    with pytest.raises(ValueError, match="empty decoder prompt"):
        runner._decoder_prompt_ids(_request(), [])


def test_audio_item_requires_exactly_one_clip() -> None:
    runner = _runner()
    req = _request()
    req.multimodal_inputs.mm_items.append(object())

    with pytest.raises(ValueError, match="exactly one audio item"):
        runner._audio_item(req)


def test_chained_decode_drives_the_cross_attention_cache() -> None:
    """The chained step comes from AudioMlxModelRunner, not from this module.

    It has to work against Whisper's ``CacheList``: the shared implementation
    hands the cache to ``_decode_with_native_cache`` without reading
    ``.offset``, which is exactly the attribute a ``CacheList`` pair lacks. It
    also has to leave the cross-attention half untouched while the
    self-attention half grows, so the second token still attends to the audio.
    """
    from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode

    runner = _runner()
    request = _request()
    # The multimodal pipeline hands the runner a Torch tensor, which is what
    # AudioMlxModelRunner._to_numpy converts.
    import torch

    request.multimodal_inputs.mm_items[0].feature = torch.zeros(1, 8, 40)

    pending = runner.prefill_start(
        req_id="r0",
        new_token_ids=[50258, 50259, 50360],
        full_token_ids=[50258, 50259, 50360],
        prefix_slot_ids=[],
        new_slot_ids=[],
        req_pool_idx=0,
        req=request,
    )
    cache = pending.cache
    cross_keys = cache[0][1][0]
    self_offset_after_prefill = cache[0][0].offset

    chained = runner.decode_batch_start_chained(
        MlxPendingDecode(
            lazy_tokens=pending.lazy_token,
            req_ids=["r0"],
            caches=[cache],
            lazy_logprobs=None,
            logprob_spec=None,
            edit_rows=None,
        )
    )
    mx.eval(chained.lazy_tokens)

    assert chained.lazy_tokens.shape == (1,)
    assert chained.caches == [cache]
    # self-attention advanced by the one decoded token
    assert cache[0][0].offset == self_offset_after_prefill + 1
    # cross-attention is projected once and then fixed
    assert cache[0][1][0].shape == cross_keys.shape
    assert mx.array_equal(cache[0][1][0], cross_keys)
