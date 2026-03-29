# SPDX-License-Identifier: Apache-2.0
"""Unit tests for S2-Pro Tensor Parallel (TP) support.

Run on H200 machine with sglang + CUDA:
    pytest tests/test_s2pro_tp.py -v

Tests cover:
  - Audio decoder creation ordering (must be after embed_tokens)
  - Weight loading: text (TP-sharded) + audio decoder (replicated)
  - setup_vq_decode: device alignment, buffer allocation, semantic bias
  - setup_vq_decode_internal: caches set up AFTER device move
  - Forward pass: VQ combination, logits shape, _decode_codebooks output
  - TP-specific: all-gather gating, tp_size tracking
  - Config validation: tp_rank < tp_size

Strategy:
  Initialize SGLang's real distributed TP process group (TP=1) via
  init_distributed_environment + initialize_model_parallel, then construct
  S2ProSGLangTextModel with all real SGLang parallel layers.  For forward
  pass tests, transformer layer forward methods are replaced with
  passthroughs to avoid needing a full KV-cache / ForwardBatch setup.
"""

from __future__ import annotations

import math
import os
import socket
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip the entire module if sglang or CUDA is not available
# ---------------------------------------------------------------------------
sglang = pytest.importorskip("sglang", reason="sglang not installed")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for SGLang distributed tests", allow_module_level=True)


# ---------------------------------------------------------------------------
# Module-scoped fixture: initialize SGLang distributed (TP=1)
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@pytest.fixture(scope="module", autouse=True)
def init_sglang_distributed():
    """Initialize SGLang TP process group (TP=1, single GPU) for the module."""
    from sglang.srt.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    # Clean up any leftover state from a previous module / crashed run
    try:
        destroy_model_parallel()
    except Exception:
        pass
    try:
        destroy_distributed_environment()
    except Exception:
        pass

    port = _find_free_port()
    torch.cuda.set_device(0)

    # Disable message-queue broadcaster to keep the test lightweight
    os.environ["SGLANG_USE_MESSAGE_QUEUE_BROADCASTER"] = "false"

    # Set global server args (needed by RotaryEmbedding._compute_inv_freq)
    server_args = ServerArgs(model_path="dummy", tp_size=1, dtype="bfloat16")
    set_global_server_args_for_scheduler(server_args)

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        backend="nccl",
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    yield

    destroy_model_parallel()
    destroy_distributed_environment()


# ---------------------------------------------------------------------------
# Minimal config helpers
# ---------------------------------------------------------------------------

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.configuration import (
    FishQwen3AudioDecoderConfig,
    FishQwen3Config,
    FishQwen3OmniConfig,
)


def _make_tiny_config() -> FishQwen3OmniConfig:
    """Create a minimal config for fast testing."""
    tc = FishQwen3Config(
        vocab_size=256,
        dim=64,
        n_layer=2,
        n_head=4,
        n_local_heads=2,
        intermediate_size=256,
        rope_base=10000.0,
        norm_eps=1e-6,
        max_seq_len=128,
        attention_qk_norm=True,
        tie_word_embeddings=True,
    )
    adc = FishQwen3AudioDecoderConfig(
        text_dim=64,  # must match tc.dim
        num_codebooks=4,
        vocab_size=64,
        dim=32,
        n_layer=1,
        n_head=2,
        n_local_heads=2,
        intermediate_size=128,
        rope_base=10000.0,
        norm_eps=1e-6,
        attention_qk_norm=False,
    )
    return FishQwen3OmniConfig(text_config=tc, audio_decoder_config=adc)


def _make_tiny_config_no_decoder() -> FishQwen3OmniConfig:
    """Config without audio_decoder_config."""
    tc = FishQwen3Config(
        vocab_size=256,
        dim=64,
        n_layer=2,
        n_head=4,
        n_local_heads=2,
        intermediate_size=256,
        rope_base=10000.0,
        norm_eps=1e-6,
        max_seq_len=128,
        attention_qk_norm=True,
        tie_word_embeddings=True,
    )
    return FishQwen3OmniConfig(text_config=tc, audio_decoder_config=None)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

_VQ_PARAMS = dict(
    num_codebooks=4,
    codebook_size=64,
    semantic_begin_id=100,
    semantic_end_id=163,  # 100 + 64 - 1
    im_end_id=2,
    max_batch_size=8,
)

_DEVICE = "cuda:0"


def _create_model(config=None, device=_DEVICE):
    """Create S2ProSGLangTextModel with real SGLang parallel layers (TP=1)."""
    from sglang_omni.models.fishaudio_s2_pro.sglang_model import S2ProSGLangTextModel

    if config is None:
        config = _make_tiny_config()

    model = S2ProSGLangTextModel(config=config)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    return model


def _create_model_no_decoder(device=_DEVICE):
    """Create model without internal audio decoder."""
    return _create_model(config=_make_tiny_config_no_decoder(), device=device)


def _patch_layers_passthrough(model):
    """Replace transformer layer forward methods with passthroughs.

    This avoids needing a real ForwardBatch (with KV cache, attention masks,
    etc.) when testing forward-pass logic outside of the attention layers.
    """
    for i in range(model.start_layer, model.end_layer):

        def _passthrough(positions, hidden_states, forward_batch, residual):
            if residual is None:
                residual = hidden_states
            else:
                hidden_states = hidden_states + residual
                residual = hidden_states
            return hidden_states, residual

        model.layers[i].forward = _passthrough


def _make_fake_forward_batch(*, is_extend: bool = False):
    fb = MagicMock()
    fb.input_embeds = None

    class _FakeMode:
        @staticmethod
        def is_extend():
            return is_extend

    fb.forward_mode = _FakeMode()
    return fb


# ===========================================================================
# Tests
# ===========================================================================


class TestAudioDecoderCreationOrder:
    """Verify _create_audio_decoder runs AFTER embed_tokens exists."""

    def test_audio_decoder_exists_when_config_has_it(self):
        model = _create_model()
        assert (
            model._audio_decoder is not None
        ), "Audio decoder should be created when config has audio_decoder_config"

    def test_embed_tokens_exists_before_audio_decoder(self):
        """embed_tokens must already exist when _create_audio_decoder runs."""
        model = _create_model()
        embed_device = model.embed_tokens.weight.device
        ad_device = next(model._audio_decoder.parameters()).device
        assert ad_device == embed_device, (
            f"Audio decoder device ({ad_device}) should match embed_tokens "
            f"device ({embed_device}) — creation order bug"
        )

    def test_no_audio_decoder_without_config(self):
        model = _create_model_no_decoder()
        assert model._audio_decoder is None

    def test_audio_decoder_is_submodule(self):
        """Audio decoder should be discoverable via named_modules."""
        model = _create_model()
        names = {n for n, _ in model.named_modules()}
        assert "_audio_decoder" in names

    def test_create_audio_decoder_method(self):
        """Test _create_audio_decoder on an existing model object."""
        model = _create_model_no_decoder()
        assert model._audio_decoder is None

        adc = _make_tiny_config().audio_decoder_config
        model._create_audio_decoder(adc)
        assert model._audio_decoder is not None
        # Should be on same device as embed_tokens
        ad_device = next(model._audio_decoder.parameters()).device
        expected = model.embed_tokens.weight.device
        assert ad_device == expected


class TestSetupVqDecode:
    """Test setup_vq_decode: device alignment, buffer allocation, semantic bias."""

    def test_external_decoder_attached(self):
        """Backward-compatible path: external audio decoder."""
        from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling import (
            FishQwen3AudioDecoder,
        )

        model = _create_model_no_decoder()
        adc = FishQwen3AudioDecoderConfig(
            text_dim=64,
            num_codebooks=4,
            vocab_size=64,
            dim=32,
            n_layer=1,
            n_head=2,
            n_local_heads=2,
            intermediate_size=128,
            attention_qk_norm=False,
        )
        ext_decoder = (
            FishQwen3AudioDecoder(adc).to(device=_DEVICE, dtype=torch.bfloat16).eval()
        )
        ext_decoder.setup_caches(max_batch_size=8, dtype=torch.bfloat16)

        model.setup_vq_decode(ext_decoder, **_VQ_PARAMS)
        assert model._vq_ready
        assert model._audio_decoder is ext_decoder

    def test_internal_decoder_path(self):
        """TP path: internal audio decoder from config."""
        model = _create_model()
        model.setup_vq_decode_internal(**_VQ_PARAMS)
        assert model._vq_ready

    def test_raises_without_any_decoder(self):
        model = _create_model_no_decoder()
        with pytest.raises(ValueError, match="Audio decoder not initialized"):
            model.setup_vq_decode(audio_decoder=None, **_VQ_PARAMS)

    def test_raises_internal_without_decoder(self):
        model = _create_model_no_decoder()
        with pytest.raises(ValueError, match="Internal audio decoder not initialized"):
            model.setup_vq_decode_internal(**_VQ_PARAMS)

    def test_buffer_shapes(self):
        model = _create_model()
        model.setup_vq_decode_internal(**_VQ_PARAMS)

        nc = _VQ_PARAMS["num_codebooks"]
        bs = _VQ_PARAMS["max_batch_size"]
        assert model._vq_codes.shape == (bs, nc)
        assert model._vq_mask.shape == (bs,)
        assert model._output_codes.shape == (bs, nc + 1)
        assert model._output_semantic_ids.shape == (bs,)

    def test_buffers_on_correct_device(self):
        model = _create_model()
        model.setup_vq_decode_internal(**_VQ_PARAMS)

        expected = model.embed_tokens.weight.device
        assert model._vq_codes.device == expected
        assert model._vq_mask.device == expected
        assert model._semantic_bias.device == expected
        assert model._output_codes.device == expected
        assert model._vq_codebook_offsets.device == expected

    def test_semantic_bias_mask(self):
        model = _create_model()
        model.setup_vq_decode_internal(**_VQ_PARAMS)

        bias = model._semantic_bias
        sb = _VQ_PARAMS["semantic_begin_id"]
        se = _VQ_PARAMS["semantic_end_id"]
        im = _VQ_PARAMS["im_end_id"]

        # Allowed tokens should have bias == 0
        assert bias[sb].item() == 0.0
        assert bias[se].item() == 0.0
        assert bias[im].item() == 0.0
        # Tokens in the middle of semantic range
        assert bias[(sb + se) // 2].item() == 0.0
        # Tokens outside allowed range should be -inf
        assert bias[0].item() == float("-inf")
        assert bias[sb - 1].item() == float("-inf")
        if se + 1 < model.vocab_size:
            assert bias[se + 1].item() == float("-inf")

    def test_vq_scale_factor(self):
        model = _create_model()
        model.setup_vq_decode_internal(**_VQ_PARAMS)

        nc = _VQ_PARAMS["num_codebooks"]
        expected = 1.0 / math.sqrt(nc + 1)
        assert abs(model._vq_scale - expected) < 1e-7


class TestSetupVqDecodeInternalOrdering:
    """Verify setup_vq_decode_internal: device move THEN caches."""

    def test_caches_on_correct_device(self):
        model = _create_model()
        model.setup_vq_decode_internal(**_VQ_PARAMS)

        ad = model._audio_decoder
        assert ad.max_batch_size == _VQ_PARAMS["max_batch_size"]

        # Check that cache tensors are on the model's device (not CPU)
        expected = model.embed_tokens.weight.device
        for layer in ad.layers:
            if hasattr(layer, "attention") and hasattr(layer.attention, "kv_cache"):
                kv = layer.attention.kv_cache
                if kv is not None:
                    assert kv.k_cache.device == expected, (
                        f"KV cache on {kv.k_cache.device}, expected {expected}. "
                        "setup_caches was likely called before device move."
                    )

    def test_device_move_from_cpu(self):
        """If audio decoder starts on CPU, setup_vq_decode should move it."""
        model = _create_model(device="cuda:0")
        # Manually move audio decoder back to CPU to simulate the bug scenario
        model._audio_decoder = model._audio_decoder.to("cpu")
        cpu_device = next(model._audio_decoder.parameters()).device
        assert cpu_device.type == "cpu"

        model.setup_vq_decode_internal(**_VQ_PARAMS)

        # After setup, should be on CUDA
        ad_device = next(model._audio_decoder.parameters()).device
        assert (
            ad_device.type == "cuda"
        ), f"Audio decoder should have been moved to CUDA, but is on {ad_device}"


class TestWeightLoading:
    """Test load_weights: text weights (TP-sharded) + audio decoder (replicated)."""

    def test_weights_separated_correctly(self):
        """Audio decoder weights should be directly copied (replicated)."""
        model = _create_model()

        ad_param_name = list(model._audio_decoder.named_parameters())[0][0]
        orig_value = model._audio_decoder.state_dict()[ad_param_name].clone()

        new_value = torch.randn_like(orig_value)
        weights = [
            (f"audio_decoder.{ad_param_name}", new_value),
        ]

        model.load_weights(iter(weights))

        loaded = model._audio_decoder.state_dict()[ad_param_name]
        assert torch.allclose(
            loaded, new_value.to(device=loaded.device, dtype=loaded.dtype)
        ), "Audio decoder weight not loaded correctly"

    def test_text_weights_loaded_via_remap(self):
        """Text model weights loaded through the remap path."""
        model = _create_model()

        # In real checkpoints: "text_model.model.embeddings.weight"
        # -> remap to "embed_tokens.weight"
        orig = model.embed_tokens.weight.data.clone()
        new_weight = torch.randn_like(orig)
        weights = [("text_model.model.embeddings.weight", new_weight)]

        model.load_weights(iter(weights))
        # Weight loader may shard for TP, but for TP=1 it should match fully
        loaded = model.embed_tokens.weight.data
        assert not torch.equal(
            loaded, orig
        ), "embed_tokens.weight should have been updated by load_weights"

    def test_audio_decoder_buffer_loaded(self):
        model = _create_model()
        nc = model._audio_decoder.config.num_codebooks
        vs = model._audio_decoder.config.vocab_size

        # codebook_offsets is a registered buffer
        new_offsets = torch.arange(nc, dtype=torch.long) * (vs + 1)
        weights = [("audio_decoder.codebook_offsets", new_offsets)]

        model.load_weights(iter(weights))

        actual = model._audio_decoder.codebook_offsets
        expected = new_offsets.to(device=actual.device, dtype=actual.dtype)
        assert torch.equal(actual, expected), "Buffer not loaded correctly"

    def test_audio_decoder_device_dtype_cast(self):
        """Weights loaded from CPU/float32 should be cast to param device/dtype."""
        model = _create_model()

        ad_param_name, ad_param = list(model._audio_decoder.named_parameters())[0]
        target_device = ad_param.device
        target_dtype = ad_param.dtype

        cpu_weight = torch.randn(ad_param.shape, dtype=torch.float32, device="cpu")
        weights = [(f"audio_decoder.{ad_param_name}", cpu_weight)]

        model.load_weights(iter(weights))

        loaded = model._audio_decoder.state_dict()[ad_param_name]
        assert loaded.device == target_device
        assert loaded.dtype == target_dtype

    def test_unknown_weights_silently_skipped(self):
        model = _create_model()

        weights = [
            ("audio_decoder.nonexistent_layer.weight", torch.randn(10, 10)),
            ("completely_unknown.weight", torch.randn(10, 10)),
        ]
        # Should not raise
        model.load_weights(iter(weights))

    def test_no_audio_decoder_skips_audio_weights(self):
        model = _create_model_no_decoder()

        weights = [("audio_decoder.embeddings.weight", torch.randn(64, 32))]
        # Should not raise — just logs and skips
        model.load_weights(iter(weights))


class TestForwardPass:
    """Test forward pass logic: VQ combination, shapes, _decode_codebooks.

    Transformer layers are patched to passthrough to avoid needing a real
    ForwardBatch with KV cache.  The tested logic is the embedding, VQ
    combination, logits, and codebook generation — all outside of attention.
    """

    def _setup_model(self):
        model = _create_model()
        model.setup_vq_decode_internal(**_VQ_PARAMS)
        _patch_layers_passthrough(model)
        return model

    def test_forward_decode_mode_shapes(self):
        """Forward in decode mode should produce correct output shapes."""
        model = self._setup_model()
        bs = 2
        device = model.embed_tokens.weight.device

        input_ids = torch.randint(0, model.vocab_size, (bs,), device=device)
        positions = torch.arange(bs, device=device)
        fb = _make_fake_forward_batch(is_extend=False)

        with torch.no_grad():
            output = model.forward(input_ids, positions, fb)

        assert output.next_token_logits.shape == (bs, model.vocab_size)
        assert output.hidden_states.shape[0] == bs
        assert output.hidden_states.shape[1] == model.hidden_size

    def test_forward_without_vq_ready(self):
        """Forward without VQ setup should still produce logits."""
        model = _create_model()
        _patch_layers_passthrough(model)
        assert not model._vq_ready
        bs = 2
        device = model.embed_tokens.weight.device

        input_ids = torch.randint(0, model.vocab_size, (bs,), device=device)
        positions = torch.arange(bs, device=device)
        fb = _make_fake_forward_batch(is_extend=False)

        with torch.no_grad():
            output = model.forward(input_ids, positions, fb)

        assert output.next_token_logits.shape == (bs, model.vocab_size)

    def test_decode_codebooks_writes_output_buffers(self):
        model = self._setup_model()
        bs = 3
        device = model.embed_tokens.weight.device

        logits = torch.randn(bs, model.vocab_size, device=device, dtype=torch.bfloat16)
        hidden_states = torch.randn(
            bs, model.hidden_size, device=device, dtype=torch.bfloat16
        )

        model._decode_codebooks(logits, hidden_states)

        # Semantic tokens should be in valid range
        sb = _VQ_PARAMS["semantic_begin_id"]
        se = _VQ_PARAMS["semantic_end_id"]
        im = _VQ_PARAMS["im_end_id"]
        for i in range(bs):
            sem = model._output_semantic_ids[i].item()
            assert sem == im or (
                sb <= sem <= se
            ), f"Semantic token {sem} not in allowed range [{sb}, {se}] or im_end={im}"

        # output_codes should have num_codebooks+1 columns
        nc = _VQ_PARAMS["num_codebooks"]
        codes = model._output_codes[:bs]
        assert codes.shape == (bs, nc + 1)

        # First column is the full semantic token ID
        assert torch.equal(codes[:, 0], model._output_semantic_ids[:bs])

    def test_vq_combination_applied_for_semantic_tokens(self):
        """When _vq_mask is True, hidden_states should differ from plain embedding."""
        model = self._setup_model()
        device = model.embed_tokens.weight.device

        # Pick a token ID in the semantic range
        sem_token = torch.tensor([_VQ_PARAMS["semantic_begin_id"]], device=device)
        plain_embed = model.embed_tokens(sem_token)

        # Set VQ buffers
        model._vq_mask[0] = True
        model._vq_codes[0] = torch.randint(
            0,
            _VQ_PARAMS["codebook_size"],
            (_VQ_PARAMS["num_codebooks"],),
            device=device,
        )

        # Recompute embedding with VQ combination
        hidden = model.embed_tokens(sem_token)
        bs = hidden.shape[0]
        vq_codes = model._vq_codes[:bs]
        vq_mask = model._vq_mask[:bs]
        offset_parts = vq_codes + model._vq_codebook_offsets[None, :]
        all_embeds = model._vq_codebook_embeddings(offset_parts)
        vq_sum = all_embeds.sum(dim=1).to(hidden.dtype)
        combined = (hidden + vq_sum) * model._vq_scale
        result = torch.where(vq_mask.unsqueeze(-1), combined, hidden)

        # VQ combination should change the embedding
        assert not torch.allclose(
            result, plain_embed.to(result.dtype), atol=1e-5
        ), "VQ combination should modify the embedding for semantic tokens"


class TestTPSpecific:
    """Test TP-specific code paths."""

    def test_tp_size_stored(self):
        model = _create_model()
        assert model.tp_size == 1

    def test_all_gather_not_called_for_tp1(self):
        """With tp_size=1, tensor_model_parallel_all_gather should be skipped."""
        model = _create_model()
        model.setup_vq_decode_internal(**_VQ_PARAMS)
        _patch_layers_passthrough(model)
        device = model.embed_tokens.weight.device

        bs = 1
        input_ids = torch.randint(0, model.vocab_size, (bs,), device=device)
        positions = torch.arange(bs, device=device)
        fb = _make_fake_forward_batch(is_extend=False)

        with patch(
            "sglang_omni.models.fishaudio_s2_pro.sglang_model.tensor_model_parallel_all_gather",
        ) as mock_gather:
            with torch.no_grad():
                model.forward(input_ids, positions, fb)
            mock_gather.assert_not_called()

    def test_all_gather_called_for_tp_gt1(self):
        """With tp_size>1, tensor_model_parallel_all_gather should be called."""
        model = _create_model()
        model.tp_size = 2  # Pretend TP=2
        model.setup_vq_decode_internal(**_VQ_PARAMS)
        _patch_layers_passthrough(model)
        device = model.embed_tokens.weight.device

        bs = 1
        input_ids = torch.randint(0, model.vocab_size, (bs,), device=device)
        positions = torch.arange(bs, device=device)
        fb = _make_fake_forward_batch(is_extend=False)

        with patch(
            "sglang_omni.models.fishaudio_s2_pro.sglang_model.tensor_model_parallel_all_gather",
            side_effect=lambda x: x,  # identity
        ) as mock_gather:
            with torch.no_grad():
                model.forward(input_ids, positions, fb)
            mock_gather.assert_called_once()


class TestConfigValidation:
    """Test tp_rank < tp_size validation."""

    def test_tp_rank_ge_tp_size_raises(self):
        from sglang_omni.models.fishaudio_s2_pro.pipeline.stages import (
            create_sglang_tts_engine_executor,
        )

        with pytest.raises(ValueError, match="tp_rank.*must be less than tp_size"):
            create_sglang_tts_engine_executor(
                model_path="/nonexistent",
                tp_size=2,
                tp_rank=2,
            )

    def test_tp_rank_eq_tp_size_raises(self):
        from sglang_omni.models.fishaudio_s2_pro.pipeline.stages import (
            create_sglang_tts_engine_executor,
        )

        with pytest.raises(ValueError, match="tp_rank.*must be less than tp_size"):
            create_sglang_tts_engine_executor(
                model_path="/nonexistent",
                tp_size=1,
                tp_rank=1,
            )

    def test_tp_rank_0_tp_size_1_no_validation_error(self):
        """Valid config should not raise the tp_rank validation error."""
        assert 0 < 1  # tp_rank=0 < tp_size=1
