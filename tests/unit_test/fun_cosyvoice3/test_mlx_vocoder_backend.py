# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import threading

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten  # noqa: E402

from sglang_omni.models.fun_cosyvoice3.mlx.vocoder import (  # noqa: E402
    FunCosyVoice3MlxVocoder,
)
from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.config import (  # noqa: E402
    FlowConfig,
    HiFTConfig,
)
from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.dit import (  # noqa: E402
    Attention,
    _layer_norm,
)
from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.flow import (  # noqa: E402
    CausalMaskedDiffWithDiT,
)
from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.flow_matching import (  # noqa: E402
    CausalConditionalCFM,
)
from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.hift import (  # noqa: E402
    CausalHiFTGenerator,
)
from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.loader import (  # noqa: E402
    _map_flow_weight,
    _map_hift_weight,
)


def _write_tiny_artifact(tmp_path, *, hift_prefix="hifigan"):
    flow_config = FlowConfig(
        input_size=4,
        output_size=4,
        spk_embed_dim=3,
        vocab_size=16,
        pre_lookahead_len=1,
        pre_lookahead_channels=16,
        token_mel_ratio=2,
        dit_hidden_size=16,
        dit_depth=1,
        dit_num_heads=2,
        dit_head_dim=8,
        dit_mlp_ratio=2.0,
        dit_mel_dim=4,
        dit_mu_dim=4,
        dit_spk_dim=4,
        dit_static_chunk_size=4,
        n_timesteps=1,
    )
    hift_config = HiFTConfig(
        in_channels=4,
        base_channels=8,
        nb_harmonics=1,
        sampling_rate=24000,
        upsample_rates=[2],
        upsample_kernel_sizes=[4],
        istft_params={"n_fft": 4, "hop_len": 2},
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1]],
        source_resblock_kernel_sizes=[3],
        source_resblock_dilation_sizes=[[1]],
        conv_pre_look_right=1,
    )
    flow = CausalMaskedDiffWithDiT(flow_config)
    hift = CausalHiFTGenerator(hift_config)
    weights = {
        **{f"flow.{name}": value for name, value in tree_flatten(flow.parameters())},
        **{
            f"{hift_prefix}.{name}": value
            for name, value in tree_flatten(hift.parameters())
        },
    }
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "cosyvoice3",
                "flow": {
                    "input_size": 4,
                    "output_size": 4,
                    "spk_embed_dim": 3,
                    "vocab_size": 16,
                    "pre_lookahead_len": 1,
                    "pre_lookahead_channels": 16,
                    "token_mel_ratio": 2,
                    "n_timesteps": 1,
                    "dit": {
                        "dim": 16,
                        "depth": 1,
                        "heads": 2,
                        "dim_head": 8,
                        "ff_mult": 2.0,
                        "mel_dim": 4,
                        "mu_dim": 4,
                        "spk_dim": 4,
                        "static_chunk_size": 4,
                    },
                },
                "hifigan": {
                    "in_channels": 4,
                    "base_channels": 8,
                    "nb_harmonics": 1,
                    "sampling_rate": 24000,
                    "upsample_rates": [2],
                    "upsample_kernel_sizes": [4],
                    "istft_n_fft": 4,
                    "istft_hop_len": 2,
                    "resblock_kernel_sizes": [3],
                    "resblock_dilation_sizes": [[1]],
                    "source_resblock_kernel_sizes": [3],
                    "source_resblock_dilation_sizes": [[1]],
                    "conv_pre_look_right": 1,
                },
            }
        ),
        encoding="utf-8",
    )


def test_plus_artifact_weight_key_mapping():
    assert (
        _map_flow_weight("flow.decoder.estimator.transformer_blocks.0.ff.ff_0_0.weight")
        == "decoder.estimator.transformer_blocks.0.ff.ff.0.weight"
    )
    assert (
        _map_hift_weight("hifigan.f0_predictor.condnet_4.conv.weight")
        == "f0_predictor.condnet.2.weight"
    )
    assert (
        _map_hift_weight("hifigan.resblocks.0.convs1.0.conv.weight")
        == "resblocks.0.convs1.0.weight"
    )
    assert (
        _map_hift_weight("hift.resblocks.0.convs1.0.weight")
        == "resblocks.0.convs1.0.weight"
    )


def test_fused_attention_matches_explicit_reference():
    mx.random.seed(7)
    attention = Attention(dim=8, heads=2, dim_head=4)
    inputs = mx.random.normal((2, 5, 8))
    mask = mx.tril(mx.ones((2, 5, 5), dtype=mx.bool_))

    actual = attention(inputs, mask=mask)
    query = attention.to_q(inputs).reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
    key = attention.to_k(inputs).reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
    value = attention.to_v(inputs).reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
    scores = query @ key.transpose(0, 1, 3, 2) / math.sqrt(4)
    scores = mx.where(mask[:, None], scores, -float("inf"))
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
    expected = weights @ value
    expected = expected.transpose(0, 2, 1, 3).reshape(2, 5, 8)
    expected = attention.to_out[0](expected)

    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()


def test_fast_layer_norm_matches_reference_formula():
    mx.random.seed(11)
    inputs = mx.random.normal((2, 5, 16)).astype(mx.float16)

    actual = _layer_norm(inputs)
    inputs_float = inputs.astype(mx.float32)
    mean = mx.mean(inputs_float, axis=-1, keepdims=True)
    variance = mx.var(inputs_float, axis=-1, keepdims=True)
    expected = ((inputs_float - mean) * mx.rsqrt(variance + 1e-6)).astype(inputs.dtype)

    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=1e-3, atol=1e-3).item()


def test_flow_noise_is_cast_to_model_dtype():
    class ZeroEstimator:
        out_channels = 4

        def __call__(self, x, *args):
            return mx.zeros_like(x)

    flow_matching = CausalConditionalCFM(ZeroEstimator())
    mu = mx.zeros((1, 4, 6), dtype=mx.float16)
    output = flow_matching(
        mu=mu,
        mask=mx.ones((1, 1, 6), dtype=mx.float16),
        spks=mx.zeros((1, 4), dtype=mx.float16),
        cond=mx.zeros_like(mu),
        n_timesteps=1,
    )

    mx.eval(output)
    assert flow_matching._rand_noise.dtype == mx.float32
    assert output.dtype == mx.float16


def test_load_and_decode_tiny_converted_artifact(tmp_path):
    _write_tiny_artifact(tmp_path)
    vocoder = FunCosyVoice3MlxVocoder.from_pretrained(str(tmp_path))

    waveform = vocoder.decode(
        token=[1, 2, 3],
        prompt_token=[4, 5],
        prompt_feat=np.zeros((4, 4), dtype=np.float32),
        embedding=np.ones(3, dtype=np.float32),
    )

    assert waveform.ndim == 1
    assert waveform.size > 0
    assert waveform.dtype == np.float32
    assert waveform.flags.c_contiguous
    assert np.isfinite(waveform).all()
    assert vocoder.sample_rate == 24000
    assert vocoder.token_mel_ratio == 2


def test_loaded_vocoder_decodes_on_scheduler_thread(tmp_path):
    _write_tiny_artifact(tmp_path)
    vocoder = FunCosyVoice3MlxVocoder.from_pretrained(str(tmp_path))
    stream = mx.new_thread_local_stream(mx.gpu)
    results = []
    errors = []

    def decode():
        try:
            with mx.stream(stream):
                results.append(
                    vocoder.decode(
                        token=[1, 2, 3],
                        prompt_token=[4, 5],
                        prompt_feat=np.zeros((4, 4), dtype=np.float32),
                        embedding=np.ones(3, dtype=np.float32),
                    )
                )
        except Exception as exc:  # pragma: no cover - asserted in parent thread
            errors.append(exc)

    thread = threading.Thread(target=decode, name="scheduler-vocoder-test")
    thread.start()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert errors == []
    assert len(results) == 1
    assert results[0].shape[0] > 0
    assert np.isfinite(results[0]).all()


def test_loader_accepts_canonical_hift_prefix(tmp_path):
    _write_tiny_artifact(tmp_path, hift_prefix="hift")

    vocoder = FunCosyVoice3MlxVocoder.from_pretrained(str(tmp_path))

    assert vocoder.sample_rate == 24000


def test_loader_validates_explicit_dtype_against_artifact(tmp_path):
    _write_tiny_artifact(tmp_path)

    FunCosyVoice3MlxVocoder.from_pretrained(
        str(tmp_path),
        expected_dtype="float16",
    )
    with pytest.raises(ValueError, match="owned by the converted artifact"):
        FunCosyVoice3MlxVocoder.from_pretrained(
            str(tmp_path),
            expected_dtype="bfloat16",
        )


def test_decode_validates_prompt_alignment(tmp_path):
    _write_tiny_artifact(tmp_path)
    vocoder = FunCosyVoice3MlxVocoder.from_pretrained(str(tmp_path))

    with pytest.raises(ValueError, match="token_mel_ratio"):
        vocoder.decode_mx(
            token=[1],
            prompt_token=[2, 3],
            prompt_feat=np.zeros((3, 4), dtype=np.float32),
            embedding=np.ones(3, dtype=np.float32),
        )


def test_loader_rejects_unconverted_checkpoint(tmp_path):
    (tmp_path / "flow.pt").touch()
    (tmp_path / "hift.pt").touch()

    with pytest.raises(FileNotFoundError, match="converted artifact"):
        FunCosyVoice3MlxVocoder.from_pretrained(str(tmp_path))


def test_loader_rejects_raw_unsanitized_hift_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "flow.stub": mx.zeros((1,)),
            "hift.conv_pre.weight_g": mx.zeros((1,)),
        },
    )

    with pytest.raises(ValueError, match="raw unsanitized"):
        FunCosyVoice3MlxVocoder.from_pretrained(str(tmp_path))
