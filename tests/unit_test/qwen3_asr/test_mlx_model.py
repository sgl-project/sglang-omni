# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.qwen3_asr.mlx.config import (  # noqa: E402
    AudioEncoderConfig,
    ModelConfig,
    TextConfig,
)
from sglang_omni.models.qwen3_asr.mlx.model import Qwen3ASRModel  # noqa: E402
from sglang_omni.models.qwen3_asr.mlx.runner import (  # noqa: E402
    Qwen3ASRMlxModelRunner,
    make_qwen3_asr_mlx_runner_class,
)


def _tiny_model(*, tie_word_embeddings: bool = True) -> Qwen3ASRModel:
    mx.random.seed(0)
    audio = AudioEncoderConfig(
        num_mel_bins=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        d_model=8,
        max_source_positions=100,
        n_window=5,
        n_window_infer=10,
        conv_chunksize=50,
        downsample_hidden_size=4,
        output_dim=8,
    )
    text = TextConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=128,
        tie_word_embeddings=tie_word_embeddings,
    )
    return Qwen3ASRModel(
        ModelConfig(audio_config=audio, text_config=text, audio_token_id=10)
    )


def test_native_mlx_audio_prefill_forward() -> None:
    model = _tiny_model()
    features = mx.zeros((1, 8, 20))
    mask = mx.ones((1, 20))
    audio_features = model.get_audio_features(features, mask)
    input_ids = mx.array([[1, *([10] * audio_features.shape[0]), 2]], dtype=mx.int32)
    embeddings = model._build_inputs_embeds(
        input_ids,
        audio_features,
        audio_start=1,
        num_audio_tokens=audio_features.shape[0],
    )
    logits = model(
        input_ids,
        input_embeddings=embeddings,
        cache=model.make_cache(),
    )

    mx.eval(logits)
    assert logits.shape == (1, input_ids.shape[1], 64)


def test_native_mlx_prefill_only_projects_last_position() -> None:
    model = _tiny_model()
    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    embeddings = model.model.embed_tokens(input_ids)

    full_logits = model(input_ids, input_embeddings=embeddings)
    last_logits = model._forward_last_logits(embeddings)

    mx.eval(full_logits, last_logits)
    assert last_logits.shape == (1, 1, 64)
    assert mx.allclose(
        last_logits,
        full_logits[:, -1:, :],
        rtol=1e-2,
        atol=1e-2,
    ).item()


def test_runner_restores_audio_placeholder_before_embedding() -> None:
    runner = object.__new__(Qwen3ASRMlxModelRunner)
    runner.model = _tiny_model()
    item = SimpleNamespace(
        feature=torch.zeros((1, 8, 20)),
        feature_attention_mask=torch.ones((1, 20)),
        model_specific_data={},
        pad_value=1_000_001,
    )
    req = SimpleNamespace(
        multimodal_inputs=SimpleNamespace(
            audio_token_id=10,
            mm_items=[item],
        )
    )

    input_ids, embeddings = runner._audio_prefill_inputs(
        req, [1, 1_000_001, 1_000_001, 1_000_001, 1_000_001, 2]
    )

    mx.eval(embeddings)
    assert input_ids.tolist() == [[1, 10, 10, 10, 10, 2]]
    assert embeddings.shape == (1, 6, 8)


def test_runner_only_rewrites_the_exact_audio_placeholder() -> None:
    item = SimpleNamespace(pad_value=1_000_001)
    req = SimpleNamespace(
        multimodal_inputs=SimpleNamespace(audio_token_id=10, mm_items=[item])
    )

    normalized = Qwen3ASRMlxModelRunner._normalize_audio_token_ids(
        req, [1, 1_000_001, -7, 2]
    )

    assert normalized == [1, 10, -7, 2]


def test_runner_converts_bfloat16_features_to_numpy() -> None:
    converted = Qwen3ASRMlxModelRunner._to_numpy(
        torch.ones((2, 3), dtype=torch.bfloat16)
    )

    assert converted.dtype.name == "float32"


def test_runner_rejects_missing_audio_item() -> None:
    req = SimpleNamespace(
        multimodal_inputs=SimpleNamespace(audio_token_id=10, mm_items=[])
    )

    with pytest.raises(ValueError, match="exactly one audio item"):
        Qwen3ASRMlxModelRunner._audio_item(req)


def test_runner_resolves_revision_and_checks_remote_code(monkeypatch, tmp_path) -> None:
    import mlx_lm.utils as mlx_lm_utils
    import sglang.srt.hardware_backend.mlx.remote_code_gate as remote_code_gate

    observed = {}
    monkeypatch.setattr(
        remote_code_gate,
        "resolve_model_directory",
        lambda model_path, revision=None: observed.update(
            model_path=model_path,
            revision=revision,
        )
        or tmp_path,
    )
    monkeypatch.setattr(
        remote_code_gate,
        "ensure_remote_code_allowed",
        lambda model_dir, trust_remote_code: observed.update(
            model_dir=model_dir,
            trust_remote_code=trust_remote_code,
        ),
    )
    monkeypatch.setattr(
        mlx_lm_utils,
        "load_model",
        lambda model_path, **kwargs: ("loaded-model", {}),
    )
    runner = object.__new__(Qwen3ASRMlxModelRunner)
    runner.model_path = "org/model"
    runner.revision = "revision-sha"
    runner.trust_remote_code = True

    runner._load_model()

    assert observed == {
        "model_path": "org/model",
        "revision": "revision-sha",
        "model_dir": tmp_path,
        "trust_remote_code": True,
    }
    assert runner.model == "loaded-model"


def test_native_mlx_rejects_audio_feature_count_mismatch() -> None:
    model = _tiny_model()
    input_ids = mx.array([[1, 10, 2]], dtype=mx.int32)
    audio_features = mx.zeros((2, 8))

    with pytest.raises(ValueError, match="counts differ"):
        model._build_inputs_embeds(
            input_ids,
            audio_features,
            audio_start=1,
            num_audio_tokens=1,
        )


def test_runner_chains_native_single_request_decode() -> None:
    runner_class = make_qwen3_asr_mlx_runner_class()
    runner = object.__new__(runner_class)
    runner.model = _tiny_model()
    runner._req_token_ids = {"req": [1]}
    runner._req_caches = {"req": runner.model.make_cache()}
    runner._decode_step_ct = 0
    runner._clear_steps = 0

    first = runner.decode_batch_start(["req"])
    second = runner.decode_batch_start_chained(first)
    mx.eval(second.lazy_tokens)
    runner.decode_batch_finalize(first)
    runner.decode_batch_finalize(second)

    assert first.lazy_tokens.shape == (1,)
    assert second.lazy_tokens.shape == (1,)
    assert runner._req_caches["req"][0].offset == 2
    assert len(runner._req_token_ids["req"]) == 3


def test_hf_weight_sanitize_is_local_and_transposes_conv2d() -> None:
    weights = {
        "thinker.audio_tower.conv2d1.weight": mx.zeros((4, 1, 3, 3)),
        "thinker.model.embed_tokens.weight": mx.zeros((64, 8)),
        "thinker.lm_head.weight": mx.zeros((64, 8)),
    }

    sanitized = _tiny_model().sanitize(weights)

    assert sanitized["audio_tower.conv2d1.weight"].shape == (4, 3, 3, 1)
    assert "model.embed_tokens.weight" in sanitized
    assert "lm_head.weight" not in sanitized


def test_hf_weight_sanitize_keeps_untied_lm_head() -> None:
    model = _tiny_model(tie_word_embeddings=False)

    sanitized = model.sanitize(
        {
            "thinker.model.embed_tokens.weight": mx.zeros((64, 8)),
            "thinker.lm_head.weight": mx.zeros((64, 8)),
        }
    )

    assert "lm_head.weight" in sanitized
