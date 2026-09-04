# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.nn import QuantizedLinear  # noqa: E402
from mlx_lm.models.qwen2 import ModelArgs, Qwen2Model  # noqa: E402
from sglang.srt.hardware_backend.mlx.sampling import MlxSamplingParams  # noqa: E402

from sglang_omni.models.fun_cosyvoice3.mlx.model import (  # noqa: E402
    SPEECH_TOKEN_SIZE,
    TOTAL_VOCAB_SIZE,
    CosyVoice3MlxModel,
    _quantize_loaded_backbone,
)
from sglang_omni.models.fun_cosyvoice3.mlx.runner import (  # noqa: E402
    FunCosyVoice3MlxModelRunner,
)


def _tiny_model(
    *,
    hidden_size: int = 8,
    intermediate_size: int = 16,
) -> CosyVoice3MlxModel:
    mx.random.seed(0)
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=32,
        max_position_embeddings=128,
        rope_theta=10_000.0,
        tie_word_embeddings=True,
    )
    backbone = Qwen2Model(args)
    speech_embedding = mx.arange(TOTAL_VOCAB_SIZE * args.hidden_size).reshape(
        TOTAL_VOCAB_SIZE, args.hidden_size
    )
    speech_embedding = speech_embedding.astype(mx.float32)
    llm_decoder = mx.zeros((TOTAL_VOCAB_SIZE, args.hidden_size), dtype=mx.float32)
    return CosyVoice3MlxModel(backbone, speech_embedding, llm_decoder)


def test_native_mlx_builds_prompt_and_projects_only_last_logits() -> None:
    model = _tiny_model()

    embeddings = model.build_prompt_embeddings([1, 2], [7, 8])
    logits = model.forward_embeddings(embeddings)

    mx.eval(embeddings, logits)
    assert embeddings.shape == (1, 6, 8)
    assert mx.array_equal(
        embeddings[0, 0], model.speech_embedding.weight[SPEECH_TOKEN_SIZE]
    ).item()
    assert mx.array_equal(
        embeddings[0, 1:3], model.model.embed_tokens(mx.array([[1, 2]]))[0]
    ).item()
    assert mx.array_equal(
        embeddings[0, 3], model.speech_embedding.weight[SPEECH_TOKEN_SIZE + 2]
    ).item()
    assert mx.array_equal(
        embeddings[0, 4:6], model.speech_embedding(mx.array([[7, 8]]))[0]
    ).item()
    assert logits.shape == (1, 1, TOTAL_VOCAB_SIZE)


def test_native_mlx_quantizes_only_the_qwen_layers() -> None:
    model = _tiny_model(hidden_size=64, intermediate_size=128)

    _quantize_loaded_backbone(model.model, "mlx_q4")

    assert isinstance(model.model.layers[0].self_attn.q_proj, QuantizedLinear)
    assert not isinstance(model.speech_embedding, QuantizedLinear)
    assert not isinstance(model.llm_decoder, QuantizedLinear)


def test_native_mlx_rejects_unknown_quantization() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        _quantize_loaded_backbone(_tiny_model().model, "unknown")


def test_runner_masks_controls_and_penalizes_each_repeated_id_once() -> None:
    runner = object.__new__(FunCosyVoice3MlxModelRunner)
    runner._cosyvoice3_prompt_lengths = {"req": 2}
    runner._cosyvoice3_min_lengths = {"req": 4}
    runner._cosyvoice3_repetition_penalties = {"req": 2.0}
    runner._cosyvoice3_recent_tokens = {"req": []}
    speech_ids = mx.arange(SPEECH_TOKEN_SIZE, dtype=mx.int32)
    runner._cosyvoice3_seen_masks = {"req": (speech_ids == 5) | (speech_ids == 6)}
    runner._first_attention_cache = lambda cache: SimpleNamespace(offset=5)
    # Token 5 appears twice, but repetition penalty must be applied once per
    # distinct speech id. Token 6 verifies the negative-logit branch.
    runner._req_token_ids = {"req": [0, 0, 5, 5, 6]}

    raw_logits = np.zeros((1, TOTAL_VOCAB_SIZE), dtype=np.float32)
    raw_logits[0, 5] = 4.0
    raw_logits[0, 6] = -4.0
    raw_logits[0, SPEECH_TOKEN_SIZE] = 100.0
    constrained = runner._constrain_logits(
        mx.array(raw_logits),
        ["req"],
        [[]],
    )

    mx.eval(constrained)
    assert constrained[0, 5].item() == pytest.approx(2.0)
    assert constrained[0, 6].item() == pytest.approx(-8.0)
    assert mx.all(constrained[0, SPEECH_TOKEN_SIZE:] == -float("inf")).item()


def test_runner_chained_constraint_includes_the_lazy_predecessor() -> None:
    runner = object.__new__(FunCosyVoice3MlxModelRunner)
    runner._cosyvoice3_prompt_lengths = {"req": 2}
    runner._cosyvoice3_min_lengths = {"req": 0}
    runner._cosyvoice3_repetition_penalties = {"req": 2.0}
    runner._cosyvoice3_recent_tokens = {"req": []}
    speech_ids = mx.arange(SPEECH_TOKEN_SIZE, dtype=mx.int32)
    runner._cosyvoice3_seen_masks = {"req": speech_ids == 5}
    runner._first_attention_cache = lambda cache: SimpleNamespace(offset=4)
    raw_logits = np.zeros((1, TOTAL_VOCAB_SIZE), dtype=np.float32)
    raw_logits[0, 5] = 4.0
    raw_logits[0, 6] = 6.0

    constrained = runner._constrain_logits(
        mx.array(raw_logits),
        ["req"],
        [[]],
        pending_tokens=mx.array([6], dtype=mx.int32),
    )

    mx.eval(constrained)
    assert constrained[0, 5].item() == pytest.approx(2.0)
    assert constrained[0, 6].item() == pytest.approx(3.0)


def test_runner_tracks_recent_history_for_ras() -> None:
    runner = object.__new__(FunCosyVoice3MlxModelRunner)
    runner._cosyvoice3_recent_tokens = {"req": [29, 28, 29]}
    mask = runner._recent_token_masks(["req"], None)
    mx.eval(mask)
    assert mask.shape == (1, SPEECH_TOKEN_SIZE)
    assert bool(mask[0, 29].item())
    assert bool(mask[0, 28].item())
    assert not bool(mask[0, 100].item())


def test_runner_ras_redraws_a_repeated_primary_token() -> None:
    runner = object.__new__(FunCosyVoice3MlxModelRunner)
    runner._enable_sampling = True
    runner._req_sampling = {
        "req": MlxSamplingParams(
            temperature=1.0, top_k=1, top_p=1.0, min_p=0.0, seed=None
        )
    }
    runner._rng_key = mx.random.key(0)
    runner._cosyvoice3_recent_tokens = {"req": [5]}
    runner._cosyvoice3_sampling_pending_tokens = None
    runner._first_attention_cache = lambda cache: SimpleNamespace(offset=3)
    runner._edited_logits = lambda logits, edit_rows: logits

    logits = mx.zeros((1, TOTAL_VOCAB_SIZE), dtype=mx.float32)
    logits = logits.at[0, 5].add(10.0)
    logits = logits.at[0, 6].add(9.0)
    tokens, _ = runner._select_tokens_with_logprobs(logits, ["req"], [[]])

    mx.eval(tokens)
    assert int(tokens[0].item()) == 6


def test_runner_ras_keeps_a_non_repeated_primary_token() -> None:
    runner = object.__new__(FunCosyVoice3MlxModelRunner)
    runner._enable_sampling = True
    runner._req_sampling = {
        "req": MlxSamplingParams(
            temperature=1.0, top_k=1, top_p=1.0, min_p=0.0, seed=None
        )
    }
    runner._rng_key = mx.random.key(0)
    runner._cosyvoice3_recent_tokens = {"req": [5]}
    runner._cosyvoice3_sampling_pending_tokens = None
    runner._first_attention_cache = lambda cache: SimpleNamespace(offset=3)
    runner._edited_logits = lambda logits, edit_rows: logits

    logits = mx.zeros((1, TOTAL_VOCAB_SIZE), dtype=mx.float32)
    logits = logits.at[0, 6].add(10.0)
    logits = logits.at[0, 5].add(9.0)
    tokens, _ = runner._select_tokens_with_logprobs(logits, ["req"], [[]])

    mx.eval(tokens)
    assert int(tokens[0].item()) == 6


@pytest.mark.parametrize(
    ("sampling_seed", "deterministic", "expected"),
    [
        (7, False, 7),
        (None, True, 42),
        (None, False, None),
    ],
)
def test_runner_resolves_omni_sampling_seed(
    sampling_seed: int | None,
    deterministic: bool,
    expected: int | None,
) -> None:
    runner = object.__new__(FunCosyVoice3MlxModelRunner)
    runner._deterministic_seeding = deterministic
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            temperature=0.7,
            top_k=20,
            top_p=0.8,
            min_p=0.0,
            sampling_seed=sampling_seed,
        )
    )

    params = runner._sampling_params_for_request(req)

    assert params.seed == expected


def test_runner_remove_and_clear_drop_cosyvoice_metadata() -> None:
    class _BaseRunner:
        def __init__(self) -> None:
            self.removed: list[str] = []
            self.base_cleared = False

        def remove_request(self, req_id: str) -> None:
            self.removed.append(req_id)

        def clear(self) -> None:
            self.base_cleared = True

    class _Runner(FunCosyVoice3MlxModelRunner, _BaseRunner):
        pass

    runner = _Runner()
    for metadata in (
        runner._cosyvoice3_prompt_lengths,
        runner._cosyvoice3_min_lengths,
        runner._cosyvoice3_repetition_penalties,
    ):
        metadata.update(remove=1, keep=2)
    runner._cosyvoice3_seen_masks.update(
        remove=mx.zeros((SPEECH_TOKEN_SIZE,), dtype=mx.bool_),
        keep=mx.zeros((SPEECH_TOKEN_SIZE,), dtype=mx.bool_),
    )

    runner.remove_request("remove")

    assert runner.removed == ["remove"]
    for metadata in (
        runner._cosyvoice3_prompt_lengths,
        runner._cosyvoice3_min_lengths,
        runner._cosyvoice3_repetition_penalties,
    ):
        assert metadata == {"keep": 2}
    assert set(runner._cosyvoice3_seen_masks) == {"keep"}

    runner.clear()

    assert runner.base_cleared is True
    assert runner._cosyvoice3_prompt_lengths == {}
    assert runner._cosyvoice3_min_lengths == {}
    assert runner._cosyvoice3_repetition_penalties == {}
    assert runner._cosyvoice3_seen_masks == {}
