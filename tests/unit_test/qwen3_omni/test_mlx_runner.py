# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MLX Qwen3-Omni thinker scheduler runner and dispatch."""

from __future__ import annotations

import json
import queue
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

# ``sglang._platform_stubs`` installs a mock ``triton`` module the first time
# any sglang module is imported, after which ``torch._inductor`` can no longer
import torch._inductor.runtime.triton_heuristics  # noqa: F401,E402  isort:skip

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx.nn")

import mlx.nn as nn  # noqa: E402  (import after the skip guard)
import torch  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from sglang_omni.model_runner import mlx_model_worker  # noqa: E402
from sglang_omni.model_runner.model_worker import ModelWorkerConfig  # noqa: E402
from sglang_omni.models.qwen3_omni.mlx.config import (  # noqa: E402
    MoeTextConfig,
    QuantizationConfig,
)
from sglang_omni.models.qwen3_omni.mlx.runner import (  # noqa: E402
    Qwen3OmniMlxSchedulerModelRunner,
    Qwen3OmniMlxTalkerModelRunner,
    build_qwen3_omni_talker_mlx_runner,
    build_qwen3_omni_thinker_mlx_runner,
    create_qwen3_omni_mlx_worker,
    load_qwen3_omni_mlx_talker,
    make_qwen3_omni_talker_mlx_runner_class,
    make_qwen3_omni_thinker_mlx_runner_class,
    read_qwen3_omni_component_weights,
)
from sglang_omni.models.qwen3_omni.mlx.talker import Qwen3OmniMlxTalker  # noqa: E402
from sglang_omni.models.qwen3_omni.mlx.talker_prefill import (  # noqa: E402
    Qwen3OmniMlxTalkerPrefillBuilder,
)
from sglang_omni.models.qwen3_omni.mlx.thinker import Qwen3OmniMlxThinker  # noqa: E402
from sglang_omni.models.qwen3_omni.pending_text_queue import (  # noqa: E402
    PendingTextTensorQueue,
)
from sglang_omni.models.qwen3_omni.request_builders import (  # noqa: E402
    make_talker_scheduler_adapters,
)
from sglang_omni.models.qwen3_omni.talker_model_runner import (  # noqa: E402
    QwenTalkerModelRunner,
)
from sglang_omni.proto.request import OmniRequest, StagePayload  # noqa: E402
from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor  # noqa: E402
from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest  # noqa: E402
from tests.unit_test.fixtures.qwen_fakes import FakeQwenTokenizer  # noqa: E402
from tests.utils.build_tiny_qwen3_omni_checkpoint import (  # noqa: E402
    OFFICIAL_SPEAKER_IDS,
    OFFICIAL_SPECIAL_TOKEN_IDS,
    build_tiny_config,
    build_tiny_qwen3_omni_checkpoint,
)

IMAGE_ID = OFFICIAL_SPECIAL_TOKEN_IDS["thinker_config.image_token_id"]
VIDEO_ID = OFFICIAL_SPECIAL_TOKEN_IDS["thinker_config.video_token_id"]
AUDIO_ID = OFFICIAL_SPECIAL_TOKEN_IDS["thinker_config.audio_token_id"]
PLACEHOLDER_IDS = {"image": IMAGE_ID, "video": VIDEO_ID, "audio": AUDIO_ID}

HIDDEN = 32
VOCAB = 512
# The official placeholder ids sit near 151k, far above the isolated unit
# model's vocabulary, so the in-memory fixtures use small ids of their own. The
UNIT_PLACEHOLDER_IDS = {"image": 300, "video": 301, "audio": 302}


def write_indexed_mlx_checkpoint(directory: Path, tensors: dict[str, mx.array]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    shard_name = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(directory / shard_name), tensors)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {key: shard_name for key in tensors},
            }
        ),
        encoding="utf-8",
    )
    return directory


# ---------------------------------------------------------------------------
# Small in-memory thinker for the isolated runner tests


def _unit_text_config(num_layers: int = 2) -> MoeTextConfig:
    return MoeTextConfig(
        hidden_size=HIDDEN,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=num_layers,
        intermediate_size=64,
        moe_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=(),
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        max_position_embeddings=4096,
        rope_theta=1000000.0,
        mrope_section=(2, 1, 1),
        tie_word_embeddings=False,
        shared_expert_intermediate_size=None,
    )


def _seeded_thinker(num_layers: int = 2) -> Qwen3OmniMlxThinker:
    model = Qwen3OmniMlxThinker(_unit_text_config(num_layers))
    rng = np.random.default_rng(7)
    updates = {}
    for name, value in tree_flatten(model.parameters()):
        shape = tuple(value.shape)
        if name.endswith("norm.weight") and len(shape) == 1:
            filled = 1.0 + 0.05 * rng.standard_normal(shape)
        else:
            filled = 0.05 * rng.standard_normal(shape)
        updates[name] = mx.array(filled.astype(np.float32))
    model.load_weights(list(updates.items()))
    model.eval()
    return model


def _unit_runner(*, capture_hidden_layers=(0, 1), num_layers: int = 2):
    return build_qwen3_omni_thinker_mlx_runner(
        model=_seeded_thinker(num_layers),
        placeholder_token_ids=dict(UNIT_PLACEHOLDER_IDS),
        capture_hidden_layers=capture_hidden_layers,
        accept_hidden_layer=(
            capture_hidden_layers[-1] if capture_hidden_layers else None
        ),
        disable_radix_cache=True,
        pool_size=4096,
    )


# ---------------------------------------------------------------------------
# Fake scheduler request


def _make_req(
    *,
    rid: str = "req-0",
    prompt_ids: list[int],
    omni_model_inputs: dict[str, Any] | None = None,
    mm_positions: dict[str, torch.Tensor] | None = None,
    mrope_positions: torch.Tensor | None = None,
    mrope_position_delta: int = 0,
):
    multimodal_inputs = None
    if mrope_positions is not None:
        multimodal_inputs = SimpleNamespace(
            mrope_positions=mrope_positions,
            mrope_position_delta=torch.tensor([[mrope_position_delta]]),
        )
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=list(prompt_ids),
        omni_model_inputs=omni_model_inputs,
        _omni_mm_positions=mm_positions,
        multimodal_inputs=multimodal_inputs,
        req_pool_idx=0,
    )


def _prefill(runner, req, token_ids):
    return runner.prefill_start(
        req_id=req.rid,
        new_token_ids=list(token_ids),
        full_token_ids=list(token_ids),
        prefix_slot_ids=[],
        new_slot_ids=list(range(len(token_ids))),
        req_pool_idx=0,
        req=req,
    )


# ---------------------------------------------------------------------------
# Step 1: architecture dispatch


def test_mlx_worker_dispatch_preserves_asr_and_adds_omni(monkeypatch):
    observed = []
    monkeypatch.setattr(
        mlx_model_worker,
        "_MLX_WORKER_FACTORIES",
        {
            "Qwen3ASRForConditionalGeneration": lambda **kwargs: observed.append("asr"),
            "Qwen3OmniThinkerForCausalLM": lambda **kwargs: observed.append("thinker"),
        },
    )
    mlx_model_worker.create_mlx_model_worker(
        config=SimpleNamespace(model_arch_override="Qwen3OmniThinkerForCausalLM"),
        server_args=object(),
        gpu_id=0,
    )
    assert observed == ["thinker"]


def test_mlx_worker_dispatch_still_routes_qwen3_asr(monkeypatch):
    observed = []
    monkeypatch.setattr(
        mlx_model_worker,
        "_MLX_WORKER_FACTORIES",
        {
            "Qwen3ASRForConditionalGeneration": lambda **kwargs: observed.append("asr"),
            "Qwen3OmniThinkerForCausalLM": lambda **kwargs: observed.append("thinker"),
        },
    )
    mlx_model_worker.create_mlx_model_worker(
        config=SimpleNamespace(
            model_arch_override="Qwen3ASRForConditionalGeneration",
        ),
        server_args=object(),
        gpu_id=0,
    )
    assert observed == ["asr"]


def test_real_dispatch_table_covers_asr_thinker_and_talker():
    assert set(mlx_model_worker._MLX_WORKER_FACTORIES) == {
        "Qwen3ASRForConditionalGeneration",
        "Qwen3OmniThinkerForCausalLM",
        "Qwen3OmniTalker",
    }


def test_unknown_architecture_is_rejected():
    with pytest.raises(NotImplementedError, match="SomethingElse"):
        mlx_model_worker.create_mlx_model_worker(
            config=SimpleNamespace(model_arch_override="SomethingElse"),
            server_args=object(),
            gpu_id=0,
        )


def test_talker_mlx_worker_dispatches_to_the_omni_factory(monkeypatch):
    """``Qwen3OmniTalker`` now builds a real MLX worker, not a NotImplementedError."""

    built = {}

    def _fake_worker(**kwargs):
        built.update(kwargs)
        raise RuntimeError("stop after talker dispatch")

    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.mlx.runner."
        "_create_qwen3_omni_talker_mlx_worker",
        _fake_worker,
    )
    with pytest.raises(RuntimeError, match="stop after talker dispatch"):
        create_qwen3_omni_mlx_worker(
            config=SimpleNamespace(model_arch_override="Qwen3OmniTalker"),
            server_args=object(),
            gpu_id=0,
        )
    assert built["gpu_id"] == 0


def test_mlx_worker_has_no_torch_prefill_shadow(
    monkeypatch, tiny_checkpoint: Path
) -> None:
    from sglang.srt.runtime_context import get_context

    from sglang_omni.model_runner import external_model_worker
    from sglang_omni.models.qwen3_omni.mlx import runner as mlx_runner

    class _FakeExternalWorker:
        def __init__(self, **kwargs):
            self.server_args = object()
            self._init_model_runner()

        def _init_model_runner(self):
            self.model_runner = SimpleNamespace(model=object())

    monkeypatch.setattr(
        external_model_worker,
        "_make_external_worker_class",
        lambda: _FakeExternalWorker,
    )
    monkeypatch.setattr(
        external_model_worker, "_build_parallel_state", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        external_model_worker, "_resolve_nccl_port", lambda *args, **kwargs: 12345
    )
    monkeypatch.setattr(
        external_model_worker,
        "_publish_scheduler_runtime_context",
        lambda *args, **kwargs: None,
    )
    server_args = SimpleNamespace(
        quantization=None,
        mlx_enable_sampling=False,
        model_path=str(tiny_checkpoint),
        trust_remote_code=False,
        revision=None,
    )

    with get_context().override_server_args(**vars(server_args)):
        worker = mlx_runner._create_qwen3_omni_talker_mlx_worker(
            config=SimpleNamespace(nccl_port=None),
            server_args=server_args,
            gpu_id=0,
        )

    assert not hasattr(worker, "talker_prefill_shim")
    assert isinstance(
        worker.mlx_talker_prefill_builder,
        Qwen3OmniMlxTalkerPrefillBuilder,
    )
    assert not hasattr(mlx_runner, "_dense_mlx_weight")


def test_mlx_load_error_never_falls_back_to_another_worker(monkeypatch):
    """SGLANG_USE_MLX=1 must propagate an MLX failure."""

    import sglang.srt.runtime_context as runtime_context
    import sglang.srt.hardware_backend.mlx.runtime as tensor_bridge

    from sglang_omni.model_runner import external_model_worker
    from sglang_omni.models.qwen3_omni import apple_runtime
    from sglang_omni.scheduling import bootstrap

    external_calls = []
    torch_worker_calls = []

    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: True)
    monkeypatch.setattr(apple_runtime, "qwen3_omni_uses_apple_backend", lambda: True)
    monkeypatch.setattr(
        runtime_context,
        "get_context",
        lambda: SimpleNamespace(is_config_namespace_published=lambda name: False),
    )
    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda server_args, gpu_id: "test",
    )

    def _external(**kwargs):
        external_calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        external_model_worker, "create_external_model_worker", _external
    )

    def _failing_mlx(**kwargs):
        torch_worker_calls.append(kwargs)
        raise RuntimeError("MLX architecture load failed: Qwen3OmniThinkerForCausalLM")

    monkeypatch.setattr(mlx_model_worker, "create_mlx_model_worker", _failing_mlx)

    with pytest.raises(RuntimeError, match="MLX architecture load failed"):
        bootstrap.create_sglang_infrastructure(
            SimpleNamespace(),
            0,
            model_arch_override="Qwen3OmniThinkerForCausalLM",
        )

    assert len(torch_worker_calls) == 1
    assert external_calls == []


# ---------------------------------------------------------------------------
# Step 3: ModelWorkerConfig capture propagation


def test_model_worker_config_carries_capture_hidden_layers():
    config = ModelWorkerConfig(capture_hidden_layers=(0, 24))
    assert config.capture_hidden_layers == (0, 24)
    assert ModelWorkerConfig().capture_hidden_layers is None


def test_infrastructure_forwards_capture_layers_to_the_mlx_factory(monkeypatch):
    import sglang.srt.runtime_context as runtime_context
    import sglang.srt.hardware_backend.mlx.runtime as tensor_bridge

    from sglang_omni.scheduling import bootstrap

    seen = {}

    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: True)
    monkeypatch.setattr(
        runtime_context,
        "get_context",
        lambda: SimpleNamespace(is_config_namespace_published=lambda name: False),
    )
    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda server_args, gpu_id: "test",
    )

    def _factory(*, config, server_args, gpu_id, tp_rank=0):
        seen["capture"] = config.capture_hidden_layers
        raise RuntimeError("stop after config construction")

    monkeypatch.setattr(mlx_model_worker, "create_mlx_model_worker", _factory)

    with pytest.raises(RuntimeError, match="stop after config construction"):
        bootstrap.create_sglang_infrastructure(
            SimpleNamespace(),
            0,
            model_arch_override="Qwen3OmniThinkerForCausalLM",
            capture_hidden_layers=[0, 24],
        )

    assert seen["capture"] == (0, 24)


def test_runner_rejects_capture_layer_that_talker_does_not_accept():
    with pytest.raises(ValueError, match="accept_hidden_layer"):
        build_qwen3_omni_thinker_mlx_runner(
            model=_seeded_thinker(2),
            placeholder_token_ids=dict(UNIT_PLACEHOLDER_IDS),
            capture_hidden_layers=(0, 1),
            accept_hidden_layer=7,
            disable_radix_cache=True,
            pool_size=64,
        )


def test_runner_rejects_more_than_one_nonzero_capture_layer():
    with pytest.raises(ValueError, match="exactly one"):
        build_qwen3_omni_thinker_mlx_runner(
            model=_seeded_thinker(3),
            placeholder_token_ids=dict(UNIT_PLACEHOLDER_IDS),
            capture_hidden_layers=(0, 1, 2),
            accept_hidden_layer=2,
            disable_radix_cache=True,
            pool_size=64,
        )


# ---------------------------------------------------------------------------
# Prefill behaviour


def test_prefill_requires_exactly_one_request():
    runner = _unit_runner()
    req_a = _make_req(rid="a", prompt_ids=[1, 2, 3])
    req_b = _make_req(rid="b", prompt_ids=[4, 5, 6])
    _prefill(runner, req_a, [1, 2, 3])
    with pytest.raises(RuntimeError, match="one request"):
        _prefill(runner, req_b, [4, 5, 6])


def test_prefill_rejects_radix_prefix_and_sampling():
    runner = _unit_runner()
    req = _make_req(prompt_ids=[1, 2, 3])
    with pytest.raises(NotImplementedError, match="radix prefix"):
        runner.prefill_start(
            req_id=req.rid,
            new_token_ids=[2, 3],
            full_token_ids=[1, 2, 3],
            prefix_slot_ids=[0],
            new_slot_ids=[1, 2],
            req_pool_idx=0,
            req=req,
        )
    with pytest.raises(NotImplementedError, match="greedy"):
        runner.prefill_start(
            req_id=req.rid,
            new_token_ids=[1, 2, 3],
            full_token_ids=[1, 2, 3],
            prefix_slot_ids=[],
            new_slot_ids=[0, 1, 2],
            req_pool_idx=0,
            req=req,
            logprob_spec=object(),
        )


def test_prefill_merges_two_separate_runs_of_each_modality():
    """Two images and two audio clips put four separate placeholder runs in one"""

    runner = _unit_runner()
    pad_image = 10_000_011
    pad_audio = 10_000_013
    prompt = [5, pad_image, 6, pad_audio, 7, pad_image, 8, pad_audio, 9]
    image_values = np.array([[0.25] * HIDDEN, [-0.75] * HIDDEN], dtype=np.float32)
    audio_values = np.array([[0.5] * HIDDEN, [-0.5] * HIDDEN], dtype=np.float32)
    req = _make_req(
        prompt_ids=prompt,
        omni_model_inputs={
            "image_embeds": torch.from_numpy(image_values),
            "audio_embeds": torch.from_numpy(audio_values),
            "pad_values": {"image": pad_image, "audio": pad_audio},
        },
        mm_positions={
            "image": torch.tensor([1, 5], dtype=torch.long),
            "video": torch.zeros(0, dtype=torch.long),
            "audio": torch.tensor([3, 7], dtype=torch.long),
        },
    )

    _prefill(runner, req, prompt)

    assert runner.last_prefill_input_ids == [
        5,
        UNIT_PLACEHOLDER_IDS["image"],
        6,
        UNIT_PLACEHOLDER_IDS["audio"],
        7,
        UNIT_PLACEHOLDER_IDS["image"],
        8,
        UNIT_PLACEHOLDER_IDS["audio"],
        9,
    ]
    merged = np.asarray(runner.last_prefill_embeddings, dtype=np.float32)
    np.testing.assert_allclose(merged[0, 1], image_values[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(merged[0, 5], image_values[1], rtol=0, atol=1e-6)
    np.testing.assert_allclose(merged[0, 3], audio_values[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(merged[0, 7], audio_values[1], rtol=0, atol=1e-6)

    text_rows = np.asarray(
        runner.model.embed_tokens(mx.array([prompt[:1]], dtype=mx.int32)),
        dtype=np.float32,
    )
    np.testing.assert_allclose(merged[0, 0], text_rows[0, 0], rtol=0, atol=1e-6)


def test_prefill_interleaves_deepstack_rows_across_two_image_and_video_runs():
    """Image and video DeepStack rows are interleaved by prompt position, so two
    runs of each must produce one joint layer ordered by absolute position."""

    runner = _unit_runner()
    prompt = [
        UNIT_PLACEHOLDER_IDS["image"],
        UNIT_PLACEHOLDER_IDS["video"],
        5,
        UNIT_PLACEHOLDER_IDS["video"],
        UNIT_PLACEHOLDER_IDS["image"],
    ]
    image_rows = torch.tensor([[1.0] * HIDDEN, [4.0] * HIDDEN])
    video_rows = torch.tensor([[2.0] * HIDDEN, [3.0] * HIDDEN])
    req = _make_req(
        prompt_ids=prompt,
        omni_model_inputs={
            "image_embeds": image_rows,
            "video_embeds": video_rows,
            "image_deepstack_visual_embeds": [image_rows],
            "video_deepstack_visual_embeds": [video_rows],
        },
        mm_positions={
            "image": torch.tensor([0, 4], dtype=torch.long),
            "video": torch.tensor([1, 3], dtype=torch.long),
            "audio": torch.zeros(0, dtype=torch.long),
        },
    )

    layers = runner._deepstack_embeddings(req, req.omni_model_inputs)

    assert len(layers) == 1
    np.testing.assert_allclose(
        np.asarray(layers[0], dtype=np.float32)[:, 0],
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    _prefill(runner, req, prompt)


def test_prefill_restores_cache_key_placeholder_ids():
    runner = _unit_runner()
    pad_audio = 10_000_003
    prompt = [5, pad_audio, pad_audio, 6]
    audio_embeds = torch.zeros((2, HIDDEN), dtype=torch.bfloat16)
    req = _make_req(
        prompt_ids=prompt,
        omni_model_inputs={
            "audio_embeds": audio_embeds,
            "pad_values": {"audio": pad_audio},
        },
        mm_positions={
            "image": torch.zeros(0, dtype=torch.long),
            "video": torch.zeros(0, dtype=torch.long),
            "audio": torch.tensor([1, 2], dtype=torch.long),
        },
    )
    pending = _prefill(runner, req, prompt)

    restored = runner.last_prefill_input_ids
    assert restored == [
        5,
        UNIT_PLACEHOLDER_IDS["audio"],
        UNIT_PLACEHOLDER_IDS["audio"],
        6,
    ]
    # Out-of-vocabulary placeholders must never reach the embedding table.
    assert max(restored) < VOCAB
    # Later decode bookkeeping keeps the restored ids, not the cache keys.
    assert pending.full_token_ids == restored


def test_prefill_converts_bfloat16_without_calling_numpy_on_bf16():
    runner = _unit_runner()
    pad_audio = 10_000_007
    prompt = [5, pad_audio, pad_audio, 6]
    values = np.array([[0.5] * HIDDEN, [-0.25] * HIDDEN], dtype=np.float32)
    audio_embeds = torch.from_numpy(values).to(torch.bfloat16)

    def _forbidden_numpy(self, *args, **kwargs):
        raise AssertionError("numpy() called directly on a bfloat16 tensor")

    original_numpy = torch.Tensor.numpy

    def _guarded_numpy(self, *args, **kwargs):
        if self.dtype == torch.bfloat16:
            return _forbidden_numpy(self)
        return original_numpy(self, *args, **kwargs)

    req = _make_req(
        prompt_ids=prompt,
        omni_model_inputs={
            "audio_embeds": audio_embeds,
            "pad_values": {"audio": pad_audio},
        },
        mm_positions={
            "image": torch.zeros(0, dtype=torch.long),
            "video": torch.zeros(0, dtype=torch.long),
            "audio": torch.tensor([1, 2], dtype=torch.long),
        },
    )
    torch.Tensor.numpy = _guarded_numpy
    try:
        _prefill(runner, req, prompt)
    finally:
        torch.Tensor.numpy = original_numpy

    merged = np.asarray(runner.last_prefill_embeddings, dtype=np.float32)
    np.testing.assert_allclose(merged[0, 1], values[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(merged[0, 2], values[1], rtol=0, atol=1e-6)


def test_prefill_forwards_mrope_positions_from_the_request():
    runner = _unit_runner()
    prompt = [1, 2, 3, 4]
    positions = torch.tensor(
        [[0, 1, 2, 3], [0, 5, 6, 3], [0, 7, 8, 3]], dtype=torch.long
    )
    req = _make_req(prompt_ids=prompt, mrope_positions=positions)
    _prefill(runner, req, prompt)
    np.testing.assert_array_equal(
        np.asarray(runner.last_prefill_positions), positions.numpy()
    )


def test_text_only_prefill_uses_sequential_positions():
    runner = _unit_runner()
    prompt = [1, 2, 3, 4]
    req = _make_req(prompt_ids=prompt)
    _prefill(runner, req, prompt)
    expected = np.tile(np.arange(4), (3, 1))
    np.testing.assert_array_equal(np.asarray(runner.last_prefill_positions), expected)


# ---------------------------------------------------------------------------
# Hidden-state transport and the first-row stream contract


def test_prefill_exposes_full_sequence_cpu_torch_hidden_states():
    runner = _unit_runner()
    prompt = [1, 2, 3, 4, 5]
    req = _make_req(prompt_ids=prompt)
    pending = _prefill(runner, req, prompt)
    runner.prefill_finalize(pending)

    hidden = runner.pop_hidden_states(pending)
    assert set(hidden) == {"embed", 1}
    for value in hidden.values():
        assert isinstance(value, torch.Tensor)
        assert value.device.type == "cpu"
        assert value.shape == (len(prompt), HIDDEN)
        assert torch.isfinite(value).all()


def test_media_prefill_retains_real_layer24_capture_for_transport():
    runner = _unit_runner(capture_hidden_layers=(0, 24), num_layers=25)
    prompt = [1, UNIT_PLACEHOLDER_IDS["image"], 3]
    req = _make_req(
        prompt_ids=prompt,
        omni_model_inputs={"image_embeds": torch.full((1, HIDDEN), 0.25)},
        mm_positions={"image": torch.tensor([1])},
    )
    pending = _prefill(runner, req, prompt)
    runner.prefill_finalize(pending)
    hidden = runner.pop_hidden_states(pending)
    torch.testing.assert_close(hidden["mlx_prompt_hidden"][0], hidden[24])
    assert hidden["mlx_prompt_hidden"].shape == (1, len(prompt), HIDDEN)
    assert not torch.allclose(hidden[24][1], hidden["embed"][1])
    assert torch.count_nonzero(hidden[24][1]) > 0


def test_first_prefill_row_not_the_last_reaches_the_talker_stream():
    from sglang_omni.models.qwen3_omni.request_builders import (
        make_thinker_stream_output_builder,
    )

    runner = _unit_runner()
    prompt = [1, 2, 3, 4, 5]
    req = _make_req(prompt_ids=prompt)
    pending = _prefill(runner, req, prompt)
    runner.prefill_finalize(pending)
    hidden = runner.pop_hidden_states(pending)

    # Rows must genuinely differ, or "first row" would be vacuous.
    embed = hidden["embed"]
    assert torch.max(torch.abs(embed[0] - embed[-1])).item() > 1e-3

    processor = SGLangOutputProcessor(
        capture_hidden=True,
        capture_hidden_layers=[0, 1],
        model=None,
    )
    scheduler_output = _scheduler_output_for(req, hidden)
    outputs = processor.process(
        SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=hidden),
            next_token_ids=torch.tensor([7]),
        ),
        scheduler_output,
    )
    extra = outputs[req.rid].extra
    streamed = extra["hidden_states"]["embed"]
    assert torch.equal(streamed, embed[0])
    assert not torch.equal(streamed, embed[-1])

    builder = make_thinker_stream_output_builder()
    req_data = SimpleNamespace(
        req=SimpleNamespace(inflight_middle_chunks=0),
        stage_payload=SimpleNamespace(
            # ``should_generate_audio_output`` unwraps a real ``StagePayload``
            # and otherwise treats its argument as the request itself.
            metadata={"output_modalities": ["audio"]},
            request=SimpleNamespace(
                params={"stream": False},
                metadata={"output_modalities": ["audio"]},
            ),
        ),
    )
    messages = builder(req.rid, req_data, outputs[req.rid])
    talker = [m for m in messages if m.target == "talker_ar"]
    assert len(talker) == 1
    assert torch.equal(talker[0].data, embed[0])


def _scheduler_output_for(req, hidden):
    from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest

    extend_range = SimpleNamespace(length=next(iter(hidden.values())).shape[0])
    batch_req = SimpleNamespace(rid=req.rid, extend_range=extend_range)
    return SchedulerOutput(
        requests=[
            SchedulerRequest(request_id=req.rid, data=SimpleNamespace(req=batch_req))
        ],
        batch_data=SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            reqs=[batch_req],
        ),
    )


# ---------------------------------------------------------------------------
# Decode chaining and cache lifecycle


def test_prefill_then_decode_reuses_the_same_cache_and_advances_positions():
    runner = _unit_runner()
    prompt = [1, 2, 3, 4]
    positions = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
    )
    req = _make_req(
        prompt_ids=prompt, mrope_positions=positions, mrope_position_delta=2
    )
    pending = _prefill(runner, req, prompt)
    first_token = runner.prefill_finalize(pending)
    cache = runner._req_caches[req.rid]
    assert cache[0].offset == len(prompt)

    decode = runner.decode_batch_start([req.rid])
    assert decode.caches[0] is cache
    np.testing.assert_array_equal(
        np.asarray(runner.last_decode_positions),
        np.full((3, 1), len(prompt) + 2),
    )
    tokens = runner.decode_batch_finalize(decode)
    assert len(tokens) == 1
    assert cache[0].offset == len(prompt) + 1
    assert runner._req_token_ids[req.rid][-1] == tokens[0]
    assert runner._req_token_ids[req.rid][-2] == first_token

    hidden = runner.pop_hidden_states(decode)
    assert set(hidden) == {"embed", 1}
    for value in hidden.values():
        assert value.shape == (1, HIDDEN)
        assert value.device.type == "cpu"


def test_chained_decode_keeps_per_step_hidden_states_separate():
    runner = _unit_runner()
    prompt = [1, 2, 3, 4]
    req = _make_req(prompt_ids=prompt)
    pending = _prefill(runner, req, prompt)
    runner.prefill_finalize(pending)

    first = runner.decode_batch_start([req.rid])
    second = runner.decode_batch_start_chained(first)
    first_hidden = runner.pop_hidden_states(first)
    second_hidden = runner.pop_hidden_states(second)
    assert first_hidden is not None and second_hidden is not None
    assert not torch.equal(first_hidden["embed"], second_hidden["embed"])

    runner.decode_batch_finalize(first)
    runner.decode_batch_finalize(second)
    assert runner._req_caches[req.rid][0].offset == len(prompt) + 2


def test_decode_rejects_multi_request_batches_and_sampling():
    runner = _unit_runner()
    prompt = [1, 2, 3]
    req = _make_req(prompt_ids=prompt)
    runner.prefill_finalize(_prefill(runner, req, prompt))
    with pytest.raises(RuntimeError, match="one request"):
        runner.decode_batch_start([req.rid, "other"])
    with pytest.raises(NotImplementedError, match="greedy"):
        runner.decode_batch_start([req.rid], logprob_spec=object())


def test_abort_clears_the_request_cache_and_hidden_state():
    runner = _unit_runner()
    prompt = [1, 2, 3]
    req = _make_req(prompt_ids=prompt)
    runner.prefill_finalize(_prefill(runner, req, prompt))
    assert runner.has_request(req.rid)

    runner.remove_request(req.rid)

    assert not runner.has_request(req.rid)
    assert req.rid not in runner._req_caches
    assert req.rid not in runner._req_token_ids
    assert req.rid not in runner._req_mrope_delta
    # A fresh prefill after the abort is admitted again.
    runner.prefill_finalize(_prefill(runner, req, prompt))
    assert runner.has_request(req.rid)


def test_scheduler_runner_aborts_and_finishes_through_the_mlx_runner():
    removed = []
    worker = SimpleNamespace(_mlx_runner=SimpleNamespace(remove_request=removed.append))
    runner = Qwen3OmniMlxSchedulerModelRunner.__new__(Qwen3OmniMlxSchedulerModelRunner)
    runner.tp_worker = worker
    runner._last_mlx_pending = None
    runner._execution_bridge = None

    runner.abort_request("req-a")
    runner.on_request_finished("req-b", SimpleNamespace())

    assert removed == ["req-a", "req-b"]


def test_scheduler_runner_preserves_hidden_outputs_during_finalize():
    """``_finalize`` must hand the runner-populated dict to the processor."""

    hidden = {"embed": torch.zeros((3, HIDDEN)), 1: torch.ones((3, HIDDEN))}
    seen = {}

    class _Processor:
        _capture_hidden = True

        def process(self, model_output, scheduler_output, host_token_ids=None):
            seen["hidden"] = model_output.logits_output.hidden_states
            return {
                req.request_id: SimpleNamespace(extra=None)
                for req in scheduler_output.requests
            }

    runner = Qwen3OmniMlxSchedulerModelRunner.__new__(Qwen3OmniMlxSchedulerModelRunner)
    runner.output_processor = _Processor()
    runner._last_mlx_pending = None
    batch_result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=hidden),
        next_token_ids=torch.tensor([3]),
        can_run_cuda_graph=False,
    )
    from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest

    scheduler_output = SchedulerOutput(
        requests=[
            SchedulerRequest(
                request_id="req",
                data=SimpleNamespace(generation_steps=0, extra_model_outputs={}),
            )
        ],
        batch_data=SimpleNamespace(reqs=[SimpleNamespace(rid="req")]),
    )
    runner._finalize(batch_result, None, scheduler_output.batch_data, scheduler_output)
    assert seen["hidden"] is hidden


# ---------------------------------------------------------------------------
# Real checkpoint loading: dense and converted 4-bit (incl. MoE expert stacks)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tiny_qwen3_omni_mlx_runner")
    return build_tiny_qwen3_omni_checkpoint(root / "tiny")


def test_dense_local_checkpoint_loads_and_prefills(tiny_checkpoint: Path):
    runner_class = make_qwen3_omni_thinker_mlx_runner_class()
    runner = runner_class(
        model_path=str(tiny_checkpoint),
        disable_radix_cache=True,
        capture_hidden_layers=(0, 24),
        pool_size=2048,
    )
    assert runner.accept_hidden_layer == 24
    assert runner.model.num_layers == 25
    assert runner.placeholder_token_ids == PLACEHOLDER_IDS

    prompt = [11, 12, 13, 14]
    req = _make_req(prompt_ids=prompt)
    pending = runner.prefill_start(
        req_id=req.rid,
        new_token_ids=prompt,
        full_token_ids=prompt,
        prefix_slot_ids=[],
        new_slot_ids=[0, 1, 2, 3],
        req_pool_idx=0,
        req=req,
    )
    token = runner.prefill_finalize(pending)
    assert isinstance(token, int)
    hidden = runner.pop_hidden_states(pending)
    assert set(hidden) == {"embed", 24}
    assert hidden["embed"].shape == (4, 32)
    assert torch.isfinite(hidden[24]).all()


def _write_quantized_thinker_checkpoint(
    directory: Path,
    *,
    bare_expert_weight_names: bool,
    bits: int = 4,
    group_size: int = 32,
) -> Path:
    """A converted 4-bit checkpoint whose MoE expert stacks are quantized."""

    directory.mkdir(parents=True, exist_ok=True)
    dense = _seeded_thinker(2)
    nn.quantize(dense, group_size=group_size, bits=bits)

    weights: dict[str, mx.array] = {}
    for name, value in tree_flatten(dense.parameters()):
        key = f"thinker.{name}"
        if bare_expert_weight_names and key.endswith(
            (".experts.gate_up_proj.weight", ".experts.down_proj.weight")
        ):
            key = key[: -len(".weight")]
        weights[key] = value
    mx.save_safetensors(str(directory / "model.safetensors"), weights)

    text = _unit_text_config(2)
    config = build_tiny_config().to_dict()
    config["architectures"] = ["Qwen3OmniMoeForConditionalGeneration"]
    config["thinker_config"]["image_token_id"] = UNIT_PLACEHOLDER_IDS["image"]
    config["thinker_config"]["video_token_id"] = UNIT_PLACEHOLDER_IDS["video"]
    config["thinker_config"]["audio_token_id"] = UNIT_PLACEHOLDER_IDS["audio"]
    config["thinker_config"]["text_config"] = {
        "hidden_size": text.hidden_size,
        "head_dim": text.head_dim,
        "num_attention_heads": text.num_attention_heads,
        "num_key_value_heads": text.num_key_value_heads,
        "num_hidden_layers": text.num_hidden_layers,
        "intermediate_size": text.intermediate_size,
        "moe_intermediate_size": text.moe_intermediate_size,
        "num_experts": text.num_experts,
        "num_experts_per_tok": text.num_experts_per_tok,
        "norm_topk_prob": text.norm_topk_prob,
        "decoder_sparse_step": text.decoder_sparse_step,
        "mlp_only_layers": list(text.mlp_only_layers),
        "rms_norm_eps": text.rms_norm_eps,
        "vocab_size": text.vocab_size,
        "max_position_embeddings": text.max_position_embeddings,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "rope_scaling": {
            "rope_type": "default",
            "mrope_section": list(text.mrope_section),
        },
        "rope_theta": text.rope_theta,
    }
    config["talker_config"]["accept_hidden_layer"] = 1
    config["quantization"] = {"bits": bits, "group_size": group_size}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


@pytest.mark.parametrize("bare_expert_weight_names", [False, True])
def test_converted_4bit_checkpoint_loads_quantized_expert_stacks(
    tmp_path: Path, bare_expert_weight_names: bool
):
    directory = _write_quantized_thinker_checkpoint(
        tmp_path / f"q4-{int(bare_expert_weight_names)}",
        bare_expert_weight_names=bare_expert_weight_names,
    )
    runner_class = make_qwen3_omni_thinker_mlx_runner_class()
    runner = runner_class(
        model_path=str(directory),
        disable_radix_cache=True,
        capture_hidden_layers=(0, 1),
        pool_size=512,
    )

    experts = runner.model.model.layers[0].mlp.experts
    # The expert stacks must be genuinely quantized modules that consume the
    # checkpoint's scales and biases -- not silently dropped or dequantized.
    assert hasattr(experts.gate_up_proj, "scales")
    assert hasattr(experts.gate_up_proj, "biases")
    assert hasattr(experts.down_proj, "scales")
    parameter_names = {name for name, _ in tree_flatten(runner.model.parameters())}
    assert "model.layers.0.mlp.experts.gate_up_proj.scales" in parameter_names
    assert "model.layers.0.mlp.experts.gate_up_proj.biases" in parameter_names

    prompt = [3, 4, 5, 6]
    req = _make_req(prompt_ids=prompt)
    pending = runner.prefill_start(
        req_id=req.rid,
        new_token_ids=prompt,
        full_token_ids=prompt,
        prefix_slot_ids=[],
        new_slot_ids=[0, 1, 2, 3],
        req_pool_idx=0,
        req=req,
    )
    runner.prefill_finalize(pending)
    hidden = runner.pop_hidden_states(pending)
    assert torch.isfinite(hidden["embed"]).all()
    assert torch.isfinite(hidden[1]).all()

    decode = runner.decode_batch_start([req.rid])
    tokens = runner.decode_batch_finalize(decode)
    assert len(tokens) == 1
    assert 0 <= tokens[0] < VOCAB


def test_quantized_expert_scales_are_actually_consumed(tmp_path: Path):
    """Perturbing only the expert scales must change the model output."""

    base = _write_quantized_thinker_checkpoint(
        tmp_path / "q4-base", bare_expert_weight_names=False
    )
    perturbed = tmp_path / "q4-perturbed"
    perturbed.mkdir()
    (perturbed / "config.json").write_text(
        (base / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    weights = dict(mx.load(str(base / "model.safetensors")))
    scale_keys = [k for k in weights if k.endswith(".experts.gate_up_proj.scales")]
    assert scale_keys
    for key in scale_keys:
        weights[key] = weights[key] * 3.0
    mx.save_safetensors(str(perturbed / "model.safetensors"), weights)

    runner_class = make_qwen3_omni_thinker_mlx_runner_class()
    prompt = [3, 4, 5, 6]

    def _embed_capture(path: Path) -> np.ndarray:
        runner = runner_class(
            model_path=str(path),
            disable_radix_cache=True,
            capture_hidden_layers=(0, 1),
            pool_size=512,
        )
        req = _make_req(prompt_ids=prompt)
        pending = runner.prefill_start(
            req_id=req.rid,
            new_token_ids=prompt,
            full_token_ids=prompt,
            prefix_slot_ids=[],
            new_slot_ids=[0, 1, 2, 3],
            req_pool_idx=0,
            req=req,
        )
        runner.prefill_finalize(pending)
        return runner.pop_hidden_states(pending)[1].numpy()

    original = _embed_capture(base)
    changed = _embed_capture(perturbed)
    assert np.isfinite(original).all() and np.isfinite(changed).all()
    assert np.max(np.abs(original - changed)) > 1e-3


def test_attention_bias_true_is_threaded_into_the_native_model(tmp_path: Path):
    directory = _write_quantized_thinker_checkpoint(
        tmp_path / "bias", bare_expert_weight_names=False
    )
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config["thinker_config"]["text_config"]["attention_bias"] = True
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")

    runner_class = make_qwen3_omni_thinker_mlx_runner_class()
    with pytest.raises(Exception) as excinfo:
        runner_class(
            model_path=str(directory),
            disable_radix_cache=True,
            capture_hidden_layers=(0, 1),
            pool_size=512,
        )
    # The bias parameters are genuinely created, so a bias-free checkpoint
    # fails loudly instead of silently omitting them.
    assert "bias" in str(excinfo.value).lower()


def test_quantization_metadata_is_read_from_the_checkpoint(tmp_path: Path):
    directory = _write_quantized_thinker_checkpoint(
        tmp_path / "meta", bare_expert_weight_names=False, bits=4, group_size=32
    )
    runner_class = make_qwen3_omni_thinker_mlx_runner_class()
    runner = runner_class(
        model_path=str(directory),
        disable_radix_cache=True,
        capture_hidden_layers=(0, 1),
        pool_size=512,
    )
    assert runner.quantization == QuantizationConfig(bits=4, group_size=32)
    assert runner.model.model.layers[0].self_attn.q_proj.bits == 4
    assert runner.model.model.layers[0].mlp.experts.gate_up_proj.group_size == 32


def test_dense_checkpoint_captures_match_transformers(tiny_checkpoint: Path):
    """Independent oracle for the on-disk (per-expert) MoE weight layout."""

    from transformers import Qwen3OmniMoeForConditionalGeneration

    reference = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(tiny_checkpoint), dtype=torch.float32
    )
    reference.eval()
    text_model = reference.thinker.model

    prompt = [11, 12, 13, 14, 15, 16]
    input_ids = torch.tensor([prompt], dtype=torch.long)
    positions = torch.arange(len(prompt)).view(1, 1, -1).repeat(3, 1, 1)
    with torch.no_grad():
        output = text_model(
            input_ids=input_ids,
            position_ids=positions,
            output_hidden_states=True,
            use_cache=False,
        )
    # hidden_states[0] is the embedding output (== the input of layer 0) and
    # hidden_states[N] is the input of layer N, matching the capture convention.
    expected_embed = output.hidden_states[0][0].numpy()
    expected_layer = output.hidden_states[24][0].numpy()

    runner_class = make_qwen3_omni_thinker_mlx_runner_class()
    runner = runner_class(
        model_path=str(tiny_checkpoint),
        disable_radix_cache=True,
        capture_hidden_layers=(0, 24),
        pool_size=2048,
    )
    req = _make_req(prompt_ids=prompt)
    pending = runner.prefill_start(
        req_id=req.rid,
        new_token_ids=prompt,
        full_token_ids=prompt,
        prefix_slot_ids=[],
        new_slot_ids=list(range(len(prompt))),
        req_pool_idx=0,
        req=req,
    )
    runner.prefill_finalize(pending)
    hidden = runner.pop_hidden_states(pending)

    # Non-vacuity: the layer-24 rows must actually differ from the embeddings.
    assert np.max(np.abs(expected_layer - expected_embed)) > 1e-3
    np.testing.assert_allclose(
        hidden["embed"].numpy(), expected_embed, rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(hidden[24].numpy(), expected_layer, rtol=0, atol=2e-4)


# ---------------------------------------------------------------------------
# Native MLX talker: real scheduler fixtures

IM_START = OFFICIAL_SPECIAL_TOKEN_IDS["im_start_token_id"]
IM_END = OFFICIAL_SPECIAL_TOKEN_IDS["im_end_token_id"]
SYSTEM_ID = OFFICIAL_SPECIAL_TOKEN_IDS["system_token_id"]
USER_ID = OFFICIAL_SPECIAL_TOKEN_IDS["user_token_id"]
ASSISTANT_ID = OFFICIAL_SPECIAL_TOKEN_IDS["assistant_token_id"]
CODEC_EOS = OFFICIAL_SPECIAL_TOKEN_IDS["talker_config.codec_eos_token_id"]
CODEC_BOS = OFFICIAL_SPECIAL_TOKEN_IDS["talker_config.codec_bos_id"]
TALKER_VOCAB = 3072
CODE_GROUPS = 16


@pytest.fixture()
def sglang_runtime_context():
    """A published SGLang runtime context with a CPU-safe attention backend."""

    from sglang.srt.runtime_context import get_context

    with get_context().override_server_args(
        page_size=1,
        attention_backend="torch_native",
        prefill_attention_backend="torch_native",
        decode_attention_backend="torch_native",
        disable_radix_cache=True,
    ):
        yield


def _talker_memory_pools(size: int = 512):
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixCache

    kv = MHATokenToKVPool(
        size=size,
        page_size=1,
        dtype=torch.float16,
        head_num=1,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=size, dtype=torch.float16, device="cpu", kvcache=kv, need_sort=False
    )
    req_to_token_pool = ReqToTokenPool(
        size=4, max_context_len=size, device="cpu", enable_memory_saver=False
    )
    # The Apple profile pins ``disable_radix_cache=True``; the tree is still
    # constructed because ``init_next_round_input`` takes one.
    tree_cache = RadixCache(
        CacheInitParams(
            disable=True,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
    )
    return allocator, req_to_token_pool, tree_cache


class _FakeMlxTalkerWorker:
    """Stand-in for the external-forward MLX talker worker."""

    uses_external_forward = True

    def __init__(self, mlx_talker, prefill_builder):
        from sglang.srt.hardware_backend.mlx.model_runner_stub import _DummyModel

        self.gpu_id = 0
        self.mlx_talker = mlx_talker
        self.mlx_talker_prefill_builder = prefill_builder
        self.model_runner = SimpleNamespace(model=_DummyModel())


class _TalkerHarness:
    """Runner + adapters + pools for one MLX talker request."""

    def __init__(self, *, checkpoint: Path, outbox, feedback_enabled: bool = True):
        loaded = load_qwen3_omni_mlx_talker(str(checkpoint))
        self.checkpoint = checkpoint
        self.mlx_talker = loaded["model"]
        root = build_tiny_config()
        talker_config = root.talker_config
        self.prefill_builder = Qwen3OmniMlxTalkerPrefillBuilder.from_talker(
            self.mlx_talker,
            model_path=str(checkpoint),
            special_token_ids={
                "audio_token_id": root.thinker_config.audio_token_id,
                "image_token_id": root.thinker_config.image_token_id,
                "video_token_id": root.thinker_config.video_token_id,
                "tts_bos_token_id": root.tts_bos_token_id,
                "tts_eos_token_id": root.tts_eos_token_id,
                "tts_pad_token_id": root.tts_pad_token_id,
                "im_start_token_id": root.im_start_token_id,
                "im_end_token_id": root.im_end_token_id,
                "system_token_id": root.system_token_id,
                "user_token_id": root.user_token_id,
                "assistant_token_id": root.assistant_token_id,
                "codec_bos_id": talker_config.codec_bos_id,
                "codec_nothink_id": talker_config.codec_nothink_id,
                "codec_think_bos_id": talker_config.codec_think_bos_id,
                "codec_think_eos_id": talker_config.codec_think_eos_id,
                "codec_pad_id": talker_config.codec_pad_id,
            },
            speaker_map=dict(OFFICIAL_SPEAKER_IDS),
        )
        self.worker = _FakeMlxTalkerWorker(self.mlx_talker, self.prefill_builder)
        self.outbox = outbox
        self.runner = build_qwen3_omni_talker_mlx_runner(
            tp_worker=self.worker,
            output_processor=SGLangOutputProcessor(
                capture_hidden=False, capture_hidden_layers=None, model=None
            ),
            outbox=outbox,
            mlx_talker=self.mlx_talker,
            feedback_enabled=feedback_enabled,
        )
        self.allocator, self.req_to_token_pool, self.tree_cache = _talker_memory_pools()
        self._bind_bridge()
        self.request_builder, _, self.append_chunk, self.mark_done = (
            make_talker_scheduler_adapters(
                tokenizer=FakeQwenTokenizer(),
                codec_vocab_size=TALKER_VOCAB,
                prefill_builder=self.prefill_builder,
                thinker_config=root.thinker_config,
                required_aux_hidden_key=talker_config.accept_hidden_layer,
                codec_bos_id=talker_config.codec_bos_id,
                codec_eos_id=talker_config.codec_eos_token_id,
                codec_nothink_id=talker_config.codec_nothink_id,
                codec_think_bos_id=talker_config.codec_think_bos_id,
                codec_think_eos_id=talker_config.codec_think_eos_id,
                codec_pad_id=talker_config.codec_pad_id,
                audio_token_id=root.thinker_config.audio_token_id,
                image_token_id=root.thinker_config.image_token_id,
                video_token_id=root.thinker_config.video_token_id,
                tts_bos_token_id=root.tts_bos_token_id,
                tts_eos_token_id=root.tts_eos_token_id,
                tts_pad_token_id=root.tts_pad_token_id,
                im_start_token_id=root.im_start_token_id,
                im_end_token_id=root.im_end_token_id,
                system_token_id=root.system_token_id,
                user_token_id=root.user_token_id,
                assistant_token_id=root.assistant_token_id,
                speaker_map=dict(OFFICIAL_SPEAKER_IDS),
            )
        )

    def _bind_bridge(self):
        from sglang.srt.managers.overlap_utils import FutureMap
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        from sglang_omni.model_runner.sglang_execution import SGLangExecutionBridge

        allocator, req_to_token_pool, tree_cache = _talker_memory_pools()
        self.allocator = allocator
        self.req_to_token_pool = req_to_token_pool
        self.tree_cache = tree_cache
        self.bridge = SGLangExecutionBridge(
            device=torch.device("cpu"),
            worker=self.worker,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            future_map=FutureMap(
                torch.device("cpu"),
                SpeculativeAlgorithm.NONE,
                req_to_token_pool,
                needs_cpu_seq_lens=False,
            ),
        )
        self.runner.bind_execution_bridge(self.bridge)

    # -- scheduler-output construction ---------------------------------

    def build_request(
        self,
        *,
        request_id: str = "req",
        suppress_tokens: list[int] | None = None,
        thinker_done: bool = True,
        stream: bool = False,
    ):
        payload = _talker_stage_payload(request_id=request_id, stream=stream)
        payload.prefetched_stream_done = thinker_done
        req_data = self.request_builder(payload)
        if suppress_tokens is not None:
            req_data.suppress_tokens = list(suppress_tokens)
        req_data.thinker_chunks_done = thinker_done
        return req_data

    def prefill_output(self, req_data):
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        from sglang_omni.scheduling.omni_scheduler import OmniScheduler

        req = req_data.req
        req._omni_data = req_data
        OmniScheduler._normalize_req_token_arrays(req)
        req.init_next_round_input(tree_cache=self.tree_cache)
        req.set_extend_range(0, len(req.origin_input_ids))
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            tree_cache=self.tree_cache,
            model_config=SimpleNamespace(
                vocab_size=TALKER_VOCAB,
                is_encoder_decoder=False,
                is_hybrid_swa=False,
            ),
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        self.batch = batch
        return SchedulerOutput(
            requests=[SchedulerRequest(request_id=req.rid, data=req_data)],
            batch_data=batch,
        )

    def decode_output(self, scheduler_output, token_id: int):
        """Advance the real batch one decode step, as the scheduler would."""

        batch = scheduler_output.batch_data
        for req in batch.reqs:
            req.output_ids.append(int(token_id))
        batch.output_ids = torch.tensor([int(token_id)], dtype=torch.long)
        batch.prepare_for_decode()
        return SchedulerOutput(
            requests=list(scheduler_output.requests), batch_data=batch
        )

    def drain_outbox(self):
        messages = []
        while True:
            try:
                messages.append(self.outbox.get_nowait())
            except queue.Empty:
                return messages


def _talker_stage_payload(*, request_id: str = "req", stream: bool = False):
    """A talker StagePayload with a real chat-template prompt and thinker chunks."""

    prompt_ids = torch.tensor(
        [
            IM_START,
            SYSTEM_ID,
            100,
            IM_END,
            IM_START,
            USER_ID,
            101,
            102,
            IM_END,
            IM_START,
            ASSISTANT_ID,
        ],
        dtype=torch.long,
    )
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=[], params={"stream": stream}),
        data={
            "prompt": {"input_ids": prompt_ids},
            # A non-empty model_inputs dict is what makes the request builder
            # attach real ``linear_mrope_positions`` rows to the Req.
            "thinker_inputs": {"model_inputs": {"use_audio_in_video": False}},
        },
    )
    payload.prefetched_chunks = [
        SimpleNamespace(data=None, metadata={"token_id": token_id})
        for token_id in (200, 201, 202, 203, 204)
    ]
    payload.prefetched_stream_done = True
    return payload


def build_tiny_mlx_talker_runner(*, checkpoint: Path, outbox=None, **kwargs):
    """The brief's fixture entry point: a fully wired tiny MLX talker runner."""

    return _TalkerHarness(
        checkpoint=checkpoint, outbox=outbox or queue.Queue(), **kwargs
    )


def _suppress_all_but(token_id: int, vocab_size: int = TALKER_VOCAB) -> list[int]:
    return [candidate for candidate in range(vocab_size) if candidate != token_id]


# ---------------------------------------------------------------------------
# Talker: prefill, code emission, feedback


def test_mlx_talker_runner_emits_codes_and_queues_feedback(
    tiny_checkpoint: Path, sglang_runtime_context
):
    outbox = queue.Queue()
    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint, outbox=outbox)
    runner = harness.runner
    req_data = harness.build_request()
    scheduler_output = harness.prefill_output(req_data)

    result = runner.execute(scheduler_output)

    message = outbox.get_nowait()
    assert message.target == "code2wav"
    assert message.type == "stream"
    assert message.data.shape == (runner.num_code_groups,)
    assert runner.num_code_groups == CODE_GROUPS
    assert scheduler_output.requests[0].data.pending_feedback_queue
    assert result.outputs["req"].data is not None
    assert int(result.outputs["req"].data) == int(message.data[0])
    assert message.request_id == "req"
    assert message.metadata == {"stream": False}
    assert outbox.empty()


def test_prefill_feedback_row_is_a_cpu_float32_talker_row(
    tiny_checkpoint: Path, sglang_runtime_context
):
    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    req_data = harness.build_request()
    harness.runner.execute(harness.prefill_output(req_data))

    row = req_data.pending_feedback_queue[0]
    assert row.device == torch.device("cpu")
    assert row.dtype == torch.float32
    assert row.shape == (harness.mlx_talker.config.text_config.hidden_size,)
    assert torch.isfinite(row).all()


def test_decode_consumes_exactly_one_feedback_and_one_text_row(
    tiny_checkpoint: Path, sglang_runtime_context
):
    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    runner = harness.runner
    req_data = harness.build_request()
    scheduler_output = harness.prefill_output(req_data)
    prefill_result = runner.execute(scheduler_output)

    text_rows_before = len(req_data.pending_text_queue)
    feedback_before = len(req_data.pending_feedback_queue)
    assert feedback_before == 1
    assert text_rows_before >= 1

    decode_output = harness.decode_output(
        scheduler_output, int(prefill_result.outputs["req"].data)
    )
    runner.execute(decode_output)

    # One feedback row consumed and one produced; exactly one text row consumed.
    assert len(req_data.pending_feedback_queue) == feedback_before
    assert len(req_data.pending_text_queue) == text_rows_before - 1
    assert len(harness.drain_outbox()) == 2


def test_take_next_decode_rows_matches_the_torch_runner_sum(
    tiny_checkpoint: Path, sglang_runtime_context
):
    """The unsummed pair is exactly ``_take_next_decode_input_embed``'s row."""

    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    req_data = harness.build_request()
    harness.runner.execute(harness.prefill_output(req_data))

    mirror = SimpleNamespace(
        data=SimpleNamespace(
            pending_feedback_queue=list(req_data.pending_feedback_queue),
            pending_text_queue=req_data.pending_text_queue.copy(),
            thinker_chunks_done=req_data.thinker_chunks_done,
            tts_pad_embed=req_data.tts_pad_embed,
            decode_input_embeds=[],
        )
    )
    expected = QwenTalkerModelRunner._take_next_decode_input_embed(
        sched_req=mirror, device=torch.device("cpu"), dtype=torch.float32
    )
    feedback, text = Qwen3OmniMlxTalkerModelRunner._take_next_decode_rows(
        SimpleNamespace(data=req_data)
    )

    assert expected is not None
    torch.testing.assert_close(feedback + text, expected)


def test_decode_is_not_ready_without_a_feedback_or_text_row(
    tiny_checkpoint: Path, sglang_runtime_context
):
    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    runner = harness.runner
    req_data = harness.build_request(thinker_done=False)
    scheduler_output = harness.prefill_output(req_data)
    prefill_result = runner.execute(scheduler_output)
    decode_output = harness.decode_output(
        scheduler_output, int(prefill_result.outputs["req"].data)
    )
    batch = decode_output.batch_data

    assert runner.is_decode_batch_ready(batch) is True

    # Drain the text rows while the thinker stream is still open: with no text
    # row and no thinker-done padding, the batch must not be admitted.
    while req_data.pending_text_queue:
        req_data.pending_text_queue.popleft()
    assert runner.is_decode_batch_ready(batch) is False

    # An empty feedback queue is equally disqualifying.
    req_data.thinker_chunks_done = True
    req_data.pending_feedback_queue.clear()
    assert runner.is_decode_batch_ready(batch) is False


def test_decode_raises_rather_than_running_without_its_rows(
    tiny_checkpoint: Path, sglang_runtime_context
):
    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    runner = harness.runner
    req_data = harness.build_request()
    scheduler_output = harness.prefill_output(req_data)
    prefill_result = runner.execute(scheduler_output)
    decode_output = harness.decode_output(
        scheduler_output, int(prefill_result.outputs["req"].data)
    )
    req_data.pending_feedback_queue.clear()

    with pytest.raises(RuntimeError, match="feedback and text input"):
        runner.execute(decode_output)


def test_thinker_done_padding_uses_the_projected_tts_pad_row(
    tiny_checkpoint: Path, sglang_runtime_context
):
    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    runner = harness.runner
    req_data = harness.build_request()
    scheduler_output = harness.prefill_output(req_data)
    prefill_result = runner.execute(scheduler_output)

    while req_data.pending_text_queue:
        req_data.pending_text_queue.popleft()
    assert req_data.thinker_chunks_done is True

    feedback_row = req_data.pending_feedback_queue[0].clone()
    decode_output = harness.decode_output(
        scheduler_output, int(prefill_result.outputs["req"].data)
    )
    runner.execute(decode_output)

    # The combined row the runner recorded is feedback + the TTS pad row.
    torch.testing.assert_close(
        req_data.decode_input_embeds[-1], feedback_row + req_data.tts_pad_embed
    )
    assert len(req_data.pending_text_queue) == 0


def test_codec_eos_completes_without_duplicate_emission_or_leakage(
    tiny_checkpoint: Path, sglang_runtime_context
):
    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    runner = harness.runner
    # Suppress everything but the codec EOS so the greedy argmax must pick it.
    req_data = harness.build_request(suppress_tokens=_suppress_all_but(CODEC_EOS))
    scheduler_output = harness.prefill_output(req_data)

    result = runner.execute(scheduler_output)

    messages = harness.drain_outbox()
    assert len(messages) == 1
    assert int(messages[0].data[0]) == CODEC_EOS
    assert int(result.outputs["req"].data) == CODEC_EOS
    assert runner.has_request("req") is True

    runner.on_request_finished("req", req_data)

    assert runner.has_request("req") is False
    assert runner._pending_steps == {}
    assert runner._suppress_masks == {}
    assert runner._mrope_delta == {}
    assert list(req_data.pending_feedback_queue) == []
    assert len(req_data.pending_text_queue) == 0
    assert req_data.decode_input_embeds == []
    assert harness.drain_outbox() == []


def test_abort_clears_every_per_request_talker_resource(
    tiny_checkpoint: Path, sglang_runtime_context
):
    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    runner = harness.runner
    req_data = harness.build_request()
    runner.execute(harness.prefill_output(req_data))

    assert runner.has_request("req") is True
    runner.abort_request("req")

    assert runner.has_request("req") is False
    assert runner._caches == {}
    assert runner._suppress_masks == {}
    assert runner._pending_steps == {}
    assert runner._inflight_prefills == set()


def test_mrope_positions_follow_linear_talker_semantics(
    tiny_checkpoint: Path, sglang_runtime_context
):
    from sglang_omni.models.qwen3_omni.mrope_positions import linear_mrope_positions

    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    runner = harness.runner
    req_data = harness.build_request()
    scheduler_output = harness.prefill_output(req_data)

    # The request builder attached real linear M-RoPE rows.
    attached = req_data.req.multimodal_inputs.mrope_positions
    prompt_len = int(req_data.prefill_input_embeds.shape[0])
    expected_prompt, _ = linear_mrope_positions(prompt_len)
    torch.testing.assert_close(attached.to(torch.int64), expected_prompt)

    prefill_result = runner.execute(scheduler_output)
    torch.testing.assert_close(runner.last_positions, expected_prompt.to(torch.int64))

    token = int(prefill_result.outputs["req"].data)
    for step in range(2):
        decode_output = harness.decode_output(scheduler_output, token)
        decode_result = runner.execute(decode_output)
        assert runner.last_positions.shape == (3, 1)
        assert torch.equal(
            runner.last_positions,
            torch.full((3, 1), prompt_len + step, dtype=torch.int64),
        )
        token = int(decode_result.outputs["req"].data)


def test_mlx_talker_runner_refuses_async_lookahead(
    tiny_checkpoint: Path, sglang_runtime_context
):
    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    runner = harness.runner
    req_data = harness.build_request()
    scheduler_output = harness.prefill_output(req_data)

    assert runner.lookahead_eligible(scheduler_output.batch_data) is False
    with pytest.raises(NotImplementedError, match="lookahead"):
        runner.execute_launch(scheduler_output)
    with pytest.raises(NotImplementedError, match="lookahead"):
        runner.execute_resolve(object())
    # A None handle (no step in flight) is still a no-op, never a raise.
    assert runner.execute_resolve(None) is None

    # A synchronous step must not leave any speculative queue state behind:
    # exactly one feedback row and one emitted frame, and no pending step.
    runner.execute(scheduler_output)
    assert len(req_data.pending_feedback_queue) == 1
    assert len(harness.drain_outbox()) == 1
    assert runner._pending_steps == {}
    assert runner._inflight_prefills == set()


def test_bootstrap_refuses_partial_start_for_the_mlx_talker(monkeypatch):
    from sglang_omni.models.qwen3_omni import apple_runtime, bootstrap

    monkeypatch.setattr(apple_runtime, "qwen3_omni_uses_mlx_backend", lambda: True)
    with pytest.raises(ValueError, match="partial talker start"):
        bootstrap.create_talker_scheduler(
            SimpleNamespace(),
            0,
            enable_partial_start=True,
        )


def test_talker_scheduler_binds_runner_abort_cleanup():
    from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler._abort_callback = None
    aborted: list[str] = []
    runner = SimpleNamespace(abort_request=aborted.append)

    def _fake_bind(self, model_runner):
        self._model_runner = model_runner

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(OmniScheduler, "bind_model_runner", _fake_bind)
        scheduler.bind_model_runner(runner)

    assert scheduler._abort_callback is not None
    scheduler._abort_callback("req")
    assert aborted == ["req"]


# ---------------------------------------------------------------------------
# Talker: real dense / 4-bit loading and component shard ownership


def test_dense_talker_checkpoint_loads_from_per_expert_shards(tiny_checkpoint: Path):
    loaded = load_qwen3_omni_mlx_talker(str(tiny_checkpoint))
    model = loaded["model"]

    assert isinstance(model, Qwen3OmniMlxTalker)
    assert model.num_code_groups == CODE_GROUPS
    assert model.vocab_size == TALKER_VOCAB
    # The published layout serializes one linear per expert; the loader fuses
    # them onto the quantizable SwitchLinear stacks.
    names = {name for name, _ in tree_flatten(model.parameters())}
    assert "model.layers.0.mlp.experts.gate_up_proj.weight" in names
    assert "code_predictor.model.layers.0.self_attn.q_proj.weight" in names
    assert not any(name.startswith("thinker") for name in names)


def _write_converted_component_checkpoint(root: Path, source: Path) -> Path:
    """A converted MLX export: one prefix-stripped shard per component dir."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        (source / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    official: dict[str, mx.array] = {}
    for shard in sorted(source.rglob("*.safetensors")):
        official.update(mx.load(str(shard)))

    per_component: dict[str, dict[str, mx.array]] = {
        "thinker": {},
        "talker": {},
        "code2wav": {},
    }
    for key, value in official.items():
        for component in per_component:
            prefix = f"{component}."
            if key.startswith(prefix):
                per_component[component][key[len(prefix) :]] = value
                break
    for component, weights in per_component.items():
        directory = root / component
        directory.mkdir(exist_ok=True)
        mx.save_safetensors(str(directory / "model.safetensors"), weights)
    return root


def test_talker_loader_ignores_sibling_component_shards(
    tiny_checkpoint: Path, tmp_path: Path
):
    converted = _write_converted_component_checkpoint(
        tmp_path / "converted", tiny_checkpoint
    )

    weights = read_qwen3_omni_component_weights(
        converted,
        component="talker",
        official_prefixes=("talker.",),
        local_prefixes=(
            "model.",
            "codec_head.",
            "text_projection.",
            "hidden_projection.",
            "code_predictor.",
        ),
    )

    assert weights
    # The talker's own decoder is present ...
    assert "model.layers.0.self_attn.q_proj.weight" in weights
    # ... with the talker's hidden size, not the thinker's 25-layer stack.
    assert "model.layers.24.self_attn.q_proj.weight" not in weights
    assert "codec_head.weight" in weights
    assert not any(key.startswith("visual.") for key in weights)
    assert not any(key.startswith("upsample.") for key in weights)

    loaded = load_qwen3_omni_mlx_talker(str(converted))
    assert loaded["model"].num_layers == 2


def test_thinker_loader_ignores_sibling_component_shards(
    tiny_checkpoint: Path, tmp_path: Path
):
    converted = _write_converted_component_checkpoint(
        tmp_path / "converted-thinker", tiny_checkpoint
    )

    weights = read_qwen3_omni_component_weights(
        converted,
        component="thinker",
        official_prefixes=("thinker.model.", "thinker.lm_head."),
        local_prefixes=("model.", "lm_head."),
    )

    assert "model.layers.24.self_attn.q_proj.weight" in weights
    assert "lm_head.weight" in weights
    # Neither the talker's identically named keys nor the towers leak in.
    assert "codec_head.weight" not in weights
    assert not any(key.startswith("visual.") for key in weights)


def test_thinker_reader_accepts_language_model_namespace(tmp_path: Path) -> None:
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "thinker.language_model.model.embed_tokens.weight": mx.zeros((8, 4)),
            "thinker.language_model.lm_head.weight": mx.zeros((8, 4)),
            "thinker.audio_tower.layers.0.fc1.weight": mx.zeros((4, 4)),
        },
    )

    weights = read_qwen3_omni_component_weights(
        tmp_path,
        component="thinker",
        official_prefixes=(
            "thinker.model.",
            "thinker.lm_head.",
            "thinker.language_model.model.",
            "thinker.language_model.lm_head.",
        ),
        local_prefixes=("model.", "lm_head."),
    )

    assert set(weights) == {
        "thinker.language_model.model.embed_tokens.weight",
        "thinker.language_model.lm_head.weight",
    }


def test_thinker_loader_accepts_language_model_namespace(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "mlx-community"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        (tiny_checkpoint / "config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    weights = {}
    for shard in tiny_checkpoint.rglob("*.safetensors"):
        for key, value in mx.load(str(shard)).items():
            if key.startswith(("thinker.model.", "thinker.lm_head.")):
                weights[f"thinker.language_model.{key[len('thinker.') :]}"] = value
    mx.save_safetensors(str(checkpoint / "model.safetensors"), weights)

    runner_class = make_qwen3_omni_thinker_mlx_runner_class()
    runner_kwargs = {
        "disable_radix_cache": True,
        "capture_hidden_layers": (0, 24),
        "pool_size": 2048,
    }
    actual = runner_class(model_path=str(checkpoint), **runner_kwargs).model
    expected = runner_class(model_path=str(tiny_checkpoint), **runner_kwargs).model
    actual_weights = dict(tree_flatten(actual.parameters()))
    expected_weights = dict(tree_flatten(expected.parameters()))

    assert actual_weights.keys() == expected_weights.keys()
    for key, expected_weight in expected_weights.items():
        np.testing.assert_array_equal(
            np.asarray(actual_weights[key]), np.asarray(expected_weight)
        )


@pytest.mark.parametrize(
    ("component", "prefix"),
    [
        ("vision", "thinker.vision_tower."),
        ("audio", "thinker.audio_tower."),
        ("code2wav", "code2wav."),
    ],
)
def test_component_reader_keeps_only_owned_namespace(
    tmp_path: Path, component: str, prefix: str
) -> None:
    write_indexed_mlx_checkpoint(
        tmp_path,
        {
            f"{prefix}owned.weight": mx.ones((2, 2)),
            "talker.model.foreign.weight": mx.zeros((2, 2)),
        },
    )

    weights = read_qwen3_omni_component_weights(
        tmp_path,
        component=component,
        official_prefixes=(prefix,),
        local_prefixes=("owned.",),
    )

    assert list(weights) == [f"{prefix}owned.weight"]


def test_component_reader_ignores_directories_named_safetensors(
    tmp_path: Path,
) -> None:
    (tmp_path / ".downloads" / "partial.safetensors").mkdir(parents=True)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "thinker.language_model.model.embed_tokens.weight": mx.zeros((8, 4)),
        },
    )

    weights = read_qwen3_omni_component_weights(
        tmp_path,
        component="thinker",
        official_prefixes=("thinker.language_model.model.",),
        local_prefixes=("model.",),
    )

    assert set(weights) == {
        "thinker.language_model.model.embed_tokens.weight",
    }


def _write_quantized_talker_checkpoint(
    directory: Path, source: Path, *, bits: int = 4, group_size: int = 32
) -> Path:
    """A converted 4-bit talker export with real weights/scales/biases."""

    directory.mkdir(parents=True, exist_ok=True)
    dense = load_qwen3_omni_mlx_talker(str(source))["model"]
    nn.quantize(dense, group_size=group_size, bits=bits)
    weights = {
        f"talker.{name}": value for name, value in tree_flatten(dense.parameters())
    }
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["quantization"] = {"bits": bits, "group_size": group_size}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def test_converted_4bit_talker_checkpoint_loads_and_steps(
    tiny_checkpoint: Path, tmp_path: Path
):
    directory = _write_quantized_talker_checkpoint(
        tmp_path / "talker-q4", tiny_checkpoint
    )
    loaded = load_qwen3_omni_mlx_talker(str(directory))
    model = loaded["model"]

    assert loaded["quantization"] == QuantizationConfig(bits=4, group_size=32)
    experts = model.model.layers[0].mlp.experts
    assert hasattr(experts.gate_up_proj, "scales")
    assert hasattr(experts.gate_up_proj, "biases")
    assert model.codec_head.bits == 4

    hidden = model.config.text_config.hidden_size
    rng = np.random.default_rng(3)
    rows = mx.array((0.1 * rng.standard_normal((1, 5, hidden))).astype(np.float32))
    positions = mx.broadcast_to(mx.arange(5, dtype=mx.int32)[None, :], (3, 5))
    step = model.prefill(
        rows, mrope_positions=positions, input_embeddings_are_projected=True
    )
    codes = np.asarray(step.codes)
    assert codes.shape == (1, CODE_GROUPS)
    assert np.isfinite(np.asarray(step.feedback)).all()


# ---------------------------------------------------------------------------
# Talker: native MLX prefill construction


def test_native_prefill_builder_supplies_the_request_builder_instead_of_the_stub(
    tiny_checkpoint: Path, sglang_runtime_context
):
    """No request-builder path may dereference SGLang's ``_DummyModel``."""

    from sglang.srt.hardware_backend.mlx.model_runner_stub import _DummyModel

    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    req_data = harness.build_request()

    assert isinstance(harness.worker.model_runner.model, _DummyModel)
    assert not hasattr(harness.worker, "talker_prefill_shim")
    assert isinstance(
        harness.worker.mlx_talker_prefill_builder,
        Qwen3OmniMlxTalkerPrefillBuilder,
    )
    assert req_data.input_embeds_are_projected is True
    assert req_data.prefill_input_embeds.dtype == torch.float32
    assert req_data.prefill_input_embeds.device == torch.device("cpu")
    assert req_data.prefill_input_embeds.shape[1] == (
        harness.mlx_talker.config.text_config.hidden_size
    )
    assert isinstance(req_data.pending_text_queue, PendingTextTensorQueue)
    assert req_data.tts_pad_embed is not None
    assert req_data.tts_pad_embed.dtype == torch.float32


def test_talker_runner_factory_returns_the_model_runner_class():
    assert make_qwen3_omni_talker_mlx_runner_class() is Qwen3OmniMlxTalkerModelRunner


def test_bootstrap_wires_the_native_builder_and_the_mlx_talker_runner(
    monkeypatch, tiny_checkpoint: Path
):
    """``create_talker_scheduler`` must never hand ``_DummyModel`` to the builder."""

    from sglang.srt.utils import hf_transformers_utils

    from sglang_omni.models.qwen3_omni import apple_runtime, bootstrap
    from sglang_omni.models.qwen3_omni import request_builders as qwen_request_builders
    from sglang_omni.models.qwen3_omni import talker_scheduler as qwen_talker_scheduler
    from sglang_omni.models.qwen3_omni.mlx import runner as mlx_runner
    from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
    from sglang_omni.scheduling import sglang_backend

    loaded = load_qwen3_omni_mlx_talker(str(tiny_checkpoint))
    root = build_tiny_config()
    talker_config = root.talker_config
    prefill_builder = Qwen3OmniMlxTalkerPrefillBuilder.from_talker(
        loaded["model"],
        model_path=str(tiny_checkpoint),
        special_token_ids={
            "audio_token_id": root.thinker_config.audio_token_id,
            "image_token_id": root.thinker_config.image_token_id,
            "video_token_id": root.thinker_config.video_token_id,
            "tts_bos_token_id": root.tts_bos_token_id,
            "tts_eos_token_id": root.tts_eos_token_id,
            "tts_pad_token_id": root.tts_pad_token_id,
            "im_start_token_id": root.im_start_token_id,
            "im_end_token_id": root.im_end_token_id,
            "system_token_id": root.system_token_id,
            "user_token_id": root.user_token_id,
            "assistant_token_id": root.assistant_token_id,
            "codec_bos_id": talker_config.codec_bos_id,
            "codec_nothink_id": talker_config.codec_nothink_id,
            "codec_think_bos_id": talker_config.codec_think_bos_id,
            "codec_think_eos_id": talker_config.codec_think_eos_id,
            "codec_pad_id": talker_config.codec_pad_id,
        },
        speaker_map=dict(OFFICIAL_SPEAKER_IDS),
    )
    worker = _FakeMlxTalkerWorker(loaded["model"], prefill_builder)
    hf_config = root
    model_config = SimpleNamespace(
        hf_config=hf_config,
        model_path=str(tiny_checkpoint),
        vocab_size=152064,
    )
    worker.model_runner.model_config = model_config

    seen: dict[str, Any] = {}

    monkeypatch.setattr(apple_runtime, "qwen3_omni_uses_mlx_backend", lambda: True)
    monkeypatch.setattr(
        qwen_talker_scheduler, "configure_talker_server_args", lambda *a, **k: False
    )
    monkeypatch.setattr(
        scheduling_bootstrap,
        "create_sglang_infrastructure",
        lambda *a, **k: (worker, object(), object(), object(), model_config),
    )
    monkeypatch.setattr(
        hf_transformers_utils, "get_tokenizer", lambda *a, **k: FakeQwenTokenizer()
    )

    def _fake_adapters(**kwargs):
        seen["prefill_builder"] = kwargs["prefill_builder"]
        return (object(), object(), object(), object())

    monkeypatch.setattr(
        qwen_request_builders, "make_talker_scheduler_adapters", _fake_adapters
    )

    def _fake_output_processor(**kwargs):
        seen["output_processor_model"] = kwargs["model"]
        return object()

    monkeypatch.setattr(sglang_backend, "SGLangOutputProcessor", _fake_output_processor)

    class _FakeScheduler:
        outbox = queue.Queue()

        def __init__(self, **kwargs):
            seen["scheduler_kwargs"] = kwargs

        def bind_model_runner(self, model_runner):
            seen["model_runner"] = model_runner

    monkeypatch.setattr(qwen_talker_scheduler, "QwenTalkerScheduler", _FakeScheduler)

    def _fake_build_runner(**kwargs):
        seen["runner_kwargs"] = kwargs
        return mlx_runner.Qwen3OmniMlxTalkerModelRunner(
            kwargs["tp_worker"],
            SGLangOutputProcessor(capture_hidden=False, model=None),
            kwargs["outbox"],
            mlx_talker=kwargs["mlx_talker"],
            feedback_enabled=kwargs["feedback_enabled"],
        )

    monkeypatch.setattr(
        mlx_runner, "build_qwen3_omni_talker_mlx_runner", _fake_build_runner
    )

    bootstrap.create_talker_scheduler(SimpleNamespace(), 0)

    assert seen["prefill_builder"] is prefill_builder
    assert seen["output_processor_model"] is None
    assert isinstance(seen["model_runner"], Qwen3OmniMlxTalkerModelRunner)
    assert seen["model_runner"].mlx_model is loaded["model"]
    assert seen["runner_kwargs"]["tp_worker"] is worker


def test_streaming_requests_mark_the_code2wav_message(
    tiny_checkpoint: Path, sglang_runtime_context
):
    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    req_data = harness.build_request(stream=True)

    harness.runner.execute(harness.prefill_output(req_data))

    (message,) = harness.drain_outbox()
    assert message.metadata == {"stream": True}


def test_abort_drops_a_computed_but_unemitted_step(
    tiny_checkpoint: Path, sglang_runtime_context
):
    """In-flight prefill state must not outlive an abort."""

    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    runner = harness.runner
    req_data = harness.build_request()
    scheduler_output = harness.prefill_output(req_data)

    runner.custom_prefill_forward(
        None, scheduler_output.batch_data, scheduler_output.requests
    )

    assert set(runner._pending_steps) == {"req"}
    assert runner.has_request("req") is True
    assert harness.drain_outbox() == []

    runner.abort_request("req")

    assert runner._pending_steps == {}
    assert runner._caches == {}
    assert runner._suppress_masks == {}
    assert list(req_data.pending_feedback_queue) == []


def test_talker_runner_refuses_multi_request_batches(
    tiny_checkpoint: Path, sglang_runtime_context
):
    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    req_data = harness.build_request()
    scheduler_output = harness.prefill_output(req_data)
    doubled = list(scheduler_output.requests) * 2

    with pytest.raises(RuntimeError, match="one request at a time"):
        harness.runner.custom_prefill_forward(
            None, scheduler_output.batch_data, doubled
        )


def test_talker_prefill_refuses_a_radix_prefix(
    tiny_checkpoint: Path, sglang_runtime_context
):
    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    req_data = harness.build_request()
    scheduler_output = harness.prefill_output(req_data)
    req_data.req.prefix_indices = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(NotImplementedError, match="radix prefix"):
        harness.runner.custom_prefill_forward(
            None, scheduler_output.batch_data, scheduler_output.requests
        )


def test_talker_prefill_refuses_unprojected_prompt_rows(
    tiny_checkpoint: Path, sglang_runtime_context
):
    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    req_data = harness.build_request()
    scheduler_output = harness.prefill_output(req_data)
    req_data.input_embeds_are_projected = False

    with pytest.raises(RuntimeError, match="talker-space rows"):
        harness.runner.custom_prefill_forward(
            None, scheduler_output.batch_data, scheduler_output.requests
        )


def test_decode_rejects_a_non_cpu_float32_text_row(
    tiny_checkpoint: Path, sglang_runtime_context
):
    """Row ownership is asserted before the two rows are combined."""

    harness = build_tiny_mlx_talker_runner(checkpoint=tiny_checkpoint)
    runner = harness.runner
    req_data = harness.build_request()
    scheduler_output = harness.prefill_output(req_data)
    prefill_result = runner.execute(scheduler_output)
    decode_output = harness.decode_output(
        scheduler_output, int(prefill_result.outputs["req"].data)
    )
    hidden = harness.mlx_talker.config.text_config.hidden_size
    req_data.pending_text_queue = PendingTextTensorQueue.from_tensor(
        torch.zeros((2, hidden), dtype=torch.float64)
    )

    with pytest.raises(RuntimeError, match="device/dtype"):
        runner.execute(decode_output)
