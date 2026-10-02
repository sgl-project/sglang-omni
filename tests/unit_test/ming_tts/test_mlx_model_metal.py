# SPDX-License-Identifier: Apache-2.0
"""Run on a Metal-capable terminal; these tests execute the real MLX modules."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

if not torch.backends.mps.is_available():
    pytest.skip("Requires an accessible Apple Metal device", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from sglang_omni.models.ming_omni.talker.talker_module.aggregator import (  # noqa: E402 - Requires the backend checks above.
    Aggregator as TorchAggregator,
)
from sglang_omni.models.ming_omni.talker.talker_module.dit import DiT as TorchDiT  # noqa: E402 - Requires the backend checks above.
from sglang_omni.models.ming_tts.mlx.backbone import (  # noqa: E402
    BailingMoeSparseMoeBlock,
)
from sglang_omni.models.ming_tts.mlx.config import ModelConfig, TextConfig  # noqa: E402
from sglang_omni.models.ming_tts.mlx.flow_matching import (  # noqa: E402
    build_cfm_timesteps,
)
from sglang_omni.models.ming_tts.mlx.model import MingTTSModel  # noqa: E402
from sglang_omni.models.ming_tts.mlx.runner import MingTTSMlxRunner  # noqa: E402


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig.from_dict(
        dict(
            llm_config=dict(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=24,
                moe_intermediate_size=12,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                num_experts=4,
                num_experts_per_tok=2,
                num_shared_experts=1,
                first_k_dense_replace=1,
                multi_gate=True,
                max_position_embeddings=64,
                rope_scaling={"type": "3D", "factor": None, "mrope_section": [1, 1, 2]},
            ),
            ditar_config=dict(
                hidden_size=16,
                depth=2,
                num_heads=2,
                mlp_ratio=2.0,
                patch_size=2,
                history_patch_size=4,
            ),
            aggregator_config=dict(hidden_size=16, depth=1, num_heads=2, mlp_ratio=2.0),
            audio_tokenizer_config={"enc_kwargs": {"latent_dim": 4}},
        )
    )


@pytest.fixture
def model(config: ModelConfig) -> MingTTSModel:
    mx.set_default_device(mx.gpu)
    mx.random.seed(1967)
    result = MingTTSModel(config)
    result.eval()
    mx.eval(result.parameters())
    return result


def as_torch(x: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(x.astype(mx.float32)))


def assert_close(
    actual: mx.array,
    expected: torch.Tensor,
    atol: float = 2e-5,
    rtol: float = 2e-4,
) -> None:
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        expected.detach().float().numpy(),
        atol=atol,
        rtol=rtol,
    )


def torch_backbone(
    weights: dict[str, torch.Tensor],
    cfg: TextConfig,
    *,
    input_ids: torch.Tensor | None = None,
    embeds: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unfused Torch equations; no SGLang paged attention or MLX operations."""

    def linear(x: torch.Tensor, prefix: str) -> torch.Tensor:
        return F.linear(x, weights[prefix + ".weight"], weights.get(prefix + ".bias"))

    def norm(x: torch.Tensor, prefix: str) -> torch.Tensor:
        return F.rms_norm(
            x, (x.shape[-1],), weights[prefix + ".weight"], cfg.rms_norm_eps
        )

    def mlp(x: torch.Tensor, prefix: str) -> torch.Tensor:
        return linear(
            F.silu(linear(x, prefix + ".gate_proj")) * linear(x, prefix + ".up_proj"),
            prefix + ".down_proj",
        )

    x = (
        F.embedding(input_ids, weights["word_embeddings.weight"])
        if embeds is None
        else embeds
    )
    batch, length, _ = x.shape
    if positions is None:
        positions = torch.arange(length).expand(3, batch, length)
    inv = 1.0 / (
        cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim)
    )
    phases = positions[..., None].float() * inv
    chunks = torch.split(phases, cfg.mrope_section, dim=-1)
    phase = torch.cat([chunk[i] for i, chunk in enumerate(chunks)], dim=-1)[:, None]

    def rope(v: torch.Tensor) -> torch.Tensor:
        left, right = v.chunk(2, dim=-1)
        return torch.cat(
            (
                left * phase.cos() - right * phase.sin(),
                right * phase.cos() + left * phase.sin(),
            ),
            dim=-1,
        )

    for i in range(cfg.num_hidden_layers):
        prefix = f"layers.{i}"
        h = norm(x, prefix + ".input_layernorm")
        q, k, v = linear(h, prefix + ".attention.query_key_value").split(
            (
                cfg.num_attention_heads * cfg.head_dim,
                cfg.num_key_value_heads * cfg.head_dim,
                cfg.num_key_value_heads * cfg.head_dim,
            ),
            dim=-1,
        )
        q = rope(
            q.reshape(batch, length, cfg.num_attention_heads, cfg.head_dim).transpose(
                1, 2
            )
        )
        k = rope(
            k.reshape(batch, length, cfg.num_key_value_heads, cfg.head_dim).transpose(
                1, 2
            )
        )
        v = v.reshape(batch, length, cfg.num_key_value_heads, cfg.head_dim).transpose(
            1, 2
        )
        repeats = cfg.num_attention_heads // cfg.num_key_value_heads
        attn = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(repeats, 1),
            v.repeat_interleave(repeats, 1),
            is_causal=True,
        )
        x = x + linear(
            attn.transpose(1, 2).reshape(batch, length, -1), prefix + ".attention.dense"
        )
        h = norm(x, prefix + ".post_attention_layernorm")
        if i < cfg.first_k_dense_replace:
            out = mlp(h, prefix + ".mlp")
        else:
            scores = linear(h, prefix + ".mlp.gate").float().softmax(-1)
            probs, indices = scores.topk(cfg.num_experts_per_tok, dim=-1)
            if cfg.norm_topk_prob:
                probs = probs / probs.sum(-1, keepdim=True)
            probs = probs * cfg.routed_scaling_factor
            out = torch.zeros_like(h)
            for expert in range(cfg.num_experts):

                def expert_linear(value: torch.Tensor, projection: str) -> torch.Tensor:
                    return F.linear(
                        value,
                        weights[f"{prefix}.mlp.experts.{projection}.weight"][expert],
                    )

                expert_out = expert_linear(
                    F.silu(expert_linear(h, "gate_proj")) * expert_linear(h, "up_proj"),
                    "down_proj",
                )
                weight = (probs * (indices == expert)).sum(-1, keepdim=True)
                out = out + expert_out * weight
            if cfg.num_shared_experts:
                out = out + mlp(h, prefix + ".mlp.shared_experts")
        x = x + out
    return norm(x, "norm")


@pytest.mark.parametrize("distinct_axes", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_backbone_torch_and_cached_decode(
    model: MingTTSModel, distinct_axes: bool, batch: int
) -> None:
    ids = mx.array(np.arange(batch * 7).reshape(batch, 7) % 32)
    positions = np.broadcast_to(np.arange(7), (3, batch, 7)).copy()
    if distinct_axes:
        positions[1] += 5
        positions[2] += 11
    weights = {k: as_torch(v) for k, v in tree_flatten(model.model.parameters())}
    expected = torch_backbone(
        weights,
        model.config.llm_config,
        input_ids=torch.tensor(np.array(ids)).long(),
        positions=torch.tensor(positions),
    )
    full = model(ids, positions=mx.array(positions))
    assert_close(full, expected)
    cache = model.make_cache()
    prefill = model(ids[:, :4], positions=mx.array(positions[:, :, :4]), cache=cache)
    decode = model(ids[:, 4:5], positions=mx.array(positions[:, :, 4:5]), cache=cache)
    chunk = model(ids[:, 5:], positions=mx.array(positions[:, :, 5:]), cache=cache)
    assert_close(mx.concatenate((prefill, decode, chunk), axis=1), expected)
    assert all(c.offset == 7 for c in cache)
    automatic = model(ids)
    explicit = model(
        ids, positions=mx.array(np.broadcast_to(np.arange(7), (3, batch, 7)))
    )
    assert_close(automatic, as_torch(explicit))


@pytest.mark.parametrize(
    "top_k,renormalize", [(1, True), (2, True), (2, False), (4, True)]
)
def test_router_topk(model: MingTTSModel, top_k: int, renormalize: bool) -> None:
    cfg = replace(
        model.config.llm_config,
        num_experts_per_tok=top_k,
        norm_topk_prob=renormalize,
        routed_scaling_factor=1.3,
    )
    block = BailingMoeSparseMoeBlock(cfg)
    x = mx.random.normal((2, 9, cfg.hidden_size))
    indices, scores = block.route(x)
    logits = F.linear(as_torch(x), as_torch(block.gate.weight))
    expected, expected_indices = logits.softmax(-1).topk(top_k, dim=-1)
    if renormalize:
        expected = expected / expected.sum(-1, keepdim=True)
    assert_close(mx.sort(scores, axis=-1), (expected * 1.3).sort(-1).values)
    np.testing.assert_array_equal(
        np.sort(np.array(indices), axis=-1), expected_indices.sort(-1).values.numpy()
    )


def load_acoustic_reference(mlx_model: nn.Module, torch_model: torch.nn.Module) -> None:
    torch.manual_seed(1964)
    state = torch_model.state_dict()
    with torch.no_grad():
        for key, value in state.items():
            if key.endswith("inv_freq"):
                continue
            # Note (altale): Nonzero final layers avoid comparing all-zero outputs.
            value.copy_(
                torch.randn_like(value) * 0.08
                + (1.0 if "norm" in key and key.endswith("weight") else 0.0)
            )
    weights = {}
    for key, value in state.items():
        if key.endswith("inv_freq"):
            continue
        key = key.replace(".mlp.ff.0.0.", ".mlp.fc1.").replace(
            ".mlp.ff.2.", ".mlp.fc2."
        )
        weights[key] = mx.array(value.detach().numpy())
    mlx_model.load_weights(list(weights.items()), strict=True)
    mlx_model.eval()
    torch_model.eval()


@pytest.fixture
def acoustic_pair(model: MingTTSModel) -> tuple[TorchDiT, TorchAggregator]:
    from sglang_omni.models.ming_omni.talker.talker_module.execution import (
        TalkerExecutionConfig,
    )

    execution = TalkerExecutionConfig(attn_backend="torch")
    dit = TorchDiT(
        in_channels=4,
        hidden_size=16,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        llm_cond_dim=16,
        execution_config=execution,
    )
    agg = TorchAggregator(
        in_channels=4,
        hidden_size=16,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        llm_input_dim=16,
        execution_config=execution,
    )
    load_acoustic_reference(model.flowloss.cfm.model, dit)
    load_acoustic_reference(model.linear_proj_audio, agg)
    return dit, agg


@pytest.mark.parametrize("parameter", ["cfg_scale", "sigma", "temperature"])
def test_cfm_rejects_parameter_count_mismatch(
    model: MingTTSModel, parameter: str
) -> None:
    with pytest.raises(ValueError):
        model.flowloss.cfm.sample(
            noise=mx.zeros((2, 4, 2)),
            c=mx.zeros((2, 1, 16)),
            latent_history=mx.zeros((2, 4, 4)),
            timesteps=build_cfm_timesteps(1),
            sde_random=mx.zeros((0, 2, 2, 4)),
            **{parameter: mx.array([0.2, 0.5, 0.8])},
        )


@pytest.mark.parametrize(
    "steps,temperature,shape",
    [
        (1, 0.0, None),
        (4, 0.8, ()),
        (5, 0.8, (1,)),
        (10, 0.0, (1, 1, 1)),
        (10, 0.8, (2,)),
        (10, 0.8, (2, 1, 1)),
    ],
)
@torch.no_grad()
def test_acoustic_tail_torch_parity(
    model: MingTTSModel,
    acoustic_pair: tuple[TorchDiT, TorchAggregator],
    steps: int,
    temperature: float,
    shape: tuple[int, ...] | None,
) -> None:
    from sglang_omni.models.ming_tts.flow_matching import CFM as TorchCFM

    dit, agg = acoustic_pair
    noise = mx.random.normal((2, 4, 2))
    history = mx.random.normal((2, 4, 4))
    cond = mx.random.normal((2, 1, 16))
    random = mx.random.normal((steps - 1, 2, 2, 4))
    t = build_cfm_timesteps(steps)
    cfg = mx.array([0.0, 2.0])
    if shape is None:
        value = temperature
        torch_value = temperature
    else:
        values = [0.0, temperature] if shape in ((2,), (2, 1, 1)) else [temperature]
        value = mx.array(values).reshape(shape)
        torch_value = torch.tensor(values).reshape(shape)
    expected = TorchCFM(dit).sample(
        as_torch(noise),
        as_torch(cond),
        as_torch(history),
        as_torch(t),
        as_torch(random),
        cfg_scale=as_torch(cfg),
        sigma=0.25,
        temperature=torch_value,
    )
    actual = model.compute_tail_step(
        cond,
        history,
        noise=noise,
        timesteps=t,
        sde_random=random,
        cfg=cfg,
        temperature=value,
    )
    assert_close(actual.sampled, expected)
    assert_close(actual.feedback_embeddings, agg(expected).reshape(2, -1))


def test_official_weight_mapping(model: MingTTSModel) -> None:
    canonical = dict(tree_flatten(model.parameters()))
    official = {}
    for key, value in canonical.items():
        if ".mlp.experts." in key:
            prefix, suffix = key.split(".mlp.experts.")
            for e in range(value.shape[0]):
                official[f"model.{prefix}.mlp.experts.{e}.{suffix}"] = value[e]
            continue
        if key.startswith("model."):
            key = "model." + key
        key = key.replace(".mlp.fc1.", ".mlp.ff.0.0.").replace(
            ".mlp.fc2.", ".mlp.ff.2."
        )
        official[key] = value
    official["audio.unloaded.weight"] = mx.zeros((1,))
    official["model.lm_head.weight"] = mx.zeros((1,))
    official["linear_proj_audio.rotary_embed.inv_freq"] = mx.zeros((4,))
    mapped = model.sanitize(official)
    assert set(mapped) == set(canonical)
    for key in canonical:
        np.testing.assert_array_equal(np.array(mapped[key]), np.array(canonical[key]))
    model.load_weights(list(mapped.items()), strict=True)


@pytest.mark.parametrize("frames", [2, 4, 8])
def test_reference_injection_history_and_release(
    model: MingTTSModel, frames: int
) -> None:
    runner = MingTTSMlxRunner(model)
    reference = mx.arange(frames * 4, dtype=mx.float32).reshape(frames, 4) / 10
    ids = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
    speaker = mx.ones((1, 192)) * 0.01
    runner.start(
        "ref",
        ids,
        max_steps=3,
        reference_latents=reference,
        reference_start=2,
        speaker_embedding=speaker,
        speaker_positions=[0],
    )
    state = runner.states["ref"]
    expected = np.zeros((1, 4, 4), dtype=np.float32)
    tail = np.array(reference)[-4:]
    expected[:, -len(tail) :] = tail
    np.testing.assert_array_equal(np.array(state.history), expected)
    embeds = model.model.word_embeddings(ids[None])
    embeds[0, 0] = model.spk_head(speaker)[0]
    embeds[0, 2 : 2 + frames // 2] = model.project_reference_latents(reference)
    assert_close(state.hidden, as_torch(model(inputs_embeds=embeds)[:, -1:]))
    runner.release("ref")
    runner.release("ref")
    assert runner.states == {}


def test_real_tiny_closed_loop_matches_torch(
    model: MingTTSModel, acoustic_pair: tuple[TorchDiT, TorchAggregator]
) -> None:
    from sglang_omni.models.ming_tts.flow_matching import CFM as TorchCFM

    dit, agg = acoustic_pair
    runner = MingTTSMlxRunner(model)
    ids = mx.array([1, 2, 3])
    runner.start("loop", ids, max_steps=3)
    embeds = as_torch(model.model.word_embeddings(ids[None]))
    backbone_weights = {
        k: as_torch(v) for k, v in tree_flatten(model.model.parameters())
    }
    history = torch.zeros((1, 4, 4))
    for step in range(3):
        noise = mx.random.normal((1, 4, 2))
        random = mx.random.normal((3, 1, 2, 4))
        times = build_cfm_timesteps(4)
        with torch.no_grad():
            hidden = torch_backbone(
                backbone_weights, model.config.llm_config, embeds=embeds
            )[:, -1:]
            patch = TorchCFM(dit).sample(
                as_torch(noise),
                hidden,
                history,
                as_torch(times),
                as_torch(random),
                cfg_scale=2.0,
                sigma=0.25,
                temperature=0.7,
            )
            feedback = agg(patch)
            stop = F.linear(
                hidden, as_torch(model.stop_head.weight), as_torch(model.stop_head.bias)
            ).softmax(-1)[0, 0, 1]
        actual = runner.step(
            "loop", noise=noise, timesteps=times, sde_random=random, temperature=0.7
        )
        assert_close(actual.latent, patch[0], atol=5e-5, rtol=3e-4)
        assert abs(actual.stop_prob - stop.item()) < 2e-5
        assert actual.finish_reason == ("length" if step == 2 else None)
        embeds = torch.cat((embeds, feedback), dim=1)
        history = torch.cat((history, patch), dim=1)[:, -4:]
    assert runner.states == {}
    runner.start("next", ids, max_steps=1)
    assert runner.states["next"].steps == 0
    assert runner.states["next"].cache[0].offset == 3
    runner.release("next")


def test_first_inference_on_scheduler_thread(model: MingTTSModel) -> None:
    runner = MingTTSMlxRunner(model)
    stream = mx.new_thread_local_stream(mx.gpu)

    def generate() -> np.ndarray:
        with mx.stream(stream):
            runner.start("thread", mx.array([1, 2, 3]), max_steps=2)
            patches = []
            for step in range(2):
                result = runner.step(
                    "thread",
                    noise=mx.zeros((1, 4, 2)),
                    timesteps=build_cfm_timesteps(1),
                    sde_random=mx.zeros((0, 1, 2, 4)),
                )
                assert result.finish_reason == ("length" if step == 1 else None)
                patches.append(np.array(result.latent))
            assert runner.states == {}
            return np.stack(patches)

    # Note (altale): Run on the worker first to expose unevaluated initialization arrays.
    with ThreadPoolExecutor(max_workers=1) as executor:
        actual = executor.submit(generate).result(timeout=60)
    expected = generate()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("quantization", [None, "mlx_q4", "mlx_q8"])
def test_strict_checkpoint_load_and_backbone_quantization(
    config: ModelConfig, tmp_path: Path, quantization: str | None
) -> None:
    from sglang_omni.models.ming_tts.mlx.loading import load_ming_tts_model

    text = replace(
        config.llm_config,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    cfg = replace(config, llm_config=text)
    source = MingTTSModel(cfg)
    weights = dict(tree_flatten(source.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(asdict(cfg)))
    loaded = load_ming_tts_model(tmp_path, quantization=quantization)
    assert (
        loaded.model.layers[1].mlp.gate.weight.dtype
        == source.model.layers[1].mlp.gate.weight.dtype
    )
    assert not hasattr(loaded.stop_head, "scales")
    assert not hasattr(loaded.model.word_embeddings, "scales")
    assert not hasattr(loaded.flowloss.cfm.model.x_embedder, "scales")
    assert hasattr(loaded.model.layers[0].attention.query_key_value, "scales") == (
        quantization is not None
    )
    assert hasattr(loaded.model.layers[1].mlp.experts.gate_proj, "scales") == (
        quantization is not None
    )
    output = loaded(mx.array([[1, 2, 3]]))
    assert output.shape == (1, 3, 64)
    assert bool(mx.all(mx.isfinite(output)).item())
    if quantization is None:
        np.testing.assert_array_equal(
            np.array(output), np.array(source(mx.array([[1, 2, 3]])))
        )


@pytest.mark.parametrize("problem", ["missing", "unexpected", "prequantized"])
def test_checkpoint_loading_rejects_incomplete_or_unsupported_weights(
    model: MingTTSModel, tmp_path: Path, problem: str
) -> None:
    from sglang_omni.models.ming_tts.mlx.loading import load_ming_tts_model

    raw = asdict(model.config)
    weights = dict(tree_flatten(model.parameters()))
    if problem == "missing":
        weights.pop("stop_head.weight")
    elif problem == "unexpected":
        weights["unknown.weight"] = mx.zeros((1,))
    else:
        raw["quantization"] = {"bits": 4, "group_size": 64}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_ming_tts_model(tmp_path)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("stop", [False, True])
def test_scheduler_runner_control_tokens_and_cleanup(
    model: MingTTSModel, streaming: bool, stop: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sglang.srt.hardware_backend.mlx.kv_cache import ContiguousAttentionKVCache

    from sglang_omni.models.ming_tts.mlx.worker import MingTTSMlxModelRunner
    from sglang_omni.models.ming_tts.payload_types import MingTTSState

    model.stop_head.weight = mx.zeros_like(model.stop_head.weight)
    model.stop_head.bias = mx.array([-20.0, 20.0] if stop else [20.0, -20.0])
    backend = MingTTSMlxRunner(
        model,
        cache_factory=lambda: [
            ContiguousAttentionKVCache(max_seq_len=64) for _ in model.model.layers
        ],
    )
    worker = SimpleNamespace(
        gpu_id=0, model_runner=SimpleNamespace(model=None), _mlx_runner=backend
    )
    runner = MingTTSMlxModelRunner(worker, SimpleNamespace(_capture_hidden=False))
    data = SimpleNamespace(
        state=MingTTSState(input_ids=[1, 2, 3]),
        input_ids=torch.tensor([1, 2, 3]),
        max_new_tokens=6,
        generation_steps=0,
        is_streaming=streaming,
        audio_patch_token_id=3,
        audio_eos_token_id=4,
        stop_step=None,
        generated_latents=None,
        pending_stream_patch=None,
    )
    request = SimpleNamespace(request_id="first", data=data)
    result = runner.custom_prefill_forward(None, None, [request])
    assert result.next_token_ids.tolist() == [3]
    assert backend.states["first"].cache[0].offset == 3
    for step in range(1, 5 if stop else 6):
        assert result.next_token_ids.tolist() == [3]
        assert "first" in backend.states
        data.generation_steps = step
        result = runner.custom_decode_forward(None, None, [request])
    assert result.next_token_ids.tolist() == [4 if stop else 3]
    assert backend.states == {}
    assert data.stop_step == (4 if stop else None)
    if streaming:
        assert data.pending_stream_patch.is_last
        assert data.generated_latents is None
    else:
        assert data.generated_latents.shape == (5 if stop else 6, 2, 4)
    runner.reset_request("first")
    assert runner.generated_latents == {}
    request.request_id = "second"
    runner.custom_prefill_forward(None, None, [request])
    runner.reset_request("second")
    runner.reset_request("second")
    assert backend.states == {}
    assert runner.generated_latents == {}

    request.request_id = "failure"
    runner.custom_prefill_forward(None, None, [request])

    def fail(
        hidden_states: mx.array,
        latent_history: mx.array,
        *,
        noise: mx.array,
        timesteps: mx.array,
        sde_random: mx.array,
        cfg: float | mx.array,
        sigma: float | mx.array,
        temperature: float | mx.array,
    ) -> None:
        raise RuntimeError("synthetic acoustic failure")

    monkeypatch.setattr(model, "compute_tail_step", fail)
    with pytest.raises(RuntimeError, match="synthetic acoustic"):
        runner.custom_decode_forward(None, None, [request])
    assert backend.states == {}
    assert runner.generated_latents == {}
