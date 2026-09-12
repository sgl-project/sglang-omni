# SPDX-License-Identifier: Apache-2.0
"""Real MLX parity and lifecycle tests using tiny local Higgs weights."""
import pytest
import torch

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from sglang.srt.utils.tensor_bridge import mlx_to_torch, torch_to_mlx

from sglang_omni.models.higgs_tts.mlx.model import load_mlx_language_model
from sglang_omni.models.higgs_tts.mlx.scheduler_runner import HiggsMlxModelRunner
from tests.unit_test.higgs_tts.test_torch_mps_runner import (  # noqa: F401
    checkpoint,
    runner,
)


def test_strict_mapping_and_forward_parity(checkpoint):
    path, _, source, _ = checkpoint
    model = load_mlx_language_model(str(path), dtype=mx.float32)
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = source.model(ids).last_hidden_state
    actual = mlx_to_torch(model(torch_to_mlx(ids)), device="cpu")
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    from safetensors.torch import save_file

    save_file({"body.norm.weight": torch.ones(32)}, str(path / "model.safetensors"))
    with pytest.raises(ValueError):
        load_mlx_language_model(str(path))


@pytest.fixture
def mlx_runner(runner, checkpoint):
    runner.model.mlx_language_model = load_mlx_language_model(
        str(checkpoint[0]), dtype=mx.float32
    )
    from types import SimpleNamespace

    worker = SimpleNamespace(gpu_id=0, model_runner=SimpleNamespace(model=runner.model))
    return HiggsMlxModelRunner(worker, None)


@torch.inference_mode()
def test_cached_decode_and_request_reuse(mlx_runner):
    runner = mlx_runner
    embeds = runner.model.backbone.model.embed_tokens(
        torch.tensor([1, 2, 3], device="mps")
    )
    expected = runner.model.backbone.model(
        inputs_embeds=embeds[None]
    ).last_hidden_state[:, -1]
    runner._forward("one", embeds[:2], prefill=True)
    actual = runner._forward("one", embeds[2:], prefill=False)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    assert runner._past_key_values["one"][0].offset == 3
    runner.model.acquire_row("one")
    runner.reset_request("one")
    assert not runner._past_key_values and not runner.model._rid_to_row
    with pytest.raises(RuntimeError, match="no KV cache"):
        runner._forward("one", embeds[2:], prefill=False)
    torch.testing.assert_close(
        runner._forward("one", embeds, prefill=True), expected, atol=2e-4, rtol=2e-4
    )


def test_mlx_failure_cleans_state(mlx_runner):
    r = mlx_runner
    r.model.acquire_row("bad")
    with pytest.raises(Exception):
        r._forward("bad", torch.zeros(1, 7, device="mps"), prefill=True)
    assert not r._past_key_values and not r.model._rid_to_row


def test_reference_embeddings_and_multicodebook_decode(mlx_runner):
    from types import SimpleNamespace

    from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs

    req = SimpleNamespace(
        extend_range=SimpleNamespace(start=0, length=3),
        inflight_middle_chunks=0,
        origin_input_ids=[1, -100, 2],
        sampling_params=SimpleNamespace(
            sampling_seed=42, temperature=0.0, top_p=1.0, top_k=1
        ),
    )
    request = SimpleNamespace(
        request_id="ref",
        data=SimpleNamespace(req=req, reference_codes_delayed=[[1] * 8]),
    )
    batch = SimpleNamespace(
        input_ids=torch.tensor([1, -100, 2], device="mps"),
        batch_size=1,
        replace_embeds=None,
        sampling_info=SimpleNamespace(
            temperatures=torch.tensor([0.0]),
            top_ps=torch.tensor([1.0]),
            top_ks=torch.tensor([1]),
        ),
    )
    with torch.inference_mode():
        mlx_runner.before_prefill(batch, None, [request])
        embedded = get_omni_prefill_inputs(batch).input_embeds
        torch.testing.assert_close(
            embedded[1],
            mlx_runner.model.get_multimodal_embedding()(
                torch.ones(1, 8, dtype=torch.long, device="mps")
            )[0],
        )
        out = mlx_runner.custom_prefill_forward(batch, None, [request])
        assert torch.isfinite(out.logits_output.hidden_states).all()
        batch.input_ids = mlx_runner.model.get_output_codes("ref")[-1, :1]
        mlx_runner.before_decode(batch, None, [request])
        out = mlx_runner.custom_decode_forward(batch, None, [request])
        assert torch.isfinite(out.logits_output.hidden_states).all()
        assert mlx_runner._past_key_values["ref"][0].offset == 4
        mlx_runner.reset_request("ref")


def test_public_worker_registry():
    from sglang_omni.model_runner.mlx_model_worker import resolve_mlx_runner_factory
    from sglang_omni.models.higgs_tts.mlx.runner import HiggsMlxWorkerModel

    assert resolve_mlx_runner_factory("HiggsTTSModel")() is HiggsMlxWorkerModel
    with pytest.raises(NotImplementedError, match="architecture"):
        resolve_mlx_runner_factory("missing")


@pytest.mark.parametrize("dtype", ["float32", "float16", "auto", torch.float32])
def test_worker_rejects_unsupported_dtype_before_loading(dtype, monkeypatch):
    from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config
    from sglang_omni.models.higgs_tts.mlx.runner import HiggsMlxWorkerModel

    def must_not_load(*args, **kwargs):
        pytest.fail("Unsupported dtype must be rejected before reading the checkpoint")

    monkeypatch.setattr(HiggsMultimodalQwen3Config, "from_pretrained", must_not_load)
    with pytest.raises(ValueError, match="Higgs MLX.*dtype"):
        HiggsMlxWorkerModel(model_path=".", dtype=dtype)


def test_public_worker_load_and_release(checkpoint, monkeypatch):
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from sglang_omni.models.higgs_tts import model as model_mod
    from sglang_omni.models.higgs_tts.mlx.runner import HiggsMlxWorkerModel

    path, _, _, state = checkpoint
    state["tied.embedding.modality_embeddings.0.embedding.weight"] = torch.zeros(
        8 * 1026, 32
    )
    save_file(state, str(path / "model.safetensors"))
    monkeypatch.setattr(model_mod, "_resolve_max_running_requests", lambda: 1)
    owner = HiggsMlxWorkerModel(model_path=str(path), pool_size=64)
    assert (
        owner.scheduler_model.backbone.model.embed_tokens.weight.dtype == torch.bfloat16
    )
    assert owner.pool_size == 64
    assert not hasattr(owner.scheduler_model.backbone.model, "layers")
    worker = SimpleNamespace(
        gpu_id=0,
        model_runner=SimpleNamespace(model=owner.scheduler_model),
        _mlx_runner=owner,
    )
    runner = HiggsMlxModelRunner(worker, None)
    assert runner._past_key_values is owner.request_caches
    with torch.inference_mode():
        embeds = owner.scheduler_model.backbone.model.embed_tokens(
            torch.tensor([1, 2], device="mps")
        )
        runner._forward("one", embeds, prefill=True)
    owner.scheduler_model.acquire_row("one")
    assert owner.has_request("one")
    owner.store_auxiliary_state_for_request("one")
    owner.remove_request("one")
    assert not owner.has_request("one")
    assert not owner.scheduler_model._rid_to_row
