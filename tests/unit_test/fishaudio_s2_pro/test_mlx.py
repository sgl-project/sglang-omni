# SPDX-License-Identifier: Apache-2.0
"""Numerical checks for the native Fish MLX networks and cache lifecycle."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("requires Apple Metal", allow_module_level=True)

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.audio_decoder import (
    FishQwen3AudioDecoder,
)
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.configuration import (
    FishQwen3AudioDecoderConfig,
    FishQwen3Config,
)
from sglang_omni.models.fishaudio_s2_pro.mlx.model import FishModel, to_mlx, to_torch
from sglang_omni.models.fishaudio_s2_pro.torch_mps import S2ProTorchMpsTextModel


def models():
    torch.manual_seed(73)
    common = dict(
        dim=32,
        n_layer=2,
        n_head=4,
        n_local_heads=2,
        head_dim=8,
        intermediate_size=48,
        max_seq_len=32,
        use_moe=False,
    )
    text_config = FishQwen3Config(vocab_size=64, attention_qk_norm=True, **common)
    audio_config = FishQwen3AudioDecoderConfig(
        vocab_size=8, text_dim=32, num_codebooks=3, attention_qk_norm=False, **common
    )
    text = S2ProTorchMpsTextModel(SimpleNamespace(text_config=text_config)).eval()
    audio = FishQwen3AudioDecoder(audio_config).eval()
    native = FishModel(
        dict(
            text_config=text_config.to_dict(),
            audio_decoder_config=audio_config.to_dict(),
        )
    )
    weights = [
        ("text_model.model." + k.replace("embed_tokens.", "embeddings."), to_mlx(v))
        for k, v in text.named_parameters()
    ]
    weights += [("audio_decoder." + k, to_mlx(v)) for k, v in audio.named_parameters()]
    native.load_weights(weights, strict=True)
    native.eval()
    return text, audio, native


def test_native_slow_matches_torch_and_cached_prefill():
    text, _, native = models()
    tokens = torch.tensor([1, 5, 8, 3, 2])
    expected, _ = text.forward_native(tokens, request_id="r", prefill=True)
    cache = native.text_model.model.make_cache()
    embeds = native.text_model.model.embeddings(to_mlx(tokens))
    native.text_model(embeds[None, :3], cache)
    native.text_model(embeds[None, 3:4], cache)
    actual, _ = native.text_model(embeds[None, 4:], cache)
    full, _ = native.text_model(embeds[None], native.text_model.model.make_cache())
    # M5 MLX defaults to reduced-precision FP32 GEMM (TF32), whose ~1e-3
    # relative error lands near 2e-2 on these logits (|max| ~ 20). Tolerate that
    # against the logit scale rather than absolutely; a real port bug is O(1).
    # Run with MLX_ENABLE_TF32=0 for the strict cross-framework check.
    strict = os.environ.get("MLX_ENABLE_TF32") == "0"
    atol, rtol = (2e-5, 2e-5) if strict else (5e-2, 2e-3)
    torch.testing.assert_close(to_torch(actual), expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(to_torch(actual), to_torch(full), atol=atol, rtol=rtol)
    assert all(state.offset == 5 for state in cache)


def test_reference_codebook_embedding_matches_torch():
    _, audio, native = models()
    text = torch.randn(2, 32)
    codes = torch.tensor([[1, 2, 3], [4, 2, 0]])
    expected = (
        text + audio.codebook_embeddings(codes + audio.codebook_offsets).sum(1)
    ) / 4**0.5
    actual = native.audio_decoder.mix_embeddings(to_mlx(text), to_mlx(codes))
    torch.testing.assert_close(to_torch(actual), expected, atol=1e-6, rtol=1e-6)


def test_native_fast_chain_matches_mps_and_resets_every_frame():
    _, audio, native = models()
    audio = audio.to("mps")
    audio.setup_caches(1, dtype=torch.float32)
    hidden = torch.randn(1, 32)
    for semantic_code in [2, 0, 2]:
        audio.reset_caches()
        audio.forward_kvcached(audio.project_in(hidden.to("mps"))[:, None], 0)
        code = torch.tensor([semantic_code], device="mps")
        expected = [semantic_code]
        for index in range(1, 3):
            logits = audio.forward_kvcached(audio.embeddings(code)[:, None], index)
            code = logits[:, 0].argmax(-1)
            expected.append(code.item())
        actual = native.audio_decoder.generate(to_mlx(hidden), semantic_code)
        assert np.array(actual).tolist() == [expected]


def test_native_loader_rejects_missing_extra_and_wrong_shapes(tmp_path):
    import json

    from mlx.utils import tree_flatten

    _, _, native = models()
    (tmp_path / "config.json").write_text(json.dumps(native._config))
    weights = dict(tree_flatten(native.parameters()))
    path = str(tmp_path / "model.safetensors")
    mx.save_safetensors(path, weights)
    loaded = FishModel.from_pretrained(tmp_path)
    assert len(tree_flatten(loaded.parameters())) == len(weights)
    name = next(iter(weights))
    for malformed in [
        {key: value for key, value in weights.items() if key != name},
        {**weights, "unexpected.weight": mx.zeros((1,))},
        {**weights, name: mx.zeros((1,))},
    ]:
        mx.save_safetensors(path, malformed)
        with pytest.raises(ValueError):
            FishModel.from_pretrained(tmp_path)


def test_native_request_reference_feedback_and_cache_cleanup():
    from sglang_omni.models.fishaudio_s2_pro.mlx.runner import (
        FishMlxModel,
        FishMlxSchedulerRunner,
    )

    _, _, native = models()
    model = object.__new__(FishMlxModel)
    torch.nn.Module.__init__(model)
    model.native = native
    model.vocab_size = 64
    model.context_length = 32
    model._request_caches = {}
    model.configure(
        SimpleNamespace(semantic_begin_id=32, semantic_end_id=39, eos_token_ids=[40]),
        16,
    )
    model._sampling_seeds.fill_(42)
    request = SimpleNamespace(
        request_id="r",
        data=SimpleNamespace(
            req=SimpleNamespace(
                prefix_indices=[], origin_input_ids=[1, 33, 34, 5], output_ids=[]
            ),
            vq_parts=[torch.tensor([[1, 2], [3, 4], [5, 6]])],
            vq_mask_tokens=torch.tensor([False, True, True, False]),
            last_codebook_values=None,
        ),
    )
    model.forward_request(request, prefill=True)
    assert model._request_caches["r"][0].offset == 4
    assert model._output_codes.shape == (1, 4)
    # Explicit semantic feedback makes this branch independent of the sampled
    # EOS choice of the random tiny checkpoint.
    request.data.req.output_ids = [33]
    request.data.last_codebook_values = torch.tensor([1, 2, 3])
    model.forward_request(request, prefill=False)
    assert model._request_caches["r"][0].offset == 5
    runner = object.__new__(FishMlxSchedulerRunner)
    runner.model = model
    runner.abort_request("r")
    assert not model._request_caches
    runner.abort_request("r")
    with pytest.raises(RuntimeError, match="no cache"):
        model.forward_request(request, prefill=False)
    model.forward_request(request, prefill=True)
    runner.on_request_finished("r", None)
    assert not model._request_caches


def test_native_profile_rejects_unsafe_overrides(monkeypatch):
    import sglang.srt.hardware_backend.mlx.runtime as mlx_runtime

    from sglang_omni.models.fishaudio_s2_pro.engine_builder import (
        FishS2ProEngineBuilder,
    )

    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: True)
    builder = FishS2ProEngineBuilder(max_new_tokens=32, ras_window=16)
    defaults = builder.generation_defaults(dtype="bfloat16")
    builder.adjust_overrides(defaults)
    for key, value in [
        ("max_running_requests", 2),
        ("quantization", "int8"),
        ("disable_radix_cache", False),
        ("chunked_prefill_size", 128),
    ]:
        with pytest.raises(ValueError, match="Fish Apple requires"):
            builder.adjust_overrides({**defaults, key: value})


def test_native_mlx_requires_apple_metal(monkeypatch):
    import sglang.srt.hardware_backend.mlx.runtime as mlx_runtime

    from sglang_omni.models.fishaudio_s2_pro import engine_builder

    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: True)
    monkeypatch.setattr(engine_builder.current_platform, "is_mps", lambda: False)
    builder = engine_builder.FishS2ProEngineBuilder(max_new_tokens=32, ras_window=16)
    with pytest.raises(ValueError, match="Fish MLX requires Apple Metal"):
        builder.pre_infra_setup("unused")


def test_registry_adapter_binds_fish_model_and_preserves_cache_ownership(monkeypatch):
    from sglang.srt.runtime_context import get_context

    from sglang_omni.model_runner.mlx_model_worker import (
        _create_registered_runner,
        resolve_mlx_runner_factory,
    )
    from sglang_omni.models.fishaudio_s2_pro.mlx import runner

    calls = []
    model = SimpleNamespace(clear_request=lambda rid: calls.append(rid))

    def create_model(path, *, context_length):
        assert path == "checkpoint" and context_length == 4096
        return model

    monkeypatch.setattr(runner, "FishMlxModel", create_model)
    factory = resolve_mlx_runner_factory("FishS2ProMlxModel")
    with get_context().override_server_args(
        model_path="checkpoint", context_length=4096, max_total_tokens=4096
    ):
        adapter = _create_registered_runner(factory())
    assert adapter.scheduler_model is model
    assert adapter.pool_size == 4096
    adapter.prepare_for_kv_cache_release(SimpleNamespace(rid="r"))
    # Preparing a release must not prematurely clear Fish's native state;
    # completion/abort owns that operation.
    assert calls == []
