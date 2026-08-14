# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import importlib
import os
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.llada2_uni import stages
from sglang_omni.models.llada2_uni.components.image_decoder import LLaDA2ImageDecoder
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState


def test_image_decoder_constructs_sp_leader_with_native_backend(tmp_path) -> None:
    decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        stage_role="leader",
        sp_rank=0,
        sp_size=2,
        ulysses_degree=2,
    )

    assert decoder.backend == "sglang"
    assert decoder.is_leader is True


def test_image_decoder_rejects_follower_rank_zero(tmp_path) -> None:
    with pytest.raises(ValueError, match="follower.*rank"):
        LLaDA2ImageDecoder(
            str(tmp_path),
            device="cpu",
            stage_role="follower",
            sp_rank=0,
            sp_size=2,
            ulysses_degree=2,
        )


def test_parallel_follower_does_not_expose_image_bytes(tmp_path) -> None:
    decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        stage_role="follower",
        sp_rank=1,
        sp_size=2,
        ulysses_degree=2,
    )

    with pytest.raises(RuntimeError, match="leader-only"):
        decoder.decode_to_bytes([1], 1, 1)


@pytest.mark.parametrize(
    ("sp_size", "sp_rank", "expected_init_method"),
    [
        (2, 1, "env://"),
        (1, 0, "tcp://127.0.0.1:23456"),
    ],
)
def test_sglang_runtime_initialization_is_idempotent_and_rank_checked(
    monkeypatch: pytest.MonkeyPatch,
    sp_size: int,
    sp_rank: int,
    expected_init_method: str,
) -> None:
    decoder_model = importlib.import_module(
        "sglang_omni.models.llada2_uni.components.decoder_model"
    )
    monkeypatch.setenv("LOCAL_RANK", "7")
    state = {"initialized": False}
    init_calls: list[dict] = []

    class FakeArchConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeDitConfig:
        def __init__(self, *, arch_config):
            self.arch_config = arch_config

    class FakePipelineConfig:
        def __init__(self, *, dit_config):
            self.dit_config = dit_config

    class FakeServerArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def _initialize(**kwargs):
        init_calls.append(kwargs)
        state["initialized"] = True

    symbols = SimpleNamespace(
        ZImageArchConfig=FakeArchConfig,
        ZImageDitConfig=FakeDitConfig,
        ZImagePipelineConfig=FakePipelineConfig,
        ServerArgs=FakeServerArgs,
        get_global_server_args=lambda: (_ for _ in ()).throw(ValueError("unset")),
        set_global_server_args=lambda _args: None,
        set_mixed_precision_policy=lambda **_kwargs: None,
        model_parallel_is_initialized=lambda: state["initialized"],
        maybe_init_distributed_environment_and_model_parallel=_initialize,
        get_sp_world_size=lambda: sp_size,
        get_sp_parallel_rank=lambda: sp_rank,
    )
    monkeypatch.setattr(decoder_model, "_get_sglang_runtime_symbols", lambda: symbols)
    monkeypatch.setattr(
        decoder_model,
        "_free_tcp_port",
        lambda: 23456,
        raising=False,
    )

    parallel_config = decoder_model.ZImageParallelConfig(
        sp_rank=sp_rank,
        sp_size=sp_size,
        ulysses_degree=sp_size,
    )
    config = {
        "all_patch_size": [2],
        "all_f_patch_size": [1],
        "in_channels": 16,
        "dim": 60,
        "n_layers": 1,
        "n_refiner_layers": 1,
        "n_heads": 30,
        "n_kv_heads": 30,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "cap_feat_dim": 4096,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
        "axes_dims": [2, 0, 0],
        "axes_lens": [8, 8, 8],
    }

    first = decoder_model.ensure_sglang_zimage_runtime(
        model_path="checkpoint",
        config=config,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        parallel_config=parallel_config,
    )
    second = decoder_model.ensure_sglang_zimage_runtime(
        model_path="checkpoint",
        config=config,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        parallel_config=parallel_config,
    )

    assert first.dit_config.arch_config.kwargs["num_attention_heads"] == 30
    assert first.pipeline_config.dit_precision == "bf16"
    assert second.pipeline_config.dit_config is second.dit_config
    assert os.environ["LOCAL_RANK"] == "0"
    assert init_calls == [
        {
            "tp_size": 1,
            "sp_size": sp_size,
            "cfg_degree": 1,
            "ulysses_degree": sp_size,
            "ring_degree": 1,
            "distributed_init_method": expected_init_method,
        }
    ]


def test_native_loader_uses_checkpoint_load_device_keyword(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    decoder_model = importlib.import_module(
        "sglang_omni.models.llada2_uni.components.decoder_model"
    )
    observed: dict[str, object] = {}

    class FakeModel(torch.nn.Module):
        param_names_mapping = {}

        def __init__(self, *, config, hf_config):
            super().__init__()
            self.config = config
            self.hf_config = hf_config

    class FakeWeightLoadPlan:
        def __init__(self, *, checkpoint_load_device):
            self.checkpoint_load_device = checkpoint_load_device

    def _weights(files, *, weight_load_plan):
        observed["weight_files"] = files
        observed["iterator_device"] = weight_load_plan.checkpoint_load_device
        return iter(())

    def _load(**kwargs):
        observed["load_kwargs"] = kwargs
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    symbols = SimpleNamespace(
        ZImageTransformer2DModel=FakeModel,
        WeightLoadPlan=FakeWeightLoadPlan,
        get_param_names_mapping=lambda mapping: lambda name: (name, None, None),
        load_model_from_full_model_state_dict=_load,
        safetensors_weights_iterator=_weights,
        set_default_torch_dtype=lambda _dtype: contextlib.nullcontext(),
    )
    monkeypatch.setattr(decoder_model, "_get_sglang_loader_symbols", lambda: symbols)

    model = decoder_model.load_sglang_zimage_model(
        decoder_dir=tmp_path,
        config={"name": "decoder"},
        dit_config=object(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        checkpoint_load_device=torch.device("cpu"),
    )

    assert isinstance(model, FakeModel)
    assert observed["iterator_device"] == torch.device("cpu")
    assert observed["load_kwargs"]["checkpoint_load_device"] == torch.device("cpu")


def test_native_adapter_delegates_sp_shard_forward_and_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder_model = importlib.import_module(
        "sglang_omni.models.llada2_uni.components.decoder_model"
    )
    calls: list[str] = []
    forwarded: dict[str, object] = {}

    @contextlib.contextmanager
    def _forward_context(**kwargs):
        calls.append("context-enter")
        assert kwargs == {
            "current_timestep": 0,
            "attn_metadata": None,
            "forward_batch": None,
        }
        yield
        calls.append("context-exit")

    monkeypatch.setattr(
        decoder_model,
        "_get_sglang_forward_context",
        lambda: _forward_context,
        raising=False,
    )

    class FakePipelineConfig:
        def shard_latents_for_sp(self, batch, latents):
            calls.append("shard")
            assert batch.raw_latent_shape == tuple(latents.shape)
            return latents[..., :2], True

        def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
            calls.append("condition")
            assert batch.prompt_seq_lens == [[3]]
            return {
                "freqs_cis": "freqs",
                "image_seq_len_target": 32,
                "caption_valid_lens": torch.tensor([3]),
            }

        def gather_latents_for_sp(self, latents, batch):
            calls.append("gather")
            return torch.cat([latents, latents], dim=-1)

    class FakeModel(torch.nn.Module):
        rotary_emb = object()

        def forward(self, **kwargs):
            forwarded.update(kwargs)
            return -torch.stack(kwargs["hidden_states"])

    adapter = decoder_model.SGLangZImageModelAdapter(
        FakeModel(),
        FakePipelineConfig(),
    )
    full = torch.zeros(1, 16, 1, 2, 4)
    local = adapter.prepare_latents(full)
    result = adapter(
        x=list(local.unbind(0)),
        t=torch.tensor([0.25]),
        cap_feats=[torch.zeros(3, 4096)],
        patch_size=2,
        f_patch_size=1,
        return_dict=False,
    )[0]
    gathered = adapter.gather_latents(result)

    assert calls == [
        "shard",
        "condition",
        "context-enter",
        "context-exit",
        "gather",
    ]
    assert forwarded["freqs_cis"] == "freqs"
    assert forwarded["image_seq_len_target"] == 32
    assert torch.equal(forwarded["timestep"], torch.tensor([750.0]))
    assert len(result) == 1
    assert gathered.shape == full.shape


def test_image_decode_factory_wires_sp_metadata_and_follower_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.llada2_uni.components import image_decoder

    observed: dict[str, object] = {}

    class FakeDecoder:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def decode(self, *args, **kwargs):
            observed["decode"] = (args, kwargs)
            return None

        def decode_to_bytes(self, *args, **kwargs):
            raise AssertionError("parallel followers must not encode image bytes")

    monkeypatch.setattr(image_decoder, "LLaDA2ImageDecoder", FakeDecoder)
    scheduler = stages.create_image_decode_executor(
        "checkpoint",
        device="cpu",
        dtype=torch.float32,
        stage_role="follower",
        sp_rank=1,
        sp_size=2,
        nccl_port=12345,
        ulysses_degree=2,
        ring_degree=1,
        checkpoint_load_device="cpu",
    )
    state = LLaDA2UniPipelineState(
        task_kind="t2i",
        image_token_offset=100,
        thinker_out={"output_ids": [101]},
        generation_state={"image_grid": {"height": 1, "width": 1}},
    )
    payload = SimpleNamespace(request_id="request", data=state.to_dict())

    result = scheduler._fn(payload)

    assert observed["init"] == {
        "model_path": "checkpoint",
        "device": "cpu",
        "dtype": torch.float32,
        "decode_mode": "normal",
        "num_steps": 50,
        "resolution_multiplier": 2,
        "backend": "sglang",
        "attention_backend": "fa",
        "stage_role": "follower",
        "sp_rank": 1,
        "sp_size": 2,
        "ulysses_degree": 2,
        "ring_degree": 1,
        "checkpoint_load_device": "cpu",
    }
    assert result.data == {"modality": "image", "parallel_follower": True}


def test_sp_decoder_loads_native_wrapper_with_checkpoint_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import json

    from sglang_omni.models.llada2_uni.components import decoder_model

    decoder_dir = tmp_path / "decoder"
    decoder_dir.mkdir()
    (decoder_dir / "config.json").write_text(
        json.dumps(
            {
                "all_patch_size": [2],
                "all_f_patch_size": [1],
                "in_channels": 16,
                "dim": 60,
                "n_layers": 1,
                "n_refiner_layers": 1,
                "n_heads": 30,
                "n_kv_heads": 30,
                "norm_eps": 1e-5,
                "qk_norm": True,
                "cap_feat_dim": 2560,
                "rope_theta": 256.0,
                "t_scale": 1000.0,
                "axes_dims": [2, 0, 0],
                "axes_lens": [8, 8, 8],
            }
        )
    )
    observed: dict[str, object] = {}

    class FakeWrapper:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr(decoder_model, "ZImageTransformer2DModelWrapper", FakeWrapper)
    decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        stage_role="leader",
        sp_rank=0,
        sp_size=2,
        ulysses_degree=2,
        checkpoint_load_device="cpu",
    )

    model, loaded_config = decoder._ensure_diffusion_model("normal")

    assert isinstance(model, FakeWrapper)
    assert loaded_config["cap_feat_dim"] == 4096
    assert observed["checkpoint_load_device"] == torch.device("cpu")
    assert observed["parallel_config"].sp_size == 2
    assert observed["parallel_config"].sp_rank == 0


def test_sp_leader_broadcasts_conditioning_failure_before_feature_collective(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        stage_role="leader",
        sp_rank=0,
        sp_size=2,
        ulysses_degree=2,
    )
    monkeypatch.setattr(
        decoder,
        "_ensure_diffusion_model",
        lambda _mode: (object(), {"all_patch_size": [2], "all_f_patch_size": [1]}),
    )
    monkeypatch.setattr(
        decoder,
        "_ensure_sigvq",
        lambda: (_ for _ in ()).throw(RuntimeError("bad SigVQ checkpoint")),
    )
    broadcasts: list[int] = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, src: broadcasts.append(int(tensor.flatten()[0])),
    )

    with pytest.raises(RuntimeError, match="bad SigVQ checkpoint"):
        decoder.decode([1], 1, 1, num_steps=1)

    assert broadcasts == [0]


def test_sp_follower_stops_when_leader_conditioning_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        stage_role="follower",
        sp_rank=1,
        sp_size=2,
        ulysses_degree=2,
    )
    monkeypatch.setattr(
        decoder,
        "_ensure_diffusion_model",
        lambda _mode: (object(), {"all_patch_size": [2], "all_f_patch_size": [1]}),
    )
    monkeypatch.setattr(
        decoder,
        "_ensure_sigvq",
        lambda: (_ for _ in ()).throw(AssertionError("follower loaded SigVQ")),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, src: tensor.zero_()
    )

    assert decoder.decode([1], 1, 1, num_steps=1) is None


def test_sp_follower_runs_native_shard_and_gather_without_vae(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[str] = []

    class FakeWrapper:
        def prepare_latents(self, latents):
            calls.append("shard")
            return latents

        def gather_latents(self, latents):
            calls.append("gather")
            return latents

        def __call__(self, **_kwargs):
            raise AssertionError("num_steps=1 must not evaluate the velocity")

    decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        stage_role="follower",
        sp_rank=1,
        sp_size=2,
        ulysses_degree=2,
    )
    wrapper = FakeWrapper()
    monkeypatch.setattr(
        decoder,
        "_ensure_diffusion_model",
        lambda _mode: (
            wrapper,
            {"all_patch_size": [2], "all_f_patch_size": [1]},
        ),
    )
    monkeypatch.setattr(
        decoder,
        "_ensure_sigvq",
        lambda: (_ for _ in ()).throw(AssertionError("follower loaded SigVQ")),
    )
    monkeypatch.setattr(
        decoder,
        "_ensure_vae",
        lambda: (_ for _ in ()).throw(AssertionError("follower loaded VAE")),
    )
    broadcast_index = 0

    def _broadcast(tensor, src):
        nonlocal broadcast_index
        if broadcast_index == 0:
            tensor.fill_(1)
        broadcast_index += 1

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast", _broadcast)

    assert decoder.decode([1], 1, 1, num_steps=1, seed=7) is None
    assert calls == ["shard", "gather"]


def test_single_rank_sglang_backend_uses_native_latent_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[str] = []

    class FakeWrapper:
        def prepare_latents(self, latents):
            calls.append("prepare")
            return latents

        def gather_latents(self, latents):
            calls.append("gather")
            return latents

        def __call__(self, **_kwargs):
            raise AssertionError("num_steps=1 must not evaluate the velocity")

    class FakeSigVQ:
        def __call__(self, tokens):
            return torch.zeros(
                tokens.shape[0],
                tokens.shape[1],
                4096,
                dtype=torch.float32,
            )

    class FakeVAE:
        config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

        def decode(self, latent, *, return_dict):
            assert return_dict is False
            return (torch.zeros(latent.shape[0], 3, *latent.shape[-2:]),)

    decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        backend="sglang",
    )
    wrapper = FakeWrapper()
    monkeypatch.setattr(
        decoder,
        "_ensure_diffusion_model",
        lambda _mode: (
            wrapper,
            {"all_patch_size": [2], "all_f_patch_size": [1]},
        ),
    )
    monkeypatch.setattr(decoder, "_ensure_sigvq", lambda: FakeSigVQ())
    monkeypatch.setattr(decoder, "_ensure_vae", lambda: FakeVAE())

    image = decoder.decode([1], 1, 1, num_steps=1, seed=7)

    assert image is not None
    assert calls == ["prepare", "gather"]
