# SPDX-License-Identifier: Apache-2.0

import copy
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.auk import runtime
from sglang_omni.models.auk import vae as vae_module
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.vae import (
    Activation1d,
    AuKVAEConfig,
    BigVGANFlowVAE,
    SnakeBeta,
)
from sglang_omni.platforms import current_platform

TINY_VAE = dict(
    upsample_rates=[2, 2, 2],
    upsample_kernel_sizes=[4, 4, 4],
    upsample_initial_channel=192,
    resblock_kernel_sizes=[3, 7],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
    downsample_rates=[2, 2],
    downsample_channels=[4, 8, 16],
    latent_dim=8,
    flow_hidden_channels=8,
)

requires_xpu = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="the fused activations run on XPU"
)


def seeded_vae() -> BigVGANFlowVAE:
    torch.manual_seed(0)
    vae = BigVGANFlowVAE(AuKVAEConfig.from_dict(TINY_VAE)).eval()
    with torch.no_grad():
        for parameter in vae.parameters():
            parameter.normal_(0.0, 0.3)
    return vae


@pytest.fixture
def served_vae(monkeypatch: pytest.MonkeyPatch) -> Iterator[BigVGANFlowVAE]:
    checkpoint = seeded_vae()
    monkeypatch.setattr(
        runtime,
        "make_runtime_config",
        lambda path: AuKRuntimeConfig(
            model_path=path, vae={"model_init_kwargs": TINY_VAE}
        ),
    )
    monkeypatch.setattr(
        runtime,
        "load_vae_weights",
        lambda vae, path: vae.load_state_dict(checkpoint.state_dict()),
    )
    runtime.load_vae.cache_clear()
    yield checkpoint
    runtime.load_vae.cache_clear()


def test_the_loader_folds_weight_norm_without_moving_the_waveform(
    served_vae: BigVGANFlowVAE,
) -> None:
    loaded = runtime.load_vae("stub", "cpu")
    latents = torch.randn(2, 8, 13, generator=torch.Generator().manual_seed(1))

    assert not [
        name
        for name, _ in loaded.named_parameters()
        if name.endswith(("weight_g", "weight_v"))
    ]
    assert not [name for name, p in loaded.named_parameters() if p.requires_grad]
    with torch.inference_mode():
        assert torch.equal(
            loaded.inference_from_latents(latents),
            served_vae.inference_from_latents(latents),
        )


@pytest.mark.parametrize("snake_logscale", [True, False])
def test_the_loader_fuses_activations_only_for_a_log_scale_checkpoint(
    monkeypatch: pytest.MonkeyPatch, snake_logscale: bool
) -> None:
    if snake_logscale and vae_module.anti_aliased_snake_beta is None:
        pytest.skip("the fused activation kernel needs Triton")
    else:
        pass
    vae_config = {**TINY_VAE, "snake_logscale": snake_logscale}
    torch.manual_seed(0)
    checkpoint = BigVGANFlowVAE(AuKVAEConfig.from_dict(vae_config)).eval()
    with torch.no_grad():
        for parameter in checkpoint.parameters():
            parameter.uniform_(0.1, 0.5)
    monkeypatch.setattr(
        runtime,
        "make_runtime_config",
        lambda path: AuKRuntimeConfig(
            model_path=path, vae={"model_init_kwargs": vae_config}
        ),
    )
    monkeypatch.setattr(
        runtime,
        "load_vae_weights",
        lambda vae, path: vae.load_state_dict(checkpoint.state_dict()),
    )
    monkeypatch.setattr(
        runtime,
        "current_platform",
        SimpleNamespace(
            device_type="cpu", enable_auk_fused_vae_activations=lambda: True
        ),
    )
    runtime.load_vae.cache_clear()
    try:
        loaded = runtime.load_vae("stub", "cpu")
    finally:
        runtime.load_vae.cache_clear()

    activations = [
        module for module in loaded.modules() if isinstance(module, Activation1d)
    ]
    assert activations
    assert all(
        (module.inverse_beta is not None) is snake_logscale for module in activations
    )
    if not snake_logscale:
        latents = torch.randn(1, 8, 13, generator=torch.Generator().manual_seed(2))
        with torch.inference_mode():
            assert torch.equal(
                loaded.inference_from_latents(latents),
                checkpoint.inference_from_latents(latents),
            )
    else:
        pass


@requires_xpu
def test_the_xpu_loader_decodes_the_eager_waveform_through_fused_activations(
    served_vae: BigVGANFlowVAE,
) -> None:
    loaded = runtime.load_vae("stub", "xpu")
    eager = copy.deepcopy(loaded)
    for module in eager.modules():
        if isinstance(module, Activation1d):
            module.inverse_beta = None
        else:
            pass

    assert all(
        module.inverse_beta is not None
        for module in loaded.modules()
        if isinstance(module, Activation1d)
    )
    for batch_size, frames in ((1, 1), (1, 37), (2, 160)):
        latents = torch.randn(
            batch_size,
            8,
            frames,
            generator=torch.Generator().manual_seed(frames),
        ).to("xpu")
        with torch.inference_mode():
            assert torch.equal(
                loaded.inference_from_latents(latents),
                eager.inference_from_latents(latents),
            )


@requires_xpu
@pytest.mark.parametrize("channels, frames", [(768, 200), (96, 1201), (24, 4801)])
def test_a_fused_activation_matches_eager_on_a_strided_input(
    channels: int, frames: int
) -> None:
    torch.manual_seed(channels)
    activation = Activation1d(SnakeBeta(channels), causal=True).to("xpu")
    with torch.no_grad():
        activation.act.alpha.normal_(0.0, 0.5)
        activation.act.beta.normal_(0.0, 0.5)
    fused = copy.deepcopy(activation)
    fused.enable_fused_kernel()
    padded = torch.randn(1, channels, frames + 3, device="xpu")
    cropped = padded[..., :-3]

    with torch.inference_mode():
        assert torch.equal(fused(cropped), activation(cropped))
