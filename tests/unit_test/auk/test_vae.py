# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
import torch

from sglang_omni.models.auk import runtime
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.vae import AuKVAEConfig, BigVGANFlowVAE

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
    with torch.inference_mode():
        assert torch.equal(
            loaded.inference_from_latents(latents),
            served_vae.inference_from_latents(latents),
        )
