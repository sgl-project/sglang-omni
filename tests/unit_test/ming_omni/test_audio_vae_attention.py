# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from sglang_omni.models.ming_omni.talker.audio_vae import modeling_audio_vae
from sglang_omni.models.ming_omni.talker.audio_vae.configuration_audio_vae import (
    AudioVAEconfig,
)


@pytest.mark.parametrize("is_npu", [False, True])
def test_audio_vae_attention_backend_and_window(monkeypatch, is_npu):
    monkeypatch.setattr(modeling_audio_vae.current_platform, "is_npu", lambda: is_npu)
    backbone = {
        "_attn_implementation": "eager",
        "vocab_size": 1,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "use_sliding_window": True,
        "sliding_window": 4,
        "max_window_layers": 0,
        "use_cache": False,
        "bos_token_id": None,
        "eos_token_id": None,
    }
    config = AudioVAEconfig(
        enc_kwargs={"backbone": backbone, "input_dim": 8, "latent_dim": 2},
        dec_kwargs={
            "backbone": copy.deepcopy(backbone),
            "output_dim": 8,
            "latent_dim": 2,
        },
        patch_size=2,
    )
    original = copy.deepcopy((config.enc_kwargs, config.dec_kwargs))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = modeling_audio_vae.AudioVAE(config).eval()
        inputs = torch.randn(1, 32, 8)
    for component in (
        model.encoder.encoder,
        model.encoder.aggregator,
        model.decoder.decoder,
    ):
        assert component.config._attn_implementation == ("sdpa" if is_npu else "eager")
        assert component.config.sliding_window == 4
    assert (config.enc_kwargs, config.dec_kwargs) == original

    # The distant prefix is outside all four layers' combined receptive field.
    changed = inputs.clone()
    changed[:, :8] *= -10
    with torch.inference_mode():
        before = model.encoder.encoder(inputs_embeds=inputs).last_hidden_state
        after = model.encoder.encoder(inputs_embeds=changed).last_hidden_state
    torch.testing.assert_close(before[:, -1], after[:, -1])
    assert not torch.allclose(before[:, 0], after[:, 0])
