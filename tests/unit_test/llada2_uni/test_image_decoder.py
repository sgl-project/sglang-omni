# SPDX-License-Identifier: Apache-2.0
"""Focused image-decoder and generation-path tests."""

from __future__ import annotations

import asyncio
import base64
import io
import re
from types import SimpleNamespace

import pytest
import torch
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from PIL import Image
from safetensors.torch import save_file

from sglang_omni.models.llada2_uni.components import image_decoder as decoder_module
from sglang_omni.models.llada2_uni.components.decoder_model import (
    ZImageTransformer2DModelWrapper,
    decoder_config,
)
from sglang_omni.models.llada2_uni.components.image_decoder import (
    LLaDA2ImageDecoder,
    create_decoder_model_fn,
)
from sglang_omni.models.llada2_uni.components.preprocessor import (
    BOI_TOKEN,
    EOI_TOKEN,
    IMAGE_TOKEN_OFFSET,
    SOI_TOKEN,
    LLaDA2Preprocessor,
)
from sglang_omni.models.llada2_uni.config import IMAGE_STAGE
from sglang_omni.models.llada2_uni.merge import extract_image_vq_tokens
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.llada2_uni.request_builders import (
    merge_image_tokens_for_thinker,
)
from sglang_omni.models.llada2_uni.stages import create_image_decode_executor
from sglang_omni.proto import OmniRequest, StagePayload


@pytest.fixture
def tiny_config():
    return decoder_config(
        {
            "dim": 32,
            "n_layers": 1,
            "n_refiner_layers": 1,
            "n_heads": 4,
            "n_kv_heads": 4,
            "cap_feat_dim": 16,
            "axes_dims": (2, 2, 4),
            "axes_lens": (128, 32, 32),
        }
    )


@pytest.fixture
def tiny_checkpoint(tmp_path, tiny_config):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = ZImageTransformer2DModel(**tiny_config).eval()
        torch.nn.init.normal_(model.x_pad_token, std=0.01)
        torch.nn.init.normal_(model.cap_pad_token, std=0.01)
    state = {
        key.replace("cap_embedder.", "semantic_embedder."): value.contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(tmp_path / "model.safetensors"))
    return tmp_path, model, state


def test_diffusers_checkpoint_forward(tiny_checkpoint, tiny_config):
    path, reference, _ = tiny_checkpoint
    wrapper = ZImageTransformer2DModelWrapper(path, tiny_config, "cpu", torch.float32)
    x = [torch.randn(16, 1, 4, 6), torch.randn(16, 1, 6, 4)]
    cap = [torch.randn(7, 16), torch.randn(33, 16)]
    t = torch.tensor([0.125, 0.875])
    with torch.inference_mode():
        expected = reference(x=x, t=t, cap_feats=cap, return_dict=False)[0]
        actual = wrapper(x, t, cap, return_dict=False)[0]
    for actual_item, expected_item in zip(actual, expected):
        torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)


def test_cfg_batched_model_fn():
    calls = []
    positive = [torch.ones(4, 16), torch.ones(5, 16) * 2]
    negative = [torch.zeros_like(cap) for cap in positive]
    positive_latent = torch.tensor([3.0, 4.0]).reshape(2, 1, 1, 1)
    negative_latent = torch.tensor([-1.0, 2.0]).reshape(2, 1, 1, 1)

    def model(**kwargs):
        calls.append(kwargs)
        return (
            [
                positive_latent,
                positive_latent * 2,
                negative_latent,
                negative_latent * 2,
            ],
        )

    model_fn = create_decoder_model_fn(
        model, positive, negative, 1.0, 2, 1, torch.bfloat16
    )
    x = torch.randn(2, 2, 1, 1, 1)
    actual = model_fn(x, torch.tensor(0.25))
    expected = positive_latent + (positive_latent - negative_latent)
    expected *= min(
        1.0,
        torch.linalg.vector_norm(positive_latent) / torch.linalg.vector_norm(expected),
    )
    torch.testing.assert_close(actual, torch.stack([expected, expected * 2]))
    assert actual.dtype == torch.float32
    assert calls[0]["cap_feats"] == positive + negative
    assert len(calls[0]["x"]) == 4


@pytest.fixture
def decode_probe(monkeypatch, tmp_path):
    observed = SimpleNamespace(ids=None, latents=None)
    decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        dtype=torch.float32,
        num_steps=4,
        resolution_multiplier=1,
    )

    def sigvq(ids):
        observed.ids = ids.clone()
        return ids.float().unsqueeze(-1).expand(-1, -1, 16)

    def model(**kwargs):
        return ([torch.ones_like(latent) for latent in kwargs["x"]],)

    def vae_decode(latents, return_dict):
        observed.latents = latents.clone()
        assert not return_dict
        pixels = torch.empty(1, 3, latents.shape[-2] * 8, latents.shape[-1] * 8)
        pixels[:, 0], pixels[:, 1], pixels[:, 2] = -2, 0, 2
        return (pixels,)

    decoder._sigvq = sigvq
    decoder._diff_model = model
    decoder._diff_config = {"all_patch_size": [2], "all_f_patch_size": [1]}
    decoder._vae = SimpleNamespace(
        config=SimpleNamespace(scaling_factor=2.0, shift_factor=3.0),
        decode=vae_decode,
    )
    monkeypatch.setattr(decoder, "ensure_diff_model", lambda mode: None)
    return decoder, observed


def test_decode_pipeline(decode_probe):
    decoder, observed = decode_probe
    image = decoder.decode([1, 2], 1, 2, seed=9)
    assert image.size == (32, 16)
    assert image.mode == "RGB" and image.getpixel((0, 0)) == (0, 127, 255)
    torch.testing.assert_close(observed.ids, torch.tensor([[1, 1, 2, 2, 1, 1, 2, 2]]))
    expected_noise = torch.randn(
        (1, 16, 1, 2, 4), generator=torch.Generator().manual_seed(9)
    )
    torch.testing.assert_close(
        observed.latents, (expected_noise.squeeze(2) + 1) / 2 + 3
    )


def test_decode_rejects_invalid_tokens_before_loading(tmp_path, monkeypatch):
    decoder = LLaDA2ImageDecoder(str(tmp_path), device="cpu")
    monkeypatch.setattr(
        decoder, "ensure_sigvq", lambda: pytest.fail("invalid input loaded weights")
    )
    with pytest.raises(ValueError):
        decoder.decode([16384], 1, 1)


class FakeTokenizer:
    mask_token_id = 156895
    eos_token_id = 2

    def __len__(self):
        return IMAGE_TOKEN_OFFSET + 16384

    def convert_tokens_to_ids(self, token):
        return {
            SOI_TOKEN: 156901,
            EOI_TOKEN: 156902,
            BOI_TOKEN: 156904,
            "<uncondition>": 90,
        }[token]

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in re.split(
            r"(<\|reserved_token_\d+\|>|<\|/?image\|>|<boi>|<uncondition>)", text
        ):
            if part in {
                SOI_TOKEN,
                EOI_TOKEN,
                BOI_TOKEN,
                "<uncondition>",
            }:
                ids.append(self.convert_tokens_to_ids(part))
            elif part.startswith("<|reserved_token_"):
                ids.append(10000 + int(re.search(r"\d+", part)[0]))
            else:
                ids.extend(ord(char) + 1000 for char in part)
        return ids


@pytest.fixture
def preprocessor():
    processor = LLaDA2Preprocessor.__new__(LLaDA2Preprocessor)
    processor._tokenizer = FakeTokenizer()
    processor._soi_id = processor._tokenizer.convert_tokens_to_ids(SOI_TOKEN)
    processor._boi_id = processor._tokenizer.convert_tokens_to_ids(BOI_TOKEN)
    processor._eoi_id = processor._tokenizer.convert_tokens_to_ids(EOI_TOKEN)
    processor._max_seq_len = 8192
    processor._merge_size = 1
    processor._factor = 16
    processor._image_processor = SimpleNamespace(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=1,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        rescale_factor=1 / 255,
    )
    return processor


@pytest.mark.parametrize(
    "source_size,grid",
    [((1024, 1024), (32, 32)), ((256, 256), (32, 32)), ((1600, 200), (16, 64))],
)
def test_edit_preprocess_merge_extract_and_decode(
    preprocessor, decode_probe, monkeypatch, tmp_path, source_size, grid
):
    source = tmp_path / "source.png"
    Image.new("RGB", source_size, "white").save(source)
    payload = StagePayload(
        request_id="e2e",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "make it green"}],
                "images": [str(source)],
            },
            metadata={"output_modalities": ["image"]},
            params={},
        ),
        data={},
    )
    state = LLaDA2UniPipelineState.from_dict(asyncio.run(preprocessor(payload)).data)
    assert state.task_kind == "edit"
    grid_h, grid_w = grid
    assert state.stream_state["image_info"] == [{"grid_h": grid_h, "grid_w": grid_w}]
    encoded = state.encoder_inputs[IMAGE_STAGE]
    torch.testing.assert_close(
        encoded["pixel_values"], torch.ones_like(encoded["pixel_values"])
    )

    source_tokens = list(range(grid_h * grid_w))
    state.encoder_outs[IMAGE_STAGE] = {"image_token_ids": [source_tokens]}
    merge_image_tokens_for_thinker(state)
    prompt_ids = state.prompt["input_ids"].flatten().tolist()
    assert [tid for tid in prompt_ids if tid >= IMAGE_TOKEN_OFFSET] == [
        IMAGE_TOKEN_OFFSET + token_id for token_id in source_tokens
    ]

    state.thinker_out = {
        "output_ids": [IMAGE_TOKEN_OFFSET + token_id for token_id in source_tokens]
        + [2]
    }
    tokens, h, w, params = extract_image_vq_tokens(state)
    assert tokens == source_tokens
    assert (h, w) == grid
    assert params == {}

    decoder, _ = decode_probe
    monkeypatch.setattr(decoder_module, "LLaDA2ImageDecoder", lambda **kwargs: decoder)
    scheduler = create_image_decode_executor("unused", device="cpu")
    payload.data = state.to_dict()
    result = scheduler._fn(payload)
    image = Image.open(io.BytesIO(base64.b64decode(result.data["image"])))
    assert image.size == (grid_w * 16, grid_h * 16)
    assert image.getpixel((0, 0)) == (0, 127, 255)
    assert result.data["events"][0]["type"] == "image_final"
