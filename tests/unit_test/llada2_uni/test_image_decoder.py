# SPDX-License-Identifier: Apache-2.0
"""Focused image-decoder and generation-path tests."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from PIL import Image
from pydantic import ValidationError
from safetensors.torch import save_file
from sglang.srt.dllm.config import DllmConfig

from sglang_omni.config import build_process_topology_plan, build_stage_placement_plan
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.patch import ConfigPatchSet
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.models.llada2_uni import stages
from sglang_omni.models.llada2_uni.components import common as common_module
from sglang_omni.models.llada2_uni.components import decoder_runtime
from sglang_omni.models.llada2_uni.components.decoder_model import (
    DecoderConfig,
    ZImageTransformer2DModelWrapper,
    decoder_config,
)
from sglang_omni.models.llada2_uni.components.image_decoder import (
    DecoderBackend,
    DecoderMode,
    LLaDA2ImageDecoder,
    create_decoder_model_fn,
)
from sglang_omni.models.llada2_uni.components.preprocessor import (
    BOI_TOKEN,
    EOI_TOKEN,
    IMAGE_TOKEN_OFFSET,
    SOI_TOKEN,
    LLaDA2Preprocessor,
    preprocess_image_edit,
)
from sglang_omni.models.llada2_uni.components.transport import sample_velocity
from sglang_omni.models.llada2_uni.config import (
    IMAGE_DECODE_STAGE,
    IMAGE_STAGE,
    THINKER_STAGE,
    LLaDA2ImageDecoderStageConfig,
    LLaDA2UniOmniPipelineConfig,
)
from sglang_omni.models.llada2_uni.merge import extract_image_vq_tokens
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.llada2_uni.request_builders import (
    build_dllm_thinker_request,
    make_dllm_thinker_scheduler_adapters,
    merge_image_tokens_for_thinker,
    thinker_next,
)
from sglang_omni.models.llada2_uni.stages import (
    create_decode_executor,
    create_image_decode_executor,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def test_image_pipeline_config_round_trip() -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="unused")
    decoder_stage = next(
        stage for stage in config.stages if stage.name == IMAGE_DECODE_STAGE
    )
    assert isinstance(decoder_stage, LLaDA2ImageDecoderStageConfig)
    decoder_stage.factory.backend = "sglang"
    decoder_stage.factory.decode_mode = "decoder-turbo"
    decoder_stage.factory.num_steps = 12
    resolved = ConfigResolver(config).resolve(ConfigPatchSet()).config
    decoder_stage = next(
        stage for stage in resolved.stages if stage.name == IMAGE_DECODE_STAGE
    )
    assert isinstance(decoder_stage, LLaDA2ImageDecoderStageConfig)
    assert decoder_stage.factory.backend == "sglang"
    assert decoder_stage.factory.decode_mode == "decoder-turbo"
    assert decoder_stage.factory.num_steps == 12
    assert resolved.stage_config_cls(THINKER_STAGE).engine_stage
    with pytest.raises(ValidationError, match="backend"):
        ConfigManager(resolved).merge_config(
            {"image_decode.factory.backend": "invalid"}
        )

    config_path = (
        Path(__file__).resolve().parents[3] / "examples/configs/llada2_uni.yaml"
    )
    config = ConfigManager.from_file(str(config_path)).config
    placement = build_stage_placement_plan(config)
    topology = build_process_topology_plan(config, placement, stages_cfg=config.stages)
    assert (
        topology.stage_to_process[IMAGE_STAGE]
        == topology.stage_to_process[THINKER_STAGE]
    )
    assert (
        topology.stage_to_process[IMAGE_DECODE_STAGE]
        != topology.stage_to_process[THINKER_STAGE]
    )


def native_worker(
    rank: int, directory: str, cfg: DecoderConfig, dtype: torch.dtype
) -> None:
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    from sglang_omni.models.llada2_uni.components.decoder_runtime import (
        initialize_decoder_runtime,
    )

    path = Path(directory)
    with initialize_decoder_runtime(
        directory,
        gpu_id=0,
        dtype=dtype,
        attention_backend="torch_sdpa",
        dist_timeout=120,
    ) as runtime:
        with runtime.compute_context(), torch.inference_mode():
            for device in ("cuda", "cuda:0"):
                decoder = LLaDA2ImageDecoder(
                    directory,
                    device=device,
                    dtype=dtype,
                    backend="sglang",
                    runtime=runtime,
                )
                assert decoder.device == runtime.device
            with pytest.raises(ValueError, match="device/dtype must match"):
                LLaDA2ImageDecoder(
                    directory,
                    device="cuda:1",
                    dtype=dtype,
                    backend="sglang",
                    runtime=runtime,
                )
            inputs = torch.load(
                path / "inputs.pt", weights_only=True, map_location=runtime.device
            )
            model = ZImageTransformer2DModelWrapper(
                directory,
                cfg,
                runtime.device,
                dtype,
                backend="sglang",
                runtime=runtime,
            )
            output = torch.stack(
                model(
                    list(inputs["x"].to(dtype).unbind(0)),
                    inputs["t"],
                    list(inputs["cap"].to(dtype).unbind(0)),
                    return_dict=False,
                )[0]
            )
            torch.save(output.cpu(), path / "native.pt")
    assert not torch.distributed.is_initialized()


@pytest.mark.accelerator
def test_sglang_backend_matches_diffusers(tmp_path: Path) -> None:
    dtype = torch.bfloat16
    cfg = decoder_config(
        {
            "dim": 768,
            "n_layers": 1,
            "n_refiner_layers": 1,
            "n_heads": 6,
            "n_kv_heads": 6,
            "cap_feat_dim": 16,
            "axes_dims": (32, 48, 48),
            "axes_lens": (128, 64, 64),
        }
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        reference = ZImageTransformer2DModel(**cfg).eval()
        torch.nn.init.normal_(reference.x_pad_token, std=0.01)
        torch.nn.init.normal_(reference.cap_pad_token, std=0.01)
        inputs = {
            "x": torch.randn(2, 16, 1, 16, 16),
            "cap": torch.randn(2, 32, 16),
            "t": torch.tensor([0.125, 0.875]),
        }
    save_file(
        {
            name.replace("cap_embedder.", "semantic_embedder."): value.contiguous()
            for name, value in reference.state_dict().items()
        },
        str(tmp_path / "model.safetensors"),
    )
    torch.save(inputs, tmp_path / "inputs.pt")
    with torch.inference_mode():
        expected = torch.stack(
            reference(
                x=list(inputs["x"].unbind(0)),
                t=inputs["t"],
                cap_feats=list(inputs["cap"].unbind(0)),
                return_dict=False,
            )[0]
        )
    del reference

    mp.spawn(
        native_worker,
        args=(str(tmp_path), cfg, dtype),
        nprocs=1,
        join=True,
    )
    actual = torch.load(tmp_path / "native.pt", weights_only=True)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


def test_sglang_decoder_factory_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    factory_types = get_type_hints(create_image_decode_executor)
    assert factory_types["backend"] == DecoderBackend
    assert factory_types["decode_mode"] == DecoderMode
    runtime = Mock(spec=decoder_runtime.DecoderRuntimeHandle)
    runtime.compute_context.return_value = nullcontext()
    initialize = Mock(return_value=runtime)
    decoder = Mock(spec=LLaDA2ImageDecoder)
    decoder.decode.return_value = Image.new("RGB", (8, 8))
    constructor = Mock(return_value=decoder)
    monkeypatch.setattr(decoder_runtime, "initialize_decoder_runtime", initialize)
    monkeypatch.setattr(stages, "LLaDA2ImageDecoder", constructor)

    scheduler = create_image_decode_executor("unused", device="cpu", backend="sglang")
    state = LLaDA2UniPipelineState(
        task_kind="t2i",
        thinker_out={"output_ids": [IMAGE_TOKEN_OFFSET + 3]},
        stream_state={"image_info": [{"grid_h": 1, "grid_w": 1}]},
        request_metadata={"image_generation": {"decode_mode": "decoder-turbo"}},
    )
    payload = StagePayload(
        request_id="decoder-wiring",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )
    assert scheduler.fn(payload) is payload
    assert constructor.call_args.kwargs["runtime"] is runtime
    assert constructor.call_args.kwargs["backend"] == "sglang"
    initialize.assert_called_once_with(
        "unused", gpu_id=None, dtype=torch.bfloat16, attention_backend="torch_sdpa"
    )
    decoder.decode.assert_called_once_with(
        [3], 1, 1, decode_mode="decoder-turbo", num_steps=8
    )
    assert payload.data["format"] == "png" and payload.data["image"]
    assert runtime.compute_context.call_count == 2
    monkeypatch.setattr(SimpleScheduler, "start", lambda self: None)
    scheduler.start()
    runtime.close.assert_called_once_with()


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


def test_turbo_sampler_matches_reference_and_preserves_rng():
    initial_latents = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)

    def velocity_model(latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return latents * 0.125 + timestep.reshape(-1, 1, 1, 1) * 0.25

    actual_generator = torch.Generator().manual_seed(29)
    actual = sample_velocity(
        initial_latents,
        velocity_model,
        num_steps=3,
        turbo=True,
        generator=actual_generator,
    )

    reference_generator = torch.Generator().manual_seed(29)
    reference_timesteps = torch.linspace(0, 1, 4, dtype=torch.float64).to(
        initial_latents
    )
    reference_latents = initial_latents.to(torch.float64)
    for current_time, next_time in zip(
        reference_timesteps[:-1], reference_timesteps[1:]
    ):
        batch_time = (
            torch.ones(reference_latents.shape[0], dtype=reference_latents.dtype)
            * current_time
        )
        prediction = velocity_model(reference_latents, batch_time)
        expanded_time = batch_time[(...,) + (None,) * (reference_latents.ndim - 1)]
        alpha, sigma = expanded_time, 1 - expanded_time
        denominator = sigma - -alpha
        clean = (sigma * prediction - -reference_latents) / denominator
        next_batch_time = torch.ones_like(batch_time) * next_time
        next_alpha = next_batch_time[(...,) + (None,) * (reference_latents.ndim - 1)]
        next_sigma = 1 - next_alpha
        noise = torch.randn(
            reference_latents.shape,
            dtype=reference_latents.dtype,
            generator=reference_generator,
        )
        reference_latents = next_alpha * clean + next_sigma * noise

    torch.testing.assert_close(actual, reference_latents)
    torch.testing.assert_close(
        torch.randn(4, generator=actual_generator),
        torch.randn(4, generator=reference_generator),
    )


@pytest.mark.parametrize("turbo,steps,expected_calls", [(False, 2, 1), (True, 1, 1)])
def test_sampler_forward_counts(turbo, steps, expected_calls):
    initial_latents = torch.zeros(1, 1, 1, 1)
    calls = []

    def model(latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        calls.append(timestep.clone())
        return torch.zeros_like(latents)

    sample_velocity(
        initial_latents,
        model,
        num_steps=steps,
        turbo=turbo,
        generator=torch.Generator().manual_seed(29),
    )
    assert len(calls) == expected_calls


def test_normal_step_bound_before_loading(tmp_path):
    with pytest.raises(ValueError, match="normal num_steps must be at least 2"):
        LLaDA2ImageDecoder(str(tmp_path), device="cpu", num_steps=1)
    decoder = LLaDA2ImageDecoder(str(tmp_path), device="cpu")
    with pytest.raises(ValueError, match="normal num_steps must be at least 2"):
        decoder.decode([1], 1, 1, num_steps=1)
    LLaDA2ImageDecoder(
        str(tmp_path), device="cpu", decode_mode="decoder-turbo", num_steps=1
    )


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

    decoder.sigvq = sigvq
    decoder.diff_model = model
    decoder.diff_config = {
        "all_patch_size": [2],
        "all_f_patch_size": [1],
    }
    decoder.vae = SimpleNamespace(
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

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token - 1000) for token in ids if 1000 <= token < 10000)


@pytest.fixture
def preprocessor():
    processor = LLaDA2Preprocessor.__new__(LLaDA2Preprocessor)
    processor.tokenizer = FakeTokenizer()
    processor.soi_id = processor.tokenizer.convert_tokens_to_ids(SOI_TOKEN)
    processor.boi_id = processor.tokenizer.convert_tokens_to_ids(BOI_TOKEN)
    processor.eoi_id = processor.tokenizer.convert_tokens_to_ids(EOI_TOKEN)
    processor.max_seq_len = 8192
    processor.merge_size = 1
    processor.factor = 16
    processor.image_processor = SimpleNamespace(
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
    [((1024, 1024), (32, 32)), ((1600, 200), (16, 64))],
)
def test_edit_preprocess_merge_extract_and_decode(
    preprocessor, decode_probe, monkeypatch, tmp_path, source_size, grid
):
    source = tmp_path / "source.png"
    row_indices, column_indices = np.indices((source_size[1], source_size[0]))
    source_pixels = np.stack(
        (
            row_indices % 256,
            column_indices % 256,
            (row_indices + column_indices) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(source_pixels).save(source)
    generation_params = {
        "cfg_text_scale": 3.5,
        "cfg_image_scale": 1.5,
        "cfg_rescale": 0.6,
        "dllm_steps": 4,
        "decoder_steps": 2,
        "seed": 9,
    }
    payload = StagePayload(
        request_id="e2e",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "make it green"}],
                "images": [str(source)],
            },
            metadata={
                "output_modalities": ["image"],
                "image_generation": generation_params,
            },
            params={"max_new_tokens": 1},
        ),
        data={},
    )
    state = LLaDA2UniPipelineState.from_dict(asyncio.run(preprocessor(payload)).data)
    assert state.task_kind == "edit"
    grid_h, grid_w = grid
    assert state.stream_state["image_info"] == [{"grid_h": grid_h, "grid_w": grid_w}]
    processed_image = preprocess_image_edit(
        [Image.open(source).convert("RGB")], preprocessor.factor
    )[0]
    processed_pixels = (
        torch.from_numpy(np.asarray(processed_image).copy())
        .permute(2, 0, 1)
        .float()
        .mul(preprocessor.image_processor.rescale_factor)
    )
    processed_pixels = (processed_pixels - 0.5) / 0.5
    expected_patch_rows = []
    for row_index in range(grid_h):
        for column_index in range(grid_w):
            patch = processed_pixels[
                :,
                row_index * 16 : (row_index + 1) * 16,
                column_index * 16 : (column_index + 1) * 16,
            ]
            expected_patch_rows.append(
                patch.unsqueeze(1).expand(-1, 2, -1, -1).flatten()
            )
    expected_patch_rows_tensor = torch.stack(expected_patch_rows)
    torch.testing.assert_close(
        state.encoder_inputs[IMAGE_STAGE]["pixel_values"],
        expected_patch_rows_tensor,
    )
    source_tokens = list(range(grid_h * grid_w))
    state.encoder_outs[IMAGE_STAGE] = {"image_token_ids": [source_tokens]}
    merge_image_tokens_for_thinker(state)
    prompt_ids = state.prompt["input_ids"].flatten().tolist()
    assert [tid for tid in prompt_ids if tid >= IMAGE_TOKEN_OFFSET] == [
        IMAGE_TOKEN_OFFSET + token_id for token_id in source_tokens
    ]

    request = build_dllm_thinker_request(
        state,
        params=payload.request.params,
        tokenizer=preprocessor.tokenizer,
        vocab_size=len(preprocessor.tokenizer),
        dllm_config=DllmConfig(
            algorithm="LowConfidenceCFG",
            algorithm_config={
                "threshold": 1.0,
                "image_token_offset": IMAGE_TOKEN_OFFSET,
            },
            block_size=32,
            mask_id=preprocessor.tokenizer.mask_token_id,
            max_running_requests=3,
        ),
        request_id=payload.request_id,
    ).req
    assert list(request.origin_input_ids) == prompt_ids
    no_text_ids = request._uncond_input_ids  # noqa: leading-underscore  # DLLM protocol
    no_image_ids = (
        request._uncond_img_input_ids
    )  # noqa: leading-underscore  # DLLM protocol
    assert [tid for tid in no_text_ids if tid >= IMAGE_TOKEN_OFFSET] == [
        IMAGE_TOKEN_OFFSET + token_id for token_id in source_tokens
    ]
    assert not any(tid >= IMAGE_TOKEN_OFFSET for tid in no_image_ids)
    for branch_ids, pad_len in (
        (
            no_text_ids,
            request._uncond_left_pad_len,
        ),  # noqa: leading-underscore  # DLLM protocol
        (
            no_image_ids,
            request._uncond_img_left_pad_len,
        ),  # noqa: leading-underscore  # DLLM protocol
    ):
        assert len(branch_ids) == len(prompt_ids)
        assert 0 <= pad_len < len(branch_ids)
        assert branch_ids[:pad_len] == [preprocessor.tokenizer.mask_token_id] * pad_len
        assert branch_ids[pad_len] != preprocessor.tokenizer.mask_token_id
    assert request._cfg_scale == 3.5  # noqa: leading-underscore  # DLLM protocol
    assert request._cfg_image_scale == 1.5  # noqa: leading-underscore  # DLLM protocol
    assert request._cfg_rescale == 0.6  # noqa: leading-underscore  # DLLM protocol
    assert request._dllm_steps == 4  # noqa: leading-underscore  # DLLM protocol
    assert request.sampling_params.max_new_tokens == grid_h * grid_w

    state.thinker_out = {
        "output_ids": [IMAGE_TOKEN_OFFSET + token_id for token_id in source_tokens]
        + [2]
    }
    tokens, h, w, params = extract_image_vq_tokens(state)
    assert tokens == source_tokens
    assert (h, w) == grid
    assert params == generation_params

    decoder, _ = decode_probe
    monkeypatch.setattr(stages, "LLaDA2ImageDecoder", lambda **kwargs: decoder)
    scheduler = create_image_decode_executor("unused", device="cpu")
    payload.data = state.to_dict()
    result = scheduler.fn(payload)
    image = Image.open(io.BytesIO(base64.b64decode(result.data["image"])))
    assert image.size == (grid_w * 16, grid_h * 16)
    assert image.getpixel((0, 0)) == (0, 127, 255)
    assert result.data["events"][0]["type"] == "image_final"
    state.thinker_out["output_ids"] = [
        IMAGE_TOKEN_OFFSET + 3,
        IMAGE_TOKEN_OFFSET + 4,
    ]
    with pytest.raises(ValueError, match="does not match"):
        extract_image_vq_tokens(state)


@pytest.mark.parametrize("cfg_scale", [1.0, 4.0])
def test_thinking_stops_at_boundary_then_generates_image(
    preprocessor: LLaDA2Preprocessor,
    monkeypatch: pytest.MonkeyPatch,
    cfg_scale: float,
) -> None:
    payload = StagePayload(
        request_id="thinking-image",
        request=OmniRequest(
            inputs={"messages": [{"role": "user", "content": "A red sailboat."}]},
            metadata={
                "image_generation": {
                    "mode": "thinking",
                    "cfg_scale": cfg_scale,
                    "dllm_steps": 8,
                }
            },
            params={},
        ),
        data={},
    )
    payload = asyncio.run(preprocessor(payload))
    tokenizer = preprocessor.tokenizer
    prompt_tokens = payload.data["prompt"]["input_ids"].numel()
    config = DllmConfig(
        algorithm="LowConfidenceCFG",
        algorithm_config={},
        block_size=32,
        mask_id=tokenizer.mask_token_id,
        max_running_requests=4,
    )
    build, finish = make_dllm_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        dllm_config=config,
    )
    text_request = build(payload)
    assert text_request.req.sampling_params.max_new_tokens == 2048
    assert (
        text_request.req._task_kind == "thinking"
    )  # noqa: leading-underscore  # DLLM protocol
    assert preprocessor.boi_id in text_request.req.eos_token_ids
    # dllm_steps sets the image pass, the thinking text keeps the block schedule.
    assert (
        text_request.req._dllm_steps is None
    )  # noqa: leading-underscore  # DLLM protocol
    assert (
        text_request.req._uncond_input_ids is None
    )  # noqa: leading-underscore  # DLLM protocol

    trace = "A red sailboat."
    text_request.output_ids = (
        tokenizer.encode(trace)
        + preprocessor.build_t2i_header_ids(32, 32)
        + [IMAGE_TOKEN_OFFSET + 7]
    )
    image_payload = finish(text_request)
    image_payload.data = LLaDA2UniPipelineState.from_dict(image_payload.data).to_dict()
    assert thinker_next(payload.request_id, image_payload) == "thinker"
    image_request = build(image_payload)
    assert image_request.req.origin_input_ids[-1] == preprocessor.boi_id
    assert image_request.req.sampling_params.max_new_tokens == 1024
    assert (
        image_request.req._dllm_steps == 8
    )  # noqa: leading-underscore  # DLLM protocol
    assert (
        image_request.req._task_kind == "t2i"
    )  # noqa: leading-underscore  # DLLM protocol
    if cfg_scale > 1:
        assert len(
            image_request.req._uncond_input_ids
        ) == len(  # noqa: leading-underscore  # DLLM protocol
            image_request.req.origin_input_ids
        )
    else:
        assert (
            image_request.req._uncond_input_ids is None
        )  # noqa: leading-underscore  # DLLM protocol
    image_request.output_ids = [IMAGE_TOKEN_OFFSET + 7] * 1024
    output = finish(image_request)
    state = LLaDA2UniPipelineState.from_dict(output.data)
    assert state.thinking_text == trace
    assert thinker_next(payload.request_id, output) == ["decode", "image_decode"]
    assert extract_image_vq_tokens(state)[:3] == ([7] * 1024, 32, 32)

    monkeypatch.setattr(common_module, "load_llada2_tokenizer", lambda path: tokenizer)
    output.data = state.to_dict()
    decoded = create_decode_executor("unused").fn(output)
    thinking_tokens = text_request.output_ids.index(preprocessor.boi_id) + 1
    assert decoded.data["usage"] == {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": thinking_tokens + 1024,
        "total_tokens": prompt_tokens + thinking_tokens + 1024,
    }
    assert decoded.data["text"] == trace

    text_request.output_ids = tokenizer.encode("No image boundary")
    with pytest.raises(RuntimeError, match="did not produce <boi>"):
        finish(text_request)
    assert LLaDA2UniPipelineState.from_dict(payload.data).thinking_phase == "text"


@pytest.mark.parametrize("task_kind", ["chat", "t2i"])
def test_single_pass_usage_is_unchanged(
    preprocessor: LLaDA2Preprocessor,
    monkeypatch: pytest.MonkeyPatch,
    task_kind: str,
) -> None:
    tokenizer = preprocessor.tokenizer
    monkeypatch.setattr(common_module, "load_llada2_tokenizer", lambda path: tokenizer)
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[1, 2, 3]])},
        thinker_out={"output_ids": [IMAGE_TOKEN_OFFSET + 7] * 4},
        task_kind=task_kind,
    )
    payload = StagePayload(
        request_id="single-pass",
        request=OmniRequest(inputs={}, metadata={}, params={}),
        data=state.to_dict(),
    )
    result = create_decode_executor("unused").fn(payload)
    assert result.data["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }


def test_thinking_edit_is_rejected(preprocessor: LLaDA2Preprocessor) -> None:
    payload = StagePayload(
        request_id="thinking-edit",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "Make it red"}],
                "images": ["source.png"],
            },
            metadata={
                "image_generation": {
                    "mode": "thinking",
                }
            },
            params={},
        ),
        data={},
    )
    with pytest.raises(ValueError, match="only supports text-to-image"):
        asyncio.run(preprocessor(payload))
