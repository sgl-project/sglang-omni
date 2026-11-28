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
from sglang_omni.models.llada2_uni.serial_offload import SerialOffloadCoordinator
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


def test_native_sp2_decoder_config() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "examples/configs/llada2_uni_native_sp2.yaml"
    )
    config = ConfigManager.from_file(str(config_path)).config
    stage = next(stage for stage in config.stages if stage.name == IMAGE_DECODE_STAGE)
    placement = build_stage_placement_plan(config)
    assert set(placement.gpus) == {0, 1, 2}
    assert stage.allow_child_processes and stage.runtime_gpu_ids == [1, 2]
    assert stage.factory.backend == "sglang" and stage.factory.attention_backend == "fa"
    assert stage.factory.ring_degree == 1

    ring_config = ConfigManager(config).merge_config(
        {
            "image_decode.runtime_gpu_ids": [1, 2, 3, 4],
            "image_decode.factory.ring_degree": 4,
        }
    )
    ring_stage = ring_config.stage_named(IMAGE_DECODE_STAGE)
    assert ring_stage.factory.ring_degree == 4
    assert set(build_stage_placement_plan(ring_config).gpus) == {0, 1, 2, 3, 4}


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
    monkeypatch.setattr(stages, "initialize_decoder_runtime", initialize)
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


@pytest.mark.parametrize("failing_phase", [None, "sigvq", "semantic_decoder", "vae"])
def test_decoder_offload_phase_order_and_failure_cleanup(decode_probe, failing_phase):
    decoder, observed = decode_probe
    if failing_phase is None:
        reference = decoder.decode([1, 2], 1, 2, seed=9)
        reference_latents = observed.latents.clone()
    else:
        pass
    coordinator = SerialOffloadCoordinator(torch.device("cpu"))
    decoder.offload_coordinator = coordinator
    transitions = []
    resident = set()

    def transfer(label, device):
        assert coordinator.lock.locked()
        if isinstance(device, torch.device):
            assert not resident
            resident.add(label)
            transitions.append((label, "restore"))
        else:
            assert resident == {label}
            resident.remove(label)
            transitions.append((label, "release"))

    for attribute, label in (
        ("sigvq", "sigvq"),
        ("diff_model", "semantic_decoder"),
        ("vae", "vae"),
    ):
        component = Mock(wraps=getattr(decoder, attribute))
        component.to = Mock(
            side_effect=lambda device, label=label: transfer(label, device)
        )
        if label == "vae":
            component.config = decoder.vae.config
        else:
            pass
        if failing_phase == label:
            if label == "vae":
                component.decode.side_effect = ValueError("phase failed")
            else:
                component.side_effect = ValueError("phase failed")
        else:
            pass
        setattr(decoder, attribute, component)

    if failing_phase is None:
        actual = decoder.decode([1, 2], 1, 2, seed=9)
        assert actual.tobytes() == reference.tobytes()
        torch.testing.assert_close(observed.latents, reference_latents, rtol=0, atol=0)
        phases = ["sigvq", "semantic_decoder", "vae"]
    else:
        with pytest.raises(ValueError, match="phase failed"):
            decoder.decode([1, 2], 1, 2, seed=9)
        phases = ["sigvq", "semantic_decoder", "vae"]
        phases = phases[: phases.index(failing_phase) + 1]
    assert transitions == [
        (phase, action) for phase in phases for action in ("restore", "release")
    ]
    assert not resident
    assert not coordinator.lock.locked()
    assert coordinator.failure is None


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

    monkeypatch.setattr(stages, "load_llada2_tokenizer", lambda path: tokenizer)
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
    monkeypatch.setattr(stages, "load_llada2_tokenizer", lambda path: tokenizer)
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


def test_diffusers_serial_offload_rejects_interleaved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = Mock(spec=LLaDA2ImageDecoder)
    monkeypatch.setattr(stages, "LLaDA2ImageDecoder", Mock(return_value=decoder))
    monkeypatch.setattr(
        stages,
        "resolve_concrete_device",
        lambda device, gpu_id: torch.device(f"cuda:{gpu_id or 0}"),
    )
    scheduler = create_image_decode_executor(
        "unused",
        device="cuda",
        gpu_id=0,
        backend="diffusers",
        enable_serial_offload=True,
    )
    payload = StagePayload(
        request_id="offload-interleaved",
        request=OmniRequest(inputs={}),
        data=LLaDA2UniPipelineState(task_kind="interleaved").to_dict(),
    )
    with pytest.raises(ValueError, match="does not support interleaved"):
        scheduler.fn(payload)


def test_sp_decoder_guards_and_leader_error_broadcast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno
    import gc
    import json
    import threading
    import weakref

    from sglang_omni.models.llada2_uni.components import (
        image_decoder as image_decoder_mod,
    )
    from sglang_omni.models.llada2_uni.components.decoder_model import (
        require_joint_sp_support,
    )
    from sglang_omni.models.llada2_uni.components.image_decoder import (
        validate_checkpoint_sp_config,
    )

    class _LegacyModel(torch.nn.Module):
        def __init__(self, config, hf_config):
            super().__init__()

    class _JointSPModel(torch.nn.Module):
        def __init__(self, config, hf_config, *, shard_joint_sequence: bool = False):
            super().__init__()

    with pytest.raises(RuntimeError, match="shard_joint_sequence"):
        require_joint_sp_support(_LegacyModel)
    require_joint_sp_support(_JointSPModel)

    turbo_dir = tmp_path / "decoder-turbo"
    turbo_dir.mkdir()
    turbo_cfg = turbo_dir / "config.json"
    turbo_cfg.write_text(json.dumps({"n_heads": 30, "n_kv_heads": 30}))
    validate_checkpoint_sp_config(str(tmp_path), "decoder-turbo", 2)
    with pytest.raises(ValueError, match="divisible by Ulysses degree"):
        validate_checkpoint_sp_config(str(tmp_path), "decoder-turbo", 4)

    monkeypatch.setattr(
        image_decoder_mod,
        "hf_hub_download",
        lambda repo_id, filename, local_files_only=False: str(turbo_cfg),
    )
    with pytest.raises(ValueError, match="divisible by Ulysses degree"):
        validate_checkpoint_sp_config("inclusionAI/LLaDA2.0-Uni", "decoder-turbo", 4)

    fresh_dir = tmp_path / "fresh-hub" / "decoder-turbo"
    fresh_dir.mkdir(parents=True)
    fresh_cfg = fresh_dir / "config.json"
    fresh_cfg.write_text(json.dumps({"n_heads": 32, "n_kv_heads": 32}))

    def _hub_prefers_online(repo_id, filename, local_files_only=False):
        if local_files_only:
            raise FileNotFoundError("not cached")
        return str(fresh_cfg)

    from huggingface_hub import constants as hf_constants

    hub_cache = tmp_path / "hf-hub"
    repo_root = hub_cache / "models--inclusionAI--LLaDA2.0-Uni"
    old_snap = repo_root / "snapshots" / ("a" * 40) / "decoder-turbo"
    new_snap = repo_root / "snapshots" / ("b" * 40) / "decoder-turbo"
    old_snap.mkdir(parents=True)
    new_snap.mkdir(parents=True)
    (old_snap / "config.json").write_text(json.dumps({"n_heads": 30, "n_kv_heads": 30}))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home-empty"))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hub_cache))

    monkeypatch.setattr(image_decoder_mod, "hf_hub_download", _hub_prefers_online)
    validate_checkpoint_sp_config("inclusionAI/LLaDA2.0-Uni", "decoder-turbo", 4)

    def _hub_offline(repo_id, filename, local_files_only=False):
        raise FileNotFoundError("offline")

    monkeypatch.setattr(image_decoder_mod, "hf_hub_download", _hub_offline)
    with pytest.raises(ValueError, match="divisible by Ulysses degree"):
        validate_checkpoint_sp_config("inclusionAI/LLaDA2.0-Uni", "decoder-turbo", 4)

    (new_snap / "config.json").write_text(json.dumps({"n_heads": 32, "n_kv_heads": 32}))
    assert (
        image_decoder_mod.resolve_cached_snapshot_config(
            "inclusionAI/LLaDA2.0-Uni", "decoder-turbo"
        )
        is None
    )
    validate_checkpoint_sp_config("inclusionAI/LLaDA2.0-Uni", "decoder-turbo", 4)

    (new_snap / "config.json").unlink()
    refs_dir = repo_root / "refs"
    refs_dir.mkdir()
    (refs_dir / "main").write_text("b" * 40)
    sec_repo = tmp_path / "hf-home-empty" / "hub" / "models--inclusionAI--LLaDA2.0-Uni"
    sec_snap = sec_repo / "snapshots" / ("c" * 40) / "decoder-turbo"
    sec_snap.mkdir(parents=True)
    (sec_snap / "config.json").write_text(json.dumps({"n_heads": 32, "n_kv_heads": 32}))
    (sec_repo / "refs").mkdir()
    (sec_repo / "refs" / "main").write_text("c" * 40)
    with pytest.raises(FileNotFoundError, match="Active cached snapshot"):
        validate_checkpoint_sp_config("inclusionAI/LLaDA2.0-Uni", "decoder-turbo", 4)
    new_snap.rmdir()
    new_snap.parent.rmdir()
    with pytest.raises(FileNotFoundError, match="Active cached snapshot"):
        validate_checkpoint_sp_config("inclusionAI/LLaDA2.0-Uni", "decoder-turbo", 4)

    sp1_decoder = LLaDA2ImageDecoder(str(tmp_path), device="cpu", dtype=torch.float32)
    sp1_diff_calls: list[str] = []
    monkeypatch.setattr(
        sp1_decoder,
        "ensure_sigvq",
        lambda: (_ for _ in ()).throw(FileNotFoundError("no sigvq")),
    )
    monkeypatch.setattr(
        sp1_decoder, "ensure_diff_model", lambda mode: sp1_diff_calls.append(mode)
    )
    with pytest.raises(FileNotFoundError, match="no sigvq"):
        sp1_decoder.decode([1], 1, 1, seed=7)
    assert sp1_diff_calls == []

    sp1_feature_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    sp1_decoder.sigvq = lambda tokens: torch.ones((1, tokens.shape[1], 16))
    sp1_decoder.diff_model = lambda **kw: (
        sp1_feature_refs.append(weakref.ref(kw["cap_feats"][0])),
        ([torch.zeros_like(x) for x in kw["x"]],),
    )[1]
    sp1_decoder.diff_model_mode = "decoder-turbo"
    sp1_decoder.diff_config = {"cap_feat_dim": 16}
    monkeypatch.setattr(sp1_decoder, "ensure_sigvq", lambda: None)
    monkeypatch.setattr(sp1_decoder, "ensure_diff_model", lambda mode: None)
    monkeypatch.setattr(
        sp1_decoder,
        "ensure_vae",
        lambda: (_ for _ in ()).throw(OSError("missing vae config")),
    )

    def _run_sp1_vae_fail() -> str:
        try:
            sp1_decoder.decode(
                [1], 1, 1, decode_mode="decoder-turbo", num_steps=1, seed=7
            )
        except OSError as exc:
            return str(exc)
        raise AssertionError("Expected missing vae config error")

    gc.collect()
    assert ["missing vae config" in _run_sp1_vae_fail() for _ in range(2)] == [
        True,
        True,
    ]
    assert len(sp1_feature_refs) == 2
    assert all(ref() is None for ref in sp1_feature_refs)

    barrier = threading.Barrier(2)
    status_slots: list[str | None] = [None, None]
    feature_src: list[torch.Tensor | None] = [None]
    broadcast_calls: list[int] = []
    local_rank = threading.local()

    def _fake_all_gather_obj(output_list, obj, group):
        rank = local_rank.rank
        status_slots[rank] = obj
        barrier.wait(timeout=5.0)
        output_list[:] = list(status_slots)
        barrier.wait(timeout=5.0)

    def _fake_broadcast(tensor, src, group):
        rank = local_rank.rank
        broadcast_calls.append(rank)
        if rank == 0:
            feature_src[0] = tensor.clone()
        barrier.wait(timeout=5.0)
        if rank != 0:
            assert feature_src[0] is not None
            tensor.copy_(feature_src[0])
        barrier.wait(timeout=5.0)

    monkeypatch.setattr(decoder_runtime.dist, "all_gather_object", _fake_all_gather_obj)
    monkeypatch.setattr(decoder_runtime.dist, "broadcast", _fake_broadcast)
    monkeypatch.setattr(decoder_runtime.dist, "get_global_rank", lambda group, rank: 0)
    fake_ps = SimpleNamespace(
        get_sp_group=lambda: SimpleNamespace(cpu_group="gloo", device_group="nccl")
    )

    def _make_runtime(rank: int) -> decoder_runtime.DecoderRuntimeHandle:
        handle = decoder_runtime.DecoderRuntimeHandle(
            torch.device("cpu"),
            torch.float32,
            "fa",
            fake_ps,
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
        )
        handle.sp_size = 2
        handle.sp_rank = rank
        handle.ulysses_degree = 2
        return handle

    leader = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        dtype=torch.float32,
        backend="sglang",
        runtime=_make_runtime(0),
    )
    peer = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        dtype=torch.float32,
        backend="sglang",
        runtime=_make_runtime(1),
    )
    captured_caps: dict[int, torch.Tensor] = {}
    leader_feature_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def _make_diff_model(rank: int):
        def _diff_model(**kw):
            captured_caps[rank] = kw["cap_feats"][0].clone()
            if rank == 0:
                leader_feature_refs.append(weakref.ref(kw["cap_feats"][0]))
            return ([torch.zeros_like(x) for x in kw["x"]],)

        return _diff_model

    for rank, dec in ((0, leader), (1, peer)):
        dec.diff_model = _make_diff_model(rank)
        dec.diff_config = {
            "cap_feat_dim": 16,
            "all_patch_size": (2,),
            "all_f_patch_size": (1,),
        }
        monkeypatch.setattr(dec, "ensure_diff_model", lambda mode: None)

    leader.sigvq = lambda tokens: torch.full((1, 4, 16), 7.0)
    leader.vae = SimpleNamespace(
        config=SimpleNamespace(scaling_factor=1.0, shift_factor=0.0),
        decode=lambda x, return_dict=False: (torch.zeros((1, 3, 16, 16)),),
    )
    monkeypatch.setattr(leader, "ensure_sigvq", lambda: None)
    monkeypatch.setattr(leader, "ensure_vae", lambda: None)

    def _run_pair(seed: int | None = 7) -> dict[int, object]:
        outcomes: dict[int, object] = {}

        def _worker(rank: int, dec: LLaDA2ImageDecoder) -> None:
            local_rank.rank = rank
            try:
                outcomes[rank] = dec.decode([1], 1, 1, seed=seed)
            except BaseException as exc:
                outcomes[rank] = exc

        threads = [
            threading.Thread(target=_worker, args=(0, leader)),
            threading.Thread(target=_worker, args=(1, peer)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
            assert not thread.is_alive()
        return outcomes

    missing_seed_outcomes = _run_pair(seed=None)
    assert isinstance(missing_seed_outcomes[0], ValueError)
    assert "explicit shared seed" in str(missing_seed_outcomes[0])
    assert isinstance(missing_seed_outcomes[1], ValueError)
    assert "explicit shared seed" in str(missing_seed_outcomes[1])

    sp3_runtime = _make_runtime(0)
    sp3_runtime.sp_size = 3
    sp3_runtime.ulysses_degree = 3
    sp3_decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        dtype=torch.float32,
        backend="sglang",
        runtime=sp3_runtime,
    )
    sp3_sigvq_calls: list[int] = []
    monkeypatch.setattr(sp3_decoder, "ensure_sigvq", lambda: sp3_sigvq_calls.append(1))
    with pytest.raises(ValueError, match="divide evenly"):
        sp3_decoder.decode([0] * (32 * 32), 32, 32, seed=7)
    assert sp3_sigvq_calls == []
    image_decoder_mod.validate_sp_sequence_divisibility(24, 24, 2, 3)

    ok_outcomes = _run_pair()
    assert isinstance(ok_outcomes[0], Image.Image)
    assert ok_outcomes[1] is None
    assert sorted(broadcast_calls) == [0, 1]
    assert torch.equal(captured_caps[1], torch.full((4, 16), 7.0))

    broadcast_calls.clear()
    monkeypatch.setattr(
        leader,
        "ensure_sigvq",
        lambda: (_ for _ in ()).throw(FileNotFoundError("no sigvq")),
    )
    leader_fail = _run_pair()
    assert isinstance(leader_fail[0], FileNotFoundError)
    assert isinstance(leader_fail[1], RuntimeError)
    assert "rank 0: FileNotFoundError: no sigvq" in str(leader_fail[1])
    assert broadcast_calls == []

    monkeypatch.setattr(leader, "ensure_sigvq", lambda: None)
    monkeypatch.setattr(
        peer,
        "ensure_diff_model",
        lambda mode: (_ for _ in ()).throw(
            OSError(errno.EMFILE, "Too many open files")
        ),
    )
    peer_fail = _run_pair()
    assert isinstance(peer_fail[1], OSError)
    assert isinstance(peer_fail[0], RuntimeError)
    assert "rank 1: OSError" in str(peer_fail[0])
    assert broadcast_calls == []

    monkeypatch.setattr(peer, "ensure_diff_model", lambda mode: None)
    monkeypatch.setattr(
        leader,
        "ensure_vae",
        lambda: (_ for _ in ()).throw(OSError("missing vae config")),
    )
    gc.collect()
    vae_fail = _run_pair()
    assert isinstance(vae_fail[0], OSError)
    assert isinstance(vae_fail[1], RuntimeError)
    assert "during output finalization" in str(vae_fail[1])
    assert "rank 0: OSError: missing vae config" in str(vae_fail[1])
    vae_fail.clear()
    assert leader_feature_refs[-1]() is None


def test_native_sp_host_scheduler_and_umm_interleaved_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import multiprocessing

    from sglang_omni.config.runtime import (
        apply_typed_stage_kwargs,
        resolve_stage_typed_kwargs,
    )
    from sglang_omni.models.llada2_uni import native_sp_host
    from sglang_omni.models.llada2_uni.config import LLaDA2UniInterleavedPipelineConfig
    from sglang_omni.native_generation.stage import NativeGenerationScheduler

    turbo_dir = tmp_path / "decoder-turbo"
    turbo_dir.mkdir()
    (turbo_dir / "config.json").write_text(
        json.dumps({"n_heads": 30, "n_kv_heads": 30})
    )

    monkeypatch.setattr(native_sp_host, "require_joint_sp_support", lambda: None)
    monkeypatch.setattr(
        native_sp_host,
        "resolve_concrete_device",
        lambda device, gpu_id: torch.device(f"cuda:{gpu_id or 0}"),
    )

    stub_decoder = object.__new__(native_sp_host.NativeSPDecoder)
    stub_decoder.args = SimpleNamespace()
    stub_decoder.sp_size = 2
    stub_decoder.decode_mode = "normal"
    stub_decoder.num_steps = 17
    stub_decoder.resolution_multiplier = 1
    req_metrics = SimpleNamespace(stages={"semantic_decode": 1.0})
    captured_params = []
    monkeypatch.setattr(
        native_sp_host,
        "prepare_request",
        lambda args, params: (
            captured_params.append(params),
            SimpleNamespace(extra={}, metrics=req_metrics),
        )[1],
    )
    req = stub_decoder.build_request([3], 1, 1, seed=-1)
    assert captured_params[0].seed == 0
    assert (captured_params[0].width, captured_params[0].height) == (16, 16)
    assert req.extra["llada_vq"] == ([3], 1, 1, -1, "normal", 17, 1)
    stub_decoder.sp_size = 3
    with pytest.raises(ValueError, match="divide evenly"):
        stub_decoder.build_request(
            [0] * (32 * 32), 32, 32, seed=7, resolution_multiplier=2
        )
    ok_sp3 = stub_decoder.build_request(
        [0] * (24 * 24), 24, 24, seed=7, resolution_multiplier=2
    )
    assert ok_sp3.extra["llada_vq"][1:3] == (24, 24)
    stub_decoder.sp_size = 2
    ok_sp2 = stub_decoder.build_request(
        [0] * (32 * 32), 32, 32, seed=7, resolution_multiplier=2
    )
    assert ok_sp2.extra["llada_vq"][1:3] == (32, 32)
    with pytest.raises(RuntimeError, match="worker boom"):
        native_sp_host.NativeSPDecoder.extract_png(
            SimpleNamespace(error="worker boom", output_image_bytes=[b"png"])
        )
    with pytest.raises(RuntimeError, match="no PNG output"):
        native_sp_host.NativeSPDecoder.extract_png(
            SimpleNamespace(error=None, output_image_bytes=[])
        )

    from sglang_omni.models.llada2_uni.native_sp import SemanticDecode

    sync_calls: list[tuple[BaseException | None, str]] = []

    def _stub_sync(err: BaseException | None, stage: str = "") -> None:
        sync_calls.append((err, stage))
        if err is not None:
            raise err
        else:
            pass

    stage_obj = object.__new__(SemanticDecode)
    stage_obj.runtime = SimpleNamespace(
        compute_context=nullcontext, sync_rank_status=_stub_sync
    )
    stage_obj.decoder = SimpleNamespace(
        decode=Mock(side_effect=[Image.new("RGB", (4, 4)), None])
    )
    leader_out = stage_obj.forward(req, SimpleNamespace())
    peer_out = stage_obj.forward(req, SimpleNamespace())
    assert len(leader_out.output_image_bytes) == 1
    assert not peer_out.output_image_bytes
    assert leader_out.metrics is req_metrics
    assert peer_out.metrics is req_metrics
    assert sync_calls == [
        (None, "during output finalization"),
        (None, "during output finalization"),
    ]
    assert stage_obj.decoder.decode.call_args.kwargs == {
        "seed": -1,
        "decode_mode": "normal",
        "num_steps": 17,
        "resolution_multiplier": 1,
    }
    failing_image = Mock(save=Mock(side_effect=OSError("png encode failed")))
    stage_obj.decoder = SimpleNamespace(decode=Mock(return_value=failing_image))
    sync_calls.clear()
    with pytest.raises(OSError, match="png encode failed"):
        stage_obj.forward(req, SimpleNamespace())
    assert len(sync_calls) == 1
    assert isinstance(sync_calls[0][0], OSError)
    assert sync_calls[0][1] == "during output finalization"

    reader, writer = multiprocessing.Pipe(duplex=False)
    fake_worker = SimpleNamespace(pid=4242, exitcode=-9, sentinel=reader.fileno())
    fake_engine = SimpleNamespace(
        local_scheduler_process=[fake_worker],
        shutdown=Mock(),
    )
    built_requests = []

    class _FakeNativeSPDecoder:
        def __init__(self, model_path, gpu_id, devices, ring_degree, **kwargs):
            self.kwargs = kwargs
            self.args = SimpleNamespace()
            self.engine = fake_engine

        def build_request(self, tokens, height, width, **kwargs):
            built_requests.append((tokens, height, width, kwargs))
            return SimpleNamespace(extra={"llada_vq": (tokens, height, width)})

        extract_png = staticmethod(native_sp_host.NativeSPDecoder.extract_png)

        def close(self):
            self.engine.shutdown()

    monkeypatch.setattr(native_sp_host, "NativeSPDecoder", _FakeNativeSPDecoder)

    with pytest.raises(ValueError, match="divisible by Ulysses degree"):
        native_sp_host.create_native_decoder(
            str(tmp_path),
            gpu_id=0,
            runtime_gpu_ids=[0, 1, 2, 3],
            ring_degree=1,
        )

    interleaved_cfg = ConfigManager(
        LLaDA2UniInterleavedPipelineConfig(model_path=str(tmp_path))
    ).merge_config(
        {
            "image_decode.gpu": 1,
            "image_decode.runtime_gpu_ids": [1, 2],
            "image_decode.allow_child_processes": True,
            "image_decode.factory.backend": "sglang",
            "image_decode.factory.attention_backend": "fa",
            "image_decode.factory.dtype": "bf16",
            "image_decode.factory.decode_mode": "decoder-turbo",
            "image_decode.factory.num_steps": 8,
            "image_decode.factory.ring_degree": 2,
        }
    )
    stage_cfg = interleaved_cfg.stage_named(IMAGE_DECODE_STAGE)
    assert stage_cfg.next == "orchestrator"
    typed_args = apply_typed_stage_kwargs(
        create_image_decode_executor,
        {"runtime_gpu_ids": [0, 1]},
        resolve_stage_typed_kwargs(stage_cfg),
        stage_name=IMAGE_DECODE_STAGE,
    )
    scheduler = create_image_decode_executor(str(tmp_path), gpu_id=0, **typed_args)
    try:
        assert isinstance(scheduler, NativeGenerationScheduler)
        assert scheduler.requires_tp_work_fanout is False
        assert scheduler.claim_result(
            StagePayload(request_id="r", request=OmniRequest(inputs={}), data={}),
            terminal=False,
        )
        block_event = __import__("threading").Event()

        def _blocked_forward(batch):
            block_event.wait(timeout=5.0)
            return SimpleNamespace(error=None, output_image_bytes=[b"png"])

        monkeypatch.setattr(
            native_sp_host.sync_scheduler_client, "forward", _blocked_forward
        )
        writer.close()
        state = LLaDA2UniPipelineState(
            task_kind="interleaved",
            thinker_out={"output_ids": [IMAGE_TOKEN_OFFSET + 3]},
            stream_state={
                "interleaved": {
                    "segments": [{"grid_h": 1, "grid_w": 1, "vq_tokens": [3]}]
                }
            },
            request_metadata={"image_generation": {"seed": -1}},
        )
        payload = StagePayload(
            request_id="dead-worker",
            request=OmniRequest(inputs={}),
            data=state.to_dict(),
        )
        with pytest.raises(RuntimeError, match="exited with code -9"):
            scheduler.fn(payload)
        block_event.set()
    finally:
        scheduler.stop()
        reader.close()
