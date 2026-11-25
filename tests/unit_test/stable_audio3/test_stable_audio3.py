# SPDX-License-Identifier: Apache-2.0
"""Stable Audio 3: request contract, official layout values, batch and length-group invariance, graph buckets and checkpoint detection."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import BatchEncoding

from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.models.stable_audio3 import runtime as runtime_module
from sglang_omni.models.stable_audio3.autoencoder import AudioDecoder
from sglang_omni.models.stable_audio3.conditioning import Conditioner
from sglang_omni.models.stable_audio3.config import StableAudio3PipelineConfig
from sglang_omni.models.stable_audio3.cuda_graphs import (
    DiTCudaGraphRunner,
    PromptEncoderCudaGraphRunner,
    bucket_batch,
    bucket_latents,
    serving_batch_buckets,
)
from sglang_omni.models.stable_audio3.dit import DiffusionTransformer
from sglang_omni.models.stable_audio3.layers import (
    AttentionBackend,
    flash_attention_supported,
)
from sglang_omni.models.stable_audio3.model_config import StableAudio3Config
from sglang_omni.models.stable_audio3.request_builders import (
    MAX_PROMPT_CHARACTERS,
    build_state,
)
from sglang_omni.models.stable_audio3.runtime import (
    GenerationRequest,
    StableAudio3Runtime,
    plan_length_groups,
    sampling_schedule,
)
from sglang_omni.models.stable_audio3.speech_validation import (
    StableAudio3SpeechRequestValidator,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.serve.openai_errors import is_bad_request_error
from sglang_omni.serve.speech_errors import SpeechAPIError
from tests.unit_test.fixtures.accelerator import require_device_streams

PROMPT_TOKENS = 6
TEXT_DIM = 12
# note (ruilin): on an H100 the graphed fp16 audio of the tiny model is 4.4e-4 relative error from eager for every one of 24 seeded requests.
GRAPH_RELATIVE_ERROR_LIMIT = 2e-3


def payload_with(
    params: dict[str, object],
    metadata: dict[str, object] | None = None,
    inputs: object = "ambient pads",
) -> StagePayload:
    return StagePayload(
        request_id="request",
        request=OmniRequest(
            inputs=inputs,
            params=params,
            metadata=(
                {"task": "tts", "tts_params": {}} if metadata is None else metadata
            ),
        ),
        data={},
    )


def test_request_defaults_to_the_official_duration_and_a_random_seed() -> None:
    first, second = (build_state(payload_with({})) for _ in range(2))
    assert first.duration_seconds == 120.0
    assert first.seed != second.seed
    state = build_state(
        payload_with(
            {"seed": 7, "stage_params": {"generation": {"duration_seconds": 30}}}
        )
    )
    assert (state.prompt, state.duration_seconds, state.seed) == ("ambient pads", 30, 7)


def test_requests_that_skip_the_speech_validator_are_refused() -> None:
    """A /generate or chat request skips the speech validator, and /generate can carry the speech metadata itself: chat messages became the prompt, and a negative seed reached the batch."""
    with pytest.raises(ValueError, match="/v1/audio/speech") as error:
        build_state(payload_with({}, metadata={}))
    assert is_bad_request_error(error.value)
    for params, message in (
        ({"seed": -5}, "seed"),
        ({"stage_params": {"generation": {"duration_seconds": 1000}}}, "duration"),
        ({"stage_params": {"generation": {"steps": 4}}}, "steps"),
        ({"stage_params": "fast"}, "stage_params"),
    ):
        with pytest.raises(ValueError, match=message) as error:
            build_state(payload_with(params))
        assert is_bad_request_error(error.value)
    for inputs in (
        "  ",
        [{"role": "user", "content": "ambient pads"}],
        "a" * (MAX_PROMPT_CHARACTERS + 1),
    ):
        with pytest.raises(ValueError, match="prompt") as error:
            build_state(payload_with({}, inputs=inputs))
        assert is_bad_request_error(error.value)


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        ({"voice": "alloy"}, "voice"),
        ({"stream": True, "response_format": "wav"}, "stream"),
        ({"seed": 2**63}, "seed"),
        ({"seed": -1}, "seed"),
        ({"temperature": 0.7}, "temperature"),
        ({"instructions": "dreamy"}, "instructions"),
        ({"stage_params": {"vocoder": {}}}, "stage_params"),
        (
            {"stage_params": {"generation": {"duration_seconds": 381}}},
            "stage_params.generation.duration_seconds",
        ),
        (
            {"stage_params": {"generation": {"duration_seconds": 0.05}}},
            "stage_params.generation.duration_seconds",
        ),
        (
            {"stage_params": {"generation": {"steps": 4}}},
            "stage_params.generation.steps",
        ),
    ],
)
def test_unsupported_requests_are_client_errors(
    fields: dict[str, object], param: str
) -> None:
    validator = StableAudio3SpeechRequestValidator(
        default_model="stable-audio-3",
        supported_speech_fields=StableAudio3PipelineConfig.supported_speech_fields,
    )
    with pytest.raises(SpeechAPIError) as error:
        validator.parse_generation_request({"input": "ambient pads", **fields})
    assert (error.value.status_code, error.value.param) == (400, param)


def tiny_config() -> StableAudio3Config:
    """The Medium layout with tiny widths; the sample geometry matches the release."""
    return StableAudio3Config.model_validate(
        {
            "model_type": "diffusion_cond_inpaint",
            "sample_size": 16777216,
            "sample_rate": 44100,
            "audio_channels": 2,
            "model": {
                "pretransform": {
                    "type": "autoencoder",
                    "chunked": True,
                    "config": {
                        "pretransform": {
                            "type": "patched",
                            "config": {"patch_size": 256, "channels": 2},
                        },
                        "encoder": {"config": {"strides": [16]}},
                        "decoder": {
                            "type": "taae_v2",
                            "config": {
                                "out_channels": 512,
                                "channels": 8,
                                "c_mults": [4],
                                "strides": [16],
                                "latent_dim": 8,
                                "transformer_depths": [2],
                                "sinusoidal_blocks": [2],
                                "sliding_window": [1, 1],
                                "dim_heads": 16,
                                "differential": True,
                                "variable_stride": True,
                                "mask_noise": 0.1,
                            },
                        },
                        "bottleneck": {
                            "type": "softnorm",
                            "config": {
                                "noise_augment_dim": 0,
                                "noise_regularize": True,
                                "auto_scale": True,
                            },
                        },
                        "latent_dim": 8,
                        "downsampling_ratio": 4096,
                    },
                },
                "conditioning": {
                    "configs": [
                        {
                            "id": "prompt",
                            "type": "t5gemma",
                            "config": {
                                "max_length": PROMPT_TOKENS,
                                "padding_mode": "learned",
                                "subfolder": "t5gemma-b-b-ul2",
                            },
                        },
                        {
                            "id": "seconds_total",
                            "type": "number",
                            "config": {
                                "min_val": 0,
                                "max_val": 384,
                                "fourier_features_type": "expo",
                            },
                        },
                    ],
                    "cond_dim": TEXT_DIM,
                },
                "diffusion": {
                    "type": "dit",
                    "diffusion_objective": "rf_denoiser",
                    "config": {
                        "io_channels": 8,
                        "embed_dim": 64,
                        "depth": 2,
                        "num_heads": 1,
                        "cond_token_dim": TEXT_DIM,
                        "global_cond_dim": TEXT_DIM,
                        "local_add_cond_dim": 9,
                        "num_memory_tokens": 4,
                        "global_cond_type": "adaLN",
                        "timestep_features_type": "expo",
                        "norm_type": "rms_norm",
                        "attn_kwargs": {"qk_norm": "rms", "differential": True},
                        "norm_kwargs": {"force_fp32": True},
                        "ff_kwargs": {"mult": 2.0},
                    },
                },
            },
        }
    )


class CharacterTokenizer:
    def __call__(
        self,
        prompts: list[str],
        *,
        truncation: bool,
        max_length: int,
        padding: str,
        return_tensors: str,
    ) -> BatchEncoding:
        input_ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, prompt in enumerate(prompts):
            codes = [ord(character) % 64 for character in prompt[:max_length]]
            input_ids[row, : len(codes)] = torch.tensor(codes)
            attention_mask[row, : len(codes)] = 1
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})


class EmbeddingTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, TEXT_DIM)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> BatchEncoding:
        # note (ruilin): every position also reads the mean of the unmasked tokens, so a replay that kept a stale mask returns different states.
        mask = attention_mask[..., None].to(self.embedding.weight.dtype)
        embeddings = self.embedding(input_ids)
        context = (embeddings * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        return BatchEncoding({"last_hidden_state": embeddings + context})


def tiny_runtime(
    device: torch.device, dtype: torch.dtype, backend: AttentionBackend
) -> StableAudio3Runtime:
    torch.manual_seed(0)
    config = tiny_config()
    dit = DiffusionTransformer(config.model.diffusion.config, backend=backend)
    conditioner = Conditioner(config.model.conditioning)
    decoder = AudioDecoder(config.model.pretransform.config, backend=backend)
    for module in (dit, conditioner, decoder):
        module.to(device=device, dtype=dtype).eval()
    return StableAudio3Runtime(
        config=config,
        dit=dit,
        conditioner=conditioner,
        decoder=decoder,
        tokenizer=CharacterTokenizer(),
        text_encoder=EmbeddingTextEncoder().to(device).eval(),
        device=device,
    )


@pytest.fixture(scope="module")
def runtime() -> StableAudio3Runtime:
    return tiny_runtime(torch.device("cpu"), torch.float32, "sdpa")


def test_layout_values_match_the_official_implementation(
    runtime: StableAudio3Runtime,
) -> None:
    """Latent lengths, valid lengths and noise levels read from the official Medium inference."""
    assert [
        runtime.latent_length(seconds) for seconds in (4.0, 5.0, 20.0, 30.0, 380.0)
    ] == [
        108,
        120,
        280,
        388,
        4096,
    ]
    assert [
        runtime.valid_latent_length(5.0, 120),
        runtime.valid_latent_length(30.0, 388),
    ] == [
        118,
        387,
    ]
    assert torch.allclose(
        sampling_schedule(torch.device("cpu")),
        torch.tensor(
            [1.0, 0.9944, 0.9845, 0.9579, 0.8909, 0.7455, 0.5125, 0.2739, 0.0]
        ),
        atol=1e-4,
    )


def test_batching_does_not_change_any_request(runtime: StableAudio3Runtime) -> None:
    """A short request padded next to a long one, decoded in chunks, matches its own single run."""
    short = GenerationRequest(prompt="lofi", duration_seconds=4.0, seed=3)
    long = GenerationRequest(prompt="orchestral swell", duration_seconds=20.0, seed=4)
    batched = runtime.generate([short, long])
    singles = [runtime.generate([request])[0] for request in (short, long)]
    for batched_audio, single_audio, request in zip(
        batched, singles, (short, long), strict=True
    ):
        assert batched_audio.shape == (2, int(request.duration_seconds * 44100))
        assert batched_audio.abs().max() > 0
        assert torch.allclose(batched_audio, single_audio, atol=1e-6)


@pytest.mark.parametrize(
    ("latent_lengths", "groups"),
    [
        ([388] * 8, [list(range(8))]),
        ([1357, 119, 1357, 119], [[0, 2], [1, 3]]),
        ([119, 248, 4096, 119], [[0, 1, 3], [2]]),
    ],
)
def test_length_groups_split_only_mixed_lengths(
    latent_lengths: list[int], groups: list[list[int]]
) -> None:
    """Equal lengths stay one batch; a long request no longer pads the short ones next to it."""
    planned = plan_length_groups(latent_lengths, overhead_latents=431)
    assert sorted(sorted(group) for group in planned) == groups


def test_grouped_generation_matches_one_batch(
    runtime: StableAudio3Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Split into length groups, every request still returns its own audio, in order."""
    requests = [
        GenerationRequest(prompt="orchestral swell", duration_seconds=20.0, seed=4),
        GenerationRequest(prompt="lofi", duration_seconds=4.0, seed=3),
        GenerationRequest(prompt="rain", duration_seconds=4.0, seed=5),
    ]
    monkeypatch.setattr(runtime_module, "GROUP_OVERHEAD_SECONDS", 0.0)
    grouped = runtime.generate_in_length_groups(requests)
    batched = runtime.generate(requests)
    for grouped_audio, batched_audio in zip(grouped, batched, strict=True):
        assert grouped_audio.shape == batched_audio.shape
        assert torch.allclose(grouped_audio, batched_audio, atol=1e-6)


@pytest.mark.accelerator
@pytest.mark.parametrize("backend", ["flash", "sdpa"])
def test_graphed_compiled_generation_matches_eager_whatever_the_padding_holds(
    backend: AttentionBackend,
) -> None:
    """Graphs recorded at startup are the only ones serving uses; replayed compiled blocks stay close to eager, and a padded batch returns the same bits after a fuller batch left its data in the padding."""
    device = require_device_streams()
    if backend == "flash" and not flash_attention_supported(device):
        pytest.skip("the flash backend needs FlashAttention 3")
    else:
        pass
    runtime = tiny_runtime(device, torch.float16, backend)
    padded_batch = [
        GenerationRequest(prompt="lofi", duration_seconds=4.0, seed=3),
        GenerationRequest(prompt="rain", duration_seconds=5.3, seed=5),
        GenerationRequest(prompt="orchestral swell", duration_seconds=9.0, seed=4),
    ]
    fuller_batch = [
        GenerationRequest(prompt=f"drone {index}", duration_seconds=9.5, seed=index)
        for index in range(4)
    ]
    eager = runtime.generate(padded_batch)
    runtime.compile_blocks()
    graph_backend = current_platform.get_device_graph_backend(device)
    runtime.dit_graphs = DiTCudaGraphRunner(
        runtime.dit, graph_backend=graph_backend, max_batch_size=8
    )
    runtime.prompt_graphs = PromptEncoderCudaGraphRunner(
        runtime.text_encoder,
        graph_backend=graph_backend,
        max_batch_size=8,
        prompt_length=runtime.conditioner.max_length,
    )
    runtime.capture_graphs()
    recorded = (set(runtime.dit_graphs.graphs), set(runtime.prompt_graphs.graphs))
    graphed = runtime.generate(padded_batch)
    runtime.generate(fuller_batch)
    graphed_after_fuller = runtime.generate(padded_batch)
    # note (ruilin): both batches replay one DiT bucket, so the second padded run sees the fuller batch's leftovers.
    shared_bucket = (4, bucket_latents(runtime.latent_length(9.5)))
    assert bucket_latents(runtime.latent_length(9.0)) == shared_bucket[1]
    assert shared_bucket in runtime.dit_graphs.graphs
    assert (
        set(runtime.dit_graphs.graphs),
        set(runtime.prompt_graphs.graphs),
    ) == recorded
    for eager_audio, graphed_audio, later_audio in zip(
        eager, graphed, graphed_after_fuller, strict=True
    ):
        relative_error = (
            (graphed_audio.float() - eager_audio.float()).norm()
            / eager_audio.float().norm()
        ).item()
        assert relative_error < GRAPH_RELATIVE_ERROR_LIMIT
        assert torch.equal(graphed_audio, later_audio)


@pytest.mark.parametrize("max_batch_size", [3, 8, 16])
def test_batch_buckets_fit_every_batch_the_stage_sends(max_batch_size: int) -> None:
    """A graph bucket holds the whole batch, never exceeds the stage batch its buffers are sized for, and is one the startup capture records."""
    recorded = serving_batch_buckets(max_batch_size)
    for batch_size in range(1, max_batch_size + 2):
        bucket = bucket_batch(batch_size, max_batch_size)
        if batch_size > max_batch_size:
            assert bucket is None
        else:
            assert batch_size <= bucket <= max_batch_size
            assert bucket in recorded


@pytest.mark.parametrize(
    ("model_type", "prompt_encoder", "is_stable_audio3"),
    [("diffusion_cond_inpaint", "t5gemma", True), ("diffusion_cond", "t5", False)],
)
def test_checkpoint_layout_detection(
    tmp_path: Path, model_type: str, prompt_encoder: str, is_stable_audio3: bool
) -> None:
    """The checkpoint ships only model_config.json; the Medium layout resolves to this pipeline, Stable Audio Open's does not."""
    (tmp_path / "model_config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "model": {"conditioning": {"configs": [{"type": prompt_encoder}]}},
            }
        )
    )
    if is_stable_audio3:
        config_cls = resolve_config_cls_for_model_path(str(tmp_path))
        assert config_cls is StableAudio3PipelineConfig
    else:
        with pytest.raises(ValueError, match="Could not resolve"):
            resolve_config_cls_for_model_path(str(tmp_path))
