# SPDX-License-Identifier: Apache-2.0

import copy
import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

from sglang_omni.models.zonos3.backbone_kernels import (
    select_experts,
    store_and_mix_values,
)
from sglang_omni.models.zonos3.components import codec, codec_stream, istft_head
from sglang_omni.models.zonos3.components.codec_decoder_kernels import (
    convnext_stack,
    fused_transformer_forward,
    polyphase_resample,
)
from sglang_omni.models.zonos3.components.codec_encoder_kernels import (
    SNAKE_EPSILON,
    biased_snake,
)
from sglang_omni.models.zonos3.components.codec_encoder_layers import (
    encoder_transformer_forward,
)
from sglang_omni.models.zonos3.components.codec_kernels import (
    nearest_codes,
    scaled_residual,
    write_rotary_states,
)
from sglang_omni.models.zonos3.components.codec_layers import (
    ConvNeXtBlock,
    FactorizedVectorQuantizer,
    sinc_resample,
)
from sglang_omni.models.zonos3.components.codec_stream import Zonos3DecoderSlots
from sglang_omni.models.zonos3.components.codec_transformer import (
    ProjectedTransformer,
    TransformerStageSpec,
    rotary_tables,
)
from sglang_omni.models.zonos3.components.encoder_norm_kernels import residual_norm
from sglang_omni.models.zonos3.components.semantic_features import (
    W2vBertSemanticFeatures,
)
from sglang_omni.models.zonos3.gdn_prefill_graph import (
    GDNPrefillGraphTables,
    gdn_extend_in_graph,
)
from sglang_omni.models.zonos3.hf_config import Zonos3Config, zonos3_config_from_params
from sglang_omni.models.zonos3.model_runner import Zonos3ModelRunner
from sglang_omni.models.zonos3.prompt import (
    Zonos3SpeechPreprocessor,
    build_prompt_rows,
    resolve_condition_buckets,
)
from sglang_omni.models.zonos3.request_builders import (
    REPETITION_CODEBOOKS,
    STOP_TOKEN_ID,
    Zonos3GenerationInput,
    Zonos3GenerationResult,
    Zonos3RequestBuilder,
    Zonos3RequestData,
    Zonos3StreamOutputBuilder,
    complete_frame_count,
    undelay_rows,
)
from sglang_omni.models.zonos3.sampling_kernels import NO_CODE, sample_code_rows
from sglang_omni.models.zonos3.sglang_model import Zonos3SGLangModel
from sglang_omni.models.zonos3.stream_steps import (
    Zonos3StreamState,
    is_chunk_ready,
    stable_sample_count,
)
from sglang_omni.models.zonos3.streaming_vocoder import Zonos3StreamingVocoderScheduler
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.platforms import current_platform
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import RequestOutput, SchedulerRequest
from sglang_omni.utils.hf import (
    ZONOS3_ARCHITECTURE,
    try_resolve_arch_from_zonos3_layout,
)

if current_platform.is_cuda():
    from sglang.kernels.ops.attention.flash_attention_v3 import (  # noqa: leading-underscore  # upstream spelling
        _is_fa3_supported,
    )
else:
    pass

# Model section of the released Zonos3-Mini lm/zonos3_mini/params.json (edges shortened).
RELEASED_PARAMS = {
    "model": {
        "n_layers": 11,
        "dim": 1024,
        "head_dim": 32,
        "n_heads": 16,
        "n_kv_heads": 4,
        "ffn_dim_multiplier": 2.0,
        "multiple_of": 64,
        "moe_ffn_dim_multiplier": 1.5,
        "moe_multiple_of": 64,
        "norm_eps": 1e-05,
        "rope_theta": 10000.0,
        "max_seqlen": 32768,
        "n_codebooks": 10,
        "codebook_size": 4096,
        "eoa_id": 4096,
        "audio_pad_id": 4097,
        "text_vocab": 8196,
        "attention_pattern": "gdn3_global1",
        "residual_type": "full_attnres",
        "moe_router_topk": 2,
        "moe_n_experts": 64,
        "moe_alpha": 0.25,
        "loss_softcap": 15.0,
        "condition_bucket_counts": [10, 10, 30],
        "condition_names": ["loudness", "fmax", "speaking_rate"],
        "condition_edges": {
            "loudness": [
                -29.778,
                -26.947,
                -25.047,
                -23.3,
                -21.743,
                -20.2,
                -18.5,
                -16.5,
                -13.9,
            ],
            "fmax": [
                3000.0,
                4000.0,
                5000.0,
                6000.0,
                7000.0,
                8000.0,
                10000.0,
                12000.0,
                16000.0,
            ],
            "speaking_rate": [float(edge) for edge in range(8, 37)],
        },
    }
}
SILENCE_FRAME = torch.tensor([1918, 746, 3544, 234, 1267, 2914, 3944, 2849, 326, 1337])
# The released codec registry's 3 dB output headroom.
HEADROOM_GAIN = 10.0 ** (-3.0 / 20.0)


@pytest.fixture(scope="module")
def config() -> Zonos3Config:
    return zonos3_config_from_params(RELEASED_PARAMS)


def generation_input(
    config: Zonos3Config,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.75,
    top_k: int = 80,
    top_p: float = 0.69,
    repetition_penalty: float = 2.4,
    seed: int | None = None,
) -> Zonos3GenerationInput:
    return Zonos3GenerationInput(
        prompt_rows=build_prompt_rows([2, 3], None, config)[0],
        condition_row_index=0,
        condition_buckets=(10, 10, 30),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )


def test_config_geometry_matches_released_checkpoint_shapes(
    config: Zonos3Config,
) -> None:
    """Shapes from the released model.pth: layers 3 and 7 are global attention, feed_forward.w_in (2, 2048, 1024), latent_down (256, 1024), experts.down_proj (64, 256, 1536), multi_output (40980, 1024), text embedder (8197, 1024)."""
    assert config.full_attention_layer_ids == [3, 7]
    assert config.dense_intermediate_size == 2048
    assert config.expert_latent_size == 256
    assert config.moe_intermediate_size == 1536
    assert config.codebook_count * config.audio_vocab_size == 40980
    assert config.vocab_size == 8197
    assert config.frame_width == 11


def sheared_rows(
    codes: torch.Tensor, trailing_rows: int, config: Zonos3Config
) -> torch.Tensor:
    """Generated rows [rows, codebooks] that carry codes [codebooks, frames] with codebook c on row frame + c; the leading triangle and trailing_rows more rows hold arbitrary codes, as the runner samples them."""
    codebook_count, frame_count = codes.shape
    rows = torch.randint(
        0,
        config.codebook_size,
        (frame_count + codebook_count - 1 + trailing_rows, codebook_count),
    )
    for codebook_index in range(codebook_count):
        rows[codebook_index : codebook_index + frame_count, codebook_index] = codes[
            codebook_index
        ]
    return rows


def test_undelay_reads_only_complete_frames_before_the_end_of_audio_row(
    config: Zonos3Config,
) -> None:
    """Row r carries codebook c of frame r - c; the triangle and the rows after the end frame hold arbitrary codes."""
    codebook_count, frame_count, trailing_rows = config.codebook_count, 5, 4
    codes = torch.randint(0, config.codebook_size, (codebook_count, frame_count))
    rows = sheared_rows(codes, trailing_rows, config)

    torch.testing.assert_close(
        undelay_rows(rows, frame_count, config.codebook_size), codes
    )
    assert (
        undelay_rows(rows, None, config.codebook_size).shape[1]
        == frame_count + trailing_rows
    )
    assert (
        undelay_rows(rows[: codebook_count - 1], None, config.codebook_size).shape[1]
        == 0
    )


class ScriptedFrameModel(torch.nn.Module):
    """Samples the codes passed as hidden states and embeds a frame as its own row, so staged inputs are readable.

    Its float32 tables keep the runner on the torch frame-embedding tail.
    """

    def __init__(self, config: Zonos3Config) -> None:
        super().__init__()
        self.config = config
        self.embedders = torch.nn.ModuleList(
            torch.nn.Embedding(config.text_pad_id + 1, 1)
            for _ in range(config.frame_width)
        )
        self.decode_input_embedding = torch.nn.Embedding(4, config.frame_width)
        self.requires_grad_(False)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            hidden_states, self.config.audio_vocab_size
        ).float()

    def embed_rows(self, rows: torch.Tensor) -> torch.Tensor:
        return rows.float()


def make_runner(model: torch.nn.Module) -> Zonos3ModelRunner:
    return Zonos3ModelRunner(
        SimpleNamespace(
            gpu_id=0,
            model_runner=SimpleNamespace(
                model=model,
                req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(8, 1)),
            ),
        ),
        SGLangOutputProcessor(),
        model=model,
    )


def scripted_request_data(
    config: Zonos3Config,
    request_id: str,
    *,
    is_stream: bool,
    temperature: float = 0.75,
    top_k: int = 80,
    top_p: float = 0.69,
    repetition_penalty: float = 2.4,
    seed: int | None = None,
) -> Zonos3RequestData:
    return Zonos3RequestBuilder(config=config, context_length=64)(
        StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="hi", params={"stream": is_stream}),
            data=generation_input(
                config,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            ),
        )
    )


def generate_scripted_rows(
    config: Zonos3Config, scripts: dict[str, torch.Tensor], *, is_lookahead: bool
) -> tuple[
    dict[str, Zonos3RequestData],
    list[tuple[str, int, torch.Tensor]],
    dict[str, list[torch.Tensor]],
]:
    """Drives the runner and stream output like the scheduler; lookahead launches step N, the prefill included, before resolving step N-1."""
    model = ScriptedFrameModel(config).cuda()
    runner = make_runner(model)
    stream_output_builder = Zonos3StreamOutputBuilder(
        codebook_size=config.codebook_size,
        codebook_count=config.codebook_count,
        vocoder_initial_chunk_frames=2,
        vocoder_steady_chunk_frames=3,
    )
    requests = [
        SchedulerRequest(
            request_id=request_id,
            data=scripted_request_data(
                config, request_id, is_stream=True, temperature=0.0
            ),
        )
        for request_id in scripts
    ]
    launched_row_counts = dict.fromkeys(scripts, 0)
    staged_inputs = []
    streamed_codes = {request_id: [] for request_id in scripts}

    def launch_rows(
        batch: list[SchedulerRequest],
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        hidden_states = torch.stack(
            [
                scripts[request.request_id][launched_row_counts[request.request_id]]
                for request in batch
            ]
        )
        for request in batch:
            launched_row_counts[request.request_id] += 1
        forward_batch = SimpleNamespace(
            batch_size=len(batch),
            req_pool_indices=torch.tensor(
                [1 + list(scripts).index(request.request_id) for request in batch],
                device="cuda",
            ),
            extend_seq_lens=torch.ones(len(batch), dtype=torch.int64),
        )
        result = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=hidden_states.cuda()),
            next_token_ids=None,
        )
        return forward_batch, result

    def stream_step(batch: list[SchedulerRequest], result: SimpleNamespace) -> None:
        for request, token_id in zip(batch, result.next_token_ids.tolist()):
            request_id = request.request_id
            if request.data.req.finished():
                # The scheduler drops a request that stopped in an earlier step before stream output.
                pass
            else:
                streamed_codes[request_id].extend(
                    message.data
                    for message in stream_output_builder(
                        request_id,
                        request.data,
                        RequestOutput(request_id=request_id, data=token_id),
                    )
                )
                if token_id == STOP_TOKEN_ID:
                    request.data.req.finished_reason = FINISH_MATCHED_TOKEN(
                        STOP_TOKEN_ID
                    )
                    streamed_codes[request_id].extend(
                        message.data
                        for message in stream_output_builder.flush(
                            request_id, request.data
                        )
                    )
                else:
                    pass

    forward_batch, result = launch_rows(requests)
    runner.post_prefill(result, forward_batch, None, requests)
    if is_lookahead:
        pending_step = (
            requests,
            result,
            runner.post_decode_launch(result, forward_batch, requests),
        )
    else:
        runner.ensure_next_token_ids(
            result, forward_batch, None, SimpleNamespace(requests=requests)
        )
        stream_step(requests, result)
        pending_step = None
    while True:
        batch = [request for request in requests if not request.data.req.finished()]
        if batch:
            forward_batch, result = launch_rows(batch)
            runner.before_decode(forward_batch, None, batch, is_lookahead=is_lookahead)
            staged_inputs.extend(
                (
                    request.request_id,
                    launched_row_counts[request.request_id] - 1,
                    staged_input.cpu(),
                )
                for request, staged_input in zip(
                    batch, model.decode_input_embedding.weight
                )
            )
            if is_lookahead:
                launched_step = (
                    batch,
                    result,
                    runner.post_decode_launch(result, forward_batch, batch),
                )
            else:
                runner.post_decode(result, forward_batch, None, batch)
                stream_step(batch, result)
                launched_step = None
        elif pending_step is None:
            break
        else:
            launched_step = None
        if pending_step is not None:
            torch.cuda.synchronize()
            pending_batch, pending_result, staged_rows = pending_step
            runner.post_decode_resolve(
                staged_rows, pending_result, None, None, pending_batch
            )
            stream_step(pending_batch, pending_result)
        else:
            pass
        pending_step = launched_step
    return (
        {request.request_id: request.data for request in requests},
        staged_inputs,
        streamed_codes,
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_async_decode_matches_sync_rows_without_the_overrun_step(
    config: Zonos3Config,
) -> None:
    """A step launched before the host reads the previous step, the prefill included, must feed back that step's frame, the step that still carries a stopped request must not add a row, and coalesced stream output must carry every frame once; an end-of-audio sampled in codebook c at row g ends the audio at frame g - c."""
    torch.manual_seed(0)
    # Request id to (row, codebook) of its only end-of-audio code.
    end_of_audio_positions = {"early": (2, 0), "late": (8, 3)}
    end_of_audio_frames = {
        request_id: row_index - codebook_index
        for request_id, (row_index, codebook_index) in end_of_audio_positions.items()
    }
    scripts = {}
    for request_id, (row_index, codebook_index) in end_of_audio_positions.items():
        scripts[request_id] = torch.randint(
            0,
            config.codebook_size,
            (
                end_of_audio_frames[request_id] + config.codebook_count + 2,
                config.codebook_count,
            ),
        )
        scripts[request_id][row_index, codebook_index] = config.end_of_audio_id

    sync_requests, _, sync_streamed_codes = generate_scripted_rows(
        config, scripts, is_lookahead=False
    )
    lookahead_requests, staged_inputs, lookahead_streamed_codes = (
        generate_scripted_rows(config, scripts, is_lookahead=True)
    )

    codebook_indices = torch.arange(config.codebook_count)
    for request_id, end_of_audio_frame in end_of_audio_frames.items():
        request_data = lookahead_requests[request_id]
        assert request_data.end_of_audio_row == end_of_audio_frame
        assert (
            len(request_data.generated_rows)
            == end_of_audio_frame + config.codebook_count
        )
        torch.testing.assert_close(
            torch.stack(request_data.generated_rows),
            torch.stack(sync_requests[request_id].generated_rows),
        )
        spoken_codes = scripts[request_id][
            torch.arange(end_of_audio_frame)[None] + codebook_indices[:, None],
            codebook_indices[:, None],
        ]
        for streamed_codes in (
            sync_streamed_codes[request_id],
            lookahead_streamed_codes[request_id],
        ):
            torch.testing.assert_close(torch.cat(streamed_codes, dim=1), spoken_codes)
    assert {
        request_id
        for request_id, row_index, _ in staged_inputs
        if row_index == len(lookahead_requests[request_id].generated_rows)
    } == set(scripts)
    for request_id, row_index, staged_input in staged_inputs:
        previous_row = lookahead_requests[request_id].generated_rows[row_index - 1]
        torch.testing.assert_close(
            staged_input,
            torch.cat([previous_row, torch.tensor([config.text_pad_id])]).float(),
        )


class SamplerTestModel(torch.nn.Module):
    """The sampler-facing part of Zonos3SGLangModel with its frame embedding; its hidden states are the logits, so a row's draw does not depend on the rows it is batched with."""

    embed_rows = Zonos3SGLangModel.embed_rows

    def __init__(self, config: Zonos3Config) -> None:
        super().__init__()
        self.config = config
        self.embedders = torch.nn.ModuleList(
            [
                torch.nn.Embedding(config.audio_vocab_size, config.hidden_size)
                for _ in range(config.codebook_count)
            ]
            + [torch.nn.Embedding(config.text_pad_id + 1, config.hidden_size)]
        )
        self.decode_input_embedding = torch.nn.Embedding(4, config.hidden_size)
        self.requires_grad_(False)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.float().view(
            -1, self.config.codebook_count, self.config.audio_vocab_size
        )


SAMPLER_DRAWS = 3000
# A logit no sampled code reaches: unmasked, pad would win every draw.
DOMINANT_LOGIT = 1e4


def sampler_state(
    config: Zonos3Config, slot_count: int, window_rows: int
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Per-slot knobs (temperature, top-p, repetition penalty all 1), top-k 80, seeds, steps, and empty windows with their code counts, on CUDA."""
    return (
        torch.ones((3, slot_count), device="cuda"),
        torch.full((slot_count,), 80, dtype=torch.int64, device="cuda"),
        torch.arange(slot_count, dtype=torch.int64, device="cuda") * 1000 + 17,
        torch.zeros(slot_count, dtype=torch.int64, device="cuda"),
        torch.full(
            (slot_count, REPETITION_CODEBOOKS, window_rows),
            NO_CODE,
            dtype=torch.int64,
            device="cuda",
        ),
        torch.zeros(
            (slot_count, REPETITION_CODEBOOKS, config.codebook_size),
            dtype=torch.int32,
            device="cuda",
        ),
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_sampler_draws_each_slot_distribution_with_end_of_audio_in_any_codebook(
    config: Zonos3Config,
) -> None:
    """Codes 7, 8 and end-of-audio hold probabilities 0.5, 0.3 and 0.2 in every codebook and pad the largest logit; each request's draws follow the distribution its own temperature, top-k and top-p leave, read from its slot, and end-of-audio is drawn in every codebook."""
    # (temperature, top-k, top-p) per request and the probabilities of codes 7, 8 and end-of-audio they leave.
    request_knobs = [(1.0, 80, 1.0), (1.0, 2, 1.0), (2.0, 80, 0.5), (0.5, 80, 1.0)]
    expected_probabilities = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.625, 0.375, 0.0],
            [0.5635, 0.4365, 0.0],
            [0.6579, 0.2368, 0.1053],
        ]
    )
    drawn_codes = torch.tensor([7, 8, config.end_of_audio_id])
    slots = torch.tensor([5, 2, 7, 0], device="cuda")
    knobs, top_k, seeds, steps, windows, window_counts = sampler_state(config, 8, 4)
    for slot, (temperature, request_top_k, top_p) in zip(
        slots.tolist(), request_knobs, strict=True
    ):
        knobs[0, slot] = temperature
        knobs[1, slot] = top_p
        top_k[slot] = request_top_k
    logits = torch.full(
        (len(request_knobs), config.codebook_count, config.audio_vocab_size),
        float("-inf"),
        device="cuda",
    )
    logits[:, :, drawn_codes] = torch.tensor([0.5, 0.3, 0.2], device="cuda").log()
    logits[:, :, config.audio_pad_id] = DOMINANT_LOGIT

    draws = []
    for _ in range(SAMPLER_DRAWS):
        draws.append(
            sample_code_rows(
                logits,
                slots,
                knobs,
                top_k,
                seeds,
                steps,
                windows,
                window_counts,
                max_top_k=80,
                pad_id=config.audio_pad_id,
                codebook_size=config.codebook_size,
                softcap=config.logit_softcap,
            )
        )
        steps[slots] += 1
    draws = torch.stack(draws).cpu()

    assert torch.isin(
        draws, drawn_codes
    ).all(), "pad and unlisted codes are never drawn"
    frequencies = torch.stack(
        [(draws == code).float().mean(dim=(0, 2)) for code in drawn_codes], dim=1
    )
    torch.testing.assert_close(frequencies, expected_probabilities, atol=0.02, rtol=0)
    assert (draws[:, 0] == config.end_of_audio_id).any(dim=0).all()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_repetition_penalty_reaches_only_the_window_of_the_leading_codebooks(
    config: Zonos3Config,
) -> None:
    """Greedy draws under penalty 2 with a two-row window: the leading codebooks walk codes 7, 8, 9, 7 as codes enter and leave the window, a positive logit divided and a negative one multiplied; the other codebooks keep 7, and end-of-audio never enters the window."""
    knobs, top_k, seeds, steps, windows, window_counts = sampler_state(config, 2, 2)
    knobs[0] = 0.0
    knobs[2] = 2.0
    slots = torch.tensor([1], device="cuda")
    logits = torch.full(
        (1, config.codebook_count, config.audio_vocab_size),
        float("-inf"),
        device="cuda",
    )
    logits[0, :, 7:10] = torch.tensor([3.0, 2.0, 1.9], device="cuda")
    logits[0, 3:REPETITION_CODEBOOKS, 7:10] = torch.tensor(
        [-1.0, -1.5, -1.6], device="cuda"
    )
    logits[0, 0, config.end_of_audio_id] = 4.0

    draws = []
    for _ in range(4):
        draws.append(
            sample_code_rows(
                logits,
                slots,
                knobs,
                top_k,
                seeds,
                steps,
                windows,
                window_counts,
                max_top_k=80,
                pad_id=config.audio_pad_id,
                codebook_size=config.codebook_size,
                softcap=config.logit_softcap,
            )[0]
        )
        steps[slots] += 1
    draws = torch.stack(draws).cpu()

    assert draws[:, 0].tolist() == [config.end_of_audio_id] * 4
    for codebook_index in range(1, REPETITION_CODEBOOKS):
        assert draws[:, codebook_index].tolist() == [7, 8, 9, 7], codebook_index
    assert (draws[:, REPETITION_CODEBOOKS:] == 7).all()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_backbone_kernels_match_the_float64_residual_mix_and_expert_choice(
    config: Zonos3Config,
) -> None:
    """Each sublayer input is the RMS norm of the softmax mix of the stored outputs under its query's RMS-scaled logits, within one bfloat16 rounding of float64; experts are the top bias-corrected logits with renormalized softmax weights."""
    torch.manual_seed(0)
    token_count, hidden_size = 14, config.hidden_size
    query_count = 2 * config.num_hidden_layers + 1
    epsilon = config.rms_norm_eps
    outputs = torch.randn(
        query_count, token_count, hidden_size, device="cuda"
    ).bfloat16()
    queries = (0.05 * torch.randn(query_count, hidden_size, device="cuda")).bfloat16()
    norm_weight = (1 + 0.1 * torch.randn(hidden_size, device="cuda")).bfloat16()
    values = torch.empty(
        (token_count, query_count, hidden_size), dtype=torch.float32, device="cuda"
    )
    value_logits = torch.empty(
        (token_count, query_count, query_count), dtype=torch.float32, device="cuda"
    )
    mixes = [
        store_and_mix_values(
            outputs[value_index],
            values,
            queries.float(),
            value_logits,
            norm_weight.float(),
            value_index=value_index,
            store_epsilon=epsilon,
            mix_epsilon=epsilon,
            output_dtype=torch.bfloat16,
        )
        for value_index in range(query_count)
    ]
    exact_values = outputs.double().transpose(0, 1)
    exact_logits = (exact_values @ queries.double().T) * torch.rsqrt(
        exact_values.square().mean(dim=-1, keepdim=True) + epsilon
    )
    for query_index in (0, 7, query_count - 1):
        mixed = torch.einsum(
            "tn,tnd->td",
            torch.softmax(exact_logits[:, : query_index + 1, query_index], dim=-1),
            exact_values[:, : query_index + 1],
        )
        expected = mixed * torch.rsqrt(
            mixed.square().mean(dim=-1, keepdim=True) + epsilon
        )
        torch.testing.assert_close(
            mixes[query_index].double(),
            expected * norm_weight.double(),
            rtol=2**-7,
            atol=1e-6,
        )
    router_logits = 3 * torch.randn(token_count, config.num_experts, device="cuda")
    balancing_biases = torch.randn(config.num_experts, device="cuda")
    expert_weights, expert_ids = select_experts(
        router_logits, balancing_biases, top_k=config.num_experts_per_tok
    )
    expected_ids = torch.topk(
        router_logits - balancing_biases, config.num_experts_per_tok, dim=-1
    ).indices
    expected_weights = torch.softmax(router_logits, dim=-1).gather(-1, expected_ids)
    assert torch.equal(expert_ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(
        expert_weights, expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_seeded_requests_draw_the_same_rows_alone_or_batched_in_any_slot(
    config: Zonos3Config,
) -> None:
    """Each request's knobs and seed reach the draw from the slot prefill wrote them to: requests decoded together draw the rows each draws alone in another slot, other seeds draw other rows, and each slot's next frame embedding equals embed_rows bit for bit while the other slots stay untouched."""
    torch.manual_seed(1)
    model = SamplerTestModel(config).to("cuda", torch.bfloat16)
    # (temperature, top-k, top-p, repetition penalty, seed) per request; each knob cuts the draw.
    request_knobs = [
        (1.5, 50, 0.9, 2.4, 11),
        (0.6, 20, 0.95, 1.0, 12),
        (0.0, 80, 0.69, 2.4, 13),
        (1.2, 128, 0.6, 1.3, 14),
    ]
    batched_pool_rows = [3, 1, 4, 2]
    step_logits = 3 * torch.randn(
        4,
        len(request_knobs),
        config.codebook_count * config.audio_vocab_size,
        device="cuda",
    )

    def decode(
        request_indices: list[int], pool_rows: list[int], seed_offset: int
    ) -> tuple[torch.Tensor, Zonos3ModelRunner]:
        """Rows [steps, requests, codebooks] of the requests decoded together in pool_rows, and their runner."""
        runner = make_runner(model)
        requests = [
            SchedulerRequest(
                request_id=f"request-{index}",
                data=scripted_request_data(
                    config,
                    f"request-{index}",
                    is_stream=False,
                    temperature=request_knobs[index][0],
                    top_k=request_knobs[index][1],
                    top_p=request_knobs[index][2],
                    repetition_penalty=request_knobs[index][3],
                    seed=request_knobs[index][4] + seed_offset,
                ),
            )
            for index in request_indices
        ]
        pool_row_tensor = torch.tensor(pool_rows, device="cuda")
        runner.post_prefill(
            SimpleNamespace(
                logits_output=SimpleNamespace(
                    hidden_states=step_logits[0, request_indices]
                ),
                next_token_ids=None,
            ),
            SimpleNamespace(
                req_pool_indices=pool_row_tensor,
                extend_seq_lens=torch.ones(len(requests), dtype=torch.int64),
            ),
            None,
            requests,
        )
        for logits in step_logits[1:]:
            runner.post_decode(
                SimpleNamespace(
                    logits_output=SimpleNamespace(
                        hidden_states=logits[request_indices]
                    ),
                    next_token_ids=None,
                ),
                SimpleNamespace(req_pool_indices=pool_row_tensor),
                None,
                requests,
            )
        return (
            torch.stack(
                [torch.stack(request.data.generated_rows) for request in requests],
                dim=1,
            ),
            runner,
        )

    batched_rows, batched_runner = decode([0, 1, 2, 3], batched_pool_rows, 0)
    reseeded_rows, _ = decode([0, 1, 2, 3], batched_pool_rows, 100)

    for index in range(len(request_knobs)):
        alone_rows, _ = decode([index], [5], 0)
        assert torch.equal(alone_rows[:, 0], batched_rows[:, index]), index
    assert not torch.equal(reseeded_rows, batched_rows)
    last_rows = batched_rows[-1].cuda()
    assert torch.equal(
        batched_runner.feedback_embeddings[batched_pool_rows],
        model.embed_rows(
            torch.cat(
                [last_rows, torch.full_like(last_rows[:, :1], config.text_pad_id)],
                dim=1,
            )
        ).to(torch.bfloat16),
    ), "the frame embedding equals embed_rows bit for bit"
    untouched_slots = torch.ones(8, dtype=torch.bool)
    untouched_slots[batched_pool_rows] = False
    assert not bool(batched_runner.feedback_embeddings[untouched_slots.cuda()].any())


def test_reference_prompt_layout_is_pad_flushed_with_one_row_per_condition(
    config: Zonos3Config,
) -> None:
    reference_codes = torch.randint(0, config.codebook_size, (config.codebook_count, 4))
    text_token_ids = [2, 100, 101, 3]

    prompt_rows, condition_row_index = build_prompt_rows(
        text_token_ids, reference_codes, config
    )
    no_reference_rows, no_reference_index = build_prompt_rows(
        text_token_ids, None, config
    )

    assert condition_row_index == 4 + config.codebook_count - 1
    reference_rows = prompt_rows[:condition_row_index, : config.codebook_count]
    torch.testing.assert_close(
        undelay_rows(reference_rows, None, config.codebook_size), reference_codes
    )
    assert not (prompt_rows == config.end_of_audio_id).any()
    condition_rows = prompt_rows[condition_row_index : condition_row_index + 3]
    assert (condition_rows[:, : config.codebook_count] == config.audio_pad_id).all()
    assert (condition_rows[:, -1] == config.text_pad_id).all()
    assert prompt_rows[condition_row_index + 3 :, -1].tolist() == text_token_ids
    assert no_reference_index == 0
    torch.testing.assert_close(prompt_rows[condition_row_index:], no_reference_rows)


def test_condition_buckets_default_to_unknown_and_values_map_through_edges(
    config: Zonos3Config,
) -> None:
    assert resolve_condition_buckets(config, None, None) == (10, 10, 30)
    assert resolve_condition_buckets(config, {"fmax": 3}, None) == (10, 3, 30)
    assert resolve_condition_buckets(config, None, {"loudness": -40.0}) == (0, 10, 30)
    assert resolve_condition_buckets(config, None, {"loudness": -23.3}) == (4, 10, 30)
    assert resolve_condition_buckets(config, None, {"speaking_rate": 100.0}) == (
        10,
        10,
        29,
    )
    assert resolve_condition_buckets(config, {"loudness": 2}, {"loudness": -40.0}) == (
        2,
        10,
        30,
    )


@pytest.mark.parametrize(
    ("buckets", "values"),
    [
        ({"pitch": 1}, None),
        ({"loudness": 11}, None),
        ({"loudness": 1.5}, None),
        (None, {"fmax": "high"}),
    ],
)
def test_condition_rows_reject_unknown_names_and_bad_ids(
    config: Zonos3Config,
    buckets: dict[str, object] | None,
    values: dict[str, object] | None,
) -> None:
    with pytest.raises(ValueError, match="Zonos3 request"):
        resolve_condition_buckets(config, buckets, values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"temperature": -0.1}, "temperature"),
        ({"top_k": 0}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"repetition_penalty": 0.5}, "repetition_penalty"),
        ({"seed": 2**63}, "seed"),
        ({"max_new_tokens": 100}, "context length"),
    ],
)
def test_request_builder_rejects_invalid_generation_input(
    config: Zonos3Config, overrides: dict[str, int | float], message: str
) -> None:
    builder = Zonos3RequestBuilder(config=config, context_length=64)
    payload = StagePayload(
        request_id="r0",
        request=OmniRequest(inputs="hi"),
        data=generation_input(config, **overrides),
    )

    with pytest.raises(ValueError, match=message):
        builder(payload)


def preprocessor(config: Zonos3Config) -> Zonos3SpeechPreprocessor:
    """A preprocessor with a word-level tokenizer and three codec-silence prefill frames."""
    tokenizer = Tokenizer(
        WordLevel({"hello": 0, "world": 1, "[UNK]": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    return Zonos3SpeechPreprocessor(
        tokenizer=tokenizer,
        text_token_offset=4,
        begin_of_text_id=2,
        end_of_text_id=3,
        config=config,
        reference_encoder=None,
        silence_frame=SILENCE_FRAME,
        silence_prefill_frames=3,
        max_new_tokens=2048,
        max_reference_seconds=60.0,
    )


def test_preprocessor_builds_clone_prompt_with_silence_prefill_and_conditions(
    config: Zonos3Config,
) -> None:
    reference_codes = torch.randint(0, config.codebook_size, (config.codebook_count, 3))
    request = OmniRequest(
        inputs={
            "text": "hello world",
            "references": [{"vq_codes": reference_codes.tolist()}],
        },
        params={"temperature": 0.7, "top_p": 0.9, "seed": 5},
        metadata={"tts_params": {"condition_buckets": {"speaking_rate": 12}}},
    )

    result = preprocessor(config)(
        StagePayload(request_id="r0", request=request, data=None)
    ).data

    expected_rows, expected_index = build_prompt_rows(
        [2, 4, 5, 3], reference_codes, config
    )
    torch.testing.assert_close(result.prompt_rows[:-3], expected_rows)
    silence_rows = result.prompt_rows[-3:]
    assert (silence_rows[:, -1] == config.text_pad_id).all()
    for row_index in range(3):
        assert (
            silence_rows[row_index, : row_index + 1].tolist()
            == SILENCE_FRAME[: row_index + 1].tolist()
        )
        assert (
            silence_rows[row_index, row_index + 1 : config.codebook_count]
            == config.audio_pad_id
        ).all()
    assert result.condition_row_index == expected_index
    assert result.condition_buckets == (10, 10, 12)
    assert result.temperature == 0.7
    assert result.top_p == 0.9
    assert result.seed == 5
    assert result.repetition_penalty == 2.4


def test_prepend_silence_false_leaves_the_prompt_without_prefill_rows(
    config: Zonos3Config,
) -> None:
    request = OmniRequest(
        inputs="hello", metadata={"tts_params": {"prepend_silence": False}}
    )

    result = preprocessor(config)(
        StagePayload(request_id="r0", request=request, data=None)
    ).data

    torch.testing.assert_close(
        result.prompt_rows, build_prompt_rows([2, 4, 3], None, config)[0]
    )


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {
            "inputs": {
                "text": "hello",
                "references": [{"vq_codes": [[1]]}, {"vq_codes": [[1]]}],
            }
        },
        {"inputs": {"text": "hello", "references": [{"vq_codes": [[1, 2]]}]}},
        {"inputs": {"text": "hello", "references": [{"name": "no audio"}]}},
        {"inputs": "hello", "params": {"temperature": True}},
        {"inputs": "hello", "params": {"seed": 1.5}},
        {
            "inputs": "hello",
            "metadata": {"tts_params": {"condition_buckets": [1, 2, 3]}},
        },
        {"inputs": "hello", "metadata": {"tts_params": {"prepend_silence": "yes"}}},
    ],
)
def test_preprocessor_rejects_unsupported_requests(
    config: Zonos3Config, request_kwargs: dict[str, object]
) -> None:
    payload = StagePayload(
        request_id="r0", request=OmniRequest(**request_kwargs), data=None
    )

    with pytest.raises(ValueError, match="Zonos3 request"):
        preprocessor(config)(payload)


def test_zonos3_layout_resolves_its_architecture(tmp_path) -> None:
    """The released params.json names no architecture; the layout marker does."""
    assert try_resolve_arch_from_zonos3_layout(str(tmp_path)) is None
    marker = tmp_path / "lm" / "zonos3_mini" / "params.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")

    assert try_resolve_arch_from_zonos3_layout(str(tmp_path)) == ZONOS3_ARCHITECTURE


@pytest.mark.parametrize("initial_codec_chunk_frames", [None, 0, 1])
def test_stream_output_builder_sends_each_frame_once_by_the_time_a_vocoder_chunk_is_ready(
    config: Zonos3Config,
    small_decoder: codec.Zonos3CodecDecoder,
    initial_codec_chunk_frames: int | None,
) -> None:
    """Every frame reaches the vocoder once and the arbitrary codes of the leading triangle and of the rows past the end never do; a send no later than per-frame sends would make each chunk ready."""
    request_params = {
        "stream": True,
        "initial_codec_chunk_frames": initial_codec_chunk_frames,
    }
    payload = StagePayload(
        request_id="stream",
        request=OmniRequest(inputs="hi", params=request_params),
        data=generation_input(config),
    )
    request_data = Zonos3RequestBuilder(config=config, context_length=64)(payload)
    stream_output_builder = Zonos3StreamOutputBuilder(
        codebook_size=config.codebook_size,
        codebook_count=config.codebook_count,
        vocoder_initial_chunk_frames=2,
        vocoder_steady_chunk_frames=3,
    )
    vocoder = Zonos3StreamingVocoderScheduler(
        small_decoder,
        slots=None,
        headroom_gain=HEADROOM_GAIN,
        initial_chunk_frames=2,
        steady_chunk_frames=3,
        max_step_streams=4,
    )
    vocoder_state = Zonos3StreamState()
    vocoder.latch_stream_contract(
        "stream", vocoder_state, request_params, origin="stream metadata"
    )
    end_of_audio_row = 14
    codes = torch.randint(
        0, config.codebook_size, (config.codebook_count, end_of_audio_row)
    )
    rows = sheared_rows(codes, 1, config)
    request_output = RequestOutput(request_id="stream", data=0)

    streamed_codes = []
    for row_index, delayed_row in enumerate(rows):
        request_data.generated_rows.append(delayed_row)
        if row_index == end_of_audio_row:
            request_data.end_of_audio_row = end_of_audio_row
        else:
            pass
        for message in stream_output_builder("stream", request_data, request_output):
            streamed_codes.append(message.data)
            vocoder.ingest("stream", vocoder_state, message.data)
        per_frame_state = dataclasses.replace(
            vocoder_state,
            frame_count=complete_frame_count(
                len(request_data.generated_rows),
                request_data.end_of_audio_row,
                config.codebook_count,
            ),
        )
        is_ready = is_chunk_ready(vocoder_state, steady_chunk_frames=3)
        assert is_ready == is_chunk_ready(per_frame_state, steady_chunk_frames=3)
        # A ready playing stream may wait for its playback deadline before it decodes.
        if is_ready and row_index % 2 == 1:
            vocoder_state.emitted_sample_count = stable_sample_count(vocoder_state)
        else:
            pass
    streamed_codes.extend(
        message.data for message in stream_output_builder.flush("stream", request_data)
    )

    torch.testing.assert_close(torch.cat(streamed_codes, dim=1), codes)
    assert stream_output_builder.flush("stream", request_data) == []


def build_small_decoder(
    monkeypatch: pytest.MonkeyPatch, *, model_dimension: int, head_count: int
) -> codec.Zonos3CodecDecoder:
    """The real causal decoder with small widths, its stages model_dimension wide over head_count heads; FFT, resampler and attention context geometry are unchanged."""
    torch.manual_seed(42)
    with monkeypatch.context() as patches:
        for constant_name, constant_value in {
            "SEMANTIC_DIMENSION": 8,
            "LATENT_DIMENSION": 8,
            "CONVNEXT_HIDDEN_DIMENSION": 32,
            "CONVNEXT_DEPTH": 1,
        }.items():
            patches.setattr(codec, constant_name, constant_value)
        patches.setattr(istft_head, "ISTFT_HEAD_WIDTH", 16)
        patches.setattr(istft_head, "ISTFT_HEAD_DEPTH", 1)
        patches.setattr(
            codec,
            "DECODER_TRANSFORMER_STAGES",
            tuple(
                TransformerStageSpec(
                    input_dimension=8,
                    model_dimension=model_dimension,
                    output_dimension=16,
                    head_count=head_count,
                    layer_count=1,
                    feedforward_dimension=16,
                    context_frames=stage_spec.context_frames,
                )
                for stage_spec in codec.DECODER_TRANSFORMER_STAGES
            ),
        )
        decoder = codec.Zonos3CodecDecoder().eval()
    with torch.no_grad():
        # Fan-in scaled weights keep the code signal alive through every stage.
        for parameter in decoder.parameters():
            if parameter.dim() > 1:
                parameter.normal_(0.0, parameter[0].numel() ** -0.5)
            else:
                parameter.normal_(0.0, 0.03)
        decoder.to_external_kernel.normal_(0.0, 0.01)
    return decoder


@pytest.fixture
def small_decoder(monkeypatch: pytest.MonkeyPatch) -> codec.Zonos3CodecDecoder:
    return build_small_decoder(monkeypatch, model_dimension=8, head_count=2)


def deliver_codes(
    scheduler: Zonos3StreamingVocoderScheduler,
    request_id: str,
    chunk_index: int,
    codes: torch.Tensor,
) -> None:
    """Deliver one engine chunk, then run decode steps while the serving loop would."""
    scheduler.handle_stream_chunk(
        request_id,
        StreamItem(
            chunk_id=chunk_index,
            from_stage="tts_engine",
            data=codes,
            metadata={"modality": "audio_codes", "stream": True},
        ),
    )
    while scheduler.has_ready_work():
        scheduler.run_ready_step()


def drain_outbox(scheduler: Zonos3StreamingVocoderScheduler) -> list[OutgoingMessage]:
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


def audio_waveform(audio_data: dict[str, object]) -> torch.Tensor:
    """The float32 waveform an audio payload carries."""
    return torch.frombuffer(
        bytearray(audio_data["audio_waveform"]), dtype=torch.float32
    )


@pytest.mark.parametrize("frame_count", [1, 3, 9])
@pytest.mark.parametrize("is_done_before_payload", [False, True])
def test_streamed_audio_matches_offline_decode_and_flushes_tail(
    small_decoder: codec.Zonos3CodecDecoder,
    frame_count: int,
    is_done_before_payload: bool,
) -> None:
    """Later frames must not change emitted audio, and a short request must still receive its tail."""
    codes = torch.randint(0, codec.CODEBOOK_SIZE, (codec.CODEBOOK_COUNT, frame_count))
    payload = StagePayload(
        request_id="stream",
        request=OmniRequest(inputs="hi", params={"stream": True}),
        data=Zonos3GenerationResult(
            codes=codes, prompt_row_count=7, finish_reason="stop"
        ),
    )
    scheduler = Zonos3StreamingVocoderScheduler(
        small_decoder,
        slots=None,
        headroom_gain=HEADROOM_GAIN,
        initial_chunk_frames=2,
        steady_chunk_frames=3,
        max_step_streams=4,
    )

    for frame_index in range(frame_count):
        deliver_codes(
            scheduler, "stream", frame_index, codes[:, frame_index : frame_index + 1]
        )
    early_messages = drain_outbox(scheduler)
    if is_done_before_payload:
        scheduler.handle_stream_done("stream")
        assert scheduler.outbox.empty()
        scheduler.handle_streaming_new_request("stream", payload)
    else:
        scheduler.handle_streaming_new_request("stream", payload)
        scheduler.handle_stream_done("stream")
    while scheduler.has_ready_work():
        scheduler.run_ready_step()
    final_messages = drain_outbox(scheduler)

    # Two initial frames plus the held-back boundary frame precede the first audio.
    assert bool(early_messages) == (frame_count >= 3)
    result_message = final_messages[-1]
    assert result_message.type == "result"
    streamed_waveform = torch.cat(
        [
            audio_waveform(message.data)
            for message in early_messages + final_messages
            if message.type == "stream"
        ]
    )
    # The headroom gain reaches both streamed chunks and whole decodes.
    offline_waveform = small_decoder.decode(codes[None])[0] * HEADROOM_GAIN
    torch.testing.assert_close(streamed_waveform, offline_waveform)
    torch.testing.assert_close(
        audio_waveform(scheduler.decode_full(payload).data), offline_waveform
    )
    assert result_message.data.data["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": frame_count,
        "total_tokens": 7 + frame_count,
    }
    assert result_message.data.data["finish_reason"] == "stop"
    deliver_codes(scheduler, "stream", frame_count, codes[:, :1])
    assert scheduler.outbox.empty()


@pytest.mark.parametrize("is_streaming", [False, True])
def test_result_without_frames_answers_one_frame_of_silence(
    small_decoder: codec.Zonos3CodecDecoder, is_streaming: bool
) -> None:
    """End-of-audio sampled before the first spoken frame finished the request; it was answered with an error instead of audio."""
    payload = StagePayload(
        request_id="silent",
        request=OmniRequest(inputs="hi", params={"stream": is_streaming}),
        data=Zonos3GenerationResult(
            codes=torch.empty((codec.CODEBOOK_COUNT, 0), dtype=torch.int64),
            prompt_row_count=7,
            finish_reason="stop",
        ),
    )
    scheduler = Zonos3StreamingVocoderScheduler(
        small_decoder,
        slots=None,
        headroom_gain=HEADROOM_GAIN,
        initial_chunk_frames=2,
        steady_chunk_frames=3,
        max_step_streams=4,
    )

    if is_streaming:
        scheduler.handle_streaming_new_request("silent", payload)
        scheduler.handle_stream_done("silent")
        while scheduler.has_ready_work():
            scheduler.run_ready_step()
        *stream_messages, result_message = drain_outbox(scheduler)
        assert [message.type for message in stream_messages] == ["stream"]
        assert result_message.type == "result"
        waveform = audio_waveform(stream_messages[0].data)
        result_data = result_message.data.data
    else:
        result_data = scheduler.decode_full(payload).data
        waveform = audio_waveform(result_data)

    torch.testing.assert_close(waveform, torch.zeros(codec.EXTERNAL_SAMPLES_PER_FRAME))
    assert result_data["usage"]["completion_tokens"] == 0
    assert result_data["finish_reason"] == "stop"


@pytest.mark.parametrize("frame_count, is_stream_answered", [(2, True), (40, False)])
def test_non_streaming_decode_yields_to_a_stream_waiting_for_audio(
    small_decoder: codec.Zonos3CodecDecoder, frame_count: int, is_stream_answered: bool
) -> None:
    """Queued non-streaming requests kept the vocoder from every streaming decode, so streams under mixed load got no audio until that traffic stopped; a finished stream with audio buffered may still wait."""
    codes = torch.randint(0, codec.CODEBOOK_SIZE, (codec.CODEBOOK_COUNT, frame_count))
    scheduler = Zonos3StreamingVocoderScheduler(
        small_decoder,
        slots=None,
        headroom_gain=HEADROOM_GAIN,
        initial_chunk_frames=2,
        steady_chunk_frames=3,
        max_step_streams=4,
    )
    deliver_codes(scheduler, "stream", 0, codes)
    scheduler.handle_streaming_new_request(
        "stream",
        StagePayload(
            request_id="stream",
            request=OmniRequest(inputs="hi", params={"stream": True}),
            data=Zonos3GenerationResult(
                codes=codes, prompt_row_count=7, finish_reason="stop"
            ),
        ),
    )
    scheduler.handle_stream_done("stream")
    early_messages = drain_outbox(scheduler)

    scheduler.handle_new_request_batch(
        [
            IncomingMessage(
                "whole",
                "new_request",
                StagePayload(
                    request_id="whole",
                    request=OmniRequest(inputs="hi", params={"stream": False}),
                    data=Zonos3GenerationResult(
                        codes=codes, prompt_row_count=7, finish_reason="stop"
                    ),
                ),
            )
        ]
    )

    # Forty frames play for 2.6 s once their first chunk is sent, far beyond the decode horizon.
    assert [message.type for message in early_messages] == (
        [] if is_stream_answered else ["stream"]
    )
    assert [
        (message.request_id, message.type) for message in drain_outbox(scheduler)
    ] == [("whole", "result")] + (
        [("stream", "stream"), ("stream", "result")] if is_stream_answered else []
    )
    assert scheduler.has_ready_work() != is_stream_answered


def test_codec_prefix_needs_one_boundary_frame_of_holdback(
    small_decoder: codec.Zonos3CodecDecoder,
) -> None:
    """The decoder is causal up to one frame: a prefix matches the full decode except its last frame."""
    codes = torch.randint(0, codec.CODEBOOK_SIZE, (1, codec.CODEBOOK_COUNT, 9))

    full_waveform = small_decoder.decode(codes)[0]
    prefix_waveform = small_decoder.decode(codes[:, :, :3])[0]

    prefix_stable_sample_count = 2 * codec.EXTERNAL_SAMPLES_PER_FRAME
    torch.testing.assert_close(
        prefix_waveform[:prefix_stable_sample_count],
        full_waveform[:prefix_stable_sample_count],
    )
    assert not torch.allclose(
        prefix_waveform[prefix_stable_sample_count:],
        full_waveform[prefix_stable_sample_count : prefix_waveform.numel()],
    )


def test_aborted_stream_does_not_disturb_another_stream(
    small_decoder: codec.Zonos3CodecDecoder,
) -> None:
    scheduler = Zonos3StreamingVocoderScheduler(
        small_decoder,
        slots=None,
        headroom_gain=HEADROOM_GAIN,
        initial_chunk_frames=2,
        steady_chunk_frames=3,
        max_step_streams=4,
    )
    codes = torch.randint(0, codec.CODEBOOK_SIZE, (codec.CODEBOOK_COUNT, 3))

    deliver_codes(scheduler, "aborted", 0, codes[:, :1])
    deliver_codes(scheduler, "surviving", 0, codes[:, :1])
    scheduler.abort("aborted")
    deliver_codes(scheduler, "aborted", 1, codes[:, 1:])
    assert scheduler.outbox.empty()
    deliver_codes(scheduler, "surviving", 1, codes[:, 1:])

    [message] = drain_outbox(scheduler)
    assert message.request_id == "surviving" and message.type == "stream"
    streamed_waveform = audio_waveform(message.data)
    torch.testing.assert_close(
        streamed_waveform,
        small_decoder.decode(codes[None])[0, : streamed_waveform.numel()]
        * HEADROOM_GAIN,
    )


def test_unplayed_stream_decodes_before_a_final_that_is_not_due(
    small_decoder: codec.Zonos3CodecDecoder,
) -> None:
    """A stream that has not played yet must not wait behind the final decode of a stream with a second of audio buffered."""
    scheduler = Zonos3StreamingVocoderScheduler(
        small_decoder,
        slots=None,
        headroom_gain=HEADROOM_GAIN,
        initial_chunk_frames=2,
        steady_chunk_frames=3,
        max_step_streams=1,
    )
    playing_codes = torch.randint(0, codec.CODEBOOK_SIZE, (codec.CODEBOOK_COUNT, 16))
    # Fifteen emitted frames are 1 s of audio, past the playback decode horizon.
    deliver_codes(scheduler, "playing", 0, playing_codes)
    scheduler.handle_streaming_new_request(
        "playing",
        StagePayload(
            request_id="playing",
            request=OmniRequest(inputs="hi", params={"stream": True}),
            data=Zonos3GenerationResult(
                codes=playing_codes, prompt_row_count=7, finish_reason="stop"
            ),
        ),
    )
    scheduler.handle_stream_done("playing")
    drain_outbox(scheduler)
    scheduler.handle_stream_chunk(
        "unplayed",
        StreamItem(
            chunk_id=0,
            from_stage="tts_engine",
            data=torch.randint(0, codec.CODEBOOK_SIZE, (codec.CODEBOOK_COUNT, 3)),
            metadata={"modality": "audio_codes", "stream": True},
        ),
    )

    scheduler.run_ready_step()

    [message] = drain_outbox(scheduler)
    assert message.request_id == "unplayed" and message.type == "stream"


def test_streams_decoded_together_match_their_own_decode(
    small_decoder: codec.Zonos3CodecDecoder,
) -> None:
    """Streams of different lengths share steps and final flushes without changing their audio."""
    scheduler = Zonos3StreamingVocoderScheduler(
        small_decoder,
        slots=None,
        headroom_gain=HEADROOM_GAIN,
        initial_chunk_frames=2,
        steady_chunk_frames=3,
        max_step_streams=4,
    )
    stream_codes = {
        request_id: torch.randint(
            0, codec.CODEBOOK_SIZE, (codec.CODEBOOK_COUNT, frame_count)
        )
        for request_id, frame_count in {"short": 3, "long": 60, "middle": 9}.items()
    }
    for request_id, delivered_frame_count in {
        "short": 3,
        "long": 60,
        "middle": 6,
    }.items():
        scheduler.handle_stream_chunk(
            request_id,
            StreamItem(
                chunk_id=0,
                from_stage="tts_engine",
                data=stream_codes[request_id][:, :delivered_frame_count],
                metadata={"modality": "audio_codes", "stream": True},
            ),
        )
    scheduler.run_ready_step()
    scheduler.handle_stream_chunk(
        "middle",
        StreamItem(
            chunk_id=1,
            from_stage="tts_engine",
            data=stream_codes["middle"][:, 6:],
            metadata={"modality": "audio_codes", "stream": True},
        ),
    )
    for request_id in ("short", "long"):
        scheduler.handle_streaming_new_request(
            request_id,
            StagePayload(
                request_id=request_id,
                request=OmniRequest(inputs="hi", params={"stream": True}),
                data=Zonos3GenerationResult(
                    codes=stream_codes[request_id],
                    prompt_row_count=7,
                    finish_reason="stop",
                ),
            ),
        )
        scheduler.handle_stream_done(request_id)
    while scheduler.has_ready_work():
        scheduler.run_ready_step()

    streamed_waveforms: dict[str, list[torch.Tensor]] = {}
    for message in drain_outbox(scheduler):
        if message.type == "stream":
            streamed_waveforms.setdefault(message.request_id, []).append(
                audio_waveform(message.data)
            )
        else:
            pass
    for request_id, emitted_frame_count in {
        "short": 3,
        "long": 60,
        "middle": 8,
    }.items():
        torch.testing.assert_close(
            torch.cat(streamed_waveforms[request_id]),
            small_decoder.decode(stream_codes[request_id][None])[
                0, : emitted_frame_count * codec.EXTERNAL_SAMPLES_PER_FRAME
            ]
            * HEADROOM_GAIN,
        )


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_cuda(),
    reason="needs the CUDA platform",
)
def test_fused_codec_kernels_round_like_the_eager_ops() -> None:
    """The fused snake, rotary-state, scaled-residual, residual-norm and nearest-code kernels must reproduce the eager chains bit for bit."""
    torch.manual_seed(0)
    alpha = torch.rand(1, 64, 1, device="cuda") * 3 + 0.01
    convolved = (torch.randn(2, 64, 5000, device="cuda") * 4).bfloat16()
    bias = torch.randn(64, device="cuda").bfloat16()
    skip_states = (
        (torch.randn(2, 5000, 64, device="cuda") * 4).bfloat16().transpose(1, 2)
    )
    eager_states = skip_states + (convolved + bias[:, None])
    eager_snake = F.pad(
        eager_states
        + (alpha + SNAKE_EPSILON).reciprocal() * torch.sin(alpha * eager_states).pow(2),
        (6, 0),
    ).bfloat16()
    fused_states, fused_snake = biased_snake(
        convolved, bias, alpha, 6, skip_states, True, uses_fused_encoder=True
    )
    assert torch.equal(fused_states, eager_states)
    assert torch.equal(fused_snake, eager_snake)

    for width in (768, 1280):
        norm = torch.nn.LayerNorm(width, device="cuda")
        torch.nn.init.normal_(norm.weight)
        torch.nn.init.normal_(norm.bias)
        residual = torch.randn(300, width, device="cuda") * 3 + torch.randn(
            300, 1, device="cuda"
        )
        update = torch.randn(300, width, device="cuda").bfloat16()
        update_scale = torch.randn(width, device="cuda")
        fused_sum, fused_normed = residual_norm(residual, norm, update, update_scale)
        assert torch.equal(fused_sum, residual + update * update_scale)
        assert torch.equal(fused_normed, norm(fused_sum).bfloat16())

    quantizer = FactorizedVectorQuantizer(1024, 4096).cuda()
    quantizer.cache_normalized_codebook()
    projected = quantizer.in_proj(torch.randn(1, 1024, 300, device="cuda"))
    queries = F.normalize(projected.transpose(1, 2).reshape(-1, 8), dim=-1)
    entries = F.normalize(quantizer.codebook.weight, dim=-1)
    eager_codes = (
        -(
            queries.pow(2).sum(1, keepdim=True)
            - 2 * queries @ entries.t()
            + entries.pow(2).sum(1)[None]
        )
    ).argmax(1)[None]
    selected = quantizer.codebook(eager_codes).transpose(1, 2)
    fused_codes, straight_through = nearest_codes(
        projected,
        quantizer.codebook_columns,
        quantizer.codebook_squares,
        quantizer.codebook.weight,
    )
    assert torch.equal(fused_codes, eager_codes)
    assert torch.equal(straight_through, projected + (selected - projected))

    batch, frame_count, head_count, head_dimension, leading_frames = 2, 70, 4, 16, 64
    projections = torch.randn(
        batch, frame_count, 3 * head_count * head_dimension, device="cuda"
    ).bfloat16()
    angles = torch.rand(frame_count, head_dimension // 2, device="cuda") * 50
    rotary_cos = angles.cos().repeat_interleave(2, dim=-1)
    rotary_signed_sin = torch.stack((-angles.sin(), angles.sin()), dim=-1).flatten(-2)
    query_key_value = projections.view(
        batch, frame_count, 3, head_count, head_dimension
    ).permute(2, 0, 3, 1, 4)
    eager_states = torch.zeros(
        (3, batch, head_count, leading_frames + frame_count, head_dimension),
        dtype=torch.bfloat16,
        device="cuda",
    )
    fused_states = torch.zeros_like(eager_states)
    eager_states[:2, :, :, leading_frames:] = (
        query_key_value[:2] * rotary_cos
        + query_key_value[:2]
        .unflatten(-1, (head_dimension // 2, 2))
        .flip(-1)
        .flatten(-2)
        * rotary_signed_sin
    )
    eager_states[2, :, :, leading_frames:] = query_key_value[2]
    write_rotary_states(
        projections, rotary_cos, rotary_signed_sin, fused_states, leading_frames
    )
    assert torch.equal(fused_states, eager_states)

    update = torch.randn(batch, frame_count, 96, device="cuda").bfloat16()
    for scale in (
        torch.randn(96, device="cuda"),
        torch.randn(96, device="cuda").bfloat16(),
    ):
        for hidden_states in (
            torch.randn(batch, frame_count, 96, device="cuda"),
            torch.randn(batch, frame_count, 96, device="cuda").bfloat16(),
        ):
            assert torch.equal(
                scaled_residual(hidden_states, update, scale),
                hidden_states + update * scale,
            )


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_cuda(),
    reason="needs the CUDA platform",
)
@pytest.mark.parametrize("frame_count", [30, 200])
def test_encoder_transformer_pass_matches_the_autocast_stage(frame_count: int) -> None:
    """The encoder's fused stage pass, with the attention kernel's own window, must equal the bf16 autocast stage bit for bit, inside and past the attention context."""
    torch.manual_seed(0)
    stage = ProjectedTransformer(
        TransformerStageSpec(
            input_dimension=96,
            model_dimension=128,
            output_dimension=48,
            head_count=2,
            layer_count=2,
            feedforward_dimension=256,
            context_frames=80,
        )
    ).cuda()
    for module in stage.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.05)
            module.bfloat16()
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.normal_(module.weight)
            torch.nn.init.normal_(module.bias)
        else:
            pass
    for layer in stage.layers:
        torch.nn.init.normal_(layer.attention_scale)
        torch.nn.init.normal_(layer.ffn_scale)
    hidden_states = torch.randn(1, 96, frame_count, device="cuda").bfloat16()
    rotary_cos, rotary_signed_sin = rotary_tables(
        torch.arange(256, device="cuda", dtype=torch.float32), 64
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.equal(
            encoder_transformer_forward(
                stage,
                hidden_states,
                rotary_cos,
                rotary_signed_sin,
                uses_fused_encoder=True,
            ),
            stage(hidden_states),
        )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("captured_before_serving", [False, True])
def test_codec_graph_replay_matches_eager_decode(
    small_decoder: codec.Zonos3CodecDecoder, captured_before_serving: bool
) -> None:
    """A decoder graph captured on a shape's second decode or before serving must read each call's codes and reproduce the eager waveform; capturing while serving must keep the cached pinned host blocks other threads reuse."""
    graph_decoder = small_decoder.cuda()
    eager_decoders = [copy.deepcopy(graph_decoder) for _ in range(3)]
    codes = torch.randint(
        0, codec.CODEBOOK_SIZE, (3, 1, codec.CODEBOOK_COUNT, 5), device="cuda"
    )
    if captured_before_serving:
        graph_decoder.capture_decode_graphs()
    else:
        pass
    graph_count = len(graph_decoder.decode_graphs)
    torch.empty(1, pin_memory=True)
    pinned_free_count = torch.cuda.host_memory_stats()["num_host_free"]

    graph_waveforms = [graph_decoder.decode(request_codes) for request_codes in codes]

    assert len(graph_decoder.decode_graphs) == max(graph_count, 1)
    assert torch.cuda.host_memory_stats()["num_host_free"] == pinned_free_count
    for request_codes, eager_decoder, graph_waveform in zip(
        codes, eager_decoders, graph_waveforms
    ):
        torch.testing.assert_close(graph_waveform, eager_decoder.decode(request_codes))


# One bfloat16 ulp of states in [8, 16): the largest fused-minus-eager difference measured over seeds 0-3.
BFLOAT16_STATE_TOLERANCE = 2.0**-4


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_cuda()
    or torch.cuda.get_device_capability()[0] != codec.FLASH_ATTENTION_DEVICE_MAJOR,
    reason="needs an sm90 CUDA GPU",
)
def test_fused_decoder_path_matches_the_eager_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused trunk stage, ConvNeXt stack and matmul resampler must reproduce the eager decoder layers.

    A trunk stage runs within and past its 16-frame attention window; bfloat16 states may differ by one ulp of the largest ones (about 8 here), while a window one frame too wide moves them by more than 1.
    """
    torch.manual_seed(0)
    stage = ProjectedTransformer(
        TransformerStageSpec(
            input_dimension=96,
            model_dimension=128,
            output_dimension=64,
            head_count=2,
            layer_count=2,
            feedforward_dimension=256,
            context_frames=16,
        )
    )
    blocks = torch.nn.ModuleList(ConvNeXtBlock(64, 128) for _ in range(2))
    with torch.no_grad():
        for parameter in [*stage.parameters(), *blocks.parameters()]:
            if parameter.dim() > 1:
                parameter.normal_(0.0, parameter[0].numel() ** -0.5)
            else:
                parameter.normal_(1.0, 0.1)
    stage = stage.cuda().bfloat16()
    blocks = blocks.cuda().bfloat16()
    with torch.inference_mode():
        for frame_count in (12, 100):
            stage_input = torch.randn(2, 96, frame_count, device="cuda").bfloat16()
            torch.testing.assert_close(
                fused_transformer_forward(stage, stage_input),
                stage(stage_input),
                atol=BFLOAT16_STATE_TOLERANCE,
                rtol=0.0,
            )
        block_input = torch.randn(2, 64, 300, device="cuda").bfloat16()
        eager_blocks = block_input
        for block in blocks:
            eager_blocks = block(eager_blocks)
        torch.testing.assert_close(
            convnext_stack(blocks, block_input),
            eager_blocks.transpose(1, 2),
            atol=BFLOAT16_STATE_TOLERANCE,
            rtol=0.0,
        )
        # Without TF32 the strided-window matmul and the strided convolution compute the same float32 sums.
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
        waveform = torch.randn(2, 5000, device="cuda")
        kernel = (
            torch.randn(
                codec.EXTERNAL_RESAMPLE_RATIO,
                1,
                codec.TO_EXTERNAL_KERNEL_TAPS,
                device="cuda",
            )
            * codec.TO_EXTERNAL_KERNEL_TAPS**-0.5
        )
        torch.testing.assert_close(
            polyphase_resample(
                waveform,
                kernel,
                codec.INTERNAL_RESAMPLE_RATIO,
                codec.EXTERNAL_RESAMPLE_RATIO,
            ),
            sinc_resample(
                waveform,
                kernel,
                codec.INTERNAL_RESAMPLE_RATIO,
                codec.EXTERNAL_RESAMPLE_RATIO,
            ),
        )


# note (zhaochenyang20): with bfloat16 stages the seeded small decoder's slotted stream stayed 78-124 dB from its whole decode over eight code seeds on one H100; slots that dropped their ConvNeXt inputs scored 41-44 dB, a quarter of the attention window 64-67 dB and one tail context block fewer 64-65 dB.
SLOTTED_STREAM_MIN_SNR_DB = 70.0


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_cuda()
    or torch.cuda.get_device_capability()[0] != codec.FLASH_ATTENTION_DEVICE_MAJOR
    or not _is_fa3_supported(),
    reason="needs an sm90 CUDA GPU with FA3",
)
def test_slotted_stream_tracks_the_whole_decode_and_frees_its_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream in a decoder slot decodes only its new frames, yet must stream the whole decode's audio within bfloat16 noise; it must free the slot when aborted and when finished."""
    decoder = build_small_decoder(monkeypatch, model_dimension=128, head_count=2)
    decoder = decoder.cuda()
    decoder.transformer_stages.to(torch.bfloat16)
    decoder.head.blocks.to(torch.bfloat16)
    decoder.uses_fused_trunk = True
    monkeypatch.setattr(
        codec_stream, "CONVNEXT_DEPTH", len(decoder.quantizer.semantic_head) - 1
    )
    slots = Zonos3DecoderSlots(
        decoder,
        stream_slots=1,
        row_buckets=(1, 2),
        pass_frame_buckets=(4, 8),
        max_stream_frames=32,
    )
    slots.capture_graphs()
    scheduler = Zonos3StreamingVocoderScheduler(
        decoder,
        slots=slots,
        headroom_gain=HEADROOM_GAIN,
        initial_chunk_frames=2,
        steady_chunk_frames=3,
        max_step_streams=2,
    )
    frame_count = 20
    codes = torch.randint(0, codec.CODEBOOK_SIZE, (codec.CODEBOOK_COUNT, frame_count))

    deliver_codes(scheduler, "aborted", 0, codes[:, :3])
    assert slots.acquire() is None
    scheduler.abort("aborted")
    drain_outbox(scheduler)
    reacquired_stream = slots.acquire()
    assert reacquired_stream is not None
    slots.release(reacquired_stream)
    for frame_index in range(frame_count):
        deliver_codes(
            scheduler, "stream", frame_index, codes[:, frame_index : frame_index + 1]
        )
    assert slots.acquire() is None
    scheduler.handle_streaming_new_request(
        "stream",
        StagePayload(
            request_id="stream",
            request=OmniRequest(inputs="hi", params={"stream": True}),
            data=Zonos3GenerationResult(
                codes=codes, prompt_row_count=7, finish_reason="stop"
            ),
        ),
    )
    scheduler.handle_stream_done("stream")
    while scheduler.has_ready_work():
        scheduler.run_ready_step()

    messages = drain_outbox(scheduler)
    assert [message.request_id for message in messages] == ["stream"] * len(messages)
    assert messages[-1].type == "result"
    streamed_waveform = torch.cat(
        [audio_waveform(message.data) for message in messages[:-1]]
    )
    whole_waveform = decoder.decode(codes[None])[0].cpu() * HEADROOM_GAIN
    assert streamed_waveform.shape == whole_waveform.shape
    signal_to_noise_db = 10 * torch.log10(
        whole_waveform.pow(2).sum() / (streamed_waveform - whole_waveform).pow(2).sum()
    )
    assert signal_to_noise_db > SLOTTED_STREAM_MIN_SNR_DB
    freed_stream = slots.acquire()
    assert freed_stream is not None
    slots.release(freed_stream)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_captured_gdn_prefill_matches_eager_prefill() -> None:
    """A gated delta net prefill captured on one sequence must replay other batches bit-exactly, touching only their state slots."""
    torch.manual_seed(0)
    head_count, head_dimension, kernel_size, slot_count, bucket = 4, 128, 4, 24, 512
    convolution_dimension = 3 * head_count * head_dimension
    layer = RadixLinearAttention(
        layer_id=0,
        num_q_heads=head_count,
        num_k_heads=head_count,
        num_v_heads=head_count,
        head_q_dim=head_dimension,
        head_k_dim=head_dimension,
        head_v_dim=head_dimension,
        conv_weights=torch.randn(convolution_dimension, kernel_size, device="cuda")
        .mul(0.3)
        .bfloat16(),
        A_log=torch.randn(head_count, device="cuda"),
        dt_bias=torch.randn(head_count, device="cuda").bfloat16(),
    )
    convolution_states = torch.zeros(
        slot_count, convolution_dimension, kernel_size - 1, device="cuda"
    ).bfloat16()
    recurrent_states = torch.zeros(
        slot_count, head_count, head_dimension, head_dimension, device="cuda"
    )
    mixed_qkv = torch.zeros(
        bucket, convolution_dimension, device="cuda", dtype=torch.bfloat16
    )
    gates = torch.zeros(bucket, 2 * head_count, device="cuda", dtype=torch.bfloat16)
    tables = GDNPrefillGraphTables.allocate(
        token_count=bucket, max_sequence_count=16, device=torch.device("cuda")
    )

    def fill(
        lengths: list[int], slots: torch.Tensor, prefix_lengths: list[int]
    ) -> None:
        tables.fill(
            extend_sequence_lengths=lengths,
            state_slots=slots,
            extend_prefix_lengths=torch.tensor(prefix_lengths, device="cuda"),
        )

    def replay_body() -> torch.Tensor:
        return gdn_extend_in_graph(
            layer=layer,
            tables=tables,
            convolution_states=convolution_states,
            recurrent_states=recurrent_states,
            mixed_qkv=mixed_qkv,
            a=gates[:, :head_count],
            b=gates[:, head_count:],
        )

    fill([bucket], torch.tensor([slot_count - 1], device="cuda"), [0])
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        replay_body()
    torch.cuda.current_stream().wait_stream(side_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = replay_body()

    for lengths, prefix_lengths in (
        ([81], [0]),
        ([120, 90, 70, 150], [0, 0, 0, 0]),
        ([1, 2, 3, 200, 7], [0, 1, 0, 0, 0]),
    ):
        token_count = sum(lengths)
        slots = torch.randperm(slot_count - 1, device="cuda")[: len(lengths)] + 1
        mixed_qkv.normal_().mul_(50.0)
        gates.normal_().mul_(50.0)
        mixed_qkv[:token_count].normal_()
        gates[:token_count].normal_()
        convolution_states.normal_()
        recurrent_states.normal_()
        eager_convolution_states = convolution_states.clone()
        eager_recurrent_states = recurrent_states.clone()
        cumulative_lengths = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        convolved_qkv = causal_conv1d_fn(
            mixed_qkv[:token_count].transpose(0, 1),
            layer.conv_weights,
            None,
            activation="silu",
            conv_states=eager_convolution_states,
            has_initial_state=torch.tensor(prefix_lengths, device="cuda") > 0,
            cache_indices=slots.int(),
            query_start_loc=cumulative_lengths,
            seq_lens_cpu=lengths,
        ).transpose(0, 1)
        q, k, v = fused_qkv_split_gdn_prefill(
            convolved_qkv, *([head_count] * 3), *([head_dimension] * 3)
        )
        g, beta = fused_gdn_gating(
            layer.A_log,
            gates[:token_count, :head_count],
            gates[:token_count, head_count:],
            layer.dt_bias,
        )
        eager_output, _, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=eager_recurrent_states,
            initial_state_indices=slots.int(),
            cu_seqlens=cumulative_lengths,
            use_qk_l2norm_in_kernel=True,
        )
        other_slots = torch.ones(slot_count, dtype=torch.bool, device="cuda")
        other_slots[slots] = False
        recurrent_before = recurrent_states.clone()
        convolution_before = convolution_states.clone()

        fill(lengths, slots, prefix_lengths)
        graph.replay()

        assert torch.equal(graph_output[:, :token_count], eager_output)
        assert not graph_output[:, token_count:].any()
        assert torch.equal(recurrent_states[slots], eager_recurrent_states[slots])
        assert torch.equal(convolution_states[slots], eager_convolution_states[slots])
        assert torch.equal(recurrent_states[other_slots], recurrent_before[other_slots])
        assert torch.equal(
            convolution_states[other_slots], convolution_before[other_slots]
        )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_teacher_tracks_the_autocast_teacher(tmp_path) -> None:
    """The fused teacher pass, and its torch path off CUDA, must track the bf16 autocast Wav2Vec2BertModel on every lane."""
    torch.manual_seed(0)
    # Weights large enough that every sub-block, the relative-key bias included, moves the output; with the bias dropped it misses by about 19%.
    teacher = Wav2Vec2BertModel(
        Wav2Vec2BertConfig(
            hidden_size=128,
            output_hidden_size=128,
            num_attention_heads=2,
            intermediate_size=256,
            num_hidden_layers=2,
            initializer_range=0.1,
        )
    )
    for layer in teacher.encoder.layers:
        torch.nn.init.normal_(layer.self_attn.distance_embedding.weight)
    teacher.to(torch.bfloat16).save_pretrained(tmp_path)
    teacher = teacher.cuda().eval()
    eager_features = W2vBertSemanticFeatures(tmp_path, codec.SAMPLE_RATE_HZ)
    features = W2vBertSemanticFeatures(tmp_path, codec.SAMPLE_RATE_HZ).cuda()
    input_features = torch.randn(2, 300, 160, device="cuda").bfloat16()

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = teacher(
                input_features=input_features,
                attention_mask=torch.ones(2, 300, dtype=torch.bool, device="cuda"),
            ).last_hidden_state.float()
        fused = features.teacher_states(input_features)
        eager = eager_features.teacher_states(input_features.cpu()).cuda()

    assert (fused - expected).norm() / expected.norm() < 0.02
    assert (eager - expected).norm() / expected.norm() < 0.02
