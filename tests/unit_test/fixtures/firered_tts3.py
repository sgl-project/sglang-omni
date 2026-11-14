# SPDX-License-Identifier: Apache-2.0
"""Shared FireRedTTS3 test assets and slow-boundary doubles."""

from __future__ import annotations

from typing import Literal

import pytest
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import GenerationBatchResult
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.firered_tts3.dit import DiT
from sglang_omni.models.firered_tts3.flow_execution import euler_flow_loop, flow_t_span
from sglang_omni.models.firered_tts3.flow_graph import FlowGraphRunner
from sglang_omni.models.firered_tts3.model_runner import FireRedTTS3ModelRunner
from sglang_omni.models.firered_tts3.modules import PatchEncoder
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.request_builders import (
    FlowState,
    build_sglang_firered_tts3_request,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.types import SchedulerRequest


def make_speaker_tokenizer() -> PreTrainedTokenizerFast:
    """Local tokenizer with special tokens and merged Han tokens for CPU tests."""
    special_tokens = [
        "<unk>",
        "<|Chinese|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|f531_speaker|>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<|audio_holder|>",
        "<|text_holder|>",
        "<|text_pad|>",
        "<|sot|>",
        "<|eot|>",
        "<|sosp|>",
        "<|eosp|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    characters = list(
        dict.fromkeys(
            "".join(chr(codepoint) for codepoint in range(32, 127))
            + "\n你好世界中性活泼晚上好，欢迎收听今天的节目。今天天气很好。欢迎使用点点语音合成系统。源音频文本。公园游乐根据上述音色描述，首先整理成语音属性，再合成以下文本对应的音频："
        )
    )
    tokens = special_tokens + characters + ["你好", "中性", "生僻"]
    tokenizer = Tokenizer(
        models.WordLevel(
            {token: index for index, token in enumerate(tokens)}, unk_token="<unk>"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        Regex(r"<\|[^|]+\|>|你好|中性|生僻|[\s\S]"), behavior="isolated"
    )
    tokenizer.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        additional_special_tokens=special_tokens[1:],
        clean_up_tokenization_spaces=False,
    )


HISTORY_LENGTH = 8
PATCH_SIZE = 4
REDAE_DIM = 64
COND_CHANNELS = 32
N_TIMESTEPS = 10


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        pass
    return torch.device("cuda")


@pytest.fixture
def tiny_dit(device: torch.device) -> DiT:
    return tiny_cpu_dit().to(device=device)


def tiny_runner(
    dit: torch.nn.Module,
    device: torch.device,
    *,
    mode: Literal["fp32", "bf16", "compile_fp32", "compile_bf16"] = "fp32",
    dit_fallback: torch.nn.Module | None = None,
    capture_sizes: tuple[int, ...] = (1, 2, 4),
    max_graphs: int = 8,
) -> FlowGraphRunner:
    return FlowGraphRunner(
        dit,
        flow_t_span(N_TIMESTEPS, device),
        n_timesteps=N_TIMESTEPS,
        patch_size=PATCH_SIZE,
        history_length=HISTORY_LENGTH,
        mode=mode,
        dit_fallback=dit_fallback,
        capture_sizes=capture_sizes,
        max_graphs=max_graphs,
    )


def make_inputs(
    device: torch.device, generator: torch.Generator, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history_latents = torch.randn(
        batch_size, HISTORY_LENGTH, REDAE_DIM, device=device, generator=generator
    )
    initial_noise = torch.randn(
        batch_size, PATCH_SIZE, REDAE_DIM, device=device, generator=generator
    )
    conditioning = torch.randn(
        batch_size,
        HISTORY_LENGTH + PATCH_SIZE,
        COND_CHANNELS,
        device=device,
        generator=generator,
    )
    return history_latents, initial_noise, conditioning


def eager_patch(
    dit: torch.nn.Module,
    history_latents: torch.Tensor,
    initial_noise: torch.Tensor,
    conditioning: torch.Tensor,
    guidance_scales: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return euler_flow_loop(
        dit,
        torch.cat([history_latents, initial_noise], dim=1),
        conditioning,
        flow_t_span(N_TIMESTEPS, device),
        guidance_scales,
        N_TIMESTEPS,
        PATCH_SIZE,
    )


def tiny_cpu_dit() -> DiT:
    torch.manual_seed(0)
    return (
        DiT(
            in_channels=REDAE_DIM + COND_CHANNELS,
            out_channels=REDAE_DIM,
            mlp_ratio=2.0,
            depth=2,
            num_heads=4,
            hidden_size=64,
        )
        .to(dtype=torch.float32)
        .eval()
    )


@pytest.fixture
def disable_flow_optimization_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRERED_TTS3_FFN_GEMM", "0")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_ATOL", "0")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_RTOL", "0")
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "0")
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv1d")


@pytest.fixture
def flow_dispatch_dit(tiny_dit: DiT) -> DiT:
    # note (Codex): a zero head isolates dispatch lifecycle from backend rounding.
    with torch.no_grad():
        tiny_dit.final_layer.linear.weight.zero_()
        tiny_dit.final_layer.linear.bias.zero_()
    return tiny_dit


class SmallFlowModel(torch.nn.Module):
    """Real CPU side path with small hidden dimensions."""

    def __init__(self) -> None:
        super().__init__()
        self.dit: DiT = tiny_cpu_dit()
        self.patch_encoder: PatchEncoder = PatchEncoder(
            64, 8, hidden_size=16, num_heads=2, depth=1
        )
        self.dit_head: torch.nn.Linear = torch.nn.Linear(8, 32)
        self.stop_head: torch.nn.Linear = torch.nn.Linear(8, 1)
        self.patch_size: int = 4
        self.history_patches: int = 2
        self.history_length: int = 8
        self.redae_dim: int = 64
        self.llm_hidden_size: int = 8
        self.spk_proj_dit: torch.nn.Linear | None = None
        self.graph_feedback_buffer: torch.Tensor | None = None
        self.token_embeddings: torch.nn.Embedding = torch.nn.Embedding(151936, 8)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.token_embeddings


def make_flow_runner(
    model: SmallFlowModel, *, batching: bool = False
) -> FireRedTTS3ModelRunner:
    runner = FireRedTTS3ModelRunner.__new__(FireRedTTS3ModelRunner)
    runner.model = model.eval()
    runner.async_enabled = False
    runner.flow_batching = batching
    runner.flow_graphs = None
    runner.control_token_fastpath = True
    runner.skip_min_patch_stop = True
    runner.prefill_launch_at = None
    runner.t_span_cache = {}
    runner.active_requests = {}
    return runner


def make_flow_request(
    index: int = 0,
    timesteps: int = 3,
    *,
    stopped: bool = False,
    dual_stream: bool = False,
    ratio: tuple[int, int] = (2, 1),
    variant: Literal["base", "instruct", "speaker"] = "base",
) -> SchedulerRequest:
    state = FireRedTTS3State(
        variant="speaker" if dual_stream else variant,
        input_ids=[11, 22, 33],
        dual_stream=dual_stream,
        text_stream_ids=[101, 102, 103],
        text_pad_id=151671,
        n_timesteps=timesteps,
        inference_cfg=1.2 + index,
        min_patches=6,
        stop_threshold=0.5 if stopped else 1.1,
    )
    payload = StagePayload(
        request_id=f"row-{index}",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )
    request_data = build_sglang_firered_tts3_request(
        payload,
        patch_size=4,
        vocab_size=151936,
        interleave_ratio=ratio if dual_stream else None,
    )
    request_data.probe = ProfilingProbe(enabled=False)
    request_data.flow_state = FlowState(
        generator=torch.Generator().manual_seed(index),
        prompt_patch_count=0,
        speaker_conditioning=None,
        backbone_conditioning=torch.zeros(1, 2, 8),
        latent_history=torch.zeros(1, 8, 64),
        patch_count=6 if stopped else 0,
    )
    request_data.req.prefix_len = 0
    request_data.req.extend_end = request_data.prefill_token_count
    return SchedulerRequest(request_id=payload.request_id, data=request_data)


def make_flow_result(
    hidden_states: torch.Tensor, sampled_token_ids: list[int] | None = None
) -> GenerationBatchResult:
    return GenerationBatchResult(
        next_token_ids=torch.tensor(
            sampled_token_ids
            if sampled_token_ids is not None
            else [123] * len(hidden_states)
        ),
        logits_output=LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        ),
    )
