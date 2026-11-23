# SPDX-License-Identifier: Apache-2.0
"""LFM3 runner and engine contracts: mixed-batch speech steps, retraction replay and refused server settings."""

from __future__ import annotations

import types
from dataclasses import fields

import pytest
import torch
from torch import nn

from sglang_omni.models.lfm3_audio import model_runner as model_runner_module
from sglang_omni.models.lfm3_audio.depth import FrameSampling, SpeechSampler
from sglang_omni.models.lfm3_audio.engine_builder import Lfm3AudioEngineBuilder
from sglang_omni.models.lfm3_audio.generation import SpeechState
from sglang_omni.models.lfm3_audio.hf_config import Lfm3AudioTokens
from sglang_omni.models.lfm3_audio.model_runner import Lfm3AudioModelRunner
from sglang_omni.models.lfm3_audio.request_builders import Lfm3AudioRequestData
from sglang_omni.models.lfm3_audio.speech_pool import SpeechRows, SpeechStatePool
from sglang_omni.scheduling.types import SchedulerRequest

HIDDEN_SIZE = 4
TOKENS = Lfm3AudioTokens()


def make_runner() -> Lfm3AudioModelRunner:
    runner = Lfm3AudioModelRunner.__new__(Lfm3AudioModelRunner)
    embed_tokens = nn.Embedding(200_000, HIDDEN_SIZE)
    codebook_embedding = nn.Embedding(
        TOKENS.num_codebooks * TOKENS.codebook_size, HIDDEN_SIZE
    )
    offsets = torch.arange(TOKENS.num_codebooks) * TOKENS.codebook_size
    runner.model = types.SimpleNamespace(
        get_input_embeddings=lambda: embed_tokens,
        frame_embeddings=lambda codes: codebook_embedding(codes + offsets).sum(-2),
        decode_input_embed_buffer=torch.zeros(2, HIDDEN_SIZE),
        text_only_buffer=torch.zeros(2, dtype=torch.bool),
        depth_decoder=None,
        depth_graphs=None,
    )
    runner.tokens = TOKENS
    runner.stop_token_id = TOKENS.eos_token_ids[0]
    runner.speech_pool = None
    return runner


def make_request(request_id: str, last_prompt_token: int) -> SchedulerRequest:
    speech = SpeechState.from_prompt(
        TOKENS,
        last_prompt_token,
        text_only=False,
        sampling=FrameSampling(
            temperature=0.0,
            top_k=None,
            repetition_penalty=1.0,
            repetition_window=4,
            generator=torch.Generator(),
        ),
    )
    request_data = Lfm3AudioRequestData(
        input_ids=torch.tensor([11, 12, last_prompt_token]),
        req=types.SimpleNamespace(
            prefix_len=0,
            extend_len=3,
            output_ids=[],
            is_retracted=False,
            finished=lambda: False,
        ),
        speech=speech,
    )
    return SchedulerRequest(request_id=request_id, data=request_data)


def run_speech_step(
    runner: Lfm3AudioModelRunner,
    requests: list[SchedulerRequest],
    sampled_token_ids: list[int],
    frames: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
    """One step: the speaking rows get frames, then the scheduler appends the (overridden) token ids."""
    monkeypatch.setattr(runner, "sample_speaking_frames", lambda *args: frames)
    result = types.SimpleNamespace(
        logits_output=types.SimpleNamespace(
            hidden_states=torch.zeros(len(requests), HIDDEN_SIZE)
        ),
        next_token_ids=torch.tensor(sampled_token_ids),
    )
    runner.speech_step(result, requests)
    next_token_ids = result.next_token_ids.tolist()
    for request, token_id in zip(requests, next_token_ids, strict=True):
        request.data.req.output_ids.append(token_id)
    return next_token_ids


def test_mixed_batch_maps_each_frame_to_its_speaking_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a batch of a text row and a speaking row, the frame, its feedback and audio_out go to the speaking row only."""
    runner = make_runner()
    text_request = make_request("text", 1000)
    speech_request = make_request("speech", TOKENS.audio_start_token_id)
    frame = torch.full((1, TOKENS.num_codebooks), 7)
    next_token_ids = run_speech_step(
        runner, [text_request, speech_request], [42, 43], frame, monkeypatch
    )
    assert next_token_ids == [42, TOKENS.audio_out_token_id]
    assert text_request.data.speech.text_out == [42]
    assert not text_request.data.pending_feedback_queue
    assert torch.equal(speech_request.data.feedback_frame_codes[0], frame[0])
    assert len(speech_request.data.pending_feedback_queue) == 1


def test_text_end_in_text_mode_stops_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler stops on im_end only, so a text-mode text_end is sent on as im_end."""
    runner = make_runner()
    request = make_request("text", 1000)
    next_token_ids = run_speech_step(
        runner,
        [request],
        [TOKENS.text_end_token_id],
        torch.empty(0, TOKENS.num_codebooks),
        monkeypatch,
    )
    assert next_token_ids == [runner.stop_token_id]


def test_retracted_request_replays_the_inputs_its_decode_steps_fed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retracted request is prefilled again over its prompt and output; each output position must get the embedding its decode step fed (a frame embedding for a speech frame), not the token table's row for audio_out."""
    runner = make_runner()
    request = make_request("speech", TOKENS.audio_start_token_id)
    sglang_request = request.data.req
    prompt_embeds = runner.prefill_input_embeds(request)
    fed_inputs = []
    for code in (3, 4):
        frame = torch.full((1, TOKENS.num_codebooks), code)
        run_speech_step(runner, [request], [42], frame, monkeypatch)
        runner.before_decode(None, None, [request])
        fed_inputs.append(runner.model.decode_input_embed_buffer[0].clone())
    # The third frame is sampled, and its embedding queued, when the request is retracted.
    run_speech_step(
        runner,
        [request],
        [42],
        torch.full((1, TOKENS.num_codebooks), 5),
        monkeypatch,
    )
    fed_inputs.append(request.data.pending_feedback_queue[0].clone())

    sglang_request.extend_len = 3 + len(sglang_request.output_ids)
    replay = runner.prefill_input_embeds(request)

    assert sglang_request.output_ids == [TOKENS.audio_out_token_id] * 3
    assert torch.allclose(replay, torch.cat([prompt_embeds, torch.stack(fed_inputs)]))
    assert not request.data.pending_feedback_queue


def test_device_speech_steps_match_the_host_state_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seeded runner advances speech on the device; its emitted ids, fed inputs and host replay match the host state machine step for step, through both the synchronous and the launch/resolve path."""
    host_runner = make_runner()
    device_runner = make_runner()
    device_runner.model = types.SimpleNamespace(
        **{
            **vars(host_runner.model),
            "decode_input_embed_buffer": torch.zeros(3, HIDDEN_SIZE),
            "text_only_buffer": torch.zeros(3, dtype=torch.bool),
        }
    )
    host_runner.model.decode_input_embed_buffer = torch.zeros(3, HIDDEN_SIZE)
    host_runner.model.text_only_buffer = torch.zeros(3, dtype=torch.bool)
    device_runner.speech_pool = SpeechStatePool(
        slot_count=4,
        hidden_size=HIDDEN_SIZE,
        window=4,
        num_codebooks=TOKENS.num_codebooks,
        embed_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        device_runner,
        "next_host_staging",
        lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )
    prompts = (TOKENS.audio_start_token_id, TOKENS.mixed_start_token_id, 1000)
    host_requests = [make_request(f"r{i}", token) for i, token in enumerate(prompts)]
    device_requests = [make_request(f"r{i}", token) for i, token in enumerate(prompts)]
    forward_batch = types.SimpleNamespace(req_pool_indices=torch.tensor([3, 1, 4]))
    generator = torch.Generator().manual_seed(0)
    choices = torch.tensor(
        [1000, 1001, TOKENS.audio_start_token_id, TOKENS.text_end_token_id]
    )

    def step_inputs(step: int) -> tuple[list[int], torch.Tensor]:
        sampled = choices[torch.randint(0, 4, (3,), generator=generator)].tolist()
        frames = torch.randint(0, 50, (3, TOKENS.num_codebooks), generator=generator)
        if step in (9, 25):
            # An end-of-speech frame for the first request.
            frames[0, 5] = TOKENS.audio_eos_code
        else:
            pass
        return sampled, frames

    def device_step(hook: str, sampled: list[int], frames: torch.Tensor) -> list[int]:
        monkeypatch.setattr(
            model_runner_module, "seeded_frame_codes", lambda *args, **kwargs: frames
        )
        result = types.SimpleNamespace(
            logits_output=types.SimpleNamespace(
                hidden_states=torch.zeros(3, HIDDEN_SIZE)
            ),
            next_token_ids=torch.tensor(sampled),
        )
        if hook == "launch":
            launch_buf = device_runner.post_decode_launch(
                result, forward_batch, device_requests
            )
            device_runner.post_decode_resolve(
                launch_buf, result, forward_batch, None, device_requests
            )
        else:
            getattr(device_runner, hook)(result, forward_batch, None, device_requests)
        next_token_ids = result.next_token_ids.tolist()
        for request, token_id in zip(device_requests, next_token_ids, strict=True):
            request.data.req.output_ids.append(token_id)
        return next_token_ids

    for step in range(40):
        sampled, frames = step_inputs(step)
        speaking = [
            row
            for row, request in enumerate(host_requests)
            if request.data.speech.needs_frame()
        ]
        expected = run_speech_step(
            host_runner, host_requests, sampled, frames[speaking], monkeypatch
        )
        hook = (
            "post_prefill" if step == 0 else ("launch" if step % 2 else "post_decode")
        )
        assert device_step(hook, sampled, frames) == expected, step
        pooled = device_runner.speech_pool.gather(forward_batch.req_pool_indices)
        mirrored = SpeechRows.from_states(
            [request.data.speech for request in device_requests],
            4,
            TOKENS.num_codebooks,
        )
        for field in fields(SpeechRows):
            assert torch.equal(
                getattr(pooled, field.name), getattr(mirrored, field.name)
            ), (step, field.name)
        host_runner.before_decode(None, None, host_requests)
        device_runner.before_decode(forward_batch, None, device_requests)
        assert torch.allclose(
            device_runner.model.decode_input_embed_buffer,
            host_runner.model.decode_input_embed_buffer,
        ), step

    for host_request, device_request in zip(
        host_requests, device_requests, strict=True
    ):
        host_speech, device_speech = (
            host_request.data.speech,
            device_request.data.speech,
        )
        assert device_speech.text_out == host_speech.text_out
        assert [frame.tolist() for frame in device_speech.frames] == [
            frame.tolist() for frame in host_speech.frames
        ]
        assert [
            codes.tolist() for codes in device_request.data.feedback_frame_codes
        ] == [codes.tolist() for codes in host_request.data.feedback_frame_codes]


SUPPORTED_SETTINGS = dict(
    attention_backend="triton",
    prefill_attention_backend=None,
    decode_attention_backend=None,
    disable_radix_cache=True,
    chunked_prefill_size=-1,
    enable_mixed_chunk=False,
    disable_overlap_schedule=True,
    disable_cuda_graph=False,
    enable_torch_compile=False,
    tp_size=1,
    pp_size=1,
    dp_size=1,
    speculative_algorithm=None,
)


@pytest.mark.parametrize(
    "override",
    [
        {"disable_radix_cache": False},
        {"chunked_prefill_size": 4096},
        {"enable_mixed_chunk": True},
        {"disable_overlap_schedule": False},
        {"enable_torch_compile": True},
        {"tp_size": 2},
        {"speculative_algorithm": "EAGLE"},
        {"attention_backend": "flashinfer"},
        {"decode_attention_backend": "fa3"},
    ],
)
def test_engine_refuses_unsupported_server_settings(
    override: dict[str, object],
) -> None:
    builder = Lfm3AudioEngineBuilder(
        speech_sampler=SpeechSampler.EXACT, enable_async_decode=False
    )
    builder.validate_before_infrastructure(types.SimpleNamespace(**SUPPORTED_SETTINGS))
    with pytest.raises(ValueError, match="LFM3"):
        builder.validate_before_infrastructure(
            types.SimpleNamespace(**{**SUPPORTED_SETTINGS, **override})
        )
