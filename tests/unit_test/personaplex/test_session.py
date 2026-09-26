# SPDX-License-Identifier: Apache-2.0
"""A full-duplex session steps the LM frame by frame exactly as one offline request."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.personaplex.architecture import SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.model_runner import PersonaPlexModelRunner
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.request_builders import build_lm_request
from sglang_omni.models.personaplex.session import (
    Code2WavSessionHooks,
    MimiEncodeSessionHooks,
    PersonaPlexSessionAdapter,
    PromptSessionHooks,
    pcm16_to_waveform,
    waveform_to_pcm16,
)
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext

from .test_model_runner import FakeModel

SESSION = SessionIdentity("call")
PROMPT_IDS = [11, 12, 13]
CONTEXT_LENGTH = 4096


def caller_codes(num_frames: int) -> torch.Tensor:
    return torch.arange(num_frames * 8).view(num_frames, 8) + 500


def make_runner() -> tuple[PersonaPlexModelRunner, FakeModel]:
    model = FakeModel()
    runner = PersonaPlexModelRunner.__new__(PersonaPlexModelRunner)
    runner.model = model
    return runner, model


def unit_chunk(index: int, *, eos: bool = False) -> TimedChunk:
    return TimedChunk("audio", 80.0 * index, 80.0, index, b"", format="pcm16", eos=eos)


def scheduled_req(origin, output=(), prefix_len: int = 0):
    return SimpleNamespace(
        origin_input_ids=list(origin),
        output_ids=list(output),
        prefix_indices=range(prefix_len),
    )


def prefill(runner, request, req) -> torch.Tensor:
    extend_len = (
        len(req.origin_input_ids) + len(req.output_ids) - len(req.prefix_indices)
    )
    forward_batch = SimpleNamespace(
        replace_embeds=None, input_ids=torch.zeros(extend_len)
    )
    runner.before_prefill(forward_batch, SimpleNamespace(reqs=[req]), [request])
    return get_omni_prefill_inputs(forward_batch).input_embeds.long()


def sample(runner, request, token: int, *, decode: bool) -> None:
    result = SimpleNamespace(next_token_ids=torch.tensor([token]))
    if decode:
        runner.post_decode(result, None, None, [request])
    else:
        runner.post_prefill(result, None, None, [request])


def decode_row(runner, model, request, req) -> torch.Tensor:
    runner.before_decode(None, SimpleNamespace(reqs=[req]), [request])
    return model.fusion_buffer[0].long().clone()


def run_offline(num_frames: int, tokens: list[int]):
    """Every forward's input rows and the frames of one offline request."""
    runner, model = make_runner()
    payload = StagePayload(
        "offline",
        request=OmniRequest(inputs={}, params={}),
        data=PersonaPlexState(
            text_prompt_ids=PROMPT_IDS, user_codes=caller_codes(num_frames)
        ).to_dict(),
    )
    request = SimpleNamespace(
        data=build_lm_request(payload, vocab_size=32000, voice_cache={})
    )
    origin = request.data.req.origin_input_ids
    inputs = [prefill(runner, request, scheduled_req(origin))]
    sample(runner, request, tokens[0], decode=False)
    for step in range(1, num_frames):
        inputs.append(
            decode_row(runner, model, request, scheduled_req(origin, tokens[:step]))[
                None
            ]
        )
        sample(runner, request, tokens[step], decode=True)
    frames = request.data.talker_model_inputs["frames"]
    return inputs, torch.stack(frames), model.depformer.calls


class SessionDriver:
    """Plays the SGLang streaming session: history kept, one request per unit."""

    def __init__(self, adapter: PersonaPlexSessionAdapter) -> None:
        self.adapter = adapter
        self.runner, self.model = make_runner()
        self.history: list[int] = []
        self.unit = 0

    def payload(self, codes: torch.Tensor) -> StagePayload:
        state = PersonaPlexState(user_codes=codes)
        if self.unit == 0:
            state.text_prompt_ids = PROMPT_IDS
            state.carries_prompt = True
        else:
            pass
        return StagePayload(
            f"unit-{self.unit}",
            request=OmniRequest(inputs=None, params={}),
            data=state.to_dict(),
        )

    def build(self, codes: torch.Tensor):
        data = self.adapter.build(SESSION, unit_chunk(self.unit), self.payload(codes))
        return SimpleNamespace(data=data)

    def step(self, codes: torch.Tensor, tokens: list[int]):
        """Run one unit; returns its forward inputs and the payload it hands on."""
        request = self.build(codes)
        req = request.data.req
        history = self.history + list(req.origin_input_ids)
        assert req.sampling_params.max_new_tokens == len(tokens)
        # Note (wilsonzheng0327): SGLang caches all but the last history token.
        prefix_len = max(len(self.history) - 1, 0)
        inputs = [prefill(self.runner, request, scheduled_req(history, (), prefix_len))]
        sample(self.runner, request, tokens[0], decode=False)
        for step in range(1, len(tokens)):
            inputs.append(
                decode_row(
                    self.runner,
                    self.model,
                    request,
                    scheduled_req(history, tokens[:step], prefix_len),
                )[None]
            )
            sample(self.runner, request, tokens[step], decode=True)
        request.data.output_ids = list(tokens)
        self.history = history + list(tokens)
        self.unit += 1
        return inputs, self.adapter.result(SESSION, request.data)


def open_adapter(context_length: int = CONTEXT_LENGTH) -> PersonaPlexSessionAdapter:
    adapter = PersonaPlexSessionAdapter(vocab_size=32000, context_length=context_length)
    adapter.open(SESSION, OmniRequest(inputs=None, params={}))
    return adapter


def test_session_units_replay_the_offline_forwards():
    num_frames = 5
    tokens = [77 + i for i in range(num_frames)]
    offline_inputs, offline_frames, offline_calls = run_offline(num_frames, tokens)

    driver = SessionDriver(open_adapter())
    codes = caller_codes(num_frames)
    session_inputs = []
    session_frames = []
    text_ids = []
    # Note (wilsonzheng0327): Forward j reads caller frame j - 1, so the offline
    # request's num_frames forwards are the first num_frames - 1 units.
    unit_tokens = [tokens[:2]] + [[token] for token in tokens[2:]]
    for unit, step_tokens in enumerate(unit_tokens):
        inputs, payload = driver.step(codes[unit : unit + 1], step_tokens)
        session_inputs.extend(inputs)
        state = PersonaPlexState.from_dict(payload.data)
        assert state.codes.shape[0] == len(step_tokens)
        session_frames.append(state.codes)
        text_ids.extend(state.text_ids)

    assert len(session_inputs) == len(offline_inputs)
    for session_rows, offline_rows in zip(session_inputs, offline_inputs, strict=True):
        assert torch.equal(session_rows, offline_rows)
    assert torch.equal(torch.cat(session_frames), offline_frames)
    assert text_ids == tokens
    for session_call, offline_call in zip(
        driver.model.depformer.calls, offline_calls, strict=True
    ):
        assert torch.equal(session_call.forced, offline_call.forced)
        assert torch.equal(session_call.text, offline_call.text)


def test_a_failed_unit_rolls_back_to_the_last_finished_one():
    codes = caller_codes(3)
    driver = SessionDriver(open_adapter())
    driver.step(codes[0:1], [77, 78])
    session = driver.adapter.sessions[SESSION.id]
    rows_before = session.timeline.user_rows.clone()

    abandoned = driver.build(codes[1:2])
    abandoned_rows = prefill(
        driver.runner,
        abandoned,
        scheduled_req(driver.history, (), len(driver.history) - 1),
    )
    sample(driver.runner, abandoned, 99, decode=False)
    assert len(session.model_inputs["agent_rows"]) == 3

    retried_inputs, payload = driver.step(codes[1:2], [79])
    assert len(session.model_inputs["agent_rows"]) == 3
    assert torch.equal(session.timeline.user_rows[: rows_before.shape[0]], rows_before)
    assert torch.equal(retried_inputs[0], abandoned_rows)
    assert PersonaPlexState.from_dict(payload.data).codes.shape[0] == 1


def test_unit_past_the_context_is_rejected():
    driver = SessionDriver(open_adapter(context_length=0))
    with pytest.raises(ValueError, match="context"):
        driver.build(caller_codes(1))


def test_first_unit_without_a_prompt_is_rejected():
    adapter = open_adapter()
    payload = StagePayload(
        "unit-0",
        request=OmniRequest(inputs=None, params={}),
        data=PersonaPlexState(user_codes=caller_codes(1)).to_dict(),
    )
    with pytest.raises(ValueError, match="prompt"):
        adapter.build(SESSION, unit_chunk(0), payload)


def test_empty_end_of_input_relays_no_frames():
    adapter = open_adapter()
    payload = StagePayload("eos", request=OmniRequest(inputs=None, params={}), data={})
    relayed = adapter.finish_input(SESSION, payload)
    assert PersonaPlexState.from_dict(relayed.data).codes.shape == (0, 8)


def context(emitted: list[TimedChunk]) -> SessionContext:
    return SessionContext(
        session_identity=SESSION, cancelled=threading.Event(), emit=emitted.append
    )


def pcm_chunk(waveform: torch.Tensor, index: int, *, eos: bool = False) -> TimedChunk:
    return TimedChunk(
        "audio",
        80.0 * index,
        80.0,
        index,
        waveform_to_pcm16(waveform),
        format="pcm16",
        eos=eos,
    )


def test_prompt_rides_the_first_unit_only():
    hooks = PromptSessionHooks(
        lambda request: PersonaPlexState(text_prompt_ids=PROMPT_IDS)
    )
    hooks.open(SESSION, OmniRequest(inputs=None, params={}))
    waveform = torch.linspace(-0.5, 0.5, SAMPLES_PER_FRAME)
    states = []
    for index in range(2):
        payload = StagePayload(
            f"unit-{index}", request=OmniRequest(inputs=None, params={}), data={}
        )
        hooks.append(pcm_chunk(waveform, index), payload, context([]))
        states.append(PersonaPlexState.from_dict(payload.data))
    assert states[0].text_prompt_ids == PROMPT_IDS and states[0].carries_prompt
    assert states[1].text_prompt_ids == [] and not states[1].carries_prompt
    torch.testing.assert_close(states[1].waveform, waveform, atol=1e-4, rtol=0)
    assert states[1].num_samples == SAMPLES_PER_FRAME


def test_units_must_be_whole_frames_of_pcm16():
    with pytest.raises(ValueError, match="whole 80 ms frames"):
        pcm16_to_waveform(
            TimedChunk("audio", 0.0, 1.0, 0, b"\0\0" * 10, format="pcm16")
        )
    with pytest.raises(ValueError, match="pcm16"):
        pcm16_to_waveform(TimedChunk("audio", 0.0, 1.0, 0, b"", format="wav"))


def test_streaming_encode_matches_the_whole_recording(random_codec):
    frames = 3
    waveform = torch.randn(frames * SAMPLES_PER_FRAME).clamp(-1, 1) * 0.5
    quantized = pcm16_to_waveform(pcm_chunk(waveform, 0))
    whole = random_codec.encode(quantized.view(1, 1, -1))[0].T

    hooks = MimiEncodeSessionHooks(random_codec, torch.device("cpu"))
    hooks.open(SESSION, OmniRequest(inputs=None, params={}))
    streamed = []
    for index in range(frames):
        unit = quantized[index * SAMPLES_PER_FRAME : (index + 1) * SAMPLES_PER_FRAME]
        payload = StagePayload(
            f"unit-{index}",
            request=OmniRequest(inputs=None, params={}),
            data=PersonaPlexState(waveform=unit).to_dict(),
        )
        hooks.append(pcm_chunk(unit, index), payload, context([]))
        state = PersonaPlexState.from_dict(payload.data)
        assert state.waveform is None
        streamed.append(state.user_codes)
    assert torch.equal(torch.cat(streamed), whole)


class PieceTokenizer:
    def id_to_piece(self, token: int) -> str:
        return {100: "▁hello", 101: "▁there", 102: "!"}[token]


def test_code2wav_streams_audio_and_spoken_text(random_codec):
    frames = 3
    codes = torch.randint(0, 2048, (frames, 8))
    whole = random_codec.decode(codes.T[None])[0, 0]
    hooks = Code2WavSessionHooks(random_codec, torch.device("cpu"), PieceTokenizer())
    hooks.open(SESSION, OmniRequest(inputs=None, params={}))
    emitted: list[TimedChunk] = []
    text_ids = [[3, 100], [0, 101], [102]]
    for index in range(frames):
        payload = StagePayload(
            f"unit-{index}",
            request=OmniRequest(inputs=None, params={}),
            data=PersonaPlexState(
                codes=codes[index : index + 1], text_ids=text_ids[index]
            ).to_dict(),
        )
        hooks.append(
            unit_chunk(index, eos=index == frames - 1), payload, context(emitted)
        )

    audio = [c for c in emitted if c.modality == "audio" and not c.eos]
    text = [c.payload["text"] for c in emitted if c.modality == "text"]
    streamed = np.frombuffer(b"".join(c.payload for c in audio), dtype="<i2")
    expected = np.frombuffer(waveform_to_pcm16(whole), dtype="<i2")
    assert np.abs(streamed.astype(np.int32) - expected).max() <= 1
    assert [c.t_start_ms for c in audio] == [0.0, 80.0, 160.0]
    assert "".join(text) == "hello there!"
    assert emitted[-1].eos and emitted[-1].t_start_ms == 240.0
