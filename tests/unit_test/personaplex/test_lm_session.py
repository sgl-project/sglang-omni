# SPDX-License-Identifier: Apache-2.0
"""A call stepped unit by unit through the LM session adapter is the offline request."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.personaplex.lm_session import PersonaPlexSessionAdapter
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.request_builders import (
    apply_lm_result,
    build_lm_request,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk

from .test_model_runner import FakeModel, make_runner

CALL = SessionIdentity("call")
PROMPT_IDS = [11, 12, 13]
CONTEXT_LENGTH = 4096


def caller_codes(num_frames: int) -> torch.Tensor:
    return torch.arange(num_frames * 8).view(num_frames, 8) + 500


def scheduled_req(origin: list[int], output: list[int], prefix_len: int):
    return SimpleNamespace(
        origin_input_ids=list(origin),
        output_ids=list(output),
        prefix_indices=range(prefix_len),
    )


def step(runner, model, request, origin, tokens, prefix_len) -> list[torch.Tensor]:
    """Play one request: prefill from prefix_len, then decode, sampling tokens."""
    forward_batch = SimpleNamespace(
        replace_embeds=None, input_ids=torch.zeros(len(origin) - prefix_len)
    )
    runner.before_prefill(
        forward_batch,
        SimpleNamespace(reqs=[scheduled_req(origin, [], prefix_len)]),
        [request],
    )
    inputs = [get_omni_prefill_inputs(forward_batch).input_embeds.long()]
    runner.post_prefill(
        SimpleNamespace(next_token_ids=torch.tensor(tokens[:1])), None, None, [request]
    )
    for index in range(1, len(tokens)):
        runner.before_decode(
            None,
            SimpleNamespace(reqs=[scheduled_req(origin, tokens[:index], prefix_len)]),
            [request],
        )
        inputs.append(model.fusion_buffer[:1].long().clone())
        runner.post_decode(
            SimpleNamespace(next_token_ids=torch.tensor(tokens[index : index + 1])),
            None,
            None,
            [request],
        )
    return inputs


def run_offline(num_frames: int, tokens: list[int]):
    model = FakeModel()
    runner = make_runner(model)
    payload = StagePayload(
        "offline",
        request=OmniRequest(inputs={}),
        data=PersonaPlexState(
            text_prompt_ids=PROMPT_IDS, user_codes=caller_codes(num_frames)
        ).to_dict(),
    )
    request = SimpleNamespace(data=build_lm_request(payload, vocab_size=32000))
    origin = list(request.data.req.origin_input_ids)
    inputs = step(runner, model, request, origin, tokens, 0)
    frames = torch.stack(request.data.talker_model_inputs["frames"])
    request.data.output_ids = list(tokens)
    text_ids = PersonaPlexState.from_dict(apply_lm_result(request.data).data).text_ids
    return inputs, frames, text_ids, model.depformer.calls


def open_adapter(context_length: int = CONTEXT_LENGTH) -> PersonaPlexSessionAdapter:
    adapter = PersonaPlexSessionAdapter(vocab_size=32000, context_length=context_length)
    adapter.open(CALL, OmniRequest(inputs=None))
    return adapter


def unit_payload(index: int, codes: torch.Tensor) -> StagePayload:
    state = PersonaPlexState(user_codes=codes)
    if index == 0:
        state.text_prompt_ids = PROMPT_IDS
        state.carries_prompt = True
    else:
        pass
    return StagePayload(
        f"unit-{index}", request=OmniRequest(inputs=None), data=state.to_dict()
    )


def unit_chunk(index: int) -> TimedChunk:
    return TimedChunk("audio", 80.0 * index, 80.0, index, b"", "pcm16")


class CallDriver:
    """Plays SGLang's streaming session: the history is kept between units."""

    def __init__(self, adapter: PersonaPlexSessionAdapter) -> None:
        self.adapter = adapter
        self.runner = make_runner(FakeModel())
        self.history: list[int] = []
        self.units = 0

    def build(self, codes: torch.Tensor) -> SimpleNamespace:
        data = self.adapter.build(
            CALL, unit_chunk(self.units), unit_payload(self.units, codes)
        )
        return SimpleNamespace(data=data)

    def unit(self, codes: torch.Tensor, tokens: list[int]):
        request = self.build(codes)
        req = request.data.req
        assert req.sampling_params.max_new_tokens == len(tokens)
        origin = self.history + list(req.origin_input_ids)
        # Note (wilsonzheng0327): The retained KV holds all but the last token.
        prefix_len = max(len(self.history) - 1, 0)
        inputs = step(
            self.runner, self.runner.model, request, origin, tokens, prefix_len
        )
        request.data.output_ids = list(tokens)
        self.history = origin + list(tokens)
        self.units += 1
        result = self.adapter.result(CALL, request.data)
        return inputs, PersonaPlexState.from_dict(result.data)


@pytest.mark.parametrize(
    "frames_per_unit", [[1, 1, 1, 1], [2, 1, 1], [1, 3], [4]], ids=str
)
def test_a_call_steps_exactly_as_the_offline_request(frames_per_unit: list[int]):
    num_caller_frames = sum(frames_per_unit)
    # Note (wilsonzheng0327): Forward j reads caller frame j - 1, so an offline
    # request of F + 1 frames runs the forwards a call runs on F frames.
    tokens = [77 + index for index in range(num_caller_frames + 1)]
    offline_inputs, offline_frames, offline_text, offline_calls = run_offline(
        num_caller_frames + 1, tokens
    )

    driver = CallDriver(open_adapter())
    codes = caller_codes(num_caller_frames + 1)
    inputs, frames, text_ids = [], [], []
    frame, token = 0, 0
    for index, count in enumerate(frames_per_unit):
        num_tokens = count + 1 if index == 0 else count
        unit_inputs, state = driver.unit(
            codes[frame : frame + count], tokens[token : token + num_tokens]
        )
        assert state.codes.shape[0] == num_tokens == len(state.text_ids)
        inputs.extend(unit_inputs)
        frames.append(state.codes)
        text_ids.extend(state.text_ids)
        frame += count
        token += num_tokens

    assert len(inputs) == len(offline_inputs)
    for rows, offline_rows in zip(inputs, offline_inputs, strict=True):
        assert torch.equal(rows, offline_rows)
    assert torch.equal(torch.cat(frames), offline_frames)
    assert text_ids == offline_text
    for call, offline_call in zip(
        driver.runner.model.depformer.calls, offline_calls, strict=True
    ):
        assert torch.equal(call.text, offline_call.text)
        assert torch.equal(call.forced, offline_call.forced)


def test_only_the_first_unit_sends_ids():
    first = CallDriver(open_adapter()).build(caller_codes(2)).data
    timeline = first.talker_model_inputs["timeline"]
    assert len(first.req.origin_input_ids) == timeline.num_prompt_positions
    assert first.req.sampling_params.max_new_tokens == 3

    driver = CallDriver(open_adapter())
    driver.unit(caller_codes(1), [77, 78])
    later = driver.build(caller_codes(2)).data
    assert list(later.req.origin_input_ids) == []
    assert later.req.sampling_params.max_new_tokens == 2


def test_a_unit_past_the_context_is_rejected():
    with pytest.raises(ValueError, match="context"):
        CallDriver(open_adapter(context_length=0)).build(caller_codes(1))


def test_a_call_without_its_prompt_is_rejected():
    adapter = open_adapter()
    payload = StagePayload(
        "unit-1",
        request=OmniRequest(inputs=None),
        data=PersonaPlexState(user_codes=caller_codes(1)).to_dict(),
    )
    with pytest.raises(ValueError, match="prompt"):
        adapter.build(CALL, unit_chunk(1), payload)


def test_an_empty_end_of_input_is_relayed_without_a_step():
    payload = StagePayload("end", request=OmniRequest(inputs=None), data={})
    relayed = open_adapter().finish_input(CALL, payload)
    state = PersonaPlexState.from_dict(relayed.data)
    assert state.codes.shape == (0, 8) and state.text_ids == []


def test_close_after_a_failed_open_is_a_no_op():
    adapter = PersonaPlexSessionAdapter(vocab_size=32000, context_length=4096)
    adapter.close(CALL)
    assert adapter.sessions == {}
