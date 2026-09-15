# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 preprocessing: cloning-mode routing and prefill layout."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.models.voxcpm2.hf_config import VoxCPM2RuntimeConfig
from sglang_omni.models.voxcpm2.payload_types import VoxCPM2State
from sglang_omni.models.voxcpm2.request_builders import (
    VoxCPM2PreprocessingContext,
    VoxCPM2SGLangRequestData,
    apply_voxcpm2_result,
    audio_prefix_fingerprint,
    build_prefill_inputs,
    build_sglang_voxcpm2_request,
    build_stream_output,
    build_voxcpm2_state,
)

_TOKEN_IDS = {
    C.AUDIO_START_TOKEN: 101,
    C.AUDIO_END_TOKEN: 102,
    C.AUDIO_PROMPT_START_TOKEN: 103,
    C.AUDIO_PROMPT_END_TOKEN: 104,
}


class _FakeTokenizer:
    """Character-per-token stand-in; ids are the character codes."""

    def __call__(self, text, *, add_special_tokens=True):
        ids = [ord(ch) for ch in text]
        return {"input_ids": ([1] if add_special_tokens else []) + ids}

    def convert_tokens_to_ids(self, token):
        return _TOKEN_IDS[token]


class _FakeRequest:
    def __init__(self, inputs, params=None, metadata=None):
        self.inputs = inputs
        self.params = params or {}
        self.metadata = metadata or {}


class _FakePayload:
    def __init__(self, request):
        self.request = request
        self.request_id = "req"
        self.data = {}


def _context():
    return VoxCPM2PreprocessingContext(
        config=VoxCPM2RuntimeConfig(
            model_path="fake",
            audio_vae={"sample_rate": 16000, "out_sample_rate": 48000},
            patch_size=4,
            feat_dim=64,
        ),
        tokenizer=_FakeTokenizer(),
    )


def _state_for(references):
    payload = _FakePayload(_FakeRequest({"text": "hi", "references": references}))
    return build_voxcpm2_state(payload, _context())


@pytest.mark.parametrize("transcript", [None, "reference words"])
def test_text_prefix_and_target_length_exclude_automatic_bos(transcript):
    references = (
        [{"audio_path": "reference.wav", "text": transcript}]
        if transcript is not None
        else []
    )
    state = _state_for(references)
    expected_text = (transcript or "") + "hi"
    assert state.text_token.tolist() == [ord(ch) for ch in expected_text] + [101]
    assert state.target_text_length == 2


def _speech_state(**kwargs):
    from sglang_omni.client.client import Client
    from sglang_omni.serve.protocol import CreateSpeechRequest
    from sglang_omni.serve.speech_service import SpeechRequestValidator

    request = CreateSpeechRequest(input="hi", **kwargs)
    generated = SpeechRequestValidator(default_model="voxcpm2").build_generate_request(
        request
    )
    return build_voxcpm2_state(
        _FakePayload(Client._build_omni_request(generated)), _context()
    )


def test_speech_endpoint_defaults_preserve_voxcpm2_sampling():
    state = _speech_state()
    assert state.inference_timesteps == C.DEFAULT_INFERENCE_TIMESTEPS
    assert state.cfg_value == C.DEFAULT_CFG_VALUE
    assert state.max_len == C.DEFAULT_MAX_LEN


def test_speech_endpoint_explicit_length_and_stage_recipe_reach_builder():
    state = _speech_state(
        max_new_tokens=1,
        seed=17,
        stage_params={"tts_engine": {"inference_timesteps": 4, "cfg_value": 1.5}},
    )
    assert (state.max_len, state.seed) == (1, 17)
    assert (state.inference_timesteps, state.cfg_value) == (4, 1.5)


def test_model_length_override_takes_precedence_over_generic_length():
    state = _speech_state(max_new_tokens=1, stage_params={"tts_engine": {"max_len": 3}})
    assert state.max_len == 3


@pytest.mark.parametrize("abort_before_compute", [True, False])
def test_preprocessing_abort_never_hands_off_a_cancelled_payload(abort_before_compute):
    import threading

    from sglang_omni.scheduling.messages import IncomingMessage
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    executed = []

    def compute(payload):
        executed.append(payload.request_id)
        if payload.request_id == "cancelled":
            entered.set()
            assert release.wait(timeout=5)
        payload.data = build_voxcpm2_state(payload, _context()).to_dict()
        if payload.request_id == "cancelled":
            completed.set()
        return payload

    scheduler = SimpleScheduler(compute, max_concurrency=8)
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()

    def enqueue(request_id):
        payload = _FakePayload(_FakeRequest("hi"))
        payload.request_id = request_id
        scheduler.inbox.put(
            IncomingMessage(request_id=request_id, type="new_request", data=payload)
        )

    try:
        if abort_before_compute:
            scheduler.abort("cancelled")
        enqueue("cancelled")
        if not abort_before_compute:
            assert entered.wait(timeout=5)
            scheduler.abort("cancelled")
            release.set()
            assert completed.wait(timeout=5)
        enqueue("live")
        result = scheduler.outbox.get(timeout=5)
        assert (result.request_id, result.type) == ("live", "result")
    finally:
        release.set()
        scheduler.stop()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert scheduler.outbox.empty()
    assert ("cancelled" in executed) is not abort_before_compute


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_ar_admission_normalizes_compute_tensors_and_preserves_wire_context(device):
    state = VoxCPM2State(
        text_token=torch.tensor([1, 2], dtype=torch.int32),
        prompt_latents=torch.full((3, 4, 64), 1.001, requires_grad=True),
    )
    payload = _FakePayload(_FakeRequest("hi"))
    payload.data = state.to_dict()
    original_key = audio_prefix_fingerprint(_prefill(state))
    data = build_sglang_voxcpm2_request(
        payload,
        tokenizer=_FakeTokenizer(),
        patch_size=4,
        feat_dim=64,
        vocab_size=256,
        device=device,
        dtype=torch.bfloat16,
    )
    for tensor in (
        data.prefill.text_token,
        data.prefill.text_mask,
        data.prefill.audio_mask,
        data.prefill.audio_feat,
        data.cond,
    ):
        assert tensor.device.type == device
        assert not tensor.requires_grad
    assert data.prefill.audio_feat.dtype == data.cond.dtype == torch.bfloat16
    assert data.req.extra_key == original_key
    assert data.req.origin_input_ids == [1, 2, 0, 0, 0]
    assert len(data.context_patches) == 3
    for patch in data.context_patches:
        assert patch.device.type == "cpu"
        assert patch.dtype == torch.float32
        assert not patch.requires_grad
    if device == "cpu":
        assert data.cond.unique().item() == 1.0
        torch.testing.assert_close(data.context_patches[-1], state.prompt_latents[-1])


def test_reference_without_transcript_is_a_timbre_prefix():
    state = _state_for([{"audio_path": "ref.wav"}])
    assert state.reference_audio == "ref.wav"
    assert state.prompt_audio == ""
    assert state.prompt_text == ""


def test_reference_with_transcript_becomes_continuation_audio():
    state = _state_for([{"audio_path": "ref.wav", "text": "hello"}])
    assert state.prompt_audio == "ref.wav"
    assert state.reference_audio == ""
    assert state.prompt_text == "hello"


def test_continuation_prefixes_the_transcript_to_the_target_text():
    state = _state_for([{"audio_path": "ref.wav", "text": "hello"}])
    assert state.text_token.tolist()[: len("hello")] == [ord(c) for c in "hello"]


def test_text_always_ends_with_the_audio_start_token():
    state = _state_for([])
    assert int(state.text_token[-1]) == _TOKEN_IDS[C.AUDIO_START_TOKEN]


def _prefill(state):
    return build_prefill_inputs(
        state, tokenizer=_FakeTokenizer(), patch_size=4, feat_dim=64
    )


def test_prefill_masks_cover_every_position_exactly_once():
    state = VoxCPM2State(text_token=torch.tensor([1, 2, 3], dtype=torch.int32))
    state.ref_latents = torch.zeros((5, 4, 64))
    state.prompt_latents = torch.zeros((2, 4, 64))
    prefill = _prefill(state)

    total = int(prefill.text_token.shape[0])
    assert int(prefill.audio_feat.shape[0]) == total
    assert torch.equal(prefill.text_mask + prefill.audio_mask, torch.ones(total))


def test_reference_prefix_leads_and_prompt_audio_trails():
    state = VoxCPM2State(text_token=torch.tensor([1, 2, 3], dtype=torch.int32))
    state.ref_latents = torch.zeros((5, 4, 64))
    state.prompt_latents = torch.zeros((2, 4, 64))
    prefill = _prefill(state)

    assert int(prefill.text_token[0]) == _TOKEN_IDS[C.AUDIO_PROMPT_START_TOKEN]
    # note (Xinhao Tan): index 6 is 1 start token + 5 reference patches + 1 end.
    assert int(prefill.text_token[6]) == _TOKEN_IDS[C.AUDIO_PROMPT_END_TOKEN]
    assert prefill.audio_mask[-2:].tolist() == [1, 1]
    assert prefill.text_mask[-2:].tolist() == [0, 0]


def test_zero_shot_prefill_is_text_only():
    state = VoxCPM2State(text_token=torch.tensor([1, 2, 3], dtype=torch.int32))
    prefill = _prefill(state)
    assert prefill.text_mask.tolist() == [1, 1, 1]
    assert prefill.audio_mask.tolist() == [0, 0, 0]


def test_empty_text_is_rejected():
    payload = _FakePayload(_FakeRequest({"text": "  ", "references": []}))
    with pytest.raises(ValueError, match="nonempty input text"):
        build_voxcpm2_state(payload, _context())


def test_more_than_one_reference_is_rejected():
    payload = _FakePayload(
        _FakeRequest(
            {"text": "hi", "references": [{"audio_path": "a"}, {"audio_path": "b"}]}
        )
    )
    with pytest.raises(ValueError, match="at most one reference"):
        build_voxcpm2_state(payload, _context())


def test_a_reference_without_audio_is_rejected():
    """Silently answering in an unrelated voice is the worse failure here."""
    payload = _FakePayload(
        _FakeRequest({"text": "hi", "references": [{"text": "a transcript"}]})
    )
    with pytest.raises(ValueError, match="reference audio is missing"):
        build_voxcpm2_state(payload, _context())


def test_an_empty_reference_source_is_rejected():
    payload = _FakePayload(
        _FakeRequest({"text": "hi", "references": [{"audio_path": ""}]})
    )
    with pytest.raises(ValueError, match="reference audio is missing"):
        build_voxcpm2_state(payload, _context())


def test_ref_text_without_reference_audio_is_rejected():
    payload = _FakePayload(
        _FakeRequest(
            {"text": "hi", "references": []},
            metadata={"tts_params": {"ref_text": "a transcript"}},
        )
    )
    with pytest.raises(ValueError, match="without reference audio"):
        build_voxcpm2_state(payload, _context())


def test_no_reference_at_all_is_still_a_valid_zero_shot_request():
    """The rejection must not swallow the mode the endpoint uses by default."""
    state = _state_for([])
    assert state.reference_audio == ""
    assert state.prompt_audio == ""
    assert state.prompt_text == ""


def test_unset_sampling_fields_fall_back_to_the_released_recipe():
    state = _state_for([])
    assert state.inference_timesteps == C.DEFAULT_INFERENCE_TIMESTEPS
    assert state.cfg_value == C.DEFAULT_CFG_VALUE
    assert state.max_len == C.DEFAULT_MAX_LEN
    assert state.seed is None


def test_caller_supplied_sampling_fields_win():
    payload = _FakePayload(
        _FakeRequest(
            {"text": "hi", "references": []},
            metadata={"tts_params": {"inference_timesteps": 4, "cfg_value": 1.0}},
        )
    )
    state = build_voxcpm2_state(payload, _context())
    assert state.inference_timesteps == 4
    assert state.cfg_value == 1.0


def _prefill_with_reference(fill: float):
    state = VoxCPM2State(text_token=torch.tensor([1, 2], dtype=torch.int32))
    state.ref_latents = torch.full((3, 4, 64), fill)
    return _prefill(state)


def test_different_reference_audio_gets_a_different_radix_key():
    """Audio positions share token id 0, so only this key keeps prefixes apart."""
    first = audio_prefix_fingerprint(_prefill_with_reference(0.25))
    second = audio_prefix_fingerprint(_prefill_with_reference(0.75))
    assert first is not None and second is not None
    assert first != second


def test_identical_reference_audio_gets_the_same_radix_key():
    first = audio_prefix_fingerprint(_prefill_with_reference(0.25))
    second = audio_prefix_fingerprint(_prefill_with_reference(0.25))
    assert first == second


def test_zero_shot_requests_share_one_radix_subtree():
    state = VoxCPM2State(text_token=torch.tensor([1, 2], dtype=torch.int32))
    assert audio_prefix_fingerprint(_prefill(state)) is None


def _streaming_data(stream=True):
    return VoxCPM2SGLangRequestData(
        state=VoxCPM2State(stream=stream),
        stream_metadata={"modality": "audio_latents", "stream": stream},
    )


def test_a_non_streaming_request_sends_no_chunks():
    data = _streaming_data(stream=False)
    data.latent_patches.append(torch.zeros(4, 64))
    assert list(build_stream_output("r", data, None)) == []


def test_each_patch_is_sent_once_and_only_once():
    """A step that sampled nothing must not re-send the patch before it."""
    data = _streaming_data()
    data.latent_patches.append(torch.zeros(4, 64))
    assert len(list(build_stream_output("r", data, None))) == 1
    assert list(build_stream_output("r", data, None)) == []

    data.latent_patches.append(torch.ones(4, 64))
    second = list(build_stream_output("r", data, None))
    assert len(second) == 1
    assert second[0].metadata["chunk_id"] == 1


def test_a_chunk_carries_the_bare_patch_tensor():
    """The cross-process stage relay rejects anything that is not a tensor."""
    data = _streaming_data()
    data.latent_patches.append(torch.zeros(4, 64))
    message = next(iter(build_stream_output("r", data, None)))
    assert isinstance(message.data, torch.Tensor)
    assert message.metadata["modality"] == "audio_latents"
    assert message.metadata["stream"] is True


@pytest.mark.parametrize("continuation", [False, True])
def test_first_diffusion_condition_is_the_final_prefill_patch(continuation):
    state = VoxCPM2State(text_token=torch.tensor([1, 2], dtype=torch.int32))
    if continuation:
        state.prompt_latents = torch.arange(3 * 4 * 64).reshape(3, 4, 64).float()
    payload = _FakePayload(_FakeRequest("hi"))
    payload.data = state.to_dict()
    data = build_sglang_voxcpm2_request(
        payload, tokenizer=_FakeTokenizer(), patch_size=4, feat_dim=64, vocab_size=256
    )
    expected = state.prompt_latents[-1:] if continuation else torch.zeros(1, 4, 64)
    torch.testing.assert_close(data.cond, expected, rtol=0, atol=0)
    assert data.state.context_len == (3 if continuation else 0)


def test_context_is_sent_once_and_excluded_from_completion_usage():
    state = VoxCPM2State(
        text_token=torch.tensor([1, 2], dtype=torch.int32),
        stream=True,
        prompt_latents=torch.arange(5 * 4 * 64).reshape(5, 4, 64).float(),
    )
    payload = _FakePayload(_FakeRequest("hi"))
    payload.data = state.to_dict()
    data = build_sglang_voxcpm2_request(
        payload, tokenizer=_FakeTokenizer(), patch_size=4, feat_dim=64, vocab_size=256
    )
    generated = torch.full((4, 64), -1.0)
    data.latent_patches.append(generated)
    chunks = list(build_stream_output("req", data, None))
    assert len(chunks) == 4
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(range(4))
    assert all(chunk.metadata["context_len"] == 3 for chunk in chunks)
    torch.testing.assert_close(
        torch.stack([c.data for c in chunks[:3]]), state.prompt_latents[-3:]
    )
    assert list(build_stream_output("req", data, None)) == []
    apply_voxcpm2_result(data)
    assert data.state.completion_tokens == 1
    expected = torch.cat([state.prompt_latents[-3:], generated.unsqueeze(0)])
    torch.testing.assert_close(
        data.state.generated_latents, expected.permute(2, 0, 1).reshape(64, -1)
    )


def test_zero_minimum_length_survives_state_serialization():
    assert VoxCPM2State.from_dict(VoxCPM2State(min_len=0).to_dict()).min_len == 0
