from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni_v1.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni_v1.models.qwen3_omni import request_builders
from sglang_omni_v1.models.qwen3_omni.merge import build_thinker_inputs
from sglang_omni_v1.models.qwen3_omni.payload_types import PipelineState
from sglang_omni_v1.models.qwen3_omni.request_builders import (
    build_sglang_thinker_request,
    make_talker_scheduler_adapters,
)
from sglang_omni_v1.proto import OmniRequest, StagePayload


def test_build_thinker_inputs_includes_media_cache_keys() -> None:
    state = PipelineState(
        encoder_inputs={
            "image_encoder": {"cache_key": "img-cache"},
            "audio_encoder": {"cache_key": "aud-cache"},
        }
    )

    thinker_inputs = build_thinker_inputs(state, encoder_outs={})

    assert thinker_inputs["media_cache_keys"] == {
        "image": "image:img-cache",
        "video": "video:img-cache",
        "audio": "audio:aud-cache",
    }


def test_build_sglang_thinker_request_hashes_media_tokens_and_preserves_mrope_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, torch.Tensor] = {}

    def fake_compute_mrope_positions(
        input_ids: torch.Tensor,
        model_inputs: dict[str, object],
        thinker_config: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del model_inputs, thinker_config
        captured["input_ids"] = input_ids.clone()
        return torch.zeros((3, input_ids.numel()), dtype=torch.long), torch.tensor(0)

    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )
    monkeypatch.setattr(
        "sglang_omni_v1.models.qwen3_omni.request_builders._compute_mrope_positions",
        fake_compute_mrope_positions,
    )

    audio_token_id = 7
    input_ids = torch.tensor([101, audio_token_id, 102], dtype=torch.long)
    state = PipelineState(
        prompt={
            "prompt_text": "prompt",
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        },
        thinker_inputs={
            "model_inputs": {"audio_embeds": torch.ones((1, 4))},
            "media_cache_keys": {"audio": "audio:cache-key"},
        },
    )
    thinker_config = SimpleNamespace(
        image_token_id=5,
        video_token_id=6,
        audio_token_id=audio_token_id,
    )
    tokenizer = SimpleNamespace(eos_token_id=0)

    req_data = build_sglang_thinker_request(
        state,
        params={"seed": "17"},
        tokenizer=tokenizer,
        vocab_size=256,
        request_id="rid-1",
        thinker_config=thinker_config,
    )

    assert req_data.req.sampling_params.sampling_seed == 17
    pad_values = req_data.req.omni_model_inputs["pad_values"]
    assert "audio" in pad_values
    assert pad_values["audio"] >= 256
    assert int(req_data.input_ids[1]) == pad_values["audio"]
    assert captured["input_ids"].tolist() == input_ids.tolist()


def test_talker_scheduler_adapter_normalizes_seed_before_request_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int | None] = {}

    class FakeTalkerPrefillBuilder:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def build_prompt_prefill(
            self,
            payload: StagePayload,
            thinker_chunks: list[object],
            *,
            thinker_done: bool,
        ) -> dict[str, object]:
            del payload, thinker_chunks, thinker_done
            return {
                "input_embeds": torch.zeros((1, 4), dtype=torch.float32),
                "input_ids": torch.tensor([1], dtype=torch.long),
                "pending_text_queue": [],
                "tts_pad_embed": torch.zeros(4, dtype=torch.float32),
                "prompt_model_inputs": {},
                "tts_eos_embed": torch.ones(4, dtype=torch.float32),
            }

        def append_text_chunk(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def mark_thinker_done(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    def fake_build_sglang_talker_request(
        thinker_hidden_states: torch.Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        del thinker_hidden_states
        captured["sampling_seed"] = kwargs["sampling_seed"]
        return SimpleNamespace()

    monkeypatch.setattr(
        request_builders,
        "TalkerPrefillBuilder",
        FakeTalkerPrefillBuilder,
    )
    monkeypatch.setattr(
        request_builders,
        "build_sglang_talker_request",
        fake_build_sglang_talker_request,
    )

    request_builder, *_ = make_talker_scheduler_adapters(
        tokenizer=SimpleNamespace(eos_token_id=0),
        codec_vocab_size=8,
        model=SimpleNamespace(config=SimpleNamespace(codec_eos_token_id=7)),
        model_path="unused",
        thinker_config=SimpleNamespace(),
        required_aux_hidden_key=0,
    )
    payload = StagePayload(
        request_id="rid-1",
        request=OmniRequest(inputs={}, params={"seed": "23"}),
        data={},
    )
    payload.prefetched_chunks = [
        SimpleNamespace(data=torch.zeros(4, dtype=torch.float32), metadata={})
    ]
    payload.prefetched_stream_done = True

    request_builder(payload)

    assert captured["sampling_seed"] == 23


def test_inject_multimodal_embeds_uses_pad_values_and_clamps_embed_lookup() -> None:
    class RecordingEmbed:
        def __init__(self) -> None:
            self.num_embeddings = 10
            self.seen: torch.Tensor | None = None

        def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
            self.seen = input_ids.clone()
            return torch.zeros((input_ids.shape[0], 4), dtype=torch.float32)

    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    runner._embed_tokens = RecordingEmbed()
    runner._image_token_id = 5
    runner._video_token_id = 6
    runner._audio_token_id = 7

    pad_audio_token_id = 999
    audio_embed = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    req = SimpleNamespace(
        omni_model_inputs={
            "audio_embeds": audio_embed,
            "pad_values": {"audio": pad_audio_token_id},
        },
        _omni_consumed=None,
        is_chunked=0,
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([1, pad_audio_token_id, 2], dtype=torch.long),
        extend_seq_lens_cpu=[3],
    )
    schedule_batch = SimpleNamespace(reqs=[req])

    input_embeds, _, _ = runner._inject_multimodal_embeds(forward_batch, schedule_batch)

    assert runner._embed_tokens.seen is not None
    assert (
        int(runner._embed_tokens.seen.max().item())
        < runner._embed_tokens.num_embeddings
    )
    assert torch.equal(input_embeds[1], audio_embed[0])
