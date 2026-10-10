from __future__ import annotations

import asyncio
import base64
import io
import wave
from types import SimpleNamespace
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
import pytest
import torch
from PIL import Image
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.minicpm_o.components import preprocessor as preprocessor_mod
from sglang_omni.models.minicpm_o.components.preprocessor import (
    ASR_PROMPT_EN,
    ASR_PROMPT_ZH,
    AUDIO_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    TTS_READ_PROMPT_EN,
    TTS_READ_PROMPT_ZH,
    MiniCPMOPreprocessor,
    tts_read_prompt,
)
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import build_sglang_thinker_request
from sglang_omni.models.minicpm_o.talker_request import (
    build_sglang_talker_request,
    build_talker_request,
)
from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner
from sglang_omni.proto import OmniRequest, StagePayload

# The generation suffix from MiniCPM-o-4_5's tokenizer template.
GENERATION_TEMPLATE = """
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
    {%- if use_tts_template is defined and use_tts_template is true %}
        {{- '<|tts_bos|>' }}
    {%- endif %}
{%- endif %}
"""


@pytest.mark.parametrize("use_tts_template", [False, True])
def test_chat_prompt_matches_checkpoint_non_thinking_default(
    use_tts_template: bool,
) -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor.tokenizer = PreTrainedTokenizerBase(chat_template=GENERATION_TEMPLATE)

    prompt = preprocessor.render_chat_template(
        [{"role": "user", "content": "Answer the question."}],
        use_tts_template=use_tts_template,
    )

    expected = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if use_tts_template:
        expected += "<|tts_bos|>"
    assert prompt == expected


def test_raw_prompt_bypasses_chat_template() -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    raw_prompt = "<|im_start|>assistant\n<think>\n"

    assert preprocessor.render_chat_template(raw_prompt) == raw_prompt


def test_prompt_token_ids_bypass_chat_template() -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    token_ids = [151644, 151667, 198]
    payload = StagePayload(
        request_id="prompt-token-ids",
        request=OmniRequest(inputs={"messages": token_ids}),
        data=None,
    )

    result = asyncio.run(preprocessor(payload))

    assert result.data["prompt"]["prompt_text"] == ""
    assert result.data["prompt"]["input_ids"].tolist() == token_ids
    torch.testing.assert_close(
        result.data["prompt"]["attention_mask"], torch.ones(3, dtype=torch.long)
    )


# Renders each turn ahead of the checkpoint's generation suffix.
CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endfor %}" + GENERATION_TEMPLATE
)


class ProcessorOutput(TypedDict):
    input_ids: torch.Tensor
    image_bound: list[list[torch.Tensor]]
    pixel_values: list[list[torch.Tensor]]
    tgt_sizes: list[list[torch.Tensor]]
    audio_bounds: list[list[torch.Tensor]]
    audio_feature_lens: list[list[torch.Tensor]]
    audio_features: list[torch.Tensor]


class RecordingProcessor:
    def __init__(self) -> None:
        self.images: list[list[Image.Image]] | None = None
        self.audios: list[list[npt.NDArray[np.float32]]] | None = None
        self.audio_parts: list[list[int]] | None = None

    def __call__(
        self,
        prompt_text: str,
        *,
        images: list[list[Image.Image]] | None,
        audios: list[list[npt.NDArray[np.float32]]] | None,
        return_tensors: Literal["pt"],
        audio_parts: list[list[int]] | None = None,
    ) -> ProcessorOutput:
        self.images = images
        self.audios = audios
        self.audio_parts = audio_parts
        image_count = len(images[0]) if images else 0
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "image_bound": [[torch.tensor([0, 1])] * image_count],
            "pixel_values": [[torch.zeros(1, 2) for _ in range(image_count)]],
            "tgt_sizes": [[torch.tensor([1, 1]) for _ in range(image_count)]],
            "audio_bounds": [[]],
            "audio_feature_lens": [[]],
            "audio_features": [],
        }


async def images_as_given(raw_images: list[Image.Image] | None) -> list[Image.Image]:
    return list(raw_images or [])


async def silent_audios(
    raw_audios: list[str | npt.NDArray[np.float32]] | None, *, target_sr: int
) -> list[npt.NDArray[np.float32]]:
    return [
        (
            waveform
            if isinstance(waveform, np.ndarray)
            else np.zeros(target_sr // 10, dtype=np.float32)
        )
        for waveform in raw_audios or []
    ]


@pytest.fixture
def recording_processor() -> RecordingProcessor:
    return RecordingProcessor()


@pytest.fixture
def media_preprocessor(
    monkeypatch: pytest.MonkeyPatch, recording_processor: RecordingProcessor
) -> MiniCPMOPreprocessor:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = (
        recording_processor  # noqa: leading-underscore  # production name
    )
    preprocessor.speech_enabled = False
    preprocessor.tokenizer = PreTrainedTokenizerBase(chat_template=CHAT_TEMPLATE)
    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", images_as_given)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", silent_audios)
    return preprocessor


@pytest.mark.parametrize(
    ("language", "task_prompt"), [("en", ASR_PROMPT_EN), ("zh", ASR_PROMPT_ZH)]
)
def test_transcription_prompt_precedes_audio(
    media_preprocessor: MiniCPMOPreprocessor, language: str, task_prompt: str
) -> None:
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 1600)
    payload = StagePayload(
        request_id="transcription",
        request=OmniRequest(
            inputs={"audio_bytes": wav_buffer.getvalue()},
            params={"language": language},
        ),
        data=None,
    )

    result = asyncio.run(media_preprocessor(payload))

    prompt_text = result.data["prompt"]["prompt_text"]
    user_turn = prompt_text.partition("<|im_start|>assistant\n")[0]
    assert user_turn == (
        f"<|im_start|>user\n{task_prompt}\n\n{AUDIO_PLACEHOLDER}<|im_end|>\n"
    )


def test_chat_media_placeholders_lead_the_user_text(
    media_preprocessor: MiniCPMOPreprocessor,
) -> None:
    payload = StagePayload(
        request_id="chat",
        request=OmniRequest(
            inputs={
                "messages": [
                    {"role": "user", "content": "Answer the question in the audio."}
                ],
                "images": [Image.new("RGB", (2, 2))],
                "audios": ["question.wav"],
            }
        ),
        data=None,
    )

    result = asyncio.run(media_preprocessor(payload))

    prompt_text = result.data["prompt"]["prompt_text"]
    user_turn = prompt_text.partition("<|im_start|>assistant\n")[0]
    assert user_turn == (
        f"<|im_start|>user\n{IMAGE_PLACEHOLDER}\n{AUDIO_PLACEHOLDER}\n"
        "Answer the question in the audio.<|im_end|>\n"
    )


class SpeechTokenizer:
    """Chat template, prompt ids and speech ids of a tiny MiniCPM-o vocabulary."""

    def apply_chat_template(
        self, messages: list[dict[str, str]], **template_options: bool
    ) -> str:
        assert template_options["use_tts_template"]
        return messages[0]["content"] + "<|tts_bos|>"

    def __call__(
        self, prompt_text: str, return_tensors: str
    ) -> dict[str, torch.Tensor]:
        assert prompt_text == f"{TTS_READ_PROMPT_ZH}你好<|tts_bos|>"
        return {"input_ids": torch.tensor([[7, 8, 151703]])}

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|tts_bos|>": 151703, "<|tts_eos|>": 151704}[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert text == "你好"
        assert not add_special_tokens
        return [100, 101]


@pytest.mark.parametrize(
    ("text", "language", "prompt"),
    [
        ("你好", None, TTS_READ_PROMPT_ZH),
        ("你好", "Auto", TTS_READ_PROMPT_ZH),
        ("hello", None, TTS_READ_PROMPT_EN),
        ("hello", "Chinese", TTS_READ_PROMPT_ZH),
        ("你好", "English", TTS_READ_PROMPT_EN),
    ],
)
def test_speech_read_prompt_follows_the_language_then_the_text(
    text: str, language: str | None, prompt: str
) -> None:
    assert tts_read_prompt(text, language) == prompt


def test_speech_request_prefills_its_text() -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor.tokenizer = SpeechTokenizer()
    preprocessor.speech_enabled = True
    payload = StagePayload(
        request_id="speech",
        request=OmniRequest(
            inputs="你好",
            metadata={
                "task": "tts",
                "output_modalities": ["audio"],
                "tts_params": {"language": "Chinese"},
            },
        ),
        data=None,
    )

    result = asyncio.run(preprocessor(payload))
    prompt = result.data["prompt"]
    assert prompt["input_ids"].tolist() == [7, 8, 151703, 100, 101, 151704]
    assert prompt["known_tts_output_ids"] == [100, 101]

    runner = object.__new__(MiniCPMOThinkerModelRunner)
    runner.pending_hidden = {}
    thinker_request = SimpleNamespace(
        stage_payload=result,
        extra_model_outputs={},
        req=SimpleNamespace(inflight_middle_chunks=1),
    )
    # The six-token prompt prefills in two chunks, one hidden row per token.
    for rows, middle_chunks in ((range(0, 4), 1), (range(4, 6), 0)):
        thinker_request.req.inflight_middle_chunks = middle_chunks
        runner.post_process_outputs(
            None,
            SimpleNamespace(
                requests=[SimpleNamespace(request_id="speech", data=thinker_request)]
            ),
            {
                "speech": SimpleNamespace(
                    extra={"hidden_states": torch.tensor([[row] * 4 for row in rows])}
                )
            },
        )
    runner.on_request_finished("speech", thinker_request)
    state = MiniCPMOPipelineState.from_dict(result.data)
    state.thinker_out = {
        "output_ids": [999],
        "extra_model_outputs": thinker_request.extra_model_outputs,
    }
    assert len(thinker_request.extra_model_outputs["hidden_states_seq"]) == 2
    span = build_talker_request(
        state,
        tts_bos_token_id=151703,
        tts_eos_token_id=151704,
    )
    assert span["tts_token_ids"].tolist() == [100, 101]
    torch.testing.assert_close(
        span["tts_hidden"],
        torch.tensor([[3, 3, 3, 3], [4, 4, 4, 4]]),
    )


def test_speech_sampling_fields_reach_the_talker() -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor.tokenizer = SpeechTokenizer()
    preprocessor.speech_enabled = True
    payload = StagePayload(
        request_id="speech",
        request=OmniRequest(
            inputs="你好",
            params={"max_new_tokens": 30, "temperature": 0.3, "top_k": 30},
            metadata={
                "task": "tts",
                "output_modalities": ["audio"],
                "tts_params": {
                    "language": "Chinese",
                    "explicit_generation_params": ["max_new_tokens", "temperature"],
                },
            },
        ),
        data=None,
    )

    result = asyncio.run(preprocessor(payload))
    state = MiniCPMOPipelineState.from_dict(result.data)
    state.thinker_out = {
        "output_ids": [999],
        "extra_model_outputs": {"hidden_states_seq": [torch.zeros(4)] * 6},
    }
    talker = SimpleNamespace(
        build_condition_embeddings=lambda token_ids, hidden: torch.zeros(
            len(token_ids) + 2, 8
        )
    )
    sampling_params = build_sglang_talker_request(
        state,
        model=talker,
        codec_vocab_size=64,
        codec_eos_id=63,
        tts_bos_token_id=151703,
        tts_eos_token_id=151704,
        params=result.request.params,
    ).req.sampling_params
    assert sampling_params.max_new_tokens == 30
    assert sampling_params.min_new_tokens == 30
    assert sampling_params.temperature == pytest.approx(0.3)
    assert sampling_params.top_k == 25


def test_speech_prefill_advances_by_chunk_in_its_own_cache_namespace() -> None:
    state = MiniCPMOPipelineState(
        prompt={
            "prompt_text": "",
            "input_ids": torch.tensor([151703, 100, 101, 151704]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "known_tts_output_ids": [100, 101],
        }
    )
    first, second = (
        build_sglang_thinker_request(
            state,
            params={},
            tokenizer=SpeechTokenizer(),
            vocab_size=151808,
            request_id="speech",
        ).req
        for _ in range(2)
    )
    assert first.extra_key != second.extra_key

    # SGLang stashes a finished chunk so that the next chunk starts after it.
    chunk_cache = ChunkCache(
        SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(4).view(1, 4)),
            token_to_kv_pool_allocator=None,
            page_size=1,
        )
    )
    first.kv.req_pool_idx = 0
    first.set_extend_range(0, 2)
    maybe_cache_unfinished_req(first, chunk_cache, chunked=True)
    assert first.prefix_indices.tolist() == [0, 1]


def test_ordered_inline_media_preserves_turns_and_text_separators(
    media_preprocessor: MiniCPMOPreprocessor,
    recording_processor: RecordingProcessor,
) -> None:
    first_image = Image.new("RGB", (2, 2), "red")
    second_image = Image.new("RGB", (2, 2), "blue")
    messages = [
        {"role": "user", "content": ["Before", first_image, "After"]},
        {"role": "assistant", "content": "Remembered."},
        {
            "role": "user",
            "content": [second_image, {"type": "text", "text": "Compare."}],
        },
    ]
    payload = StagePayload(
        request_id="inline-turns", request=OmniRequest(inputs=messages), data=None
    )
    result = asyncio.run(media_preprocessor(payload))
    prompt_text = result.data["prompt"]["prompt_text"]
    assert prompt_text.startswith(
        f"<|im_start|>user\nBefore\n{IMAGE_PLACEHOLDER}\nAfter<|im_end|>\n"
        "<|im_start|>assistant\nRemembered.<|im_end|>\n"
        f"<|im_start|>user\n{IMAGE_PLACEHOLDER}\nCompare.<|im_end|>\n"
    )
    assert recording_processor.images == [[first_image, second_image]]


def test_audio_content_reaches_processor_in_turn_order(
    media_preprocessor: MiniCPMOPreprocessor,
    recording_processor: RecordingProcessor,
) -> None:
    first_waveform = np.zeros(1600, dtype=np.float32)
    second_waveform = np.ones(1600, dtype=np.float32)
    third_waveform = np.full(1600, 0.5, dtype=np.float32)
    request_payload = StagePayload(
        request_id="audio-turns",
        request=OmniRequest(
            inputs=[
                {"role": "user", "content": [first_waveform, "Next", second_waveform]},
                {"role": "assistant", "content": "OK."},
                {"role": "user", "content": ["Finally", third_waveform]},
            ]
        ),
        data=None,
    )
    result = asyncio.run(media_preprocessor(request_payload))
    assert result.data["prompt"]["prompt_text"].startswith(
        f"<|im_start|>user\n{AUDIO_PLACEHOLDER}\nNext\n{AUDIO_PLACEHOLDER}<|im_end|>\n"
        "<|im_start|>assistant\nOK.<|im_end|>\n"
        f"<|im_start|>user\nFinally\n{AUDIO_PLACEHOLDER}<|im_end|>\n"
    )
    assert recording_processor.audio_parts == [[0, 0, 2]]
    for actual_waveform, expected_waveform in zip(
        recording_processor.audios[0],
        [first_waveform, second_waveform, third_waveform],
        strict=True,
    ):
        np.testing.assert_array_equal(actual_waveform, expected_waveform)


def test_audio_cache_identity_includes_turn_grouping(
    media_preprocessor: MiniCPMOPreprocessor,
    recording_processor: RecordingProcessor,
) -> None:
    first_waveform = np.zeros(16000, dtype=np.float32)
    second_waveform = np.ones(8000, dtype=np.float32)
    same_turn_payload = StagePayload(
        request_id="same-turn-audio",
        request=OmniRequest(
            inputs=[{"role": "user", "content": [first_waveform, second_waveform]}]
        ),
        data=None,
    )
    separate_turns_payload = StagePayload(
        request_id="separate-turns-audio",
        request=OmniRequest(
            inputs=[
                {"role": "user", "content": [first_waveform]},
                {"role": "assistant", "content": "OK."},
                {"role": "user", "content": [second_waveform]},
            ]
        ),
        data=None,
    )

    same_turn_result = asyncio.run(media_preprocessor(same_turn_payload))
    assert recording_processor.audio_parts == [[0, 0]]
    separate_turns_result = asyncio.run(media_preprocessor(separate_turns_payload))
    assert recording_processor.audio_parts == [[0, 2]]

    same_turn_cache_key = same_turn_result.data["encoder_inputs"]["audio_encoder"][
        "cache_key"
    ]
    separate_turns_cache_key = separate_turns_result.data["encoder_inputs"][
        "audio_encoder"
    ]["cache_key"]
    assert same_turn_cache_key != separate_turns_cache_key
    assert (
        same_turn_result.data["mm_inputs"]["audio"]["cache_key"] == same_turn_cache_key
    )
    assert (
        separate_turns_result.data["mm_inputs"]["audio"]["cache_key"]
        == separate_turns_cache_key
    )


def test_text_parts_follow_chat_newline_separator(
    media_preprocessor: MiniCPMOPreprocessor,
) -> None:
    prompt_text = media_preprocessor.render_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A"},
                    {"type": "text", "text": "B"},
                ],
            }
        ]
    )
    assert prompt_text.startswith("<|im_start|>user\nA\nB<|im_end|>\n")


@pytest.mark.parametrize(
    "top_level_media_name", ["images", "audios", "audio", "videos", "video"]
)
def test_inline_and_top_level_media_are_rejected(
    media_preprocessor: MiniCPMOPreprocessor, top_level_media_name: str
) -> None:
    payload = StagePayload(
        request_id="mixed-media",
        request=OmniRequest(
            inputs={
                "messages": [
                    {"role": "user", "content": [Image.new("RGB", (2, 2)), "Describe."]}
                ],
                top_level_media_name: ["media"],
            }
        ),
        data=None,
    )
    with pytest.raises(ValueError, match="Inline media cannot be combined"):
        asyncio.run(media_preprocessor(payload))


def test_unknown_inline_content_is_rejected(
    media_preprocessor: MiniCPMOPreprocessor,
) -> None:
    payload = StagePayload(
        request_id="unknown-part",
        request=OmniRequest(
            inputs=[{"role": "user", "content": [{"type": "unknown"}]}]
        ),
        data=None,
    )
    with pytest.raises(ValueError, match="Unsupported MiniCPM-o content type"):
        asyncio.run(media_preprocessor(payload))


@pytest.mark.parametrize("audio_part_type", ["audio_url", "input_audio"])
def test_openai_inline_media_decode_in_original_order(
    media_preprocessor: MiniCPMOPreprocessor,
    recording_processor: RecordingProcessor,
    audio_part_type: Literal["audio_url", "input_audio"],
) -> None:
    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(image_buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(
        image_buffer.getvalue()
    ).decode("ascii")
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(np.full(1600, 8192, dtype="<i2").tobytes())
    encoded_audio = base64.b64encode(audio_buffer.getvalue()).decode("ascii")
    audio_source = (
        {"url": "data:audio/wav;base64," + encoded_audio}
        if audio_part_type == "audio_url"
        else {"data": encoded_audio, "format": "wav"}
    )
    request_payload = StagePayload(
        request_id="openai-media",
        request=OmniRequest(
            inputs=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Before"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "Between"},
                        {"type": audio_part_type, audio_part_type: audio_source},
                        {"type": "text", "text": "After"},
                    ],
                }
            ]
        ),
        data=None,
    )
    result = asyncio.run(media_preprocessor(request_payload))
    assert result.data["prompt"]["prompt_text"].startswith(
        f"<|im_start|>user\nBefore\n{IMAGE_PLACEHOLDER}\nBetween\n{AUDIO_PLACEHOLDER}\nAfter<|im_end|>\n"
    )
    np.testing.assert_array_equal(
        np.asarray(recording_processor.images[0][0]),
        np.full((4, 4, 3), [255, 0, 0], dtype=np.uint8),
    )
    np.testing.assert_allclose(recording_processor.audios[0][0], 0.25)
    assert recording_processor.audio_parts == [[0]]
