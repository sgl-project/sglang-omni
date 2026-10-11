from __future__ import annotations

import asyncio
import io
import wave
from types import SimpleNamespace

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
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

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


class FakeProcessor:
    def __call__(
        self,
        prompt_text: str,
        *,
        images: list[list[Image.Image]] | None,
        audios: list[list[npt.NDArray[np.float32]]] | None,
        return_tensors: str,
        **options: object,
    ) -> dict[str, object]:
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
    raw_audios: list[object] | None, *, target_sr: int
) -> list[npt.NDArray[np.float32]]:
    return [np.zeros(target_sr // 10, dtype=np.float32) for _ in raw_audios or []]


@pytest.fixture
def media_preprocessor(monkeypatch: pytest.MonkeyPatch) -> MiniCPMOPreprocessor:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = (
        FakeProcessor()  # noqa: leading-underscore  # production name
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


def speech_talker_request(talker_params: dict[str, int]) -> SGLangARRequestData:
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
    return build_sglang_talker_request(
        state,
        model=talker,
        codec_vocab_size=64,
        codec_eos_id=63,
        tts_bos_token_id=151703,
        tts_eos_token_id=151704,
        params={**result.request.params, **talker_params},
    )


def test_speech_sampling_fields_reach_the_talker() -> None:
    talker_request = speech_talker_request({})

    sampling_params = talker_request.req.sampling_params
    assert sampling_params.max_new_tokens == 30
    assert talker_request.talker_model_inputs["min_new_tokens"] == 30
    assert sampling_params.temperature == pytest.approx(0.3)
    assert sampling_params.top_k == 25


def test_a_negative_talker_minimum_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="talker_min_new_tokens"):
        speech_talker_request({"talker_min_new_tokens": -1})


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
