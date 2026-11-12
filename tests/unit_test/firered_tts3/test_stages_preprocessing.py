# SPDX-License-Identifier: Apache-2.0
"""Preprocessing stage: Base template composition, budgets, validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.request_builders import (
    build_sglang_firered_tts3_request,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.utils.json import JsonValue
from tests.unit_test.fixtures.firered_tts3 import make_speaker_tokenizer
from tests.unit_test.fixtures.firered_tts3_preprocessing import (
    make_preprocessing_payload,
    preprocess_payload,
)


@pytest.mark.parametrize("max_patches,engine_tokens", [(400, 420), (256, 276)])
def test_preprocessing_composes_base_template(
    max_patches: int, engine_tokens: int
) -> None:
    payload = make_preprocessing_payload()
    tokenizer = make_speaker_tokenizer()
    state = preprocess_payload(payload, max_patches=max_patches, tokenizer=tokenizer)
    expected = "<|Chinese|><|sot|>今天天气很好。欢迎使用点点语音合成系统。<|eot|>"
    assert tokenizer.decode(state.input_ids) == expected
    assert state.variant == "base"
    assert state.task_type == "tts"
    assert state.language == "Chinese"
    assert state.prompt_audio_path == "ref.wav"
    assert state.prompt_text == "今天天气很好。"
    request = build_sglang_firered_tts3_request(
        payload, patch_size=4, vocab_size=151936
    )
    assert request.req.sampling_params.max_new_tokens == engine_tokens
    assert request.state.max_patches == max_patches


def test_preprocessing_param_precedence() -> None:
    payload = make_preprocessing_payload(
        params={"seed": 7, "stage_params": {"talker": {"inference_cfg": 1.5}}},
        tts_params={"stop_threshold": "0.7"},
    )
    preprocess_payload(payload)
    request = build_sglang_firered_tts3_request(
        payload, patch_size=4, vocab_size=151936
    )
    assert request.state.seed == 7
    assert request.state.inference_cfg == 1.5
    assert request.state.stop_threshold == 0.7


@pytest.mark.parametrize(
    "parameter_name,parameter_value",
    [
        ("n_timesteps", -1),
        ("n_timesteps", 0),
        ("n_timesteps", 1.5),
        ("n_timesteps", []),
        ("n_timesteps", True),
        ("inference_cfg", {}),
        ("stop_threshold", "nan"),
        ("min_patches", "invalid"),
        ("min_patches", -1),
        ("max_patches", 0),
        ("max_patches", -100),
        ("max_patches", []),
        ("seed", {}),
    ],
)
def test_preprocessing_rejects_invalid_generation_parameters(
    parameter_name: str, parameter_value: JsonValue
) -> None:
    """Invalid solver steps must not reach Flow and return the initial noise."""
    payload = make_preprocessing_payload(params={parameter_name: parameter_value})
    with pytest.raises(ValueError, match=parameter_name):
        preprocess_payload(payload)


def test_preprocessing_preserves_generation_parameter_precedence() -> None:
    payload = make_preprocessing_payload(
        inputs={
            "text": "Hello",
            "references": [{"audio_path": "ref.wav", "text": "Hello"}],
            "n_timesteps": "6",
        },
        params={
            "n_timesteps": "3",
            "stage_params": {"talker": {"n_timesteps": "4"}},
            "inference_cfg": "1.25",
            "min_patches": "3",
            "max_patches": "100",
            "seed": "7",
        },
        tts_params={"n_timesteps": "5", "stop_threshold": "0.7"},
    )
    state = preprocess_payload(payload)
    assert state.n_timesteps == 6
    assert state.inference_cfg == 1.25
    assert state.stop_threshold == 0.7
    assert state.min_patches == 3
    assert state.max_patches == 100
    assert state.seed == 7


def test_preprocessing_requires_text_and_reference() -> None:
    payload = make_preprocessing_payload(inputs={"text": "  "})
    with pytest.raises(ValueError, match="non-empty input text"):
        preprocess_payload(payload)
    payload = make_preprocessing_payload(inputs={"text": "你好"})
    with pytest.raises(ValueError, match="reference audio"):
        preprocess_payload(payload)
    payload = make_preprocessing_payload(
        inputs={"text": "你好", "references": [{"audio_path": "ref.wav"}]}
    )
    with pytest.raises(ValueError, match="transcript"):
        preprocess_payload(payload)


def test_preprocessing_accepts_streaming_and_bad_language_rejected() -> None:
    state = preprocess_payload(make_preprocessing_payload(params={"stream": True}))
    assert state.stream is True
    state = preprocess_payload(make_preprocessing_payload())
    assert state.stream is False
    with pytest.raises(ValueError, match="unsupported FireRedTTS3 language"):
        preprocess_payload(
            make_preprocessing_payload(tts_params={"language": "Klingon"})
        )
    state = preprocess_payload(
        make_preprocessing_payload(tts_params={"language": "ZH_Shanghai"})
    )
    assert state.language == "ZH_Shanghai"


INSTRUCT_CONFIG = FireRedTTS3Config(use_lm_head=True, dit_in_channels=1088)


def instruct_payload(
    *,
    task_type: str | None = None,
    instructions: str | None = None,
    text: str = "晚上好，欢迎收听今天的节目。",
    reference_audio_path: str | None = None,
    params: dict[str, JsonValue] | None = None,
    tts_params: dict[str, JsonValue] | None = None,
) -> StagePayload:
    inputs: dict[str, JsonValue] = {"text": text}
    if reference_audio_path is not None:
        inputs["references"] = [
            {"audio_path": reference_audio_path, "text": "源音频文本。"}
        ]
    else:
        pass
    if instructions is not None:
        inputs["instructions"] = instructions
    else:
        pass
    if task_type is not None:
        (tts_params := dict(tts_params or {}))["task_type"] = task_type
    else:
        pass
    return make_preprocessing_payload(
        inputs=inputs, params=params or {}, tts_params=tts_params
    )


def test_instruct_task_validation() -> None:
    with pytest.raises(ValueError, match="task_type"):
        preprocess_payload(instruct_payload(task_type="Base2"), config=INSTRUCT_CONFIG)
    with pytest.raises(ValueError, match="instructions"):
        preprocess_payload(
            instruct_payload(task_type="VoiceDesign"), config=INSTRUCT_CONFIG
        )
    with pytest.raises(ValueError, match="non-empty input text"):
        preprocess_payload(
            instruct_payload(task_type="VoiceDesign", instructions="x", text=" "),
            config=INSTRUCT_CONFIG,
        )
    with pytest.raises(ValueError, match="edit source"):
        preprocess_payload(
            instruct_payload(task_type="AcousticEdit", text=""), config=INSTRUCT_CONFIG
        )
    with pytest.raises(ValueError, match="reference audio"):
        preprocess_payload(instruct_payload(text="你好"), config=INSTRUCT_CONFIG)
    with pytest.raises(ValueError, match="transcript"):
        preprocess_payload(
            make_preprocessing_payload(
                inputs={"text": "你好", "references": [{"audio_path": "ref.wav"}]}
            ),
            config=INSTRUCT_CONFIG,
        )


@pytest.mark.parametrize(
    "sampling_overrides, expected_sampling",
    [
        ({"temperature": 0.0}, (0.0, 0.8, 20)),
        ({"top_p": 0.5, "top_k": 5}, (0.7, 0.5, 5)),
    ],
)
def test_instruct_explicit_sampling_overrides_task_default(
    sampling_overrides: dict[str, float | int],
    expected_sampling: tuple[float, float, int],
) -> None:
    """Supplied sampling fields must override VoiceDesign defaults without metadata."""
    state = preprocess_payload(
        instruct_payload(
            task_type="VoiceDesign",
            instructions="a calm deep voice",
            params=sampling_overrides,
        ),
        config=INSTRUCT_CONFIG,
    )
    assert (
        state.text_temperature,
        state.text_top_p,
        state.text_top_k,
    ) == expected_sampling


@pytest.mark.parametrize(
    "task,instructions,params,tts_params,user_instruction,engine_tokens,sampling",
    [
        (
            "tts",
            None,
            {},
            {},
            "Convert text to speech.\n源音频文本。晚上好，欢迎收听今天的节目。",
            420,
            (0.0, None, None),
        ),
        (
            "VoiceDesign",
            "a calm deep voice",
            {},
            {},
            "a calm deep voice\n\n根据上述音色描述，首先整理成语音属性，再合成以下文本对应的音频：\n晚上好，欢迎收听今天的节目。",
            622,
            (0.7, 0.8, 20),
        ),
        (
            "SemanticEdit",
            "Replace '公园' with '游乐园'.",
            {},
            {},
            "Identify the content of the audio. Replace '公园' with '游乐园'.",
            622,
            (0.0, None, None),
        ),
        (
            "AcousticEdit",
            None,
            {},
            {"speed": 0.8},
            "adjust the speed to 0.8x",
            420,
            (0.0, None, None),
        ),
        (
            "AcousticEdit",
            None,
            {
                "stage_params": {
                    "preprocessing": {
                        "acoustic_attribute": "Pitch",
                        "acoustic_value": -2,
                    }
                }
            },
            {},
            "shift the pitch by -2 steps",
            420,
            (0.0, None, None),
        ),
    ],
)
def test_instruct_preprocessing_routes_tasks_to_model_prompts(
    tmp_path: Path,
    task: str,
    instructions: str | None,
    params: dict[str, JsonValue],
    tts_params: dict[str, JsonValue],
    user_instruction: str,
    engine_tokens: int,
    sampling: tuple[float, float | None, int | None],
) -> None:
    audio_path = str(tmp_path / "edit.wav")
    sf.write(audio_path, np.zeros(69120, dtype=np.float32), 24000, subtype="FLOAT")
    is_edit = task in ("SemanticEdit", "AcousticEdit")
    tokenizer = make_speaker_tokenizer()
    state = preprocess_payload(
        instruct_payload(
            task_type=None if task == "tts" else task,
            instructions=instructions,
            text="" if is_edit else "晚上好，欢迎收听今天的节目。",
            reference_audio_path=audio_path if task != "VoiceDesign" else None,
            params=params,
            tts_params=tts_params,
        ),
        tokenizer=tokenizer,
        config=INSTRUCT_CONFIG,
    )
    audio_input = "<|sosp|>" + "<|image_pad|>" * 18 + "<|eosp|>\n" if is_edit else ""
    if task == "tts":
        assistant_prefix = "<|sosp|>" + "<|video_pad|>" * 18
        assert state.prompt_text == "源音频文本。"
    elif task == "AcousticEdit":
        assistant_prefix = "<|sosp|>"
    else:
        assistant_prefix = "<|sot|>"
    expected = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{audio_input}{user_instruction} /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant_prefix}"
    assert tokenizer.decode(state.input_ids) == expected
    assert state.variant == "instruct"
    assert (
        state.task_type
        == {
            "tts": "tts",
            "VoiceDesign": "voice_design",
            "SemanticEdit": "semantic_edit",
            "AcousticEdit": "acoustic_edit",
        }[task]
    )
    assert state.audio_in_path == (audio_path if is_edit else None)
    assert state.prompt_audio_path == (audio_path if task == "tts" else None)
    assert state.inference_cfg == (2.0 if task == "tts" else 1.2)
    assert state.max_engine_tokens == engine_tokens
    assert (state.text_temperature, state.text_top_p, state.text_top_k) == sampling
