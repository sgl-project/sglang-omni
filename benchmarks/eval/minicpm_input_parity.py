"""Capture actual MiniCPM chat preprocessing without loading model weights."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from minicpmo.utils import get_video_frame_audio_segments
from PIL import Image
from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
from sglang_omni.proto.request import OmniRequest, StagePayload


class CaptureComplete(RuntimeError):
    """Stop the reference chat immediately after processor execution."""


def serialize(value: object) -> object:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if tensor.numel() <= 20000:
            return tensor.tolist()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, Image.Image):
        return {
            "size": list(value.size),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {name: serialize(element) for name, element in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(element) for element in value]
    return value


class ReferenceCapture:
    def __init__(self, processor: transformers.ProcessorMixin) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.result = None

    def __call__(
        self,
        prompts: list[str],
        images: list[list[Image.Image]] | None,
        audios: list[list[np.ndarray]] | None,
        audio_parts: list[list[int]] | None,
        **options: object,
    ) -> None:
        processed = self.processor(prompts, images, audios, audio_parts, **options)
        self.result = {
            "prompt_text": prompts[0],
            "media": {"images": images, "audios": audios, "audio_parts": audio_parts},
            "processed": dict(processed),
        }
        raise CaptureComplete


def cases() -> dict[str, list[dict[str, object]]]:
    red = Image.new("RGB", (32, 32), "red")
    blue = Image.new("RGB", (32, 32), "blue")
    audio = np.sin(np.arange(16000, dtype=np.float32) * 0.03) * np.float32(0.1)
    return {
        "text": [{"role": "user", "content": "Say hello."}],
        "text_parts": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First"},
                    {"type": "text", "text": "Second"},
                ],
            }
        ],
        "image": [{"role": "user", "content": [red, "What color?"]}],
        "interleaved": [
            {"role": "user", "content": ["First", red, "Then", blue, "Compare."]}
        ],
        "multi_turn": [
            {"role": "user", "content": [red, "Remember this."]},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": [blue, "Compare with the previous image."]},
        ],
        "audio": [{"role": "user", "content": ["Transcribe.", audio]}],
        "audio_same_turn": [{"role": "user", "content": [audio, "Next", audio[:8000]]}],
        "audio_multi_turn": [
            {"role": "user", "content": [audio, "Remember."]},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": [audio[:8000], "Compare."]},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backend", choices=["official", "omni"], required=True)
    parser.add_argument("--family", choices=["minicpm", "qwen"], default="minicpm")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    processor = AutoProcessor.from_pretrained(
        arguments.model_path, trust_remote_code=True
    )
    results = {}
    if arguments.family == "qwen":
        from sglang_omni.models.qwen3_omni.components.preprocessor import (
            Qwen3OmniPreprocessor,
        )

        preprocessor = Qwen3OmniPreprocessor(arguments.model_path, max_seq_len=32768)
        processor = preprocessor.processor
        for name, messages in cases().items():
            structured = []
            images, audios = [], []
            for message in messages:
                content = message["content"]
                if isinstance(content, str):
                    structured.append(message)
                    continue
                parts = []
                for part in content:
                    if isinstance(part, Image.Image):
                        images.append(part)
                        parts.append({"type": "image", "image": part})
                    elif isinstance(part, np.ndarray):
                        audios.append(part)
                        parts.append({"type": "audio", "audio": part})
                    elif isinstance(part, str):
                        parts.append({"type": "text", "text": part})
                    else:
                        parts.append(part)
                structured.append({"role": message["role"], "content": parts})
            if arguments.backend == "official":
                prompt = processor.apply_chat_template(
                    structured, tokenize=False, add_generation_prompt=True
                )
                processed = processor(
                    text=prompt,
                    images=images or None,
                    audio=audios or None,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                results[name] = serialize(
                    {"prompt_text": prompt, "processed": dict(processed)}
                )
            else:
                try:
                    payload = StagePayload(
                        request_id=name,
                        request=OmniRequest(
                            inputs=structured, params={"max_new_tokens": 256}
                        ),
                        data=None,
                    )
                    results[name] = serialize(asyncio.run(preprocessor(payload)).data)
                except (ValueError, TypeError, RuntimeError) as error:
                    results[name] = {"error": str(error)}
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"Captured {len(results)} cases in {arguments.output}")
        return
    fixtures = cases()
    if arguments.video:
        frames, segments, _ = get_video_frame_audio_segments(str(arguments.video))
        content = [element for pair in zip(frames, segments) for element in pair]
        fixtures = {"video": [{"role": "user", "content": content}]}
    if arguments.backend == "official":
        model_class = get_class_from_dynamic_module(
            "modeling_minicpmo.MiniCPMO", arguments.model_path
        )
        capture = ReferenceCapture(processor)
        reference = SimpleNamespace(
            processor=capture,
            prepare_processor=lambda **options: None,
            device=torch.device("cpu"),
        )
        for name, messages in fixtures.items():
            try:
                model_class.chat(
                    reference,
                    msgs=messages,
                    do_sample=False,
                    enable_thinking=False,
                    use_tts_template=False,
                    omni_mode=bool(arguments.video),
                    **(
                        {"max_slice_nums": 1, "use_image_id": False}
                        if arguments.video
                        else {}
                    ),
                )
            except CaptureComplete:
                results[name] = serialize(capture.result)
    else:
        preprocessor = MiniCPMOPreprocessor(arguments.model_path)
        preprocessor._processor = processor
        for name, messages in fixtures.items():
            try:
                if arguments.video:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video_url",
                                    "video_url": {
                                        "url": str(arguments.video),
                                        "use_audio": True,
                                    },
                                }
                            ],
                        }
                    ]
                payload = StagePayload(
                    request_id=name, request=OmniRequest(inputs=messages), data=None
                )
                result = asyncio.run(preprocessor(payload))
                results[name] = serialize(result.data)
            except (ValueError, TypeError, RuntimeError) as error:
                results[name] = {"error": str(error)}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Captured {len(results)} cases in {arguments.output}")


if __name__ == "__main__":
    main()
