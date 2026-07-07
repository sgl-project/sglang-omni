# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import importlib.util
import queue as queue_mod
import sys
import threading
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

from sglang_omni.models.llada2_uni.components.image_decoder import (
    DecodedImage,
    LLaDA2ImageDecoder,
    pil_image_to_bytes,
)
from sglang_omni.models.llada2_uni.components.image_token_generator import (
    LLaDA2ImageTokenGenerator,
)
from sglang_omni.models.llada2_uni.components.preprocessor import (
    DEFAULT_IMAGE_GEN_LENGTH,
    IMAGE_STAGE,
    LLaDA2Preprocessor,
    build_image_generation_config,
)
from sglang_omni.models.llada2_uni.config import IMAGE_DECODE_STAGE
from sglang_omni.models.llada2_uni.hybrid_scheduler import LLaDA2HybridThinkerScheduler
from sglang_omni.models.llada2_uni.merge import decode_events
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.llada2_uni.stages import create_image_decode_executor
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage


class RecordingTokenizer:
    def __init__(self) -> None:
        self.encoded: list[str] = []
        self.ids: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        self.encoded.append(text)
        if text not in self.ids:
            self.ids[text] = len(self.ids) + 1
        return [self.ids[text]]


def test_image_generation_config_builds_semantic_grid() -> None:
    request = OmniRequest(
        inputs={"messages": [{"role": "user", "content": "draw a red kite"}]},
        params={},
        metadata={
            "output_modalities": ["image"],
            "image_config": {
                "width": 512,
                "height": 768,
                "steps": 8,
                "cfg_scale": 2.0,
                "decode_mode": "decoder-turbo",
            },
        },
    )

    generation = build_image_generation_config(request)

    assert generation["type"] == "image"
    assert generation["width"] == 512
    assert generation["height"] == 768
    assert generation["token_grid_h"] == 24
    assert generation["token_grid_w"] == 16
    assert generation["num_image_tokens"] == 384
    assert generation["gen_length"] == DEFAULT_IMAGE_GEN_LENGTH
    assert generation["steps"] == 8
    assert generation["cfg_scale"] == 2.0
    assert generation["decoder_steps"] == 8
    assert generation["decode_mode"] == "decoder-turbo"
    assert generation["format"] == "png"


def test_image_generation_preprocessor_builds_t2i_prompt_state() -> None:
    tokenizer = RecordingTokenizer()
    preprocessor = object.__new__(LLaDA2Preprocessor)
    preprocessor._max_seq_len = 2048
    preprocessor._tokenizer = tokenizer

    request = OmniRequest(
        inputs={
            "messages": [
                {"role": "system", "content": "ignored"},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Draw a lighthouse at dusk."}],
                },
            ]
        },
        params={},
        metadata={
            "output_modalities": ["image"],
            "image_config": {"width": 512, "height": 512},
        },
    )
    payload = StagePayload(request_id="req-1", request=request, data=None)

    output = asyncio.run(preprocessor(payload))
    state = LLaDA2UniPipelineState.from_dict(output.data)

    assert state.encoder_inputs == {IMAGE_STAGE: {"_skip": True, "_result": {}}}
    assert state.generation["type"] == "image"
    assert state.generation["token_grid_h"] == 16
    assert state.generation["token_grid_w"] == 16
    assert state.generation["num_image_tokens"] == 256
    assert "Draw a lighthouse at dusk." in state.generation["text_prompt"]
    assert "<|reserved_token_16|>" in state.generation["image_header"]
    assert state.prompt is not None
    assert state.prompt["input_ids"].shape == (1, 7)

    (
        header_soi,
        header_height,
        header_width,
        header_boi,
        conditional_prefix,
        conditional_prompt,
        conditional_assistant,
        unconditional_prefix,
        unconditional_prompt,
        unconditional_assistant,
    ) = tokenizer.encoded
    assert state.prompt["input_ids"].tolist()[0] == [
        tokenizer.ids[conditional_prefix],
        tokenizer.ids[conditional_prompt],
        tokenizer.ids[conditional_assistant],
        tokenizer.ids[header_soi],
        tokenizer.ids[header_height],
        tokenizer.ids[header_width],
        tokenizer.ids[header_boi],
    ]
    assert state.generation["uncond_ids"] == [
        tokenizer.ids[unconditional_prefix],
        tokenizer.ids[unconditional_prompt],
        tokenizer.ids[unconditional_assistant],
        tokenizer.ids[header_soi],
        tokenizer.ids[header_height],
        tokenizer.ids[header_width],
        tokenizer.ids[header_boi],
    ]
    assert conditional_prefix == unconditional_prefix
    assert "You are a text-to-image generation assistant." in conditional_prefix
    assert "Draw a lighthouse at dusk." in conditional_prompt
    assert conditional_assistant == "<role>ASSISTANT</role>"
    assert "<uncondition>" in unconditional_prompt
    assert unconditional_assistant == conditional_assistant
    assert header_soi == "<|image|>"
    assert header_height == "<|reserved_token_16|>"
    assert header_width == "<|reserved_token_16|>"
    assert header_boi == "<boi>"


def test_image_generation_decode_returns_image_token_event() -> None:
    events = decode_events(
        thinker_out={"output_ids": [157184, 157185, 157186]},
        tokenizer=object(),
        generation={
            "type": "image",
            "image_token_offset": 157184,
            "num_image_tokens": 2,
            "token_grid_h": 1,
            "token_grid_w": 2,
            "width": 64,
            "height": 32,
            "decoder_steps": 8,
            "resolution_multiplier": 2,
            "decode_mode": "decoder-turbo",
        },
    )

    assert len(events) == 1
    event = events[0]
    assert event.type == "image_tokens_final"
    assert event.modality == "image"
    assert event.is_final is True
    assert event.payload["image_token_ids"] == [0, 1]
    assert event.payload["token_grid_h"] == 1
    assert event.payload["token_grid_w"] == 2


def test_image_token_generator_uses_upstream_generate_image(monkeypatch) -> None:
    calls = []

    class FakeTokenizer:
        pass

    class FakeModel:
        def eval(self):
            return self

        def generate_image(self, prompt, **kwargs):
            calls.append(("generate_image", prompt, kwargs))
            return {"token_ids": [7, 8], "h": 1, "w": 2}

    def fake_tokenizer_from_pretrained(model_path, **kwargs):
        calls.append(("tokenizer", model_path, kwargs))
        return FakeTokenizer()

    def fake_model_from_pretrained(model_path, **kwargs):
        calls.append(("model", model_path, kwargs))
        return FakeModel()

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        fake_model_from_pretrained,
    )

    generator = LLaDA2ImageTokenGenerator(
        "/tmp/llada2",
        device="cpu",
        dtype=None,
        local_files_only=True,
    )
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[1, 2, 3]])},
        generation={
            "type": "image",
            "text_prompt": "Draw a blue cube.",
            "width": 64,
            "height": 32,
            "steps": 5,
            "block_length": 16,
            "cfg_scale": 2.5,
            "gen_length": 64,
            "token_grid_h": 1,
            "token_grid_w": 2,
            "num_image_tokens": 2,
            "image_token_offset": 157184,
            "seed": 123,
        },
    )
    payload = StagePayload(
        request_id="req-img",
        request=OmniRequest(inputs="draw", metadata={"output_modalities": ["image"]}),
        data=state.to_dict(),
    )

    out = generator(payload)
    out_state = LLaDA2UniPipelineState.from_dict(out.data)

    assert calls[0][0] == "tokenizer"
    assert calls[1][0] == "model"
    assert calls[2][0] == "generate_image"
    assert calls[2][1] == "Draw a blue cube."
    assert calls[2][2]["image_h"] == 32
    assert calls[2][2]["image_w"] == 64
    assert calls[2][2]["steps"] == 5
    assert calls[2][2]["block_length"] == 16
    assert calls[2][2]["cfg_scale"] == 2.5
    assert out_state.thinker_out == {
        "output_ids": [157191, 157192],
        "is_final": True,
        "finish_reason": "length",
    }


def test_hybrid_thinker_routes_image_and_lazily_starts_text_scheduler() -> None:
    text_factory_calls = []
    image_calls = []

    class FakeTextScheduler:
        def __init__(self) -> None:
            self.inbox: queue_mod.Queue = queue_mod.Queue()
            self.outbox: queue_mod.Queue = queue_mod.Queue()
            self.running = False

        def start(self) -> None:
            self.running = True
            while self.running:
                try:
                    msg = self.inbox.get(timeout=0.01)
                except queue_mod.Empty:
                    continue
                self.outbox.put(
                    OutgoingMessage(
                        request_id=msg.request_id,
                        type="result",
                        data=("text", msg.data),
                    )
                )

        def stop(self) -> None:
            self.running = False

        def abort(self, request_id: str) -> None:
            pass

    def text_scheduler_factory():
        scheduler = FakeTextScheduler()
        text_factory_calls.append(scheduler)
        return scheduler

    def image_compute_fn(payload):
        image_calls.append(payload)
        return ("image", payload)

    scheduler = LLaDA2HybridThinkerScheduler(
        text_scheduler_factory=text_scheduler_factory,
        image_compute_fn=image_compute_fn,
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        image_state = LLaDA2UniPipelineState(
            prompt={"input_ids": torch.tensor([[1]])},
            generation={"type": "image"},
        )
        image_payload = StagePayload(
            request_id="req-img",
            request=OmniRequest(inputs="draw"),
            data=image_state.to_dict(),
        )
        scheduler.inbox.put(
            IncomingMessage(
                request_id="req-img",
                type="new_request",
                data=image_payload,
            )
        )
        image_msg = scheduler.outbox.get(timeout=1)

        assert image_msg.request_id == "req-img"
        assert image_msg.type == "result"
        assert image_msg.data == ("image", image_payload)
        assert image_calls == [image_payload]
        assert text_factory_calls == []

        text_state = LLaDA2UniPipelineState(
            prompt={"input_ids": torch.tensor([[2]])},
            generation={},
        )
        text_payload = StagePayload(
            request_id="req-text",
            request=OmniRequest(inputs="hello"),
            data=text_state.to_dict(),
        )
        scheduler.inbox.put(
            IncomingMessage(
                request_id="req-text",
                type="new_request",
                data=text_payload,
            )
        )
        text_msg = scheduler.outbox.get(timeout=1)

        assert text_msg.request_id == "req-text"
        assert text_msg.type == "result"
        assert text_msg.data == ("text", text_payload)
        assert len(text_factory_calls) == 1
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_image_decoder_encodes_png_and_jpeg_bytes() -> None:
    image = Image.new("RGB", (2, 1), color=(255, 0, 0))

    png_bytes, png_format, png_mime = pil_image_to_bytes(image, image_format="png")
    jpeg_bytes, jpeg_format, jpeg_mime = pil_image_to_bytes(image, image_format="jpg")

    assert png_bytes.startswith(b"\x89PNG")
    assert png_format == "png"
    assert png_mime == "image/png"
    assert jpeg_bytes.startswith(b"\xff\xd8")
    assert jpeg_format == "jpeg"
    assert jpeg_mime == "image/jpeg"
    assert base64.b64encode(png_bytes).decode("ascii")


def test_decoder_model_imports_without_flash_attn(monkeypatch) -> None:
    decoder_model_path = (
        Path(__file__).parents[3]
        / "sglang_omni/models/llada2_uni/components/decoder_model.py"
    )
    monkeypatch.setitem(sys.modules, "flash_attn", None)

    spec = importlib.util.spec_from_file_location(
        "llada2_decoder_model_no_flash_attn_test",
        decoder_model_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._HAS_FLASH_ATTN is False
    assert module.flash_attn_func is None


def test_image_decoder_seed_uses_global_torch_rng() -> None:
    decoder = object.__new__(LLaDA2ImageDecoder)
    torch.manual_seed(999)

    decoder._seed_decode(123)
    seeded_sample = torch.randn(4)

    torch.manual_seed(123)
    expected_sample = torch.randn(4)

    assert torch.equal(seeded_sample, expected_sample)


def test_image_decode_stage_is_lazy_for_text_requests(monkeypatch) -> None:
    calls = []

    class FakeImageDecoder:
        def __init__(self, model_path, *, device="cuda", dtype=None):
            calls.append(("init", model_path, device, dtype))

    monkeypatch.setattr(
        "sglang_omni.models.llada2_uni.components.image_decoder.LLaDA2ImageDecoder",
        FakeImageDecoder,
    )

    executor = create_image_decode_executor("/tmp/llada2", device="cpu", dtype=None)
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[1, 2, 3]])},
        thinker_out={"output_ids": [1, 2], "is_final": True},
        generation={},
    )
    payload = StagePayload(
        request_id="req-text",
        request=OmniRequest(inputs="hello", metadata={"output_modalities": ["text"]}),
        data=state.to_dict(),
    )

    out = executor._fn(payload)

    assert out is payload
    assert calls == []


def test_image_decode_stage_converts_vq_tokens_to_image_payload(monkeypatch) -> None:
    calls = []

    class FakeImageDecoder:
        def __init__(self, model_path, *, device="cuda", dtype=None):
            calls.append(("init", model_path, device, dtype))

        def decode(self, token_ids, **kwargs):
            calls.append(("decode", token_ids, kwargs))
            image = Image.new("RGB", (4, 2), color=(0, 255, 0))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            return DecodedImage(
                image_bytes=image_bytes,
                data=base64.b64encode(image_bytes).decode("ascii"),
                format="png",
                mime_type="image/png",
                width=4,
                height=2,
            )

    monkeypatch.setattr(
        "sglang_omni.models.llada2_uni.components.image_decoder.LLaDA2ImageDecoder",
        FakeImageDecoder,
    )

    executor = create_image_decode_executor("/tmp/llada2", device="cpu", dtype=None)
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[1, 2, 3]])},
        thinker_out={"output_ids": [157184, 157185], "is_final": True},
        generation={
            "type": "image",
            "image_token_offset": 157184,
            "num_image_tokens": 2,
            "token_grid_h": 1,
            "token_grid_w": 2,
            "width": 64,
            "height": 32,
            "decoder_steps": 8,
            "resolution_multiplier": 2,
            "decode_mode": "decoder-turbo",
            "format": "png",
            "seed": 123,
        },
    )
    payload = StagePayload(
        request_id="req-img",
        request=OmniRequest(inputs="draw", metadata={"output_modalities": ["image"]}),
        data=state.to_dict(),
    )

    out = executor._fn(payload)
    out_state = LLaDA2UniPipelineState.from_dict(out.data)
    decoded = out_state.engine_outputs[IMAGE_DECODE_STAGE]

    assert calls[0] == ("init", "/tmp/llada2", "cpu", torch.bfloat16)
    assert calls[1][0] == "decode"
    assert calls[1][1] == [0, 1]
    assert calls[1][2]["token_grid_h"] == 1
    assert calls[1][2]["token_grid_w"] == 2
    assert calls[1][2]["decode_mode"] == "decoder-turbo"
    assert calls[1][2]["seed"] == 123
    assert decoded["modality"] == "image"
    assert decoded["images"][0]["format"] == "png"
    assert decoded["images"][0]["mime_type"] == "image/png"
    assert decoded["images"][0]["width"] == 4
    assert decoded["events"][-1]["type"] == "image_final"
