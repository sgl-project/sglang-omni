# SPDX-License-Identifier: Apache-2.0
"""Multimodal thinker wiring tests."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang_omni.models.ming_omni.thinker import BailingMoeV2Attention

MING_PREPROCESSOR_PATH = Path("sglang_omni/models/ming_omni/components/preprocessor.py")


def test_ming_thinker_attention_feeds_o_proj_one_row_per_token() -> None:
    head_dim, num_heads, num_tokens = 8, 2, 3

    def return_query(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: SimpleNamespace,
    ) -> torch.Tensor:
        return q

    attention = BailingMoeV2Attention.__new__(BailingMoeV2Attention)
    attention.attn = return_query
    q = torch.arange(num_tokens * num_heads * head_dim, dtype=torch.float32).reshape(
        num_tokens, num_heads, head_dim
    )

    output = attention.forward_core(q, q, q, SimpleNamespace())

    assert torch.equal(output, q.flatten(1))


def load_preprocessor_with_fake_deps(monkeypatch, *, config=None, tokenizer=None):
    common_module = ModuleType("sglang_omni.models.ming_omni.components.common")
    common_module.load_ming_config = lambda model_path: config
    common_module.load_ming_tokenizer = lambda model_path: tokenizer
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.ming_omni.components.common",
        common_module,
    )

    io_module = ModuleType("sglang_omni.models.ming_omni.io")
    io_module.MingOmniPipelineState = object
    io_module.PromptInputs = dict
    io_module.__dict__.update(
        AudioEncoderInputs=dict, ImageEncoderInputs=dict, SkippedEncoderInputs=dict
    )
    monkeypatch.setitem(sys.modules, "sglang_omni.models.ming_omni.io", io_module)

    next_stage_module = ModuleType("sglang_omni.models.ming_omni.pipeline.next_stage")
    next_stage_module.AUDIO_STAGE = "audio_encoder"
    next_stage_module.IMAGE_STAGE = "image_encoder"
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.ming_omni.pipeline.next_stage",
        next_stage_module,
    )

    audio_module = ModuleType("sglang_omni.preprocessing.audio")
    audio_module.load_audio_path = lambda *args, **kwargs: None
    audio_module.compute_audio_cache_key = lambda audios: None
    monkeypatch.setitem(sys.modules, "sglang_omni.preprocessing.audio", audio_module)

    image_module = ModuleType("sglang_omni.preprocessing.image")
    image_module.ensure_image_list_async = lambda images: images
    image_module.compute_image_cache_key = lambda images: None
    monkeypatch.setitem(sys.modules, "sglang_omni.preprocessing.image", image_module)

    proto_module = ModuleType("sglang_omni.proto")
    proto_module.StagePayload = object
    monkeypatch.setitem(sys.modules, "sglang_omni.proto", proto_module)

    module_name = "_ming_preprocessor_under_test"
    spec = importlib.util.spec_from_file_location(module_name, MING_PREPROCESSOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    sys.modules.pop(module_name, None)
    return module


class FakeMingTokenizer:
    unk_token_id = -1
    ids = {
        "<audio>": 10,
        "</audio>": 11,
        "<audioPatch>": 12,
        "<image>": 20,
        "</image>": 21,
        "<imagePatch>": 22,
    }

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in self.ids and "Patch" not in text:
            return [self.ids[text]]
        return [1000 + (ord(ch) % 127) for ch in text]


def test_ming_preprocessor_builds_placeholder_input_ids_directly(monkeypatch) -> None:
    module = load_preprocessor_with_fake_deps(monkeypatch)
    processor = module.MingPreprocessor.__new__(module.MingPreprocessor)
    processor.tokenizer = FakeMingTokenizer()
    processor.audio_patch_id = processor.tokenizer.convert_tokens_to_ids(
        module.AUDIO_PATCH
    )
    processor.audio_start_id = processor.tokenizer.convert_tokens_to_ids(
        module.AUDIO_START
    )
    processor.audio_end_id = processor.tokenizer.convert_tokens_to_ids(module.AUDIO_END)
    processor.image_patch_id = processor.tokenizer.convert_tokens_to_ids(
        module.IMAGE_PATCH
    )

    prompt_text, input_ids, audio_positions = processor.build_prompt(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "image://1"}},
                    {"type": "text", "text": " describe "},
                    {"type": "audio_url", "audio_url": {"url": "audio://1"}},
                ],
            }
        ],
        image_token_counts=[3],
        audio_token_counts=[2],
    )

    assert "<image><imagePatch><imagePatch><imagePatch></image>" in prompt_text
    assert input_ids.count(22) == 3
    assert input_ids.count(12) == 2
    assert audio_positions == [input_ids.index(12)]


def test_ming_preprocessor_uses_config_image_patch_token_id(monkeypatch) -> None:
    config = SimpleNamespace(
        llm_config=SimpleNamespace(image_patch_token=222),
        audio_config=SimpleNamespace(),
        vision_config=SimpleNamespace(),
    )
    tokenizer = FakeMingTokenizer()
    tokenizer.ids = {**tokenizer.ids, "<imagePatch>": 999}
    module = load_preprocessor_with_fake_deps(
        monkeypatch,
        config=config,
        tokenizer=tokenizer,
    )

    processor = module.MingPreprocessor("fake-model")

    assert processor.image_patch_id == 222


def test_ming_config_and_stages_do_not_import_ming_runner() -> None:
    runner_module = "sglang_omni.model_runner.ming_thinker_model_runner"
    sys.modules.pop(runner_module, None)

    importlib.import_module("sglang_omni.models.ming_omni.config")
    importlib.import_module("sglang_omni.models.ming_omni.stages")

    assert runner_module not in sys.modules


def load_runner_with_fake_sglang(monkeypatch):
    scheduler_module = ModuleType("sglang.srt.managers.scheduler")

    class GenerationBatchResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    scheduler_module.GenerationBatchResult = GenerationBatchResult
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.scheduler",
        scheduler_module,
    )

    module_name = "sglang_omni.model_runner.ming_thinker_model_runner"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    runner_cls = module.MingThinkerModelRunner
    purge_cached_module(module_name, module)
    assert module_name not in sys.modules
    return torch, runner_cls


def purge_cached_module(module_name: str, module: ModuleType) -> None:
    sys.modules.pop(module_name, None)
    parent_name, _, child_name = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if isinstance(parent, ModuleType) and getattr(parent, child_name, None) is module:
        delattr(parent, child_name)


class FakeEmbedTokens:
    num_embeddings = 16

    def __init__(self, torch_module):
        self.torch = torch_module

    def __call__(self, input_ids):
        values = input_ids.to(dtype=self.torch.float32)
        return self.torch.stack((values, values + 100), dim=-1)


def fake_runner(torch_module, runner_cls, **token_ids):
    runner = runner_cls.__new__(runner_cls)
    runner.embed_tokens = FakeEmbedTokens(torch_module)
    runner.image_token_id = token_ids.get("image", 3)
    runner.video_token_id = token_ids.get("video", 4)
    runner.audio_token_id = token_ids.get("audio", 5)
    return runner


def fake_batch(torch_module, input_ids, req):
    forward_batch = SimpleNamespace(
        input_ids=torch_module.tensor(input_ids, dtype=torch_module.long),
        extend_seq_lens_cpu=[len(input_ids)],
    )
    schedule_batch = SimpleNamespace(reqs=[req])
    return forward_batch, schedule_batch


def fake_req(model_inputs, *, inflight_middle_chunks=0, rid="req-1"):
    return SimpleNamespace(
        rid=rid,
        omni_model_inputs=model_inputs,
        _omni_consumed=None,
        inflight_middle_chunks=inflight_middle_chunks,
    )


def test_ming_runner_uses_pad_value_when_token_id_is_none(monkeypatch) -> None:
    torch, runner_cls = load_runner_with_fake_sglang(monkeypatch)
    runner = fake_runner(torch, runner_cls, audio=None)
    audio_embeds = torch.tensor([[30.0, 31.0]])
    req = fake_req(
        {"audio_embeds": audio_embeds, "pad_values": {"audio": 12}},
        rid="pad-audio",
    )
    forward_batch, schedule_batch = fake_batch(torch, [12, 1], req)

    input_embeds = runner.inject_multimodal_embeds(forward_batch, schedule_batch)

    assert torch.equal(input_embeds[0], audio_embeds[0])
    assert req.omni_model_inputs is None
    assert req._omni_consumed is None  # noqa: leading-underscore  # production name


def test_ming_runner_raises_when_embeds_are_short(monkeypatch) -> None:
    torch, runner_cls = load_runner_with_fake_sglang(monkeypatch)
    runner = fake_runner(torch, runner_cls, image=3)
    req = fake_req({"image_embeds": torch.ones(1, 2)}, rid="short-image")
    forward_batch, schedule_batch = fake_batch(torch, [3, 3], req)

    with pytest.raises(ValueError, match="image.*short-image.*needed=2.*available=1"):
        runner.inject_multimodal_embeds(forward_batch, schedule_batch)

    assert req.omni_model_inputs is not None


def test_ming_runner_successful_final_injection_consumes_and_clears(
    monkeypatch,
) -> None:
    torch, runner_cls = load_runner_with_fake_sglang(monkeypatch)
    runner = fake_runner(torch, runner_cls, image=3, audio=5)
    image_embeds = torch.tensor([[20.0, 21.0]])
    audio_embeds = torch.tensor([[30.0, 31.0]])
    req = fake_req(
        {"image_embeds": image_embeds, "audio_embeds": audio_embeds},
        rid="ok-mm",
    )
    forward_batch, schedule_batch = fake_batch(torch, [3, 5, 1], req)

    input_embeds = runner.inject_multimodal_embeds(forward_batch, schedule_batch)

    assert torch.equal(input_embeds[0], image_embeds[0])
    assert torch.equal(input_embeds[1], audio_embeds[0])
    assert req.omni_model_inputs is None
    assert req._omni_consumed is None  # noqa: leading-underscore  # production name


def test_ming_runner_keeps_chunk_state_until_final_chunk(monkeypatch) -> None:
    torch, runner_cls = load_runner_with_fake_sglang(monkeypatch)
    runner = fake_runner(torch, runner_cls, image=3)
    image_embeds = torch.tensor([[20.0, 21.0], [22.0, 23.0]])
    model_inputs = {"image_embeds": image_embeds}
    req = fake_req(model_inputs, inflight_middle_chunks=1, rid="chunked-image")
    forward_batch, schedule_batch = fake_batch(torch, [3, 1], req)

    input_embeds = runner.inject_multimodal_embeds(forward_batch, schedule_batch)

    assert torch.equal(input_embeds[0], image_embeds[0])
    assert req.omni_model_inputs is model_inputs
    assert req._omni_consumed == {
        "image": 1
    }  # noqa: leading-underscore  # production name

    req.inflight_middle_chunks = 0
    forward_batch, schedule_batch = fake_batch(torch, [3, 2], req)

    input_embeds = runner.inject_multimodal_embeds(forward_batch, schedule_batch)

    assert torch.equal(input_embeds[0], image_embeds[1])
    assert req.omni_model_inputs is None
    assert req._omni_consumed is None  # noqa: leading-underscore  # production name
    assert model_inputs == {"image_embeds": image_embeds}


def test_ming_thinker_forward_publishes_sglang_forward_context() -> None:
    from sglang.srt.model_executor.forward_context import (
        get_forward_context,
        has_forward_context,
    )

    from sglang_omni.model_runner.ming_thinker_model_runner import (
        MingThinkerModelRunner,
    )

    runner = MingThinkerModelRunner.__new__(MingThinkerModelRunner)
    attn_backend = SimpleNamespace(init_forward_metadata=lambda batch: None)
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=attn_backend)
    )

    seen = []

    def model(**kwargs):
        seen.append(get_forward_context().attn_backend)
        return torch.ones(1, 2)

    def logits_processor(input_ids, hidden_states, lm_head, forward_batch):
        seen.append(get_forward_context().attn_backend)
        assert input_ids is forward_batch.input_ids
        assert lm_head == "lm_head"
        return "logits"

    runner.outer_model = SimpleNamespace(
        model=model,
        logits_processor=logits_processor,
        lm_head="lm_head",
    )
    forward_batch = SimpleNamespace(
        positions=torch.tensor([0]),
        mrope_positions=None,
        input_ids=torch.tensor([1]),
    )

    assert not has_forward_context()
    result = runner.forward_with_omni_embeds(forward_batch, torch.ones(1, 2))

    assert seen == [attn_backend, attn_backend]
    assert result.logits_output == "logits"
