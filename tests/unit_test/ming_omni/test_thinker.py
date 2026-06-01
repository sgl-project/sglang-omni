# SPDX-License-Identifier: Apache-2.0
"""Multimodal thinker wiring tests."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

BOOTSTRAP_PATH = Path("sglang_omni/models/ming_omni/bootstrap.py")
RUNNER_PATH = Path("sglang_omni/model_runner/ming_thinker_model_runner.py")
MING_THINKER_PATH = Path("sglang_omni/models/ming_omni/thinker.py")
MING_IMAGE_ENCODER_PATH = Path(
    "sglang_omni/models/ming_omni/components/image_encoder.py"
)
MING_PREPROCESSOR_PATH = Path("sglang_omni/models/ming_omni/components/preprocessor.py")
VENDOR_SGLANG_LAYERS_PATH = Path("sglang_omni/vendor/sglang/layers.py")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_ming_bootstrap_wires_ming_thinker_model_runner() -> None:
    source = _read(BOOTSTRAP_PATH)

    assert "MingThinkerModelRunner" in source
    assert "SGLangOutputProcessor" in source
    assert "model=model_worker.model_runner.model" in source
    assert "model_runner = MingThinkerModelRunner(model_worker, output_proc)" in source
    assert "model_runner=model_runner" in source
    assert 'model_arch_override="BailingMoeV2ForCausalLM"' in source


def test_ming_thinker_runner_source_injects_multimodal_embeds() -> None:
    source = _read(RUNNER_PATH)

    assert "class MingThinkerModelRunner(ModelRunner)" in source
    assert "audio_embeds" in source
    assert "image_embeds" in source
    assert "_resolve_match_id" in source
    assert "_validate_final_consumption" in source
    assert "continue" in source
    assert ".clamp(" in source
    assert "self._embed_tokens.num_embeddings - 1" in source
    assert "outer.model(" in source
    assert "input_ids=None" in source
    assert "input_embeds=input_embeds" in source
    assert "outer.logits_processor(" in source
    assert "GenerationBatchResult(" in source
    assert "can_run_cuda_graph=False" in source
    assert "req.omni_model_inputs = None" in source
    assert "req._omni_consumed = None" in source


def test_ming_thinker_runner_uses_ming_token_id_contract() -> None:
    source = _read(RUNNER_PATH)

    assert "hf_config" in source
    assert "llm_config" in source
    assert "image_token_id" in source
    assert "video_token_id" in source
    assert "audio_token_id" in source
    assert "image_patch_token" in source
    assert "video_patch_token" in source
    assert "thinker_config" not in source


def test_ming_thinker_weight_loader_uses_qwen3_helper_path() -> None:
    source = _read(MING_THINKER_PATH)

    assert "sglang_omni.models.qwen3_omni.thinker" not in source
    assert "sglang_omni.models.qwen3_omni.components.thinker_model" in source
    assert "extract_fused_experts" in source


def test_ming_image_encoder_keeps_its_tp_context_for_runtime_forward() -> None:
    source = _read(MING_IMAGE_ENCODER_PATH)
    init_body = source.split("    @staticmethod", 1)[0]

    assert (
        "_init_sglang_tp(" in init_body
    ), "TP context must be initialized in __init__, not deferred to forward"
    assert "_cleanup_sglang_tp()" not in init_body


def test_ming_image_encoder_tp_init_requires_parallel_state() -> None:
    source = _read(MING_IMAGE_ENCODER_PATH)
    init_fn = source.split("def _init_sglang_tp", 1)[1].split(
        "    @classmethod\n    def _cleanup_sglang_tp", 1
    )[0]

    assert "parallel_state.model_parallel_is_initialized()" in init_fn
    assert 'if getattr(dp, "_ATTN_TP_SIZE", None) is not None' not in init_fn


def test_vendored_sglang_layers_do_not_import_removed_sampling_symbol() -> None:
    source = _read(VENDOR_SGLANG_LAYERS_PATH)

    assert "top_k_top_p_sampling_from_probs" not in source


def test_ming_vision_block_kwargs_pass_head_size_only_when_supported(
    monkeypatch,
) -> None:
    weight_loader_module = ModuleType("sglang_omni.models.weight_loader")
    weight_loader_module.default_weight_loader = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.weight_loader",
        weight_loader_module,
    )

    from sglang_omni.models.ming_omni.components.vision_encoder import (
        _build_qwen3_vision_block_kwargs,
    )

    class LegacyVisionBlock:
        def __init__(self, dim, num_heads, intermediate_dim, prefix=""):
            pass

    class NewVisionBlock:
        def __init__(self, dim, num_heads, head_size, intermediate_dim, prefix=""):
            pass

    common = {
        "dim": 1152,
        "num_heads": 16,
        "head_size": 72,
        "intermediate_dim": 4304,
        "hidden_act": "gelu_pytorch_tanh",
        "norm_layer": None,
        "quant_config": None,
        "prefix": "visual.blocks.0",
    }

    legacy_kwargs = _build_qwen3_vision_block_kwargs(
        LegacyVisionBlock,
        **common,
    )
    new_kwargs = _build_qwen3_vision_block_kwargs(NewVisionBlock, **common)

    assert "head_size" not in legacy_kwargs
    assert new_kwargs["head_size"] == 72


def _load_preprocessor_with_fake_deps(monkeypatch, *, config=None, tokenizer=None):
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


class _FakeMingTokenizer:
    unk_token_id = -1
    _ids = {
        "<audio>": 10,
        "</audio>": 11,
        "<audioPatch>": 12,
        "<image>": 20,
        "</image>": 21,
        "<imagePatch>": 22,
    }

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in self._ids and "Patch" not in text:
            return [self._ids[text]]
        return [1000 + (ord(ch) % 127) for ch in text]


def test_ming_preprocessor_builds_placeholder_input_ids_directly(monkeypatch) -> None:
    module = _load_preprocessor_with_fake_deps(monkeypatch)
    processor = module.MingPreprocessor.__new__(module.MingPreprocessor)
    processor._tokenizer = _FakeMingTokenizer()
    processor._audio_patch_id = processor._tokenizer.convert_tokens_to_ids(
        module.AUDIO_PATCH
    )
    processor._audio_start_id = processor._tokenizer.convert_tokens_to_ids(
        module.AUDIO_START
    )
    processor._audio_end_id = processor._tokenizer.convert_tokens_to_ids(
        module.AUDIO_END
    )
    processor._image_patch_id = processor._tokenizer.convert_tokens_to_ids(
        module.IMAGE_PATCH
    )

    prompt_text, input_ids, audio_positions = processor._build_prompt(
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
    tokenizer = _FakeMingTokenizer()
    tokenizer._ids = {**tokenizer._ids, "<imagePatch>": 999}
    module = _load_preprocessor_with_fake_deps(
        monkeypatch,
        config=config,
        tokenizer=tokenizer,
    )

    processor = module.MingPreprocessor("fake-model")

    assert processor._image_patch_id == 222


def test_ming_config_and_stages_do_not_import_ming_runner() -> None:
    runner_module = "sglang_omni.model_runner.ming_thinker_model_runner"
    sys.modules.pop(runner_module, None)

    importlib.import_module("sglang_omni.models.ming_omni.config")
    importlib.import_module("sglang_omni.models.ming_omni.stages")

    assert runner_module not in sys.modules


def _load_runner_with_fake_sglang(monkeypatch):
    torch = pytest.importorskip("torch")

    scheduler_module = ModuleType("sglang.srt.managers.scheduler")

    class GenerationBatchResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    scheduler_module.GenerationBatchResult = GenerationBatchResult
    monkeypatch.setitem(sys.modules, "sglang", ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", ModuleType("sglang.srt"))
    monkeypatch.setitem(
        sys.modules, "sglang.srt.managers", ModuleType("sglang.srt.managers")
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.scheduler",
        scheduler_module,
    )

    module_name = "sglang_omni.model_runner.ming_thinker_model_runner"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    runner_cls = module.MingThinkerModelRunner
    _purge_cached_module(module_name, module)
    assert module_name not in sys.modules
    return torch, runner_cls


def _purge_cached_module(module_name: str, module: ModuleType) -> None:
    sys.modules.pop(module_name, None)
    parent_name, _, child_name = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if isinstance(parent, ModuleType) and getattr(parent, child_name, None) is module:
        delattr(parent, child_name)


class _FakeEmbedTokens:
    num_embeddings = 16

    def __init__(self, torch_module):
        self._torch = torch_module

    def __call__(self, input_ids):
        values = input_ids.to(dtype=self._torch.float32)
        return self._torch.stack((values, values + 100), dim=-1)


def _fake_runner(torch_module, runner_cls, **token_ids):
    runner = runner_cls.__new__(runner_cls)
    runner._embed_tokens = _FakeEmbedTokens(torch_module)
    runner._image_token_id = token_ids.get("image", 3)
    runner._video_token_id = token_ids.get("video", 4)
    runner._audio_token_id = token_ids.get("audio", 5)
    return runner


def _fake_batch(torch_module, input_ids, req):
    forward_batch = SimpleNamespace(
        input_ids=torch_module.tensor(input_ids, dtype=torch_module.long),
        extend_seq_lens_cpu=[len(input_ids)],
    )
    schedule_batch = SimpleNamespace(reqs=[req])
    return forward_batch, schedule_batch


def _fake_req(model_inputs, *, is_chunked=0, rid="req-1"):
    return SimpleNamespace(
        rid=rid,
        omni_model_inputs=model_inputs,
        _omni_consumed=None,
        is_chunked=is_chunked,
    )


def test_ming_runner_uses_pad_value_when_token_id_is_none(monkeypatch) -> None:
    torch, runner_cls = _load_runner_with_fake_sglang(monkeypatch)
    runner = _fake_runner(torch, runner_cls, audio=None)
    audio_embeds = torch.tensor([[30.0, 31.0]])
    req = _fake_req(
        {"audio_embeds": audio_embeds, "pad_values": {"audio": 12}},
        rid="pad-audio",
    )
    forward_batch, schedule_batch = _fake_batch(torch, [12, 1], req)

    input_embeds = runner._inject_multimodal_embeds(forward_batch, schedule_batch)

    assert torch.equal(input_embeds[0], audio_embeds[0])
    assert req.omni_model_inputs is None
    assert req._omni_consumed is None


def test_ming_runner_raises_when_embeds_are_short(monkeypatch) -> None:
    torch, runner_cls = _load_runner_with_fake_sglang(monkeypatch)
    runner = _fake_runner(torch, runner_cls, image=3)
    req = _fake_req({"image_embeds": torch.ones(1, 2)}, rid="short-image")
    forward_batch, schedule_batch = _fake_batch(torch, [3, 3], req)

    with pytest.raises(ValueError, match="image.*short-image.*needed=2.*available=1"):
        runner._inject_multimodal_embeds(forward_batch, schedule_batch)

    assert req.omni_model_inputs is not None


def test_ming_runner_successful_final_injection_consumes_and_clears(
    monkeypatch,
) -> None:
    torch, runner_cls = _load_runner_with_fake_sglang(monkeypatch)
    runner = _fake_runner(torch, runner_cls, image=3, audio=5)
    image_embeds = torch.tensor([[20.0, 21.0]])
    audio_embeds = torch.tensor([[30.0, 31.0]])
    req = _fake_req(
        {"image_embeds": image_embeds, "audio_embeds": audio_embeds},
        rid="ok-mm",
    )
    forward_batch, schedule_batch = _fake_batch(torch, [3, 5, 1], req)

    input_embeds = runner._inject_multimodal_embeds(forward_batch, schedule_batch)

    assert torch.equal(input_embeds[0], image_embeds[0])
    assert torch.equal(input_embeds[1], audio_embeds[0])
    assert req.omni_model_inputs is None
    assert req._omni_consumed is None


def test_ming_runner_keeps_chunk_state_until_final_chunk(monkeypatch) -> None:
    torch, runner_cls = _load_runner_with_fake_sglang(monkeypatch)
    runner = _fake_runner(torch, runner_cls, image=3)
    image_embeds = torch.tensor([[20.0, 21.0], [22.0, 23.0]])
    model_inputs = {"image_embeds": image_embeds}
    req = _fake_req(model_inputs, is_chunked=1, rid="chunked-image")
    forward_batch, schedule_batch = _fake_batch(torch, [3, 1], req)

    input_embeds = runner._inject_multimodal_embeds(forward_batch, schedule_batch)

    assert torch.equal(input_embeds[0], image_embeds[0])
    assert req.omni_model_inputs is model_inputs
    assert req._omni_consumed == {"image": 1}

    req.is_chunked = 0
    forward_batch, schedule_batch = _fake_batch(torch, [3, 2], req)

    input_embeds = runner._inject_multimodal_embeds(forward_batch, schedule_batch)

    assert torch.equal(input_embeds[0], image_embeds[1])
    assert req.omni_model_inputs is None
    assert req._omni_consumed is None
    assert model_inputs == {"image_embeds": image_embeds}
