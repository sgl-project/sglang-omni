# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Iterator

import torch
from torch import nn


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


@contextmanager
def _nano_sglang_model_class() -> Iterator[type]:
    class FakeLocalModel(nn.Module):
        @property
        def dtype(self) -> torch.dtype:
            return torch.float32

        def _audio_embedding_weight(self, channel: int) -> torch.Tensor:
            return self.embedding_list[channel + 1].weight[
                : int(self.config.audio_vocab_size)
            ]

    stub_modules: dict[str, ModuleType] = {
        name: _package(name)
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.layers",
            "sglang.srt.layers.quantization",
            "sglang.srt.model_executor",
        )
    }
    module_attrs: dict[str, dict[str, object]] = {
        "sglang.srt.distributed": {"get_pp_group": lambda: None},
        "sglang.srt.layers.activation": {"NewGELU": object},
        "sglang.srt.layers.linear": {
            "ColumnParallelLinear": object,
            "QKVParallelLinear": object,
            "RowParallelLinear": object,
        },
        "sglang.srt.layers.logits_processor": {"LogitsProcessorOutput": object},
        "sglang.srt.layers.quantization.base_config": {"QuantizationConfig": object},
        "sglang.srt.layers.radix_attention": {"RadixAttention": object},
        "sglang.srt.layers.rotary_embedding": {"get_rope": lambda *a, **k: None},
        "sglang.srt.layers.utils": {
            "PPMissingLayer": object,
            "get_layer_id": lambda name: None,
        },
        "sglang.srt.layers.vocab_parallel_embedding": {
            "VocabParallelEmbedding": object
        },
        "sglang.srt.model_executor.forward_batch_info": {
            "ForwardBatch": object,
            "PPProxyTensors": object,
        },
        "sglang.srt.runtime_context": {
            "get_parallel": lambda: SimpleNamespace(tp_size=1)
        },
        "sglang.srt.utils": {"add_prefix": lambda name, prefix: name},
        "sglang_omni.models.moss_tts_local.sglang_model": {
            "MossTTSLocalSGLangModel": FakeLocalModel
        },
        "sglang_omni.models.moss_tts_local.state_pool": {
            "MossTTSLocalDecodeStatePool": object
        },
    }
    for name, attrs in module_attrs.items():
        module = ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        stub_modules[name] = module

    previous = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)
    module_name = "_moss_tts_nano_sglang_model_test"
    path = (
        Path(__file__).parents[3] / "sglang_omni/models/moss_tts_nano/sglang_model.py"
    )
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        yield module.MossTTSNanoSGLangModel
    finally:
        sys.modules.pop(module_name, None)
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class _RecordingLocalTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, torch.Tensor]] = []

    def step(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        self.calls.append((position, hidden_states.detach().clone()))
        return hidden_states


def _make_model(model_cls: type) -> nn.Module:
    model = object.__new__(model_cls)
    nn.Module.__init__(model)
    model.n_vq = 3
    model.config = SimpleNamespace(
        audio_assistant_slot_token_id=9,
        audio_end_token_id=7,
        audio_vocab_size=5,
    )
    model.embedding_list = nn.ModuleList(
        [nn.Embedding(16, 4), *(nn.Embedding(5, 4) for _ in range(model.n_vq))]
    )
    with torch.no_grad():
        for table_index, embedding in enumerate(model.embedding_list):
            values = torch.arange(embedding.weight.numel(), dtype=torch.float32)
            embedding.weight.copy_(values.reshape_as(embedding.weight) + table_index)
    model.local_transformer = _RecordingLocalTransformer()
    model.local_text_lm_head = nn.Linear(4, 2, bias=False)
    return model


def test_eager_frame_decode_feeds_text_choice_before_audio_codebooks() -> None:
    with _nano_sglang_model_class() as model_cls:
        model = _make_model(model_cls)
        hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

        stop_choice, codes = model.decode_frame(
            hidden_states,
            sample_text=lambda logits: torch.tensor([0, 1]),
            sample_audio=lambda logits, channel: torch.full(
                (2,), channel + 1, dtype=torch.long
            ),
        )

    calls = model.local_transformer.calls
    assert [position for position, _ in calls] == [0, 1, 2, 3]
    torch.testing.assert_close(calls[0][1], hidden_states)
    torch.testing.assert_close(
        calls[1][1],
        model.embedding_list[0](torch.tensor([9, 7])),
    )
    torch.testing.assert_close(
        calls[2][1],
        model.embedding_list[1](torch.tensor([1, 1])),
    )
    torch.testing.assert_close(
        calls[3][1],
        model.embedding_list[2](torch.tensor([2, 2])),
    )
    assert stop_choice.tolist() == [0, 1]
    assert codes.tolist() == [[1, 2, 3], [1, 2, 3]]


def test_graphable_frame_decode_uses_the_same_local_step_sequence() -> None:
    with _nano_sglang_model_class() as model_cls:
        model = _make_model(model_cls)
        hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        audio_call = 0

        def sample(logits: torch.Tensor, **kwargs) -> torch.Tensor:
            nonlocal audio_call
            del kwargs
            if int(logits.shape[-1]) == 2:
                return torch.tensor([0, 1])
            audio_call += 1
            return torch.full((2,), audio_call, dtype=torch.long)

        model._sample_seeded_branchless = sample
        stop_choice, codes, feedback = model._decode_frame_graphable(
            hidden_states,
            text_temperature=torch.ones(2),
            text_top_p=torch.ones(2),
            text_top_k=torch.ones(2, dtype=torch.long),
            audio_temperature=torch.ones(2),
            audio_top_p=torch.ones(2),
            audio_top_k=torch.ones(2, dtype=torch.long),
            seeds=torch.tensor([11, 12]),
            base_positions=torch.tensor([0, 4]),
        )

    calls = model.local_transformer.calls
    assert [position for position, _ in calls] == [0, 1, 2, 3]
    torch.testing.assert_close(
        calls[1][1],
        model.embedding_list[0](torch.tensor([9, 7])),
    )
    assert stop_choice.tolist() == [0, 1]
    assert codes.tolist() == [[1, 2, 3], [1, 2, 3]]
    expected_feedback = model.embedding_list[0](torch.tensor([9, 9]))
    for channel, code in enumerate((1, 2, 3)):
        expected_feedback = expected_feedback + model.embedding_list[channel + 1](
            torch.tensor([code, code])
        )
    torch.testing.assert_close(feedback, expected_feedback)
