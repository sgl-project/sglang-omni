# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the eager Qwen3-TTS Torch MPS path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.qwen3_omni.pending_text_queue import PendingTextTensorQueue
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.qwen3_tts.torch_mps_runner import (
    Qwen3TTSTorchMpsModelRunner,
    _materialize_rotary_buffers,
)
from sglang_omni.models.qwen3_tts.torch_mps_vocoder import (
    create_torch_mps_vocoder_scheduler,
)


def _logits(token_id: int) -> torch.Tensor:
    logits = torch.zeros((1, 1, 1030))
    logits[..., token_id] = 9.0
    return logits


class _FakeTalker:
    dtype = torch.float32

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if "inputs_embeds" in kwargs:
            return SimpleNamespace(
                logits=_logits(2),
                past_key_values="prefill-cache",
                past_hidden=torch.ones((1, 1, 4)),
                generation_step=0,
            )
        semantic = int(kwargs["input_ids"].item())
        return SimpleNamespace(
            logits=_logits(3),
            past_key_values=f"cache-{semantic}",
            past_hidden=torch.full((1, 1, 4), float(semantic)),
            generation_step=int(kwargs["generation_step"]) + 1,
            hidden_states=(
                None,
                torch.tensor([[semantic, semantic + 10, semantic + 20]]),
            ),
        )


def _runner() -> Qwen3TTSTorchMpsModelRunner:
    runner = Qwen3TTSTorchMpsModelRunner.__new__(Qwen3TTSTorchMpsModelRunner)
    runner.device = torch.device("cpu")
    runner.model = SimpleNamespace(
        torch_mps_talker=_FakeTalker(),
        config=SimpleNamespace(vocab_size=1030, codec_eos_token_id=1029),
    )
    runner._request_states = {}
    return runner


def _request() -> SimpleNamespace:
    sampling = SimpleNamespace(
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    req = SimpleNamespace(rid="r0", output_ids=[], sampling_params=sampling)
    data = SimpleNamespace(
        req=req,
        prompt_input_embeds=torch.ones((2, 4)),
        pending_text_queue=PendingTextTensorQueue.from_tensor(torch.ones((2, 4))),
        tts_pad_embed=torch.zeros(4),
        semantic_sampling_seed=11,
        subtalker_sampling_seed=12,
        subtalker_dosample=False,
        subtalker_top_p=1.0,
        subtalker_top_k=50,
        subtalker_temperature=0.9,
        output_codes=[],
    )
    return SimpleNamespace(request_id="r0", data=data)


def test_torch_mps_runner_uses_upstream_talker_for_complete_frames() -> None:
    runner = _runner()
    request = _request()

    prefill = runner.custom_prefill_forward(None, None, [request])
    request.data.req.output_ids.append(int(prefill.next_token_ids.item()))
    decode = runner.custom_decode_forward(None, None, [request])

    assert prefill.next_token_ids.tolist() == [2]
    assert decode.next_token_ids.tolist() == [3]
    assert [codes.tolist() for codes in request.data.output_codes] == [
        [2, 12, 22],
        [3, 13, 23],
    ]
    talker = runner.talker
    assert len(talker.calls) == 3
    assert talker.calls[1]["past_key_values"] == "prefill-cache"
    assert talker.calls[2]["past_key_values"] == "cache-2"
    assert talker.calls[1]["cache_position"].tolist() == [2]
    assert talker.calls[2]["cache_position"].tolist() == [3]


def test_torch_mps_runner_does_not_generate_a_frame_for_eos() -> None:
    runner = _runner()
    request = _request()
    runner.talker.calls.clear()
    original = runner.talker

    def eos_prefill(**kwargs):
        original.calls.append(kwargs)
        return SimpleNamespace(
            logits=_logits(1029),
            past_key_values="cache",
            past_hidden=torch.ones((1, 1, 4)),
            generation_step=0,
        )

    # Special methods are resolved on the class, so use a minimal callable type.
    runner.model.torch_mps_talker = type(
        "EosTalker",
        (),
        {"dtype": torch.float32, "__call__": staticmethod(eos_prefill)},
    )()

    result = runner.custom_prefill_forward(None, None, [request])

    assert result.next_token_ids.tolist() == [1029]
    assert request.data.output_codes == []


def test_torch_mps_runner_drives_the_upstream_talker_forward() -> None:
    from sglang_omni.models.qwen3_tts.compat import (
        apply_qwen_tts_transformers_compatibility_patches,
    )

    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    predictor = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "num_code_groups": 3,
        "pad_token_id": 0,
    }
    config = Qwen3TTSTalkerConfig(
        code_predictor_config=predictor,
        vocab_size=1030,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        text_hidden_size=8,
        text_vocab_size=40,
        num_code_groups=3,
        codec_eos_token_id=1029,
        codec_pad_id=1028,
        codec_bos_id=1027,
        pad_token_id=1028,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [1, 1],
            "interleaved": True,
        },
    )
    runner = _runner()
    talker = Qwen3TTSTalkerForConditionalGeneration(config).eval()
    with torch.no_grad():
        talker.codec_head.weight.zero_()
    runner.model.torch_mps_talker = talker
    request = _request()
    request.data.prompt_input_embeds = torch.ones((2, 8))
    request.data.pending_text_queue = PendingTextTensorQueue.from_tensor(
        torch.ones((2, 8))
    )
    request.data.tts_pad_embed = torch.zeros(8)

    result = runner.custom_prefill_forward(None, None, [request])

    assert result.next_token_ids.shape == (1,)
    assert len(request.data.output_codes) == 1
    assert request.data.output_codes[0].shape == (3,)
    state = runner._request_states["r0"]
    assert state.generation_step == 1
    assert state.attention_mask.shape == (1, 3)


def test_torch_mps_talker_materializes_meta_rotary_buffers() -> None:
    from sglang_omni.models.qwen3_tts.compat import (
        apply_qwen_tts_transformers_compatibility_patches,
    )

    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    predictor = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "num_code_groups": 3,
        "pad_token_id": 0,
    }
    config = Qwen3TTSTalkerConfig(
        code_predictor_config=predictor,
        vocab_size=1030,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        text_hidden_size=8,
        text_vocab_size=40,
        num_code_groups=3,
        codec_eos_token_id=1029,
        codec_pad_id=1028,
        codec_bos_id=1027,
        pad_token_id=1028,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [1, 1],
            "interleaved": True,
        },
    )
    with torch.device("meta"):
        talker = Qwen3TTSTalkerForConditionalGeneration(config)

    assert [name for name, value in talker.named_buffers() if value.is_meta]
    _materialize_rotary_buffers(talker)

    assert not [name for name, value in talker.named_buffers() if value.is_meta]
    rotary_modules = [
        module for module in talker.modules() if hasattr(module, "inv_freq")
    ]
    assert len(rotary_modules) == 2
    assert all(not module.original_inv_freq.is_meta for module in rotary_modules)


class _FakeTokenizer:
    def decode(self, items):
        assert items[0]["audio_codes"].tolist() == [[1, 2], [3, 4]]
        return [torch.arange(8, dtype=torch.float32)], 24000


def _payload(*, stream: bool = False):
    from sglang_omni.proto import OmniRequest, StagePayload

    return StagePayload(
        request_id="r0",
        request=OmniRequest(inputs="hello", params={"stream": stream}),
        data=Qwen3TTSState(
            audio_codes=torch.tensor([[1, 2], [3, 4]]),
            ref_code_len=1,
        ).to_dict(),
    )


def test_torch_mps_vocoder_decodes_one_final_payload() -> None:
    scheduler = create_torch_mps_vocoder_scheduler(_FakeTokenizer())

    result = scheduler._fn(_payload())

    assert result.data["sample_rate"] == 24000
    assert result.data["modality"] == "audio"
    waveform = np.frombuffer(result.data["audio_waveform"], dtype=np.float32)
    assert waveform.tolist() == [4.0, 5.0, 6.0, 7.0]


def test_torch_mps_vocoder_rejects_streaming() -> None:
    scheduler = create_torch_mps_vocoder_scheduler(_FakeTokenizer())

    with pytest.raises(ValueError, match="non-streaming only"):
        scheduler._fn(_payload(stream=True))


def test_vocoder_factory_selects_torch_mps_compatibility_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.utils import tensor_bridge

    from sglang_omni.models.qwen3_tts import stages, torch_mps_vocoder

    tokenizer = object()
    scheduler = object()
    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: False)
    monkeypatch.setattr(stages, "resolve_device_spec", lambda device, gpu_id: "mps")
    monkeypatch.setattr(
        stages, "_load_qwen3_tts_tokenizer", lambda *args, **kwargs: tokenizer
    )
    monkeypatch.setattr(
        torch_mps_vocoder,
        "create_torch_mps_vocoder_scheduler",
        lambda loaded: scheduler if loaded is tokenizer else None,
    )

    assert stages.create_vocoder_executor("model") is scheduler
