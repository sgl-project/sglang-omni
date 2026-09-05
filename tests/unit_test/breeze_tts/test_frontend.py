# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.breeze_tts.frontend import BreezeFrontend
from sglang_omni.proto import OmniRequest, StagePayload


class Tokenizer:
    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert add_special_tokens
        return {"input_ids": torch.tensor([[2] + [ord(c) % 50 + 3 for c in text]])}


class AudioTokenizer:
    def get_input_sample_rate(self):
        return 24000

    def encode(self, audio, *, sr):
        assert sr == 24000
        code = int(audio[0])
        return SimpleNamespace(audio_codes=[torch.full((2, 4), code)])


def make_payload(reference="reference.wav", cfg_scale=4):
    return StagePayload(
        request_id="frontend",
        data={},
        request=OmniRequest(
            inputs="Hello",
            metadata={
                "tts_params": {
                    "instructions": "Whisper",
                    "cfg_scale": cfg_scale,
                    "ref_audio": reference,
                    "ref_text": "Reference" if reference else None,
                }
            },
        ),
    )


def test_reference_eos_embedding_and_cfg_segment_isolation(tiny_config, monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.utils.audio.load_audio", lambda *a, **k: np.array([1.0])
    )
    frontend = BreezeFrontend(tiny_config).eval()
    tokenizer = Tokenizer()
    output = frontend.prepare(make_payload(), tokenizer, AudioTokenizer()).data
    ref_text = frontend.encode_text(tokenizer, "[S0]Reference")
    ref_codes = frontend.embed_audio(torch.tensor([[1] * 4, [1] * 4, [0] * 4]))
    target = frontend.encode_text(tokenizer, "[S0]<ins_bos>Whisper<ins_eos>Hello")
    plain = frontend.encode_text(tokenizer, "[S0]Hello")
    torch.testing.assert_close(
        output["prompt_embeds"], torch.cat((ref_text, ref_codes, target))
    )
    torch.testing.assert_close(
        output["negative_embeds"], torch.cat((ref_text, ref_codes, plain))
    )
    assert not output["prompt_embeds"].requires_grad
    assert not output["negative_embeds"].requires_grad
    # Different reference codes cannot be reused from another request, and
    # encoding a reference must not contaminate target-text attention.
    monkeypatch.setattr(
        "sglang_omni.utils.audio.load_audio", lambda *a, **k: np.array([2.0])
    )
    other = frontend.prepare(make_payload(), tokenizer, AudioTokenizer()).data
    assert not torch.equal(other["prompt_embeds"], output["prompt_embeds"])
    torch.testing.assert_close(other["prompt_embeds"][-len(target) :], target)


def test_design_never_calls_reference_encoder(tiny_config):
    frontend = BreezeFrontend(tiny_config).eval()
    out = frontend.prepare(make_payload(None, 1), Tokenizer(), None).data
    assert out["prompt_embeds"] is out["negative_embeds"]


def test_long_prompt_is_rejected_not_silently_truncated(tiny_config):
    frontend = BreezeFrontend(tiny_config).eval()
    payload = make_payload(None)
    payload.request.inputs = "x" * 1100
    with pytest.raises(ValueError, match="context"):
        frontend.prepare(payload, Tokenizer(), None)


def test_generation_limit_accounts_for_both_cfg_branch_lengths(tiny_config):
    frontend = BreezeFrontend(tiny_config).eval()
    payload = make_payload(None)
    payload.request.inputs = "x" * 900
    output = frontend.prepare(payload, Tokenizer(), None).data
    assert output["sampling"]["max_new_tokens"] == 1024 - max(
        len(output["prompt_embeds"]), len(output["negative_embeds"])
    )
