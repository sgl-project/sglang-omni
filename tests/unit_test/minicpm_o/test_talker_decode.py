# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MiniCPM-o talker fast decode loop (GPU-free).

The fast path's CUDA graph cannot run on CPU; these tests pin
SGLANG_OMNI_MINICPMO_TALKER_GRAPH=0 so ``mode="fast"`` exercises the
StaticCache + explicit-mask eager loop, and compare it against the
``compat`` loop (op-for-op the remote generate) on a tiny real LlamaModel.
"""

import types

import pytest
import torch
import torch.nn as nn

from sglang_omni.models.minicpm_o.components.talker_decode import (
    MINICPMO_TALKER_GRAPH_ENV,
    TalkerDecodeLoop,
    _SamplingState,
)

NUM_AUDIO = 32
EOS = NUM_AUDIO - 1
HIDDEN = 32


def _empty_gen_logits(**_kwargs):
    return [], []


def _sampling_params(temperature=0.05):
    return types.SimpleNamespace(
        temperature=temperature, top_p=0.85, top_k=25, repetition_penalty=1.05
    )


class _TinyTTS(nn.Module):
    """Minimal stand-in for the remote MiniCPMTTS module."""

    def __init__(self):
        super().__init__()
        modeling_llama = pytest.importorskip(
            "transformers.models.llama.modeling_llama",
            reason="local transformers install cannot import LlamaModel",
            exc_type=ImportError,
        )
        from transformers import LlamaConfig

        LlamaModel = modeling_llama.LlamaModel
        llama_config = LlamaConfig(
            hidden_size=HIDDEN,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=128,
            vocab_size=8,
            attn_implementation="eager",
        )
        self.model = LlamaModel(llama_config)
        self.emb_code = nn.ModuleList([nn.Embedding(NUM_AUDIO, HIDDEN)])
        self.head_code = nn.ModuleList([nn.Linear(HIDDEN, NUM_AUDIO, bias=False)])
        self.config = types.SimpleNamespace(
            num_vq=1,
            num_audio_tokens=NUM_AUDIO,
            max_position_embeddings=128,
        )
        self.eval()


@pytest.fixture(autouse=True)
def _no_graph(monkeypatch):
    monkeypatch.setenv(MINICPMO_TALKER_GRAPH_ENV, "0")


def _generate(loop, tts, *, mode, seed, max_new_token=24, min_new_token=0):
    torch.manual_seed(seed)
    cond = torch.randn(1, 5, HIDDEN)
    torch.manual_seed(seed + 1)
    return loop.generate(
        cond,
        torch.tensor([EOS], dtype=torch.long),
        min_new_token=min_new_token,
        max_new_token=max_new_token,
        sampling_params=_sampling_params(),
        mode=mode,
    )


def test_fast_matches_compat_on_tiny_llama():
    torch.manual_seed(0)
    tts = _TinyTTS()
    loop = TalkerDecodeLoop(tts, gen_logits_fn=_empty_gen_logits)
    for seed in (7, 13):
        compat = _generate(loop, tts, mode="compat", seed=seed)
        fast = _generate(loop, tts, mode="fast", seed=seed)
        assert compat.shape == fast.shape
        assert torch.equal(compat, fast)


def test_fast_backend_reuse_is_stateless_across_requests():
    # Request B runs on a backend that already served request A (different
    # condition). Any state leak — stale KV attended through the mask, or an
    # unreset StaticCache write position — makes B diverge from a fresh
    # compat run of B.
    torch.manual_seed(0)
    tts = _TinyTTS()
    loop = TalkerDecodeLoop(tts, gen_logits_fn=_empty_gen_logits)
    _generate(loop, tts, mode="fast", seed=7)
    backend = loop._fast
    assert backend is not None
    fast_b = _generate(loop, tts, mode="fast", seed=21)
    assert loop._fast is backend
    compat_b = _generate(loop, tts, mode="compat", seed=21)
    assert torch.equal(fast_b, compat_b)


class _ScriptedModel(nn.Module):
    """Backbone stub emitting a scripted hidden state per decode step."""

    def __init__(self, eos_at_step):
        super().__init__()
        self.eos_at_step = eos_at_step
        self.calls = 0

    def forward(self, *, inputs_embeds, position_ids, **_kwargs):
        step = self.calls
        self.calls += 1
        hidden = torch.zeros(1, inputs_embeds.shape[1], NUM_AUDIO)
        token = EOS if step >= self.eos_at_step else (step % (NUM_AUDIO - 1))
        hidden[:, -1, token] = 50.0
        return types.SimpleNamespace(last_hidden_state=hidden, past_key_values=None)


def _scripted_tts(eos_at_step):
    tts = nn.Module()
    tts.model = _ScriptedModel(eos_at_step)
    tts.emb_code = nn.ModuleList([nn.Embedding(NUM_AUDIO, NUM_AUDIO)])
    tts.head_code = nn.ModuleList([nn.Identity()])
    tts.config = types.SimpleNamespace(
        num_vq=1, num_audio_tokens=NUM_AUDIO, max_position_embeddings=128
    )
    return tts


def test_eos_stops_and_is_excluded():
    tts = _scripted_tts(eos_at_step=4)
    loop = TalkerDecodeLoop(tts, gen_logits_fn=_empty_gen_logits)
    out = loop.generate(
        torch.zeros(1, 3, NUM_AUDIO),
        torch.tensor([EOS], dtype=torch.long),
        min_new_token=0,
        max_new_token=24,
        sampling_params=_sampling_params(temperature=0.01),
        mode="compat",
    )
    # steps 0..3 emit non-EOS tokens, step 4 emits EOS and is excluded.
    assert out.shape == (1, 4, 1)
    assert EOS not in out.view(-1).tolist()


def test_min_new_token_blocks_early_eos():
    tts = _scripted_tts(eos_at_step=0)
    loop = TalkerDecodeLoop(tts, gen_logits_fn=_empty_gen_logits)
    out = loop.generate(
        torch.zeros(1, 3, NUM_AUDIO),
        torch.tensor([EOS], dtype=torch.long),
        min_new_token=6,
        max_new_token=24,
        sampling_params=_sampling_params(temperature=0.01),
        mode="compat",
    )
    # EOS is masked for the first 6 steps, fires at step 6 and is excluded.
    assert out.shape == (1, 6, 1)
    assert EOS not in out.view(-1).tolist()


def test_sampling_state_passes_windowed_history():
    seen = []

    def _spy_processor(logits_token, logits):
        seen.append(logits_token.shape)
        return logits

    def _gen_logits(**_kwargs):
        return [], [_spy_processor]

    sampling = _SamplingState(
        num_vq=1,
        num_audio_tokens=NUM_AUDIO,
        temperature=1.0,
        gen_logits_fn=_gen_logits,
        repetition_penalty=1.05,
        top_p=0.85,
        top_k=25,
        device=torch.device("cpu"),
    )
    new_tokens = torch.zeros(1, 10, 1, dtype=torch.long)
    logits = torch.randn(1, NUM_AUDIO, 1)
    sampling.sample(
        logits,
        step=0,
        new_tokens=new_tokens,
        min_new_token=0,
        eos_token=torch.tensor([EOS]),
    )
    sampling.sample(
        logits,
        step=3,
        new_tokens=new_tokens,
        min_new_token=0,
        eos_token=torch.tensor([EOS]),
    )
    # step 0 skips processors entirely; step 3 passes the (num_vq, t) history.
    assert seen == [(1, 3)]
