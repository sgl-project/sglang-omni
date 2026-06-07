# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for MOSS-TTS Local (no checkpoint / GPU required)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner
from sglang_omni.models.moss_tts_local.request_builders import _LOCAL_SAMPLING
from sglang_omni.models.moss_tts_local.sglang_model import (
    MossTTSLocalSGLangModel,
    _DepthTransformer,
)
from sglang_omni.scheduling.types import RequestOutput


def test_registry_resolves_local_config_without_collision():
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    cls = PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name("MossTTSLocalPipelineConfig")
    assert cls.__name__ == "MossTTSLocalPipelineConfig"
    assert cls.architecture == "MossTTSLocalModel"
    archs = PIPELINE_CONFIG_REGISTRY.get_supported_archs()
    # Distinct from Delay; both coexist (no duplicate-arch ValueError on import).
    assert "MossTTSLocalModel" in archs
    assert "MossTTSDelayModel" in archs


def test_pipeline_stages_wired():
    from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig

    cfg = MossTTSLocalPipelineConfig(model_path="dummy/path")
    names = [s.name for s in cfg.stages]
    assert names == ["preprocessing", "tts_engine", "vocoder"]
    # Preprocessing reuses the shared Delay factory; engine/vocoder are Local.
    factories = {s.name: s.factory for s in cfg.stages}
    assert factories["preprocessing"].startswith("sglang_omni.models.moss_tts.stages")
    assert "moss_tts_local.stages" in factories["tts_engine"]
    assert "moss_tts_local.stages" in factories["vocoder"]


def test_depth_transformer_cache_equivalent_to_eager():
    """Incremental depth KV-cache must match full-prefix recompute (no positions)."""
    torch.manual_seed(0)
    B, LH, FFN, NH, NKV, HD, NL, CH = 2, 64, 128, 4, 2, 16, 3, 8
    dep = _DepthTransformer(NL, LH, FFN, NH, NKV, HD, 1e-6).eval()
    for p in dep.parameters():
        torch.nn.init.normal_(p, std=0.05)
    toks = [torch.randn(B, LH) for _ in range(CH)]

    @torch.no_grad()
    def eager():
        outs = []
        for k in range(CH):
            outs.append(dep(torch.stack(toks[: k + 1], dim=1))[:, -1, :])
        return torch.stack(outs, dim=1)

    scaling = HD**-0.5
    pk = torch.zeros(NL, B, NKV, CH, HD)
    pv = torch.zeros_like(pk)
    import torch.nn.functional as F

    @torch.no_grad()
    def cached():
        outs = []
        for k in range(CH):
            x = toks[k]
            for li in range(NL):
                layer = dep.layers[li]
                a = layer.self_attn
                normed = layer.input_layernorm(x)
                q = a.q_norm(a.q_proj(normed).view(B, NH, HD))
                kk = a.k_norm(a.k_proj(normed).view(B, NKV, HD))
                vv = a.v_proj(normed).view(B, NKV, HD)
                pk[li, :, :, k, :] = kk
                pv[li, :, :, k, :] = vv
                keys = pk[li, :, :, : k + 1, :].repeat_interleave(NH // NKV, 1)
                vals = pv[li, :, :, : k + 1, :].repeat_interleave(NH // NKV, 1)
                o = F.scaled_dot_product_attention(
                    q.unsqueeze(2), keys, vals, scale=scaling
                )
                x = x + a.o_proj(o.transpose(1, 2).reshape(B, NH * HD))
                x = x + layer.mlp(layer.post_attention_layernorm(x))
            outs.append(dep.norm(x))
        return torch.stack(outs, dim=1)

    assert torch.allclose(eager(), cached(), atol=1e-4)


def _fake_data(pen, prompt_rows, **over):
    base = dict(
        text_temperature=0.0,
        text_top_p=1.0,
        text_top_k=-1,
        audio_temperature=0.0,
        audio_top_p=1.0,
        audio_top_k=-1,
        sampling_seed=0,
        generation_steps=0,
        audio_repetition_penalty=pen,
        prompt_rows=prompt_rows,
        output_rows=[],
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_audio_repetition_penalty_demotes_seen_codes():
    """A heavily-seen audio code should be demoted (greedy argmax flips)."""
    r = MossTTSLocalModelRunner.__new__(MossTTSLocalModelRunner)
    r.model = SimpleNamespace(
        device=torch.device("cpu"), channels=33, audio_pad_code=1024
    )
    prompt = torch.full((4, 33), 1024, dtype=torch.long)
    prompt[:, 1] = 5  # token 5 seen repeatedly in audio channel 1

    def argmax_for(pen):
        sampler = r._make_sampler([_fake_data(pen, prompt)])
        logits = torch.zeros(1, 1025)
        logits[0, 5] = 10.0  # 5 is the raw max
        logits[0, 3] = 6.0
        return int(sampler(1, logits)[0])

    assert argmax_for(1.0) == 5  # no penalty: max stays
    assert argmax_for(2.0) == 3  # penalty: 10/2=5 < 6 -> flips to 3


def test_rep_penalty_skips_text_channel():
    """Channel 0 (text) must never get the audio repetition penalty."""
    r = MossTTSLocalModelRunner.__new__(MossTTSLocalModelRunner)
    r.model = SimpleNamespace(
        device=torch.device("cpu"), channels=33, audio_pad_code=1024
    )
    prompt = torch.full((4, 33), 1024, dtype=torch.long)
    prompt[:, 0] = 5
    sampler = r._make_sampler([_fake_data(2.0, prompt)])
    logits = torch.zeros(1, 200)
    logits[0, 5] = 10.0
    logits[0, 3] = 6.0
    assert int(sampler(0, logits)[0]) == 5  # text channel unpenalized


def test_local_sampling_defaults():
    # Upstream MossTTSLocal generate() defaults (calmer audio sampling than Delay).
    assert _LOCAL_SAMPLING["audio_temperature"] == 1.0
    assert _LOCAL_SAMPLING["audio_top_p"] == 0.95
    assert _LOCAL_SAMPLING["audio_top_k"] == 50
    assert _LOCAL_SAMPLING["audio_repetition_penalty"] == 1.1
    assert _LOCAL_SAMPLING["text_temperature"] == 1.5


def test_result_adapter_drops_text_channel():
    """apply_*_result keeps only channels 1.. (RVQ codes), dropping channel 0 (text)."""
    from sglang_omni.models.moss_tts.payload_types import MossTTSState
    from sglang_omni.models.moss_tts_local.request_builders import (
        apply_sglang_moss_tts_local_result,
    )

    state = MossTTSState()
    rows = [torch.arange(33, dtype=torch.long) + 100 * t for t in range(4)]  # (4, 33)
    data = SimpleNamespace(
        state=state,
        output_rows=rows,
        prompt_rows=None,
        input_ids=[1, 2, 3],
        engine_start_s=0.0,
    )
    payload = SimpleNamespace(request_id="r", request=None)
    apply_sglang_moss_tts_local_result(payload, data)

    codes = state.delayed_audio_codes
    assert tuple(codes.shape) == (4, 32)  # channel 0 dropped
    assert torch.equal(codes, torch.stack(rows, dim=0)[:, 1:])


def test_post_process_skips_audio_end():
    """A frame whose channel-0 token == audio_end must not be appended (EOS)."""
    runner = MossTTSLocalModelRunner.__new__(MossTTSLocalModelRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(audio_end_token_id=14))
    runner._pending_rows = torch.tensor([[12, 2, 4], [14, 4, 4]], dtype=torch.long)
    runner._pending_embeds = torch.ones((2, 3))
    requests = [
        SimpleNamespace(
            request_id="active",
            data=SimpleNamespace(output_rows=[], pending_feedback_queue=[]),
        ),
        SimpleNamespace(
            request_id="eos",
            data=SimpleNamespace(output_rows=[], pending_feedback_queue=[]),
        ),
    ]
    runner.post_process_outputs(
        object(),
        SimpleNamespace(requests=requests),
        {
            "active": RequestOutput("active", data=12),
            "eos": RequestOutput("eos", data=14),
        },
    )
    assert [row.tolist() for row in requests[0].data.output_rows] == [[12, 2, 4]]
    assert len(requests[0].data.pending_feedback_queue) == 1
    assert requests[1].data.output_rows == []


def test_normalize_config_fills_local_defaults():
    cfg = SimpleNamespace(
        language_config=SimpleNamespace(hidden_size=2048, vocab_size=155648),
    )
    out = MossTTSLocalSGLangModel._normalize_config(cfg)
    assert out.n_vq == 32 and out.channels == 33
    assert out.local_num_layers == 4 and out.local_hidden_size == 1536
    assert out.local_ffn_hidden_size == 8960
    assert len(out.vocab_size_list) == 33
    assert out.vocab_size_list[0] == 155648 and out.vocab_size_list[1] == 1025
