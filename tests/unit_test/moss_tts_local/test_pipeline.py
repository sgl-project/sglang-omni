# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from sglang_omni.client.audio import encode_audio, encode_wav
from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.models.moss_tts_local.config import (
    MossTTSLocalColocatedPipelineConfig,
    MossTTSLocalPipelineConfig,
)
from sglang_omni.models.moss_tts_local.local_transformer import (
    MossTTSLocalTransformer,
    _rotate_half_interleaved,
)
from sglang_omni.models.moss_tts_local.payload_types import (
    MossTTSLocalState,
    moss_tts_local_special_token_defaults,
)
from sglang_omni.models.moss_tts_local.request_builders import (
    MossTTSLocalSGLangRequestData,
    apply_sglang_moss_tts_local_result,
    build_generation_kwargs,
    build_moss_tts_local_state,
    clear_moss_tts_local_preprocessing_context,
    preprocess_moss_tts_local_payload,
    set_moss_tts_local_preprocessing_context,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.utils.audio_payload import audio_waveform_payload

N_VQ = 12


# ---------------------------------------------------------------------------
# Local transformer numerics
# ---------------------------------------------------------------------------


def _hf_rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Verbatim port of the upstream gpt2_decoder.rotate_half."""
    even = hidden_states[..., ::2]
    odd = hidden_states[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(hidden_states)


def _reference_full_forward(
    module: MossTTSLocalTransformer, inputs: torch.Tensor
) -> torch.Tensor:
    """Full-sequence forward replicating the upstream eager math.

    ``inputs`` is ``[batch, seq, hidden]``; positions are 0..seq-1 with a
    causal mask, interleaved RoPE, fp32 softmax via explicit matmuls.
    """
    batch, seq, hidden = inputs.shape
    num_heads = module.num_heads
    head_dim = module.head_dim

    inv_freq = 1.0 / (
        1_000_000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)

    x = inputs
    for block in module.h:
        normed = block.ln_1(x)
        qkv = block.attn.c_attn(normed)
        query, key, value = qkv.split(hidden, dim=-1)
        query = query.view(batch, seq, num_heads, head_dim)
        key = key.view(batch, seq, num_heads, head_dim)
        value = value.view(batch, seq, num_heads, head_dim)
        cos_b = cos.view(1, seq, 1, head_dim)
        sin_b = sin.view(1, seq, 1, head_dim)
        query = query * cos_b + _hf_rotate_half(query) * sin_b
        key = key * cos_b + _hf_rotate_half(key) * sin_b

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / head_dim**0.5
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, value).transpose(1, 2).reshape(batch, seq, hidden)
        x = x + block.attn.c_proj(attn)
        x = x + block.mlp(block.ln_2(x))
    return module.ln_f(x)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_local_transformer_incremental_matches_full_recompute(num_layers: int):
    torch.manual_seed(0)
    module = MossTTSLocalTransformer(
        hidden_size=64,
        num_heads=4,
        inner_size=96,
        num_layers=num_layers,
        max_positions=N_VQ + 1,
        rope_base=1_000_000.0,
    )
    module.eval()
    batch, seq = 3, N_VQ + 1
    inputs = torch.randn(batch, seq, 64)

    reference = _reference_full_forward(module, inputs)
    stepped = torch.stack([module.step(inputs[:, t], t) for t in range(seq)], dim=1)
    torch.testing.assert_close(stepped, reference, rtol=1e-4, atol=1e-5)


def test_local_transformer_kv_cache_grows_with_batch():
    module = MossTTSLocalTransformer(
        hidden_size=32,
        num_heads=2,
        inner_size=48,
        num_layers=1,
        max_positions=N_VQ + 1,
        rope_base=1_000_000.0,
    )
    out_small = module.step(torch.randn(2, 32), 0)
    assert out_small.shape == (2, 32)
    out_large = module.step(torch.randn(8, 32), 0)
    assert out_large.shape == (8, 32)
    assert module._kv_capacity >= 8


def test_local_transformer_rejects_out_of_range_position():
    module = MossTTSLocalTransformer(
        hidden_size=32,
        num_heads=2,
        inner_size=48,
        num_layers=1,
        max_positions=N_VQ + 1,
        rope_base=1_000_000.0,
    )
    with pytest.raises(ValueError):
        module.step(torch.randn(1, 32), N_VQ + 1)


def test_rotate_half_interleaved_matches_upstream():
    x = torch.randn(5, 4, 8)
    torch.testing.assert_close(_rotate_half_interleaved(x), _hf_rotate_half(x))


# ---------------------------------------------------------------------------
# Registry / config
# ---------------------------------------------------------------------------


def test_registry_resolves_local_architecture():
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config("MossTTSLocalModel")
    assert config_cls is MossTTSLocalPipelineConfig
    # The Delay family keeps its own architecture.
    delay_cls = PIPELINE_CONFIG_REGISTRY.get_config("MossTTSDelayModel")
    assert delay_cls is not MossTTSLocalPipelineConfig


def test_pipeline_stage_wiring():
    config = MossTTSLocalPipelineConfig(model_path="OpenMOSS-Team/moss-local-test")
    stages = {stage.name: stage for stage in config.stages}
    assert set(stages) == {"preprocessing", "tts_engine", "vocoder"}
    assert stages["preprocessing"].next == "tts_engine"
    assert stages["tts_engine"].next == "vocoder"
    assert stages["vocoder"].terminal
    for stage in stages.values():
        assert "moss_tts_local" in stage.factory
    assert stages["preprocessing"].process == "pipeline"
    assert stages["preprocessing"].gpu == 0
    assert stages["preprocessing"].factory_args["device"] == "cuda:1"
    assert stages["tts_engine"].process == "pipeline"
    assert stages["tts_engine"].gpu == 0
    assert stages["vocoder"].process == "pipeline"
    assert stages["vocoder"].gpu == 0
    assert stages["vocoder"].factory_args["device"] == "cuda:1"

    placement = build_stage_placement_plan(config)
    assert placement.stages["tts_engine"].gpu_ids == (0,)
    assert placement.stages["preprocessing"].gpu_ids == (0,)
    assert placement.stages["vocoder"].gpu_ids == (0,)

    colocated = MossTTSLocalColocatedPipelineConfig(
        model_path="OpenMOSS-Team/moss-local-test"
    )
    colocated_stages = {stage.name: stage for stage in colocated.stages}
    assert colocated_stages["preprocessing"].factory_args["device"] == "cuda:0"
    assert colocated_stages["vocoder"].factory_args["device"] == "cuda:0"


def test_special_token_defaults_match_v15_checkpoint():
    defaults = dict(moss_tts_local_special_token_defaults())
    assert defaults["audio_start_token_id"] == 151669
    assert defaults["audio_end_token_id"] == 151670
    assert defaults["audio_user_slot_token_id"] == 151654
    assert defaults["audio_assistant_slot_token_id"] == 151656
    assert defaults["audio_pad_code"] == 1024


# ---------------------------------------------------------------------------
# Generation kwargs / state
# ---------------------------------------------------------------------------


def test_build_generation_kwargs_defaults():
    kwargs = build_generation_kwargs({}, tts_params={})
    assert kwargs["max_new_tokens"] == 4096
    assert kwargs["text_temperature"] == 1.0
    assert kwargs["text_top_p"] == 1.0
    assert kwargs["text_top_k"] == 50
    assert kwargs["audio_temperature"] == 1.7
    assert kwargs["audio_top_p"] == 0.8
    assert kwargs["audio_top_k"] == 25
    assert kwargs["audio_repetition_penalty"] == 1.0


def test_build_generation_kwargs_explicit_overrides():
    kwargs = build_generation_kwargs(
        {"temperature": 0.9, "top_p": 0.7},
        tts_params={
            "explicit_generation_params": ["temperature", "top_p"],
            "audio_top_k": 11,
        },
    )
    assert kwargs["text_temperature"] == 0.9
    assert kwargs["audio_temperature"] == 0.9
    assert kwargs["audio_top_p"] == 0.7
    assert kwargs["audio_top_k"] == 11


def test_state_round_trip():
    state = MossTTSLocalState(
        text="hello",
        language="English",
        token_count=125,
        audio_codes=torch.zeros((3, N_VQ), dtype=torch.long),
        sample_rate=48000,
        prompt_tokens=7,
        completion_tokens=3,
    )
    restored = MossTTSLocalState.from_dict(state.to_dict())
    assert restored.text == "hello"
    assert restored.language == "English"
    assert restored.token_count == 125
    assert restored.sample_rate == 48000
    assert torch.as_tensor(restored.audio_codes).shape == (3, N_VQ)


def test_build_state_token_count_and_language():
    payload = StagePayload(
        request_id="r0",
        request=OmniRequest(
            inputs={"text": "${token:50} hello world", "references": []},
            params={"language": "English"},
            metadata={},
        ),
        data={},
    )
    state = build_moss_tts_local_state(payload)
    assert state.token_count == 50
    assert state.text == "hello world"
    assert state.language == "English"


# ---------------------------------------------------------------------------
# Preprocessing handoff + result adapter
# ---------------------------------------------------------------------------


class _FakeProcessor:
    """Builds deterministic [1, T, 13] rows from the message text length."""

    @staticmethod
    def build_user_message(**kwargs):
        return dict(kwargs, role="user")

    def __call__(self, conversations, mode):
        assert mode == "generation"
        message = conversations[0][0]
        text = str(message.get("text", ""))
        seq = max(4, len(text) % 7 + 4)
        rows = torch.full((1, seq, N_VQ + 1), 1024, dtype=torch.long)
        rows[0, :, 0] = torch.arange(seq)
        rows[0, -1, 0] = 151669  # trailing audio_start row
        return {"input_ids": rows}


def _payload(text: str = "hello") -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs={"text": text}, params={}, metadata={}),
        data={},
    )


def test_preprocess_and_result_adapter():
    set_moss_tts_local_preprocessing_context(processor=_FakeProcessor())
    try:
        payload = preprocess_moss_tts_local_payload(_payload())
        assert payload.data.get("_moss_tts_local_prepared_request") == "req-1"

        from sglang_omni.models.moss_tts_local.request_builders import (
            pop_prepared_moss_tts_local_request,
        )

        prepared = pop_prepared_moss_tts_local_request(payload)
        assert prepared is not None
        assert prepared.prompt_rows.ndim == 2
        assert prepared.prompt_rows.shape[1] == N_VQ + 1
        assert len(prepared.input_ids_list) == prepared.prompt_rows.shape[0]

        data = MossTTSLocalSGLangRequestData(
            input_ids=prepared.input_ids,
            max_new_tokens=16,
            temperature=0.0,
            output_ids=[],
            state=prepared.state,
            prompt_rows=prepared.prompt_rows,
            stage_payload=payload,
            engine_start_s=0.0,
        )
        data.output_rows = [
            torch.cat([torch.tensor([151656]), torch.arange(N_VQ, dtype=torch.long)])
            for _ in range(3)
        ]
        result = apply_sglang_moss_tts_local_result(payload, data)
        codes = torch.as_tensor(result.data["audio_codes"])
        assert codes.shape == (3, N_VQ)
        assert result.data["completion_tokens"] == 3
        assert result.data["prompt_tokens"] == prepared.prompt_rows.shape[0]
    finally:
        clear_moss_tts_local_preprocessing_context()


def test_result_adapter_empty_generation():
    payload = _payload()
    data = MossTTSLocalSGLangRequestData(
        input_ids=torch.zeros(4, dtype=torch.long),
        max_new_tokens=16,
        temperature=0.0,
        output_ids=[],
        prompt_rows=torch.full((4, N_VQ + 1), 1024, dtype=torch.long),
        stage_payload=payload,
        engine_start_s=0.0,
    )
    result = apply_sglang_moss_tts_local_result(payload, data)
    codes = torch.as_tensor(result.data["audio_codes"])
    assert codes.shape == (0, N_VQ)


# ---------------------------------------------------------------------------
# Repetition penalty parity
# ---------------------------------------------------------------------------


def test_audio_repetition_penalty_matches_upstream_semantics():
    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner

    logits = torch.tensor(
        [[2.0, -1.0, 0.5, 3.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float32
    )
    history_row0 = torch.tensor([[0, 9], [2, 9]], dtype=torch.long)  # channel 0: {0, 2}
    histories = [history_row0, None]
    expected = logits.clone()
    penalty = 1.5
    expected[0, 0] = expected[0, 0] / penalty  # positive -> divide
    expected[0, 2] = expected[0, 2] / penalty

    MossTTSLocalModelRunner._apply_audio_repetition_penalty(
        logits, histories, [penalty, 1.0], channel=0
    )
    torch.testing.assert_close(logits, expected)

    # Negative scores multiply.
    logits2 = torch.tensor([[-2.0, 1.0]], dtype=torch.float32)
    MossTTSLocalModelRunner._apply_audio_repetition_penalty(
        logits2, [torch.tensor([[0]], dtype=torch.long)], [2.0], channel=0
    )
    torch.testing.assert_close(
        logits2, torch.tensor([[-4.0, 1.0]], dtype=torch.float32)
    )


def test_row_radix_token_ids_hash_rows_and_keep_eos():
    from sglang_omni.models.moss_tts.request_builders import build_row_cache_key_ids
    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner

    end_id = 151670
    slot_id = 151656
    rows = torch.full((3, N_VQ + 1), 7, dtype=torch.long)
    rows[:, 0] = torch.tensor([slot_id, end_id, slot_id])
    rows[2, 1:] = torch.arange(N_VQ)
    next_text = rows[:, 0].clone()

    out = MossTTSLocalModelRunner._row_radix_token_ids(rows, next_text, end_id)
    expected = [k % 151643 for k in build_row_cache_key_ids(rows)]
    assert int(out[1]) == end_id  # stop decision keeps the raw eos id
    assert int(out[0]) == expected[0]
    assert int(out[2]) == expected[2]
    assert int(out[0]) != int(out[2])  # different codes -> different keys
    assert int(out[0]) != slot_id  # no longer the constant slot id
    # Generated ids must stay inside the vocab: the scheduler finishes any
    # request whose output id crosses the vocab boundary.
    assert all(0 <= int(v) < 151936 for v in out)


def test_gather_rep_histories_excludes_prompt_and_inactive_rows():
    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner

    class _Data:
        def __init__(self, rows):
            self.output_rows = rows

    row = torch.cat([torch.tensor([151656]), torch.arange(N_VQ, dtype=torch.long)])
    active = _Data([row, row + 0])
    inactive = _Data([row])
    empty = _Data([])

    histories = MossTTSLocalModelRunner._gather_rep_histories(
        [active, inactive, empty], [1.5, 1.0, 1.5], torch.device("cpu")
    )
    assert histories is not None
    assert histories[0].shape == (2, N_VQ)  # generated frames only, channel cols
    assert histories[1] is None  # unit penalty -> skipped
    assert histories[2] is None  # no frames yet
    # All penalties at 1.0 -> no history gathering at all.
    assert (
        MossTTSLocalModelRunner._gather_rep_histories(
            [active], [1.0], torch.device("cpu")
        )
        is None
    )


def test_build_generation_kwargs_precedence():
    # Direct field names apply tts_params-then-params (params wins, matching
    # the MOSS Delay semantics); both override the explicit generic aliases.
    kwargs = build_generation_kwargs(
        {"temperature": 0.5, "audio_temperature": 1.2},
        tts_params={
            "explicit_generation_params": ["temperature"],
            "audio_temperature": 1.9,
        },
    )
    assert kwargs["text_temperature"] == 0.5
    assert kwargs["audio_temperature"] == 1.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_decode_frame_graphed_matches_branchless_eager():
    """The captured frame graph must reproduce the branchless eager decode."""
    from sglang_omni.models.moss_tts_local.local_transformer import (
        sample_seeded_branchless,
    )

    torch.manual_seed(11)
    device = torch.device("cuda")
    module = MossTTSLocalTransformer(
        hidden_size=64,
        num_heads=4,
        inner_size=96,
        num_layers=1,
        max_positions=N_VQ + 1,
        rope_base=1_000_000.0,
    ).to(device=device, dtype=torch.bfloat16)
    tables = [
        torch.randn(64, 64, device=device, dtype=torch.bfloat16) for _ in range(N_VQ)
    ]

    def frame(hidden, seeds, base):
        current = module.step(hidden, 0)
        codes = []
        for channel in range(N_VQ):
            logits = (current.float() @ tables[channel].float().T)[:, :32]
            code = sample_seeded_branchless(
                logits,
                temperature=torch.full((hidden.shape[0],), 1.7, device=device),
                top_p=torch.full((hidden.shape[0],), 0.8, device=device),
                top_k=torch.full(
                    (hidden.shape[0],), 25, device=device, dtype=torch.long
                ),
                seeds=seeds,
                positions=base + channel + 1,
            )
            codes.append(code)
            if channel + 1 < N_VQ:
                embed = torch.nn.functional.embedding(code, tables[channel][:32])
                current = module.step(embed.to(torch.bfloat16), channel + 1)
        return torch.stack(codes, dim=-1)

    batch = 4
    static_hidden = torch.zeros(batch, 64, device=device, dtype=torch.bfloat16)
    static_seeds = torch.zeros(batch, device=device, dtype=torch.long)
    static_base = torch.zeros(batch, device=device, dtype=torch.long)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            frame(static_hidden, static_seeds, static_base)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graphed_codes = frame(static_hidden, static_seeds, static_base)

    hidden = torch.randn(batch, 64, device=device, dtype=torch.bfloat16)
    seeds = torch.arange(batch, device=device, dtype=torch.long) * 999
    base = torch.full((batch,), 13, device=device, dtype=torch.long)

    static_hidden.copy_(hidden)
    static_seeds.copy_(seeds)
    static_base.copy_(base)
    graph.replay()
    from_graph = graphed_codes.clone()

    eager = frame(hidden, seeds, base)
    torch.testing.assert_close(from_graph, eager)


def test_batched_reference_encoder_coalesces_and_isolates_errors():
    import threading

    from sglang_omni.models.moss_tts_local.stages import _BatchedReferenceEncoder

    calls = []

    class _FakeCodecProcessor:
        def encode_audios_from_path(self, paths):
            calls.append(list(paths))
            out = []
            for p in paths:
                if "bad" in p:
                    raise RuntimeError(f"cannot read {p}")
                out.append(torch.full((4, N_VQ), len(p), dtype=torch.long))
            return out

    encoder = _BatchedReferenceEncoder(
        _FakeCodecProcessor(), max_batch_size=4, max_batch_wait_ms=20
    )
    results = {}

    def run(path):
        try:
            results[path] = encoder.encode(path)
        except Exception as exc:
            results[path] = exc

    threads = [threading.Thread(target=run, args=(p,)) for p in ("aa", "bbb", "bad1")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert isinstance(results["bad1"], Exception)
    assert results["aa"].shape == (4, N_VQ) and int(results["aa"][0, 0]) == 2
    assert results["bbb"].shape == (4, N_VQ) and int(results["bbb"][0, 0]) == 3
    # The failing batch retried per item; good items still succeeded.
    assert any(len(c) > 1 for c in calls) or len(calls) >= 3


def test_branchless_sampler_matches_eager_sampler():
    """The CUDA-graphable sampler must reproduce the eager path exactly."""
    pytest.importorskip("sglang")
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.models.moss_tts_local.local_transformer import (
        sample_seeded_branchless,
    )

    torch.manual_seed(7)
    rows, vocab = 6, 64
    logits = torch.randn(rows, vocab, dtype=torch.float32) * 3
    temperature = torch.tensor([1.7, 1.0, 0.5, 1.7, 0.0, 1.7])
    top_p = torch.tensor([0.8, 1.0, 0.9, 0.8, 0.8, 0.8])
    top_k = torch.tensor([25, 50, 8, 64, 25, 1], dtype=torch.long)
    seeds = torch.arange(rows, dtype=torch.long) * 1234567
    positions = torch.arange(rows, dtype=torch.long) * 13

    eager = MossTTSModelRunner._sample_tokens(
        logits.clone(),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seeds=seeds,
        positions=positions,
    )
    branchless = sample_seeded_branchless(
        logits.clone(),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seeds=seeds,
        positions=positions,
    )
    torch.testing.assert_close(eager, branchless)


# ---------------------------------------------------------------------------
# Stereo audio payload + encoding
# ---------------------------------------------------------------------------


def test_audio_waveform_payload_keeps_stereo_shape():
    wav = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    payload = audio_waveform_payload(wav, keep_channels=True)
    assert payload["audio_waveform_shape"] == [2, 4]
    restored = np.frombuffer(payload["audio_waveform"], dtype=np.float32).reshape(2, 4)
    np.testing.assert_allclose(restored, wav.numpy())
    # Default behavior still flattens.
    flat = audio_waveform_payload(wav)
    assert flat["audio_waveform_shape"] == [8]


def test_encode_wav_stereo_header_and_interleave():
    stereo = np.stack(
        [np.full(4, 0.5, dtype=np.float32), np.full(4, -0.5, dtype=np.float32)]
    )
    blob = encode_wav(stereo, 48000)
    assert blob[:4] == b"RIFF" and blob[8:12] == b"WAVE"
    num_channels = struct.unpack("<H", blob[22:24])[0]
    sample_rate = struct.unpack("<I", blob[24:28])[0]
    assert num_channels == 2
    assert sample_rate == 48000
    pcm = np.frombuffer(blob[44:], dtype=np.int16).reshape(-1, 2)
    assert (pcm[:, 0] > 0).all() and (pcm[:, 1] < 0).all()


def test_encode_audio_stereo_wav_and_mono_fallback():
    stereo = np.stack(
        [np.ones(64, dtype=np.float32) * 0.1, np.ones(64, dtype=np.float32) * -0.1]
    )
    blob, mime = encode_audio(stereo, response_format="wav", sample_rate=48000)
    assert mime == "audio/wav"
    assert struct.unpack("<H", blob[22:24])[0] == 2
    # Mono input keeps the legacy single-channel header.
    mono_blob, _ = encode_audio(
        np.ones(64, dtype=np.float32) * 0.1, response_format="wav", sample_rate=48000
    )
    assert struct.unpack("<H", mono_blob[22:24])[0] == 1
