# SPDX-License-Identifier: Apache-2.0
"""Tests for the MLX Qwen3-TTS scheduler runner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
import numpy as np  # noqa: E402

from sglang_omni.models.qwen3_tts.mlx.config import (  # noqa: E402
    CodePredictorConfig,
    TalkerConfig,
)
from sglang_omni.models.qwen3_tts.mlx.runner import (  # noqa: E402
    Qwen3TTSMlxModelRunner,
    Qwen3TTSRequestSpec,
)
from sglang_omni.models.qwen3_tts.mlx.sampling import SamplingParams  # noqa: E402
from sglang_omni.models.qwen3_tts.mlx.talker import (  # noqa: E402
    Qwen3TTSTalkerForConditionalGeneration,
)

HIDDEN = 8
GROUPS = 4


def _talker() -> Qwen3TTSTalkerForConditionalGeneration:
    mx.random.seed(0)
    config = TalkerConfig(
        code_predictor_config=CodePredictorConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            num_code_groups=GROUPS,
        ),
        vocab_size=1040,  # > 1024 so the reserved special block exists
        hidden_size=HIDDEN,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        text_hidden_size=6,
        text_vocab_size=40,
        num_code_groups=GROUPS,
        codec_eos_token_id=1030,
    )
    model = Qwen3TTSTalkerForConditionalGeneration(config)
    mx.eval(model.parameters())
    return model


class _FakeBase:
    """Stands in for MlxModelRunner.

    Real SGLang attention caches and a real ``BatchedDecodeContext`` are used so
    the batched decode path under test is the production one, not a stub.
    """

    def __init__(self, model) -> None:
        from sglang.srt.hardware_backend.mlx import kv_cache as kv

        self._kv = kv
        kv.patch_model_attention(model)
        layers, _ = kv.find_attention_layers(model)
        self._cache_layout = SimpleNamespace(
            num_layers=len(layers),
            attention_layer_indices=tuple(range(len(layers))),
            attention_pool_index_by_layer={i: i for i in range(len(layers))},
            full_kv_pool_index_by_layer={i: i for i in range(len(layers))},
            has_auxiliary_state=False,
        )
        self.model = model
        self.disable_radix_cache = True
        self._req_caches: dict[str, list] = {}
        self._req_token_ids: dict[str, list[int]] = {}
        self._req_pool_idx: dict[str, int] = {}
        self._req_synced_offset: dict[str, int] = {}
        self._decode_step_ct = 0
        self._clear_steps = 0
        self.removed: list[str] = []

    def _acquire_cache(self):
        config = self.model.config
        return [
            self._kv.ContiguousAttentionKVCache(
                n_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                max_seq_len=64,
                dtype=mx.float32,
            )
            for _ in range(self._cache_layout.num_layers)
        ]

    def _build_batched_decode_context(self, caches, req_ids):
        num_layers = self._cache_layout.num_layers
        return self._kv.BatchedDecodeContext(
            batch_size=len(req_ids),
            seq_lens=[cache[0].offset for cache in caches],
            attention_layer_caches=[
                [per_req[layer] for per_req in caches] for layer in range(num_layers)
            ],
        )

    def _store_auxiliary_state(self, req_pool_idx, cache) -> None:
        pass

    def prefill_finalize(self, pending) -> int:
        token = int(pending.lazy_token.item())
        self._req_token_ids[pending.req_id] = list(pending.full_token_ids) + [token]
        self._req_caches[pending.req_id] = pending.cache
        self._req_pool_idx[pending.req_id] = pending.req_pool_idx
        return token

    def decode_batch_finalize(self, pending) -> list[int]:
        raw = pending.lazy_tokens.tolist()
        tokens = [int(t) for t in (raw if isinstance(raw, list) else [raw])]
        for rid, token in zip(pending.req_ids, tokens):
            self._req_token_ids[rid].append(token)
        return tokens

    def remove_request(self, req_id: str) -> None:
        self.removed.append(req_id)
        self._req_caches.pop(req_id, None)
        self._req_token_ids.pop(req_id, None)

    def clear(self) -> None:
        self._req_caches.clear()
        self._req_token_ids.clear()


class _Runner(Qwen3TTSMlxModelRunner, _FakeBase):
    pass


def _spec(*, temperature: float = 0.0, seed: int | None = None, trailing=None):
    mx.random.seed(1)
    prompt = mx.random.normal((1, 5, HIDDEN))
    pad = mx.random.normal((1, 1, HIDDEN))
    params = SamplingParams(temperature=temperature, top_k=0, top_p=1.0)
    return Qwen3TTSRequestSpec(
        prompt_embeds=prompt,
        trailing_text_embeds=trailing,
        pad_embed=pad,
        semantic=params,
        subtalker=params,
        seed=seed,
    )


def _prefill(runner: _Runner, req_id: str, spec) -> int:
    runner.register_request(req_id, spec)
    pending = runner.prefill_start(
        req_id=req_id,
        new_token_ids=[1, 2],
        full_token_ids=[1, 2],
        prefix_slot_ids=[],
        new_slot_ids=[0, 1],
        req_pool_idx=0,
    )
    mx.eval(pending.lazy_token)
    return runner.prefill_finalize(pending)


# --------------------------------------------------------------------------


def test_prefill_emits_one_frame_and_a_group_zero_token() -> None:
    runner = _Runner(_talker())
    token = _prefill(runner, "r0", _spec())

    frames = runner.drain_frames("r0")
    assert len(frames) == 1
    assert frames[0].shape == (GROUPS,)
    # Group 0 of the frame is exactly the token reported to the scheduler.
    assert int(frames[0][0]) == token
    # Draining is destructive.
    assert runner.drain_frames("r0") == []


def test_prefill_requires_a_registered_prompt() -> None:
    runner = _Runner(_talker())
    with pytest.raises(RuntimeError, match="register_request"):
        runner.prefill_start(
            req_id="missing",
            new_token_ids=[],
            full_token_ids=[],
            prefix_slot_ids=[],
            new_slot_ids=[],
            req_pool_idx=0,
        )


def test_prefill_rejects_a_radix_prefix() -> None:
    runner = _Runner(_talker())
    runner.register_request("r0", _spec())
    with pytest.raises(NotImplementedError, match="radix prefix"):
        runner.prefill_start(
            req_id="r0",
            new_token_ids=[1],
            full_token_ids=[1],
            prefix_slot_ids=[7],
            new_slot_ids=[1],
            req_pool_idx=0,
        )


def test_decode_produces_one_frame_per_step() -> None:
    runner = _Runner(_talker())
    _prefill(runner, "r0", _spec())
    runner.drain_frames("r0")

    for _ in range(3):
        pending = runner.decode_batch_start(["r0"])
        mx.eval(pending.lazy_tokens, pending.lazy_codes, pending.lazy_feedback)
        tokens = runner.decode_batch_finalize(pending)
        assert len(tokens) == 1

    frames = runner.drain_frames("r0")
    assert len(frames) == 3
    assert all(frame.shape == (GROUPS,) for frame in frames)


def test_chained_decode_matches_stepwise_decode() -> None:
    """A chained step must equal building the step after finalising."""
    stepwise = _Runner(_talker())
    _prefill(stepwise, "r0", _spec())
    first = stepwise.decode_batch_start(["r0"])
    mx.eval(first.lazy_tokens, first.lazy_feedback)
    stepwise.decode_batch_finalize(first)
    second = stepwise.decode_batch_start(["r0"])
    mx.eval(second.lazy_tokens)
    expected = second.lazy_tokens.tolist()

    chained = _Runner(_talker())
    _prefill(chained, "r0", _spec())
    root = chained.decode_batch_start(["r0"])
    nxt = chained.decode_batch_start_chained(root)
    mx.eval(root.lazy_tokens, nxt.lazy_tokens)
    chained.decode_batch_finalize(root)
    got = nxt.lazy_tokens.tolist()

    assert got == expected


def test_trailing_text_is_consumed_once_then_padded() -> None:
    trailing = mx.random.normal((1, 2, HIDDEN))
    spec = _spec(trailing=trailing)
    runner = _Runner(_talker())
    _prefill(runner, "r0", spec)

    # Prefill consumed row 0; two decode steps consume row 1 then fall back.
    assert spec.trailing_index == 1
    for _ in range(2):
        pending = runner.decode_batch_start(["r0"])
        mx.eval(pending.lazy_tokens, pending.lazy_feedback)
        runner.decode_batch_finalize(pending)
    assert spec.trailing_index == 2


def test_batched_decode_keeps_frames_per_request() -> None:
    runner = _Runner(_talker())
    for req_id in ("a", "b"):
        _prefill(runner, req_id, _spec())
        runner.drain_frames(req_id)

    pending = runner.decode_batch_start(["a", "b"])
    mx.eval(pending.lazy_tokens, pending.lazy_codes, pending.lazy_feedback)
    tokens = runner.decode_batch_finalize(pending)

    assert len(tokens) == 2
    assert pending.lazy_codes.shape == (2, GROUPS)
    for index, req_id in enumerate(("a", "b")):
        frames = runner.drain_frames(req_id)
        assert len(frames) == 1
        assert int(frames[0][0]) == tokens[index]


def test_codec_history_excludes_the_prompt_text_ids() -> None:
    """The repetition penalty must see codec tokens, not prompt text ids."""
    runner = _Runner(_talker())
    token = _prefill(runner, "r0", _spec())

    # The base class's list starts with the prompt's text ids.
    assert runner._req_token_ids["r0"][:2] == [1, 2]
    # The codec history holds only what the talker emitted.
    assert runner._tts_emitted["r0"] == [token]


def test_special_codec_tokens_are_suppressed() -> None:
    runner = _Runner(_talker())
    suppressed = set(runner._suppress_tokens())
    assert 1030 not in suppressed  # EOS stays sampleable
    assert 1039 in suppressed and 16 in suppressed
    # Cached, not rebuilt per row.
    assert runner._suppress_tokens() is runner._suppress_tokens()


def test_removing_a_request_drops_all_of_its_state() -> None:
    runner = _Runner(_talker())
    _prefill(runner, "r0", _spec())
    runner.remove_request("r0")

    assert "r0" not in runner._tts_specs
    assert "r0" not in runner._tts_frames
    assert "r0" not in runner._tts_emitted
    assert runner.removed == ["r0"]


def test_seed_makes_sampled_generation_reproducible() -> None:
    tokens = []
    for _ in range(2):
        runner = _Runner(_talker())
        _prefill(runner, "r0", _spec(temperature=0.9, seed=4321))
        step = []
        for _ in range(3):
            pending = runner.decode_batch_start(["r0"])
            mx.eval(pending.lazy_tokens, pending.lazy_feedback)
            step.extend(runner.decode_batch_finalize(pending))
        tokens.append(step)
    assert tokens[0] == tokens[1]


def test_unsupported_decode_features_are_rejected() -> None:
    runner = _Runner(_talker())
    _prefill(runner, "r0", _spec())
    with pytest.raises(NotImplementedError):
        runner.decode_batch_start(["r0"], logprob_spec=object())
    with pytest.raises(NotImplementedError):
        runner.decode_batch_start(["r0"], logits_hook=lambda x: x)


# --------------------------------------------------------------------------
# request-spec adapter
# --------------------------------------------------------------------------


def test_request_spec_converts_torch_prompt_and_trailing_rows() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.qwen3_tts.mlx.request_spec import build_request_spec

    data = SimpleNamespace(
        prefill_input_embeds=torch.zeros(5, HIDDEN, dtype=torch.bfloat16),
        prompt_input_embeds=None,
        tts_pad_embed=torch.ones(1, HIDDEN, dtype=torch.bfloat16),
        pending_text_queue=SimpleNamespace(
            rows=torch.zeros(3, HIDDEN, dtype=torch.bfloat16), cursor=1, _chunks=[]
        ),
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        subtalker_dosample=True,
        subtalker_temperature=0.5,
        subtalker_top_k=10,
        subtalker_top_p=0.8,
        semantic_sampling_seed=99,
    )
    spec = build_request_spec(data)

    assert spec.prompt_embeds.shape == (1, 5, HIDDEN)
    assert spec.prompt_embeds.dtype == mx.bfloat16
    assert spec.pad_embed.shape == (1, 1, HIDDEN)
    # cursor=1 means one row was already consumed.
    assert spec.trailing_text_embeds.shape == (1, 2, HIDDEN)
    assert spec.semantic.temperature == pytest.approx(0.7)
    assert spec.semantic.repetition_penalty == pytest.approx(1.1)
    assert spec.subtalker.top_k == 10
    assert spec.seed == 99


def test_request_spec_is_none_before_the_prompt_exists() -> None:
    from sglang_omni.models.qwen3_tts.mlx.request_spec import build_request_spec

    data = SimpleNamespace(prefill_input_embeds=None, prompt_input_embeds=None)
    assert build_request_spec(data) is None


def test_greedy_subtalker_when_sampling_is_disabled() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.qwen3_tts.mlx.request_spec import build_request_spec

    data = SimpleNamespace(
        prefill_input_embeds=torch.zeros(2, HIDDEN),
        tts_pad_embed=torch.zeros(1, HIDDEN),
        pending_text_queue=None,
        subtalker_dosample=False,
    )
    spec = build_request_spec(data)
    assert spec.subtalker.greedy
    assert spec.trailing_text_embeds is None
    assert np.asarray(spec.prompt_embeds).shape == (1, 2, HIDDEN)


# --------------------------------------------------------------------------
# scheduler-side bridge
# --------------------------------------------------------------------------


class _StubMlxRunner:
    def __init__(self, frames: dict[str, list], eos_id: int = 1030) -> None:
        self._frames = frames
        self._tts_specs: dict[str, object] = {}
        self._talker_config = SimpleNamespace(codec_eos_token_id=eos_id)
        self.registered: list[str] = []

    def drain_frames(self, req_id):
        return self._frames.pop(req_id, [])

    def has_request(self, req_id):
        return False

    def register_request(self, req_id, spec):
        self.registered.append(req_id)
        self._tts_specs[req_id] = spec


def _sched_req(req_id: str):
    data = SimpleNamespace(
        output_codes=[],
        latest_stream_code_chunk=None,
        prefill_input_embeds=None,
        prompt_input_embeds=None,
    )
    return SimpleNamespace(request_id=req_id, data=data)


def _bridge(stub):
    from sglang_omni.models.qwen3_tts.mlx.scheduler_runner import (
        Qwen3TTSMlxSchedulerModelRunner,
    )

    bridge = Qwen3TTSMlxSchedulerModelRunner.__new__(Qwen3TTSMlxSchedulerModelRunner)
    object.__setattr__(bridge, "tp_worker", SimpleNamespace(_mlx_runner=stub))
    return bridge


def test_bridge_appends_frames_to_request_output_codes() -> None:
    frame_a = np.array([5, 6, 7, 8])
    frame_b = np.array([1, 2, 3, 4])
    stub = _StubMlxRunner({"r0": [frame_a, frame_b]})
    bridge = _bridge(stub)

    sched_req = _sched_req("r0")
    outputs = {"r0": SimpleNamespace(data=42)}
    bridge.post_process_outputs(None, SimpleNamespace(requests=[sched_req]), outputs)

    assert len(sched_req.data.output_codes) == 2
    assert np.array_equal(sched_req.data.output_codes[0], frame_a)
    assert np.array_equal(sched_req.data.latest_stream_code_chunk, frame_b)


def test_bridge_drops_the_frame_of_an_eos_step() -> None:
    stub = _StubMlxRunner({"r0": [np.array([1, 2, 3, 4])]}, eos_id=1030)
    bridge = _bridge(stub)

    sched_req = _sched_req("r0")
    outputs = {"r0": SimpleNamespace(data=1030)}
    bridge.post_process_outputs(None, SimpleNamespace(requests=[sched_req]), outputs)

    assert sched_req.data.output_codes == []
    assert sched_req.data.latest_stream_code_chunk is None


def test_bridge_is_a_noop_when_no_frames_were_produced() -> None:
    stub = _StubMlxRunner({})
    bridge = _bridge(stub)
    sched_req = _sched_req("r0")
    bridge.post_process_outputs(
        None, SimpleNamespace(requests=[sched_req]), {"r0": SimpleNamespace(data=7)}
    )
    assert sched_req.data.output_codes == []


def test_bridge_registers_prompts_before_launch() -> None:
    torch = pytest.importorskip("torch")
    stub = _StubMlxRunner({})
    bridge = _bridge(stub)

    sched_req = _sched_req("r0")
    sched_req.data.prefill_input_embeds = torch.zeros(4, HIDDEN)
    sched_req.data.tts_pad_embed = torch.zeros(1, HIDDEN)
    sched_req.data.pending_text_queue = None

    bridge._register_requests([sched_req])
    assert stub.registered == ["r0"]

    # Registration is idempotent: a second pass must not re-register.
    bridge._register_requests([sched_req])
    assert stub.registered == ["r0"]


def test_bridge_skips_requests_without_a_prompt_yet() -> None:
    stub = _StubMlxRunner({})
    bridge = _bridge(stub)
    bridge._register_requests([_sched_req("r0")])
    assert stub.registered == []
