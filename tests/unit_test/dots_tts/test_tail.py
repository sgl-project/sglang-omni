# SPDX-License-Identifier: Apache-2.0
"""Contracts for the batched dots.tts acoustic tail.

The tail is a batched, KV-cached rewrite of a per-request full-recompute solver,
so the tests that matter are equivalence checks: against an independent
full-recompute MeanFlow sample, and against the mask definition spelled out
element by element.
"""

from __future__ import annotations

import torch

from sglang_omni.models.dots_tts.components.backbone.dit import (
    DiT,
    fuse_dit_for_inference,
)
from sglang_omni.models.dots_tts.components.backbone.encoder import VAESemanticEncoder
from sglang_omni.models.dots_tts.components.model_config import (
    _DiTConfig,
    _EncoderConfig,
)
from sglang_omni.models.dots_tts.tail import (
    DotsTtsAcousticTail,
    DotsTtsTailSpec,
    batched_causal_update_mask,
)

FM_HIDDEN = 32
LATENT_DIM = 6
PATCH_SIZE = 2
NFE = 2


class _TailModelStub(torch.nn.Module):
    """The subset of the serving model the tail reads."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_patch_size = 1
        self.latent_patch_size = PATCH_SIZE
        self.latent_dim = LATENT_DIM
        self.fm_hidden_size = FM_HIDDEN
        self.velocity_field_predictor = DiT(
            in_dim=FM_HIDDEN,
            out_dim=LATENT_DIM,
            transformer_config=_DiTConfig(
                num_layers=2,
                num_heads=2,
                hidden_size=FM_HIDDEN,
                ffn_hidden_size=64,
                modulation=True,
                qk_norm=True,
                rotary_bias=True,
            ),
            mode="meanflow",
        )
        _randomize(self.velocity_field_predictor)
        self.coordinate_proj = torch.nn.Linear(LATENT_DIM, FM_HIDDEN)
        self.latent_proj = torch.nn.Linear(LATENT_DIM, FM_HIDDEN)


def _randomize(module: torch.nn.Module) -> torch.nn.Module:
    """Give every parameter a non-trivial value.

    ``DiT.initialize_weights`` zeroes the output layer and the adaLN modulations
    so training starts from an identity flow. Left that way the sampled velocity
    is identically zero and every solver trivially agrees, so the equivalence
    tests below would pass without exercising any of the cached math.
    """
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, 0.2)
    return module


def _patch_encoder() -> VAESemanticEncoder:
    encoder_config = _EncoderConfig(
        num_layers=1,
        num_heads=2,
        hidden_size=FM_HIDDEN,
        ffn_hidden_size=64,
        causal=True,
    )
    config = type(
        "_EncCfg", (), {"patch_size": PATCH_SIZE, "PatchEncoder": encoder_config}
    )()
    return VAESemanticEncoder(in_dim=LATENT_DIM, out_dim=FM_HIDDEN, config=config)


def _meanflow_modulations(dit: torch.nn.Module, g_cond: torch.Tensor) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, NFE + 1, dtype=g_cond.dtype)
    return dit.build_mods(
        grid[:-1], duration=grid[1:] - grid[:-1], g_cond=g_cond.reshape(1, -1)
    )


def _reference_meanflow(
    dit: torch.nn.Module,
    coordinate_proj: torch.nn.Module,
    *,
    sequence: torch.Tensor,
    fm_seq_len: int,
    g_cond: torch.Tensor,
) -> torch.Tensor:
    """Full-recompute MeanFlow sample: no KV cache, no per-row padding.

    Independent oracle for :meth:`DotsTtsAcousticTail.sample_patches` — it runs
    the uncached ``FusedAdaLNDiT.forward`` over the whole flow prefix on every
    ODE step, which is what the cached batched path must reproduce.
    """
    total = fm_seq_len + PATCH_SIZE
    x_base = sequence.new_zeros(1, total, FM_HIDDEN)
    x_base[:, :fm_seq_len] = sequence[:, :fm_seq_len]

    attn_mask = torch.zeros((1, total, total), dtype=torch.bool)
    block_start = fm_seq_len - 1
    if block_start > 0:
        attn_mask[:, :block_start, :block_start] = (
            torch.ones(block_start, block_start, dtype=torch.bool).triu(1).logical_not()
        )
    attn_mask[:, block_start:fm_seq_len, :fm_seq_len] = True
    attn_mask[:, block_start:fm_seq_len, fm_seq_len:] = True
    attn_mask[:, fm_seq_len:, :] = True

    pos_ids = torch.zeros((1, total), dtype=torch.float32)
    pos_ids[:, :fm_seq_len] = torch.arange(fm_seq_len, dtype=torch.float32)
    pos_ids[:, fm_seq_len:] = torch.arange(
        fm_seq_len, fm_seq_len + PATCH_SIZE, dtype=torch.float32
    )

    z = torch.randn(1, PATCH_SIZE, LATENT_DIM, dtype=sequence.dtype)
    times = torch.linspace(0.0, 1.0, NFE + 1, dtype=sequence.dtype)
    for step in range(NFE):
        x = x_base.clone()
        x[:, fm_seq_len:] = coordinate_proj(z)
        t = times[step].expand(1)
        dt = (times[step + 1] - times[step]).expand(1)
        velocity = dit(
            x=x,
            timesteps=t,
            duration=dt,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            g_cond=g_cond,
        )[:, fm_seq_len:]
        z = (z + velocity * dt.view(-1, 1, 1)).clone()
    return z


def _build_tail(
    model: _TailModelStub, *, num_slots: int, patch_capacity: int
) -> DotsTtsAcousticTail:
    return DotsTtsAcousticTail(
        dit=fuse_dit_for_inference(model),
        coordinate_proj=model.coordinate_proj,
        patch_encoder=_randomize(_patch_encoder()),
        spec=DotsTtsTailSpec(
            nfe=NFE,
            patch_capacity=patch_capacity,
            num_slots=num_slots,
            hidden_patch_size=1,
            latent_patch_size=PATCH_SIZE,
            latent_dim=LATENT_DIM,
            fm_hidden_size=FM_HIDDEN,
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_batched_mask_matches_the_per_row_definition() -> None:
    capacity, prev_len, current_len = 8, 3, 3
    valid = torch.tensor([0, 3, 7])
    batched = batched_causal_update_mask(
        capacity_tokens=capacity,
        valid_persistent=valid,
        prev_len=prev_len,
        current_len=current_len,
    )

    q_len = prev_len + current_len
    assert batched.shape == (len(valid), 1, q_len, capacity + q_len)
    for row, valid_persistent in enumerate(valid.tolist()):
        for q in range(q_len):
            for kv in range(capacity + q_len):
                tail_idx = kv - capacity
                if tail_idx < 0:
                    expected = kv < valid_persistent
                elif q < prev_len:
                    expected = tail_idx < prev_len and tail_idx <= q
                else:
                    expected = True
                assert bool(batched[row, 0, q, kv]) is expected


def test_kv_cached_tail_matches_the_full_recompute_solver() -> None:
    torch.manual_seed(1234)
    model = _TailModelStub().eval()
    tail = _build_tail(model, num_slots=1, patch_capacity=8)
    dit = tail.dit
    unit = tail.spec.unit_len

    prompt_patches = 3
    g_cond = torch.randn(1, FM_HIDDEN)
    all_mods = _meanflow_modulations(dit, g_cond)
    fm_rows = torch.randn(prompt_patches * unit, FM_HIDDEN)

    slot = tail.acquire_slot()
    tail.seed_fm_history(slot, fm_rows=fm_rows, all_mods=all_mods)

    reference_sequence = torch.zeros(1, tail.spec.dit_cache_tokens + unit, FM_HIDDEN)
    reference_sequence[0, : fm_rows.size(0)] = fm_rows
    reference_len = fm_rows.size(0)

    for step in range(3):
        hidden_row = torch.randn(1, FM_HIDDEN)
        reference_sequence[0, reference_len] = hidden_row[0]
        reference_len += 1

        torch.manual_seed(step)
        expected = _reference_meanflow(
            dit,
            model.coordinate_proj,
            sequence=reference_sequence,
            fm_seq_len=reference_len,
            g_cond=g_cond,
        )

        torch.manual_seed(step)
        actual = tail.sample_patches(
            [slot],
            fm_hidden_rows=hidden_row,
            latent_proj=model.latent_proj,
        )

        torch.manual_seed(step)
        untouched_noise = torch.randn(1, PATCH_SIZE, LATENT_DIM)
        assert not torch.allclose(
            expected, untouched_noise
        ), "the DiT left the noise unchanged, so this comparison is vacuous"
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        latent_rows = model.latent_proj(expected)[0]
        reference_sequence[0, reference_len : reference_len + PATCH_SIZE] = latent_rows
        reference_len += PATCH_SIZE
        assert tail.fm_seq_len(slot) == reference_len


def test_slots_stay_independent_when_batched_together() -> None:
    torch.manual_seed(4321)
    model = _TailModelStub().eval()
    solo = _build_tail(model, num_slots=1, patch_capacity=8)
    shared = _build_tail(model, num_slots=2, patch_capacity=8)
    unit = solo.spec.unit_len

    g_cond = torch.randn(1, FM_HIDDEN)
    all_mods = _meanflow_modulations(solo.dit, g_cond)
    # A second slot with a longer history so the batch has to pad and mask.
    fm_rows = torch.randn(2 * unit, FM_HIDDEN)
    other_rows = torch.randn(4 * unit, FM_HIDDEN)

    solo_slot = solo.acquire_slot()
    solo.set_slot_seed(solo_slot, 99)
    solo.seed_fm_history(solo_slot, fm_rows=fm_rows, all_mods=all_mods)
    first = shared.acquire_slot()
    second = shared.acquire_slot()
    shared.set_slot_seed(first, 99)
    shared.seed_fm_history(first, fm_rows=fm_rows, all_mods=all_mods)
    shared.seed_fm_history(second, fm_rows=other_rows, all_mods=all_mods)

    for _step in range(2):
        hidden = torch.randn(2, FM_HIDDEN)
        expected = solo.sample_patches(
            [solo_slot],
            fm_hidden_rows=hidden[:1],
            latent_proj=model.latent_proj,
        )
        batched = shared.sample_patches(
            [first, second],
            fm_hidden_rows=hidden,
            latent_proj=model.latent_proj,
        )
        torch.testing.assert_close(batched[:1], expected, rtol=2e-4, atol=2e-4)


def test_slot_pool_is_bounded_and_reusable() -> None:
    model = _TailModelStub().eval()
    tail = _build_tail(model, num_slots=2, patch_capacity=8)

    first = tail.acquire_slot()
    second = tail.acquire_slot()
    assert {first, second} == {0, 1}
    try:
        tail.acquire_slot()
    except RuntimeError as error:
        assert "ran out of slots" in str(error)
    else:  # pragma: no cover - the pool must be bounded
        raise AssertionError("acquire_slot must fail once the pool is exhausted")

    tail.release_slot(first)
    assert tail.acquire_slot() == first


def test_prefill_failure_releases_the_slot() -> None:
    """A prefill that raises must not strand its slot in the pool.

    The slot is only reachable through the runner's ``_request_states``, so a
    leak here is unrecoverable: after ``num_slots`` failures every later request
    dies in ``acquire_slot`` and the engine needs a restart.
    """
    from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner

    model = _TailModelStub().eval()
    tail = _build_tail(model, num_slots=2, patch_capacity=8)

    class _Model:
        latent_dim = LATENT_DIM
        latent_patch_size = PATCH_SIZE

        def __init__(self) -> None:
            self.tail = tail
            self._decode_input_embedding = torch.nn.Embedding(2, FM_HIDDEN)

        def patch_encoder_input(self, latents, *, normalized):
            raise RuntimeError("prompt audio exceeds the patch-encoder cache")

        def speaker_condition(self, embedding, scale):
            return torch.zeros(1, FM_HIDDEN)

        def meanflow_modulations(self, g_cond, *, nfe):
            return torch.zeros(nfe, 8)

    runner = object.__new__(DotsTTSModelRunner)
    runner.model = _Model()
    runner.device = torch.device("cpu")
    runner._nfe = NFE
    runner._request_states = {}

    payload_state = type(
        "_State",
        (),
        {
            "spk_emb": torch.zeros(1, 8),
            "prompt_latent": torch.zeros(1, PATCH_SIZE, LATENT_DIM),
            "speaker_scale": 1.5,
            "seed": None,
        },
    )()
    sched_req = type(
        "_Req",
        (),
        {
            "request_id": "req",
            "data": type(
                "_Data",
                (),
                {
                    "state": payload_state,
                    "prompt_patch_count": 1,
                    "prompt_span_start": 1,
                    "input_ids": torch.zeros(2, dtype=torch.long),
                },
            )(),
        },
    )()

    free_before = len(tail._free_slots)
    for _attempt in range(free_before + 2):
        try:
            runner._materialize_request_state(sched_req)
        except RuntimeError:
            pass
        else:  # pragma: no cover - the stub always raises
            raise AssertionError("_materialize_request_state must propagate")
    assert len(tail._free_slots) == free_before
    assert runner._request_states == {}
