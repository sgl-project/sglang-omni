# SPDX-License-Identifier: Apache-2.0
"""Packed layout parity against the original padded AuK execution."""

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.auk import packed
from sglang_omni.models.auk.dit import AuKDit
from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem


def reference_varlen(q, k, v, layout):
    offsets = layout.cu_seqlens.tolist()
    return torch.cat(
        [
            F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1),
                k[start:end].transpose(0, 1),
                v[start:end].transpose(0, 1),
                dropout_p=0.0,
            ).transpose(0, 1)
            for start, end in zip(offsets, offsets[1:])
        ]
    )


def make_flow(device="cpu", dtype=torch.float32):
    torch.manual_seed(52)
    flow = AuKFlowMatching(
        AuKDit(
            dim=32,
            heads=2,
            dim_head=16,
            latent_dim=8,
            text_hidden_dim=16,
            num_layers=2,
            num_single_layers=2,
        ),
        num_llm_layers=2,
    ).eval()
    # AuK initializes output and modulation to zero; randomize them so parity
    # actually exercises attention, cross-token mixing, and residual updates.
    for parameter in flow.parameters():
        torch.nn.init.uniform_(parameter, -0.3, 0.3)
    return flow.to(device=device, dtype=dtype)


def make_items(device="cpu", no_reference=False):
    result = []
    for i, (text, frames, ref) in enumerate([(7, 19, 5), (9, 11, 8), (4, 15, 0)]):
        ref = 0 if no_reference else ref
        mask = torch.ones(text, dtype=torch.bool, device=device)
        # Include left padding and a hole; neither is equivalent to [:sum(mask)].
        mask[0] = False
        mask[2] = False
        result.append(
            AuKSampleItem(
                torch.randn(text, 16, device=device),
                mask,
                frames,
                torch.randn(ref, 8, device=device) if ref else None,
                seed=i + 1,
                ref_length=max(0, ref - 2),
            )
        )
    return result


@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
@pytest.mark.parametrize("no_reference", [False, True])
@torch.inference_mode()
def test_packed_matches_padded_with_holes_and_stored_padding(
    monkeypatch, cfg_strength, no_reference
):
    monkeypatch.setattr("sglang_omni.models.auk.dit.flash_attention", reference_varlen)
    flow = make_flow(dtype=torch.float64)
    items = make_items(no_reference=no_reference)
    sampling = dict(steps=4, cfg_strength=cfg_strength)
    expected = flow.sample_batch(items, **sampling)
    actual = flow.sample_batch(items, enable_packed_dit=True, **sampling)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-8)
    # A different request ordering must not mix either requests or CFG branches.
    permuted = flow.sample_batch(items[::-1], enable_packed_dit=True, **sampling)
    for a, b in zip(permuted[::-1], actual):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize("enable_packed_dit", [False, True])
@torch.inference_mode()
def test_rope_tables_are_built_once_per_trajectory(monkeypatch, enable_packed_dit):
    """Positions are fixed across Euler steps: three rotary tables per trajectory,
    handed to the fused Q/K kernel as the same tensors on every step."""
    monkeypatch.setattr("sglang_omni.models.auk.dit.flash_attention", reference_varlen)
    flow, items = make_flow(), make_items()
    rotary = flow.transformer.rotary_embed
    original = rotary.forward
    calls = []
    monkeypatch.setattr(
        rotary, "forward", lambda *a, **k: calls.append(1) or original(*a, **k)
    )
    keys = set()

    class RecordingFusion:
        """Stand-in for fused_norm_rope: record the table, run the native math."""

        def __init__(self, attention):
            self.attention = attention

        def __call__(self, q, k, q_norm, k_norm, rope):
            keys.add((rope.freqs.data_ptr(), tuple(rope.freqs.shape)))
            self.attention.qk_fusion = None
            try:
                return self.attention.norm_rope(q, k, q_norm, k_norm, rope)
            finally:
                self.attention.qk_fusion = self

    blocks = (
        *flow.transformer.transformer_blocks,
        *flow.transformer.single_transformer_blocks,
    )
    for block in blocks:
        block.attn.qk_fusion = RecordingFusion(block.attn)
    flow.sample_batch(
        items, steps=4, cfg_strength=2.0, enable_packed_dit=enable_packed_dit
    )
    # Audio, text and joint tables: one build each, not one per Euler step.
    assert len(calls) == 3
    assert len(keys) == 3


@torch.inference_mode()
def test_singleton_keeps_original_path(monkeypatch):
    def unexpected(*args):
        raise AssertionError("singleton must keep the existing path")

    monkeypatch.setattr("sglang_omni.models.auk.dit.flash_attention", unexpected)
    flow, items = make_flow(), make_items()[:1]
    a = flow.sample_batch(items, steps=2, cfg_strength=2.0)
    b = flow.sample_batch(items, steps=2, cfg_strength=2.0, enable_packed_dit=True)
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)


def _fake_upstream(*, blackwell, fa3, fa4):
    from types import SimpleNamespace

    return lambda: SimpleNamespace(
        is_blackwell=lambda: blackwell,
        is_fa3_supported=lambda device=None: fa3,
        is_fa4_available=lambda: fa4,
        flash_attn_varlen_func=None,
    )


@pytest.mark.parametrize(
    "capability, hip, blackwell, fa3, fa4, expected",
    [
        # SGLang's predicates decide: is_blackwell -> FA4, _is_fa3_supported -> FA3.
        ((8, 0), False, False, True, True, 3),
        ((9, 0), False, False, True, True, 3),
        ((10, 0), False, True, True, True, 4),
        ((10, 3), False, True, False, True, 4),
        ((12, 0), False, True, False, True, 4),
        ((12, 0), False, True, False, False, "FlashAttention 4 on sm120"),
        ((9, 4), True, False, True, True, "unsupported on HIP"),
        ((7, 5), False, False, False, True, "unsupported on sm75"),
    ],
)
def test_flash_version_policy(
    monkeypatch, capability, hip, blackwell, fa3, fa4, expected
):
    packed.resolve_flash_version.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(torch.version, "hip", "6.2" if hip else None)
    monkeypatch.setattr(
        packed, "upstream_flash", _fake_upstream(blackwell=blackwell, fa3=fa3, fa4=fa4)
    )
    if isinstance(expected, int):
        assert packed.resolve_flash_version(torch.device("cuda", 0)) == expected
    else:
        with pytest.raises(ValueError, match=expected):
            packed.resolve_flash_version(torch.device("cuda", 0))
    packed.resolve_flash_version.cache_clear()


@pytest.mark.parametrize(
    "version, major, expected",
    [(4, 12, "sm120"), (4, 10, "generic:4"), (3, 9, "generic:3"), (3, 8, "generic:3")],
)
def test_varlen_kernel_dispatch_matches_flash_attention_backend(
    monkeypatch, version, major, expected
):
    """FA4 on sm12x binds SGLang's flash_attention_v4_sm120 entry point, as its
    FlashAttentionBackend does; everything else goes through the generic
    flash_attn_varlen_func(ver=...) dispatcher."""
    import sys
    from types import ModuleType, SimpleNamespace

    sm120 = ModuleType("sglang.kernels.ops.attention.flash_attention_v4_sm120")
    sm120.flash_attn_varlen_func = lambda *a, **k: "sm120"
    monkeypatch.setitem(sys.modules, sm120.__name__, sm120)
    monkeypatch.setattr(
        packed,
        "upstream_flash",
        lambda: SimpleNamespace(
            flash_attn_varlen_func=lambda *a, ver, **k: f"generic:{ver}"
        ),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (major, 0))
    packed.varlen_func.cache_clear()
    layout = SimpleNamespace(cu_seqlens=None, max_seqlen=4, flash_version=version)
    q = torch.zeros(1)
    assert packed.flash_attention(q, q, q, layout) == expected
    packed.varlen_func.cache_clear()


# Skip the real-kernel test with the gate's own reason on hosts it rejects.
try:
    packed.resolve_flash_version(torch.device("cuda", 0))
    flash_unsupported = None
except (ImportError, ValueError) as exc:
    flash_unsupported = str(exc)
finally:
    packed.resolve_flash_version.cache_clear()


@pytest.mark.skipif(flash_unsupported is not None, reason=str(flash_unsupported))
@torch.inference_mode()
def test_packed_flash_cuda_matches_padded(monkeypatch, record_property):
    """Kernel-level parity of every varlen call, then end-to-end parity.

    Each FlashAttention call is compared row by row against fp32 SDPA over the
    same bf16 inputs. The kernel rounds P and the output to bf16 (2^-8 relative
    each) and accumulates in fp32, so the per-row max-abs error is bounded by
    about one bf16 ulp of the largest value; ``KERNEL_ULPS`` times eps times
    max|v| leaves a 2x margin. Observed on FA4 (RTX 5090, 16 calls): 0.26 ulp.
    The numbers are recorded as test properties.
    """
    from sglang_omni.models.auk.packed import flash_attention

    KERNEL_ULPS = 2
    eps = torch.finfo(torch.bfloat16).eps
    row_errors = []

    def checked_attention(q, k, v, layout):
        output = flash_attention(q, k, v, layout)
        reference = reference_varlen(q.float(), k.float(), v.float(), layout)
        assert output.dtype == torch.bfloat16
        row_error = (output.float() - reference).abs().amax(dim=(1, 2))
        bound = KERNEL_ULPS * eps * v.abs().max()
        row_errors.append(
            (row_error.max().item(), row_error.mean().item(), bound.item())
        )
        assert torch.isfinite(output).all()
        assert row_error.max() <= bound, (
            f"varlen kernel row max-abs error {row_error.max().item():.4g} exceeds "
            f"{bound.item():.4g} (ver={layout.flash_version}, rows={tuple(q.shape)})"
        )
        return output

    monkeypatch.setattr("sglang_omni.models.auk.dit.flash_attention", checked_attention)
    flow, items = make_flow("cuda", torch.bfloat16), make_items("cuda")
    a = flow.sample_batch(items, steps=4, cfg_strength=2.0)
    b = flow.sample_batch(items, steps=4, cfg_strength=2.0, enable_packed_dit=True)
    assert row_errors
    worst_row, mean_row, bound = max(row_errors)
    version = packed.resolve_flash_version(torch.device("cuda", 0))
    record_property("flash_version", version)
    record_property("flash_calls", len(row_errors))
    record_property("flash_row_max_abs_error", worst_row)
    record_property("flash_row_mean_abs_error", max(mean for _, mean, _ in row_errors))
    record_property("flash_row_max_abs_bound", bound)
    print(
        f"varlen ver={version}: {len(row_errors)} calls, row max-abs error "
        f"{worst_row:.4g} (bound {bound:.4g}), mean {mean_row:.4g}"
    )
    for i, (output, reference) in enumerate(zip(b, a)):
        assert output.dtype == torch.float32
        assert torch.isfinite(output).all()
        # Different GEMM/attention reductions accumulate through CFG and Euler.
        # Use the existing BF16-backbone diagnostic; real-model WER is evaluated
        # separately, rather than treating random-weight latents as audio quality.
        cosine = F.cosine_similarity(output.flatten(), reference.flatten(), dim=0)
        max_abs = (output - reference).abs().max().item()
        record_property(f"latent_{i}_cosine", cosine.item())
        record_property(f"latent_{i}_max_abs_error", max_abs)
        print(f"latent {i}: cosine {cosine.item():.6f}, max-abs {max_abs:.4g}")
        assert cosine > 0.99
