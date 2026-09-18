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
    assert flow.transformer.text_cond is None


@torch.inference_mode()
def test_singleton_keeps_original_path(monkeypatch):
    def unexpected(*args):
        raise AssertionError("singleton must keep the existing path")

    monkeypatch.setattr("sglang_omni.models.auk.dit.flash_attention", unexpected)
    flow, items = make_flow(), make_items()[:1]
    a = flow.sample_batch(items, steps=2, cfg_strength=2.0)
    b = flow.sample_batch(items, steps=2, cfg_strength=2.0, enable_packed_dit=True)
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)


@pytest.mark.parametrize(
    "capability, hip, fa3, fa4, expected",
    [
        ((8, 0), False, True, True, 3),
        ((9, 0), False, True, True, 3),
        ((10, 0), False, True, True, 4),
        ((10, 0), False, True, False, "FlashAttention 4 on sm100"),
        ((12, 0), False, False, True, "unsupported on sm120"),
        ((9, 4), True, True, True, "unsupported on HIP"),
        ((7, 5), False, False, True, "unsupported on sm75"),
    ],
)
def test_flash_version_policy(monkeypatch, capability, hip, fa3, fa4, expected):
    import sglang.kernels.ops.attention.flash_attention_v3 as v3
    import sglang.kernels.ops.attention.flash_attention_v4 as v4

    packed.resolve_flash_version.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(torch.version, "hip", "6.2" if hip else None)
    monkeypatch.setattr(v3, "_is_fa3_supported", lambda device=None: fa3)
    monkeypatch.setattr(v4, "is_flash_attention_v4_available", lambda: fa4)
    if isinstance(expected, int):
        assert packed.resolve_flash_version(torch.device("cuda", 0)) == expected
    else:
        with pytest.raises(ValueError, match=expected):
            packed.resolve_flash_version(torch.device("cuda", 0))
    packed.resolve_flash_version.cache_clear()


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
def test_packed_flash_cuda_matches_padded(monkeypatch):
    from sglang_omni.models.auk.packed import flash_attention

    def checked_attention(q, k, v, layout):
        output = flash_attention(q, k, v, layout)
        reference = reference_varlen(q, k, v, layout)
        torch.testing.assert_close(output, reference, atol=0.005, rtol=0.02)
        return output

    monkeypatch.setattr("sglang_omni.models.auk.dit.flash_attention", checked_attention)
    flow, items = make_flow("cuda", torch.bfloat16), make_items("cuda")
    a = flow.sample_batch(items, steps=4, cfg_strength=2.0)
    b = flow.sample_batch(items, steps=4, cfg_strength=2.0, enable_packed_dit=True)
    for output, reference in zip(b, a):
        assert output.dtype == torch.float32
        assert torch.isfinite(output).all()
        # Different GEMM/attention reductions accumulate through CFG and Euler.
        # Use the existing BF16-backbone diagnostic; real-model WER is evaluated
        # separately, rather than treating random-weight latents as audio quality.
        cosine = F.cosine_similarity(output.flatten(), reference.flatten(), dim=0)
        assert cosine > 0.99
