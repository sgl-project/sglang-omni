# SPDX-License-Identifier: Apache-2.0
"""Opt-in component parity on the public Moshi base (docs/cookbook/personaplex.md).

Mimi, the input embeddings and the depformer are checked against tensors that
personaplex_reference_dump.py saved from the reference package.
"""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional

from sglang_omni.models.personaplex.architecture import (
    DEPFORMER,
    MIMI_WEIGHTS_GLOB,
    MOSHI_WEIGHTS_NAME,
    NUM_AUDIO_STREAMS,
)
from sglang_omni.models.personaplex.components.depformer import (
    Depformer,
    DepformerLayer,
    rms_norm_f32,
)
from sglang_omni.models.personaplex.components.mimi import (
    MimiCodec,
    load_mimi_codec,
    resolve_mimi_weights,
)
from sglang_omni.models.personaplex.sglang_model import PersonaPlexForCausalLM
from sglang_omni.utils.checkpoint import resolve_checkpoint

pytestmark = pytest.mark.accelerator

DEFAULT_MOSHI_BASE = "kyutai/moshiko-pytorch-bf16"
DUMP_SCRIPT = Path(__file__).with_name("personaplex_reference_dump.py")
MIMI_ATOL = 1e-5  # float32 codec through two cuDNN builds, TF32 off on both
# The reference tensors come from its streaming path, the one it serves with; its
# non-streaming forward lacks the 250-position context window.
LOGIT_ATOL_F32 = 1e-3  # float32 depformer; logits are O(10)


def checkpoint_tensors(
    path: Path, prefixes: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Only the named groups of a 15 GB checkpoint, read lazily."""
    with safe_open(str(path), "pt", device="cpu") as handle:
        return {
            name: handle.get_tensor(name)
            for name in handle.keys()
            if name.startswith(prefixes)
        }


@pytest.fixture(scope="module", autouse=True)
def exact_float32():
    """What the dump script sets too: no TF32, no autotuned cuDNN algorithms."""
    backends = torch.backends
    saved = (
        backends.cuda.matmul.allow_tf32,
        backends.cudnn.allow_tf32,
        backends.cudnn.benchmark,
        backends.cudnn.deterministic,
    )
    backends.cuda.matmul.allow_tf32 = False
    backends.cudnn.allow_tf32 = False
    backends.cudnn.benchmark = False
    backends.cudnn.deterministic = True
    yield
    (
        backends.cuda.matmul.allow_tf32,
        backends.cudnn.allow_tf32,
        backends.cudnn.benchmark,
        backends.cudnn.deterministic,
    ) = saved


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    if not torch.cuda.is_available():
        pytest.skip("PersonaPlex component parity requires CUDA")
    return Path(
        resolve_checkpoint(os.environ.get("PERSONAPLEX_MOSHI_BASE", DEFAULT_MOSHI_BASE))
    )


@pytest.fixture(scope="module")
def reference(checkpoint: Path) -> dict[str, torch.Tensor]:
    """The reference dump, produced first so the two never share the GPU."""
    dump = os.environ.get("PERSONAPLEX_REFERENCE_DUMP")
    if not dump:
        pytest.skip(
            "Set PERSONAPLEX_REFERENCE_DUMP to the reference tensors file "
            "(made by personaplex_reference_dump.py)"
        )
    path = Path(dump).expanduser()
    if not path.exists():
        python = os.environ.get("PERSONAPLEX_REFERENCE_PYTHON")
        source = os.environ.get("PERSONAPLEX_REFERENCE_SOURCE")
        if not (python and source):
            pytest.skip(
                f"{path} is missing; set PERSONAPLEX_REFERENCE_PYTHON and "
                "PERSONAPLEX_REFERENCE_SOURCE to create it"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        clip = Path(source).expanduser() / "assets" / "test" / "input_assistant.wav"
        subprocess.run(
            [
                python,
                str(DUMP_SCRIPT),
                "--checkpoint",
                str(checkpoint),
                "--clip",
                str(clip),
                "--out",
                str(path),
            ],
            check=True,
        )
    return {name: value.cuda() for name, value in load_file(str(path)).items()}


@pytest.fixture(scope="module")
def codec(checkpoint: Path, reference) -> MimiCodec:
    return load_mimi_codec(
        resolve_mimi_weights(checkpoint, MIMI_WEIGHTS_GLOB), device="cuda"
    )


@pytest.fixture(scope="module")
def depformer_f32(checkpoint: Path, reference) -> Depformer:
    model = Depformer(DEPFORMER)
    model.load_reference_weights(
        checkpoint_tensors(checkpoint / MOSHI_WEIGHTS_NAME, ("depformer", "linears."))
    )
    return model.cuda().eval()


@pytest.fixture(scope="module")
def depformer_bf16(depformer_f32: Depformer) -> Depformer:
    return copy.deepcopy(depformer_f32).to(torch.bfloat16)


def test_mimi_encode_matches_reference(codec, reference):
    codes = codec.encode(reference["wav"])
    identical = (codes == reference["codes"]).all(dim=1)
    print(
        f"\n[mimi encode] codes identical for {int(identical.sum())} of "
        f"{identical.shape[-1]} frames"
    )
    assert torch.equal(codes, reference["codes"])


def test_mimi_decode_matches_reference(codec, reference):
    codes = reference["codes"]
    whole = codec.decode(codes)
    state = codec.init_decode_state()
    chunked = torch.cat(
        [
            codec.decode_step(codes[:, :, f : f + 1], state)
            for f in range(codes.shape[-1])
        ],
        dim=-1,
    )
    whole_diff = (whole - reference["decoded"]).abs().max().item()
    chunked_diff = (chunked - reference["decoded"]).abs().max().item()
    print(
        f"\n[mimi decode] whole max diff {whole_diff:.2e}, "
        f"chunked max diff {chunked_diff:.2e}"
    )
    assert whole_diff <= MIMI_ATOL
    assert chunked_diff <= MIMI_ATOL


def test_input_embeddings_match_reference(checkpoint, reference):
    tables = checkpoint_tensors(checkpoint / MOSHI_WEIGHTS_NAME, ("emb.", "text_emb."))
    model = SimpleNamespace(
        audio_emb=nn.ModuleList(
            nn.Embedding.from_pretrained(tables[f"emb.{k}.weight"])
            for k in range(NUM_AUDIO_STREAMS)
        ).cuda(),
        text_emb=nn.Embedding.from_pretrained(tables["text_emb.weight"]).cuda(),
    )
    with torch.inference_mode():
        embedded = PersonaPlexForCausalLM.embed_rows(model, reference["emb_rows"])
    diff = (embedded.float() - reference["emb_out"].float()).abs().max().item()
    print(f"\n[embeddings] max diff {diff:.2e} over {embedded.shape[0]} rows")
    assert torch.equal(embedded, reference["emb_out"])


def depformer_logits(
    model: Depformer, reference: dict[str, torch.Tensor], transformer_out: torch.Tensor
) -> torch.Tensor:
    """[B, steps, card] float logits of a teacher-forced frame."""
    recorded = []

    def record(logits: torch.Tensor) -> torch.Tensor:
        recorded.append(logits)
        return logits.argmax(dim=-1)

    with torch.inference_mode():
        model.generate(
            reference["dep_text_token"],
            transformer_out,
            reference["dep_forced_codes"],
            record,
        )
    return torch.stack(recorded, dim=1)


def per_step_diff(logits: torch.Tensor, expected: torch.Tensor) -> list[float]:
    return (logits - expected).abs().amax(dim=(0, 2)).tolist()


def ring_step(self, x_BD, step, cache_2BHSD):
    """DepformerLayer.step with the reference's full-ring behaviour at the last step.

    On the 8-step base the reference's ring holds exactly one frame; once full,
    its position math marks step 0 as future, so the last step never sees it.
    """
    spec = self.spec
    h = rms_norm_f32(x_BD, self.norm1_alpha, spec.rms_norm_eps)
    qkv = functional.linear(h, self.in_proj_weight[step])
    q, k, v = rearrange(qkv, "b (p h d) -> p b h d", p=3, h=spec.num_heads)
    cache_2BHSD[0, :, :, step] = k
    cache_2BHSD[1, :, :, step] = v
    first = 1 if step == spec.steps - 1 else 0
    attn = functional.scaled_dot_product_attention(
        q[:, :, None],
        cache_2BHSD[0, :, :, first : step + 1],
        cache_2BHSD[1, :, :, first : step + 1],
    )
    x_BD = x_BD + functional.linear(
        rearrange(attn, "b h 1 d -> b (h d)"), self.out_proj_weight[step]
    )
    h = rms_norm_f32(x_BD, self.norm2_alpha, spec.rms_norm_eps)
    gate = functional.linear(h, self.gate_in_weight[step])
    gate, up = gate.chunk(2, dim=-1)
    return x_BD + functional.linear(
        functional.silu(gate) * up, self.gate_out_weight[step]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_depformer_logits_match_reference(
    dtype, depformer_f32, depformer_bf16, reference, monkeypatch
):
    model = depformer_f32 if dtype == torch.float32 else depformer_bf16
    expected = reference[
        "dep_logits_f32" if dtype == torch.float32 else "dep_logits_bf16"
    ]
    transformer_out = reference["transformer_out"].to(dtype)
    # Note (wilsonzheng0327): bf16 matmul and attention kernels differ between torch
    # builds by an ULP; the float32 pass is the exact one.
    atol = (
        LOGIT_ATOL_F32
        if dtype == torch.float32
        else torch.finfo(torch.bfloat16).eps * expected.abs().max().item()
    )

    plain = per_step_diff(depformer_logits(model, reference, transformer_out), expected)
    monkeypatch.setattr(DepformerLayer, "step", ring_step)
    emulated = per_step_diff(
        depformer_logits(model, reference, transformer_out), expected
    )
    print(
        f"\n[depformer {dtype}] tolerance {atol:.2e}; max logit diff per step "
        f"{[f'{d:.1e}' for d in plain]}; with the reference ring emulated "
        f"{[f'{d:.1e}' for d in emulated]}"
    )

    assert max(plain[:-1]) <= atol
    assert plain[-1] > atol, "step 7 should only match with the ring emulated"
    assert max(emulated) <= atol
