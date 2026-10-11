# SPDX-License-Identifier: Apache-2.0
"""Opt-in component comparison against NVIDIA/personaplex on the public Moshi base.

Mimi, the input embeddings and the depformer are checked against tensors that
personaplex_reference_dump.py saves from the reference package. Needs a
separate reference environment; set it up as "Reference parity" in
docs/cookbook/personaplex.md describes. Skips unless PERSONAPLEX_REFERENCE_SOURCE
and PERSONAPLEX_REFERENCE_PYTHON are set.
"""

from __future__ import annotations

import copy
import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
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

DEFAULT_CHECKPOINT = "kyutai/moshiko-pytorch-bf16"
DUMP_SCRIPT = Path(__file__).with_name("personaplex_reference_dump.py")
MIMI_ATOL = 1e-5  # float32 codec through two cuDNN builds, TF32 off on both
# The reference tensors come from its streaming path, the one it serves with; its
# non-streaming forward lacks the 250-position context window.
LOGIT_ATOL_F32 = 1e-3  # float32 depformer; logits are O(10)


@dataclass(frozen=True, kw_only=True)
class ComponentInputs:
    source: Path
    checkpoint: Path
    reference_python: str


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


@pytest.fixture(scope="module")
def component_inputs() -> Iterator[ComponentInputs]:
    source = os.environ.get("PERSONAPLEX_REFERENCE_SOURCE")
    python = os.environ.get("PERSONAPLEX_REFERENCE_PYTHON")
    if not source or not python:
        pytest.skip(
            "Set PERSONAPLEX_REFERENCE_SOURCE and PERSONAPLEX_REFERENCE_PYTHON "
            "for the component comparison"
        )
    else:
        pass
    if not torch.cuda.is_available():
        pytest.skip("PersonaPlex component comparison requires CUDA")
    else:
        pass
    flags = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = os.environ.get("PERSONAPLEX_MOSHI_CHECKPOINT", DEFAULT_CHECKPOINT)
    yield ComponentInputs(
        source=Path(source).expanduser().resolve(),
        checkpoint=Path(resolve_checkpoint(checkpoint)),
        reference_python=python,
    )
    (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    ) = flags
    # Note (wilsonzheng0327): Return the module's cached GPU memory, so a pipeline started
    # later in the same pytest run sees the free memory a separate run would.
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def reference_tensors(
    component_inputs: ComponentInputs, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, torch.Tensor]:
    """Fresh reference tensors, generated before the port allocates its own."""
    source = component_inputs.source
    path = (
        tmp_path_factory.mktemp("personaplex_components")
        / "moshi_base_reference.safetensors"
    )
    print(f"\nreference dump and log: {path.parent}")
    command = [
        component_inputs.reference_python,
        str(DUMP_SCRIPT.resolve()),
        "--checkpoint",
        str(component_inputs.checkpoint),
        "--clip",
        str(source / "assets" / "test" / "input_assistant.wav"),
        "--out",
        str(path),
        "--frames",
        "200",
        "--batch",
        "2",
        "--seed",
        "0",
        "--device",
        "cuda",
    ]
    with path.with_suffix(".log").open("w") as log:
        subprocess.run(
            command,
            cwd=source,
            env=dict(os.environ, PYTHONPATH=str(source / "moshi")),
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return {name: value.cuda() for name, value in load_file(str(path)).items()}


@pytest.fixture(scope="module")
def codec(component_inputs: ComponentInputs) -> MimiCodec:
    return load_mimi_codec(
        resolve_mimi_weights(component_inputs.checkpoint, MIMI_WEIGHTS_GLOB),
        device="cuda",
    )


@pytest.fixture(scope="module")
def depformer(component_inputs: ComponentInputs) -> Depformer:
    model = Depformer(DEPFORMER)
    model.load_reference_weights(
        checkpoint_tensors(
            component_inputs.checkpoint / MOSHI_WEIGHTS_NAME,
            ("depformer", "linears."),
        )
    )
    return model.cuda().eval()


def test_mimi_encode_matches_reference(
    reference_tensors: dict[str, torch.Tensor], codec: MimiCodec
) -> None:
    codes = codec.encode(reference_tensors["wav"])
    identical = (codes == reference_tensors["codes"]).all(dim=1)
    print(
        f"\n[mimi encode] codes identical for {int(identical.sum())} of "
        f"{identical.shape[-1]} frames"
    )
    assert torch.equal(
        codes, reference_tensors["codes"]
    ), "Mimi codes differ from the reference"


def test_mimi_decode_matches_reference(
    reference_tensors: dict[str, torch.Tensor], codec: MimiCodec
) -> None:
    codes = reference_tensors["codes"]
    whole = codec.decode(codes)
    state = codec.init_decode_state()
    chunked = torch.cat(
        [
            codec.decode_step(codes[:, :, f : f + 1], state)
            for f in range(codes.shape[-1])
        ],
        dim=-1,
    )
    whole_diff = (whole - reference_tensors["decoded"]).abs().max().item()
    chunked_diff = (chunked - reference_tensors["decoded"]).abs().max().item()
    print(
        f"\n[mimi decode] whole max diff {whole_diff:.2e}, "
        f"chunked max diff {chunked_diff:.2e}"
    )
    assert whole_diff <= MIMI_ATOL, f"whole decode differs by {whole_diff:.2e}"
    assert chunked_diff <= MIMI_ATOL, f"chunked decode differs by {chunked_diff:.2e}"


def test_input_embeddings_match_reference(
    reference_tensors: dict[str, torch.Tensor], component_inputs: ComponentInputs
) -> None:
    tables = checkpoint_tensors(
        component_inputs.checkpoint / MOSHI_WEIGHTS_NAME, ("emb.", "text_emb.")
    )
    model = SimpleNamespace(
        audio_emb=nn.ModuleList(
            nn.Embedding.from_pretrained(tables[f"emb.{k}.weight"])
            for k in range(NUM_AUDIO_STREAMS)
        ).cuda(),
        text_emb=nn.Embedding.from_pretrained(tables["text_emb.weight"]).cuda(),
    )
    with torch.inference_mode():
        embedded = PersonaPlexForCausalLM.embed_rows(
            model, reference_tensors["emb_rows"]
        )
    diff = (embedded.float() - reference_tensors["emb_out"].float()).abs().max().item()
    print(f"\n[embeddings] max diff {diff:.2e} over {embedded.shape[0]} rows")
    assert torch.equal(
        embedded, reference_tensors["emb_out"]
    ), "input embeddings differ"


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


def ring_step(
    self: DepformerLayer, x_BD: torch.Tensor, step: int, cache_2BHSD: torch.Tensor
) -> torch.Tensor:
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


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"]
)
def test_depformer_logits_match_reference(
    reference_tensors: dict[str, torch.Tensor],
    depformer: Depformer,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Steps 0-6 match as written; the last step only with the reference ring emulated."""
    model = depformer if dtype == torch.float32 else copy.deepcopy(depformer).to(dtype)
    expected = reference_tensors[
        "dep_logits_f32" if dtype == torch.float32 else "dep_logits_bf16"
    ]
    transformer_out = reference_tensors["transformer_out"].to(dtype)
    # Note (wilsonzheng0327): bf16 matmul and attention kernels differ between torch
    # builds by an ULP; the float32 pass is the exact one.
    atol = (
        LOGIT_ATOL_F32
        if dtype == torch.float32
        else torch.finfo(torch.bfloat16).eps * expected.abs().max().item()
    )

    plain = per_step_diff(
        depformer_logits(model, reference_tensors, transformer_out), expected
    )
    monkeypatch.setattr(DepformerLayer, "step", ring_step)
    emulated = per_step_diff(
        depformer_logits(model, reference_tensors, transformer_out), expected
    )
    print(
        f"\n[depformer {dtype}] tolerance {atol:.2e}; max logit diff per step "
        f"{[f'{d:.1e}' for d in plain]}; with the reference ring emulated "
        f"{[f'{d:.1e}' for d in emulated]}"
    )

    assert max(plain[:-1]) <= atol, f"depformer {dtype}: steps 0-6 exceed {atol:.2e}"
    assert plain[-1] > atol, "step 7 should only match with the ring emulated"
    assert (
        max(emulated) <= atol
    ), f"depformer {dtype}: ring emulation exceeds {atol:.2e}"
