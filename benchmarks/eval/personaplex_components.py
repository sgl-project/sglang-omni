# SPDX-License-Identifier: Apache-2.0
"""Component comparison on the public Moshi base; see personaplex.md.

Mimi, the input embeddings and the depformer are checked against tensors that
personaplex_reference_dump.py saved from the reference package.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

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

DUMP_SCRIPT = Path(__file__).with_name("personaplex_reference_dump.py")
MIMI_ATOL = 1e-5  # float32 codec through two cuDNN builds, TF32 off on both
# The reference tensors come from its streaming path, the one it serves with; its
# non-streaming forward lacks the 250-position context window.
LOGIT_ATOL_F32 = 1e-3  # float32 depformer; logits are O(10)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    else:
        pass


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


def reference(
    checkpoint: Path, source: Path, *, python: str, dump: Path
) -> dict[str, torch.Tensor]:
    """Generate fresh reference tensors before allocating port tensors."""
    path = dump.expanduser().resolve()
    command = [
        python,
        str(DUMP_SCRIPT.resolve()),
        "--checkpoint",
        str(checkpoint),
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def compare_mimi_encode(codec: MimiCodec, reference: dict[str, torch.Tensor]) -> None:
    codes = codec.encode(reference["wav"])
    identical = (codes == reference["codes"]).all(dim=1)
    print(
        f"\n[mimi encode] codes identical for {int(identical.sum())} of "
        f"{identical.shape[-1]} frames"
    )
    check(
        torch.equal(codes, reference["codes"]), "Mimi codes differ from the reference"
    )


def compare_mimi_decode(codec: MimiCodec, reference: dict[str, torch.Tensor]) -> None:
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
    check(whole_diff <= MIMI_ATOL, f"whole decode differs by {whole_diff:.2e}")
    check(chunked_diff <= MIMI_ATOL, f"chunked decode differs by {chunked_diff:.2e}")


def compare_input_embeddings(
    checkpoint: Path, reference: dict[str, torch.Tensor]
) -> None:
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
    check(torch.equal(embedded, reference["emb_out"]), "input embeddings differ")


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


def compare_depformer_logits(
    model: Depformer, dtype: torch.dtype, reference: dict[str, torch.Tensor]
) -> None:
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
    plain_step = DepformerLayer.step
    DepformerLayer.step = ring_step
    try:
        emulated = per_step_diff(
            depformer_logits(model, reference, transformer_out), expected
        )
    finally:
        DepformerLayer.step = plain_step
    print(
        f"\n[depformer {dtype}] tolerance {atol:.2e}; max logit diff per step "
        f"{[f'{d:.1e}' for d in plain]}; with the reference ring emulated "
        f"{[f'{d:.1e}' for d in emulated]}"
    )

    check(max(plain[:-1]) <= atol, f"depformer {dtype}: steps 0-6 exceed {atol:.2e}")
    check(plain[-1] > atol, "step 7 should only match with the ring emulated")
    check(
        max(emulated) <= atol, f"depformer {dtype}: ring emulation exceeds {atol:.2e}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PersonaPlex component comparison.")
    p.add_argument(
        "--reference-source",
        required=True,
        help="clean NVIDIA/personaplex checkout; its assets/test clip is used",
    )
    p.add_argument(
        "--reference-python",
        required=True,
        help="interpreter of the reference environment (its torch pin differs)",
    )
    p.add_argument(
        "--checkpoint",
        default="kyutai/moshiko-pytorch-bf16",
        help="public Moshi base checkpoint: local directory or resolvable model id",
    )
    p.add_argument(
        "--dump",
        required=True,
        help="safetensors file the reference dump is written to",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.reference_source).expanduser().resolve()
    checkpoint = Path(resolve_checkpoint(args.checkpoint))
    if not torch.cuda.is_available():
        raise RuntimeError("PersonaPlex component comparison requires CUDA")
    else:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tensors = reference(
        checkpoint, source, python=args.reference_python, dump=Path(args.dump)
    )
    codec = load_mimi_codec(
        resolve_mimi_weights(checkpoint, MIMI_WEIGHTS_GLOB), device="cuda"
    )
    compare_mimi_encode(codec, tensors)
    compare_mimi_decode(codec, tensors)
    compare_input_embeddings(checkpoint, tensors)
    depformer = Depformer(DEPFORMER)
    depformer.load_reference_weights(
        checkpoint_tensors(checkpoint / MOSHI_WEIGHTS_NAME, ("depformer", "linears."))
    )
    depformer = depformer.cuda().eval()
    compare_depformer_logits(depformer, torch.float32, tensors)
    compare_depformer_logits(
        copy.deepcopy(depformer).to(torch.bfloat16), torch.bfloat16, tensors
    )


if __name__ == "__main__":
    main()
