# SPDX-License-Identifier: Apache-2.0
"""Save Moshi-base component outputs from the reference package.

Runs under the reference's own interpreter (its torch pin differs from ours)
and writes one safetensors file that test_personaplex_components.py compares
the port's Mimi codec, input embeddings and depformer against.

    python tests/test_model/personaplex_reference_dump.py \\
        --checkpoint ~/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/<rev> \\
        --clip ~/personaplex/assets/test/input_assistant.wav \\
        --out ~/.cache/personaplex-parity/moshi_base_reference.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sphn
import torch
from moshi.models import loaders
from moshi.models.lm import LMModel
from safetensors.torch import load_file, save_file

SAMPLES_PER_FRAME = 1920
AUDIO_CARD = 2048
TEXT_CARD = 32_000
NUM_STREAMS = 17
DEPFORMER_STEPS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="local Moshi-base checkpoint directory"
    )
    parser.add_argument("--clip", required=True, help="recording to encode with Mimi")
    parser.add_argument("--out", required=True, help="safetensors file to write")
    parser.add_argument("--frames", type=int, default=200, help="Mimi frames to encode")
    parser.add_argument("--batch", type=int, default=2, help="rows for the LM checks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def exact_float32() -> None:
    """The reference's own tests turn these off: TF32 and autotuned cuDNN
    algorithms would otherwise put kernel noise into the float32 codec."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_lm(checkpoint: Path, device: str) -> LMModel:
    """The base LM with its own 8 depformer steps.

    The reference's loader widens every model to 16 steps for PersonaPlex; the
    base checkpoint has 8, and keeping 8 makes the depformer ring exactly one
    frame deep, which is the configuration the port's step-7 emulation targets.
    """
    lm_kwargs = loaders._lm_kwargs  # noqa: leading-underscore  # moshi's name
    model = LMModel(device="meta", dtype=torch.bfloat16, **lm_kwargs)
    state = load_file(str(checkpoint / loaders.MOSHI_NAME), device=device)
    model.load_state_dict(state, strict=False, assign=True)
    return model.to(device=device, dtype=torch.bfloat16).eval()


def load_clip(path: str, frames: int, sample_rate: int) -> torch.Tensor:
    pcm, rate = sphn.read(path)
    if rate != sample_rate:
        pcm = sphn.resample(pcm, src_sample_rate=rate, dst_sample_rate=sample_rate)
    samples = frames * SAMPLES_PER_FRAME
    if pcm.shape[-1] < samples:
        raise ValueError(f"{path} has {pcm.shape[-1]} samples, need {samples}")
    return torch.from_numpy(np.ascontiguousarray(pcm[:1, :samples], dtype=np.float32))


def depformer_logits(
    lm: LMModel,
    text_token: torch.Tensor,
    forced_codes: torch.Tensor,
    transformer_out: torch.Tensor,
) -> torch.Tensor:
    """[B, steps, card] float logits of one teacher-forced frame."""
    logits = []
    previous = text_token
    with lm.depformer.streaming(text_token.shape[0]):
        for step in range(DEPFORMER_STEPS):
            out = lm.forward_depformer(step, previous[:, None, None], transformer_out)
            logits.append(out[:, 0, 0].float())
            previous = forced_codes[:, step]
    return torch.stack(logits, dim=1)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    exact_float32()
    torch.manual_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    device = args.device

    mimi = loaders.get_mimi(checkpoint / loaders.MIMI_NAME, device)
    wav = load_clip(args.clip, args.frames, mimi.sample_rate)[None].to(device)
    # Frame by frame, as the reference serves: its non-streaming forward has no
    # 250-position context window, so past 125 code frames it diverges from
    # its own streaming path, which the port follows.
    frames = range(wav.shape[-1] // SAMPLES_PER_FRAME)
    with mimi.streaming(1):
        codes = torch.cat(
            [
                mimi.encode(
                    wav[..., f * SAMPLES_PER_FRAME : (f + 1) * SAMPLES_PER_FRAME]
                )
                for f in frames
            ],
            dim=-1,
        )
    with mimi.streaming(1):
        decoded = torch.cat(
            [mimi.decode(codes[:, :, f : f + 1]) for f in range(codes.shape[-1])],
            dim=-1,
        )

    lm = load_lm(checkpoint, device)
    batch = args.batch
    rows = torch.randint(0, AUDIO_CARD, (batch, NUM_STREAMS), device=device)
    rows[:, 0] = torch.randint(0, TEXT_CARD, (batch,), device=device)
    # Row 0 is the initial row every stream starts from: one past each vocabulary.
    rows[0, 0] = TEXT_CARD
    rows[0, 1:] = AUDIO_CARD
    embeddings = lm.embed_codes(rows[:, :, None])
    with lm.streaming(batch):
        transformer_out, _ = lm.forward_embeddings(embeddings)

    text_token = torch.randint(0, TEXT_CARD, (batch,), device=device)
    forced_codes = torch.randint(0, AUDIO_CARD, (batch, DEPFORMER_STEPS), device=device)
    logits_bf16 = depformer_logits(lm, text_token, forced_codes, transformer_out)
    # A float32 pass as well: bf16 kernels differ between torch builds by an ULP,
    # which would hide whether the two depformers compute the same function.
    for module in (
        lm.depformer,
        lm.depformer_in,
        lm.depformer_emb,
        lm.depformer_text_emb,
        lm.linears,
    ):
        module.float()
    logits_f32 = depformer_logits(lm, text_token, forced_codes, transformer_out.float())

    tensors = {
        "wav": wav,
        "codes": codes,
        "decoded": decoded,
        "emb_rows": rows,
        "emb_out": embeddings[:, 0],
        "transformer_out": transformer_out[:, 0],
        "dep_text_token": text_token,
        "dep_forced_codes": forced_codes,
        "dep_logits_bf16": logits_bf16,
        "dep_logits_f32": logits_f32,
    }
    save_file(
        {name: value.cpu().contiguous() for name, value in tensors.items()},
        args.out,
        metadata={
            "checkpoint": str(checkpoint),
            "clip": args.clip,
            "frames": str(args.frames),
            "seed": str(args.seed),
        },
    )
    print(f"wrote {args.out}: {codes.shape[-1]} frames, batch {batch}")


if __name__ == "__main__":
    main()
