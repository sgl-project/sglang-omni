"""Microbench: MOSS Delay mixed sampling graph replay, full vs compacted draw.

Synthetic batch = 1 text row + 7 delay-31 audio rows (issue #3296 repro shape)
with the real MOSS-TTS-v1.5 vocabulary (text 155648, 32 audio codebooks x 1024).
Times graph.replay only: device sampling work, host launch included, no
backbone forward. Not a claim about end-to-end share.
"""

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch

from sglang_omni.models.moss_tts.sampler import (
    DelayGraphBatch,
    MossTTSDelayMixedGraphSampler,
)
from sglang_omni.models.moss_tts.sampling_cuda_graph import (
    MossTTSDelayMixedSamplingCudaGraphRunner,
)

TEXT_VOCAB = 155648
N_VQ = 32
AUDIO_VOCAB = 1024
INT64_MAX = (1 << 63) - 1


def make_config():
    return SimpleNamespace(
        n_vq=N_VQ,
        channels=N_VQ + 1,
        pad_token_id=151643,
        audio_start_token_id=151652,
        audio_end_token_id=151653,
        audio_assistant_gen_slot_token_id=151656,
        audio_assistant_delay_slot_token_id=151662,
        audio_pad_code=AUDIO_VOCAB,
        im_end_token_id=151645,
        vocab_size_list=[TEXT_VOCAB] + [AUDIO_VOCAB] * N_VQ,
    )


def measure(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 3),
        "min_ms": round(samples[0], 3),
        "mean_ms": round(statistics.fmean(samples), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    device = torch.device("npu:0")
    config = make_config()
    sampler = MossTTSDelayMixedGraphSampler(config).to(device)
    model = SimpleNamespace(
        config=config, device=device, sample_mixed_fixed_shape=sampler
    )
    graph = MossTTSDelayMixedSamplingCudaGraphRunner.capture(
        model=model, capture_bs=(args.batch,), disable_padding=False
    )

    batch = args.batch
    states = [[0, INT64_MAX, 0]] + [[N_VQ, index + 1, 1] for index in range(1, batch)]
    state = torch.tensor(states, dtype=torch.long, device=device)
    seeds = torch.arange(batch, device=device, dtype=torch.long) + 900000
    steps = torch.arange(batch, device=device, dtype=torch.long) + 30
    generator = torch.Generator(device=device).manual_seed(1234)
    text_logits = torch.randn(batch, TEXT_VOCAB, device=device, generator=generator)
    audio_logits = torch.randn(
        batch, N_VQ, AUDIO_VOCAB, device=device, generator=generator
    )
    dbatch = DelayGraphBatch(delay_state=state, seeds=seeds, generation_steps=steps)
    rows = torch.zeros(1, dtype=torch.long, device=device)

    full = measure(
        lambda: graph.replay(text_logits, audio_logits, dbatch),
        args.warmup,
        args.iters,
    )
    compacted = measure(
        lambda: graph.replay(text_logits, audio_logits, dbatch, text_rows=rows),
        args.warmup,
        args.iters,
    )
    result = {
        "device": "npu:0",
        "batch": batch,
        "text_vocab": TEXT_VOCAB,
        "n_vq": N_VQ,
        "audio_vocab": AUDIO_VOCAB,
        "full_replay": full,
        "compacted_replay": compacted,
        "saving_ms": round(full["median_ms"] - compacted["median_ms"], 3),
        "speedup": round(full["median_ms"] / compacted["median_ms"], 3),
        "note": "graph replay only; excludes backbone forward and e2e",
    }
    print(json.dumps(result, indent=1))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
