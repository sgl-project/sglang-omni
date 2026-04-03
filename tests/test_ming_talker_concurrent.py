"""Test concurrent TTS requests on MingOmniTalker.

Verifies that the talker can handle multiple concurrent requests
(currently expected to serialize due to max_conc=1).

Usage:
    python tests/test_ming_talker_concurrent.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --device cuda:0 --num-requests 3
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TEST_TEXTS = [
    "Hello, this is the first concurrent request.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Testing parallel speech generation capabilities.",
    "One two three four five six seven eight nine ten.",
]


def resolve_model_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    logger.info("Resolving HF repo %s ...", model_path)
    return snapshot_download(model_path)


def load_talker(model_path: str, device: str = "cuda", max_conc: int = 1):
    """Load MingOmniTalker + AudioVAE."""
    from transformers import AutoTokenizer

    from sglang_omni.models.ming_omni.talker import (
        AudioVAE,
        MingOmniTalker,
        MingOmniTalkerConfig,
        SpkembExtractor,
    )
    from sglang_omni.models.weight_loader import load_weights_by_prefix

    local_path = resolve_model_path(model_path)
    talker_path = os.path.join(local_path, "talker")

    config = MingOmniTalkerConfig.from_pretrained_dir(talker_path)
    config.max_conc = max_conc
    logger.info("max_conc=%d", config.max_conc)
    talker = MingOmniTalker(config)
    talker.eval()
    weights = load_weights_by_prefix(talker_path, prefix="")
    talker.load_weights(weights.items())
    talker.to(device=device, dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(talker_path, "llm"))
    talker.set_tokenizer(tokenizer)

    campplus_path = os.path.join(talker_path, "campplus.onnx")
    try:
        talker.set_spkemb_extractor(SpkembExtractor(campplus_path))
    except Exception:
        pass

    try:
        from talker_tn.talker_tn import TalkerTN

        talker.set_normalizer(TalkerTN())
    except ImportError:
        pass

    talker.initial_graph()

    vae_path = os.path.join(talker_path, "vae")
    vae = AudioVAE.from_pretrained(vae_path, dtype=torch.bfloat16)
    vae.to(device)
    vae.eval()

    return talker, vae


@torch.no_grad()
def generate_one(talker, vae, text: str, request_id: int) -> dict:
    """Generate speech for a single request. Returns timing info."""
    t0 = time.time()
    logger.info("[Request %d] START: %r", request_id, text[:60])

    all_wavs = []
    for tts_speech, _, _, _ in talker.omni_audio_generation(
        tts_text=text,
        voice_name=None,
        audio_detokenizer=vae,
        stream=False,
    ):
        if tts_speech is not None:
            all_wavs.append(tts_speech)

    elapsed = time.time() - t0

    if not all_wavs:
        logger.error("[Request %d] FAILED: no audio generated", request_id)
        return {
            "request_id": request_id,
            "text": text,
            "success": False,
            "elapsed": elapsed,
            "duration": 0.0,
        }

    waveform = torch.cat(all_wavs, dim=-1)
    sample_rate = getattr(vae.config, "sample_rate", 44100)
    duration = waveform.shape[-1] / sample_rate

    logger.info(
        "[Request %d] DONE: %.2fs audio in %.2fs (RTF=%.2f)",
        request_id,
        duration,
        elapsed,
        elapsed / max(duration, 0.01),
    )

    return {
        "request_id": request_id,
        "text": text,
        "success": True,
        "elapsed": round(elapsed, 3),
        "duration": round(duration, 3),
        "samples": waveform.shape[-1],
    }


def run_sequential(talker, vae, texts: list[str]) -> list[dict]:
    """Run requests one by one (baseline)."""
    results = []
    for i, text in enumerate(texts):
        results.append(generate_one(talker, vae, text, i))
    return results


def run_concurrent_threads(talker, vae, texts: list[str]) -> list[dict]:
    """Run requests concurrently using threads."""
    with ThreadPoolExecutor(max_workers=len(texts)) as pool:
        futures = [
            pool.submit(generate_one, talker, vae, text, i)
            for i, text in enumerate(texts)
        ]
        return [f.result() for f in futures]


def print_results(label: str, results: list[dict], wall_time: float) -> None:
    w = 60
    print(f"\n{'=' * w}")
    print(f"{label:^{w}}")
    print(f"{'=' * w}")
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        print(
            f"  [{status}] Request {r['request_id']}: "
            f"gen={r['elapsed']:.2f}s, audio={r['duration']:.2f}s"
        )
    print(f"{'-' * w}")
    total_gen = sum(r["elapsed"] for r in results)
    total_audio = sum(r["duration"] for r in results)
    successes = sum(1 for r in results if r["success"])
    print(f"  {'Requests:':<25} {successes}/{len(results)} succeeded")
    print(f"  {'Wall-clock time:':<25} {wall_time:.2f}s")
    print(f"  {'Sum of gen times:':<25} {total_gen:.2f}s")
    print(f"  {'Total audio generated:':<25} {total_audio:.2f}s")
    if wall_time > 0 and total_gen > 0:
        parallelism = total_gen / wall_time
        print(f"  {'Effective parallelism:':<25} {parallelism:.2f}x")
        if parallelism > 1.5:
            print(f"  {'Verdict:':<25} PARALLEL")
        else:
            print(f"  {'Verdict:':<25} SERIAL (expected, max_conc=1)")
    print(f"{'=' * w}")


def main():
    parser = argparse.ArgumentParser(
        description="Test concurrent TTS on MingOmniTalker"
    )
    parser.add_argument("--model-path", default="inclusionAI/Ming-flash-omni-2.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--num-requests",
        type=int,
        default=3,
        help="Number of concurrent requests to test.",
    )
    parser.add_argument(
        "--max-conc",
        type=int,
        default=1,
        help="Max concurrency for talker (CUDA graph pool size).",
    )
    args = parser.parse_args()

    texts = TEST_TEXTS[: args.num_requests]
    logger.info(
        "Testing with %d requests, max_conc=%d on %s",
        len(texts),
        args.max_conc,
        args.device,
    )

    talker, vae = load_talker(args.model_path, args.device, max_conc=args.max_conc)

    # Warmup
    logger.info("Warmup run...")
    generate_one(talker, vae, "Warmup sentence.", -1)

    # Sequential baseline
    logger.info("Running %d requests SEQUENTIALLY...", len(texts))
    t0 = time.time()
    seq_results = run_sequential(talker, vae, texts)
    seq_wall = time.time() - t0
    print_results("Sequential (Baseline)", seq_results, seq_wall)

    # Concurrent threads
    logger.info("Running %d requests CONCURRENTLY (threads)...", len(texts))
    t0 = time.time()
    conc_results = run_concurrent_threads(talker, vae, texts)
    conc_wall = time.time() - t0
    print_results("Concurrent (Threads)", conc_results, conc_wall)

    # Summary comparison
    print(f"\n{'=' * 60}")
    print(f"{'COMPARISON':^60}")
    print(f"{'=' * 60}")
    print(f"  Sequential wall time:  {seq_wall:.2f}s")
    print(f"  Concurrent wall time:  {conc_wall:.2f}s")
    if seq_wall > 0:
        speedup = seq_wall / conc_wall
        print(f"  Speedup:               {speedup:.2f}x")
        if speedup > 1.5:
            print(f"  Result:                Parallel execution detected!")
        else:
            print(
                f"  Result:                Serial execution (expected, talker max_conc=1)"
            )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
