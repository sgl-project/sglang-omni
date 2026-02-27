#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal e2e script for FishAudio DualAR TTS via OmniEngine.

Usage:
    # Text-only (no voice cloning):
    python examples/run_fishaudio_e2e.py --text "Hello, how are you today?"

    # With reference audio (voice cloning):
    python examples/run_fishaudio_e2e.py --text "Hello" \
        --reference-audio ref.wav --reference-text "Reference transcript."

    # Save output as wav:
    python examples/run_fishaudio_e2e.py --text "Hello" --output output.wav

Nsys profiling:
    CUDA_VISIBLE_DEVICES=1 nsys profile -o fishaudio_e2e --trace cuda,nvtx,osrt \
        --force-overwrite true \
        python examples/run_fishaudio_e2e.py --text "Hello"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def resolve_checkpoint(checkpoint: str) -> str:
    """Resolve an HF model ID to a local snapshot path, or return as-is if local."""
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def load_model_and_tokenizer(checkpoint: str, device: str):
    """Load DualARTransformer and FishTokenizer from a checkpoint."""
    from fish_speech.models.text2semantic.llama import DualARTransformer
    from fish_speech.tokenizer import FishTokenizer

    checkpoint = resolve_checkpoint(checkpoint)
    logger.info("Loading model from %s ...", checkpoint)
    t0 = time.perf_counter()
    model = DualARTransformer.from_pretrained(checkpoint, load_weights=True)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    t1 = time.perf_counter()
    logger.info("Model loaded in %.2fs", t1 - t0)

    tokenizer = FishTokenizer.from_pretrained(checkpoint)
    return model, tokenizer, checkpoint


def load_codec(checkpoint_dir: str, device: str):
    """Load the DAC codec from codec.pth inside the checkpoint directory."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    codec_path = os.path.join(checkpoint_dir, "codec.pth")
    logger.info("Loading codec from %s ...", codec_path)
    t0 = time.perf_counter()

    # Load config yaml directly instead of going through hydra.initialize
    import fish_speech.models.dac.modded_dac as _dac_mod

    configs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(_dac_mod.__file__))),
        "configs",
    )
    cfg = OmegaConf.load(os.path.join(configs_dir, "modded_dac_vq.yaml"))
    model = instantiate(cfg)

    state_dict = torch.load(
        codec_path, map_location=device, mmap=True, weights_only=True
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }
    model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    model.to(device)

    t1 = time.perf_counter()
    logger.info("Codec loaded in %.2fs", t1 - t0)
    return model


def encode_reference_audio(codec, audio_path: str, device: str) -> torch.Tensor:
    """Encode a wav file into VQ codes via the DAC codec."""
    import torchaudio

    audio, sr = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    audio = torchaudio.functional.resample(audio, sr, codec.sample_rate)

    audios = audio[None].to(device)
    audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
    logger.info(
        "Reference audio: %.2fs", audios.shape[2] / codec.sample_rate
    )

    with torch.no_grad():
        indices, _ = codec.encode(audios, audio_lengths)
        if indices.ndim == 3:
            indices = indices[0]

    logger.info("Encoded reference VQ codes: %s", indices.shape)
    return indices.cpu()


def decode_output_to_wav(codec, output_codes: torch.Tensor, output_path: str):
    """Decode VQ codes back to audio and save as wav."""
    import soundfile as sf

    # output_codes: [num_codebooks+1, T] — rows 1..N are codebook indices
    codebook_codes = output_codes[1:]  # [num_codebooks, T]
    feature_lengths = torch.tensor(
        [codebook_codes.shape[1]], device=codebook_codes.device
    )

    with torch.no_grad():
        audio, _ = codec.decode(codebook_codes[None], feature_lengths)

    audio_np = audio[0, 0].float().cpu().numpy()
    sf.write(output_path, audio_np, codec.sample_rate)
    logger.info("Saved output audio to %s (%.2fs)", output_path, len(audio_np) / codec.sample_rate)


async def run_e2e(args):
    from sglang_omni.models.fishaudio_s1 import (
        DualARRequestData,
        FishTokenizerAdapter,
        Reference,
        create_dual_ar_engine,
    )

    model, tokenizer, checkpoint_dir = load_model_and_tokenizer(
        args.checkpoint, args.device
    )
    adapter = FishTokenizerAdapter(tokenizer)

    # Use num_codebooks from model config
    num_codebooks = model.config.num_codebooks
    codebook_size = model.config.codebook_size
    logger.info(
        "Model config: num_codebooks=%d, codebook_size=%d",
        num_codebooks, codebook_size,
    )

    # Load codec if we need reference encoding or wav output
    codec = None
    if args.reference_audio or args.output:
        codec = load_codec(checkpoint_dir, args.device)

    # Build references
    references = None
    if args.reference_audio:
        vq_codes = encode_reference_audio(codec, args.reference_audio, args.device)
        ref = Reference(
            audio_bytes=b"",
            text=args.reference_text,
            vq_codes=vq_codes,
        )
        references = [ref]

    engine = create_dual_ar_engine(
        model=model,
        tokenizer=tokenizer,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        use_radix_cache=args.use_radix_cache,
        use_compile=args.compile,
    )
    await engine.start()

    try:
        if args.test_cache:
            await _run_cache_test(
                engine, adapter, references, num_codebooks, args
            )
        else:
            await _run_single_request(
                engine, adapter, references, num_codebooks, codec, args
            )
    finally:
        # Print final cache stats if radix cache is enabled
        if engine.radix_cache is not None:
            stats = engine.radix_cache.stats()
            logger.info("=== Radix Cache Stats ===")
            for k, v in stats.items():
                logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)
        await engine.stop()


async def _run_single_request(engine, adapter, references, num_codebooks, codec, args):
    """Run a single TTS request."""
    from sglang_omni.models.fishaudio_s1 import DualARRequestData

    input_values, audio_masks, audio_parts = adapter.build_prompt(
        text=args.text,
        references=references,
        num_codebooks=num_codebooks,
    )
    logger.info(
        "Prompt shape: %s (seq_len=%d)",
        input_values.shape, input_values.shape[1],
    )

    data = DualARRequestData(
        input_values=input_values,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        max_new_tokens=args.max_new_tokens,
        num_codebooks=num_codebooks,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    request_id = "req-0"
    logger.info("Submitting request '%s' ...", request_id)
    t0 = time.perf_counter()

    await engine.add_request(request_id, data)
    result = await engine.get_result(request_id)

    t1 = time.perf_counter()

    all_codes = torch.cat(result.output_codes, dim=1)
    num_steps = all_codes.shape[1]
    elapsed = t1 - t0
    logger.info(
        "Generated %d steps in %.3fs (%.1f steps/s)",
        num_steps, elapsed,
        num_steps / elapsed if elapsed > 0 else 0,
    )
    logger.info("Output codes shape: %s", all_codes.shape)
    logger.info(
        "Semantic tokens (row 0, first 20): %s", all_codes[0, :20].tolist()
    )

    if args.output and codec is not None:
        decode_output_to_wav(codec, all_codes, args.output)


async def _run_cache_test(engine, adapter, references, num_codebooks, args):
    """Send two requests with same voice ref but different text to test cache.

    Expected: request 2 should hit the radix cache on the shared voice
    reference prefix, producing a non-zero hit rate.
    """
    from sglang_omni.models.fishaudio_s1 import DualARRequestData

    texts = [
        "This is the first test sentence for cache validation.",
        "This is a completely different sentence to verify cache reuse.",
    ]

    for i, text in enumerate(texts):
        input_values, audio_masks, audio_parts = adapter.build_prompt(
            text=text,
            references=references,
            num_codebooks=num_codebooks,
        )
        logger.info(
            "Request %d: text=%r, prompt shape=%s (seq_len=%d)",
            i, text, input_values.shape, input_values.shape[1],
        )

        data = DualARRequestData(
            input_values=input_values,
            audio_masks=audio_masks,
            audio_parts=audio_parts,
            max_new_tokens=args.max_new_tokens,
            num_codebooks=num_codebooks,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        request_id = f"cache-test-{i}"
        t0 = time.perf_counter()

        await engine.add_request(request_id, data)
        result = await engine.get_result(request_id)

        t1 = time.perf_counter()

        all_codes = torch.cat(result.output_codes, dim=1)
        logger.info(
            "Request %d: generated %d steps in %.3fs",
            i, all_codes.shape[1], t1 - t0,
        )

        # Print cache stats after each request
        if engine.radix_cache is not None:
            stats = engine.radix_cache.stats()
            logger.info(
                "  After request %d — hits=%d, misses=%d, hit_rate=%.2f, "
                "token_hit_rate=%.2f, cached_tokens=%d",
                i,
                stats["num_matches"],
                stats["num_misses"],
                stats["hit_rate"],
                stats["token_hit_rate"],
                stats["total_cached_tokens"],
            )

    # Final verdict
    if engine.radix_cache is not None:
        stats = engine.radix_cache.stats()
        if stats["num_matches"] > 0:
            logger.info(
                "PASS: Cache hit detected! %d hits, token_hit_rate=%.2f",
                stats["num_matches"], stats["token_hit_rate"],
            )
        else:
            logger.warning(
                "FAIL: No cache hits detected after 2 requests with same voice ref. "
                "This may indicate a radix cache bug."
            )
    else:
        logger.warning("Cache test requires --use-radix-cache to be enabled.")


def main():
    parser = argparse.ArgumentParser(description="FishAudio DualAR e2e test")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="fishaudio/openaudio-s1-mini",
        help="HF model ID or local path",
    )
    parser.add_argument("--text", type=str, default="Hello, how are you today?")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument(
        "--reference-audio",
        type=str,
        default=None,
        help="Path to reference wav for voice cloning",
    )
    parser.add_argument(
        "--reference-text",
        type=str,
        default="",
        help="Transcript of the reference audio",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save output wav (requires codec)",
    )
    parser.add_argument(
        "--use-radix-cache",
        action="store_true",
        default=False,
        help="Enable radix-tree prefix cache for voice ref reuse",
    )
    parser.add_argument(
        "--test-cache",
        action="store_true",
        default=False,
        help="Run cache correctness test: 2 requests with same voice ref, different text",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Use torch.compile(mode='reduce-overhead') for decode steps",
    )
    args = parser.parse_args()

    asyncio.run(run_e2e(args))


if __name__ == "__main__":
    main()
