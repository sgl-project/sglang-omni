"""
Unified TTS accuracy (WER) benchmark using the seed-tts-eval dataset.

Supports multiple backends:

* ``api``         — Calls an HTTP server's ``/v1/audio/speech`` endpoint
                    (e.g. Voxtral-4B-TTS served via ``sgl-omni serve``).
                    No voice-cloning; uses ``--voice`` for speaker selection.
* ``local-ming``  — Loads MingOmniTalker + AudioVAE locally on GPU and
                    generates speech with voice-cloning from the reference
                    audio/text in each seed-tts-eval sample.

Evaluation pipeline (common to all backends):
1. Generate speech for the target text
2. Transcribe the generated audio with Whisper (EN) or FunASR (ZH)
3. Compute Word Error Rate against the original text

Dataset
-------
    huggingface-cli download zhaochenyang20/seed-tts-eval \\
        --repo-type dataset --local-dir seedtts_testset

Usage examples
--------------
    # ---- API backend (e.g. Voxtral) ----
    # 1. Start the server
    sgl-omni serve --model-path mistralai/Voxtral-4B-TTS-2603 --port 8000

    # 2. Run benchmark
    python benchmarks/accuracy/tts/benchmark_tts_wer.py \\
        --backend api \\
        --meta seedtts_testset/en/meta.lst \\
        --output-dir results/voxtral_en \\
        --lang en --max-samples 10 \\
        --voice cheerful_female --port 8000

    # ---- Local Ming backend ----
    python benchmarks/accuracy/tts/benchmark_tts_wer.py \\
        --backend local-ming \\
        --meta seedtts_testset/zh/meta.lst \\
        --model-path inclusionAI/Ming-flash-omni-2.0 \\
        --output-dir results/ming_zh \\
        --lang zh --device cuda:0 --max-samples 50
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

import numpy as np
import requests
from tqdm import tqdm
from tts_wer_utils import (
    DEFAULT_DATASET_DIR,
    DEFAULT_WHISPER_EN,
    DEFAULT_WHISPER_ZH,
    SUMMARY_LABEL_WIDTH,
    SUMMARY_LINE_WIDTH,
    WAV_HEADER_SIZE,
    SampleInput,
    SampleOutput,
    calculate_metrics,
    compute_wer,
    get_wav_duration,
    load_asr_model,
    parse_meta_lst,
    print_wer_metrics,
    save_results,
    transcribe,
    wait_for_service,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend: api  (e.g. Voxtral TTS served via sgl-omni)
# ---------------------------------------------------------------------------


def generate_speech_via_api(
    text: str,
    base_url: str,
    model_name: str,
    voice: str = "cheerful_female",
    max_new_tokens: int = 4096,
    timeout: int = 120,
) -> tuple[bytes, float]:
    """Call ``/v1/audio/speech`` and return *(wav_bytes, latency_s)*.

    Voxtral TTS does not use ref_audio/ref_text (no voice cloning).
    It only needs text + voice name.
    """
    api_url = f"{base_url}/v1/audio/speech"
    payload = {
        "model": model_name,
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "max_new_tokens": max_new_tokens,
    }

    t0 = time.perf_counter()
    response = requests.post(api_url, json=payload, timeout=timeout)
    latency = time.perf_counter() - t0

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")

    wav_bytes = response.content
    if len(wav_bytes) <= WAV_HEADER_SIZE:
        raise ValueError(f"Empty or invalid audio response ({len(wav_bytes)} bytes)")

    return wav_bytes, latency


# ---------------------------------------------------------------------------
# Backend: local-ming  (MingOmniTalker loaded on GPU)
# ---------------------------------------------------------------------------


def _resolve_model_path(model_path: str) -> str:
    """Resolve a HF repo ID or local path to a local directory."""
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    logger.info("Resolving HF repo %s to local cache...", model_path)
    return snapshot_download(model_path)


def load_ming_talker(model_path: str, device: str = "cuda"):
    """Load MingOmniTalker + AudioVAE."""
    import torch
    from transformers import AutoTokenizer

    from sglang_omni.models.ming_omni.talker import (
        AudioVAE,
        MingOmniTalker,
        MingOmniTalkerConfig,
        SpkembExtractor,
    )
    from sglang_omni.models.weight_loader import load_weights_by_prefix

    local_path = _resolve_model_path(model_path)
    talker_path = os.path.join(local_path, "talker")

    logger.info("Loading MingOmniTalker from %s ...", talker_path)
    t0 = time.time()
    config = MingOmniTalkerConfig.from_pretrained_dir(talker_path)
    talker = MingOmniTalker(config)
    talker.eval()
    weights = load_weights_by_prefix(talker_path, prefix="")
    talker.load_weights(weights.items())
    talker.to(device=device, dtype=torch.bfloat16)
    logger.info("MingOmniTalker loaded in %.1fs", time.time() - t0)

    # Set external dependencies
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(talker_path, "llm"))
    talker.set_tokenizer(tokenizer)

    voice_json_path = os.path.join(talker_path, "data", "voice_name.json")
    if os.path.exists(voice_json_path):
        import json as _json

        with open(voice_json_path, "r") as f:
            voice_dict = _json.load(f)
        for key in voice_dict:
            voice_dict[key]["prompt_wav_path"] = os.path.join(
                talker_path, voice_dict[key]["prompt_wav_path"]
            )
        talker.set_voice_presets(voice_dict)

    campplus_path = os.path.join(talker_path, "campplus.onnx")
    try:
        talker.set_spkemb_extractor(SpkembExtractor(campplus_path))
    except Exception as e:
        logger.warning("SpkembExtractor not available: %s", e)

    try:
        from talker_tn.talker_tn import TalkerTN

        talker.set_normalizer(TalkerTN())
    except ImportError:
        logger.warning("TalkerTN not available, using identity normalizer")

    logger.info("Initializing CUDA graphs...")
    t0g = time.time()
    talker.initial_graph()
    logger.info("CUDA graphs initialized in %.1fs", time.time() - t0g)

    vae_path = os.path.join(talker_path, "vae")
    logger.info("Loading AudioVAE from %s ...", vae_path)
    t0v = time.time()
    vae = AudioVAE.from_pretrained(vae_path, dtype=torch.bfloat16)
    vae.to(device)
    vae.eval()
    logger.info("AudioVAE loaded in %.1fs", time.time() - t0v)

    return talker, vae


def generate_speech_ming(
    talker: Any,
    vae: Any,
    sample: SampleInput,
) -> tuple[np.ndarray | None, int]:
    """Generate speech using MingOmniTalker with voice cloning.

    Returns *(waveform_float32, sample_rate)*.
    """
    import torch

    all_wavs = []
    with torch.no_grad():
        for tts_speech, _, _, _ in talker.omni_audio_generation(
            tts_text=sample.target_text,
            voice_name=None,
            prompt_text=sample.ref_text,
            prompt_wav_path=sample.ref_audio,
            audio_detokenizer=vae,
            stream=False,
        ):
            if tts_speech is not None:
                all_wavs.append(tts_speech)

    if not all_wavs:
        return None, 44100

    waveform = torch.cat(all_wavs, dim=-1)
    sample_rate = getattr(vae.config, "sample_rate", 44100)
    return waveform.squeeze().cpu().float().numpy(), sample_rate


# ---------------------------------------------------------------------------
# Summary printing (parameterised header)
# ---------------------------------------------------------------------------


def print_summary(metrics: dict, args: argparse.Namespace) -> None:
    lw = SUMMARY_LABEL_WIDTH
    w = SUMMARY_LINE_WIDTH
    print(f"\n{'=' * w}")

    if args.backend == "api":
        print(f"{'API TTS WER Benchmark Result':^{w}}")
        print(f"{'=' * w}")
        print(f"  {'Model:':<{lw}} {args.model}")
        print(f"  {'Voice:':<{lw}} {args.voice}")
    elif args.backend == "local-ming":
        print(f"{'Ming TTS WER Benchmark Result':^{w}}")
        print(f"{'=' * w}")
        print(f"  {'Model:':<{lw}} {args.model_path}")
    else:
        print(f"{'TTS WER Benchmark Result':^{w}}")
        print(f"{'=' * w}")

    print(f"  {'Backend:':<{lw}} {args.backend}")
    print(f"  {'Language:':<{lw}} {args.lang}")
    asr_label = args.asr_model or (
        DEFAULT_WHISPER_ZH if args.lang == "zh" else DEFAULT_WHISPER_EN
    )
    print(f"  {'ASR model:':<{lw}} {asr_label}")
    print(f"  {'Completed samples:':<{lw}} {metrics['completed']}")
    print(f"  {'Failed samples:':<{lw}} {metrics['failed']}")
    print_wer_metrics(metrics)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def _build_config_dict(args: argparse.Namespace) -> dict:
    """Return the config dict to persist alongside results."""
    cfg: dict = {
        "backend": args.backend,
        "meta": args.meta,
        "lang": args.lang,
        "max_samples": args.max_samples,
        "asr_device": args.asr_device,
        "asr_model": args.asr_model,
    }
    if args.backend == "api":
        cfg.update(
            {
                "model": args.model,
                "voice": args.voice,
                "base_url": args.base_url or f"http://{args.host}:{args.port}",
                "max_new_tokens": args.max_new_tokens,
            }
        )
    elif args.backend == "local-ming":
        cfg.update(
            {
                "model_path": args.model_path,
                "device": args.device,
            }
        )
    return cfg


def benchmark(args: argparse.Namespace) -> None:
    # Resolve meta path from --dataset-dir and --lang if not explicitly set
    if args.meta is None:
        args.meta = os.path.join(args.dataset_dir, args.lang, "meta.lst")
        logger.info("Resolved meta path: %s", args.meta)

    if not os.path.isfile(args.meta):
        logger.error("Meta file not found: %s", args.meta)
        return

    samples = parse_meta_lst(args.meta, args.max_samples)
    logger.info("Loaded %d samples from %s", len(samples), args.meta)

    # ---- backend-specific setup ----
    talker = vae = None
    base_url: str | None = None

    if args.backend == "api":
        base_url = args.base_url or f"http://{args.host}:{args.port}"
        wait_for_service(base_url)
    elif args.backend == "local-ming":
        talker, vae = load_ming_talker(args.model_path, device=args.device)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # Pre-load ASR model
    load_asr_model(args.lang, device=args.asr_device, whisper_model=args.asr_model)

    # Create audio output dir
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    outputs: list[SampleOutput] = []
    for sample in tqdm(samples, desc="Generating & evaluating"):
        output = SampleOutput(
            sample_id=sample.sample_id,
            target_text=sample.target_text,
        )
        audio_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")

        try:
            # ---- generate speech (backend-specific) ----
            if args.backend == "api":
                assert base_url is not None
                wav_bytes, latency = generate_speech_via_api(
                    text=sample.target_text,
                    base_url=base_url,
                    model_name=args.model,
                    voice=args.voice,
                    max_new_tokens=args.max_new_tokens,
                )
                output.latency = latency
                output.audio_duration = get_wav_duration(wav_bytes)

                with open(audio_path, "wb") as f:
                    f.write(wav_bytes)

            elif args.backend == "local-ming":
                import torch
                import torchaudio

                t0 = time.time()
                waveform, sample_rate = generate_speech_ming(talker, vae, sample)
                output.latency = time.time() - t0

                if waveform is None or len(waveform) == 0:
                    output.error = "Empty waveform"
                    outputs.append(output)
                    continue

                output.audio_duration = len(waveform) / sample_rate

                waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
                torchaudio.save(audio_path, waveform_tensor, sample_rate)

            # ---- ASR transcribe ----
            logger.info("[ASR] Transcribing %s on %s", audio_path, args.asr_device)
            hypothesis = transcribe(
                audio_path,
                args.lang,
                asr_device=args.asr_device,
                whisper_model=args.asr_model,
            )
            output.hypothesis = hypothesis

            # ---- WER ----
            output.wer = compute_wer(sample.target_text, hypothesis, args.lang)
            output.is_success = True

            logger.info(
                "[%s] WER=%.4f | ref=%r | hyp=%r",
                sample.sample_id,
                output.wer,
                sample.target_text[:80],
                hypothesis[:80],
            )
        except Exception as e:
            output.error = str(e)
            logger.error("Error on sample %s: %s", sample.sample_id, e, exc_info=True)

        outputs.append(output)

    # ---- report ----
    metrics = calculate_metrics(outputs)
    print_summary(metrics, args)
    save_results(
        outputs,
        metrics,
        output_dir=args.output_dir,
        config=_build_config_dict(args),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TTS accuracy (WER) using seed-tts-eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- backend selection ----
    parser.add_argument(
        "--backend",
        type=str,
        default="api",
        choices=["api", "local-ming"],
        help=(
            "TTS backend to use. "
            "'api' calls an HTTP /v1/audio/speech endpoint (e.g. Voxtral). "
            "'local-ming' loads MingOmniTalker locally on GPU."
        ),
    )

    # ---- common args ----
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help=(
            "Path to seed-tts-eval meta.lst file. "
            "If not set, resolved from --dataset-dir and --lang."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=(
            "Path to seed-tts-eval dataset root (contains zh/ and en/ subdirs). "
            f"Default: {DEFAULT_DATASET_DIR}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tts_wer",
        help="Directory to save results and audio files.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["zh", "en"],
        help="Language for ASR and text normalization.",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="cpu",
        help="Device for ASR model (cpu recommended to save GPU memory).",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default=None,
        help=(
            "Whisper model size for ASR (e.g. tiny, base.en, small, medium, "
            "large-v3). Default: base.en for EN, medium for ZH."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process.",
    )

    # ---- api backend args ----
    api_group = parser.add_argument_group(
        "api backend",
        "Options for --backend api (HTTP-based TTS, e.g. Voxtral).",
    )
    api_group.add_argument(
        "--model",
        type=str,
        default="Voxtral-4B-TTS-2603",
        help="Model name for the API request.",
    )
    api_group.add_argument(
        "--voice",
        type=str,
        default="cheerful_female",
        help="Voice to use for synthesis (e.g. cheerful_female).",
    )
    api_group.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host.",
    )
    api_group.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port.",
    )
    api_group.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (e.g. http://localhost:8000). Overrides --host/--port.",
    )
    api_group.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens for AR generation.",
    )

    # ---- local-ming backend args ----
    ming_group = parser.add_argument_group(
        "local-ming backend",
        "Options for --backend local-ming (local MingOmniTalker inference).",
    )
    ming_group.add_argument(
        "--model-path",
        type=str,
        default="inclusionAI/Ming-flash-omni-2.0",
        help="Path or HF repo ID for the Ming-flash-omni model.",
    )
    ming_group.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for TTS model (local-ming backend).",
    )

    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
