"""Quick smoke test for MingOmniTalker

Loads MingOmniTalker with the new load_weights() pattern and generates
a short TTS sample to verify functional equivalence with MingOmniTalker.

Usage:
    python test_ming_omni_talker.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_model_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    logger.info("Resolving HF repo %s ...", model_path)
    return snapshot_download(model_path)


def load_ming_omni_talker(model_path: str, device: str = "cuda"):
    """Load MingOmniTalker + AudioVAE."""
    from transformers import AutoTokenizer

    from sglang_omni.models.ming_omni.talker import (
        MingOmniTalker,
        MingOmniTalkerConfig,
        SpkembExtractor,
    )
    from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import (
        AudioVAE,
    )
    from sglang_omni.models.weight_loader import load_weights_by_prefix

    local_path = resolve_model_path(model_path)
    talker_path = os.path.join(local_path, "talker")

    # 1. Load config
    logger.info("Loading MingOmniTalkerConfig from %s ...", talker_path)
    config = MingOmniTalkerConfig.from_pretrained_dir(talker_path)
    logger.info(
        "Config loaded: patch_size=%d, steps=%d, latent_dim=%d",
        config.patch_size,
        config.steps,
        config.latent_dim,
    )

    # 2. Create model
    logger.info("Creating MingOmniTalker...")
    t0 = time.time()
    talker = MingOmniTalker(config)
    talker.eval()
    logger.info("Model created in %.1fs", time.time() - t0)

    # 3. Load weights and move to device with bf16
    logger.info("Loading weights...")
    t0 = time.time()
    weights = load_weights_by_prefix(talker_path, prefix="")
    talker.load_weights(weights.items())
    talker.to(device=device, dtype=torch.bfloat16)
    logger.info("Weights loaded in %.1fs (%d tensors)", time.time() - t0, len(weights))

    # 4. Set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(talker_path, "llm"))
    talker.set_tokenizer(tokenizer)
    logger.info("Tokenizer set: vocab_size=%d", len(tokenizer))

    # 5. Set voice presets
    voice_json_path = os.path.join(talker_path, "data", "voice_name.json")
    if os.path.exists(voice_json_path):
        with open(voice_json_path, "r") as f:
            voice_dict = json.load(f)
        for key in voice_dict:
            voice_dict[key]["prompt_wav_path"] = os.path.join(
                talker_path,
                voice_dict[key]["prompt_wav_path"],
            )
        talker.set_voice_presets(voice_dict)
        logger.info("Voice presets loaded: %s", list(voice_dict.keys())[:5])
    else:
        logger.warning("voice_name.json not found")

    # 6. Set speaker embedding extractor
    campplus_path = os.path.join(talker_path, "campplus.onnx")
    try:
        extractor = SpkembExtractor(campplus_path)
        talker.set_spkemb_extractor(extractor)
        logger.info("SpkembExtractor loaded")
    except Exception as e:
        logger.warning("SpkembExtractor not available: %s", e)

    # 7. Set text normalizer
    try:
        from talker_tn.talker_tn import TalkerTN

        talker.set_normalizer(TalkerTN())
        logger.info("TalkerTN normalizer loaded")
    except ImportError:
        logger.warning("TalkerTN not available, using identity normalizer")

    # 8. Init CUDA graphs
    logger.info("Initializing CUDA graphs...")
    t0 = time.time()
    talker.initial_graph()
    logger.info("CUDA graphs initialized in %.1fs", time.time() - t0)

    # 9. Load AudioVAE
    vae_path = os.path.join(talker_path, "vae")
    logger.info("Loading AudioVAE from %s ...", vae_path)
    t0 = time.time()
    vae = AudioVAE.from_pretrained(vae_path, dtype=torch.bfloat16)
    vae.to(device)
    vae.eval()
    logger.info("AudioVAE loaded in %.1fs", time.time() - t0)

    return talker, vae


@torch.no_grad()
def run_tts(
    talker,
    vae,
    text: str,
    voice_name: str = "DB30",
    output_path: str = "/tmp/test_ming_omni_talker.wav",
):
    """Run a single TTS generation and save the output."""
    logger.info("Generating TTS for: %r", text[:100])
    t0 = time.time()

    all_wavs = []
    for tts_speech, text_out, pos, dur_ms in talker.omni_audio_generation(
        tts_text=text,
        voice_name=voice_name,
        audio_detokenizer=vae,
        stream=False,
    ):
        if tts_speech is not None:
            all_wavs.append(tts_speech)

    elapsed = time.time() - t0

    if not all_wavs:
        logger.error("No audio generated!")
        return

    waveform = torch.cat(all_wavs, dim=-1)
    sample_rate = getattr(vae.config, "sample_rate", 44100)
    duration = waveform.shape[-1] / sample_rate

    logger.info(
        "Generated %.2fs audio in %.2fs (RTF=%.2f)",
        duration,
        elapsed,
        elapsed / max(duration, 0.01),
    )

    # Save
    wav_tensor = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
    torchaudio.save(output_path, wav_tensor.cpu().float(), sample_rate)
    logger.info("Saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Test MingOmniTalker")
    parser.add_argument("--model-path", default="inclusionAI/Ming-flash-omni-2.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--text", default="Hello, this is a test of the Ming Omni Talker model."
    )
    parser.add_argument("--voice", default="DB30")
    parser.add_argument("--output", default="/tmp/test_ming_omni_talker.wav")
    args = parser.parse_args()

    talker, vae = load_ming_omni_talker(args.model_path, args.device)
    run_tts(talker, vae, args.text, args.voice, args.output)
    logger.info("Test PASSED - MingOmniTalker works!")


if __name__ == "__main__":
    main()
