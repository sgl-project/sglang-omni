# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS Demo — sglang-omni generic AR engine integration.

Uses the standard ``create_ar_engine`` / ``ARRequestData`` path with a
``step_hook`` for TTS state propagation — the same engine code path that
Qwen3-Omni uses.

Usage:
    python examples/run_qwen3_tts_ar_demo.py \
        --model-id Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
        --text "Hello world, this is a test." \
        --speaker Ryan \
        --language auto \
        --output test_output_ar.wav
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import soundfile as sf
import torch


async def run_tts(args):
    """Run TTS through the sglang-omni generic AR engine."""

    # --- Load Talker component (only the Talker sub-model) ---
    print(f"Loading Talker from {args.model_id}...")
    from sglang_omni.models.qwen3_tts.qwen3_tts import Qwen3TTSTalkerComponent

    model = Qwen3TTSTalkerComponent(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
    )
    print(f"  Talker loaded on {args.device}")

    # --- Load speech tokenizer for decoding ---
    print("Loading speech tokenizer...")
    from transformers.utils.hub import cached_file

    from sglang_omni.models.qwen3_tts.qwen3_tts_tokenizer import Qwen3TTSSpeechTokenizer

    speech_tok_path = cached_file(
        args.model_id,
        "speech_tokenizer/config.json",
    )
    speech_tok_dir = os.path.dirname(speech_tok_path)
    speech_tokenizer = Qwen3TTSSpeechTokenizer.from_pretrained(
        speech_tok_dir,
        device_map=args.device,
    )
    print("  Speech tokenizer loaded")

    # --- Tokenize text ---
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    formatted = f"<|im_start|>assistant\n{args.text}<|im_end|>\n<|im_start|>assistant\n"
    encoded = tokenizer(formatted, return_tensors="pt")
    input_ids = encoded["input_ids"]
    print(f"  Tokenized: {input_ids.shape[1]} tokens")

    # --- Build prompt embeddings ---
    print("Building prompt embeddings...")
    from sglang_omni.models.qwen3_tts.qwen3_tts import build_tts_prompt

    prompt = build_tts_prompt(
        talker=model.talker,
        config=model.config,
        input_ids=input_ids,
        speaker=args.speaker,
        language=args.language,
        non_streaming_mode=args.non_streaming,
    )
    print(
        f"  Prompt: inputs_embeds={prompt['inputs_embeds'].shape}, "
        f"trailing_text_hidden={prompt['trailing_text_hidden'].shape}"
    )

    # --- Set seed after all loading is done (before generation) ---
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # --- Create generic AR engine and run generation ---
    print("Creating AR engine and generating...")
    from sglang_omni.engines.omni import create_ar_engine
    from sglang_omni.engines.omni.runtime.ar import ARRequestData
    from sglang_omni.models.qwen3_tts.qwen3_tts import make_tts_step_hook

    engine = create_ar_engine(
        model=model,
        max_seq_len=args.max_new_tokens + 512,
        device=args.device,
    )
    await engine.start()

    inputs_embeds = prompt["inputs_embeds"]

    req_data = ARRequestData(
        input_ids=torch.zeros(1, dtype=torch.long),
        inputs_embeds=inputs_embeds,
        attention_mask=prompt["attention_mask"],
        prefill_seq_len=inputs_embeds.shape[1],
        persistent_inputs={
            "trailing_text_hidden": prompt["trailing_text_hidden"],
            "tts_pad_embed": prompt["tts_pad_embed"],
            "generation_step": -1,
            "subtalker_dosample": True,
            "subtalker_top_k": args.top_k,
            "subtalker_top_p": args.top_p,
            "subtalker_temperature": 0.9,
            "output_hidden_states": False,
        },
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        suppress_tokens=prompt["suppress_tokens"],
        eos_token_ids=[prompt["codec_eos_token_id"]],
        step_hook=make_tts_step_hook(model),
    )

    t0 = time.perf_counter()
    await engine.add_request("tts-0", req_data)
    result = await engine.get_result("tts-0")
    gen_time = time.perf_counter() - t0
    await engine.stop()

    print(
        f"  Generated {len(result.output_ids)} tokens in {gen_time:.2f}s "
        f"({len(result.output_ids)/gen_time:.1f} tok/s)"
    )

    codec_ids_list = result.extra_model_outputs.get("all_codec_ids", [])
    print(f"  Codec frames accumulated: {len(codec_ids_list)}")

    # --- Stack codec IDs and trim at EOS ---
    if not codec_ids_list:
        print("No codec IDs generated.")
        sys.exit(1)

    codec_ids = torch.cat(codec_ids_list, dim=0)  # [T, num_code_groups]

    # Trim at first EOS in first codebook
    codec_eos = prompt["codec_eos_token_id"]
    first_cb = codec_ids[:, 0]
    eos_mask = first_cb == codec_eos
    if eos_mask.any():
        eos_idx = eos_mask.nonzero(as_tuple=True)[0][0].item()
        codec_ids = codec_ids[:eos_idx]
    print(f"  Final speech codes: {codec_ids.shape}")

    # --- Decode to audio ---
    print("Decoding to audio...")
    encoded_for_decode = {"audio_codes": codec_ids.to(args.device)}
    wavs, sample_rate = speech_tokenizer.decode(encoded_for_decode)

    if wavs:
        audio = wavs[0]
        sf.write(args.output, audio, sample_rate)
        duration = len(audio) / sample_rate
        print(f"Saved {duration:.2f}s audio to {args.output} (sr={sample_rate})")
    else:
        print("Decoding produced no audio.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS AR Engine Demo")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--speaker", type=str, default=None)
    parser.add_argument("--language", type=str, default="auto")
    parser.add_argument("--output", type=str, default="test_output_ar.wav")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--non-streaming",
        action="store_true",
        default=False,
        help="Use non-streaming mode (all text in prefill). Default is streaming.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    asyncio.run(run_tts(args))


if __name__ == "__main__":
    main()
