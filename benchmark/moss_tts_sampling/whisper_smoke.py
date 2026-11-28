"""Smoke-test the Whisper-large-v3 FP32 scorer on the NPU with two clips."""

import argparse
import json
import time

import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper_normalizer.english import EnglishTextNormalizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as manifest:
        rows = json.load(manifest)[:2]
    proc = WhisperProcessor.from_pretrained(
        args.model_path, language="en", task="transcribe"
    )
    t0 = time.perf_counter()
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    print("load_s", round(time.perf_counter() - t0, 2), flush=True)

    auds = []
    for r in rows:
        a, sr = sf.read(r["ref_wav"], dtype="float32", always_2d=False)
        if sr != 16000:
            a = resample_poly(a, 16000, sr).astype("float32")
        auds.append(a)
    feats = proc.feature_extractor(
        auds, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(args.device, dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.inference_mode():
        gen = model.generate(
            feats,
            language="en",
            task="transcribe",
            do_sample=False,
            num_beams=1,
            max_new_tokens=444,
        )
    torch.npu.synchronize()
    print("gen_s", round(time.perf_counter() - t0, 2), flush=True)
    norm = EnglishTextNormalizer()
    for r, t in zip(rows, proc.batch_decode(gen, skip_special_tokens=True)):
        print("REF:", r["ref_text"][:90])
        print("ASR:", t[:90])
        print("NRM:", norm(r["ref_text"]) == norm(t))


if __name__ == "__main__":
    main()
