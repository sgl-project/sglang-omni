#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test a running Higgs TTS HTTP server on either Torch MPS or MLX.

Run manually, not during unit tests. WAV checks are structural; listen to the
saved clips or transcribe them separately to assess speech quality. Backend
selection belongs to the server; this client does not select an accelerator.
"""

import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf


def check_stream(url, body, *, timeout, output, abort=False):
    """Inspect raw PCM and optionally disconnect after the first received chunk."""
    start = time.perf_counter()
    chunks = []
    arrivals = []
    with requests.post(
        url,
        json={**body, "stream": True, "response_format": "pcm"},
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        assert response.headers["content-type"].split(";")[0] == "audio/pcm"
        assert int(response.headers["x-sample-rate"]) == 24000
        assert int(response.headers["x-channels"]) == 1
        assert int(response.headers["x-bit-depth"]) == 16
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                chunks.append(chunk)
                arrivals.append(time.perf_counter() - start)
                if abort:
                    break
    assert chunks, "Stream returned no audio"
    raw = b"".join(chunks)
    assert len(raw) % 2 == 0, "Incomplete PCM sample"
    samples = np.frombuffer(raw, dtype="<i2")
    rms = float(np.sqrt(np.mean((samples.astype(np.float64) / 32768) ** 2)))
    assert rms > 1e-5, "Stream is effectively silent"
    if not abort:
        assert len(chunks) > 1, "Expected incremental audio delivery"
    sf.write(output, samples, 24000, subtype="PCM_16")
    return {
        "file": output.name,
        "received_chunks": len(chunks),
        "chunk_arrival_seconds": arrivals,
        "first_audio_seconds": arrivals[0],
        "wall_seconds": time.perf_counter() - start,
        "seconds": len(samples) / 24000,
        "rms": rms,
        "pcm_sha256": hashlib.sha256(raw).hexdigest(),
        "disconnected_after_first_chunk": abort,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="bosonai/higgs-audio-v3-tts-4b")
    parser.add_argument(
        "--output", type=Path, default=Path("notes/artifacts/higgs-mps-e2e")
    )
    parser.add_argument(
        "--text", default="Hello, this is a test of speech generation on my Mac."
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--reference-audio", type=Path, help="Reference WAV to upload")
    parser.add_argument("--reference-text", help="Matching reference transcript")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Check PCM streaming and disconnect recovery",
    )
    args = parser.parse_args()
    if bool(args.reference_audio) != bool(args.reference_text):
        parser.error("--reference-audio and --reference-text must be supplied together")
    args.output.mkdir(parents=True, exist_ok=True)
    url = args.url.rstrip("/") + "/v1/audio/speech"
    report = {
        "server": args.url,
        "model": args.model,
        "text": args.text,
        "checks": [],
        "quality": "manual listening or transcription required",
    }
    body = dict(
        model=args.model,
        voice="default",
        input=args.text,
        response_format="wav",
        seed=42,
        temperature=0.8,
        top_k=50,
        max_new_tokens=512,
    )
    try:
        previous = None
        for number in (1, 2):
            start = time.perf_counter()
            response = requests.post(url, json=body, timeout=args.timeout)
            response.raise_for_status()
            elapsed = time.perf_counter() - start
            samples, rate = sf.read(io.BytesIO(response.content), dtype="float32")
            assert rate == 24000, f"Unexpected sample rate: {rate}"
            assert samples.ndim == 1, f"Expected mono audio, got {samples.shape}"
            assert (
                len(samples) > 0 and np.isfinite(samples).all()
            ), "Empty or nonfinite audio"
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
            assert rms > 1e-5, "Audio is effectively silent"
            path = args.output / f"speech-{number}.wav"
            path.write_bytes(response.content)
            record = dict(
                file=path.name,
                sample_rate=rate,
                seconds=len(samples) / rate,
                wall_seconds=elapsed,
                rtf=elapsed / (len(samples) / rate),
                rms=rms,
                peak=float(np.max(np.abs(samples))),
                sha256=hashlib.sha256(response.content).hexdigest(),
            )
            if previous is not None:
                record["same_seed_equal_samples"] = bool(
                    np.array_equal(previous, samples)
                )
                assert record[
                    "same_seed_equal_samples"
                ], "Same-backend seeded outputs differ"
            previous = samples
            report["checks"].append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.reference_audio:
            reference_body = {
                **body,
                "references": [
                    {
                        "data": base64.b64encode(
                            args.reference_audio.read_bytes()
                        ).decode("ascii"),
                        "media_type": "audio/wav",
                        "text": args.reference_text,
                    }
                ],
            }
            response = requests.post(url, json=reference_body, timeout=args.timeout)
            response.raise_for_status()
            samples, rate = sf.read(io.BytesIO(response.content), dtype="float32")
            assert rate == 24000 and samples.ndim == 1 and len(samples) > 0
            assert np.isfinite(samples).all()
            assert np.sqrt(np.mean(samples.astype(np.float64) ** 2)) > 1e-5
            (args.output / "reference.wav").write_bytes(response.content)
            report["reference"] = {
                "file": "reference.wav",
                "seconds": len(samples) / rate,
            }
            if args.streaming:
                report["reference_stream"] = check_stream(
                    url,
                    reference_body,
                    timeout=args.timeout,
                    output=args.output / "reference-stream.wav",
                )
                assert (
                    report["reference_stream"]["seconds"]
                    == report["reference"]["seconds"]
                ), "Reference stream duration differs from full response"
        if args.streaming:
            report["streams"] = []
            for number in (1, 2):
                report["streams"].append(
                    check_stream(
                        url,
                        body,
                        timeout=args.timeout,
                        output=args.output / f"stream-{number}.wav",
                    )
                )
                assert (
                    report["streams"][-1]["seconds"] == report["checks"][-1]["seconds"]
                ), "Stream duration differs from full response"
            assert (
                report["streams"][0]["pcm_sha256"] == report["streams"][1]["pcm_sha256"]
            ), "Seeded streams differ"
            report["disconnect"] = check_stream(
                url,
                {**body, "input": args.text + " " + args.text},
                timeout=args.timeout,
                output=args.output / "disconnected-prefix.wav",
                abort=True,
            )
            response = requests.post(url, json=body, timeout=args.timeout)
            response.raise_for_status()
            recovered, rate = sf.read(io.BytesIO(response.content), dtype="float32")
            assert rate == 24000 and np.array_equal(
                previous, recovered
            ), "Post-disconnect request differs"
            report["post_disconnect_equal_samples"] = True
        response = requests.post(url, json={**body, "input": ""}, timeout=30)
        report["empty_input_status"] = response.status_code
        assert 400 <= response.status_code < 500, "Empty text must be rejected"
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
