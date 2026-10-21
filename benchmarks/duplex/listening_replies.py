# SPDX-License-Identifier: Apache-2.0
"""Cut every reply of the scored sessions into its own wav, with the status and
transcript the server gave it, and transcribe it with the pinned Parakeet.

A reply is the audio of one response_id, its deltas back to back as a client
with enough buffer plays them, up to where the simulated playout stopped at a
barge-in cut. Gaps from late deltas are left out: the serving report measures
them as underruns.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile
from pydantic import JsonValue
from scipy.signal import resample_poly

from benchmarks.duplex.reference_asr import load_nemo_model
from benchmarks.duplex.reference_audio import SCHEMA as V15_MANIFEST_KIND
from benchmarks.duplex.reference_core import read_json
from benchmarks.duplex.reference_v10 import MANIFEST_KIND as V10_MANIFEST_KIND
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.duplex.v15_audio import write_json

REPLY_SAMPLE_RATE = 16000
REPLIES_FILE = "replies.json"
# note (jeffro): Parakeet raises on a 198-sample reply read from its wav, cut
# 12 ms after it started; no word fits in 0.1 s, so a shorter reply is heard as
# saying nothing.
MIN_TRANSCRIBED_DURATION_S = 0.1


def scored_variants(manifests: list[Path]) -> list[dict[str, JsonValue]]:
    """The eligible variants of v1.5 and v1.0 reference exports, with the
    dataset, category and recorder directory of each."""
    variants = []
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        for sample in manifest["samples"]:
            if manifest["kind"] == V15_MANIFEST_KIND:
                dataset, category, states = (
                    "v1.5",
                    sample["category"],
                    sample["variants"],
                )
            elif manifest["kind"] == V10_MANIFEST_KIND:
                # note (luojiaxuan): a v1.0 sample is one recording, its own variant.
                dataset, category, states = "v1.0", sample["subset"], {"input": sample}
            else:
                raise ValueError(f"{manifest_path} is not a reference export manifest")
            for name, state in states.items():
                if state["eligible"]:
                    variants.append(
                        {
                            "dataset": dataset,
                            "category": category,
                            "sample_id": sample["sample_id"],
                            "variant": name,
                            "directory": state["source"]["directory"],
                            "trace_sha256": state["source"]["trace_sha256"],
                        }
                    )
                else:
                    pass
    return variants


def spoken_transcript(response: dict[str, JsonValue]) -> str | None:
    """What response.done says the reply's audio contains, or None when it
    carries no audio transcript."""
    transcripts = [
        content["transcript"]
        for item in response["output"]
        for content in item["content"]
        if content["type"] == "output_audio"
    ]
    return "".join(transcripts) if transcripts else None


def played_samples(
    chunk: dict[str, JsonValue], cuts: list[dict[str, JsonValue]]
) -> int:
    """How much of an audio delta the simulated playout played: all of it,
    unless the first barge-in cut received after it fell before its end."""
    later = [cut for cut in cuts if cut["trace_line"] > chunk["trace_line"]]
    if not later:
        return chunk["samples"]
    else:
        heard = later[0]["cut_sample"] - chunk["playout_start_sample"]
        return min(chunk["samples"], max(0, heard))


def variant_replies(directory: Path) -> list[dict[str, JsonValue]]:
    """The replies of one recorded session in the order they started, each with
    the PCM16 samples it played at the recording's output rate."""
    playout = read_json(directory / "playout.json")
    # note (luojiaxuan): only captures whose replies stop at barge-in list cuts.
    cuts = playout["cuts"] if "cuts" in playout else []
    media, sample_rate = soundfile.read(directory / "output-media.wav", dtype="int16")
    replies: dict[str, dict[str, JsonValue]] = {}
    for chunk in playout["chunks"]:
        reply = replies.setdefault(
            chunk["response_id"],
            {"first_line": chunk["trace_line"], "pieces": [], "done": None},
        )
        start = chunk["media_start_sample"]
        reply["pieces"].append(media[start : start + played_samples(chunk, cuts)])
    with (directory / "continuous.jsonl").open(encoding="utf-8") as trace:
        for line_number, line in enumerate(trace, start=1):
            record = json.loads(line)
            event = record["event"]
            if (
                record["direction"] == "receive"
                and event.get("type") == "response.done"
            ):
                reply = replies.setdefault(
                    event["response"]["id"],
                    {"first_line": line_number, "pieces": [], "done": None},
                )
                reply["done"] = event["response"]
            else:
                pass
    ordered = sorted(replies.items(), key=lambda item: item[1]["first_line"])
    return [
        {
            "response_id": response_id,
            "status": None if reply["done"] is None else reply["done"]["status"],
            "transcript": (
                None if reply["done"] is None else spoken_transcript(reply["done"])
            ),
            "samples": (
                np.concatenate(reply["pieces"])
                if reply["pieces"]
                else np.zeros(0, dtype=np.int16)
            ),
            "sample_rate": sample_rate,
        }
        for response_id, reply in ordered
    ]


def to_reply_rate(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == REPLY_SAMPLE_RATE:
        return samples
    else:
        divisor = math.gcd(sample_rate, REPLY_SAMPLE_RATE)
        resampled = resample_poly(
            samples.astype(np.float64),
            REPLY_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        )
        return np.clip(np.round(resampled), -32768, 32767).astype(np.int16)


def extract_replies(manifests: list[Path], out_dir: Path) -> list[dict[str, JsonValue]]:
    """Write one 16 kHz wav per reply that has audio under out_dir/wav/ and
    return the reply records."""
    records = []
    for variant in scored_variants(manifests):
        directory = Path(variant["directory"])
        if file_sha256(directory / "continuous.jsonl") != variant["trace_sha256"]:
            raise ValueError(
                f"{directory}: continuous.jsonl changed after the reference export"
            )
        else:
            pass
        for index, reply in enumerate(variant_replies(directory)):
            samples = to_reply_rate(reply["samples"], reply["sample_rate"])
            wav = None
            if len(samples):
                wav = (
                    Path("wav")
                    / variant["dataset"]
                    / variant["sample_id"]
                    / variant["variant"]
                    / f"{index}.wav"
                )
                (out_dir / wav).parent.mkdir(parents=True, exist_ok=True)
                soundfile.write(
                    out_dir / wav, samples, REPLY_SAMPLE_RATE, subtype="PCM_16"
                )
            else:
                pass
            records.append(
                {
                    **{
                        key: variant[key]
                        for key in ("dataset", "category", "sample_id", "variant")
                    },
                    "response_id": reply["response_id"],
                    "status": reply["status"],
                    "transcript": reply["transcript"],
                    "duration_s": len(samples) / REPLY_SAMPLE_RATE,
                    "wav": None if wav is None else str(wav),
                }
            )
    return records


def transcribe_replies(
    manifests: list[Path],
    out_dir: Path,
    nemo: Path,
    nemo_sha256: str,
    device: str,
) -> None:
    """Cut the replies and write replies.json with each reply's Parakeet text;
    a reply without audio, or shorter than MIN_TRANSCRIBED_DURATION_S, has no
    text."""
    if file_sha256(nemo) != nemo_sha256:
        raise SystemExit(f"{nemo} does not match its pinned SHA-256")
    else:
        pass
    out_dir.mkdir(parents=True, exist_ok=True)
    records = extract_replies(manifests, out_dir)
    spoken = [
        record
        for record in records
        if record["wav"] is not None
        and record["duration_s"] >= MIN_TRANSCRIBED_DURATION_S
    ]
    model = load_nemo_model(nemo, device)
    hypotheses = (
        model.transcribe([str(out_dir / record["wav"]) for record in spoken])
        if spoken
        else []
    )
    for record, hypothesis in zip(spoken, hypotheses, strict=True):
        record["asr_text"] = hypothesis.text
    for record in records:
        record.setdefault("asr_text", None)
    write_json(
        out_dir / REPLIES_FILE,
        {
            "asr": {"checkpoint_sha256": nemo_sha256, "device_type": device},
            "manifests": [str(path) for path in manifests],
            "replies": records,
        },
    )
