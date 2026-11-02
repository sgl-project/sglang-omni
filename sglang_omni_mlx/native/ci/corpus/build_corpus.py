"""Builds the frozen Qwen A/A/B corpus from public, human-transcribed test sets.

Sources are LibriSpeech test-clean, FLEURS cmn_hans_cn and ASCEND (hashes in
raw.sha256); selection is deterministic.
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SEED = 20261004
RATE = 16000


def resample(audio: np.ndarray, rate: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate == RATE:
        return audio.astype(np.float32)
    # Note (Jiaxin Deng): the sources are already 16 kHz; linear resampling is
    # only a fallback.
    duration = len(audio) / rate
    target = int(round(duration * RATE))
    x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0, duration, num=target, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def load(path_or_bytes) -> np.ndarray:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        audio, rate = sf.read(io.BytesIO(path_or_bytes), dtype="float32")
    else:
        audio, rate = sf.read(str(path_or_bytes), dtype="float32")
    return resample(audio, rate)


def librispeech(root: Path) -> list[dict]:
    items = []
    for trans in sorted(root.glob("LibriSpeech/test-clean/*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            flac = trans.parent / f"{uid}.flac"
            info = sf.info(str(flac))
            items.append(
                {
                    "source": "librispeech-test-clean",
                    "source_id": uid,
                    "chapter": "-".join(uid.split("-")[:2]),
                    "lang": "en",
                    "reference": text,
                    "duration": info.frames / info.samplerate,
                    "path": flac,
                }
            )
    return items


def fleurs(root: Path, tsv: Path) -> list[dict]:
    items = []
    seen_sentences = set()
    with tsv.open() as handle:
        for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            sentence_id, file_name, raw = row[0], row[1], row[2]
            if sentence_id in seen_sentences:
                continue
            seen_sentences.add(sentence_id)
            wav = root / "test" / file_name
            info = sf.info(str(wav))
            items.append(
                {
                    "source": "fleurs-cmn_hans_cn-test",
                    "source_id": file_name.removesuffix(".wav"),
                    "lang": "zh",
                    "reference": raw,
                    "duration": info.frames / info.samplerate,
                    "path": wav,
                }
            )
    return sorted(items, key=lambda item: item["source_id"])


def ascend(parquet: Path) -> list[dict]:
    table = pq.read_table(parquet).to_pylist()
    return [
        {
            "source": "ascend-test",
            "source_id": row["id"],
            "lang": row["language"],
            "reference": row["transcription"],
            "duration": float(row["duration"]),
            "bytes": row["audio"]["bytes"],
        }
        for row in table
    ]


def in_range(items, low, high):
    return [item for item in items if low <= item["duration"] <= high]


def take(rng, items, count, used):
    pool = [item for item in items if (item["source"], item["source_id"]) not in used]
    chosen = rng.sample(pool, min(count, len(pool)))
    for item in chosen:
        used.add((item["source"], item["source_id"]))
    return chosen


def concat_long(rng, items, group_key, count, low, high, used, joiner):
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(group_key(item), []).append(item)
    keys = sorted(groups)
    rng.shuffle(keys)
    results = []
    for key in keys:
        parts, total = [], 0.0
        for item in sorted(groups[key], key=lambda it: it["source_id"]):
            if (item["source"], item["source_id"]) in used:
                continue
            parts.append(item)
            total += item["duration"] + 0.4
            if total >= low:
                break
        if low <= total <= high:
            for item in parts:
                used.add((item["source"], item["source_id"]))
            results.append(parts)
        if len(results) == count:
            break
    return [
        {
            "source": parts[0]["source"],
            "source_id": "+".join(p["source_id"] for p in parts),
            "lang": parts[0]["lang"],
            "reference": joiner.join(p["reference"] for p in parts),
            "parts": parts,
        }
        for parts in results
    ]


def main() -> None:
    work = Path(sys.argv[1])
    extracted = work / "corpus/extracted"
    out = work / "corpus/v1"
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    used: set = set()

    en = librispeech(extracted)
    zh = fleurs(extracted, work / "corpus/raw/fleurs-cmn-test.tsv")
    cs = ascend(work / "corpus/raw/ascend-test.parquet")
    zh_numbers = [item for item in zh if re.search(r"[0-9０-９]", item["reference"])]

    selection: list[tuple[str, dict]] = []
    selection += [("en_short", it) for it in take(rng, in_range(en, 2, 5), 64, used)]
    selection += [("en_mid", it) for it in take(rng, in_range(en, 10, 20), 64, used)]
    selection += [("zh_numbers", it) for it in take(rng, zh_numbers, 24, used)]
    selection += [("zh_short", it) for it in take(rng, in_range(zh, 2, 7), 40, used)]
    selection += [("zh_mid", it) for it in take(rng, in_range(zh, 10, 20), 64, used)]
    selection += [
        ("zh_short", it)
        for it in take(
            rng, [c for c in in_range(cs, 2, 7) if c["lang"] == "zh"], 30, used
        )
    ]
    selection += [
        ("mixed", it)
        for it in take(
            rng, [c for c in in_range(cs, 2, 20) if c["lang"] == "mixed"], 64, used
        )
    ]
    selection += [
        ("en_long", it)
        for it in concat_long(rng, en, lambda it: it["chapter"], 16, 60, 120, used, " ")
    ]
    selection += [
        ("zh_long", it)
        for it in concat_long(
            rng, zh, lambda it: str(int(it["source_id"]) % 24), 16, 60, 120, used, ""
        )
    ]

    records = []
    gap = np.zeros(int(0.4 * RATE), dtype=np.float32)
    for index, (stratum, item) in enumerate(selection):
        if "parts" in item:
            pieces = []
            for part in item["parts"]:
                pieces += [load(part["path"]), gap]
            audio = np.concatenate(pieces[:-1])
        else:
            audio = load(item.get("bytes") or item["path"])
        clip_id = f"{index:04d}_{stratum}"
        sf.write(clips / f"{clip_id}.wav", audio, RATE, subtype="PCM_16")
        records.append(
            {
                "id": clip_id,
                "stratum": stratum,
                "lang": item["lang"],
                "source": item["source"],
                "source_id": item["source_id"],
                "duration": round(len(audio) / RATE, 3),
                "reference": item["reference"],
            }
        )

    noise_rng = np.random.default_rng(SEED)
    for index in range(4):
        clip_id = f"{len(records):04d}_silence"
        sf.write(
            clips / f"{clip_id}.wav",
            np.zeros(5 * RATE, np.float32),
            RATE,
            subtype="PCM_16",
        )
        records.append(
            {
                "id": clip_id,
                "stratum": "silence",
                "lang": "none",
                "source": "synthetic",
                "source_id": f"silence-{index}",
                "duration": 5.0,
                "reference": "",
            }
        )
    for index in range(4):
        clip_id = f"{len(records):04d}_noise"
        noise = noise_rng.normal(0, 0.01, 5 * RATE).astype(np.float32)
        sf.write(clips / f"{clip_id}.wav", noise, RATE, subtype="PCM_16")
        records.append(
            {
                "id": clip_id,
                "stratum": "noise",
                "lang": "none",
                "source": "synthetic",
                "source_id": f"noise-{index}",
                "duration": 5.0,
                "reference": "",
            }
        )
    mids = [r for r in records if r["stratum"] in ("en_mid", "zh_mid")]
    for base in rng.sample(mids, 8):
        audio, _ = sf.read(clips / f"{base['id']}.wav", dtype="float32")
        power = float(np.mean(audio**2))
        noisy = audio + noise_rng.normal(0, np.sqrt(power / 10), len(audio)).astype(
            np.float32
        )
        noisy = np.clip(noisy, -1, 1)
        clip_id = f"{len(records):04d}_snr10_{base['lang']}"
        sf.write(clips / f"{clip_id}.wav", noisy, RATE, subtype="PCM_16")
        records.append({**base, "id": clip_id, "stratum": f"snr10_{base['lang']}"})

    with (out / "manifest.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    by_stratum: dict[str, list[float]] = {}
    for record in records:
        by_stratum.setdefault(record["stratum"], []).append(record["duration"])
    for stratum, durations in by_stratum.items():
        print(
            f"{stratum:12s} n={len(durations):3d} total={sum(durations):7.1f}s "
            f"min={min(durations):6.1f} max={max(durations):6.1f}"
        )
    print(
        "clips",
        len(records),
        "audio seconds",
        round(sum(r["duration"] for r in records), 1),
    )


if __name__ == "__main__":
    main()
