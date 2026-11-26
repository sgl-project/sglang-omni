#!/usr/bin/env python3
"""A stratified sample of AASR-Bench utterances to play to every dictation app, and
the driver manifest for them.

    python3 aasr_sample.py select --benchmark benchmark.jsonl --n 150 --out sample.jsonl
    python3 aasr_sample.py fetch --sample sample.jsonl --dir W/aasr --out W/manifest.jsonl

select: a stratified sample over language x scene. Each stratum gets its
proportional share of N, rounded by largest remainder; within a stratum the
utterances are taken in the order of the SHA-256 of their source_record_id, so
the sample is fixed by the benchmark alone.

fetch: downloads benchmark.jsonl, the checksums and the sampled clips from the
Hugging Face dataset at the pinned revision (no account needed), checks every
clip against the dataset's sha256sums.txt, and writes the replay-driver manifest:
id (the clip's file name), wav, lang, reference (the spoken transcript, `oral`),
source_record_id and scene. The dataset carries no license; it is used for
evaluation only and not redistributed, so the clips stay on the measuring Mac.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import struct
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

DATASET = "Andrew0425/AASR-Bench"
REVISION = "802577d4ee1424653b0152e9e37b7b4a1fa5efbf"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
SAMPLE_FIELDS = ("source_record_id", "audio", "language", "scene", "voice_id")
ATTEMPTS = 4


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def key(row: dict) -> str:
    return hashlib.sha256(row["source_record_id"].encode()).hexdigest()


def select(rows: list[dict], n: int) -> list[dict]:
    strata = defaultdict(list)
    for row in rows:
        strata[(row["language"], row["scene"])].append(row)
    quotas = {s: n * len(members) / len(rows) for s, members in strata.items()}
    counts = {s: int(q) for s, q in quotas.items()}
    by_remainder = sorted(quotas, key=lambda s: (-(quotas[s] - counts[s]), s))
    for s in by_remainder[: n - sum(counts.values())]:
        counts[s] += 1
    chosen = []
    for s in sorted(strata):
        chosen += sorted(strata[s], key=key)[: counts[s]]
    return chosen


def download(path: str, dest: Path) -> None:
    """Hugging Face sometimes stalls a read; each file is tried ATTEMPTS times."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with (
                urllib.request.urlopen(f"{BASE}/{path}", timeout=60) as resp,
                tmp.open("wb") as f,
            ):
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == ATTEMPTS:
                raise
            time.sleep(5 * attempt)


def wav_seconds(path: Path) -> float:
    """The clips are written as a stream: the RIFF and data sizes are 0xFFFFFFFF
    placeholders, so the length comes from the bytes after the data chunk header."""
    with path.open("rb") as f:
        f.seek(12)
        while header := f.read(8):
            chunk, size = struct.unpack("<4sI", header)
            if chunk == b"fmt ":
                fmt = f.read(size)
                channels, rate, block = struct.unpack("<HI4xH", fmt[2:14])
                continue
            if chunk == b"data":
                return min(size, path.stat().st_size - f.tell()) / block / rate
            f.seek(size + size % 2, 1)
    raise ValueError(f"{path} has no data chunk")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_select(args) -> None:
    sample = [
        {k: row[k] for k in SAMPLE_FIELDS}
        for row in select(read_jsonl(Path(args.benchmark)), args.n)
    ]
    write_jsonl(Path(args.out), sample)
    langs = defaultdict(int)
    for row in sample:
        langs[row["language"]] += 1
    print(f"wrote {len(sample)} utterances to {args.out}: {dict(langs)}")


def cmd_fetch(args) -> None:
    root = Path(args.dir).resolve()
    for path in ("data/benchmark.jsonl", "metadata/sha256sums.txt"):
        if not (root / path).exists():
            download(path, root / path)
    sums = {}
    for line in (root / "metadata/sha256sums.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("*").removeprefix("./")] = digest
    bench = {
        row["source_record_id"]: row
        for row in read_jsonl(root / "data/benchmark.jsonl")
    }
    items = [bench[row["source_record_id"]] for row in read_jsonl(Path(args.sample))]
    todo = [
        item["audio"] for item in items if not (root / "data" / item["audio"]).exists()
    ]
    with cf.ThreadPoolExecutor(8) as pool:
        list(
            pool.map(
                lambda audio: download(f"data/{audio}", root / "data" / audio), todo
            )
        )
    manifest, seconds = [], 0.0
    for item in items:
        clip = root / "data" / item["audio"]
        expected = sums[f"data/{item['audio']}"]
        if sha256(clip) != expected:
            clip.unlink()
            sys.exit(f"{clip} does not match the dataset's checksum; run fetch again")
        seconds += wav_seconds(clip)
        manifest.append(
            {
                "id": clip.stem,
                "wav": str(clip),
                "lang": item["language"],
                "reference": item["oral"],
                "source_record_id": item["source_record_id"],
                "scene": item["scene"],
                "voice_id": item["voice_id"],
            }
        )
    write_jsonl(Path(args.out), manifest)
    print(
        f"wrote {len(manifest)} rows to {args.out}; {seconds / 60:.1f} minutes of audio"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--sample", required=True)
    p.add_argument("--dir", required=True, help="where the dataset files go")
    p.add_argument("--out", required=True, help="the replay-driver manifest")
    args = parser.parse_args()
    {"select": cmd_select, "fetch": cmd_fetch}[args.cmd](args)


if __name__ == "__main__":
    main()
