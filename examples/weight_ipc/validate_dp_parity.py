#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Leader vs follower audio parity for weight-IPC DP.

Example:
  python examples/weight_ipc/validate_dp_parity.py \\
    --leader-url http://127.0.0.1:8801 \\
    --follower-url http://127.0.0.1:8802 \\
    --n 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

DEFAULT_PROMPTS = [
    "Hello, how are you?",
    "Have a nice day and enjoy the sunshine.",
    "The quick brown fox jumps over the lazy dog.",
    "SGLang Omni weight sharing parity check.",
    "Please read this sentence carefully and clearly.",
]


def _speech(
    url: str,
    text: str,
    *,
    seed: int,
    max_new_tokens: int,
    timeout_s: float,
) -> bytes:
    resp = requests.post(
        f"{url.rstrip('/')}/v1/audio/speech",
        json={
            "input": text,
            "response_format": "wav",
            "stream": False,
            "temperature": 0.0,
            "top_k": 1,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.content


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader-url", required=True)
    parser.add_argument("--follower-url", required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    prompts = (
        DEFAULT_PROMPTS * ((args.n + len(DEFAULT_PROMPTS) - 1) // len(DEFAULT_PROMPTS))
    )[: args.n]
    rows: list[dict] = []
    identical = 0
    for i, text in enumerate(prompts):
        t0 = time.perf_counter()
        leader_audio = _speech(
            args.leader_url,
            text,
            seed=args.seed + i,
            max_new_tokens=args.max_new_tokens,
            timeout_s=args.timeout_s,
        )
        follower_audio = _speech(
            args.follower_url,
            text,
            seed=args.seed + i,
            max_new_tokens=args.max_new_tokens,
            timeout_s=args.timeout_s,
        )
        l_sha = hashlib.sha256(leader_audio).hexdigest()
        f_sha = hashlib.sha256(follower_audio).hexdigest()
        ok = l_sha == f_sha and len(leader_audio) == len(follower_audio)
        identical += int(ok)
        row = {
            "index": i,
            "text": text,
            "ok": ok,
            "leader_sha256": l_sha,
            "follower_sha256": f_sha,
            "leader_bytes": len(leader_audio),
            "follower_bytes": len(follower_audio),
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
        rows.append(row)
        status = "OK" if ok else "MISMATCH"
        print(
            f"[{i+1:02d}/{args.n}] {status} "
            f"leader={len(leader_audio)}B follower={len(follower_audio)}B "
            f"sha={l_sha[:12]}",
            flush=True,
        )

    summary = {
        "n": args.n,
        "identical": identical,
        "pass": identical == args.n,
        "leader_url": args.leader_url,
        "follower_url": args.follower_url,
        "rows": rows,
    }
    print(
        f"PARITY {identical}/{args.n} bit-identical "
        f"({'PASS' if summary['pass'] else 'FAIL'})",
        flush=True,
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
