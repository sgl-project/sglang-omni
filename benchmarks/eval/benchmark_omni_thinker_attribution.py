# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni thinker concurrency + attribution benchmark (PD Phase 0).

Rebuilds the canonical thinker baseline that #1018 s1.1 still lists as open,
and attributes where request time goes as offered load rises.

Two arrival models, because they answer different questions:

* ``--mode open`` (default) drives Poisson arrivals at a fixed offered rate.
  This is the only mode that can show the capacity knee: past saturation a
  closed-loop client absorbs the backlog into its own think time and hides it
  (the same reason ``benchmarks/eval/bench_sweep.py`` chose open loop).
* ``--mode closed`` holds N requests in flight, matching the arrival model
  used by ``docs/developer_reference/qwen3_asr_concurrency_profile.md`` so the
  thinker numbers can be read against that ASR table.

Server-side attribution comes from the existing generic profiler events via
``benchmarks.eval.asr_profiling`` (start/stop request profile, stage
breakdown, utilization sampling, environment fingerprint). Nothing here is
ASR-specific despite that module's name.

Run from the repo root:

    python -m benchmarks.eval.benchmark_omni_thinker_attribution \
      --port 8611 --mode open --rates 1,2,4,8,16,24 --duration-s 60 \
      --profile-events --profile-event-dir ./events \
      --sample-util --util-gpu-ids 1 --fingerprint \
      --output thinker_attr.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from benchmarks.eval.asr_profiling import (
    UtilizationSampler,
    build_stage_breakdown,
    collect_environment_fingerprint,
    collect_server_identity,
    start_request_profile,
    stop_request_profile,
    write_json,
)

# Prompt padding is the prefill-cost lever. Holding arrival rate and output
# length fixed while moving prompt length is the only manipulation that
# separates "the admission queue is long because capacity is capped" from
# "the admission queue is long because prefill blocks decode" -- the passive
# stage decomposition shows both as a large queued->scheduled segment.
#
# Text padding rather than an image prompt on purpose: an image arm would move
# encoder work and prefill cost together, so it cannot attribute either.
_PROMPT_HEAD = [
    "Explain in detail how a distributed key-value cache maintains consistency "
    "across replicas when a network partition heals.",
    "Describe the tradeoffs between tensor parallelism and pipeline parallelism "
    "for serving a mixture-of-experts language model.",
    "Walk through what happens inside a GPU when a kernel launch is issued, "
    "from the host driver call to warp scheduling.",
    "Compare write-through and write-back caching policies, and say when each "
    "one is the wrong choice.",
]

# `--prompt-tokens` is a target, not a count. Each repetition adds a counter
# prefix and many filler words take more than one token, so the request
# overshoots the flag, and the overshoot grows with the value. Measured against
# the Qwen3-Omni tokenizer: 42 gives 42, 1150 gives about 1966, 4400 gives about
# 7645, 2000 gives 2975, and 8000 gives 12075. Calibrate before choosing arms,
# and check the target against the model's context limit: 8192 for Qwen3-Omni
# thinker, which 4400 reaches once 128 output tokens are added.
#
# The nonce also shifts the count, because it sits inside the marker and inside
# every counter. Let the client generate it rather than passing a `--nonce` of
# a different width, or the arms move relative to each other.
#
# `prompt_tokens` reported in the result JSON is the measured value from usage
# and is what should be quoted.
_FILLER = (
    "The scheduler admits a request, builds its batch, runs one forward pass, "
    "and then releases the slot back to the pool for the next arrival. "
)


def _make_prompt(index: int, prompt_tokens: int, nonce: str = "") -> str:
    """Build a prompt of roughly `prompt_tokens` tokens, unique per request.

    Uniqueness is load-bearing and must start at token 0. RadixCache is on by
    default, so any shared prefix is served from the prefix tree and the arm
    silently measures cache hits instead of prefill cost. An earlier version
    rotated filler words per request; because the rotation period was shorter
    than the arm size it produced exact duplicate prompts, and the prompt-length
    lever collapsed. Put the unique marker first and vary every repetition.
    """
    marker = f"[req {nonce}{index}] "
    head = _PROMPT_HEAD[index % len(_PROMPT_HEAD)]
    if prompt_tokens <= 64:
        return marker + head
    words = _FILLER.split()
    reps = max(1, prompt_tokens // max(1, len(words)))
    chunks = [f"{marker}{head}"]
    for r in range(reps):
        # A per-repetition counter keeps every window of the prompt distinct,
        # so no suffix of one prompt can match a prefix of another.
        chunks.append(f"({nonce}{index}.{r}) " + " ".join(words))
    return " ".join(chunks)


@dataclass
class RequestRecord:
    request_id: str
    arm: str
    sent_at: float
    ttft_s: float | None = None
    total_s: float | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    # Client-observed inter-token gaps. The server also emits per-token
    # stage_stream_chunk_sent events; both are kept because the client series
    # is the user-visible metric and the server series is the attributable one.
    itl_ms: list[float] = field(default_factory=list)
    ok: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "arm": self.arm,
            "sent_at": self.sent_at,
            "ttft_s": self.ttft_s,
            "total_s": self.total_s,
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "itl_ms": self.itl_ms,
            "ok": self.ok,
            "error": self.error,
        }


@dataclass
class ArmResult:
    arm: str
    mode: str
    target: float
    records: list[RequestRecord] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        ok = [r for r in self.records if r.ok]
        failed = len(self.records) - len(ok)
        wall = 0.0
        if self.records:
            starts = [r.sent_at for r in self.records]
            ends = [
                r.sent_at + r.total_s for r in ok if r.total_s is not None
            ] or starts
            wall = max(ends) - min(starts)

        def pct(values: list[float], q: float) -> float | None:
            if not values:
                return None
            values = sorted(values)
            if len(values) == 1:
                return values[0]
            pos = q / 100.0 * (len(values) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(values) - 1)
            return values[lo] + (values[hi] - values[lo]) * (pos - lo)

        itls = [gap for r in ok for gap in r.itl_ms]
        ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
        totals = [r.total_s for r in ok if r.total_s is not None]
        toks = sum(r.completion_tokens or 0 for r in ok)
        return {
            "arm": self.arm,
            "mode": self.mode,
            "target": self.target,
            "requests_sent": len(self.records),
            "requests_ok": len(ok),
            "requests_failed": failed,
            "wall_s": wall,
            "achieved_rate_rps": (len(ok) / wall) if wall > 0 else None,
            "output_tok_per_s": (toks / wall) if wall > 0 else None,
            "ttft_s": {
                "mean": statistics.fmean(ttfts) if ttfts else None,
                "p50": pct(ttfts, 50),
                "p95": pct(ttfts, 95),
                "p99": pct(ttfts, 99),
                "max": max(ttfts) if ttfts else None,
            },
            "total_s": {
                "mean": statistics.fmean(totals) if totals else None,
                "p50": pct(totals, 50),
                "p95": pct(totals, 95),
                "p99": pct(totals, 99),
                "max": max(totals) if totals else None,
            },
            "itl_ms": {
                "mean": statistics.fmean(itls) if itls else None,
                "p50": pct(itls, 50),
                "p95": pct(itls, 95),
                "p99": pct(itls, 99),
                "max": max(itls) if itls else None,
                "count": len(itls),
            },
            "prompt_tokens_mean": (
                statistics.fmean(
                    [r.prompt_tokens for r in ok if r.prompt_tokens is not None]
                )
                if any(r.prompt_tokens is not None for r in ok)
                else None
            ),
            "errors": sorted({r.error for r in self.records if r.error})[:5],
        }


# --- unique-image pool -------------------------------------------------------
# The encoder result cache is keyed by a media reference/content hash and
# concurrent duplicates are collapsed by a single-flight table, so an image arm
# must never repeat a file: otherwise every request after the first measures a
# dict lookup instead of vision encode + large prefill.
_IMAGE_POOL: list[str] = []
_IMAGE_CURSOR = 0


def _load_image_pool(image_dir: str, offset: int = 0) -> None:
    """Load the pool and start the cursor at `offset`.

    The cursor restarts at 0 in every client process, so two runs against the
    SAME server would replay the same files and the second would be served from
    the encoder cache. Runs sharing a server must therefore be given disjoint
    offsets; separate servers each have their own cache and can both start at 0.
    """
    global _IMAGE_POOL, _IMAGE_CURSOR
    entries = sorted(
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not entries:
        raise SystemExit(f"--image-dir {image_dir} contains no images")
    if offset >= len(entries):
        raise SystemExit(f"--image-offset {offset} exceeds pool size {len(entries)}")
    _IMAGE_POOL = entries
    _IMAGE_CURSOR = offset
    print(
        f"image pool: {len(entries)} files from {image_dir}, starting at {offset}",
        flush=True,
    )


def _next_image() -> str:
    """Hand out the next unused image; exhausting the pool is a hard error.

    Wrapping around would silently reintroduce cache hits partway through the
    sweep, which is exactly the failure this pool exists to prevent.
    """
    global _IMAGE_CURSOR
    if _IMAGE_CURSOR >= len(_IMAGE_POOL):
        raise SystemExit(
            f"image pool exhausted after {_IMAGE_CURSOR} requests; "
            "generate a larger pool -- reusing images would measure cache hits"
        )
    path = _IMAGE_POOL[_IMAGE_CURSOR]
    _IMAGE_CURSOR += 1
    return path


def _image_pool_used() -> int:
    return _IMAGE_CURSOR


# -----------------------------------------------------------------------------


def _build_payload(args, prompt: str) -> dict[str, Any]:
    content: Any = prompt
    if getattr(args, "image_dir", None):
        # qwen3-omni takes media through the top-level "images" field
        # (serve/protocol.py:75, consumed at serve/openai_api.py:938). It has no
        # handler for inline {"type": "image_url"} content parts the way ming_omni
        # and llada2_uni do, so the --image-path form below is silently dropped by
        # this model: the request still succeeds, but it prefills text only.
        return {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "images": [_next_image()],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    if args.image_path:
        # Note: the omni chat surface takes multimodal parts; a local path is
        # accepted by the server-side reader, matching examples/ usage.
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"file://{args.image_path}"}},
        ]
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def _one_request(
    session: aiohttp.ClientSession, args, arm: str, prompt: str
) -> RequestRecord:
    rid = f"{arm}-{uuid.uuid4().hex[:12]}"
    rec = RequestRecord(request_id=rid, arm=arm, sent_at=time.time())
    started = time.perf_counter()
    last_token = started
    payload = _build_payload(args, prompt)
    try:
        async with session.post(
            f"{args.base_url}/v1/chat/completions",
            json=payload,
            headers={"x-request-id": rid},
            timeout=aiohttp.ClientTimeout(total=args.timeout_s),
        ) as resp:
            if resp.status != 200:
                rec.error = f"http {resp.status}: {(await resp.text())[:200]}"
                return rec
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[len("data:") :].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage:
                    rec.completion_tokens = usage.get("completion_tokens")
                    rec.prompt_tokens = usage.get("prompt_tokens")
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    now = time.perf_counter()
                    if rec.ttft_s is None:
                        rec.ttft_s = now - started
                    else:
                        rec.itl_ms.append((now - last_token) * 1000.0)
                    last_token = now
            rec.total_s = time.perf_counter() - started
            rec.ok = rec.ttft_s is not None
            if not rec.ok and rec.error is None:
                rec.error = "no content delta"
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        rec.error = f"{type(exc).__name__}: {exc}"
        rec.total_s = time.perf_counter() - started
    return rec


async def _run_open(session, args, rate: float, rng: random.Random) -> ArmResult:
    """Poisson arrivals at `rate` req/s for `duration_s`, never blocking."""
    arm = f"open-{rate:g}rps-p{args.prompt_tokens}"
    result = ArmResult(arm=arm, mode="open", target=rate)
    tasks: list[asyncio.Task] = []
    deadline = time.perf_counter() + args.duration_s
    i = 0
    while time.perf_counter() < deadline:
        prompt = _make_prompt(i, args.prompt_tokens, args.nonce)
        tasks.append(asyncio.create_task(_one_request(session, args, arm, prompt)))
        i += 1
        await asyncio.sleep(rng.expovariate(rate))
    if tasks:
        result.records = list(await asyncio.gather(*tasks))
    return result


async def _run_closed(session, args, concurrency: int) -> ArmResult:
    """Hold `concurrency` requests in flight for `requests` total."""
    arm = f"closed-c{concurrency}-p{args.prompt_tokens}"
    result = ArmResult(arm=arm, mode="closed", target=concurrency)
    sem = asyncio.Semaphore(concurrency)
    total = max(args.requests, concurrency)

    async def guarded(idx: int) -> RequestRecord:
        async with sem:
            return await _one_request(
                session, args, arm, _make_prompt(idx, args.prompt_tokens, args.nonce)
            )

    result.records = list(await asyncio.gather(*(guarded(i) for i in range(total))))
    return result


async def _warmup(session, args) -> None:
    for _ in range(args.warmup):
        await _one_request(
            session,
            args,
            "warmup",
            _make_prompt(-1 - _, args.prompt_tokens, args.nonce),
        )


def _raise_descriptor_limit() -> None:
    """Raise the open-file soft limit toward its hard limit, and say so."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= hard:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except (ValueError, OSError) as exc:
        print(f"could not raise RLIMIT_NOFILE from {soft}: {exc}", flush=True)
        return
    print(f"raised RLIMIT_NOFILE {soft} -> {hard}", flush=True)


async def _resolve_model(session, args) -> str:
    if args.model:
        return args.model
    async with session.get(f"{args.base_url}/v1/models") as resp:
        body = await resp.json()
    data = body.get("data") or []
    if not data:
        raise SystemExit("no models advertised by the server")
    return data[0]["id"]


async def main_async(args) -> int:
    rng = random.Random(args.seed)
    # An open-loop client holds one socket per in-flight request, so in-flight
    # count is a socket count. With an unlimited connector and the usual 1024
    # soft descriptor limit, a saturated arm exhausts descriptors and aiohttp
    # reports `ClientConnectorError ... [Too many open files]` -- which reads
    # like the server refusing, and is not. Raise the limit first, then cap the
    # connector below it so the client queues instead of failing.
    _raise_descriptor_limit()
    connector = aiohttp.TCPConnector(limit=args.max_connections)
    payload: dict[str, Any] = {
        "config": {
            "base_url": args.base_url,
            "mode": args.mode,
            "rates": args.rates,
            "concurrencies": args.concurrencies,
            "duration_s": args.duration_s,
            "requests": args.requests,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "seed": args.seed,
            "prompt_tokens_target": args.prompt_tokens,
            "image_path": args.image_path,
        },
        "arms": [],
    }

    async with aiohttp.ClientSession(connector=connector) as session:
        args.model = await _resolve_model(session, args)
        payload["config"]["model"] = args.model
        print(f"model: {args.model}", file=sys.stderr)

        if args.fingerprint:
            payload["server_identity"] = collect_server_identity(args.base_url)
            payload["environment"] = collect_environment_fingerprint(args.model)

        if args.warmup:
            print(f"warmup x{args.warmup}", file=sys.stderr)
            await _warmup(session, args)

        targets: list[float] = (
            [float(x) for x in args.rates.split(",") if x]
            if args.mode == "open"
            else [float(x) for x in args.concurrencies.split(",") if x]
        )

        for target in targets:
            for repeat in range(args.repeats):
                run_id = (
                    f"thinker-{args.mode}-{target:g}-p{args.prompt_tokens}-r{repeat}"
                )
                event_dir = None
                if args.profile_events:
                    event_dir = os.path.join(args.profile_event_dir, run_id)
                    os.makedirs(event_dir, exist_ok=True)
                    start_request_profile(args.base_url, run_id, event_dir)
                sampler = None
                if args.sample_util:
                    gpu_ids = [
                        int(x) for x in (args.util_gpu_ids or "").split(",") if x
                    ]
                    sampler = UtilizationSampler(gpu_ids=gpu_ids)
                    sampler.start()

                print(f"-> {run_id}", file=sys.stderr)
                if args.mode == "open":
                    arm = await _run_open(session, args, target, rng)
                else:
                    arm = await _run_closed(session, args, int(target))

                entry = arm.summary()
                entry["repeat"] = repeat
                entry["run_id"] = run_id
                if sampler is not None:
                    entry["utilization"] = sampler.stop().to_dict()
                if event_dir is not None:
                    stop_request_profile(args.base_url, run_id)
                    try:
                        entry["stage_breakdown"] = build_stage_breakdown(event_dir)
                    except Exception as exc:  # noqa: BLE001
                        entry["stage_breakdown_error"] = f"{type(exc).__name__}: {exc}"
                if args.save_raw_dir:
                    os.makedirs(args.save_raw_dir, exist_ok=True)
                    with open(
                        os.path.join(args.save_raw_dir, f"{run_id}.jsonl"), "w"
                    ) as handle:
                        for record in arm.records:
                            handle.write(json.dumps(record.to_dict()) + "\n")
                payload["arms"].append(entry)
                print(
                    f"   ok={entry['requests_ok']}/{entry['requests_sent']} "
                    f"rate={entry['achieved_rate_rps']} "
                    f"ttft_p95={entry['ttft_s']['p95']} "
                    f"itl_p95={entry['itl_ms']['p95']} "
                    f"ptok={entry['prompt_tokens_mean']}",
                    file=sys.stderr,
                )

    write_json(args.output, payload)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8611)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", choices=["open", "closed"], default="open")
    parser.add_argument("--rates", default="1,2,4,8,16,24")
    parser.add_argument("--concurrencies", default="1,8,16,32,48,64")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--max-connections",
        type=int,
        default=4096,
        help=(
            "cap on simultaneous connections. The client queues above this "
            "instead of exhausting descriptors and reporting a connection "
            "error that looks like the server refusing."
        ),
    )
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument(
        "--nonce", default=None, help="prefix marker making prompts unique across runs"
    )
    parser.add_argument("--image-path", default=None)
    parser.add_argument(
        "--image-dir",
        default=None,
        help="draw a distinct image per request from this pool",
    )
    parser.add_argument(
        "--image-offset",
        type=int,
        default=0,
        help="start index into the pool; keep disjoint per server",
    )
    parser.add_argument("--profile-events", action="store_true")
    parser.add_argument("--profile-event-dir", default="/tmp/thinker_profile")
    parser.add_argument("--sample-util", action="store_true")
    parser.add_argument("--util-gpu-ids", default=None)
    parser.add_argument("--fingerprint", action="store_true")
    parser.add_argument("--save-raw-dir", default=None)
    parser.add_argument("--output", default="thinker_attribution.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.image_dir:
        _load_image_pool(args.image_dir, args.image_offset)
    if args.nonce is None:
        args.nonce = uuid.uuid4().hex[:6] + "-"
    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"
    args.base_url = args.base_url.rstrip("/")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
