# SPDX-License-Identifier: Apache-2.0
"""Emit scripts/xpu/README.md from the committed baseline and the model table.

Every performance figure on that page is generated, so the page cannot drift from
xpu_model_baseline.json. Re-run it after recording, rather than editing a table by hand:

    python scripts/xpu/make_baseline_readme.py > scripts/xpu/README.md
"""

from __future__ import annotations

import ast
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "xpu_model_baseline.json"
SCRIPT = HERE / "xpu_model_baseline.py"
LANE_TABLE = HERE.parents[1] / "tests" / "test_ci" / "xpu_model" / "model_table.py"

# What each recorded environment is, for a reader who has not met the host.
HOST_NOTES = {
    "bmg17690": "8x Arc Pro B60, 24 GB each",
}
GROUPS = (
    ("transcription", "Speech to text", "audio s / wall s"),
    ("speech", "Text to speech", "audio s / wall s"),
    ("chat", "Chat (text out)", "decode tok/s"),
    ("music", "Text to music", "-"),
)

with BASELINE.open() as handle:
    data = json.load(handle)
with SCRIPT.open() as handle:
    tree = ast.parse(handle.read())

rows = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "XpuModel":
        entry = {}
        for keyword in node.keywords:
            try:
                entry[keyword.arg] = ast.literal_eval(keyword.value)
            except ValueError:
                entry[keyword.arg] = ast.unparse(keyword.value)
        rows.append(entry)
    else:
        continue
by_key = {row["key"]: row for row in rows}

environments = sorted(
    data["environments"].values(), key=lambda env: -len(env["models"])
)


def d270_flags() -> dict[str, list[str]]:
    """Each lane row's D270-only flags, read from the lane's own table."""
    flags: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(LANE_TABLE.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "XpuModel":
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            if "device_args" not in keywords:
                continue
            else:
                key = ast.literal_eval(keywords["key"])
            for family, args in ast.literal_eval(keywords["device_args"]):
                if family == "d270":
                    flags[key] = list(args)
                else:
                    continue
        else:
            continue
    return flags


def speed(model: dict) -> tuple:
    measured = model.get("speed", {})
    rate = measured.get("audio_seconds_per_wall_second")
    if rate is None:
        rate = measured.get("decode_tokens_per_second")
    else:
        pass
    return measured.get("latency_s"), rate


def accuracy(model: dict):
    ceilings = model.get("atmost", {})
    return ceilings.get("word_error_rate", ceilings.get("round_trip_wer"))


def worst_spread(model: dict):
    spreads = [v for k, v in model.get("spread", {}).items() if k.startswith("speed")]
    return max(spreads) if spreads else None


def fmt(value, pattern: str = "%.3f") -> str:
    return (pattern % value) if value is not None else "-"


def host_table(environment: dict) -> str:
    models = environment["models"]
    out = []
    for probe, title, rate_name in GROUPS:
        keys = sorted(k for k in models if by_key.get(k, {}).get("probe") == probe)
        if not keys:
            continue
        else:
            out.append(f"\n**{title}**\n")
        out.append(f"| model | key | latency | {rate_name} | WER | spread | launches |")
        out.append("|---|---|---|---|---|---|---|")
        for key in keys:
            model = models[key]
            latency, rate = speed(model)
            spread = worst_spread(model)
            out.append(
                f"| {by_key.get(key, {}).get('name', key)} | `{key}` | "
                f"{fmt(latency)} s | {fmt(rate, '%.2f')} | "
                f"{fmt(accuracy(model))} | "
                f"{fmt(spread * 100 if spread is not None else None, '%.1f%%')} | "
                f"{model.get('launches', 1)} |"
            )
    return "\n".join(out)


def recipe_table() -> str:
    extra_on_d270 = d270_flags()
    out = [
        "| key | checkpoint | cards | extra serve flags (every host) | added on D270 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        if row.get("unservable"):
            continue
        else:
            args = [str(arg) for arg in (row.get("serve_args") or ())]
        environment = row.get("env") or {}
        flags = " ".join(args)
        if environment:
            flags = " ".join(f"{k}={v}" for k, v in environment.items()) + (
                f" (env) {flags}" if flags else " (env)"
            )
        else:
            pass
        added = " ".join(extra_on_d270.get(row["key"], []))
        out.append(
            f"| `{row['key']}` | `{row['model_path']}` | {row.get('cards', 1)} | "
            f"{('`' + flags + '`') if flags else 'none'} | "
            f"{('`' + added + '`') if added else '-'} |"
        )
    return "\n".join(out)


def unservable_table() -> str:
    out = ["| key | checkpoint | why not |", "|---|---|---|"]
    for row in rows:
        if row.get("unservable"):
            out.append(
                f"| `{row['key']}` | `{row['model_path']}` | {row['unservable']} |"
            )
        else:
            continue
    return "\n".join(out)


def environment_heading(environment: dict) -> str:
    fingerprint = environment["fingerprint"]
    host = fingerprint["host"]
    return (
        f"### `{host}` -- {HOST_NOTES.get(host, fingerprint.get('device', ''))}\n\n"
        f"torch `{fingerprint.get('torch')}`, sglang `{fingerprint.get('sglang')}`, "
        f"speed floor {environment['tolerances']['speed']:.0%}, "
        f"{len(environment['models'])} models"
    )


def single_launch_rows() -> str:
    out = []
    for environment in environments:
        keys = [
            f"`{key}`"
            for key, model in sorted(environment["models"].items())
            if model.get("launches", 1) == 1
        ]
        out.append(
            f"- `{environment['fingerprint']['host']}`: {', '.join(keys) or 'none'}"
        )
    return "\n".join(out)


per_environment = "\n\n".join(
    environment_heading(env) + "\n" + host_table(env) for env in environments
)

print(
    f"""<!-- Generated by scripts/xpu/make_baseline_readme.py from xpu_model_baseline.json; regenerate rather than editing the tables by hand. -->

# Intel XPU model tests and baseline performance

What sglang-omni serves on Intel XPU, how each model is launched, how to check that a tree
still serves it correctly, and how fast it was on the reference B60 host.

There are two kinds of test, and they answer different questions:

| | functional tests | performance baseline |
|---|---|---|
| question | does every model still start and answer correctly? | is this tree slower or less accurate than the last known-good one? |
| where | `tests/unit_test/xpu/test_model_support.py`, `tests/test_ci/xpu_model/` | `scripts/xpu/xpu_model_baseline.py` + `xpu_model_baseline.json` |
| needs | nothing (contract tests), or cards and weights (serving lane) | cards and weights |
| runs on | every PR (contract tests); on demand (serving lane) | on demand, per host |
| fails on | a broken recipe, a renamed stage, a wrong answer | a regression past the entry's own floor |

Both read the same per-model recipes, and a contract test fails if the two disagree.

## Functional tests

### Contract tests -- every PR, no card, no weights

```bash
pytest tests/unit_test/xpu/test_model_support.py
```

About 20 seconds. For every model family XPU serves it checks the pipeline config imports
and registers, the stage table matches what the recipes assume, every stage factory resolves,
a checkpoint resolves to the right family, every stage-scoped flag in a recipe names a stage
the pipeline has, and the recipes here and in the perf baseline agree. It also pins the
D270-only flags, and keeps the committed baseline to B60 hosts. It runs, and passes, on a host
with no XPU at all, which is how the main unit lane sees it.

### Serving lane -- one real request per model

```bash
pytest tests/test_ci/xpu_model -m xpu_serving -v -rs                      # every model
pytest tests/test_ci/xpu_model -m xpu_serving -v -k "qwen3-asr or dots-tts"  # a subset
```

Each row is served with its recipe, sent one request, and the answer is checked: a
transcript must contain the spoken word, speech must decode as a WAV with audible frames,
chat must answer `42` to `17 plus 25`. A row **skips with its reason** rather than failing
when its checkpoint is not local, its topology needs more cards than the host has, or the
table says this image cannot serve it, so an incomplete host reports what it could not do.
The full lane, 20 rows served and 6 skipped, took 39 minutes on bmg17690.

| variable | effect |
|---|---|
| `MODEL_ROOTS=/a:/b` | mirror roots searched before the Hugging Face cache; a directory counts only if it holds a config, a weight file and every shard its index names |
| `OMNI_XPU_<KEY>_MODEL=/path` | one row's checkpoint, e.g. `OMNI_XPU_QWEN3_ASR_MODEL` |
| `OMNI_XPU_ALLOW_HF_DOWNLOAD=1` | let a row download a missing checkpoint, and let its server reach the Hub (off by default: an Omni pull is 66 GB, and the server otherwise runs with `HF_HUB_OFFLINE=1`) |
| `OMNI_XPU_DEVICE_FAMILY=d270` | force the D270 recipe on a card the lane does not recognise |
| `REQUIRE_FREE_GPU=0` | serve on cards another process already has open |

The lane picks the recipe from the card it finds: a D270 card (`Intel(R) Graphics [0x674c]`)
gets the D270 flags listed under [Host-specific serve flags](#host-specific-serve-flags), a B60
gets none, and a card it does not know gets the B60 recipe rather than a guess.

## Performance baseline

```bash
conda activate sgl-xpu                                     # oneCCL needs CCL_ROOT from the env
export PATH=/opt/intel/oneapi/compiler/latest/bin:$PATH    # Ming's joint RoPE JITs with icpx

python scripts/xpu/xpu_model_baseline.py                   # check every model on this host
python scripts/xpu/xpu_model_baseline.py --list            # inventory, and which host recorded what
python scripts/xpu/xpu_model_baseline.py --models qwen3-asr,qwen3-tts
```

It exits non-zero on a regression, naming the field and the margin, and reports an
improvement as something to re-record rather than as a pass. No tolerance flag is needed:
each entry carries its own floor and the run prints the one it used.

| flag | what it does |
|---|---|
| `--record` | write this environment's entry instead of checking; `--fresh` re-measures rows already recorded |
| `--launches N` | restart the server N times and anchor the recorded speed at the **slowest**, so a fast mode reads as an improvement rather than firing the gate |
| `--card-pool 3,6` | restrict to specific cards; each model takes what its topology needs |
| `--idle-mib N` | refuse a card already holding more than N MiB (default 1024) |
| `--idle-wait-s N` | wait that long for a busy card instead of refusing it; a recording sweep can wait, a check should not |
| `--max-checkpoint-gib N` | skip a model whose checkpoint is larger than this, cached or not (default 100; only Ming-flash-omni exceeds it) |
| `--prefetch` | fetch weights before launching, for hosts whose server has no outbound network |
| `--prune-hf-cache` | delete cached revisions this measurement does not need; destructive, never the default, safe only where `HF_HOME` is owned by this work |
| `--host-label NAME` | name the machine in the fingerprint, for a container or cluster pod whose hostname is generated |
| `--calibrate --repeats 9` | report observed spread per field and exit without gating |

Speed is the median of 5 scored samples after 2 discarded warmups, at concurrency 1.
Concurrency 1 is deliberate: Qwen3-TTS throughput at concurrency 4 has swung more than 1.5x
between runs of one unchanged tree. A full single-launch check of every recorded model takes
about two hours on bmg17690, most of it MiniMax-Music3 (~45 min) and the TP=8 thinker.

## Reference host

`bmg17690`: 8x Arc Pro B60 (24 GB each) in a native conda env `sgl-xpu`. Every number on this
page was measured there. The speed floors absorb host load and an unexplained per-model
condition step, not sample noise, which is about 1%.

### Running on bmg17690

```bash
conda activate sgl-xpu
cd /path/to/sglang-omni
pytest tests/test_ci/xpu_model -m xpu_serving -v -rs
python scripts/xpu/xpu_model_baseline.py --record --launches 3
```

Other people share the cards: check `torch.xpu.mem_get_info` per card, or let the tools refuse
a busy card, and pass `--card-pool` to stay off cards that are in use. On the reference host
card 0's PCIe link trains at 5 GT/s where cards 1-7 run at 32 (read `current_link_speed` on
each card's switch upstream port, since links can retrain), so record which cards a number
came from.

### Running on D270

The serving lane detects a D270 card and adds its flags itself, so the lane command is the
same. The baseline takes them through `--serve-extra`, and its results go to a file outside
the tree: the committed baseline holds B60 environments only, and a contract test keeps it
that way.

```bash
pytest tests/test_ci/xpu_model -m xpu_serving -v -rs
python scripts/xpu/xpu_model_baseline.py --record --launches 3 \\
  --baseline /path/outside/the/tree/xpu_model_baseline.json \\
  --serve-extra "qwen3-tts=--tts_engine.engine.disable_cuda_graph true" \\
  --serve-extra "auk=--auk_engine.factory.enable_dit_cuda_graph false" \\
  --serve-extra "auk-flash=--auk_engine.factory.enable_dit_cuda_graph false"
```

In a container, pass `--host-label`: the fingerprint hashes the hostname, and a container's is
random. Where the server has no outbound network, pass `--prefetch`.

## Baseline performance

Latency is one request at concurrency 1. The rate column is audio seconds produced per wall
second for speech, so above 1 is faster than realtime -- it is the inverse of RTF and must not
be quoted beside one -- and decode tokens per second for chat. WER for speech is round-trip:
the audio is transcribed back by Qwen3-ASR.

{per_environment}

## Host-specific serve flags

A B60 needs none. D270 needs these, each for a capture that **loses the device** rather than
falling back to eager. The serving lane applies them on a D270 card; a baseline run takes them
through `--serve-extra` and records them in the entry's `serve_extra`:

| model | flag | why |
|---|---|---|
| `qwen3-tts` | `--tts_engine.engine.disable_cuda_graph true` | its reference-code batcher captures a graph on a private stream; that capture loses the device, surfacing later as Level-Zero error 20 from an unrelated-looking synchronize |
| `auk`, `auk-flash` | `--auk_engine.factory.enable_dit_cuda_graph false` | the DiT step graph fails capture with error 40, logs that it will run eager, and generation then dies with error 20 |

**Apply such a flag by request shape, not by model family.** Qwen3-TTS CustomVoice and
VoiceDesign carried the Base variant's flag for days and never needed it: only Base sends
`ref_audio`, and a request without reference audio never reaches the batcher that breaks.
The captured and eager paths do not generate identical audio, so which path a number came
from is part of the number.

## Launching a model by hand

```bash
ZE_AFFINITY_MASK=0 sgl-omni serve --model-path <checkpoint> <flags below> --port 8000
```

Stage gpu ids in the flags are relative to `ZE_AFFINITY_MASK`. A multi-card row needs that many
cards in the mask (`ZE_AFFINITY_MASK=0,1` for two).

{recipe_table()}

Three of those flags are workarounds with a known end:

- `minimax-music3` `--minimax_music3_ar.engine.attention_backend triton`: the AR stage turns on
  deterministic inference without naming a backend, and SGLang 0.5.23's deterministic fallback
  picks fa3, a CUDA-only kernel. Fixed upstream in sgl-project/sglang#46623.
- `qwen3-omni-text` `--skip-server-warmup`: the server's own warmup reaches the vision encoder,
  whose head dim of 72 is one sgl-kernel's XPU RoPE does not compile for. Text requests never
  touch it.
- `qwen3-omni-text` `--thinker.engine.mem_fraction_static 0.435`: SGLang 0.5.20 added an
  XPU-only memory reserve that walks the automatic fraction down to 0.302, below what the KV
  cache needs.

## What cannot be served here

{unservable_table()}

## Reading a result

- **Accuracy is the half worth gating on.** Exact transcripts, frame counts and WER have held
  across every condition tested, including passes contaminated by another process on the card.
- **A tight spread is not evidence of a good measurement.** A pass contaminated by a neighbour
  on the same card can read slow with a tight spread and no warning. Both tools now refuse a
  card already in use, and the baseline flags a neighbour that arrives mid-run.
- **Host CPU load is a variance source no gate sees.** A check against an unchanged baseline
  can come in past its floor with the cards idle and the host CPU saturated. Read `uptime`
  before believing a speed delta on a shared box.
- **A speed gate is sized from the trimmed spread**, with the best and worst sample dropped,
  while the full range is reported beside it. One stalled sample in nine would otherwise have
  given Nemotron 3.5 ASR a 201% tolerance, wide enough to miss a doubling.
- **`launches` is the confidence column.** A row at 1 was measured within one server lifetime,
  so it cannot tell a real change from the model sitting in its other mode; a row at 3 is
  anchored at the slowest of three starts. Whisper's three launches on bmg17690 differed by 19%,
  which is exactly the case the anchor exists for.

## Data provenance

Every number above comes from `scripts/xpu/xpu_model_baseline.json`. Page generated
{date.today().isoformat()}.

Rows still at a single launch, to re-record with `--launches 3` before treating as a floor:

{single_launch_rows()}
""",
    end="",
)
