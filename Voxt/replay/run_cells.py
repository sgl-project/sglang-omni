"""Runs a plan of replay cells unattended on this Mac.

A cell is one system (a dictation app in one configuration) and one language
block. For each cell: quit every app under test, configure the system, launch
it, start a fresh replay-sink, play three warm-up segments that are not scored, then
measure the block with replay-driver. A driver fault ends the driver with exit code
2; the cell is resumed (finished segments are skipped) up to MAX_RESUMES times.
Finished cells are skipped, so the whole plan can be started again after an
interruption. --check validates the plan and this Mac without pressing a key;
--pilot N runs only the first cell of each system on its first N segments, to
see that every app answers the key before the full run; --summary prints what a
run directory holds.

    python3 run_cells.py --plan plan.json --check
    caffeinate -dimsu python3 run_cells.py --plan plan.json --root PILOT_DIR --pilot 5
    caffeinate -dimsu python3 run_cells.py --plan plan.json --root RUN_DIR
    python3 run_cells.py --summary --root RUN_DIR

plan.json (plan.example.json is a template):
    {"driver": ".../replay-driver", "sink_app": ".../ReplaySink.app", "manifest": ".../manifest.jsonl",
     "settle_s": 3,
     "systems": {"voxt": {"app": ".../Voxt.app", "family": "2x", "db": ".../voxtnew.db",
                          "asr": "asr.accurate", "enhance": false,
                          "key": "right_option", "trigger": "toggle", "presses": 1},
                 "other-app": {"app": "/Applications/Other.app", "family": "other",
                               "key": "right_option", "trigger": "hold"}},
     "cells": [["voxt", "en"], ["voxt", "zh"], ...]}

family: "2x" is Voxt 2.x, whose main language, speech model and LLM clean-up
(enhance) are set in its settings database; "1x" is Voxt 1.x, whose main language
and speech model are set with defaults (clean-up stays off, set by hand once);
"other" is any other dictation app, configured by hand once before the run.
A 2x system may also name "llm", the id of a language model service already set
up in Voxt ("ollama", "codex" for the ChatGPT sign-in), and "llm_model"; the cell
then points Transcribe's clean-up at that service and model. "clean_after_stop"
(Voxt 2.1.4) sets whether that clean-up waits for the stop key, and
"reclean_sentences" how many sentences before the text left at Stop it cleans again.
key, trigger, presses and the optional modifiers are passed to replay-driver.

Blocks: "en" holds the English segments, "zh" the Chinese and mixed ones; Voxt's
main language follows the block.

Before the first cell, the settings of every 2x and 1x system are copied into
RUN_DIR/settings-backup (SQLite's backup API for the database, `defaults export`
for 1.x); after the last cell, or when the run stops on an error or Ctrl-C, they
are put back and the copies deleted. A copy left by a killed run is kept, and the
next start restores it.

Relative "wav" paths in the manifest are read against the manifest's directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from score import mark_contamination, outcome, pick_results

BLOCKS = {"en": {"en", "none"}, "zh": {"zh", "mixed"}}
MAIN_LANGUAGE = {
    "2x": {"en": "en", "zh": "zh"},
    "1x": {"en": '["en"]', "zh": '["zh-hans"]'},
}
FAMILIES = {"2x", "1x", "other"}
BUNDLE_1X = "com.voxt.Voxt"
INPUT = "BlackHole 2ch"
WARMUP_SEGMENTS = 3
MAX_RESUMES = 5
LAUNCH_WAIT_S = 20
SOCKET = str(Path.home() / ".replay-sink.sock")


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {message}", flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def info(app: Path) -> dict:
    with (app / "Contents/Info.plist").open("rb") as f:
        return plistlib.load(f)


def default_input(driver: str) -> str:
    for line in run([driver, "list-devices"], check=True).stdout.splitlines():
        device = json.loads(line)
        if device["default_input"]:
            return device["name"]
    return ""


def quit_all(plan: dict) -> None:
    for system in plan["systems"].values():
        bundle = info(Path(system["app"]))["CFBundleIdentifier"]
        run(
            [
                "osascript",
                "-e",
                f'if application id "{bundle}" is running then tell application id "{bundle}" to quit',
            ]
        )
    time.sleep(3)
    for system in plan["systems"].values():
        run(["pkill", "-f", f"{system['app']}/Contents/"])
    run(["pkill", "-f", "ReplaySink.app/Contents/MacOS"])
    time.sleep(2)


def settings_doc(db: str) -> dict:
    con = sqlite3.connect(db)
    try:
        (doc,) = con.execute(
            "select doc from documents where key='settings:app'"
        ).fetchone()
    finally:
        con.close()
    return json.loads(doc)


def llm_providers(db: str) -> list[str]:
    """The language model services set up in Voxt 2.x, in its settings order."""
    return [p["id"] for p in settings_doc(db).get("providers", {}).get("llm", [])]


def sqlite_copy(src: str | Path, dst: str | Path) -> None:
    source, target = sqlite3.connect(src), sqlite3.connect(dst)
    try:
        source.backup(target)
    finally:
        source.close()
        target.close()


def save_settings(plan: dict, root: Path) -> list[tuple[str, str, Path]]:
    """Copies the settings each 2x and 1x system changes into root/settings-backup,
    keeping a copy that a killed run left there; returns (family, target, copy)."""
    backup = root / "settings-backup"
    backup.mkdir(exist_ok=True)
    targets = {}
    for system in plan["systems"].values():
        if system["family"] == "2x":
            targets.setdefault(("2x", system["db"]), None)
        elif system["family"] == "1x":
            targets.setdefault(("1x", BUNDLE_1X), None)
    saved = []
    for n, (family, target) in enumerate(targets):
        copy = backup / (
            f"{n}-{Path(target).name}" if family == "2x" else f"{n}-{target}.plist"
        )
        if not copy.exists():
            if family == "2x":
                sqlite_copy(target, copy)
            else:
                run(["defaults", "export", target, str(copy)], check=True)
        saved.append((family, target, copy))
    return saved


def restore_settings(saved: list[tuple[str, str, Path]]) -> None:
    for family, target, copy in saved:
        if family == "2x":
            sqlite_copy(copy, target)
        else:
            run(["defaults", "import", target, str(copy)], check=True)
        copy.unlink()


def configure(system: dict, block: str) -> dict:
    """Sets what this cell changes with the app quit; returns the settings as
    stored, for the cell's record."""
    family = system["family"]
    if family == "other":
        return {}
    main = MAIN_LANGUAGE[family][block]
    if family == "2x":
        db = system["db"]
        sets = [
            ("$.general.mainLanguage", main),
            ("$.features.transcribe.asr", system["asr"]),
            ("$.features.transcribe.enhance", bool(system["enhance"])),
        ]
        if system["enhance"]:
            sets.append(("$.features.transcribe.cleanup", True))
        if "clean_after_stop" in system:
            sets.append(
                (
                    "$.features.transcribe.cleanAfterStop",
                    bool(system["clean_after_stop"]),
                )
            )
        if "reclean_sentences" in system:
            sets.append(
                (
                    "$.features.transcribe.recleanSentences",
                    int(system["reclean_sentences"]),
                )
            )
        if "llm" in system:
            index = llm_providers(db).index(system["llm"])
            sets += [
                ("$.features.transcribe.llm", system["llm"]),
                (f"$.providers.llm[{index}].model", system["llm_model"]),
            ]
        # note (luojiaxuan): Values are bound as parameters; json(?) turns "true" and "false" into JSON booleans.
        slots = ", ".join(
            "?, json(?)" if isinstance(v, bool) else "?, ?" for _, v in sets
        )
        params = [
            x for k, v in sets for x in (k, json.dumps(v) if isinstance(v, bool) else v)
        ]
        con = sqlite3.connect(db)
        try:
            with con:
                con.execute(
                    f"update documents set doc=json_set(doc, {slots}) where key='settings:app'",
                    params,
                )
        finally:
            con.close()
        return settings_doc(db)
    bundle = BUNDLE_1X
    features = json.loads(
        run(["defaults", "read", bundle, "featureSettings"], check=True).stdout
    )
    features["transcription"]["asrSelectionID"] = system["asr"]
    run(
        [
            "defaults",
            "write",
            bundle,
            "featureSettings",
            "-string",
            json.dumps(features),
        ],
        check=True,
    )
    run(
        ["defaults", "write", bundle, "userMainLanguageCodes", "-string", main],
        check=True,
    )
    exported = plistlib.loads(
        run(["defaults", "export", bundle, "-"], check=True).stdout.encode()
    )
    keys = (
        "featureSettings",
        "userMainLanguageCodes",
        "activeInputDeviceUID",
        "hotkeyTriggerMode",
        "localModelIdleUnloadDelaySeconds",
    )
    return {k: v for k, v in exported.items() if k in keys}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def block_rows(manifest: list[dict], block: str) -> list[dict]:
    return [r for r in manifest if r["lang"] in BLOCKS[block]]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def finished_ids(results: Path) -> set[str]:
    if not results.exists():
        return set()
    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
    return {r["id"] for r in rows if not r.get("driver_fault")}


def drive(
    plan: dict, system: dict, cell_dir: Path, manifest: Path, out: Path, sink_log: Path
) -> int:
    cmd = [
        plan["driver"],
        "run",
        "--manifest",
        str(manifest),
        "--out",
        str(out),
        "--sink-socket",
        SOCKET,
        "--sink-log",
        str(sink_log),
        "--key",
        system["key"],
        "--trigger",
        system["trigger"],
        "--settle-s",
        str(plan["settle_s"]),
        "--mem-app",
        Path(system["app"]).stem,
        "--mem-out",
        str(cell_dir / "mem.jsonl"),
    ]
    if system["trigger"] == "toggle":
        cmd += ["--presses", str(system["presses"])]
    if system.get("modifiers"):
        cmd += ["--modifiers", ",".join(system["modifiers"])]
    with (cell_dir / "driver.log").open("a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def run_cell(
    plan: dict,
    root: Path,
    manifest: list[dict],
    name: str,
    block: str,
    limit: int | None,
) -> str:
    system = plan["systems"][name]
    cell_dir = root / f"{name}-{block}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    rows = block_rows(manifest, block)[:limit]
    if len(finished_ids(cell_dir / "results.jsonl")) == len(rows):
        return "done before"
    if default_input(plan["driver"]) != INPUT:
        return f"skipped: the system default input is not {INPUT}"

    quit_all(plan)
    settings = configure(system, block)
    app = Path(system["app"])
    plist = info(app)
    record = {
        "system": name,
        "block": block,
        "app": str(app),
        "bundle": plist["CFBundleIdentifier"],
        "version": plist.get("CFBundleShortVersionString"),
        "build": plist.get("CFBundleVersion"),
        "executable_sha256": sha256(
            app / "Contents/MacOS" / plist["CFBundleExecutable"]
        ),
        "default_input": default_input(plan["driver"]),
        "plan": system,
        "settings": settings,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (cell_dir / "cells.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    run(["open", "-g", str(app)], check=True)
    time.sleep(LAUNCH_WAIT_S)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sink_log = cell_dir / f"sink-{stamp}.log"
    run(
        [
            "open",
            "-n",
            plan["sink_app"],
            "--args",
            "--log",
            str(sink_log),
            "--socket",
            SOCKET,
        ],
        check=True,
    )
    time.sleep(3)

    write_jsonl(cell_dir / "warmup-manifest.jsonl", rows[:WARMUP_SEGMENTS])
    drive(
        plan,
        system,
        cell_dir,
        cell_dir / "warmup-manifest.jsonl",
        cell_dir / f"warmup-{stamp}.jsonl",
        sink_log,
    )
    write_jsonl(cell_dir / "manifest.jsonl", rows)
    for attempt in range(MAX_RESUMES + 1):
        code = drive(
            plan,
            system,
            cell_dir,
            cell_dir / "manifest.jsonl",
            cell_dir / "results.jsonl",
            sink_log,
        )
        if code == 0:
            break
        log(f"{cell_dir.name}: driver exit {code}, resume {attempt + 1}")
        run(["open", "-b", "local.voxt.replay-sink"])
        time.sleep(2)
    return f"{len(finished_ids(cell_dir / 'results.jsonl'))}/{len(rows)} segments"


def check(plan: dict, manifest: list[dict]) -> list[str]:
    """Everything that can be known before the first key press."""
    problems = []
    for path in ("driver", "sink_app", "manifest"):
        if not Path(plan[path]).exists():
            problems.append(f"{path}: {plan[path]} does not exist")
    for name, system in plan["systems"].items():
        app = Path(system["app"])
        if system["family"] not in FAMILIES:
            problems.append(f"{name}: family must be one of {sorted(FAMILIES)}")
            continue
        if not (app / "Contents/Info.plist").exists():
            problems.append(f"{name}: {app} is not an app")
            continue
        plist = info(app)
        print(
            f"{name}: {plist['CFBundleIdentifier']} {plist.get('CFBundleShortVersionString')} ({system['family']})"
        )
        missing = [k for k in ("key", "trigger") if k not in system]
        missing += {"2x": ["db", "asr", "enhance"], "1x": ["asr"], "other": []}[
            system["family"]
        ]
        missing += ["presses"] if system.get("trigger") == "toggle" else []
        problems += [f"{name}: no {k}" for k in missing if k not in system]
        if (
            system["family"] == "2x"
            and "db" in system
            and not Path(system["db"]).exists()
        ):
            problems.append(f"{name}: {system['db']} does not exist")
        elif "llm" in system and system["llm"] not in llm_providers(system["db"]):
            problems.append(
                f"{name}: set up the language model {system['llm']!r} in Voxt first"
            )
    for name, block in plan["cells"]:
        if name not in plan["systems"]:
            problems.append(f"cell {name} {block}: no such system")
        if block not in BLOCKS or not block_rows(manifest, block):
            problems.append(f"cell {name} {block}: no segments in this block")
    if Path(plan["driver"]).exists() and default_input(plan["driver"]) != INPUT:
        problems.append(
            f"the system default input is {default_input(plan['driver'])!r}, not {INPUT}"
        )
    return problems


def summary(root: Path) -> None:
    """One line per cell: segments with a fault-free row out of the block, how
    many got their text after the stop key, only before it, not at all, were
    still changing or may hold another segment's text, and the median wait."""
    print(
        f"{'cell':24s} {'done':>9s} {'text':>5s} {'early':>5s} {'empty':>5s} {'timeout':>7s} "
        f"{'contam':>6s} {'median wait s':>13s}"
    )
    for cell in sorted(d for d in root.iterdir() if (d / "manifest.jsonl").exists()):
        total = sum(
            1 for l in (cell / "manifest.jsonl").read_text().splitlines() if l.strip()
        )
        results = []
        if (cell / "results.jsonl").exists():
            results = [
                json.loads(l)
                for l in (cell / "results.jsonl").read_text().splitlines()
                if l.strip()
            ]
        rows = pick_results(mark_contamination(results))
        status = [outcome(r) for r in rows.values()]
        waits = sorted(
            (r["t_last"] - r["t_stop"]) / 1e9
            for r in rows.values()
            if outcome(r) == "complete"
        )
        median = f"{waits[len(waits) // 2]:.2f}" if waits else "-"
        print(
            f"{cell.name:24s} {len(rows):>4d}/{total:<4d} {status.count('complete'):>5d} "
            f"{status.count('early'):>5d} {status.count('empty'):>5d} {status.count('timeout'):>7d} "
            f"{status.count('contaminated'):>6d} {median:>13s}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan")
    parser.add_argument("--root")
    parser.add_argument(
        "--check", action="store_true", help="validate the plan and this Mac, then exit"
    )
    parser.add_argument(
        "--pilot",
        type=int,
        metavar="N",
        help="only the first cell of each system, on its first N segments",
    )
    parser.add_argument(
        "--summary", action="store_true", help="print what --root holds, then exit"
    )
    args = parser.parse_args()
    if args.summary:
        if not args.root:
            parser.error("--summary needs --root")
        summary(Path(args.root))
        return 0
    if not args.plan:
        parser.error("--plan is required")
    plan = json.loads(Path(args.plan).read_text())
    manifest = [
        json.loads(l)
        for l in Path(plan["manifest"]).read_text().splitlines()
        if l.strip()
    ]
    # note (luojiaxuan): replay-driver reads relative wav paths against the manifest it is given, and
    # every cell writes its own manifest in its directory.
    base = Path(plan["manifest"]).resolve().parent
    for row in manifest:
        row["wav"] = str(base / row["wav"])
    if args.check:
        problems = check(plan, manifest)
        print("\n".join(problems) or "plan OK")
        return 1 if problems else 0
    if not args.root:
        parser.error("--root is required to run")
    # note (luojiaxuan): replay-sink starts in / under LaunchServices, so every path it gets is absolute.
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    cells = plan["cells"]
    if args.pilot:
        first = {}
        for name, block in cells:
            first.setdefault(name, block)
        cells = list(first.items())
    saved = save_settings(plan, root)
    try:
        for name, block in cells:
            log(f"cell {name} {block}: start")
            log(
                f"cell {name} {block}: {run_cell(plan, root, manifest, name, block, args.pilot)}"
            )
    finally:
        quit_all(plan)
        restore_settings(saved)
        log(f"settings restored for {len(saved)} target(s)")
    summary(root)
    log("plan finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
