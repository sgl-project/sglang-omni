"""Reads, saves and sets the settings of the apps under test (Voxt 2.x's settings
document, Voxt 1.x's defaults) for run_cells.py."""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import sqlite3
import subprocess
import sys
from pathlib import Path

MAIN_LANGUAGE = {
    "2x": {"en": "en", "zh": "zh"},
    "1x": {"en": '["en"]', "zh": '["zh-hans"]'},
}
BUNDLE_1X = "com.voxt.Voxt"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def settings_text(db: str) -> str:
    con = sqlite3.connect(db)
    try:
        (doc,) = con.execute(
            "select doc from documents where key='settings:app'"
        ).fetchone()
    finally:
        con.close()
    return doc


def settings_doc(db: str) -> dict:
    return json.loads(settings_text(db))


def llm_providers(db: str) -> list[str]:
    """The language model services set up in Voxt 2.x, in its settings order."""
    return [p["id"] for p in settings_doc(db).get("providers", {}).get("llm", [])]


def save_settings(plan: dict, root: Path) -> list[tuple[str, str, Path]]:
    """Copies the settings each 2x and 1x system changes into root/settings-backup
    (the settings document of each Voxt 2.x database, Voxt 1.x's defaults) and
    returns (family, target, copy). A copy left by a killed run is checked and kept;
    a new one is written to a temporary file and renamed once complete."""
    backup = root / "settings-backup"
    backup.mkdir(exist_ok=True)
    targets = {}
    for system in plan["systems"].values():
        if system["family"] == "2x":
            targets.setdefault(("2x", system["db"]), None)
        elif system["family"] == "1x":
            targets.setdefault(("1x", BUNDLE_1X), None)
    saved = []
    for family, target in targets:
        tag = hashlib.sha256(target.encode()).hexdigest()[:8]
        copy = backup / (
            f"{tag}-settings-app.json" if family == "2x" else f"{tag}-{target}.plist"
        )
        if copy.exists():
            try:
                (json.loads if family == "2x" else plistlib.loads)(copy.read_bytes())
            except ValueError as e:
                sys.exit(
                    f"{copy} is not a complete backup ({e}); check the app's settings by hand, then delete it"
                )
        else:
            tmp = copy.with_name(copy.name + ".tmp")
            if family == "2x":
                tmp.write_text(settings_text(target))
            else:
                run(["defaults", "export", target, str(tmp)], check=True)
            os.replace(tmp, copy)
        saved.append((family, target, copy))
    return saved


def reset_settings(saved: list[tuple[str, str, Path]]) -> None:
    """Puts the saved settings back and keeps the copies."""
    for family, target, copy in saved:
        if family == "2x":
            # note (luojiaxuan): Only the settings row goes back; the history the run wrote stays.
            con = sqlite3.connect(target)
            try:
                with con:
                    con.execute(
                        "update documents set doc=? where key='settings:app'",
                        [copy.read_text()],
                    )
            finally:
                con.close()
        else:
            run(["defaults", "import", target, str(copy)], check=True)


def restore_settings(saved: list[tuple[str, str, Path]]) -> None:
    reset_settings(saved)
    for _, _, copy in saved:
        copy.unlink()


def configure(system: dict, block: str, capture_uid: str | None = None) -> dict:
    """Sets what this cell changes with the app quit, and capture_uid as the app's
    microphone when given; returns the settings as stored, for the cell's record."""
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
        if capture_uid:
            sets.append(("$.audio.devices", [capture_uid]))
        # note (luojiaxuan): Values are bound as parameters; json(?) turns booleans and lists into JSON.
        as_json = (bool, list)
        slots = ", ".join(
            "?, json(?)" if isinstance(v, as_json) else "?, ?" for _, v in sets
        )
        params = [
            x
            for k, v in sets
            for x in (k, json.dumps(v) if isinstance(v, as_json) else v)
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
    if capture_uid:
        run(
            [
                "defaults",
                "write",
                bundle,
                "activeInputDeviceUID",
                "-string",
                capture_uid,
            ],
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
