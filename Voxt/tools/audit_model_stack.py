#!/usr/bin/env python3
"""Audit retired runtimes, catalog invariants and resolved model dependencies.

Source checks run on any platform. --app requires macOS; --link-map checks
static-link provenance, which cannot reliably be inferred from stripped symbols.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(r"fluidaudio|nemotextprocessing|sherpa[-_]?onnx|onnxruntime", re.I)
RETIRED_API = re.compile(
    r"\b(?:FluidAudioMeetingSpeakerDiarizationEngine|MeetingOfflineVBxModelStorage|"
    r"VoxtralRealtimeModel|CanaryModel|MoonshineModel|Wav2Vec2CTCModel|LasrCTCModel|"
    r"GraniteSpeechModel|FireRedASR2Model|GLMASRModel|MMSLanguageAdapterOption|"
    r"nativeVoxtralLive|hiddenSupport)\b"
)


def source_errors(root):
    errors = []
    project = (root / "Voxt.xcodeproj/project.pbxproj").read_text()
    if FORBIDDEN.search(project):
        errors.append("Retired dependency in Xcode project")
    required_revisions = [
        "2a6e75d28ae6a399ba7c7aec842384ef7a3142b5",
        "c6446cf7bfb7cea76408013b614d4b2c530eaa03",
    ]
    for revision in required_revisions:
        if f"revision = {revision};" not in project:
            errors.append(f"Missing fixed model dependency revision: {revision}")
    if "XCLocalSwiftPackageReference" in project:
        errors.append("Local package override in shared release project")
    for path in (root / "Voxt").rglob("*.swift"):
        text = path.read_text()
        if RETIRED_API.search(text) or re.search(r"(?:import|canImport\()\s*FluidAudio", text):
            errors.append(f"Retired runtime API: {path.relative_to(root)}")
    for name, count in [("Voxt/Transcription/MLXModelSupport.swift", 11),
                        ("Voxt/Core/Models/CustomLLMModelSupport.swift", 12)]:
        text = (root / name).read_text()
        ids = re.findall(r'\bOption\(\s*id:\s*"([^"]+)"', text)
        if len(ids) != count or len(set(ids)) != count:
            errors.append(f"Unexpected or duplicate active model IDs: {name} ({len(ids)})")
    return errors


def resolved_errors(path):
    pins = json.loads(path.read_text())["pins"]
    states = {pin["identity"]: pin["state"] for pin in pins}
    errors = []
    for pin in pins:
        if FORBIDDEN.search(pin["identity"] + pin.get("location", "")):
            errors.append(f"Retired resolved dependency: {pin['identity']}")
    # Update only together with the tested MLX compatibility set.
    expected = {
        "mlx-audio-swift": ("revision", "2a6e75d28ae6a399ba7c7aec842384ef7a3142b5"),
        "mlx-swift": ("version", "0.31.6"),
        "mlx-swift-lm": ("revision", "c6446cf7bfb7cea76408013b614d4b2c530eaa03"),
        "swift-transformers": ("version", "1.3.4"),
        "swift-huggingface": ("version", "0.10.2"),
        "sparkle": ("version", "2.10.0"),
        "grdb.swift": ("version", "7.11.1"),
        "swift-log": ("version", "1.15.1"),
    }
    for identity, (key, value) in expected.items():
        actual = states.get(identity, {}).get(key)
        if actual != value:
            errors.append(f"Unexpected {identity} {key}: {actual!r}; expected {value}")
    return errors


def app_errors(app):
    if sys.platform != "darwin":
        raise RuntimeError("--app requires macOS (file / otool)")
    if not (app / "Contents/MacOS").is_dir():
        raise ValueError(f"Not a macOS app bundle: {app}")
    errors = []
    for path in app.rglob("*"):
        if FORBIDDEN.search(str(path.relative_to(app))) or path.suffix.lower() == ".onnx":
            errors.append(f"Retired runtime artifact: {path}")
        if not path.is_file() or path.is_symlink():
            continue
        kind = subprocess.check_output(["file", "-b", str(path)], text=True)
        if "Mach-O" in kind:
            links = subprocess.check_output(["otool", "-L", str(path)], text=True)
            if FORBIDDEN.search(links):
                errors.append(f"Retired dynamic link: {path}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved", type=Path)
    parser.add_argument("--app", type=Path)
    parser.add_argument("--link-map", type=Path, action="append", default=[])
    args = parser.parse_args()
    try:
        errors = source_errors(ROOT)
        if args.resolved:
            errors += resolved_errors(args.resolved)
        if args.app:
            errors += app_errors(args.app)
        for path in args.link_map:
            if FORBIDDEN.search(path.read_text()):
                errors.append(f"Retired static-link input: {path}")
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Model stack audit could not complete: {error}", file=sys.stderr)
        return 1
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Model stack audit passed for the supplied source/resolved/artifact inputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
