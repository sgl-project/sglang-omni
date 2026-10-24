#!/usr/bin/env python3
"""Build and sign Voxt Connector without launching it."""
import argparse
import datetime
import os
import plistlib
import re
import shutil
import subprocess
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "connector/VoxtConnector"
ICON = APP / "Resources/AppIcon.icns"
CONFIG = APP / "Config"


def xcconfig_setting(path, name):
    """The last plain `NAME = value` in an xcconfig file, or None."""
    value = None
    for line in Path(path).read_text().splitlines():
        key, sep, rest = line.partition("=")
        if sep and key.strip() == name:
            value = rest.split("//")[0].strip()
    return value


def bundle_id(config=CONFIG, env=os.environ):
    """The Connector's bundle ID: VOXT_CONNECTOR_BUNDLE_ID from the file VOXT_CONNECTOR_SIGNING_LOCAL names
    (a release builds a clean public checkout), else from Signing.local.xcconfig, else the neutral default
    in Signing.shared.xcconfig. Voxt puts this ID into a code requirement, so it must be a plain bundle ID.
    """
    private = env.get("VOXT_CONNECTOR_SIGNING_LOCAL", "")
    if private:
        if not Path(private).is_file():
            raise SystemExit(f"VOXT_CONNECTOR_SIGNING_LOCAL is not a file: {private}")
        value, source = (
            xcconfig_setting(private, "VOXT_CONNECTOR_BUNDLE_ID"),
            "VOXT_CONNECTOR_SIGNING_LOCAL",
        )
    else:
        local = Path(config) / "Signing.local.xcconfig"
        value = (
            xcconfig_setting(local, "VOXT_CONNECTOR_BUNDLE_ID")
            if local.is_file()
            else None
        )
        value, source = (
            (value, local.name)
            if value
            else (
                xcconfig_setting(
                    Path(config) / "Signing.shared.xcconfig", "VOXT_CONNECTOR_BUNDLE_ID"
                ),
                "Signing.shared.xcconfig",
            )
        )
    if not value or not re.fullmatch(r"[A-Za-z0-9.-]+", value):
        raise SystemExit(
            f"{source} sets no usable VOXT_CONNECTOR_BUNDLE_ID (got {value!r})."
        )
    return value


def xcodebuild_command(derived, result_bundle, bundle):
    return [
        "nice",
        "-n",
        "10",
        "xcodebuild",
        "build",
        "-project",
        "connector/VoxtConnector/VoxtConnector.xcodeproj",
        "-scheme",
        "VoxtConnector",
        "-configuration",
        "Release",
        "-destination",
        "platform=macOS",
        "-derivedDataPath",
        str(derived),
        "-resultBundlePath",
        str(result_bundle),
        "-jobs",
        "4",
        "CODE_SIGNING_ALLOWED=NO",
        f"VOXT_CONNECTOR_BUNDLE_ID={bundle}",
    ]


def add_icon(app, icon=ICON):
    """Copy the icon into the bundle and name it in Info.plist. Run before signing."""
    resources = app / "Contents/Resources"
    resources.mkdir(exist_ok=True)
    shutil.copyfile(icon, resources / "AppIcon.icns")
    info_path = app / "Contents/Info.plist"
    info = plistlib.loads(info_path.read_bytes())
    info["CFBundleIconFile"] = "AppIcon"
    info_path.write_bytes(plistlib.dumps(info))


def add_source_revision(app, revision):
    """Record the public commit the app was built from (the release check reads it). Run before signing."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(
            f"--source-revision must be a full commit hash, got {revision!r}."
        )
    info_path = app / "Contents/Info.plist"
    info = plistlib.loads(info_path.read_bytes())
    info["VoxtSourceRevision"] = revision
    info_path.write_bytes(plistlib.dumps(info))


def check_icon(app):
    # Without these the Dock shows a blank icon.
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    name = info.get("CFBundleIconFile")
    if not name:
        raise SystemExit("Info.plist has no CFBundleIconFile.")
    icns = (
        app
        / "Contents/Resources"
        / (name if name.endswith(".icns") else name + ".icns")
    )
    if not icns.is_file() or icns.read_bytes()[:4] != b"icns":
        raise SystemExit(f"{icns.name} is missing from Contents/Resources.")


def require(tool, install, which=shutil.which):
    if which(tool) is None:
        raise SystemExit(f"{tool} is needed to build Voxt Connector: {install}")


def sign(app, identity, run=subprocess.run):
    # Hardened runtime: Voxt trusts the Connector by code signature, and without the runtime
    # an injected library passes that check.
    targets = (app / "Contents/Helpers/voxt-tailcat-host", app)
    for target in targets:
        try:
            run(
                [
                    "codesign",
                    "--force",
                    "--options",
                    "runtime",
                    "--sign",
                    identity,
                    str(target),
                ],
                check=True,
                timeout=90,
            )
        except subprocess.CalledProcessError:
            raise SystemExit(
                f"Could not sign with {identity!r}. Name another identity in "
                "VOXT_CONNECTOR_SIGNING_IDENTITY, or build with VOXT_CONNECTOR_SIGNING_IDENTITY=- "
                "(ad hoc): Voxt then won't hand its port and token to the Connector, so enter "
                "them in Connector Settings."
            )
    run(
        ["codesign", "--verify", "--deep", "--strict", str(app)], check=True, timeout=90
    )
    for target in targets:
        details = run(
            ["codesign", "-dv", str(target)], capture_output=True, text=True, timeout=90
        )
        if "runtime" not in (getattr(details, "stderr", "") or ""):
            raise SystemExit(f"{target.name} is not signed with the hardened runtime.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-revision",
        help="public commit to record in Info.plist (release builds)",
    )
    parser.add_argument(
        "--bundle-id",
        help="the Connector's bundle ID (default: from connector/VoxtConnector/Config)",
    )
    parser.add_argument(
        "--print-bundle-id",
        action="store_true",
        help="print the bundle ID a build would use and stop",
    )
    args = parser.parse_args()
    bundle = args.bundle_id or bundle_id()
    if not re.fullmatch(r"[A-Za-z0-9.-]+", bundle):
        raise SystemExit(f"--bundle-id must be a plain bundle ID, got {bundle!r}.")
    if args.print_bundle_id:
        print(bundle)
        return
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if (
        subprocess.run(["pgrep", "-x", "xcodebuild"], capture_output=True).returncode
        == 0
    ):
        raise SystemExit("An xcodebuild is already running. Wait for it to finish.")
    tmp = root / "build/connector-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp) + "/"
    os.environ["GOMAXPROCS"] = "4"
    os.environ["GOFLAGS"] = "-p=4"
    runtime = root / "build/connector-runtime/voxt-tailcat-host"
    os.environ["VOXT_TAILCAT_HOST_OUT"] = str(runtime.parent)
    evidence = root / "build/connector-build"
    evidence.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    derived = root / "build/connector-derived"
    require("xcodegen", "brew install xcodegen")
    subprocess.run(["nice", "-n", "10", "tools/build-tailcat-host.sh"], check=True)
    subprocess.run(
        ["xcodegen", "generate", "--spec", "connector/VoxtConnector/project.yml"],
        check=True,
    )
    subprocess.run(
        xcodebuild_command(derived, evidence / (stamp + ".xcresult"), bundle),
        check=True,
    )
    app = derived / "Build/Products/Release/Voxt Connector.app"
    built = plistlib.loads((app / "Contents/Info.plist").read_bytes()).get(
        "CFBundleIdentifier"
    )
    if built != bundle:
        raise SystemExit(f"The Connector was built as {built}, not {bundle}.")
    subprocess.run(
        ["python3", "tools/stage-runtime.py", str(app), str(runtime)], check=True
    )
    add_icon(app)
    check_icon(app)
    if args.source_revision:
        add_source_revision(app, args.source_revision)
    try:
        sign(
            app, os.environ.get("VOXT_CONNECTOR_SIGNING_IDENTITY", "Apple Development")
        )
    except subprocess.TimeoutExpired:
        raise SystemExit(
            "Signing exceeded 90 seconds and was stopped. Retry with the user present."
        )
    print("build: " + str(app), flush=True)


if __name__ == "__main__":
    main()
