#!/usr/bin/env python3
"""Loopback end-to-end run of the Local API: Voxt built from this tree, curl, and the contract fixtures.

--host xctest (unattended) builds the TestDebug test bundle and serves the Local API from the XCTest host through LocalAPILoopbackServeTests, because launching Voxt.app asks for Accessibility, which needs a person present. --host app prints the steps for the same run against a launched Voxt.app. Results go to results/ next to this file, with ids replaced and no token or local path.
"""

import argparse
import base64
import json
import os
import plistlib
import re
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[4]
VOXT = TREE / "Voxt"
CONTRACT = VOXT / "Mobile/contract"
TESTDATA = VOXT / "Mobile/testdata"
RESULTS = HERE / "results"
SAMPLE_ID = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
HEALTH_TIMEOUT_SECONDS = 300
CHINESE_CHARACTER = re.compile(r"[一-鿿]")


@dataclass(kw_only=True)
class Response:
    status: int
    body: object


@dataclass(kw_only=True)
class Check:
    name: str
    passed: bool
    compared: str
    response: Response


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def build_arguments(work: Path, packages: Path, action: str) -> list[str]:
    return [action, "-project", str(VOXT / "Voxt.xcodeproj"), "-scheme", "Voxt", "-configuration", "TestDebug", "-destination", "platform=macOS", "-derivedDataPath", str(work / "derived"), "-clonedSourcePackagesDirPath", str(packages), "-packageAuthorizationProvider", "netrc", "-jobs", "4", "-skipPackagePluginValidation", "CODE_SIGNING_ALLOWED=NO"]


def isolated_home(work: Path, models: Path) -> Path:
    """A home whose default model folder links to the existing mlx-audio and mlx-llm folders."""
    home = work / "home"
    storage = home / "Library/Application Support/Voxt/model-storage"
    storage.mkdir(parents=True, exist_ok=True)
    for name in ("mlx-audio", "mlx-llm"):
        link = storage / name
        if link.is_symlink() or link.exists():
            link.unlink()
        else:
            pass
        link.symlink_to(models / name)
    return home


def serving_xctestrun(work: Path, variables: dict[str, str]) -> Path:
    """The built .xctestrun with the run's variables added for the Voxt test host."""
    source = next((work / "derived/Build/Products").glob("*.xctestrun"))
    plan = plistlib.loads(source.read_bytes())
    # Xcode writes either format: TestConfigurations with TestTargets, or one entry per target at the top.
    if "TestConfigurations" in plan:
        targets = [target for configuration in plan["TestConfigurations"] for target in configuration["TestTargets"]]
    else:
        targets = [value for value in plan.values() if isinstance(value, dict) and "TestHostPath" in value]
    for target in targets:
        for field in ("EnvironmentVariables", "TestingEnvironmentVariables"):
            target.setdefault(field, {}).update(variables)
    # Paths inside are relative to the file's own folder.
    destination = source.with_name("serve.xctestrun")
    destination.write_bytes(plistlib.dumps(plan))
    return destination


def send(url: str, token: str, path: str, method: str = "GET", body: bytes | None = None, content_type: str | None = None) -> Response:
    """One request through curl, so each check can be repeated by hand."""
    command = ["curl", "-sS", "-o", "-", "-w", "\n%{http_code}", "-X", method, "-H", f"Authorization: Bearer {token}"]
    if content_type is not None:
        command += ["-H", f"Content-Type: {content_type}"]
    else:
        pass
    if body is not None:
        command += ["--data-binary", "@-"]
    else:
        pass
    finished = subprocess.run(command + [url + path], input=body, capture_output=True, timeout=600)
    text, _, status = finished.stdout.decode().rpartition("\n")
    return Response(status=int(status or 0), body=json.loads(text) if text.strip() else None)


def field_names(value: object) -> list[str]:
    return sorted(value) if isinstance(value, dict) else []


def text_of(response: Response) -> str:
    return str(response.body.get("text", "")) if isinstance(response.body, dict) else ""


def run_checks(url: str, token: str) -> list[Check]:
    contract = {path.stem: json.loads(path.read_text()) for path in CONTRACT.glob("*.json")}
    speech = (TESTDATA / "jfk.wav").read_bytes()
    checks = []

    health = send(url, token, "/v1/health")
    checks.append(Check(name="health", response=health, compared="keys, app and ok as in health.json", passed=health.status == 200 and field_names(health.body) == field_names(contract["health"]) and health.body.get("app") == "Voxt" and health.body.get("ok") is True))

    capabilities = send(url, token, "/v1/capabilities")
    expected = {key: value for key, value in contract["capabilities"].items() if key != "version"}
    actual = {key: value for key, value in capabilities.body.items() if key != "version"} if isinstance(capabilities.body, dict) else {}
    checks.append(Check(name="capabilities", response=capabilities, compared="equal to capabilities.json apart from version", passed=capabilities.status == 200 and actual == expected))

    english = send(url, token, "/v1/dictations?language=auto", "POST", speech, "audio/wav")
    checks.append(Check(name="dictation-jfk", response=english, compared="keys of dictation-response.json; the text has 'country'", passed=english.status == 200 and field_names(english.body) == field_names(contract["dictation-response"]) and "country" in text_of(english).lower()))

    chinese = send(url, token, "/v1/dictations?language=auto", "POST", (TESTDATA / "zh.wav").read_bytes(), "audio/wav")
    checks.append(Check(name="dictation-zh", response=chinese, compared="the text is in Chinese characters", passed=chinese.status == 200 and CHINESE_CHARACTER.search(text_of(chinese)) is not None))

    compose_request = json.loads((CONTRACT / "compose-request.json").read_text())
    compose_request["audio"] = base64.b64encode(speech).decode()
    translated = send(url, token, "/v1/compose", "POST", json.dumps(compose_request).encode(), "application/json")
    checks.append(Check(name="compose-translate", response=translated, compared="keys of compose-response.json; jfk.wav translated to zh-Hans", passed=translated.status == 200 and field_names(translated.body) == field_names(contract["compose-response"]) and translated.body.get("mode") == "translate" and CHINESE_CHARACTER.search(text_of(translated)) is not None))

    prompts = send(url, token, "/v1/compose/prompts")
    prompt_modes = field_names(prompts.body.get("prompts")) if isinstance(prompts.body, dict) else []
    checks.append(Check(name="compose-prompts", response=prompts, compared="one built-in prompt per mode", passed=prompts.status == 200 and prompt_modes == ["dictate", "edit", "translate", "write"]))

    if "corrections" in contract["capabilities"]["endpoints"]:
        correction = send(url, token, "/v1/corrections", "POST", (CONTRACT / "correction-request.json").read_bytes(), "application/json")
        pair = correction.body.get("correction") if isinstance(correction.body, dict) else "missing"
        checks.append(Check(name="correction", response=correction, compared="shape of correction-response.json (the pair, or null)", passed=correction.status == 200 and (pair is None or field_names(pair) == ["original", "replacement"])))
    else:
        pass

    for name, path, credential in [("unauthorized", "/v1/health", "wrong-token"), ("notFound", "/v1/nothing", token), ("wrongMethod", "/v1/dictations", token)]:
        error = send(url, credential, path)
        checks.append(Check(name=f"error-{name}", response=error, compared="equal to errors.json", passed={"status": error.status, "body": error.body} == contract["errors"][name]))
    return checks


def write_results(checks: list[Check], host: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    for check in checks:
        body = check.response.body
        if isinstance(body, dict) and "id" in body:
            body = {**body, "id": SAMPLE_ID}
        else:
            pass
        saved = {"status": check.response.status, "body": body}
        (RESULTS / f"{check.name}.json").write_text(json.dumps(saved, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    passed = sum(check.passed for check in checks)
    lines = [f"Host: {host}. {passed} of {len(checks)} checks passed.", "", "| Check | Result | Compared |", "| --- | --- | --- |"]
    lines += [f"| {check.name} | {'pass' if check.passed else 'FAIL'} | {check.compared} |" for check in checks]
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def wait_for_health(url: str, token: str, server: subprocess.Popen[bytes], log: Path) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    probe = urllib.request.Request(url + "/v1/health", headers={"Authorization": f"Bearer {token}"})
    while True:
        try:
            urllib.request.urlopen(probe, timeout=2)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if server.poll() is not None or time.monotonic() > deadline:
                sys.exit(f"The Local API did not answer on {url}; see {log}.")
            else:
                time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=["xctest", "app"], default="xctest")
    parser.add_argument("--models", type=Path, required=True, help="folder holding the mlx-audio and mlx-llm model folders")
    parser.add_argument("--asr-model", default="mlx-community/Qwen3-ASR-1.7B-6bit")
    parser.add_argument("--text-model", default="mlx-community/Qwen3.5-4B-OptiQ-4bit")
    parser.add_argument("--work", type=Path, default=TREE / "build/loopback-e2e", help="build output and the hand-off files")
    parser.add_argument("--packages", type=Path, help="Swift package checkouts to reuse (default: under --work)")
    parser.add_argument("--port", type=int, default=0, help="loopback port; 0 picks a free one")
    parser.add_argument("--serve-seconds", type=int, default=900, help="how long the Local API stays up, for other clients such as the iPhone simulator")
    parser.add_argument("--skip-build", action="store_true")
    options = parser.parse_args()

    work = options.work.resolve()
    (work / "tmp").mkdir(parents=True, exist_ok=True)
    packages = (options.packages or work / "packages").resolve()
    port = options.port or free_port()
    url = f"http://127.0.0.1:{port}"
    if options.host == "app":
        print(f"xcodebuild {' '.join(build_arguments(work, packages, 'build'))}")
        print(f"Then open the built Voxt.app, turn on Settings > General > Local API on port {port}, copy the token, and send the requests in run_checks to {url}.")
        return 2
    else:
        pass

    environment = {**os.environ, "TMPDIR": str(work / "tmp") + "/", "GIT_TERMINAL_PROMPT": "0"}
    if not options.skip_build:
        log = work / "build-for-testing.log"
        with open(log, "w") as output:
            built = subprocess.run(["nice", "-n", "10", "xcodebuild", *build_arguments(work, packages, "build-for-testing")], stdout=output, stderr=subprocess.STDOUT, env=environment)
        if built.returncode != 0:
            sys.exit(f"build-for-testing failed; see {log}.")
        else:
            pass
    else:
        pass

    token = secrets.token_urlsafe(32)
    home = isolated_home(work, options.models.resolve())
    xctestrun = serving_xctestrun(work, {"CFFIXED_USER_HOME": str(home), "VOXT_LOCAL_API_TOKEN": token, "VOXT_E2E_SERVE_SECONDS": str(options.serve_seconds), "VOXT_E2E_PORT": str(port), "VOXT_E2E_ASR_MODEL": options.asr_model, "VOXT_E2E_TEXT_MODEL": options.text_model})
    hand_off = work / "local-api.env"
    ready = work / "ready"
    hand_off.unlink(missing_ok=True)
    ready.unlink(missing_ok=True)
    serve_log = work / "serve.log"
    with open(serve_log, "w") as output:
        server = subprocess.Popen(["xcodebuild", "test-without-building", "-xctestrun", str(xctestrun), "-destination", "platform=macOS", "-only-testing:VoxtTests/LocalAPILoopbackServeTests"], stdout=output, stderr=subprocess.STDOUT, env=environment)
        try:
            wait_for_health(url, token, server, serve_log)
            hand_off.write_text(f"VOXT_E2E_URL={url}\nVOXT_E2E_TOKEN={token}\n")
            hand_off.chmod(0o600)
            ready.touch()
            print(f"Local API up on {url}; hand-off file {hand_off}")
            checks = run_checks(url, token)
            write_results(checks, "XCTest host (LocalAPILoopbackServeTests)")
            print(f"Serving until the test's {options.serve_seconds} s are up.")
            server.wait()
        finally:
            hand_off.unlink(missing_ok=True)
            ready.unlink(missing_ok=True)
            if server.poll() is None:
                server.terminate()
            else:
                pass
    return 0 if all(check.passed for check in checks) and server.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
