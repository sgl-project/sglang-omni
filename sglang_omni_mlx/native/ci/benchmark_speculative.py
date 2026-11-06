# SPDX-License-Identifier: Apache-2.0
"""Run reproducible plain/draft/draft/plain corpus and long-audio comparisons."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import platform
import selectors
import subprocess
import time
from pathlib import Path

import requests
from check_golden import chip
from summarize_speculative import write_report

CI_DIRECTORY = Path(__file__).resolve().parent


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("whisper", "ming3"), required=True)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--draft-model-path", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--long-manifest", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=3600)
    arguments = parser.parse_args()
    if arguments.model == "whisper" and arguments.draft_model_path is None:
        parser.error("Whisper requires --draft-model-path")
    else:
        pass
    arguments.outputs.mkdir(parents=True, exist_ok=False)
    golden_name = (
        "whisper-large-v3" if arguments.model == "whisper" else "ming3-audio-tiny"
    )
    golden = json.loads(
        (CI_DIRECTORY / "model_golden" / f"{golden_name}.json").read_text()
    )
    corpus = [
        json.loads(line)
        for line in (CI_DIRECTORY / "corpus" / "manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    expected_hashes = {
        line.split()[1]: line.split()[0]
        for line in (CI_DIRECTORY / "corpus" / "v1.sha256").read_text().splitlines()
        if line.strip()
    }
    for clip in corpus:
        path = arguments.data_root / "corpus" / "v1" / "clips" / f"{clip['id']}.wav"
        relative = f"clips/{clip['id']}.wav"
        if file_hash(path) != expected_hashes[relative]:
            raise ValueError(f"corpus hash mismatch: {clip['id']}")
        else:
            clip["path"] = str(path.resolve())
            clip["suite"] = "corpus"
    long_clips = json.loads(arguments.long_manifest.read_text())
    for clip in long_clips:
        path = arguments.long_manifest.parent / clip["path"]
        if file_hash(path) != clip["sha256"]:
            raise ValueError(f"long-audio hash mismatch: {path}")
        else:
            clip["path"] = str(path.resolve())
            clip["suite"] = "long"
    if arguments.smoke:
        selected = []
        for language in ("en", "zh", "mixed", "none"):
            selected.extend([clip for clip in corpus if clip["lang"] == language][:2])
        corpus = selected
        long_clips = long_clips[:1]
    else:
        pass
    clips = corpus + long_clips
    binary = arguments.runtime_bin / (
        "whisper_server" if arguments.model == "whisper" else "ming3_audio_server"
    )
    repository = CI_DIRECTORY.parents[2]
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if dirty and not arguments.smoke:
        raise ValueError("full benchmarks require a clean committed source tree")
    else:
        pass
    if not arguments.smoke:
        build_record = json.loads(
            (arguments.runtime_bin / "build-provenance.json").read_text()
        )
        if build_record["source_commit"] != commit or build_record["binaries"][
            binary.name
        ] != file_hash(binary):
            raise ValueError(
                "binary provenance does not match the checked-out commit; rebuild"
            )
        else:
            pass
    else:
        pass
    provenance = dict(
        commit=commit,
        dirty=dirty,
        binary_sha256=file_hash(binary),
        chip=chip(),
        memory_bytes=int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        ),
        platform=platform.platform(),
        smoke=arguments.smoke,
        arguments={name: str(value) for name, value in vars(arguments).items()},
        corpus_manifest_sha256=file_hash(CI_DIRECTORY / "corpus" / "manifest.jsonl"),
        long_manifest_sha256=file_hash(arguments.long_manifest),
        pass_order=["plain", "draft", "draft", "plain"],
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    (arguments.outputs / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    for pass_index, mode in enumerate(provenance["pass_order"]):
        directory = arguments.outputs / f"{pass_index + 1}-{mode}"
        directory.mkdir()
        command = [
            str(binary.resolve()),
            "--supervised",
            "--model-directory",
            str(arguments.model_path.resolve()),
            "--model-name",
            golden["model"],
        ]
        if mode == "draft" and arguments.model == "whisper":
            command += [
                "--draft-model-path",
                str(arguments.draft_model_path.resolve()),
                "--draft-steps",
                "7",
            ]
        elif mode == "draft":
            command += ["--mtp", "1"]
        else:
            pass
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        (directory / "swap-before.txt").write_text(
            subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
        )
        with (directory / "server.stderr").open("w") as errors:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
            )
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    if not selector.select(timeout=600):
                        raise TimeoutError("server startup exceeded 600 seconds")
                    else:
                        ready = json.loads(process.stdout.readline())
                if ready.get("event") != "ready":
                    raise RuntimeError(f"server failed to start: {ready}")
                else:
                    (directory / "ready.json").write_text(
                        json.dumps(ready, indent=2) + "\n"
                    )
                endpoint = f"http://127.0.0.1:{ready['port']}"
                with (
                    requests.Session() as session,
                    (directory / "requests.jsonl").open("w") as output,
                ):
                    for index, clip in enumerate([clips[0], *clips]):
                        audio = Path(clip["path"]).read_bytes()
                        # note (ruiling): Warmup is recorded separately and excluded from all aggregates.
                        started = time.perf_counter()
                        if arguments.model == "whisper":
                            language = golden["language_by_clip_language"][clip["lang"]]
                            fields = dict(
                                stream="true",
                                include_generation_metadata="true",
                                temperature="0.0",
                                max_new_tokens="1024",
                            )
                            if language is not None:
                                fields["language"] = language
                            else:
                                pass
                            response = session.post(
                                endpoint + "/v1/audio/transcriptions",
                                data=fields,
                                files={"file": ("audio.wav", audio, "audio/wav")},
                                timeout=arguments.request_timeout,
                            )
                            response.raise_for_status()
                            events = [
                                json.loads(line[6:])
                                for line in response.content.decode(
                                    "utf-8"
                                ).splitlines()
                                if line.startswith("data: ") and line != "data: [DONE]"
                            ]
                            done = [
                                event
                                for event in events
                                if event.get("type") == "transcript.text.done"
                            ]
                            if len(done) != 1 or any(
                                event.get("type") == "error" for event in events
                            ):
                                raise RuntimeError(f"incomplete SSE response: {events}")
                            else:
                                metadata = done[0]["generation_metadata"]
                                transcript = {"text": done[0]["text"], **metadata}
                        else:
                            settings = golden["request"]
                            body = {
                                "model": golden["model"],
                                "temperature": 0,
                                "max_completion_tokens": (
                                    4096
                                    if clip["suite"] == "long"
                                    else settings["max_completion_tokens"]
                                ),
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": settings["system_prompt"],
                                    },
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": settings["prompt"],
                                            },
                                            {
                                                "type": "input_audio",
                                                "input_audio": {
                                                    "format": "wav",
                                                    "data": base64.b64encode(
                                                        audio
                                                    ).decode(),
                                                },
                                            },
                                        ],
                                    },
                                ],
                            }
                            response = session.post(
                                endpoint + "/v1/chat/completions",
                                json=body,
                                timeout=arguments.request_timeout,
                            )
                            response.raise_for_status()
                            reply = response.json()
                            choice = reply["choices"][0]
                            language, separator, text = choice["message"][
                                "content"
                            ].partition("\t")
                            if not separator:
                                text, language = language, ""
                            else:
                                pass
                            metadata = reply.get("generation_metadata", {})
                            transcript = dict(
                                text=text,
                                language=language,
                                finish_reason=choice["finish_reason"],
                                generated_token_count=reply["usage"][
                                    "completion_tokens"
                                ],
                            )
                        row = {
                            **transcript,
                            "id": clip["id"],
                            "lang": clip["lang"],
                            "suite": clip["suite"],
                            "seconds": time.perf_counter() - started,
                            "duration": clip["duration"],
                            "warmup": index == 0,
                            **{
                                field: metadata.get(field, 0)
                                for field in (
                                    "verify_round_count",
                                    "accepted_draft_token_count",
                                    "proposed_draft_token_count",
                                )
                            },
                        }
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        if index > 0:
                            print(
                                f"{pass_index + 1}/4 {mode} {index}/{len(clips)} {clip['id']}: {row['seconds']:.2f}s",
                                flush=True,
                            )
                        else:
                            pass
            finally:
                if process.poll() is None:
                    process.stdin.write('{"command":"shutdown"}\n')
                    process.stdin.flush()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                else:
                    pass
                (directory / "swap-after.txt").write_text(
                    subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
                )
            if process.returncode != 0:
                raise RuntimeError(
                    f"server exited {process.returncode}: see {directory}"
                )
            else:
                pass
    failed = write_report(
        arguments.outputs,
        CI_DIRECTORY / "model_golden" / f"{golden_name}.json",
        CI_DIRECTORY / "corpus" / "manifest.jsonl",
        arguments.long_manifest,
        arguments.smoke,
    )
    if failed:
        raise SystemExit(1)
    else:
        pass


if __name__ == "__main__":
    main()
else:
    pass
