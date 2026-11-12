# SPDX-License-Identifier: Apache-2.0
"""The installed runtime: what each binary serves and checks before it reads a checkpoint, and that the directory runs on its own."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from native_server import EVENT_TIMEOUT_SECONDS, EXIT_TIMEOUT_SECONDS, SupervisedServer

# The model kinds Voxt starts each server with.
SERVED_KINDS = {
    "qwen3_asr_server": ("qwen3_asr", "silero_vad", "sortformer"),
    "whisper_server": ("whisper",),
    "cohere_transcribe_server": ("cohere_transcribe",),
    "moss_transcribe_diarize_server": ("moss_transcribe_diarize",),
    "nemotron_asr_server": ("nemotron_asr",),
    "parakeet_server": ("parakeet",),
    "sensevoice_server": ("sensevoice",),
    "voxtral_tts_server": ("voxtral_tts",),
    "qwen3_tts_server": ("qwen3_tts",),
    "cosyvoice3_server": ("cosyvoice3",),
    "fishaudio_drama3_server": ("fishaudio_drama3",),
    "fish_s2_pro_server": ("fish_s2_pro",),
    "zonos3_server": ("zonos3",),
    "music3_server": ("minimax_music3",),
    "ming3_audio_server": ("ming3_audio",),
    "ming_tts_server": ("ming_tts",),
}
SYSTEM_LIBRARY_PREFIXES = ("/usr/lib/", "/System/Library/")


def mach_o_load_commands(path: Path) -> tuple[list[str], list[str]]:
    """The libraries path links and its run-path search entries, without its own install name."""
    libraries = subprocess.run(
        ["otool", "-L", str(path)], capture_output=True, text=True, check=True
    ).stdout.splitlines()[1:]
    install_name = subprocess.run(
        ["otool", "-D", str(path)], capture_output=True, text=True, check=True
    ).stdout.splitlines()[1:]
    load_commands = subprocess.run(
        ["otool", "-l", str(path)], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    run_paths = [
        load_commands[index + 2].split()[1]
        for index, line in enumerate(load_commands)
        if line.strip() == "cmd LC_RPATH"
    ]
    linked = [line.split()[0] for line in libraries]
    return [library for library in linked if library not in install_name], run_paths


def test_every_installed_server_is_listed(runtime_bin: Path) -> None:
    assert sorted(path.name for path in runtime_bin.glob("*_server")) == sorted(
        SERVED_KINDS
    )


@pytest.mark.parametrize("binary", sorted(SERVED_KINDS))
def test_a_server_serves_only_its_listed_kinds(runtime_bin: Path, binary: str) -> None:
    kinds = SERVED_KINDS[binary]
    expected = (
        f"only --model-kind {kinds[0]} is served"
        if len(kinds) == 1
        else f"--model-kind must be {', '.join(kinds[:-1])} or {kinds[-1]}"
    )
    completed = subprocess.run(
        [
            str(runtime_bin / binary),
            "--supervised",
            "--model-kind",
            "not_a_model_kind",
            "--model-directory",
            "x",
        ],
        capture_output=True,
        text=True,
        timeout=EXIT_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert expected in completed.stderr


@pytest.mark.parametrize(
    ("binary", "model_kind"),
    [
        (binary, kind)
        for binary, kinds in sorted(SERVED_KINDS.items())
        for kind in kinds
    ],
)
def test_a_load_without_a_checkpoint_reports_failed(
    runtime_bin: Path, tmp_path: Path, binary: str, model_kind: str
) -> None:
    """Voxt shows the failed event's reason; a crash in the loader sends none."""
    server = SupervisedServer(runtime_bin / binary, model_kind, tmp_path)
    assert server.first_event["event"] == "failed"
    assert server.first_event["reason"].startswith("model load failed: ")
    assert server.process.wait(timeout=EVENT_TIMEOUT_SECONDS) == 1


@pytest.mark.parametrize(
    ("binary", "flags", "message"),
    [
        ("qwen3_asr_server", ("--max-segment-seconds", "0"), "--max-segment-seconds must span one sample"),
        ("qwen3_asr_server", ("--max-segment-seconds", "0.00001"), "--max-segment-seconds must span one sample"),
        ("qwen3_asr_server", ("--decode-interval-ms", "0"), "--decode-interval-ms must be positive"),
        ("qwen3_asr_server", ("--first-decode-ms", "-1"), "--first-decode-ms must not be negative"),
        ("qwen3_asr_server", ("--draft-bits", "-1"), "--draft-bits must be 4, 6 or 8"),
        ("qwen3_asr_server", ("--draft-bits", "8"), "--draft-bits needs --draft-model-path"),
        ("moss_transcribe_diarize_server", ("--draft-bits", "-1"), "--draft-bits must be 4, 6 or 8"),
        ("moss_transcribe_diarize_server", ("--draft-bits", "8"), "--draft-bits needs --draft-model-path"),
        ("qwen3_asr_server", ("--cache-limit-mb", "-2"), "--cache-limit-mb must be 0 to 1073741824"),
        ("qwen3_asr_server", ("--max-batch", "0"), "--max-batch must be at least 1"),
        ("qwen3_asr_server", ("--max-batch", "-3"), "--max-batch must be at least 1"),
        ("music3_server", ("--request-timeout-s", "0"), "--request-timeout-s must be above 0"),
        ("music3_server", ("--stage-offload-components", "ar"), "requires both ar,dit"),
        ("music3_server", ("--stage-offload-components", "ar,dit,other"), "requires both ar,dit"),
        ("music3_server", ("--stage-offload-components", ""), "requires both ar,dit"),
        ("cosyvoice3_server", ("--max-queued-requests", "-1"), "--max-queued-requests must not be negative"),
        ("ming3_audio_server", ("--context-length", "0"), "--context-length must be a positive integer"),
        ("fish_s2_pro_server", ("--max-queued-requests", "-1"), "--max-queued-requests must not be negative"),
        ("fish_s2_pro_server", ("--request-timeout-s", "0"), "--request-timeout-s must be above 0"),
        ("zonos3_server", ("--max-queued-requests", "-1"), "--max-queued-requests must not be negative"),
        ("zonos3_server", ("--request-timeout-s", "0"), "--request-timeout-s must be above 0"),
    ],
)  # fmt: skip
def test_invalid_server_flags_exit_before_the_load(
    runtime_bin: Path, binary: str, flags: tuple[str, ...], message: str
) -> None:
    completed = subprocess.run(
        [str(runtime_bin / binary), "--supervised", "--model-directory", "x", *flags],
        capture_output=True,
        text=True,
        timeout=EXIT_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert message in completed.stderr


@pytest.mark.parametrize(
    ("binary", "flags", "message"),
    [
        *[
            ("qwen3_asr_transcribe", ("--repetition-penalty", penalty), "--repetition-penalty must be positive and finite")
            for penalty in ("0", "-1", "nan", "inf")
        ],
        ("music3_synthesize", ("--stage-offload-components", "dit"), "requires both ar,dit"),
        ("ming3_audio_chat", ("--context-length", "0"), "--context-length must be a positive integer"),
    ],
)  # fmt: skip
def test_invalid_tool_flags_fail_before_the_load(
    runtime_bin: Path, binary: str, flags: tuple[str, ...], message: str
) -> None:
    completed = subprocess.run(
        [str(runtime_bin / binary), "--model-path", "does-not-exist", *flags],
        capture_output=True,
        text=True,
        timeout=EXIT_TIMEOUT_SECONDS,
    )
    assert completed.returncode != 0
    assert message in completed.stderr


def test_the_runtime_links_only_the_system_and_itself(runtime_bin: Path) -> None:
    """A build-tree path left in a binary works on the build machine and nowhere else."""
    assert (runtime_bin / "mlx.metallib").is_file()
    for path in sorted(runtime_bin.iterdir()):
        if path.suffix == ".metallib":
            continue
        else:
            pass
        libraries, run_paths = mach_o_load_commands(path)
        assert set(run_paths) <= {"@loader_path"}, path.name
        for library in libraries:
            if library.startswith(SYSTEM_LIBRARY_PREFIXES):
                pass
            else:
                assert library.startswith("@rpath/"), (path.name, library)
                assert (runtime_bin / library.removeprefix("@rpath/")).is_file(), (
                    path.name,
                    library,
                )
