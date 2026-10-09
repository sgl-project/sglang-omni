# SPDX-License-Identifier: Apache-2.0
"""CPU checks for model CI execution inside a provisioned test Pod."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/npu/run_model_ci.sh"
SWR_IMAGE = (
    "swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang-omni"
)


@pytest.fixture
def pod_environment(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repository = tmp_path / "checkout"
    script_directory = repository / "scripts/npu"
    script_directory.mkdir(parents=True)
    shutil.copy2(SCRIPT, script_directory / SCRIPT.name)
    (script_directory / "install_npu.sh").write_text(
        'echo "environment check"\nexit "${MOCK_HEALTH_EXIT_CODE:-0}"\n'
    )
    (repository / "pyproject_npu.toml").write_text("# NPU metadata\n")
    (repository / "pyproject.toml").write_text("# original metadata\n")
    inputs = tmp_path / "mounted-data"
    (inputs / "configs").mkdir(parents=True)
    for model in ("qwen3-tts", "qwen3-asr"):
        (inputs / "models" / model).mkdir(parents=True)
        (inputs / "models" / model / "config.json").write_text("{}")
        (inputs / "configs" / f"{model}.yaml").write_text("gpu: 0\n")
    (inputs / "models/qwen3-tts/speech_tokenizer").mkdir()
    (inputs / "asr").mkdir()
    (inputs / "asr/cases.json").write_text("{}")
    toolkit = tmp_path / "toolkit"
    toolkit.mkdir()
    (toolkit / "set_env.sh").write_text(
        ': "$CANN_OPTIONAL_VARIABLE"\n'
        'export PYTHONPATH="/cann/python${PYTHONPATH:+:$PYTHONPATH}"\n'
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    mock_command = binaries / "mock-command"
    mock_command.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "command = Path(sys.argv[0]).name\n"
        "with Path(os.environ['MOCK_COMMAND_LOG']).open('a') as log:\n"
        "    log.write(json.dumps([command, *sys.argv[1:]]) + '\\n')\n"
        "if command == 'docker':\n"
        "    sys.exit(99)\n"
        "elif command == 'python' and sys.argv[1:3] == ['-m', 'pytest']:\n"
        "    junit = Path(next(argument.split('=', 1)[1] for argument in sys.argv\n"
        "                      if argument.startswith('--junitxml=')))\n"
        "    environment = {key: value for key, value in os.environ.items()\n"
        "                   if key.startswith('OMNI_NPU_')\n"
        "                   or key in ('OMNI_RUN_NPU_TESTS', 'PYTHONPATH',\n"
        "                              'HF_HUB_OFFLINE', 'ASCEND_RT_VISIBLE_DEVICES',\n"
        "                              'ASCEND_VISIBLE_DEVICES')}\n"
        "    (junit.parent / 'execution.json').write_text(json.dumps({\n"
        "        'environment': environment, 'arguments': sys.argv[1:],\n"
        "        'source': str(Path.cwd()),\n"
        "        'metadata': Path('pyproject.toml').read_text(),\n"
        "    }))\n"
        "    junit.write_text('<testsuites/>')\n"
        "    print('model test executed')\n"
        "    sys.exit(int(os.environ.get('MOCK_PYTEST_EXIT_CODE', '0')))\n"
        "else:\n"
        "    print(command)\n"
    )
    mock_command.chmod(0o755)
    for command in ("python", "git", "npu-smi", "docker"):
        (binaries / command).symlink_to(mock_command)
    environment = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "TMPDIR": str(tmp_path),
        "NPU_CI_IMAGE": f"{SWR_IMAGE}@sha256:" + "a" * 64,
        "NPU_CI_DATA_DIR": str(inputs),
        "NPU_CI_MODEL": "qwen3-tts",
        "ASCEND_HOME_PATH": str(toolkit),
        "ASCEND_RT_VISIBLE_DEVICES": "7",
        "ASCEND_VISIBLE_DEVICES": "7",
        "PYTHONPATH": "/pod/python",
        "MOCK_COMMAND_LOG": str(tmp_path / "commands.jsonl"),
    }
    return repository, environment


@pytest.mark.parametrize("model", ["qwen3-tts", "qwen3-asr"])
@pytest.mark.parametrize("exit_code", [0, 1])
def test_run_in_pod_preserves_environment_and_exit_status(
    pod_environment: tuple[Path, dict[str, str]], model: str, exit_code: int
) -> None:
    repository, environment = pod_environment
    environment.update(NPU_CI_MODEL=model, MOCK_PYTEST_EXIT_CODE=str(exit_code))
    result = subprocess.run(
        ["bash", str(repository / "scripts/npu/run_model_ci.sh")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    results = next((repository / "npu-ci-results").iterdir())
    execution = json.loads((results / "execution.json").read_text())
    model_kind = "tts" if model == "qwen3-tts" else "asr"
    assert f"tests/test_model/test_npu_{model_kind}.py" in execution["arguments"]
    assert execution["source"] != str(repository)
    assert execution["metadata"] == "# NPU metadata\n"
    assert (repository / "pyproject.toml").read_text() == "# original metadata\n"
    actual_environment = execution["environment"]
    assert actual_environment["PYTHONPATH"] == (
        execution["source"] + ":/cann/python:/pod/python"
    )
    assert actual_environment["ASCEND_RT_VISIBLE_DEVICES"] == "7"
    assert actual_environment["ASCEND_VISIBLE_DEVICES"] == "7"
    assert actual_environment["OMNI_RUN_NPU_TESTS"] == "1"
    assert actual_environment["HF_HUB_OFFLINE"] == "1"
    prefix = f"OMNI_NPU_{model_kind.upper()}"
    assert actual_environment[f"{prefix}_MODEL"] == (
        environment["NPU_CI_DATA_DIR"] + f"/models/{model}"
    )
    assert actual_environment[f"{prefix}_OUTPUT"] == str(results / model_kind)
    assert (results / "junit.xml").is_file()
    assert (results / "image.txt").read_text().strip() == environment["NPU_CI_IMAGE"]
    assert "model test executed" in (results / "runner.log").read_text()
    commands = Path(environment["MOCK_COMMAND_LOG"]).read_text().splitlines()
    assert all(json.loads(command)[0] != "docker" for command in commands)


@pytest.mark.parametrize("failure", ["missing-model", "unhealthy-npu"])
def test_fail_before_installing_or_testing(
    pod_environment: tuple[Path, dict[str, str]], failure: str
) -> None:
    repository, environment = pod_environment
    if failure == "missing-model":
        (Path(environment["NPU_CI_DATA_DIR"]) / "models/qwen3-tts/config.json").unlink()
        expected_status = 2
    else:
        environment["MOCK_HEALTH_EXIT_CODE"] = "3"
        expected_status = 3
    result = subprocess.run(
        ["bash", str(repository / "scripts/npu/run_model_ci.sh")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == expected_status, result.stdout + result.stderr
    commands = Path(environment["MOCK_COMMAND_LOG"]).read_text().splitlines()
    assert all(json.loads(command)[0] != "python" for command in commands)
    results = next((repository / "npu-ci-results").iterdir())
    assert (results / "runner.log").is_file()


@pytest.mark.parametrize(
    "image,succeeds",
    [
        ("", False),
        (f"{SWR_IMAGE}:main-cann9.0.0-910b", False),
        (f"{SWR_IMAGE}@sha256:" + "a" * 64, True),
    ],
)
def test_preflight_requires_image_digest(image: str, succeeds: bool) -> None:
    workflow = yaml.safe_load(
        (SCRIPT.parents[2] / ".github/workflows/omni-npu-ci.yaml").read_text()
    )
    preflight = next(
        step["run"]
        for step in workflow["jobs"]["preflight"]["steps"]
        if step["name"] == "Validate integration configuration"
    )
    environment = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "NPU_CI_IMAGE": image,
        "NPU_CI_DATA_DIR": "/data",
    }
    result = subprocess.run(
        ["bash", "-c", preflight],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) == succeeds, result.stderr


@pytest.mark.parametrize(
    "key,value",
    [
        ("NPU_CI_IMAGE", "image:latest"),
        ("NPU_CI_DATA_DIR", "/"),
        ("NPU_CI_MODEL", "unknown"),
    ],
)
def test_invalid_configuration_is_rejected(
    key: str, value: str, tmp_path: Path
) -> None:
    environment = {
        **os.environ,
        "NPU_CI_IMAGE": "test@sha256:" + "a" * 64,
        "NPU_CI_DATA_DIR": str(tmp_path),
        "NPU_CI_MODEL": "qwen3-tts",
        key: value,
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert not list(tmp_path.iterdir())
