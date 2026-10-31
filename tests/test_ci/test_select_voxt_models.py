import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / ".github/scripts/select_voxt_models.py"
SPEC = importlib.util.spec_from_file_location("select_voxt_models", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
MODEL_JOBS = json.loads((ROOT / MODULE.MODEL_JOBS_PATH).read_text())
ALL_NAMES = [model["name"] for model in MODEL_JOBS["models"]]


def selected_names(
    changed_files: list[str],
    requested: str | None = None,
    changed_pins: frozenset[str] = frozenset(),
) -> list[str]:
    return [
        model["name"]
        for model in MODULE.select_models(
            MODEL_JOBS, changed_files, requested, changed_pins
        )
    ]


@pytest.mark.parametrize(
    ("changed_file", "expected"),
    [
        ("sglang_omni_mlx/native/src/music3/music3_service.cpp", ["music3"]),
        ("sglang_omni_mlx/native/ci/model_golden/whisper-small.json", ["whisper"]),
        ("sglang_omni_mlx/native/ci/test_voxtral_tts_api.py", ["voxtral-tts"]),
        ("sglang_omni_mlx/native/src/qwen3_tts/qwen3_tts_server.cpp", ["qwen3-tts"]),
        ("sglang_omni_mlx/native/ci/test_cosyvoice3_api.py", ["cosyvoice3"]),
        ("Voxt/backend/README.md", []),
    ],
)
def test_model_paths_select_only_their_model(
    changed_file: str, expected: list[str]
) -> None:
    assert selected_names([changed_file]) == expected


@pytest.mark.parametrize(
    "changed_file",
    [
        "sglang_omni_mlx/native/src/common/http.cpp",
        "sglang_omni_mlx/native/src/qwen3_asr/qwen3_decoder.cpp",
        "sglang_omni_mlx/native/CMakeLists.txt",
        ".github/workflows/voxt-mac-ci.yaml",
    ],
)
def test_shared_paths_select_every_model(changed_file: str) -> None:
    assert selected_names([changed_file]) == ALL_NAMES


def test_requested_models_override_changed_files() -> None:
    assert selected_names(
        ["sglang_omni_mlx/native/src/common/http.cpp"], "music3, cohere"
    ) == ["cohere", "music3"]
    assert selected_names([], "all") == ALL_NAMES


def test_unknown_requested_model_fails() -> None:
    with pytest.raises(SystemExit):
        selected_names([], "not-a-model")


def test_every_model_job_names_existing_pins_and_goldens() -> None:
    pins = json.loads((ROOT / "sglang_omni_mlx/native/ci/model_pins.json").read_text())
    qwen3_asr_pins = json.loads((ROOT / MODULE.QWEN3_ASR_PINS_PATH).read_text())
    golden_directory = ROOT / "sglang_omni_mlx/native/ci/model_golden"
    for model in MODEL_JOBS["models"]:
        assert set(model["pins"]) <= set(pins), model["name"]
        assert set(model["qwen3_asr_pins"]) <= set(qwen3_asr_pins), model["name"]
        for golden in model["goldens"]:
            assert (golden_directory / golden).is_file(), golden
        for pattern in model["paths"]:
            re.compile(pattern)
    listed = {golden for model in MODEL_JOBS["models"] for golden in model["goldens"]}
    disabled = set(MODEL_JOBS["disabled_goldens"]["goldens"])
    assert not listed & disabled
    assert listed | disabled == {path.name for path in golden_directory.glob("*.json")}


def test_no_checkpoint_is_pinned_in_both_pin_files() -> None:
    """Both provisioning scripts write a repo to one directory, so a repo pinned twice can be overwritten by the other pin."""
    pins = json.loads((ROOT / MODULE.PINS_PATH).read_text())
    qwen3_asr_pins = json.loads((ROOT / MODULE.QWEN3_ASR_PINS_PATH).read_text())
    assert not pins.keys() & qwen3_asr_pins.keys()


@pytest.mark.parametrize(
    ("changed_file", "expected"),
    [
        (
            "sglang_omni_mlx/native/src/silero_vad/silero_vad.cpp",
            ["cohere", "sensevoice"],
        ),
        (
            "sglang_omni_mlx/native/src/sortformer/sortformer_state.cpp",
            ["nemotron", "parakeet"],
        ),
    ],
)
def test_shared_model_libraries_select_their_dependents(
    changed_file: str, expected: list[str]
) -> None:
    assert selected_names([changed_file]) == expected


@pytest.mark.parametrize(
    ("changed_pins", "expected"),
    [
        (frozenset({"mlx-community/silero-vad-v6"}), ["cohere", "sensevoice"]),
        (frozenset({"mlx-community/whisper-small-mlx"}), ["whisper"]),
        (frozenset({"mlx-community/Qwen3-ASR-0.6B-4bit"}), ["voxtral-tts"]),
        (frozenset({"some/unused-model"}), []),
    ],
)
def test_changed_pins_select_the_models_that_use_them(
    changed_pins: frozenset[str], expected: list[str]
) -> None:
    assert (
        selected_names(
            ["sglang_omni_mlx/native/ci/model_pins.json"], None, changed_pins
        )
        == expected
    )


@pytest.mark.parametrize(
    ("relative_pins_path", "base_pins_flag", "changed_repo", "expected"),
    [
        (
            MODULE.PINS_PATH,
            "--base-pins",
            "mlx-community/parakeet-tdt-0.6b-v3",
            ["parakeet"],
        ),
        (
            MODULE.QWEN3_ASR_PINS_PATH,
            "--base-qwen3-asr-pins",
            "mlx-community/Qwen3-ASR-0.6B-4bit",
            ["voxtral-tts"],
        ),
    ],
)
def test_base_pins_diff_selects_changed_checkpoints(
    tmp_path,
    monkeypatch,
    relative_pins_path: Path,
    base_pins_flag: str,
    changed_repo: str,
    expected: list[str],
) -> None:
    base_pins = json.loads((ROOT / relative_pins_path).read_text())
    base_pins[changed_repo] = {"revision": "old", "files": {}}
    (tmp_path / "base_pins.json").write_text(json.dumps(base_pins))
    (tmp_path / "changed_files.json").write_text(json.dumps([str(relative_pins_path)]))
    output = tmp_path / "github_output"
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(
        "sys.argv",
        [
            "select_voxt_models.py",
            "--changed-files",
            str(tmp_path / "changed_files.json"),
            base_pins_flag,
            str(tmp_path / "base_pins.json"),
            "--github-output",
            str(output),
        ],
    )
    MODULE.main()
    matrix = json.loads(output.read_text().splitlines()[0].split("=", 1)[1])
    assert [model["name"] for model in matrix["model"]] == expected
