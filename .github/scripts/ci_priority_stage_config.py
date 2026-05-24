"""Stage registry used by the priority CI dispatcher."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from ci_priority_scheduler import StageSpec

DEFAULT_STAGE_CONFIG: dict[str, Any] = {
    "runs_on": ["self-hosted"],
    "job_timeout_minutes": 60,
    "run_timeout_minutes": 60,
    "container_image": "frankleeeee/sglang-omni:dev",
    "container_options": "--gpus all --rm -v /dev/shm:/dev/shm",
    "setup_action": "true",
    "venv_name": "omni",
    "install_deps": "false",
    "pre_run_script": "",
    "run_script": "",
    "save_cache_script": "",
    "save_cache_always": "false",
    "post_stage_label": "",
    "artifact_search_root": "/tmp",
    "artifact_path_globs": "",
    "artifact_upload_name": "",
    "artifact_upload_path": "",
    "artifact_if_no_files_found": "ignore",
    "summary_only": "false",
}


def _stage(
    *,
    id: str,
    workflow: str,
    name: str,
    order: int,
    depends_on: tuple[str, ...] = (),
    **overrides: Any,
) -> dict[str, Any]:
    config = deepcopy(DEFAULT_STAGE_CONFIG)
    config.update(
        {
            "id": id,
            "workflow": workflow,
            "name": name,
            "order": order,
            "depends_on": list(depends_on),
        }
    )
    config.update(overrides)
    return config


STAGES: list[dict[str, Any]] = [
    _stage(
        id="pr-test/unit-test",
        workflow="PR Test",
        name="unit-test",
        order=10,
        job_timeout_minutes=60,
        run_timeout_minutes=30,
        container_options="--gpus all --shm-size=2g --rm -v /dev/shm",
        setup_action="false",
        pre_run_script=r"""
if [ -d /github/home/omni ] && [ -n "$(ls -A /github/home/omni/)" ]; then
  cp -p -r /github/home/omni ./
fi

rm -rf /github/home/.cache/flashinfer

if [ -d omni ] && ! omni/bin/python -c "import torch" 2>/dev/null; then
  echo "Cached venv is corrupted, removing..."
  rm -rf omni
fi
if [ ! -d omni ]; then
  uv venv omni -p 3.11
fi
source omni/bin/activate
uv pip install -v -e .
if ! python -c "import av" 2>/dev/null; then
  echo "PyAV native libraries corrupted in cached venv, force-reinstalling..."
  uv pip install --force-reinstall --no-deps --no-cache av
fi

bash .github/scripts/delete_gpu_process.sh
""".strip(),
        run_script=r"""
source omni/bin/activate
export PYTHONPATH=$PWD
pytest tests/ -v -m "not benchmark and not docs" -x
""".strip(),
        save_cache_script="cp -p -r omni /github/home/",
    ),
    _stage(
        id="pr-test-examples/unit-test",
        workflow="PR Test (Examples)",
        name="unit-test",
        order=20,
        job_timeout_minutes=60,
        run_timeout_minutes=30,
        container_options="--gpus all --shm-size=2g --rm -v /dev/shm",
        setup_action="false",
        pre_run_script=r"""
if [ -d /github/home/omni ] && [ -n "$(ls -A /github/home/omni/)" ]; then
  cp -p -r /github/home/omni ./
fi

rm -rf /github/home/.cache/flashinfer

if [ -d omni ] && ! omni/bin/python -c "import torch" 2>/dev/null; then
  echo "Cached venv is corrupted, removing..."
  rm -rf omni
fi
if [ ! -d omni ]; then
  uv venv omni -p 3.11
fi
source omni/bin/activate
uv pip install -v -e .

bash .github/scripts/delete_gpu_process.sh
""".strip(),
        run_script=r"""
source omni/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
bash .github/scripts/run_examples.sh
""".strip(),
        save_cache_script="cp -p -r omni /github/home/",
    ),
    _stage(
        id="qwen3/stage-1-thinker",
        workflow="Qwen3-Omni CI",
        name="stage 1 - thinker length integration",
        order=100,
        job_timeout_minutes=10,
        venv_name="omni-qwen3",
        install_deps="true",
        run_script=r"""
source omni-qwen3/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_thinker_length.py -v -s -x
""".strip(),
        post_stage_label="thinker-length",
    ),
    _stage(
        id="qwen3/stage-2-tts",
        workflow="Qwen3-Omni CI",
        name="stage 2 - TTS speed + WER + SIM",
        order=110,
        job_timeout_minutes=10,
        venv_name="omni-qwen3",
        install_deps="true",
        pre_run_script=r"""
source omni-qwen3/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export SEEDTTS_SIM_CACHE_DIR=/github/home/seedtts-wavlm-sim
python -m benchmarks.metrics.speaker_similarity_assets --warm-cache
""".strip(),
        run_script=r"""
source omni-qwen3/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export SEEDTTS_SIM_CACHE_DIR=/github/home/seedtts-wavlm-sim
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_tts_ci.py -v -s -x
""".strip(),
        post_stage_label="TTS (speed + WER + SIM)",
        artifact_path_globs="*/speed_results.json\n*/wer_results.json\n*/similarity_results.json",
        artifact_upload_name="qwen3-omni-tts-results",
        artifact_upload_path="/tmp/**/*_results.json",
    ),
    _stage(
        id="qwen3/stage-3-mmmu",
        workflow="Qwen3-Omni CI",
        name="stage 3 - MMMU accuracy + speed",
        order=120,
        job_timeout_minutes=10,
        venv_name="omni-qwen3",
        install_deps="true",
        run_script=r"""
source omni-qwen3/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_mmmu_ci.py -v -s -x
""".strip(),
        post_stage_label="MMMU (accuracy + speed)",
        artifact_path_globs="*/mmmu/mmmu_results.json",
        artifact_upload_name="qwen3-omni-mmmu-results",
        artifact_upload_path="/tmp/**/mmmu_results.json",
    ),
    _stage(
        id="qwen3/stage-4-mmmu-talker",
        workflow="Qwen3-Omni CI",
        name="stage 4 - MMMU Talker",
        order=130,
        job_timeout_minutes=10,
        venv_name="omni-qwen3",
        install_deps="true",
        run_script=r"""
source omni-qwen3/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_mmmu_talker_ci.py -v -s -x
""".strip(),
        post_stage_label="MMMU Talker (WER + speed)",
        artifact_path_globs="*/mmmu_audio/mmmu_results.json",
        artifact_upload_name="qwen3-omni-mmmu-talker-results",
        artifact_upload_path="/tmp/**/mmmu_audio/mmmu_results.json",
    ),
    _stage(
        id="qwen3/stage-5-mmsu",
        workflow="Qwen3-Omni CI",
        name="stage 5 - MMSU accuracy + speed",
        order=140,
        job_timeout_minutes=10,
        venv_name="omni-qwen3",
        install_deps="true",
        run_script=r"""
source omni-qwen3/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_mmsu_ci.py -v -s -x
""".strip(),
        post_stage_label="MMSU (accuracy + speed, summary only)",
        artifact_path_globs="*/mmsu/mmsu_results.json",
        artifact_upload_name="qwen3-omni-mmsu-results",
        artifact_upload_path="/tmp/**/mmsu/mmsu_results.json",
        summary_only="true",
    ),
    _stage(
        id="qwen3/stage-6-mmsu-talker",
        workflow="Qwen3-Omni CI",
        name="stage 6 - MMSU Talker",
        order=150,
        job_timeout_minutes=10,
        venv_name="omni-qwen3",
        install_deps="true",
        run_script=r"""
source omni-qwen3/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_mmsu_talker_ci.py -v -s -x
""".strip(),
        post_stage_label="MMSU Talker (WER + speed)",
        artifact_path_globs="*/mmsu_audio/mmsu_results.json",
        artifact_upload_name="qwen3-omni-mmsu-talker-results",
        artifact_upload_path="/tmp/**/mmsu_audio/mmsu_results.json",
    ),
    _stage(
        id="qwen3/stage-7-videomme",
        workflow="Qwen3-Omni CI",
        name="stage 7 - Video-MME accuracy + speed",
        order=160,
        job_timeout_minutes=10,
        venv_name="omni-qwen3",
        install_deps="true",
        run_script=r"""
source omni-qwen3/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_videomme_ci.py -v -s -x
""".strip(),
        post_stage_label="Video-MME (accuracy + speed)",
        artifact_path_globs="*/videomme/videomme_results.json",
        artifact_upload_name="qwen3-omni-videomme-results",
        artifact_upload_path="/tmp/**/videomme_results.json",
    ),
    _stage(
        id="qwen3/stage-8-videomme-talker",
        workflow="Qwen3-Omni CI",
        name="stage 8 - Video-MME Talker",
        order=170,
        job_timeout_minutes=10,
        venv_name="omni-qwen3",
        install_deps="true",
        run_script=r"""
source omni-qwen3/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_videomme_talker_ci.py -v -s -x
""".strip(),
        post_stage_label="Video-MME Talker (WER + speed)",
        artifact_path_globs="*/videomme_audio/videomme_results.json",
        artifact_upload_name="qwen3-omni-videomme-talker-results",
        artifact_upload_path="/tmp/**/videomme_audio/videomme_results.json",
    ),
    _stage(
        id="qwen3/stage-9-videoamme",
        workflow="Qwen3-Omni CI",
        name="stage 9 - Video-AMME accuracy + speed",
        order=180,
        job_timeout_minutes=10,
        venv_name="omni-qwen3",
        install_deps="true",
        run_script=r"""
source omni-qwen3/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_videoamme_ci.py -v -s -x
""".strip(),
        post_stage_label="Video-AMME (accuracy + speed)",
        artifact_path_globs="*/videoamme/videoamme_results.json",
        artifact_upload_name="qwen3-omni-videoamme-results",
        artifact_upload_path="/tmp/**/videoamme_results.json",
    ),
    _stage(
        id="qwen3/stage-10-videoamme-talker",
        workflow="Qwen3-Omni CI",
        name="stage 10 - Video-AMME Talker",
        order=190,
        job_timeout_minutes=10,
        venv_name="omni-qwen3",
        install_deps="true",
        run_script=r"""
source omni-qwen3/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_videoamme_talker_ci.py -v -s -x
""".strip(),
        post_stage_label="Video-AMME Talker (WER + speed)",
        artifact_path_globs="*/videoamme_audio/videoamme_results.json",
        artifact_upload_name="qwen3-omni-videoamme-talker-results",
        artifact_upload_path="/tmp/**/videoamme_audio/videoamme_results.json",
    ),
    _stage(
        id="qwen3/stage-11-thinker-tp2",
        workflow="Qwen3-Omni CI",
        name="stage 11 - Thinker TP=2 (Video-AMME Talker)",
        order=200,
        job_timeout_minutes=10,
        venv_name="omni-qwen3",
        install_deps="true",
        run_script=r"""
source omni-qwen3/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_qwen3_omni_videoamme_talker_tp2_ci.py -v -s -x
""".strip(),
        post_stage_label="Thinker TP=2 (Video-AMME Talker accuracy + WER + speed)",
        artifact_path_globs="*/videoamme_audio/videoamme_results.json",
        artifact_upload_name="qwen3-omni-thinker-tp2-results",
        artifact_upload_path="/tmp/**/videoamme_audio/videoamme_results.json",
    ),
    _stage(
        id="s2pro/docs",
        workflow="S2-Pro CI",
        name="docs",
        order=300,
        job_timeout_minutes=10,
        venv_name="omni-s2pro",
        install_deps="true",
        run_script=r"""
source omni-s2pro/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
pytest tests/docs/s2pro/test_docs_tts_s2pro.py -v -s -x
""".strip(),
        post_stage_label="S2-Pro docs",
        save_cache_always="true",
        save_cache_script=r"""
if omni-s2pro/bin/python -c "import torch; import av; from whisper.normalizers import EnglishTextNormalizer" 2>/dev/null; then
  rm -rf /github/home/omni-s2pro
  cp -p -r omni-s2pro /github/home/
else
  echo "::warning::Skipping cache save - venv appears incomplete or corrupted"
fi
""".strip(),
    ),
    _stage(
        id="s2pro/stage-1-non-streaming",
        workflow="S2-Pro CI",
        name="stage 1 - non-streaming",
        order=310,
        depends_on=("s2pro/docs",),
        job_timeout_minutes=10,
        venv_name="omni-s2pro",
        pre_run_script=r"""
source omni-s2pro/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export SEEDTTS_SIM_CACHE_DIR=/github/home/seedtts-wavlm-sim
python -m benchmarks.metrics.speaker_similarity_assets --warm-cache
""".strip(),
        run_script=r"""
source omni-s2pro/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export SEEDTTS_SIM_CACHE_DIR=/github/home/seedtts-wavlm-sim
export S2PRO_STAGE_OUTPUT_ROOT="${GITHUB_WORKSPACE}/stage-results/nonstream"
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_s2pro_tts_ci.py -v -s -x --concurrency 16 --s2pro-stage s2pro-stage-1-nonstream
""".strip(),
        post_stage_label="S2-Pro stage 1 (non-streaming)",
        artifact_search_root="stage-results/nonstream",
        artifact_path_globs="*/speed_results.json\n*/wer_results.json\n*/similarity_results.json",
        artifact_upload_name="s2pro-stage-1-results",
        artifact_upload_path="stage-results/nonstream",
        artifact_if_no_files_found="error",
        save_cache_script=r"""
result_dir="/github/home/s2pro-ci/${OMNI_CI_PR_NUMBER}/${OMNI_CI_HEAD_SHA}/nonstream"
rm -rf "${result_dir}"
mkdir -p "${result_dir}"
cp -p -r stage-results/nonstream/. "${result_dir}/"
""".strip(),
    ),
    _stage(
        id="s2pro/stage-2-streaming",
        workflow="S2-Pro CI",
        name="stage 2 - streaming",
        order=320,
        depends_on=("s2pro/docs",),
        job_timeout_minutes=10,
        venv_name="omni-s2pro",
        run_script=r"""
source omni-s2pro/bin/activate
export PYTHONPATH=$PWD
export HF_ENDPOINT=https://hf-mirror.com
export S2PRO_STAGE_OUTPUT_ROOT="${GITHUB_WORKSPACE}/stage-results/stream"
export TORCHINDUCTOR_CACHE_DIR=/github/home/.torchinductor
bash .github/scripts/run_flaky_pytest.sh \
  pytest tests/test_model/test_s2pro_tts_ci.py -v -s -x --concurrency 16 --s2pro-stage s2pro-stage-2-stream
""".strip(),
        post_stage_label="S2-Pro stage 2 (streaming)",
        artifact_search_root="stage-results/stream",
        artifact_path_globs="*/speed_results.json\n*/wer_results.json",
        artifact_upload_name="s2pro-stage-2-results",
        artifact_upload_path="stage-results/stream",
        artifact_if_no_files_found="error",
        save_cache_script=r"""
result_dir="/github/home/s2pro-ci/${OMNI_CI_PR_NUMBER}/${OMNI_CI_HEAD_SHA}/stream"
rm -rf "${result_dir}"
mkdir -p "${result_dir}"
cp -p -r stage-results/stream/. "${result_dir}/"
""".strip(),
    ),
    _stage(
        id="s2pro/stage-3-consistency",
        workflow="S2-Pro CI",
        name="stage 3 - consistency",
        order=330,
        depends_on=("s2pro/stage-1-non-streaming", "s2pro/stage-2-streaming"),
        job_timeout_minutes=5,
        setup_action="false",
        run_script=r"""
python -m pip install --upgrade pip
python -m pip install pytest
export PYTHONPATH=$PWD
export S2PRO_STAGE1_SPEED_RESULTS_DIR="/github/home/s2pro-ci/${OMNI_CI_PR_NUMBER}/${OMNI_CI_HEAD_SHA}/nonstream"
export S2PRO_STAGE2_SPEED_RESULTS_DIR="/github/home/s2pro-ci/${OMNI_CI_PR_NUMBER}/${OMNI_CI_HEAD_SHA}/stream"
export S2PRO_CONSISTENCY_CONCURRENCY="16"
python -m pytest tests/test_model/test_s2pro_consistency_artifacts.py -v -s -x
""".strip(),
    ),
]


def stage_specs() -> list[StageSpec]:
    return [
        StageSpec(
            id=stage["id"],
            workflow=stage["workflow"],
            name=stage["name"],
            order=stage["order"],
            depends_on=tuple(stage["depends_on"]),
        )
        for stage in STAGES
    ]


def config_for_stage(stage_id: str) -> dict[str, Any]:
    for stage in STAGES:
        if stage["id"] == stage_id:
            return deepcopy(stage)
    raise KeyError(stage_id)


def dispatch_config_for_stage(stage_id: str) -> str:
    config = config_for_stage(stage_id)
    return json.dumps(config, separators=(",", ":"))
