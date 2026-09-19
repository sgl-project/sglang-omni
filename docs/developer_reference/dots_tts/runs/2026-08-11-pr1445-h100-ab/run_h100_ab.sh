#!/usr/bin/env bash
set -uo pipefail

RUN_ID="$1"
BASE_COMMIT="$2"
CANDIDATE_COMMIT="$3"

RUN_ROOT="/data/dots_seedtts_runs/${RUN_ID}"
GIT_ROOT="${RUN_ROOT}/git"
BASE_ROOT="${RUN_ROOT}/source/base"
CANDIDATE_ROOT="${RUN_ROOT}/source/candidate"
VENV_ROOT="${RUN_ROOT}/venv"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/results" "${RUN_ROOT}/source"
nvidia-smi >"${RUN_ROOT}/launch-nvidia-smi.txt"

if [[ ! -d "${GIT_ROOT}/.git" ]]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/sgl-project/sglang-omni.git "${GIT_ROOT}"
fi
git -C "${GIT_ROOT}" fetch origin \
  "${BASE_COMMIT}" \
  "${CANDIDATE_COMMIT}"
if [[ ! -e "${BASE_ROOT}/.git" ]]; then
  git -C "${GIT_ROOT}" worktree add --detach "${BASE_ROOT}" "${BASE_COMMIT}"
fi
if [[ ! -e "${CANDIDATE_ROOT}/.git" ]]; then
  git -C "${GIT_ROOT}" worktree add --detach "${CANDIDATE_ROOT}" "${CANDIDATE_COMMIT}"
fi

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${VENV_ROOT}"
fi
source "${VENV_ROOT}/bin/activate"
python -m pip install \
  "dots.tts==0.2.1" \
  "jiwer" \
  "msgpack>=1.0.0" \
  "pytest" \
  >"${RUN_ROOT}/logs/dependencies.log" 2>&1

python - "${RUN_ROOT}" "${RUN_ID}" "${BASE_COMMIT}" "${CANDIDATE_COMMIT}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_root = Path(sys.argv[1])
payload = {
    "run_id": sys.argv[2],
    "created_at": datetime.now(timezone.utc).isoformat(),
    "host": "h100",
    "hardware": "NVIDIA H100",
    "physical_gpu_ids": [0, 1],
    "base_commit": sys.argv[3],
    "candidate_commit": sys.argv[4],
    "execution": "two-round GPU crossover per concurrency",
    "concurrency_levels": [16, 32],
    "samples_per_run": 1088,
    "stream": False,
    "seed": 42,
    "warmup_requests": 10,
    "server_config": "examples/configs/dots_tts.yaml",
    "max_running_requests": 16,
    "cuda_graph_max_bs": 16,
    "generate_only": True,
    "model": "dots-studio/dots.tts-mf",
    "dataset": "zhaochenyang20/seed-tts-eval-arrow",
    "dataset_split": "en",
}
(run_root / "contract.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

(
  cd "${CANDIDATE_ROOT}"
  PYTHONPATH="${CANDIDATE_ROOT}" python -m pytest -q \
    tests/unit_test/dots_tts/test_reference_encode_batching.py \
    tests/unit_test/dots_tts/test_vocoder.py \
    tests/unit_test/dots_tts/test_vocoder_streaming.py
) >"${RUN_ROOT}/logs/focused-tests.log" 2>&1
TEST_EXIT=$?
printf '%s\n' "${TEST_EXIT}" >"${RUN_ROOT}/focused-tests.exit"
if [[ "${TEST_EXIT}" -ne 0 ]]; then
  python - "${RUN_ROOT}" "${TEST_EXIT}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_root = Path(sys.argv[1])
(run_root / "host-summary.json").write_text(json.dumps({
    "status": "FAILED_FOCUSED_TESTS",
    "focused_test_exit_code": int(sys.argv[2]),
    "ended_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
  exit "${TEST_EXIT}"
fi

record_cpu() {
  local phase="$1"
  local label="$2"
  printf '%s phase=%s label=%s loadavg=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${phase}" \
    "${label}" \
    "$(tr ' ' ',' </proc/loadavg)" \
    >>"${RUN_ROOT}/host-cpu-snapshots.log"
}

run_one() {
  local revision="$1"
  local source_root="$2"
  local local_gpu="$3"
  local physical_gpu="$4"
  local concurrency="$5"
  local round="$6"
  local port="$7"
  local label="${revision}-c${concurrency}-r${round}-gpu${physical_gpu}"
  local output_dir="${RUN_ROOT}/results/${label}"
  local log_path="${RUN_ROOT}/logs/${label}.log"
  local status_path="${output_dir}/status.json"

  mkdir -p "${output_dir}"
  python - "${status_path}" "${revision}" "${concurrency}" "${round}" \
    "${local_gpu}" "${physical_gpu}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "RUNNING",
    "revision": sys.argv[2],
    "concurrency": int(sys.argv[3]),
    "round": int(sys.argv[4]),
    "container_local_gpu_id": int(sys.argv[5]),
    "physical_gpu_id": int(sys.argv[6]),
    "samples": 1088,
    "started_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY

  (
    cd "${source_root}"
    CUDA_VISIBLE_DEVICES="${local_gpu}" \
      PYTHONPATH="${source_root}" \
      python -m benchmarks.eval.benchmark_tts_seedtts \
        --generate-only \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --model dots-studio/dots.tts-mf \
        --server-config examples/configs/dots_tts.yaml \
        --ref-format references \
        --lang en \
        --seed 42 \
        --warmup 10 \
        --max-concurrency "${concurrency}" \
        --max-running-requests 16 \
        --cuda-graph-max-bs 16 \
        --device cuda:0 \
        --port "${port}" \
        --output-dir "${output_dir}" \
        --server-timeout 1800 \
        --skip-gpu-cleanup \
        --disable-tqdm
  ) >"${log_path}" 2>&1
  local exit_code=$?

  python - "${status_path}" "${exit_code}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["status"] = "COMPLETE" if int(sys.argv[2]) == 0 else "FAILED"
payload["exit_code"] = int(sys.argv[2])
payload["ended_at"] = datetime.now(timezone.utc).isoformat()
speed_path = path.parent / "speed_results.json"
if speed_path.exists():
    payload["speed_summary"] = json.loads(speed_path.read_text()).get("summary")
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
  return "${exit_code}"
}

run_pair() {
  local concurrency="$1"
  local round="$2"
  local gpu_zero_revision="$3"
  local gpu_one_revision="$4"
  local port_zero="$5"
  local port_one="$6"
  local gpu_zero_root
  local gpu_one_root
  local pair_exit=0

  if [[ "${gpu_zero_revision}" == "base" ]]; then
    gpu_zero_root="${BASE_ROOT}"
  else
    gpu_zero_root="${CANDIDATE_ROOT}"
  fi
  if [[ "${gpu_one_revision}" == "base" ]]; then
    gpu_one_root="${BASE_ROOT}"
  else
    gpu_one_root="${CANDIDATE_ROOT}"
  fi

  record_cpu start "c${concurrency}-r${round}"
  run_one "${gpu_zero_revision}" "${gpu_zero_root}" 0 0 \
    "${concurrency}" "${round}" "${port_zero}" &
  local pid_zero=$!
  run_one "${gpu_one_revision}" "${gpu_one_root}" 1 1 \
    "${concurrency}" "${round}" "${port_one}" &
  local pid_one=$!
  if ! wait "${pid_zero}"; then
    pair_exit=1
  fi
  if ! wait "${pid_one}"; then
    pair_exit=1
  fi
  record_cpu end "c${concurrency}-r${round}"
  return "${pair_exit}"
}

OVERALL_EXIT=0
if ! run_pair 16 1 base candidate 41783 58921; then
  OVERALL_EXIT=1
fi
if ! run_pair 16 2 candidate base 53669 39827; then
  OVERALL_EXIT=1
fi
if ! run_pair 32 1 base candidate 62137 45491; then
  OVERALL_EXIT=1
fi
if ! run_pair 32 2 candidate base 34757 56843; then
  OVERALL_EXIT=1
fi

python - "${RUN_ROOT}" "${OVERALL_EXIT}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_root = Path(sys.argv[1])
rows = []
for status_path in sorted((run_root / "results").glob("*/status.json")):
    rows.append(json.loads(status_path.read_text()))
payload = {
    "status": "COMPLETE" if int(sys.argv[2]) == 0 else "FAILED",
    "focused_test_exit_code": 0,
    "ended_at": datetime.now(timezone.utc).isoformat(),
    "results": rows,
}
(run_root / "host-summary.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

exit "${OVERALL_EXIT}"
