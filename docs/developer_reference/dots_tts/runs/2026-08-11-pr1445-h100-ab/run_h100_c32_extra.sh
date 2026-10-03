#!/usr/bin/env bash
set -uo pipefail

RUN_ID="$1"
RUN_ROOT="/data/dots_seedtts_runs/${RUN_ID}"
BASE_ROOT="${RUN_ROOT}/source/base"
CANDIDATE_ROOT="${RUN_ROOT}/source/candidate"
VENV_ROOT="${RUN_ROOT}/venv"

source "${VENV_ROOT}/bin/activate"

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
  local round="$5"
  local port="$6"
  local label="${revision}-c32-r${round}-gpu${physical_gpu}"
  local output_dir="${RUN_ROOT}/results/${label}"
  local log_path="${RUN_ROOT}/logs/${label}.log"
  local status_path="${output_dir}/status.json"

  mkdir -p "${output_dir}"
  python - "${status_path}" "${revision}" "${round}" \
    "${local_gpu}" "${physical_gpu}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "RUNNING",
    "revision": sys.argv[2],
    "concurrency": 32,
    "round": int(sys.argv[3]),
    "container_local_gpu_id": int(sys.argv[4]),
    "physical_gpu_id": int(sys.argv[5]),
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
        --max-concurrency 32 \
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
  local round="$1"
  local gpu_zero_revision="$2"
  local gpu_one_revision="$3"
  local port_zero="$4"
  local port_one="$5"
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

  record_cpu start "c32-r${round}"
  run_one "${gpu_zero_revision}" "${gpu_zero_root}" 0 0 \
    "${round}" "${port_zero}" &
  local pid_zero=$!
  run_one "${gpu_one_revision}" "${gpu_one_root}" 1 1 \
    "${round}" "${port_one}" &
  local pid_one=$!
  if ! wait "${pid_zero}"; then
    pair_exit=1
  fi
  if ! wait "${pid_one}"; then
    pair_exit=1
  fi
  record_cpu end "c32-r${round}"
  return "${pair_exit}"
}

OVERALL_EXIT=0
if ! run_pair 3 base candidate 48271 63347; then
  OVERALL_EXIT=1
fi
if ! run_pair 4 candidate base 35963 51739; then
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
