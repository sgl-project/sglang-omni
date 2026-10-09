#!/usr/bin/env bash
set -uo pipefail

RUN_ID="$1"
SOURCE_COMMIT="$2"

RUN_ROOT="/data/dots_seedtts_runs/${RUN_ID}"
GIT_ROOT="${RUN_ROOT}/git"
SOURCE_ROOT="${RUN_ROOT}/source/main"
VENV_ROOT="${RUN_ROOT}/venv"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/results" "${RUN_ROOT}/source"
nvidia-smi >"${RUN_ROOT}/launch-nvidia-smi.txt"

if [[ ! -d "${GIT_ROOT}/.git" ]]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/sgl-project/sglang-omni.git "${GIT_ROOT}"
fi
git -C "${GIT_ROOT}" fetch origin "${SOURCE_COMMIT}"
if [[ ! -e "${SOURCE_ROOT}/.git" ]]; then
  git -C "${GIT_ROOT}" worktree add --detach "${SOURCE_ROOT}" "${SOURCE_COMMIT}"
fi

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${VENV_ROOT}"
fi
source "${VENV_ROOT}/bin/activate"
python -m pip install \
  "dots.tts==0.2.1" \
  "jiwer" \
  "msgpack>=1.0.0" \
  >"${RUN_ROOT}/logs/dependencies.log" 2>&1

python - "${RUN_ROOT}" "${RUN_ID}" "${SOURCE_COMMIT}" <<'PY'
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
    "source_commit": sys.argv[3],
    "execution": "same-concurrency dual-GPU repeats",
    "concurrency_levels": [1, 2, 4, 8],
    "sample_counts": {"1": 50, "2": 1088, "4": 1088, "8": 1088},
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

record_cpu() {
  local phase="$1"
  local concurrency="$2"
  printf '%s phase=%s concurrency=%s loadavg=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${phase}" \
    "${concurrency}" \
    "$(tr ' ' ',' </proc/loadavg)" \
    >>"${RUN_ROOT}/host-cpu-snapshots.log"
}

run_one() {
  local local_gpu="$1"
  local physical_gpu="$2"
  local concurrency="$3"
  local sample_count="$4"
  local port="$5"
  local label="main-c${concurrency}-gpu${physical_gpu}"
  local output_dir="${RUN_ROOT}/results/${label}"
  local log_path="${RUN_ROOT}/logs/${label}.log"
  local status_path="${output_dir}/status.json"
  local sample_args=()

  if [[ "${sample_count}" != "1088" ]]; then
    sample_args+=(--max-samples "${sample_count}")
  fi
  mkdir -p "${output_dir}"
  python - "${status_path}" "${concurrency}" "${sample_count}" \
    "${local_gpu}" "${physical_gpu}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "RUNNING",
    "revision": "main",
    "concurrency": int(sys.argv[2]),
    "samples": int(sys.argv[3]),
    "container_local_gpu_id": int(sys.argv[4]),
    "physical_gpu_id": int(sys.argv[5]),
    "started_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY

  (
    cd "${SOURCE_ROOT}"
    CUDA_VISIBLE_DEVICES="${local_gpu}" \
      PYTHONPATH="${SOURCE_ROOT}" \
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
        --disable-tqdm \
        "${sample_args[@]}"
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
  local sample_count="$2"
  local port_zero="$3"
  local port_one="$4"
  local pair_exit=0

  record_cpu start "${concurrency}"
  run_one 0 0 "${concurrency}" "${sample_count}" "${port_zero}" &
  local pid_zero=$!
  run_one 1 1 "${concurrency}" "${sample_count}" "${port_one}" &
  local pid_one=$!
  if ! wait "${pid_zero}"; then
    pair_exit=1
  fi
  if ! wait "${pid_one}"; then
    pair_exit=1
  fi
  record_cpu end "${concurrency}"
  return "${pair_exit}"
}

OVERALL_EXIT=0
if ! run_pair 1 50 47293 63157; then
  OVERALL_EXIT=1
fi
if ! run_pair 2 1088 35671 58439; then
  OVERALL_EXIT=1
fi
if ! run_pair 4 1088 62983 41867; then
  OVERALL_EXIT=1
fi
if ! run_pair 8 1088 54379 38741; then
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
    "ended_at": datetime.now(timezone.utc).isoformat(),
    "results": rows,
}
(run_root / "host-summary.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

exit "${OVERALL_EXIT}"
