#!/usr/bin/env bash
set -euo pipefail

host=""
run_root=""
workers=()

while (($#)); do
  case "$1" in
    --host)
      host="$2"
      shift 2
      ;;
    --run-root)
      run_root="$2"
      shift 2
      ;;
    --worker)
      workers+=("$2")
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${host}" || -z "${run_root}" || ${#workers[@]} -eq 0 ]]; then
  echo "usage: $0 --host HOST --run-root PATH --worker CONCURRENCY,GPU,PORT,SAMPLES [...]" >&2
  exit 2
fi

expected_code_base="7f5529ad8f8b0e3baff3391946b51474eec45832"
repo_root="$(pwd -P)"
actual_commit="$(git -c safe.directory="${repo_root}" rev-parse HEAD)"
git -c safe.directory="${repo_root}" merge-base --is-ancestor \
  "${expected_code_base}" "${actual_commit}"
git -c safe.directory="${repo_root}" diff --quiet \
  "${expected_code_base}" "${actual_commit}" -- \
  sglang_omni benchmarks examples/configs

host_root="${run_root}/${host}"
mkdir -p "${host_root}"
printf '%s\n' "${actual_commit}" >"${host_root}/git_commit.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader >"${host_root}/launch_nvidia_smi.csv"
python -m pip install -e . --no-deps
python -m pip install \
  dots.tts==0.2.1 \
  jiwer==4.0.0 \
  openai-whisper==20250625
python -m benchmarks.dataset.prepare --dataset seedtts
python -c 'from huggingface_hub import snapshot_download; snapshot_download("dots-studio/dots.tts-mf"); snapshot_download("Qwen/Qwen3-ASR-1.7B")'

pids=()

terminate_children() {
  for pid in "${pids[@]:-}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
}

trap terminate_children INT TERM

run_worker() {
  local spec="$1"
  local concurrency gpu port samples
  IFS=',' read -r concurrency gpu port samples <<<"${spec}"
  local output_dir="${host_root}/c${concurrency}"
  mkdir -p "${output_dir}"
  printf '%s\n' \
    "host=${host}" \
    "gpu=${gpu}" \
    "port=${port}" \
    "concurrency=${concurrency}" \
    "samples=${samples}" \
    >"${output_dir}/worker_spec.txt"

  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model dots-studio/dots.tts-mf \
    --ref-format references \
    --server-config examples/configs/dots_tts.yaml \
    --lang en \
    --stream \
    --seed 42 \
    --warmup 10 \
    --max-samples "${samples}" \
    --max-concurrency "${concurrency}" \
    --max-running-requests 16 \
    --cuda-graph-max-bs 16 \
    --port "${port}" \
    --server-timeout 1800 \
    --skip-gpu-cleanup \
    --disable-tqdm \
    --output-dir "${output_dir}" \
    >"${output_dir}/worker.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "${status}" >"${output_dir}/exit_code"
  return "${status}"
}

for worker in "${workers[@]}"; do
  run_worker "${worker}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

printf '%s\n' "${status}" >"${host_root}/exit_code"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader >"${host_root}/final_nvidia_smi.csv"
exit "${status}"
