#!/usr/bin/env bash
set -euo pipefail

artifact_dir="${ROCM_CI_ARTIFACT_DIR:-/artifacts}"
expected_gpu_arch="${EXPECTED_GPU_ARCH:?EXPECTED_GPU_ARCH is required}"
mkdir -p "${artifact_dir}"

./scripts/rocm/install_rocm.sh --check | tee "${artifact_dir}/stack-check.log"
sgl-omni check-gpu --strict --json | tee "${artifact_dir}/gpu.json"

python3 - "${artifact_dir}/gpu.json" "${expected_gpu_arch}" <<'PY'
import json
import sys

report_path, expected = sys.argv[1:]
with open(report_path, encoding="utf-8") as report_file:
    report = json.load(report_file)
actual = {
    architecture.split(":", 1)[0]
    for architecture in report["environment"]["gpu_architectures"]
}
if actual != {expected}:
    raise SystemExit(f"expected only {expected}, found {sorted(actual)}")
PY

python3 -m pytest -q \
    tests/unit_test/config/test_gpu_transport_config.py \
    tests/unit_test/models/test_accelerator_support.py \
    tests/unit_test/test_platforms.py \
    tests/unit_test/pipeline/test_comm_router.py \
    tests/unit_test/relay/test_nixl_relay.py \
    tests/unit_test/rocm/test_install_contract.py \
    tests/unit_test/rocm/test_model_e2e_manifest.py \
    tests/unit_test/rocm/test_model_fallbacks.py \
    tests/unit_test/diagnostics/test_gpu.py \
    tests/unit_test/qwen3_omni/test_code2wav.py \
    tests/unit_test/qwen3_omni/test_fp8_backend_config.py \
    tests/unit_test/scheduling/test_engine_factory.py \
    tests/unit_test/minimax_music3/test_core.py \
    | tee "${artifact_dir}/pytest.log"
