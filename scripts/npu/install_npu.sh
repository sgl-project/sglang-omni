#!/usr/bin/env bash
# Install sglang-omni against a pre-installed Ascend NPU stack.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYPROJECT="${REPO_ROOT}/pyproject.toml"
PYPROJECT_NPU="${REPO_ROOT}/pyproject_npu.toml"
BACKUP="${REPO_ROOT}/.pyproject.cuda.bak"
LOCK="${REPO_ROOT}/.pyproject.npu.lock"
SGLANG_SUPPORTED_RELEASE="0.5.18"

EDITABLE="-e"
CHECK_ONLY=0
SKIP_DEVICE_CHECK=0
EXTRAS=""
TARGET="."
PYBIN="${PYTHON:-python}"
INSTALL_CMD=()

usage() {
  cat <<'EOF'
Usage: scripts/npu/install_npu.sh [OPTIONS]

Install sglang-omni against an existing Ascend software stack. Prerequisites
are checked but never installed or modified by this script.

Options:
  --extras NAME[,NAME]  Install eval, all, or fun-cosyvoice3 extras.
  --no-editable         Perform a non-editable installation.
  --skip-device-check   Skip NPU availability and MatMul checks.
  --check               Check prerequisites and show commands without installing.
  -h, --help            Show this help message.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-editable)
        EDITABLE=""
        shift
        ;;
      --check)
        CHECK_ONLY=1
        shift
        ;;
      --skip-device-check)
        SKIP_DEVICE_CHECK=1
        shift
        ;;
      --extras)
        [[ $# -ge 2 && -n "${2:-}" ]] || {
          echo "ERROR: --extras requires a value" >&2
          exit 2
        }
        EXTRAS="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "ERROR: unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
}

configure_install() {
  [[ -f "${PYPROJECT_NPU}" ]] || {
    echo "ERROR: ${PYPROJECT_NPU} not found" >&2
    exit 1
  }

  if [[ -n "${EXTRAS}" ]]; then
    local extra
    local -a extra_names
    IFS=',' read -r -a extra_names <<< "${EXTRAS}"
    for extra in "${extra_names[@]}"; do
      case "${extra}" in
        eval|all|fun-cosyvoice3) ;;
        *)
          echo "ERROR: unsupported extra '${extra}'; choose eval, all, or fun-cosyvoice3" >&2
          exit 2
          ;;
      esac
    done
    TARGET=".[${EXTRAS}]"
  fi

  INSTALL_CMD=("${PYBIN}" -m pip install)
  [[ -n "${EDITABLE}" ]] && INSTALL_CMD+=("${EDITABLE}")
  INSTALL_CMD+=("${TARGET}")
}

print_summary() {
  echo "=== sglang-omni Ascend NPU install ==="
  echo "  repo:        ${REPO_ROOT}"
  echo "  python:      $("${PYBIN}" -c 'import sys; print(sys.executable)')"
  echo "  target:      ${TARGET}"
  echo "  editable:    $([[ -n "${EDITABLE}" ]] && echo yes || echo no)"
}

check_sglang_version() {
  local supported_line="${SGLANG_SUPPORTED_RELEASE} release line"
  local installed_version="not installed"

  if ! installed_version="$("${PYBIN}" -c 'from importlib.metadata import version; print(version("sglang"))' 2>/dev/null)"; then
    {
      echo "ERROR: SGLang version check failed."
      echo "  supported: ${supported_line}"
      echo "  installed: not installed"
      echo "Install a supported Ascend NPU release:"
      echo "  https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu"
    } >&2
    return 1
  fi

  # Compare the numeric release segment rather than PEP 440 precedence. This
  # treats dev, pre, final, post, and local builds on the configured release
  # line as supported while excluding every other release line.
  if ! "${PYBIN}" - "${installed_version}" "${SGLANG_SUPPORTED_RELEASE}" <<'PY'
import sys

try:
    from packaging.version import InvalidVersion, Version
except ImportError:
    from pip._vendor.packaging.version import InvalidVersion, Version

try:
    installed = Version(sys.argv[1]).release
except InvalidVersion:
    raise SystemExit(1)

supported = Version(sys.argv[2]).release
raise SystemExit(0 if installed[: len(supported)] == supported else 1)
PY
  then
    {
      echo "ERROR: SGLang version check failed."
      echo "  supported: ${supported_line}"
      echo "  installed: ${installed_version}"
      echo "Install a supported Ascend NPU release:"
      echo "  https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu"
    } >&2
    return 1
  fi

  echo "  sglang:      ${installed_version}"
}

precheck() {
  check_sglang_version

  # These packages must be installed before running this script. Keep version
  # selection in the official compatibility guides rather than hard-coding an
  # example stack here.
  SGLANG_OMNI_SKIP_NPU_DEVICE_CHECK="${SKIP_DEVICE_CHECK}" "${PYBIN}" - <<'PY'
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from re import match

ASCEND_PYTORCH_URL = (
    "https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download"
)
TRITON_ASCEND_URL = (
    "https://gitcode.com/Ascend/triton-ascend/blob/main/docs/en/quick_start.md"
)
SGL_KERNEL_NPU_URL = "https://github.com/sgl-project/sgl-kernel-npu/releases"

errors = []
torch = None
torch_npu = None

try:
    import torch

    print(f"  torch:       {torch.__version__}")
except ImportError:
    errors.append(
        "torch is not installed. Select a compatible release from:\n"
        f"    {ASCEND_PYTORCH_URL}"
    )

try:
    import torch_npu

    print(f"  torch_npu:   {torch_npu.__version__}")
except ImportError:
    errors.append(
        "torch_npu is not installed. Select the matching release from:\n"
        f"    {ASCEND_PYTORCH_URL}"
    )

try:
    import triton

    triton_ascend_version = version("triton-ascend")
    print(f"  triton:      {triton.__version__} (triton-ascend {triton_ascend_version})")
except (ImportError, PackageNotFoundError):
    errors.append(
        "triton-ascend is not installed. Follow the official documentation:\n"
        f"    {TRITON_ASCEND_URL}"
    )

try:
    import memfabric

    print(f"  memfabric:   {memfabric.__version__}")
except ImportError:
    print("  memfabric:   not installed (only needed for PD disaggregation)")

try:
    import sgl_kernel_npu

    print(f"  sgl-kernel:  {getattr(sgl_kernel_npu, '__version__', 'installed')}")
except ImportError:
    errors.append(
        "sgl-kernel-npu is not installed. Select a compatible release from:\n"
        f"    {SGL_KERNEL_NPU_URL}"
    )

if torch is not None and torch_npu is not None:
    def major_minor(value):
        parsed = match(r"^(\d+)\.(\d+)", value)
        return parsed.groups() if parsed else None

    if major_minor(torch.__version__) != major_minor(torch_npu.__version__):
        errors.append(
            f"torch {torch.__version__} and torch_npu {torch_npu.__version__} "
            "must have matching major.minor versions"
        )

    if os.environ["SGLANG_OMNI_SKIP_NPU_DEVICE_CHECK"] == "1":
        print("  NPU health:  skipped (--skip-device-check)")
    else:
        try:
            if not torch.npu.is_available():
                raise RuntimeError("torch.npu.is_available() returned False")
            count = torch.npu.device_count()
            if count < 1:
                raise RuntimeError(f"torch.npu.device_count() returned {count}")
            lhs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
            actual = (lhs @ lhs).cpu()
            expected = torch.tensor([[7.0, 10.0], [15.0, 22.0]])
            if not torch.equal(actual, expected):
                raise RuntimeError(f"MatMul result mismatch: {actual}")
            print(f"  NPU devices: {count}; MatMul: ok")
        except Exception as exc:
            errors.append(f"NPU health check failed: {exc}")

if errors:
    print("\nERROR: NPU prerequisites are missing or incompatible.", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    print(
        "\nSee docs/get_started/installation_npu.md for prerequisite links.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
}

print_install_command() {
  printf '%q ' "${INSTALL_CMD[@]}"
  printf '\n'
}

acquire_lock() {
  if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required to serialize the pyproject swap" >&2
    exit 1
  fi
  exec 9>"${LOCK}" || {
    echo "ERROR: cannot open lock ${LOCK}" >&2
    exit 1
  }
  if ! flock -n 9; then
    echo "ERROR: another ${0##*/} holds ${LOCK}; wait for it to finish" >&2
    exit 1
  fi
}

check_stale_backup() {
  [[ ! -e "${BACKUP}" ]] && return
  {
    echo "ERROR: leftover backup found: ${BACKUP}"
    echo
    echo "A previous run was interrupted after swapping pyproject.toml."
    echo "Restore the original manifest before re-running this script:"
    echo
    echo "  cp ${BACKUP} ${PYPROJECT} && rm ${BACKUP}"
  } >&2
  exit 1
}

show_dry_run() {
  echo
  echo "[--check] would run:"
  echo "  cp pyproject.toml .pyproject.cuda.bak"
  echo "  cp pyproject_npu.toml pyproject.toml"
  printf '  '
  print_install_command
  echo "  # then restore pyproject.toml from backup"
}

restore() {
  if [[ -f "${BACKUP}" ]]; then
    cp -f "${BACKUP}" "${PYPROJECT}"
    rm -f "${BACKUP}"
    echo "restored original pyproject.toml"
  fi
  rm -rf "${REPO_ROOT}/sglang_omni.egg-info" "${REPO_ROOT}/build" 2>/dev/null || true
}

install_project() {
  trap restore EXIT INT TERM

  cp -f "${PYPROJECT}" "${BACKUP}"
  cp -f "${PYPROJECT_NPU}" "${PYPROJECT}"
  echo "swapped in pyproject_npu.toml"

  printf '>>> '
  print_install_command
  "${INSTALL_CMD[@]}"

  restore
  trap - EXIT INT TERM
}

verify_install() {
  local verify_rc=0

  echo
  echo "=== verifying install ==="
  if (cd / && "${PYBIN}" -c "import sglang_omni" 2>/dev/null); then
    echo "  [ok] import sglang_omni works from outside the repo"
  else
    echo "  [FAIL] import sglang_omni does NOT work outside the repo"
    verify_rc=1
  fi

  if "${PYBIN}" -m pip show sglang-omni >/dev/null 2>&1; then
    echo "  [ok] pip shows sglang-omni: $("${PYBIN}" -m pip show sglang-omni 2>/dev/null | awk '/^Version:/{print $2}')"
  else
    echo "  [FAIL] pip does not show sglang-omni"
    verify_rc=1
  fi

  if command -v sgl-omni >/dev/null 2>&1 || [[ -x "$(dirname "$("${PYBIN}" -c 'import sys;print(sys.executable)')")/sgl-omni" ]]; then
    echo "  [ok] sgl-omni console script present"
  else
    echo "  [warn] sgl-omni not on PATH (check the environment's bin directory)"
  fi

  if "${PYBIN}" -c "import sglang" >/dev/null 2>&1; then
    echo "  [ok] sglang is importable"
  else
    echo "  [warn] sglang is not installed. Follow the Ascend NPU guide for"
    echo "         a supported SGLang ${SGLANG_SUPPORTED_RELEASE} release-line installation:"
    echo "         https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu"
  fi

  if [[ "${verify_rc}" -ne 0 ]]; then
    echo
    echo "INSTALL VERIFICATION FAILED. Review the pip build output above."
    return 1
  fi
}

main() {
  parse_args "$@"
  configure_install
  print_summary
  precheck
  acquire_lock
  check_stale_backup

  if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    show_dry_run
    return
  fi

  install_project
  verify_install

  echo
  echo "=== done. Next: ==="
  echo "  sgl-omni serve --model-path <path-to-model> --port 8000"
}

main "$@"
