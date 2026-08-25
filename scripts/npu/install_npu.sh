#!/usr/bin/env bash
# Install sglang-omni for Huawei Ascend NPU: swap in pyproject_npu.toml and install
# against the pre-installed torch_npu / triton-ascend stack (the CUDA pyproject would
# clobber a working torch+npu stack). See docs/get_started/installation_npu.md.
#
# Prerequisites (NOT installed by this script — see installation_npu.md):
#   - CANN toolkit 9.0.0
#   - torch + torch_npu (matching versions)
#   - triton-ascend
#   - memfabric-hybrid (for PD disaggregation)
#   - sgl-kernel-npu (custom operators)
#
# Usage:
#   scripts/npu/install_npu.sh                    # lean core, editable
#   scripts/npu/install_npu.sh --extras eval      # core + eval/tests
#   scripts/npu/install_npu.sh --no-editable      # non-editable
#   scripts/npu/install_npu.sh --skip-device-check # build host has no visible NPU
#   scripts/npu/install_npu.sh --check            # dry-run: show what would run
#
# Never deletes your pyproject.toml: it is backed up and restored on exit.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EDITABLE="-e"
CHECK_ONLY=0
SKIP_DEVICE_CHECK=0
EXTRAS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-editable) EDITABLE=""; shift ;;
    --check)       CHECK_ONLY=1; shift ;;
    --skip-device-check) SKIP_DEVICE_CHECK=1; shift ;;
    --extras)
      [[ $# -ge 2 && -n "${2:-}" ]] || { echo "ERROR: --extras requires a value" >&2; exit 2; }
      EXTRAS="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

PYPROJECT="${REPO_ROOT}/pyproject.toml"
PYPROJECT_NPU="${REPO_ROOT}/pyproject_npu.toml"
BACKUP="${REPO_ROOT}/.pyproject.cuda.bak"

SGLANG_VERIFIED_VERSION="v0.5.16"

[[ -f "${PYPROJECT_NPU}" ]] || { echo "ERROR: ${PYPROJECT_NPU} not found" >&2; exit 1; }

PYBIN="${PYTHON:-python}"

# Build the install target, e.g. ".[eval]" or "."
TARGET="."
if [[ -n "${EXTRAS}" ]]; then
  IFS=',' read -r -a EXTRA_NAMES <<< "${EXTRAS}"
  for extra in "${EXTRA_NAMES[@]}"; do
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

echo "=== sglang-omni Huawei Ascend NPU install ==="
echo "  repo:        ${REPO_ROOT}"
echo "  python:      $("${PYBIN}" -c 'import sys; print(sys.executable)')"
echo "  target:      ${TARGET}"
echo "  editable:    $([[ -n "${EDITABLE}" ]] && echo yes || echo no)"

# --- Pre-flight: verify NPU stack is installed ---------------------------
# torch, torch_npu, triton-ascend, and memfabric are intentionally NOT pinned
# in pyproject_npu.toml (matching upstream SGLang's pyproject_npu.toml where
# srt_npu = []).  They must be installed BEFORE running this script.
SGLANG_OMNI_SKIP_NPU_DEVICE_CHECK="${SKIP_DEVICE_CHECK}" "${PYBIN}" - <<'PY' || exit 1
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from re import match

errors = []
torch = None
torch_npu = None

try:
    import torch
    print(f"  torch:       {torch.__version__}")
except ImportError:
    errors.append("torch is not installed. Install CPU torch first:\n"
                  "  pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu")

try:
    import torch_npu
    print(f"  torch_npu:   {torch_npu.__version__}")
except ImportError:
    errors.append("torch_npu is not installed. Install from Huawei OBS:\n"
                  "  pip install torch_npu==2.10.0  # or from https://gitcode.com/Ascend/pytorch/releases")

try:
    import triton
    triton_ascend_version = version("triton-ascend")
    print(f"  triton:      {triton.__version__} (triton-ascend {triton_ascend_version})")
except (ImportError, PackageNotFoundError):
    errors.append("triton-ascend is not installed. Install from Huawei cloud mirror:\n"
                  "  pip install triton-ascend==3.2.1.dev20260530 \\\n"
                  "    --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly \\\n"
                  "    --trusted-host mirrors.huaweicloud.com")

try:
    import memfabric
    print(f"  memfabric:   {memfabric.__version__}")
except ImportError:
    # memfabric is only needed for PD disaggregation; warn, don't error
    print("  memfabric:   not installed (only needed for PD disaggregation)")

try:
    import sgl_kernel_npu
    print(f"  sgl-kernel:  {getattr(sgl_kernel_npu, '__version__', 'installed')}")
except ImportError:
    errors.append("sgl-kernel-npu is not installed. Install a release from:\n"
                  "  https://github.com/sgl-project/sgl-kernel-npu/releases")

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
    print()
    print("ERROR: NPU prerequisites are missing.", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    print(file=sys.stderr)
    print("See docs/get_started/installation_npu.md for the full install sequence.", file=sys.stderr)
    sys.exit(1)
PY

# --- Verify setuptools >= 77 (same rationale as XPU) ---------------------
# --no-build-isolation is REQUIRED (see docs/get_started/installation_npu.md):
# with build isolation, pip spins up an isolated build env whose fallback
# setuptools emits a legacy in-tree egg-info instead of a PEP 660 editable .pth
# — the package then isn't importable outside the repo and no sgl-omni console
# script is created. Without isolation the build uses this env's setuptools, so the
# build requirement in pyproject_npu.toml is never enforced by pip -- check it here
# instead of failing later in metadata generation.
"${PYBIN}" - <<'PY' || exit 1
import sys

import setuptools
import setuptools.build_meta

version = tuple(
    int("".join(char for char in part if char.isdigit()) or 0)
    for part in setuptools.__version__.split(".")[:2]
)
if version < (77, 0):
    sys.exit(
        f"setuptools {setuptools.__version__} is too old: pyproject_npu.toml uses "
        "the PEP 639 license fields, which 76.1 and older reject with 'invalid "
        "pyproject.toml config: `project.license`'. --no-build-isolation means pip "
        "will not upgrade it for you, so run: pip install -U 'setuptools>=77.0.0'"
    )
if not hasattr(setuptools.build_meta, "build_editable"):
    sys.exit(
        f"setuptools {setuptools.__version__} lacks PEP 660 support; an editable "
        "install would emit a legacy egg-info."
    )
PY
NOISO="--no-build-isolation"
INSTALL_CMD=("${PYBIN}" -m pip install)
[[ -n "${EDITABLE}" ]] && INSTALL_CMD+=("${EDITABLE}")
INSTALL_CMD+=("${TARGET}" "${NOISO}")

print_install_command() {
  printf '%q ' "${INSTALL_CMD[@]}"
  printf '\n'
}

# --- Serialize the swap (same flock pattern as XPU) ----------------------
LOCK="${REPO_ROOT}/.pyproject.npu.lock"
if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required to serialize the pyproject swap" >&2
  exit 1
fi
exec 9>"${LOCK}" || { echo "ERROR: cannot open lock ${LOCK}" >&2; exit 1; }
if ! flock -n 9; then
  echo "ERROR: another ${0##*/} holds ${LOCK}; wait for it to finish" >&2
  exit 1
fi

# A leftover backup means a previous run died after the swap (SIGKILL, OOM, power
# loss -- INT/TERM are trapped below). pyproject.toml is then the NPU manifest and
# this backup holds the only copy of the original, so the cp below would destroy it.
# Refuse instead, and hand back the recovery step. Checked before the trap is armed
# so exiting here cannot itself touch either file.
if [[ -e "${BACKUP}" ]]; then
  {
    echo "ERROR: leftover backup found: ${BACKUP}"
    echo
    echo "A previous run was interrupted after swapping pyproject.toml, so that"
    echo "backup is the only copy of your original manifest. Restore it first:"
    echo
    echo "  cp ${BACKUP} ${PYPROJECT} && rm ${BACKUP}"
    echo
    echo "Then re-run this script."
  } >&2
  exit 1
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo
  echo "[--check] would run:"
  echo "  cp pyproject.toml .pyproject.cuda.bak"
  echo "  cp pyproject_npu.toml pyproject.toml"
  printf '  '
  print_install_command
  echo "  # then restore pyproject.toml from backup"
  exit 0
fi

# Restore the CUDA pyproject.toml no matter how we exit. Use cp (not mv) so a
# partial/interrupted restore still leaves the backup in place, and also sweep
# the in-tree build artifacts an editable build may drop.
restore() {
  if [[ -f "${BACKUP}" ]]; then
    cp -f "${BACKUP}" "${PYPROJECT}"
    rm -f "${BACKUP}"
    echo "restored original pyproject.toml"
  fi
  rm -rf "${REPO_ROOT}/sglang_omni.egg-info" "${REPO_ROOT}/build" 2>/dev/null || true
}
trap restore EXIT INT TERM

cp -f "${PYPROJECT}" "${BACKUP}"
cp -f "${PYPROJECT_NPU}" "${PYPROJECT}"
echo "swapped in pyproject_npu.toml"

printf '>>> '
print_install_command
"${INSTALL_CMD[@]}"

# Restore immediately (don't wait for EXIT) so verification below runs against a
# clean tree and a setuptools editable .pth (not the swapped file).
restore
trap - EXIT INT TERM

echo
echo "=== verifying install ==="
VERIFY_RC=0
# import must work from OUTSIDE the repo (proves a real install, not cwd-on-path)
if (cd / && "${PYBIN}" -c "import sglang_omni" 2>/dev/null); then
  echo "  [ok] import sglang_omni works from outside the repo"
else
  echo "  [FAIL] import sglang_omni does NOT work outside the repo (editable .pth missing)"
  VERIFY_RC=1
fi
if "${PYBIN}" -m pip show sglang-omni >/dev/null 2>&1; then
  echo "  [ok] pip shows sglang-omni: $(${PYBIN} -m pip show sglang-omni 2>/dev/null | awk '/^Version:/{print $2}')"
else
  echo "  [FAIL] pip does not show sglang-omni"
  VERIFY_RC=1
fi
if command -v sgl-omni >/dev/null 2>&1 || [[ -x "$(dirname "$(${PYBIN} -c 'import sys;print(sys.executable)')")/sgl-omni" ]]; then
  echo "  [ok] sgl-omni console script present"
else
  echo "  [warn] sgl-omni not on PATH (check the env's bin dir)"
fi

# SGLang is never installed by this script (it cannot be pinned for NPU), so
# report whether it is importable at all.
if "${PYBIN}" -c "import sglang" >/dev/null 2>&1; then
  echo "  [ok] sglang is importable"
else
  echo "  [warn] sglang is NOT installed — it is intentionally not a dependency here."
  echo "         Build the NPU SGLang from source (${SGLANG_VERIFIED_VERSION}):"
  echo "           git clone https://github.com/sgl-project/sglang && cd sglang"
  echo "           git checkout ${SGLANG_VERIFIED_VERSION}"
  echo "           cd python && cp pyproject_npu.toml pyproject.toml"
  echo "           pip install -e . --no-build-isolation"
fi

if [[ "${VERIFY_RC}" -ne 0 ]]; then
  echo
  echo "INSTALL VERIFICATION FAILED. If setuptools fell back to legacy egg-info,"
  echo "retry with a PEP 660 editable build, e.g.:"
  echo "  ${PYBIN} -m pip install -e . --config-settings editable_mode=strict"
  exit 1
fi

echo
echo "=== done. Next: ==="
echo "  sgl-omni serve --model-path <path-to-model> --port 8000"
