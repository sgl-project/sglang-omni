#!/usr/bin/env bash
# Install sglang-omni for CPU: swap in pyproject_cpu.toml and install against
# the PyTorch CPU wheel index. The default pyproject may pull accelerator-only
# wheels, so this script preserves a CPU torch stack. See
# docs/get_started/installation_cpu.md.
#
# Usage:
#   scripts/cpu/install_cpu.sh                    # lean core, editable
#   scripts/cpu/install_cpu.sh --extras eval      # core + eval/tests
#   scripts/cpu/install_cpu.sh --no-editable      # non-editable
#   scripts/cpu/install_cpu.sh --check            # dry-run: show what would run
#
# Never deletes your pyproject.toml: it is backed up and restored on exit.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
CPU_INDEX="https://download.pytorch.org/whl/cpu"
EDITABLE="-e"
CHECK_ONLY=0
EXTRAS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-editable) EDITABLE=""; shift ;;
    --check)       CHECK_ONLY=1; shift ;;
    --extras)      EXTRAS="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

PYPROJECT="${REPO_ROOT}/pyproject.toml"
PYPROJECT_CPU="${REPO_ROOT}/pyproject_cpu.toml"


SGLANG_VERIFIED_VERSION="v0.5.16"

[[ -f "${PYPROJECT_CPU}" ]] || { echo "ERROR: ${PYPROJECT_CPU} not found" >&2; exit 1; }

PYBIN="${PYTHON:-python}"

# Build the install target, e.g. ".[eval]" or "."
TARGET="."
if [[ -n "${EXTRAS}" ]]; then
  TARGET=".[${EXTRAS}]"
fi

echo "=== sglang-omni CPU install ==="
echo "  repo:        ${REPO_ROOT}"
echo "  python:      $(${PYBIN} -c 'import sys; print(sys.executable)')"
echo "  cpu index:   ${CPU_INDEX}"
echo "  target:      ${TARGET}"
echo "  editable:    $([[ -n "${EDITABLE}" ]] && echo yes || echo no)"

${PYBIN} - <<'PY' || true
try:
    import torch
    print(f"  torch:       {torch.__version__} (cuda build: {getattr(torch.version, 'cuda', None)})")
except Exception as e:
    print(f"  torch:       not importable ({e})")
PY

# --no-build-isolation is REQUIRED (see docs/get_started/installation_cpu.md):
# with build isolation, pip spins up an isolated build env whose fallback
# setuptools emits a legacy in-tree egg-info instead of a PEP 660 editable .pth
# — the package then isn't importable outside the repo and no sgl-omni console
# script is created. Without isolation it uses this env's setuptools, so an
# editable install needs PEP 660 support (setuptools>=64) here; check up front
# rather than silently producing the legacy layout.
if [[ -n "${EDITABLE}" ]]; then
    "${PYBIN}" -c "import setuptools.build_meta as m, setuptools, sys; sys.exit(None if hasattr(m, 'build_editable') else f'setuptools {setuptools.__version__} lacks PEP 660 support; an editable install would emit a legacy egg-info. Upgrade setuptools to 64 or newer, or pass --no-editable.')" || exit 1
fi
NOISO="--no-build-isolation"
INSTALL_CMD="${PYBIN} -m pip install ${EDITABLE} ${TARGET} ${NOISO} --index-url ${CPU_INDEX} --extra-index-url https://pypi.org/simple"

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    echo
    echo "[--check] would run:"
    echo "  cp pyproject_cpu.toml pyproject.toml"
    echo "  ${INSTALL_CMD}"
    exit 0
fi

BACKUP="$(mktemp "${REPO_ROOT}/.pyproject.cuda.bak.XXXXXX")"

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
trap restore EXIT

cp -f "${PYPROJECT}" "${BACKUP}"
cp -f "${PYPROJECT_CPU}" "${PYPROJECT}"
echo "swapped in pyproject_cpu.toml"

echo ">>> ${INSTALL_CMD}"
${INSTALL_CMD}

restore
trap - EXIT

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

# SGLang is installed separately from source for CPU, so report whether it is
# importable at all.
if "${PYBIN}" -c "import sglang" >/dev/null 2>&1; then
  echo "  [ok] sglang is importable"
else
  echo "  [warn] sglang is NOT installed — it is intentionally not a dependency here."
  echo "         Build the CPU SGLang from source (${SGLANG_VERIFIED_VERSION}):"
  echo "           git clone https://github.com/sgl-project/sglang && cd sglang"
  echo "           git checkout ${SGLANG_VERIFIED_VERSION}"
  echo "           cd python && cp pyproject_cpu.toml pyproject.toml"
  echo "           pip install -e . --no-build-isolation --index-url ${CPU_INDEX} --extra-index-url https://pypi.org/simple"
fi

if [[ "${VERIFY_RC}" -ne 0 ]]; then
  echo
  echo "INSTALL VERIFICATION FAILED. If setuptools fell back to legacy egg-info,"
  echo "retry with a PEP 660 editable build, e.g.:"
  echo "  ${PYBIN} -m pip install -e . --index-url ${CPU_INDEX} --extra-index-url https://pypi.org/simple --config-settings editable_mode=strict"
  exit 1
fi

echo
echo "=== done. Next: ==="
echo "  SGLANG_USE_CPU_ENGINE=1 sgl-omni serve --device cpu --model-path <path-to-qwen3-tts> --port 8000"
