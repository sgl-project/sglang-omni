#!/usr/bin/env bash
# The chat service for Inkling-2-Mini, from a venv of its own.
#
# SGLang serves Inkling-2-Mini with DSpark only from an unreleased commit, whose
# torch kernels and FlashInfer differ from the SGLang release that sglang-omni
# pins. That commit lives in a venv apart from the one sglang-omni runs in, so
# no other model sees its dependencies.
#
# Usage:
#   examples/streaming_cascade/inkling_chat_service.sh install
#   examples/streaming_cascade/inkling_chat_service.sh serve <sglang.launch_server arguments>
#
# The venv is INKLING_VENV, by default .venv-inkling in the repository.
# install builds it once per pinned commit and checks it; serve runs
# sglang_omni.models.streaming_cascade.chat_service from it with the arguments.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INKLING_VENV="${INKLING_VENV:-${REPO_ROOT}/.venv-inkling}"
# The head of https://github.com/sgl-project/sglang/pull/51926, which adds
# Inkling-2-Mini and DSpark. Move it to the merge commit once that lands.
SGLANG_COMMIT="9c64454bba553e475ce7b7b790015d2f941e2807"
SGLANG_REQUIREMENT="sglang @ git+https://github.com/sgl-project/sglang.git@${SGLANG_COMMIT}#subdirectory=python"
PYTHON_VERSION="3.12"
MARKER="${INKLING_VENV}/.sglang-commit"

install() {
  if [[ -f "${MARKER}" && "$(cat "${MARKER}")" == "${SGLANG_COMMIT}" ]]; then
    echo "Inkling venv at ${INKLING_VENV} already has SGLang ${SGLANG_COMMIT}"
    return 0
  else
    rm -rf "${INKLING_VENV}"
  fi
  echo "Building the Inkling venv at ${INKLING_VENV} with SGLang ${SGLANG_COMMIT}"
  uv venv --python "${PYTHON_VERSION}" "${INKLING_VENV}"
  # SGLang builds its Rust extensions only with a Rust toolchain, which the CI image lacks; Inkling's
  # chat service needs none of them (the radix tree falls back to Python, the others are opt-in).
  SGLANG_BUILD_RUST_EXTS=none uv pip install --python "${INKLING_VENV}/bin/python" "${SGLANG_REQUIREMENT}"
  PYTHONPATH="${REPO_ROOT}" "${INKLING_VENV}/bin/python" - <<'PY'
from importlib.metadata import version

from sglang.srt.models.registry import ModelRegistry
from sglang.srt.parser.inkling_output import InklingOutputParser

import sglang_omni.models.streaming_cascade.chat_service  # noqa: F401

expected = {"torch": "2.14.1", "flashinfer-python": "0.7.4", "sglang-kernel": "0.4.11"}
found = {name: version(name).split("+")[0] for name in expected}
assert found == expected, f"Inkling venv has {found}, expected {expected}"
assert "Inkling2LoopForConditionalGeneration" in ModelRegistry.get_supported_archs()
missing = [name for name in ("_parse", "_tool_call", "_recover_message", "_payload_ids", "_content_kind_id") if not hasattr(InklingOutputParser, name)]
assert not missing, f"InklingOutputParser has no {missing}, which chat_service overrides or calls"
print(f"Inkling venv checked: {found}")

# DSpark's first draft verification runs FlashInfer's sampling kernels, which FlashInfer compiles at
# first use, about 60 s on one H100; compile them here so the first request does not wait.
from flashinfer.sampling import get_sampling_module

get_sampling_module()
PY
  echo "${SGLANG_COMMIT}" > "${MARKER}"
}

serve() {
  if [[ ! -f "${MARKER}" || "$(cat "${MARKER}")" != "${SGLANG_COMMIT}" ]]; then
    echo "ERROR: no Inkling venv with SGLang ${SGLANG_COMMIT} at ${INKLING_VENV}; run $0 install first" >&2
    exit 1
  fi
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  exec "${INKLING_VENV}/bin/python" -m sglang_omni.models.streaming_cascade.chat_service "$@"
}

case "${1:-}" in
  install) install ;;
  serve) shift; serve "$@" ;;
  *) sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
