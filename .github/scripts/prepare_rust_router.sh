#!/usr/bin/env bash
set -euo pipefail

mode="${1:-path}"
if [[ "${mode}" != "path" && "${mode}" != "build" ]]; then
  echo "usage: $0 [path|build]" >&2
  exit 2
fi

crate_dir="sglang_omni_router/rust"
tree="$(git -c safe.directory="${PWD}" rev-parse "HEAD:${crate_dir}")"
router_dir="$(dirname "${OMNI_CI_HOME}")/rust-router"
binary="${router_dir}/bin/${tree}/sgl-omni-router"

if [[ "${mode}" == "build" ]]; then
  mkdir -p "${router_dir}"
  exec 200>"${router_dir}/build.lock"
  flock 200
  if [[ ! -x "${binary}" ]]; then
    export CARGO_HOME="${router_dir}/cargo"
    export RUSTUP_HOME="${router_dir}/rustup"
    export CARGO_TARGET_DIR="${router_dir}/target"
    if [[ -f "${CARGO_HOME}/env" ]]; then
      source "${CARGO_HOME}/env"
    fi
    if ! command -v cargo >/dev/null 2>&1; then
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --no-modify-path >&2
      source "${CARGO_HOME}/env"
    fi

    source .github/scripts/pin_to_ci_cpuset.sh
    (
      cd "${crate_dir}"
      cargo build --release --locked >&2
    )
    mkdir -p "$(dirname "${binary}")"
    install -m 755 "${CARGO_TARGET_DIR}/release/sgl-omni-router" "${binary}"
  fi
fi

printf '%s\n' "${binary}"
