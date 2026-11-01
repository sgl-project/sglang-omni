#!/usr/bin/env bash
# System go that already matches go.mod is reused, even when its GOROOT is
# outside the project cache. A downloaded toolchain must still land in the cache.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
module="$root/tailcat/host"
source "$root/tools/tailcat-go-env.sh"

work="$(mktemp -d "${TMPDIR:-/tmp}/tailcat-go-env.XXXXXX")"
trap 'rm -rf "$work"' EXIT

install_fake_go() {
  local dir="$1"
  mkdir -p "$dir/bin"
  cat > "$dir/bin/go" <<'EOF'
#!/bin/bash
set -euo pipefail
toolchain="${GOTOOLCHAIN:-local}"
if [[ "$toolchain" == "local" || "$toolchain" == "auto" || "$toolchain" == "$FAKE_LOCAL_TOOLCHAIN" ]]; then
  use_version="$FAKE_LOCAL_VERSION"
  use_root="$FAKE_LOCAL_GOROOT"
else
  if [[ "${FAKE_DOWNLOAD_FAILS:-}" == 1 ]]; then
    echo "go: download ${toolchain} for darwin/arm64: toolchain not available" >&2
    exit 1
  fi
  use_version="$FAKE_SWITCHED_VERSION"
  use_root="$FAKE_SWITCHED_GOROOT"
fi
case "${1:-}" in
  version) echo "go version ${use_version} darwin/arm64" ;;
  env)
    [[ "${2:-}" == "GOROOT" ]] || { echo "unsupported env ${2:-}" >&2; exit 1; }
    echo "$use_root"
    ;;
  *) echo "unsupported go invocation: $*" >&2; exit 1 ;;
esac
EOF
  chmod +x "$dir/bin/go"
}

cache="$work/project-cache"
system_root="$work/system-go"
install_fake_go "$system_root"
export PATH="$system_root/bin:$PATH"
export VOXT_TAILCAT_GO_CACHE="$cache"
unset GOPROXY GOSUMDB GOTOOLCHAIN

export FAKE_LOCAL_VERSION="go1.27.1"
export FAKE_LOCAL_TOOLCHAIN="go1.27.1"
export FAKE_LOCAL_GOROOT="$system_root"
export FAKE_SWITCHED_VERSION="go1.27.1"
export FAKE_SWITCHED_GOROOT="$system_root"
tailcat_prepare_go "$module"
if [[ "$GO_BIN" != "$system_root/bin/go" ]]; then
  echo "matching system toolchain was not reused: $GO_BIN" >&2
  exit 1
fi

stray="$work/stray-go"
install_fake_go "$stray"
export FAKE_LOCAL_VERSION="go1.26.3"
export FAKE_LOCAL_TOOLCHAIN="go1.26.3"
export FAKE_SWITCHED_VERSION="go1.27.1"
export FAKE_SWITCHED_GOROOT="$stray"
if tailcat_prepare_go "$module"; then
  echo "toolchain downloaded outside the project cache was accepted" >&2
  exit 1
fi

downloaded="$cache/modcache/golang.org/toolchain@v0.0.1-go1.27.1.darwin-arm64"
install_fake_go "$downloaded"
export FAKE_SWITCHED_GOROOT="$downloaded"
tailcat_prepare_go "$module"
if [[ "$GO_BIN" != "$downloaded/bin/go" ]]; then
  echo "downloaded toolchain was not used: $GO_BIN" >&2
  exit 1
fi

# The build scripts source this under set -e; without go on PATH they must still say why.
missing="$(PATH=/usr/bin:/bin /bin/bash -c 'set -euo pipefail; source "$1"; tailcat_prepare_go "$2"' _ \
  "$root/tools/tailcat-go-env.sh" "$module" 2>&1 || true)"
if [[ "$missing" != *"system go was not found on PATH"* ]]; then
  echo "no message when go is missing under set -e: '$missing'" >&2
  exit 1
fi

# Without network the go.mod toolchain cannot be downloaded: say what to install.
export FAKE_LOCAL_VERSION="go1.26.3" FAKE_LOCAL_TOOLCHAIN="go1.26.3" FAKE_DOWNLOAD_FAILS=1
offline="$(tailcat_prepare_go "$module" 2>&1 || true)"
if [[ "$offline" != *"install Go 1.27.1 or later"* ]]; then
  echo "no install hint when the toolchain download fails: '$offline'" >&2
  exit 1
fi
unset FAKE_DOWNLOAD_FAILS

echo "system toolchain reuse ready"
