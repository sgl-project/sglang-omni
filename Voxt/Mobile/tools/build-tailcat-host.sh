#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/tailcat/host"
out="${VOXT_TAILCAT_HOST_OUT:-$root/build/voxt-tailcat-host}"
source "$root/tools/tailcat-go-env.sh"
clang="$(xcrun --find clang)"
sdk_mac="$(xcrun --sdk macosx --show-sdk-path)"

unset CC CGO_CFLAGS CGO_LDFLAGS CGO_CXXFLAGS CGO_CPPFLAGS SDKROOT
tailcat_prepare_go "$src"
export CGO_ENABLED=1
export GOOS=darwin
export GOARCH=arm64
export CC="$clang"
export CGO_CFLAGS="-mmacosx-version-min=14.0 -isysroot $sdk_mac"
export CGO_LDFLAGS="-mmacosx-version-min=14.0 -isysroot $sdk_mac"

cd "$src"
module_line="$("$GO_BIN" list -mod=readonly -m github.com/tailscale/tailcat)"
if [[ "$module_line" != "github.com/tailscale/tailcat v0.6.0" ]]; then
  echo "tailcat module is not the pinned v0.6.0" >&2
  exit 1
fi
mkdir -p "$out"
"$GO_BIN" build -trimpath -mod=readonly -tags ts_omit_ssh -o "$out/voxt-tailcat-host" .
echo "helper ready"
