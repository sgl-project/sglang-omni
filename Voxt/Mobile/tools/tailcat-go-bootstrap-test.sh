#!/usr/bin/env bash
# Clean-checkout bootstrap for the Tailcat host helper.
# An empty project cache must receive the go.mod toolchain via system go,
# then the helper must be compiled there. The helper is not executed.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cache="$root/build/voxt-tailcat-go-clean"
# Go writes its module cache read-only; make it writable or rm fails on the previous run's cache.
[[ ! -d "$cache" ]] || chmod -R u+w "$cache"
rm -rf "$cache"
mkdir -p "$cache"

proxy_before="$(go env GOPROXY)"
sumdb_before="$(go env GOSUMDB)"

export VOXT_TAILCAT_GO_CACHE="$cache"
export VOXT_TAILCAT_HOST_OUT="$cache/host"
unset GOTOOLCHAIN

for script in build-tailcat-host.sh build-tailcat-native.sh; do
  # tailcat-go-env.sh owns the proxy and the Go caches; a build script that sets its own would skip the clean cache.
  if grep -qE '(GOPROXY|GOMODCACHE|GOPATH|GOROOT)=' "$root/tools/$script"; then
    echo "$script sets the Go proxy or a Go cache of its own" >&2
    exit 1
  fi
done

archive="$root/build/voxt-tailcat-native/carchive/darwin/libCVoxtTailcat.a"
archive_before=""
if [[ -f "$archive" ]]; then
  archive_before="$(stat -f %m "$archive")"
fi
VOXT_TAILCAT_PREPARE_ONLY=1 "$root/tools/build-tailcat-native.sh"
archive_after=""
if [[ -f "$archive" ]]; then
  archive_after="$(stat -f %m "$archive")"
fi
if [[ "$archive_before" != "$archive_after" ]]; then
  echo "native prepare built an archive; full native build belongs to root" >&2
  exit 1
fi

"$root/tools/build-tailcat-host.sh"

helper="$cache/host/voxt-tailcat-host"
if [[ ! -f "$helper" ]]; then
  echo "clean host build did not produce a helper binary" >&2
  exit 1
fi
if ! file "$helper" | grep -q "Mach-O"; then
  echo "helper is not a Mach-O executable" >&2
  exit 1
fi

toolchain=""
while IFS= read -r candidate; do
  case "$candidate" in
    "$cache/host/"*) continue ;;
  esac
  toolchain="$candidate"
  break
done < <(find "$cache" -type f -path '*/bin/go')
if [[ -z "$toolchain" ]]; then
  echo "clean cache does not contain the downloaded go toolchain" >&2
  exit 1
fi
case "$toolchain" in
  "$cache/modcache/golang.org/toolchain@"*) ;;
  *)
    echo "toolchain at '$toolchain' was not downloaded into the clean module cache" >&2
    exit 1
    ;;
esac
version="$("$toolchain" version)"
case "$version" in
  *"go1.27.1 "*) ;;
  *)
    echo "toolchain is '$version', go.mod requires go1.27.1" >&2
    exit 1
    ;;
esac

proxy_after="$(go env GOPROXY)"
sumdb_after="$(go env GOSUMDB)"
if [[ "$proxy_before" != "$proxy_after" || "$sumdb_before" != "$sumdb_after" ]]; then
  echo "go proxy settings changed: proxy '$proxy_before' -> '$proxy_after', sumdb '$sumdb_before' -> '$sumdb_after'" >&2
  exit 1
fi

echo "host clean cache build ready ($version)"
