#!/bin/bash
# Project-local Go toolchain for Tailcat builds. Source this file.
# System go downloads the toolchain named in go.mod. A system toolchain that
# already matches is reused; a download must land in the project cache.
# Proxy settings are not changed.

tailcat_prepare_go() {
  local module_dir="$1"
  local here root cache system_go system_goroot system_version required goroot version
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  root="$(cd "$here/.." && pwd)"
  cache="${VOXT_TAILCAT_GO_CACHE:-$root/build/voxt-tailcat-go}"
  mkdir -p "$cache/modcache" "$cache/gocache" "$cache/gopath"

  system_go="$(command -v go || true)" # under set -e a failed lookup would exit before the message
  if [[ -z "$system_go" || ! -x "$system_go" ]]; then
    echo "system go was not found on PATH" >&2
    return 1
  fi
  required="$(awk '/^go [0-9]/{ print $2; exit }' "$module_dir/go.mod")"
  if [[ -z "$required" ]]; then
    echo "go.mod has no toolchain version: $module_dir/go.mod" >&2
    return 1
  fi
  system_goroot="$(GOTOOLCHAIN=local "$system_go" env GOROOT)"
  system_version="$(GOTOOLCHAIN=local "$system_go" version)"

  export GOMODCACHE="$cache/modcache"
  export GOCACHE="$cache/gocache"
  export GOPATH="$cache/gopath"
  export GOTOOLCHAIN="go${required}"

  # Go downloads the go.mod toolchain through GOPROXY; offline it fails within about 30 s.
  goroot="$(cd "$module_dir" && "$system_go" env GOROOT)" || goroot=""
  if [[ ! -x "$goroot/bin/go" ]]; then
    echo "go${required} toolchain was not found: install Go ${required} or later, or let go download it (needs network)" >&2
    return 1
  fi
  version="$("$goroot/bin/go" version)"
  case "$version" in
    *"go${required} "*) ;;
    *)
      echo "toolchain is '$version', go.mod requires go${required}" >&2
      return 1
      ;;
  esac
  if [[ "$goroot" == "$cache"/* ]]; then
    :
  elif [[ "$goroot" == "$system_goroot" && "$system_version" == *"go${required} "* ]]; then
    :
  else
    echo "downloaded toolchain resolved outside the project cache: $goroot" >&2
    return 1
  fi
  GO_BIN="$goroot/bin/go"
  export GO_BIN
}
