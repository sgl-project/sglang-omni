#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 1
}

[[ "$#" -eq 1 && -n "${1:-}" ]] || {
  echo "Usage: $0 RUNTIME_DIRECTORY" >&2
  exit 2
}
runtime_directory="$1"
[[ -d "$runtime_directory/bin" ]] || die "native runtime bin directory not found: $runtime_directory/bin"

for server in \
  qwen3_asr_server \
  whisper_server \
  cohere_transcribe_server \
  moss_transcribe_diarize_server \
  nemotron_asr_server \
  parakeet_server \
  sensevoice_server; do
  [[ -f "$runtime_directory/bin/$server" && -x "$runtime_directory/bin/$server" ]] || die "native runtime server must be an executable regular file: $runtime_directory/bin/$server"
done

for resource in libmlx.dylib libjaccl.dylib mlx.metallib; do
  [[ -f "$runtime_directory/bin/$resource" && -r "$runtime_directory/bin/$resource" ]] || die "native runtime resource must be a readable regular file: $runtime_directory/bin/$resource"
done
