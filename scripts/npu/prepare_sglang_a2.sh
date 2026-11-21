#!/usr/bin/env bash
set -euo pipefail

output_directory="${1:?Usage: prepare_sglang_a2.sh OUTPUT_DIRECTORY}"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sglang_version="$(python3 "$script_directory/config.py" --get sglang-version)"
dockerfile_sha256="$(python3 "$script_directory/config.py" --get sglang-dockerfile-sha256)"

mkdir -p -- "$output_directory"
cd -- "$output_directory"
dockerfile="$(mktemp npu.Dockerfile.XXXXXX)"
trap 'rm -f -- "$dockerfile"' EXIT
curl --fail --silent --show-error --location --retry 3 \
    "https://raw.githubusercontent.com/sgl-project/sglang/v${sglang_version}/docker/npu.Dockerfile" \
    --output "$dockerfile"
printf '%s  %s\n' "$dockerfile_sha256" "$dockerfile" | sha256sum --check --strict

patch --batch --forward --fuzz=0 --reject-file=- "$dockerfile" <<'PATCH'
--- npu.Dockerfile
+++ npu.Dockerfile
@@ -79,6 +79,7 @@
     case "$DEVICE_TYPE" in \
       950) MF_SOC_VERSION="A5" ;; \
       a3)  MF_SOC_VERSION="A3" ;; \
+      910b) MF_SOC_VERSION="A2" ;; \
       *)   echo "Unsupported DEVICE_TYPE for mfcli kernel install: $DEVICE_TYPE" >&2; \
            exit 1 ;; \
     esac; \
PATCH
mv -- "$dockerfile" npu.Dockerfile
