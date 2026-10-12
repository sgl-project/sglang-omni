#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/tmp/OmniVAD-Kit"
BUILD_DIR="${SRC_DIR}/build-voxt"
REPO_URL="${OMNIVAD_REPO_URL:-https://github.com/lifeiteng/OmniVAD-Kit.git}"
REPO_REF="${OMNIVAD_REF:-085c1344981dc33250b87e98ce60bd302810bd6d}"

if [ ! -d "${SRC_DIR}/.git" ]; then
  git init "${SRC_DIR}"
  git -C "${SRC_DIR}" remote add origin "${REPO_URL}"
fi

git -C "${SRC_DIR}" remote set-url origin "${REPO_URL}"
git -C "${SRC_DIR}" fetch --depth 1 origin "${REPO_REF}"
git -C "${SRC_DIR}" checkout --detach FETCH_HEAD
RESOLVED_REF="$(git -C "${SRC_DIR}" rev-parse HEAD)"

cmake -S "${SRC_DIR}" \
  -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"

cmake --build "${BUILD_DIR}" --config Release -j"$(sysctl -n hw.ncpu)"

mkdir -p "${ROOT_DIR}/Voxt/Frameworks" "${ROOT_DIR}/Voxt/Resources/OmniVAD"
cp -f "${BUILD_DIR}/libomnivad.dylib" "${ROOT_DIR}/Voxt/Frameworks/libomnivad.dylib"
cp -f "${SRC_DIR}/models/vad.omnivad" "${ROOT_DIR}/Voxt/Resources/OmniVAD/vad.omnivad"
cp -f "${SRC_DIR}/models/stream-vad.omnivad" "${ROOT_DIR}/Voxt/Resources/OmniVAD/stream-vad.omnivad"
cp -f "${SRC_DIR}/models/aed.omnivad" "${ROOT_DIR}/Voxt/Resources/OmniVAD/aed.omnivad"
cp -f "${SRC_DIR}/LICENSE" "${ROOT_DIR}/Voxt/Resources/OmniVAD/LICENSE-OmniVAD-Kit.txt"

printf '%s\n' \
  "repository=${REPO_URL}" \
  "requested_ref=${REPO_REF}" \
  "resolved_commit=${RESOLVED_REF}" \
  > "${ROOT_DIR}/Voxt/Resources/OmniVAD/SOURCE-OmniVAD-Kit.txt"

otool -L "${ROOT_DIR}/Voxt/Frameworks/libomnivad.dylib"
