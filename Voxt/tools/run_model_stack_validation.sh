#!/usr/bin/env bash
# One local validation batch. No commits, pushes, Actions dispatches or tag creation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
UPDATE=()
if [[ "${1:-}" == --update-lock ]]; then
  UPDATE=(--update-lock)
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: bash tools/run_model_stack_validation.sh [--update-lock]" >&2
  exit 2
fi
# Never change a shared developer machine's system selection without consent.
source tools/configure_xcode.sh
mkdir -p build
REPORT="$(mktemp -d "$ROOT/build/model-stack-validation.XXXXXX")"
trap 'status=$?; printf "exit_status=%s\n" "$status" > "$REPORT/status.txt"; echo "Validation artifacts: $REPORT"' EXIT
xcodebuild -version > "$REPORT/toolchain.txt"
xcrun swift --version >> "$REPORT/toolchain.txt"
git rev-parse HEAD > "$REPORT/base-commit.txt"
git diff --stat > "$REPORT/working-tree.txt"
python3 -B -m unittest discover -s tools -p 'test_*.py' -v 2>&1 | tee "$REPORT/tool-tests.log"
python3 -B tools/audit_model_stack.py
git diff --check
PACKAGES=(-clonedSourcePackagesDirPath "$REPORT/packages" -packageCachePath "$REPORT/package-cache")
bash tools/resolve_dependencies.sh "${UPDATE[@]}" "${PACKAGES[@]}" 2>&1 | tee "$REPORT/resolve.log"
cp Voxt.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved "$REPORT/Package.resolved"
COMMON=(-project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS'
  -derivedDataPath "$REPORT/DerivedData" -onlyUsePackageVersionsFromResolvedFile
  CODE_SIGNING_ALLOWED=NO)
xcodebuild build "${COMMON[@]}" "${PACKAGES[@]}" -configuration Debug 2>&1 | tee "$REPORT/debug-build.log"
# Pass explicit opt-in through xcodebuild's test-runner environment convention.
export TEST_RUNNER_VOXT_RUN_MODEL_TESTS="${VOXT_RUN_MODEL_TESTS:-0}"
if [[ -n "${VOXT_MODEL_STORAGE_ROOT:-}" ]]; then
  export TEST_RUNNER_VOXT_MODEL_STORAGE_ROOT="$VOXT_MODEL_STORAGE_ROOT"
fi
printf 'VOXT_RUN_MODEL_TESTS=%s\n' "${VOXT_RUN_MODEL_TESTS:-0}" > "$REPORT/model-test-mode.txt"
xcodebuild test "${COMMON[@]}" "${PACKAGES[@]}" -configuration Debug \
  -resultBundlePath "$REPORT/VoxtTests.xcresult" 2>&1 | tee "$REPORT/tests.log"
xcodebuild build "${COMMON[@]}" "${PACKAGES[@]}" -configuration Release \
  LD_GENERATE_MAP_FILE=YES 2>&1 | tee "$REPORT/release-build.log"
MAP_ARGS=()
while IFS= read -r -d '' map; do
  MAP_ARGS+=(--link-map "$map")
done < <(find "$REPORT/DerivedData/Build" -name '*LinkMap*.txt' -print0)
if [[ ${#MAP_ARGS[@]} -eq 0 ]]; then
  echo "Missing link maps for static runtime audit" >&2
  exit 1
fi
APP="$REPORT/DerivedData/Build/Products/Release/Voxt.app"
python3 -B tools/audit_model_stack.py --resolved "$REPORT/Package.resolved" --app "$APP" "${MAP_ARGS[@]}"
du -sk "$APP" | tee "$REPORT/app-size-kib.txt"
printf '%s\n' 'Build/test/artifact audit completed. Review skipped model tests and collect quality/latency/memory baselines before release.'
