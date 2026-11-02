#!/usr/bin/env bash
# Bootstrap a missing lockfile on macOS, then enforce the resolved graph.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOCK="Voxt.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved"
command -v xcodebuild >/dev/null || { echo "Requires Xcode on macOS" >&2; exit 1; }
xcodebuild -version
xcrun swift --version
UPDATE_LOCK=0
if [[ "${1:-}" == --update-lock ]]; then
  UPDATE_LOCK=1
  shift
fi
if [[ -f "$LOCK" && "$UPDATE_LOCK" == 0 ]]; then
  xcodebuild -resolvePackageDependencies \
    -project Voxt.xcodeproj -scheme Voxt \
    -onlyUsePackageVersionsFromResolvedFile "$@"
else
  echo "Resolving the current project requirements; review and commit Package.resolved before release." >&2
  xcodebuild -resolvePackageDependencies \
    -project Voxt.xcodeproj -scheme Voxt \
    "$@"
fi
python3 tools/audit_model_stack.py --resolved "$LOCK"
