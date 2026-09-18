#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${OMNITYPER_PYTHON:-$APP_ROOT/.venv/bin/python}"
SWIFT_FLAGS=(--package-path "$APP_ROOT")
# Some standalone Command Line Tools releases omit automatic Testing plugin discovery.
SWIFT_BIN="$(xcrun --find swift)"
TESTING_PLUGIN="$(dirname "$SWIFT_BIN")/../lib/swift/host/plugins/testing/libTestingMacros.dylib"
if [[ -f "$TESTING_PLUGIN" ]]; then
  SWIFT_FLAGS+=(-Xswiftc -load-plugin-library -Xswiftc "$TESTING_PLUGIN")
fi
OMNITYPER_TEST_PYTHON="$PYTHON_BIN" swift test "${SWIFT_FLAGS[@]}"
"$PYTHON_BIN" -m unittest discover -s "$APP_ROOT/backend" -p 'test_*.py' -v
