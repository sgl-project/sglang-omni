set -euo pipefail

TARGET="${1:-docs/cookbook}"

echo "Checking cookbook target: ${TARGET}"

if [ ! -e "${TARGET}" ]; then
  echo "ERROR: target does not exist: ${TARGET}"
  exit 1
fi

python3 - "$TARGET" <<'PY'
from pathlib import Path
import re
import sys

target = Path(sys.argv[1])

if target.is_file():
    files = [target]
else:
    files = sorted(target.rglob("*.md"))

if not files:
    print(f"ERROR: no markdown files found under {target}")
    sys.exit(1)

failed = False

for path in files:
    text = path.read_text(encoding="utf-8")

    if not text.strip():
        print(f"ERROR: empty cookbook file: {path}")
        failed = True

    fence_count = len(re.findall(r"^`{3,}", text, flags=re.MULTILINE))
    if fence_count % 2 != 0:
        print(f"ERROR: unmatched fenced code block in {path}")
        failed = True

    if re.search(r"\b(TODO|FIXME)\b", text):
        print(f"ERROR: TODO/FIXME found in {path}")
        failed = True

    for match in re.findall(r"examples/configs/[A-Za-z0-9_\-./]+\.ya?ml", text):
        config_path = Path(match)
        if not config_path.exists():
            print(f"ERROR: referenced config does not exist in {path}: {match}")
            failed = True

if failed:
    sys.exit(1)

print("Cookbook checks passed.")
PY
