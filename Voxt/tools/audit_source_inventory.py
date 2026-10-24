#!/usr/bin/env python3
"""Review inventory, NOT a dead-code/security verdict. Never deletes source/data.

Only tracked app/test Swift sources are scanned; generated files and fixtures are
excluded. Reports locations/names, not matching lines (which may contain secrets).
Lexical single-occurrence symbols still require protocol/selector/resource,
conditional compilation, migration, access-control and business-entry review.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "blocking_file_read": r"\b(?:Data|String)\(contentsOf:",
    "directory_walk": r"\b(?:enumerator|contentsOfDirectory|allocatedSizeOfDirectory)\(",
    "repeating_timer": r"Timer\.(?:publish|scheduledTimer)|\.scheduleRepeating\(",
    "unstructured_task": r"\bTask(?:\.detached)?(?:\s*<[^>\n]+>)?\s*(?:\([^\n]*\))?\s*\{",
    "unchecked_sendable": r"@unchecked\s+Sendable|nonisolated\(unsafe\)",
    "crash_or_forced_try": r"\bfatalError\(|\bpreconditionFailure\(|\btry!",
    "process_or_script": r"\bProcess\(|\bNSAppleScript\(|\bNSWorkspace\.shared\.open\(",
    "destructive_file_operation": r"\b(?:removeItem|moveItem)\(",
    "credential_or_content_log_review": r"(?:VoxtLog\.|\blogger\.|\bprint\().*(?:token|key|credential|prompt|transcript|endpoint|text)",
    "test_skip_or_placeholder": r"\bXCTSkip\b|\bTODO\b|\bFIXME\b|\bfatalError\(|\bpreconditionFailure\(",
}
DECLARATIONS = re.compile(
    r"^[ \t]*(?:(?:nonisolated|private|fileprivate|internal|public|final|static|class|override|mutating)[ \t]+)*"
    r"(func|struct|enum|class|typealias|var|let)\s+(\w+)", re.M
)


def inventory(sources):
    tokens = Counter(token for text in sources.values() for token in re.findall(r"\b\w+\b", text))
    findings = {key: [] for key in RULES}
    candidates = []
    imports = {}
    for path, text in sorted(sources.items()):
        for number, line in enumerate(text.splitlines(), 1):
            for key, pattern in RULES.items():
                if re.search(pattern, line, re.I if key == "credential_or_content_log_review" else 0):
                    findings[key].append({"path": path, "line": number})
        for match in DECLARATIONS.finditer(text):
            kind, name = match.groups()
            if name != "_" and tokens[name] == 1:
                # Includes locals/callbacks and false positives by design.
                candidates.append({"path": path, "line": text.count("\n", 0, match.start()) + 1,
                                   "kind": kind, "name": name})
        for module in re.findall(r"^\s*(?:@preconcurrency\s+)?import\s+(\w+)", text, re.M):
            imports.setdefault(module, []).append(path)
    return {
        "warning": "Lexical review candidates only; no reachability, vulnerability or performance proof.",
        "file_count": len(sources),
        "line_count": sum(len(text.splitlines()) for text in sources.values()),
        "files_over_1000_lines": [{"path": path, "lines": len(text.splitlines())}
                                  for path, text in sorted(sources.items()) if len(text.splitlines()) > 1000],
        "single_occurrence_candidates": candidates,
        "review_locations": findings,
        "import_users": dict(sorted(imports.items())),
    }


def tracked_sources(root):
    paths = subprocess.check_output(["git", "ls-files", "-z", "--", "Voxt", "VoxtTests"], cwd=root).decode().split("\0")
    return {path: (root / path).read_text() for path in paths
            if path.endswith(".swift") and not path.startswith("VoxtTests/Fixtures/") and (root / path).is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inventory(tracked_sources(ROOT))
    report["head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report["working_tree_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    data = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            parser.error("output already exists; choose a fresh report path")
        args.output.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
