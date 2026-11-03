# SPDX-License-Identifier: Apache-2.0
"""Wait until WeChat shows a message asking you to release your Radix GPUs.

Reads chats that are open as standalone WeChat windows through the macOS
accessibility tree, prints one JSON event line and exits. Run it as a background
task of an agent session: nothing reaches the model until it fires.
"""

import argparse
import ctypes
import ctypes.util
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEN_CAP = 2000
FAIL_GRACE_S = 30
MISSING_GRACE_S = 2 * 3600

_CF = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
_CG = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
_CG.CGSessionCopyCurrentDictionary.restype = ctypes.c_void_p
_CF.CFStringCreateWithCString.restype = ctypes.c_void_p
_CF.CFStringCreateWithCString.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_uint32,
]
_CF.CFDictionaryGetValue.restype = ctypes.c_void_p
_CF.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_CF.CFBooleanGetValue.argtypes = [ctypes.c_void_p]
_CF.CFRelease.argtypes = [ctypes.c_void_p]
_LOCK_KEY = _CF.CFStringCreateWithCString(None, b"CGSSessionScreenIsLocked", 0x08000100)


def screen_locked():
    session = _CG.CGSessionCopyCurrentDictionary()
    value = _CF.CFDictionaryGetValue(session, _LOCK_KEY)
    locked = bool(value) and bool(_CF.CFBooleanGetValue(value))
    _CF.CFRelease(session)
    return locked


def read_bubbles(chats):
    out = subprocess.run(
        ["osascript", "-l", "JavaScript", str(HERE / "read_bubbles.js"), *chats],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        chat: (
            None
            if cells is None
            else [text or f"<non-text bubble h={h}>" for text, h in cells]
        )
        for chat, cells in json.loads(out).items()
    }


def new_tokens(prev, cur, seen):
    # note (luojiaxuan): new messages push older ones up, so the still-visible part of the last snapshot is a prefix of this one; no overlap means the view was scrolled, so fall back to "never seen before".
    k = next(
        (k for k in range(min(len(prev), len(cur)), 0, -1) if prev[-k:] == cur[:k]),
        0,
    )
    if k == 0 and prev:
        return [t for t in cur if t not in seen]
    return cur[k:]


def emit(event):
    event["time"] = datetime.now().astimezone().isoformat(timespec="seconds")
    print(json.dumps(event, ensure_ascii=False), flush=True)


def save(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False))
    tmp.replace(path)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--group",
        action="append",
        default=[],
        help="window title of a group chat; a message there counts only if it contains --mention",
    )
    ap.add_argument(
        "--direct",
        action="append",
        default=[],
        help="window title of a private chat (e.g. the admin); no mention needed",
    )
    ap.add_argument(
        "--mention", help='how others @ you in the group, e.g. "@YourWeChatName"'
    )
    ap.add_argument(
        "--keywords",
        default="释放,让出,release",
        help="comma-separated words that mark a release request",
    )
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=2)
    ap.add_argument(
        "--rebaseline",
        action="store_true",
        help="record the current messages as seen and exit (run after replying)",
    )
    args = ap.parse_args()
    if not args.group and not args.direct:
        ap.error("give at least one --group or --direct chat")
    if args.group and not args.mention:
        ap.error("--group needs --mention")
    return args


def main():
    args = parse_args()
    keywords = re.compile(
        "|".join(re.escape(k.strip()) for k in args.keywords.split(",") if k.strip()),
        re.I,
    )
    groups = set(args.group)
    mention = (args.mention or "").lower()
    chats = args.group + args.direct
    state = json.loads(args.state.read_text()) if args.state.exists() else {}

    def is_request(chat, text):
        if chat in groups and mention not in text.lower():
            return False
        return bool(keywords.search(text))

    if args.rebaseline:
        for chat, cur in read_bubbles(chats).items():
            if cur is None:
                raise SystemExit(f"{chat}: window not readable; open it and run again")
            entry = state.setdefault(chat, {"prev": [], "seen": []})
            entry["prev"] = cur
            entry["seen"] = (entry["seen"] + cur)[-SEEN_CAP:]
        save(state, args.state)
        return

    missing_since = None
    failing_since = None
    while True:
        try:
            snapshot = read_bubbles(chats)
        except subprocess.CalledProcessError as e:
            # note (luojiaxuan): the accessibility read fails now and then while WeChat redraws a window; only a failure that persists stops the relay.
            failing_since = failing_since or time.monotonic()
            if time.monotonic() - failing_since >= FAIL_GRACE_S:
                raise RuntimeError(f"osascript kept failing: {e.stderr.strip()}") from e
            time.sleep(args.interval)
            continue
        failing_since = None

        missing = [chat for chat, cur in snapshot.items() if cur is None]
        # note (luojiaxuan): the accessibility tree only lists windows on the current Space and nothing while the screen is locked, so a missing window usually means "out of reach for now"; only one gone for MISSING_GRACE_S of unlocked time is reported as possibly closed.
        if missing:
            if screen_locked():
                missing_since = None
            else:
                missing_since = missing_since or time.monotonic()
                if time.monotonic() - missing_since >= MISSING_GRACE_S:
                    emit(
                        {
                            "events": [
                                {"chat": c, "reason": "window_missing"} for c in missing
                            ]
                        }
                    )
                    return
            time.sleep(args.interval)
            continue
        missing_since = None

        events = []
        for chat, cur in snapshot.items():
            if chat not in state:
                state[chat] = {"prev": cur, "seen": cur}
                continue
            entry = state[chat]
            fresh = new_tokens(entry["prev"], cur, set(entry["seen"]))
            entry["prev"] = cur
            entry["seen"] = (entry["seen"] + fresh)[-SEEN_CAP:]
            hits = [t for t in fresh if is_request(chat, t)]
            if hits:
                events.append(
                    {"chat": chat, "reason": "release_request", "messages": hits}
                )
        save(state, args.state)

        if events:
            emit({"events": events})
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
