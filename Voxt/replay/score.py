#!/usr/bin/env python3
"""Scores one replay-driver run (one cell).

Error rate: corpus-level sum(S+D+I) / sum(N_ref) per manifest language, over the
text in the field at the deadline (`final_text`). Tokens are CJK characters one by
one and Latin words, after Traditional characters become
simplified (`hanzi.py`, ICU's Hant-Hans on macOS; the script alone is no error) and the normalization of
Voxt's eval (`internal/eval/score.go` at voxt-v2.0.1), ported here:
full-width to half-width, lower case, Chinese numerals and English number words to
digits, punctuation dropped by tokenization, and fillers dropped from both sides
(`--keep-fillers` keeps them, a strict variant). `[UNK]` is removed from
references. Manifest rows with an empty reference are silence/noise controls:
they count towards the false-output rate, not the error rate.

Latency: medians of t_last - t_stop and t_first - t_stop (ms) over rows whose
outcome is "complete". A row with text whose last edit came before the stop key
is "early": its text cannot be the answer to the whole segment (a dropped
shortcut shifts every later segment by one), so it has no latency; the report
counts early rows next to each median.

A row is "contaminated" when its text or timing may belong to another segment:
the next segment logged a late write (late_changes_prev or
n_changes_before_keydown), the row itself saw changes between its clear and
key-down or lost focus, or the row before it was still being written at the end
of its observation (a timeout, whose writer can go on after this row's key-down). Contaminated rows count in neither the error rate nor the
latency, and the report prints how many there are and why.

Rows with driver_fault=true are ignored; when an id has several fault-free rows,
the last one counts.
"""

import argparse
import json
import statistics
import sys
import unicodedata
from bisect import bisect_right

from hanzi import simplified

FILLERS = frozenset(
    ["嗯", "呃", "额", "啊", "唔", "um", "uh", "er", "erm", "ah", "hmm"]
)

# note (luojiaxuan): Go 1.24's unicode.Han, Hiragana, Katakana and Hangul tables (Unicode 15.0.0), merged
# into sorted inclusive ranges; 2.0's eval counts these characters one token each.
_CJK_RANGES = [
    (0x1100, 0x11FF),
    (0x2E80, 0x2E99),
    (0x2E9B, 0x2EF3),
    (0x2F00, 0x2FD5),
    (0x3005, 0x3005),
    (0x3007, 0x3007),
    (0x3021, 0x3029),
    (0x302E, 0x302F),
    (0x3038, 0x303B),
    (0x3041, 0x3096),
    (0x309D, 0x309F),
    (0x30A1, 0x30FA),
    (0x30FD, 0x30FF),
    (0x3131, 0x318E),
    (0x31F0, 0x321E),
    (0x3260, 0x327E),
    (0x32D0, 0x32FE),
    (0x3300, 0x3357),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xA960, 0xA97C),
    (0xAC00, 0xD7A3),
    (0xD7B0, 0xD7C6),
    (0xD7CB, 0xD7FB),
    (0xF900, 0xFA6D),
    (0xFA70, 0xFAD9),
    (0xFF66, 0xFF6F),
    (0xFF71, 0xFF9D),
    (0xFFA0, 0xFFBE),
    (0xFFC2, 0xFFC7),
    (0xFFCA, 0xFFCF),
    (0xFFD2, 0xFFD7),
    (0xFFDA, 0xFFDC),
    (0x16FE2, 0x16FE3),
    (0x16FF0, 0x16FF1),
    (0x1AFF0, 0x1AFF3),
    (0x1AFF5, 0x1AFFB),
    (0x1AFFD, 0x1AFFE),
    (0x1B000, 0x1B122),
    (0x1B132, 0x1B132),
    (0x1B150, 0x1B152),
    (0x1B155, 0x1B155),
    (0x1B164, 0x1B167),
    (0x1F200, 0x1F200),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B739),
    (0x2B740, 0x2B81D),
    (0x2B820, 0x2CEA1),
    (0x2CEB0, 0x2EBE0),
    (0x2F800, 0x2FA1D),
    (0x30000, 0x3134A),
    (0x31350, 0x323AF),
]
_CJK_STARTS = [lo for lo, _ in _CJK_RANGES]

CN_DIGIT = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}

ENGLISH_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}


def is_cjk(c):
    o = ord(c)
    i = bisect_right(_CJK_STARTS, o) - 1
    return i >= 0 and o <= _CJK_RANGES[i][1]


def chinese_number(chars):
    """A run of Chinese numerals' value: with units ("三百五十", "十二") or digit by digit ("二〇二六")."""
    if not any(c in CN_UNIT for c in chars):
        n = 0
        for c in chars:
            n = n * 10 + CN_DIGIT[c]
        return n
    total = section = digit = 0
    for c in chars:
        unit = CN_UNIT.get(c, 0)
        if unit == 10000:
            total += (section + digit) * unit
            section = digit = 0
        elif unit > 0:
            if digit == 0:
                digit = 1  # note (luojiaxuan): "十二": one ten
            section += digit * unit
            digit = 0
        else:
            digit = CN_DIGIT[c]
    return total + section + digit


def normalize(text):
    """Folds Chinese script, full-width forms, case and Chinese numerals (written as digits)."""
    text = simplified(text)
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        o = ord(c)
        if 0xFF01 <= o <= 0xFF5E:
            c = chr(o - 0xFEE0)
        elif o == 0x3000:
            c = " "
        if c in CN_DIGIT or c in CN_UNIT:
            j = i
            while j < n and (text[j] in CN_DIGIT or text[j] in CN_UNIT):
                j += 1
            out.append(str(chinese_number(text[i:j])))
            i = j
            continue
        out.append(c.lower())
        i += 1
    return "".join(out)


def tokens(text, keep_fillers=False):
    """CJK characters and Latin words of the normalized text; fillers dropped unless keep_fillers."""
    raw, word = [], []

    def flush():
        if word:
            raw.append("".join(word))
            word.clear()

    for c in normalize(text):
        if is_cjk(c):
            flush()
            raw.append(c)
        elif c.isalpha() or unicodedata.category(c) == "Nd" or (c == "'" and word):
            word.append(c)
        else:
            flush()
    flush()

    kept = []
    i = 0
    while i < len(raw):
        t = raw[i].strip("'")
        n = ENGLISH_NUMBERS.get(t)
        if n is not None:
            # note (luojiaxuan): "twenty five" is 25.
            if n >= 20 and n % 10 == 0 and i + 1 < len(raw):
                u = ENGLISH_NUMBERS.get(raw[i + 1])
                if u is not None and u < 10:
                    n += u
                    i += 1
            t = str(n)
        i += 1
        if t == "" or t == "percent" or (not keep_fillers and t in FILLERS):
            continue
        kept.append(t)
    return kept


class Score:
    def __init__(self, n=0, sub=0, dele=0, ins=0):
        self.n, self.sub, self.dele, self.ins = n, sub, dele, ins

    @property
    def edits(self):
        return self.sub + self.dele + self.ins

    def rate(self):
        if self.n == 0:
            return 0.0 if self.edits == 0 else 1.0
        return self.edits / self.n

    def __add__(self, other):
        return Score(
            self.n + other.n,
            self.sub + other.sub,
            self.dele + other.dele,
            self.ins + other.ins,
        )

    def __repr__(self):
        return f"Score(n={self.n}, sub={self.sub}, del={self.dele}, ins={self.ins})"


def compare(ref, hyp, keep_fillers=False):
    """Levenshtein alignment of hyp against ref over tokens, with 2.0's backtrace order."""
    r, h = tokens(ref, keep_fillers), tokens(hyp, keep_fillers)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)
    s = Score(n=len(r))
    i, j = len(r), len(h)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] and r[i - 1] == h[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            s.sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            s.dele += 1
            i -= 1
        else:
            s.ins += 1
            j -= 1
    return s


def clean_reference(ref):
    return ref.replace("[UNK]", " ")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_results(rows):
    """id -> the last fault-free row."""
    picked = {}
    for row in rows:
        if not row.get("driver_fault"):
            picked[row["id"]] = row
    return picked


def mark_contamination(results):
    """Copies of the rows, in the order they ran, each with the reasons its text or
    timing may belong to another segment under "contamination"."""
    marked = [dict(row, contamination=[]) for row in results]
    for prev, row in zip([None] + marked[:-1], marked):
        late_here = row.get("n_changes_before_keydown", 0) > 0
        if late_here:
            row["contamination"].append("changes_before_keydown")
        if row.get("focus_lost"):
            row["contamination"].append("focus_lost")
        if prev is not None and prev.get("status") == "timeout":
            row["contamination"].append("after_timeout")
        # note (luojiaxuan): A write that comes after a segment's observation ended is logged in the
        # next segment, before its clear (late_changes_prev) or between its clear and key-down. The
        # earlier row then misses that write, and the later one may keep it as its final text.
        if prev is not None and (late_here or row.get("late_changes_prev", 0) > 0):
            prev["contamination"].append("late_write_into_next")
    return marked


def outcome(row):
    """complete, contaminated, early, empty or timeout: the driver's status, with
    text that was last edited before the stop key told apart from text written
    after it, and rows marked by mark_contamination set aside."""
    if row.get("contamination"):
        return "contaminated"
    if row["status"] != "complete":
        return row["status"]
    if row.get("t_last") is None or row["t_last"] < row["t_stop"]:
        return "early"
    return "complete"


def median_ms(values):
    return statistics.median(values) / 1e6 if values else None


def score_run(manifest, results, keep_fillers=False):
    picked = pick_results(mark_contamination(results))
    langs = {}
    controls = {"n": 0, "with_text": 0}
    statuses = {}
    contaminated = {}
    missing = []
    for item in manifest:
        row = picked.get(item["id"])
        if row is None:
            missing.append(item["id"])
            continue
        result = outcome(row)
        statuses[result] = statuses.get(result, 0) + 1
        if result == "contaminated":
            for reason in row["contamination"]:
                contaminated[reason] = contaminated.get(reason, 0) + 1
            continue
        hyp = row.get("final_text") or ""
        ref = item.get("reference") or ""
        if not ref.strip():
            controls["n"] += 1
            controls["with_text"] += 1 if hyp.strip() else 0
            continue
        g = langs.setdefault(
            item["lang"],
            {"score": Score(), "segments": 0, "early": 0, "last": [], "first": []},
        )
        g["score"] += compare(clean_reference(ref), hyp, keep_fillers)
        g["segments"] += 1
        g["early"] += result == "early"
        if result == "complete":
            g["last"].append(row["t_last"] - row["t_stop"])
            if row.get("t_first") is not None:
                g["first"].append(row["t_first"] - row["t_stop"])

    def summary(score, segments, early, last, first):
        return {
            "segments": segments,
            "n_ref": score.n,
            "sub": score.sub,
            "del": score.dele,
            "ins": score.ins,
            "error_rate": score.rate() if score.n else None,
            "complete_with_latency": len(last),
            "early": early,
            "median_last_minus_stop_ms": median_ms(last),
            "median_first_minus_stop_ms": median_ms(first),
        }

    per_lang = {
        lang: summary(g["score"], g["segments"], g["early"], g["last"], g["first"])
        for lang, g in sorted(langs.items())
    }
    total = Score()
    last, first = [], []
    for g in langs.values():
        total += g["score"]
        last += g["last"]
        first += g["first"]
    return {
        "keep_fillers": keep_fillers,
        "per_lang": per_lang,
        "all": summary(
            total,
            sum(g["segments"] for g in langs.values()),
            sum(g["early"] for g in langs.values()),
            last,
            first,
        ),
        "controls": {
            **controls,
            "false_output_rate": (
                controls["with_text"] / controls["n"] if controls["n"] else None
            ),
        },
        "statuses": statuses,
        "contaminated": contaminated,
        "missing": missing,
    }


def format_report(report):
    def num(v, fmt):
        return "-" if v is None else format(v, fmt)

    lines = [
        "lang      segs   N_ref     S     D     I   error  med(t_last-t_stop)  med(t_first-t_stop)  early"
    ]
    rows = list(report["per_lang"].items()) + [("all", report["all"])]
    for lang, s in rows:
        err = None if s["error_rate"] is None else 100 * s["error_rate"]
        lines.append(
            f"{lang:<8} {s['segments']:>5} {s['n_ref']:>7} {s['sub']:>5} {s['del']:>5} {s['ins']:>5} "
            f"{num(err, '6.2f'):>6}%  {num(s['median_last_minus_stop_ms'], '10.0f'):>14} ms"
            f"  {num(s['median_first_minus_stop_ms'], '10.0f'):>15} ms  {s['early']:>5}"
        )
    c = report["controls"]
    lines.append(
        f"controls: {c['with_text']}/{c['n']} wrote text (false-output rate "
        f"{num(None if c['false_output_rate'] is None else 100 * c['false_output_rate'], '.1f')}%)"
    )
    lines.append(
        "statuses: "
        + ", ".join(f"{k}={v}" for k, v in sorted(report["statuses"].items()))
    )
    reasons = ", ".join(f"{k}={v}" for k, v in sorted(report["contaminated"].items()))
    lines.append(
        f"contaminated, left out of error rates and waits: "
        f"{report['statuses'].get('contaminated', 0)} ({reasons or 'none'})"
    )
    if report["missing"]:
        lines.append(
            f"missing results: {len(report['missing'])} ids (first: {report['missing'][0]})"
        )
    lines.append(
        "fillers kept (strict)" if report["keep_fillers"] else "fillers dropped"
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--manifest", required=True, help="JSONL with id, lang, reference"
    )
    parser.add_argument("--results", required=True, help="replay-driver results JSONL")
    parser.add_argument(
        "--keep-fillers",
        action="store_true",
        help="strict variant: fillers count as tokens",
    )
    parser.add_argument("--json", help="also write the report as JSON to this path")
    args = parser.parse_args(argv)
    report = score_run(
        load_jsonl(args.manifest), load_jsonl(args.results), args.keep_fillers
    )
    print(format_report(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
