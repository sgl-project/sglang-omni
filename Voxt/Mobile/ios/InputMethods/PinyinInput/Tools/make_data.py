"""Build McBopomofo-format pinyin data from the rime-wanxiang dictionaries.

Rows are "reading value score": reading = toneless syllables joined by '-'
(ü typed as v), score = log10(weight / total weight kept). Sorted by reading
bytes, then score high to low, as ParselessPhraseDB's binary search expects.

Usage: python3 -I make_data.py [WORDS [TWO_CHAR [OUT]]]
  WORDS: words kept from jichu (default 250000), by weight.
  TWO_CHAR: of those, how many are two-character words (default 162000: all
  but the ~900 lightest; pruning by weight alone drops common pairs like 大床).
  OUT: folder under out/. DROP (optional): word<TAB>reading lines to leave out,
  from find_redundant.py.
"""

import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HEADER = "# format org.openvanilla.mcbopomofo.sorted"
HAN = re.compile(r"^[㐀-䶿一-鿿豈-﫿\U00020000-\U0003ffff]+$")
SYLLABLE = re.compile(r"^[a-z]+$")
MAX_SYLLABLES = 8  # ReadingGrid::kMaximumSpanLength


def toneless(pinyin):
    text = unicodedata.normalize("NFD", pinyin.lower()).replace("ü", "v")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.split()


def parse(lines):
    rows, body = [], False
    for line in lines:
        line = line.rstrip("\n")
        if not body:
            body = line.startswith("...")
            continue
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3 or not parts[2].strip().isdigit():
            continue
        word, syllables = parts[0], toneless(parts[1])
        if (
            not HAN.match(word)
            or len(syllables) != len(word)
            or len(word) > MAX_SYLLABLES
        ):
            continue
        if not all(SYLLABLE.match(s) for s in syllables):
            continue
        rows.append((word, tuple(syllables), int(parts[2])))
    return rows


def merge(rows):
    merged = defaultdict(int)
    for word, syllables, weight in rows:
        merged[(word, syllables)] += weight
    return dict(merged)


def prune(entries, count, two_char=None):
    ranked = sorted(entries.items(), key=lambda item: (-item[1], item[0]))
    if two_char is None:
        return dict(ranked[:count])
    pairs = [item for item in ranked if len(item[0][0]) == 2][:two_char]
    longer = [item for item in ranked if len(item[0][0]) != 2][: count - len(pairs)]
    return dict(pairs + longer)


def render(entries):
    total = sum(entries.values())
    rows = []
    for (word, syllables), weight in entries.items():
        score = math.log10(max(weight, 1) / total)
        rows.append(("-".join(syllables), word, score))
    rows.sort(key=lambda row: ((row[0] + " ").encode(), -row[2], row[1]))
    return (
        "\n".join(
            [HEADER]
            + [f"{reading} {word} {score:.6f}" for reading, word, score in rows]
        )
        + "\n"
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    words = int(sys.argv[1]) if len(sys.argv) > 1 else 250_000
    two_char = (
        (int(sys.argv[2]) if sys.argv[2] != "-" else None)
        if len(sys.argv) > 2
        else 162_000
    )
    source = here / "source"
    zi = merge(parse((source / "zi.dict.yaml").open(encoding="utf-8")))
    jichu = merge(parse((source / "jichu.dict.yaml").open(encoding="utf-8")))
    drop = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    if drop:
        for line in drop.read_text(encoding="utf-8").splitlines():
            word, reading = line.split("\t")
            jichu.pop((word, tuple(reading.split("-"))), None)
    jichu = prune(jichu, words, two_char)
    entries = {**zi, **jichu}
    out = here / "out" / (sys.argv[3] if len(sys.argv) > 3 else "")
    out.mkdir(exist_ok=True)
    data = out / "pinyin-data.txt"
    data.write_text(render(entries), encoding="utf-8")
    manifest = {
        "source": "https://github.com/amzxyz/rime-wanxiang (CC-BY-4.0), dicts/zi.dict.yaml + dicts/jichu.dict.yaml, modified",
        "revision": (source / "COMMIT").read_text().strip(),
        "inputs": {
            name: sha256(source / name)
            for name in ("zi.dict.yaml", "jichu.dict.yaml", "LICENSE")
        },
        "generator_sha256": sha256(Path(__file__)),
        "characters": len(zi),
        "words": len(jichu),
        "two_char_quota": two_char,
        "dropped": {"file": drop.name, "sha256": sha256(drop)} if drop else None,
        "rows": len(entries),
        "bytes": data.stat().st_size,
        "output_sha256": sha256(data),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
