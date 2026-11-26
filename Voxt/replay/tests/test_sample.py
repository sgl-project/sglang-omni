import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from aasr_sample import select  # noqa: E402


def utterances(spec):
    return [
        {"source_record_id": f"{lang}-{scene}-{i}", "language": lang, "scene": scene}
        for (lang, scene), n in spec.items()
        for i in range(n)
    ]


class SelectTest(unittest.TestCase):
    def test_proportional_largest_remainder(self):
        rows = utterances({("zh", "a"): 50, ("zh", "b"): 30, ("en", "a"): 20})
        chosen = select(rows, 7)
        counts = {}
        for r in chosen:
            counts[(r["language"], r["scene"])] = (
                counts.get((r["language"], r["scene"]), 0) + 1
            )
        # note (luojiaxuan): Quotas 3.5, 2.1, 1.4: floors 3, 2, 1, the one left goes to the largest remainder.
        self.assertEqual(counts, {("zh", "a"): 4, ("zh", "b"): 2, ("en", "a"): 1})

    def test_fixed_by_the_benchmark(self):
        rows = utterances({("zh", "a"): 40, ("en", "b"): 40})
        self.assertEqual(select(rows, 10), select(list(reversed(rows)), 10))


if __name__ == "__main__":
    unittest.main()
