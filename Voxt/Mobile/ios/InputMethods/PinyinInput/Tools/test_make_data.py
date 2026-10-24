import math
import unittest

import make_data


class MakeDataTests(unittest.TestCase):
    def test_tones_are_removed_and_u_umlaut_becomes_v(self):
        self.assertEqual(make_data.toneless("nǐ hǎo"), ["ni", "hao"])
        self.assertEqual(make_data.toneless("nǚ lüè"), ["nv", "lve"])
        self.assertEqual(make_data.toneless("Xī Ān"), ["xi", "an"])

    def test_rows_need_one_syllable_per_character(self):
        rows = make_data.parse(
            [
                "# c",
                "---",
                "...",
                "你好\tnǐ hǎo\t10",
                "T恤\txù\t5",
                "A股\tā gǔ\t3",
                "啊\ta\t1",
                "bad",
            ]
        )
        self.assertEqual(rows, [("你好", ("ni", "hao"), 10), ("啊", ("a",), 1)])

    def test_rows_differing_only_in_tone_merge(self):
        merged = make_data.merge(
            [("啊", ("a",), 7), ("啊", ("a",), 3), ("阿", ("a",), 2)]
        )
        self.assertEqual(merged, {("啊", ("a",)): 10, ("阿", ("a",)): 2})

    def test_output_is_sorted_by_reading_then_score(self):
        text = make_data.render(
            {
                ("你", ("ni",)): 30,
                ("尼", ("ni",)): 60,
                ("你好", ("ni", "hao")): 10,
                ("年", ("nian",)): 10,
            }
        )
        lines = text.splitlines()
        self.assertEqual(lines[0], "# format org.openvanilla.mcbopomofo.sorted")
        self.assertEqual(
            [line.split(" ")[:2] for line in lines[1:]],
            [["ni", "尼"], ["ni", "你"], ["ni-hao", "你好"], ["nian", "年"]],
        )
        self.assertAlmostEqual(
            float(lines[1].split(" ")[2]), math.log10(60 / 110), places=4
        )

    def test_prune_keeps_the_heaviest_words(self):
        kept = make_data.prune(
            {
                ("甲乙", ("jia", "yi")): 1,
                ("丙丁", ("bing", "ding")): 9,
                ("戊己", ("wu", "ji")): 5,
            },
            2,
        )
        self.assertEqual(
            set(kept), {("丙丁", ("bing", "ding")), ("戊己", ("wu", "ji"))}
        )

    def test_two_character_quota_comes_first(self):
        entries = {
            ("甲乙", ("jia", "yi")): 1,
            ("丙丁", ("bing", "ding")): 2,
            ("戊己庚", ("wu", "ji", "geng")): 9,
        }
        self.assertEqual(
            set(make_data.prune(entries, 2, two_char=2)),
            {("甲乙", ("jia", "yi")), ("丙丁", ("bing", "ding"))},
        )
        self.assertEqual(
            set(make_data.prune(entries, 2, two_char=1)),
            {("丙丁", ("bing", "ding")), ("戊己庚", ("wu", "ji", "geng"))},
        )


if __name__ == "__main__":
    unittest.main()
