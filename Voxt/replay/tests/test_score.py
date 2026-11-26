import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import score  # noqa: E402


class TokensTest(unittest.TestCase):
    # note (luojiaxuan): The cases of Voxt 2.0's TestTokens (Voxt/internal/eval/eval_test.go at voxt-v2.0.1).
    def test_voxt_cases(self):
        cases = [
            ("今天下午 review 一下这个 PR。", "今 天 下 午 review 1 下 这 个 pr"),
            ("嗯，三百五十个用户", "350 个 用 户"),
            ("350个用户", "350 个 用 户"),
            ("Ｑｗｅｎ３ 模型", "qwen3 模 型"),
            ("So um, twelve percent, twenty five; Let's go!", "so 12 25 let's go"),
            ("12% of it", "12 of it"),
            ("十二点 二〇二六年", "12 点 2026 年"),
        ]
        for text, want in cases:
            self.assertEqual(" ".join(score.tokens(text)), want, text)

    def test_full_width_punctuation_and_space(self):
        self.assertEqual(score.tokens("ＨＥＬＬＯ，　ＷＯＲＬＤ！"), ["hello", "world"])

    def test_fillers(self):
        self.assertEqual(score.tokens("嗯 呃 啊 um uh 好"), ["好"])
        self.assertEqual(
            score.tokens("嗯 好 um", keep_fillers=True), ["嗯", "好", "um"]
        )

    def test_apostrophes(self):
        self.assertEqual(score.tokens("'quoted' don't"), ["quoted", "don't"])

    def test_chinese_numbers(self):
        self.assertEqual(score.chinese_number("一万二千三百零五"), 12305)
        self.assertEqual(score.chinese_number("二〇二六"), 2026)
        self.assertEqual(score.chinese_number("十"), 10)

    def test_is_cjk(self):
        for c in "中文ひらカナ한":
            self.assertTrue(score.is_cjk(c), c)
        for c in "a1，。、":
            self.assertFalse(score.is_cjk(c), c)


class CompareTest(unittest.TestCase):
    # note (luojiaxuan): The cases of Voxt 2.0's TestCompare.
    def test_voxt_cases(self):
        s = score.compare(
            "今天下午三点开会，讨论 VoxtNew 的发布计划。",
            "今天下午3点开会，讨论 Voxt New 的发布计划",
        )
        self.assertEqual((s.n, s.sub, s.dele, s.ins), (16, 1, 0, 1))
        self.assertEqual(score.compare("a b c d", "a c d e").rate(), 0.5)
        self.assertEqual(score.compare("", "").rate(), 0)
        self.assertEqual(score.compare("", "x").rate(), 1)
        total = score.compare("a b", "a") + score.compare("c d", "c d")
        self.assertEqual((total.n, total.edits, total.rate()), (4, 1, 0.25))

    def test_empty_hypothesis_is_all_deletions(self):
        s = score.compare("one two three", "")
        self.assertEqual((s.n, s.sub, s.dele, s.ins), (3, 0, 3, 0))

    def test_mixed_language_counts_characters_and_words(self):
        s = score.compare("我们用 SGLang 跑", "我们用 sglang 跑步")
        self.assertEqual((s.n, s.edits), (5, 1))

    def test_script_is_no_error(self):
        s = score.compare(
            "我们两点开会，讨论语音输入的体验", "我們兩點開會，討論語音輸入的體驗"
        )
        self.assertEqual(s.edits, 0)
        self.assertEqual(score.tokens("兩點"), score.tokens("2点"))


class ScoreRunTest(unittest.TestCase):
    def setUp(self):
        self.manifest = [
            {"id": "en1", "lang": "en", "reference": "HELLO WORLD"},
            {"id": "en2", "lang": "en", "reference": "GOOD MORNING"},
            {"id": "zh1", "lang": "zh", "reference": "今天天气很好 [UNK]"},
            {"id": "sil", "lang": "none", "reference": ""},
            {"id": "gone", "lang": "en", "reference": "NOT RUN"},
        ]
        ms = 1_000_000
        self.results = [
            {
                "id": "en1",
                "status": "complete",
                "driver_fault": True,
                "final_text": "",
                "t_stop": 0,
            },
            {
                "id": "en1",
                "status": "complete",
                "driver_fault": False,
                "final_text": "Hello, world.",
                "t_stop": 1000 * ms,
                "t_first": 1200 * ms,
                "t_last": 1500 * ms,
            },
            {
                "id": "en2",
                "status": "empty",
                "driver_fault": False,
                "final_text": "",
                "t_stop": 0,
                "t_first": None,
                "t_last": None,
            },
            {
                "id": "zh1",
                "status": "complete",
                "driver_fault": False,
                "final_text": "今天天气很好。",
                "t_stop": 0,
                "t_first": 300 * ms,
                "t_last": 300 * ms,
            },
            {
                "id": "sil",
                "status": "complete",
                "driver_fault": False,
                "final_text": "嗯。",
                "t_stop": 0,
                "t_first": 100 * ms,
                "t_last": 100 * ms,
            },
        ]

    def test_error_rates_controls_and_latency(self):
        report = score.score_run(self.manifest, self.results)
        en, zh, all_ = report["per_lang"]["en"], report["per_lang"]["zh"], report["all"]
        # note (luojiaxuan): en: en1 exact (the faulted row is ignored), en2 empty -> 2 deletions out of 4 tokens.
        self.assertEqual((en["n_ref"], en["del"], en["error_rate"]), (4, 2, 0.5))
        # note (luojiaxuan): zh: [UNK] is removed from the reference.
        self.assertEqual((zh["n_ref"], zh["error_rate"]), (6, 0.0))
        self.assertEqual((all_["n_ref"], all_["error_rate"]), (10, 0.2))
        self.assertEqual(en["median_last_minus_stop_ms"], 500)
        self.assertEqual(en["median_first_minus_stop_ms"], 200)
        self.assertEqual(all_["median_last_minus_stop_ms"], 400)
        self.assertEqual(
            report["controls"], {"n": 1, "with_text": 1, "false_output_rate": 1.0}
        )
        self.assertEqual(report["statuses"], {"complete": 3, "empty": 1})
        self.assertEqual(report["missing"], ["gone"])

    def test_report_formats(self):
        text = score.format_report(
            score.score_run(self.manifest, self.results, keep_fillers=True)
        )
        self.assertIn("fillers kept", text)
        self.assertIn("false-output rate 100.0%", text)


class OutcomeTest(unittest.TestCase):
    def test_text_last_edited_before_the_stop_key_is_early(self):
        row = {"status": "complete", "t_stop": 100, "t_last": 99}
        self.assertEqual(score.outcome(row), "early")
        self.assertEqual(score.outcome({**row, "t_last": 100}), "complete")
        self.assertEqual(score.outcome({**row, "t_last": None}), "early")
        self.assertEqual(score.outcome({**row, "status": "empty"}), "empty")

    def test_early_rows_get_no_latency(self):
        manifest = [{"id": i, "lang": "en", "reference": "hi"} for i in "ab"]
        base = {
            "driver_fault": False,
            "status": "complete",
            "final_text": "hi",
            "t_stop": 5_000_000_000,
        }
        results = [
            {**base, "id": "a", "t_first": 5_300_000_000, "t_last": 5_400_000_000},
            {**base, "id": "b", "t_first": 1_000_000_000, "t_last": 2_000_000_000},
        ]
        report = score.score_run(manifest, results)
        self.assertEqual(report["statuses"], {"complete": 1, "early": 1})
        self.assertEqual(report["all"]["median_last_minus_stop_ms"], 400)

    def test_early_rows_are_counted_per_language(self):
        manifest = [{"id": "a", "lang": "en", "reference": "hi"}]
        results = [
            {
                "id": "a",
                "driver_fault": False,
                "status": "complete",
                "final_text": "hi",
                "t_stop": 5,
                "t_first": 1,
                "t_last": 2,
            }
        ]
        report = score.score_run(manifest, results)
        self.assertEqual(
            (report["per_lang"]["en"]["early"], report["all"]["early"]), (1, 1)
        )


class ContaminationTest(unittest.TestCase):
    def test_late_writes_mark_both_rows_and_leave_them_out(self):
        ms = 1_000_000
        manifest = [{"id": i, "lang": "en", "reference": "hi"} for i in "abcde"]
        base = {
            "driver_fault": False,
            "status": "complete",
            "final_text": "hi",
            "t_stop": 0,
            "t_first": 100 * ms,
            "t_last": 200 * ms,
        }
        results = [
            {**base, "id": "a"},
            # note (luojiaxuan): a's clean-up landed after a's observation ended, before b's clear.
            {**base, "id": "b", "late_changes_prev": 1},
            # note (luojiaxuan): b's clean-up landed after c's clear, so c's field held it at key-down.
            {**base, "id": "c", "n_changes_before_keydown": 1},
            {**base, "id": "d", "focus_lost": True},
            {**base, "id": "e", "t_last": 700 * ms, "final_text": "hi there"},
        ]
        report = score.score_run(manifest, results)
        self.assertEqual(report["statuses"], {"contaminated": 4, "complete": 1})
        self.assertEqual(
            report["contaminated"],
            {"late_write_into_next": 2, "changes_before_keydown": 1, "focus_lost": 1},
        )
        self.assertEqual(report["all"]["segments"], 1)
        self.assertEqual(report["all"]["ins"], 1)
        self.assertEqual(report["all"]["median_last_minus_stop_ms"], 700)
        self.assertIn(
            "contaminated, left out of error rates and waits: 4",
            score.format_report(report),
        )

    def test_rows_without_the_newer_fields_are_clean(self):
        rows = score.mark_contamination([{"id": "a"}, {"id": "b"}])
        self.assertEqual([r["contamination"] for r in rows], [[], []])


if __name__ == "__main__":
    unittest.main()
