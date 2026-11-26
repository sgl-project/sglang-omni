import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app_settings  # noqa: E402
import run_cells  # noqa: E402

SETTINGS = {
    "general": {"mainLanguage": "zh"},
    "features": {"transcribe": {"asr": "asr.fast", "enhance": False}},
    "providers": {"llm": [{"id": "ollama", "model": "qwen3:4b"}]},
}


def make_db(path: Path) -> None:
    con = sqlite3.connect(path)
    with con:
        con.execute("create table documents (key text primary key, doc text)")
        con.execute(
            "insert into documents values ('settings:app', ?)", [json.dumps(SETTINGS)]
        )
    con.close()


class ConfigureTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = Path(self.dir.name) / "voxtnew.db"
        make_db(self.db)

    def tearDown(self):
        self.dir.cleanup()

    def test_values_are_stored_as_given(self):
        system = {
            "family": "2x",
            "db": str(self.db),
            "asr": "asr.accurate",
            "enhance": True,
            "clean_after_stop": False,
            "reclean_sentences": 2,
            "llm": "ollama",
            "llm_model": "it's:4b",
        }
        doc = app_settings.configure(system, "en")
        transcribe = doc["features"]["transcribe"]
        self.assertEqual(doc["general"]["mainLanguage"], "en")
        self.assertEqual(
            (transcribe["asr"], transcribe["enhance"], transcribe["cleanup"]),
            ("asr.accurate", True, True),
        )
        self.assertIs(transcribe["cleanAfterStop"], False)
        self.assertEqual(transcribe["recleanSentences"], 2)
        self.assertEqual(doc["providers"]["llm"][0]["model"], "it's:4b")

    def test_settings_come_back_after_the_run(self):
        root = Path(self.dir.name) / "run"
        root.mkdir()
        plan = {
            "systems": {
                "a": {"family": "2x", "db": str(self.db)},
                "b": {"family": "2x", "db": str(self.db)},
                "c": {"family": "other"},
            }
        }
        saved = app_settings.save_settings(plan, root)
        self.assertEqual(len(saved), 1)
        app_settings.configure(
            {"family": "2x", "db": str(self.db), "asr": "asr.x", "enhance": True}, "en"
        )
        # note (luojiaxuan): A second start, as after a killed run, keeps the first copy.
        app_settings.save_settings(plan, root)
        con = sqlite3.connect(self.db)
        with con:
            con.execute("insert into documents values ('history:1', '{}')")
        con.close()
        app_settings.restore_settings(saved)
        self.assertEqual(app_settings.settings_doc(str(self.db)), SETTINGS)
        con = sqlite3.connect(self.db)
        self.assertEqual(
            con.execute(
                "select count(*) from documents where key='history:1'"
            ).fetchone(),
            (1,),
        )
        con.close()
        self.assertEqual(list((root / "settings-backup").iterdir()), [])

    def test_each_cell_starts_from_the_saved_settings(self):
        root = Path(self.dir.name) / "run"
        root.mkdir()
        plan = {"systems": {"a": {"family": "2x", "db": str(self.db)}}}
        saved = app_settings.save_settings(plan, root)
        first = {"family": "2x", "db": str(self.db), "asr": "asr.x", "enhance": True}
        app_settings.configure(
            {**first, "clean_after_stop": True}, "en", "BlackHole2ch_UID"
        )
        app_settings.reset_settings(saved)
        doc = app_settings.configure(first, "zh")
        self.assertNotIn("cleanAfterStop", doc["features"]["transcribe"])
        self.assertNotIn("audio", doc)
        app_settings.reset_settings(saved)
        doc = app_settings.configure(first, "zh", "BlackHole2ch_UID")
        self.assertEqual(doc["audio"]["devices"], ["BlackHole2ch_UID"])

    def test_a_partial_backup_stops_the_run(self):
        root = Path(self.dir.name) / "run"
        (root / "settings-backup").mkdir(parents=True)
        plan = {"systems": {"a": {"family": "2x", "db": str(self.db)}}}
        saved = app_settings.save_settings(plan, root)
        saved[0][2].write_text('{"general": ')
        with self.assertRaises(SystemExit):
            app_settings.save_settings(plan, root)


class ReuseTest(unittest.TestCase):
    def test_rows_of_another_run_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            cell = Path(d)
            rows = [{"id": "a", "wav": "/x/a.wav", "lang": "en"}]
            system = {"family": "other", "key": "fn", "trigger": "hold"}
            plan = {"settle_s": 3}
            self.assertEqual(run_cells.reuse_problem(cell, rows, system, plan), "")
            run_cells.write_jsonl(cell / "manifest.jsonl", rows)
            run_cells.write_jsonl(
                cell / "cells.jsonl", [{"plan": system, "settle_s": 3}]
            )
            self.assertEqual(run_cells.reuse_problem(cell, rows, system, plan), "")
            other = [{"id": "b", "wav": "/x/b.wav", "lang": "en"}]
            self.assertIn(
                "manifest", run_cells.reuse_problem(cell, other, system, plan)
            )
            self.assertIn(
                "configuration",
                run_cells.reuse_problem(
                    cell, rows, {**system, "key": "right_option"}, plan
                ),
            )
            self.assertIn(
                "configuration",
                run_cells.reuse_problem(cell, rows, system, {"settle_s": 10}),
            )


if __name__ == "__main__":
    unittest.main()
