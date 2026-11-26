import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        doc = run_cells.configure(system, "en")
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
        saved = run_cells.save_settings(plan, root)
        self.assertEqual(len(saved), 1)
        run_cells.configure(
            {"family": "2x", "db": str(self.db), "asr": "asr.x", "enhance": True}, "en"
        )
        # note (luojiaxuan): A second start, as after a killed run, keeps the first copy.
        run_cells.save_settings(plan, root)
        run_cells.restore_settings(saved)
        self.assertEqual(run_cells.settings_doc(str(self.db)), SETTINGS)
        self.assertEqual(list((root / "settings-backup").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
