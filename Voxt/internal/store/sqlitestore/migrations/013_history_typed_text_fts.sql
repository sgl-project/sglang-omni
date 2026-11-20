-- Include Rewrite's typed input in history search and reindex existing rows.
DROP TRIGGER history_fts_insert;
DROP TRIGGER history_fts_delete;
DROP TRIGGER history_fts_update;
DROP TABLE history_fts;
CREATE VIRTUAL TABLE history_fts USING fts5(raw_text, source_text, prompt_text, result, typed_text, content='history', content_rowid='seq', tokenize='trigram');
CREATE TRIGGER history_fts_insert AFTER INSERT ON history BEGIN
  INSERT INTO history_fts(rowid, raw_text, source_text, prompt_text, result, typed_text) VALUES (new.seq, new.raw_text, new.source_text, new.prompt_text, new.result, new.typed_text);
END;
CREATE TRIGGER history_fts_delete AFTER DELETE ON history BEGIN
  INSERT INTO history_fts(history_fts, rowid, raw_text, source_text, prompt_text, result, typed_text) VALUES ('delete', old.seq, old.raw_text, old.source_text, old.prompt_text, old.result, old.typed_text);
END;
CREATE TRIGGER history_fts_update AFTER UPDATE OF raw_text, source_text, prompt_text, result, typed_text ON history BEGIN
  INSERT INTO history_fts(history_fts, rowid, raw_text, source_text, prompt_text, result, typed_text) VALUES ('delete', old.seq, old.raw_text, old.source_text, old.prompt_text, old.result, old.typed_text);
  INSERT INTO history_fts(rowid, raw_text, source_text, prompt_text, result, typed_text) VALUES (new.seq, new.raw_text, new.source_text, new.prompt_text, new.result, new.typed_text);
END;
INSERT INTO history_fts(rowid, raw_text, source_text, prompt_text, result, typed_text) SELECT seq, raw_text, source_text, prompt_text, result, typed_text FROM history;
