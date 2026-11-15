-- VoxtNew database, migration 1 (M1): documents, history, the dictionary.
--
-- Times are Unix milliseconds. Text search uses FTS5 with the trigram
-- tokenizer (substring search that works for CJK); queries under 3
-- characters use LIKE on the columns instead. Full-text indexes are
-- external-content tables kept by triggers: the plugin runs a group of
-- writes atomically only as a fixed list of statements, so derived data
-- lives in triggers rather than in read-modify-write code.

-- Small JSON documents (store.Documents): "settings:app" (the settings),
-- "appctx:rules" (app enhancement groups, M5), "notifications".
CREATE TABLE documents (
  key        TEXT PRIMARY KEY,
  doc        TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

-- One row per finished session (store.Entry). seq is the rowid the
-- full-text index points at; id is the ULID the app uses.
-- * raw_text: recognised speech; source_text: the selection (translate,
--   rewrite); prompt_text: the spoken instruction (rewrite); result: what
--   was delivered.
-- * app_bundle, app_name, url_host: the app in front at the shortcut, and
--   its tab's host; rule_id, rule_name: the app enhancement group matched.
-- * audio_ms … total_ms: stage times; audio: a file name in
--   <PathUserData>/audio/, deleted with the row by the repository.
CREATE TABLE history (
  seq          INTEGER PRIMARY KEY,
  id           TEXT NOT NULL UNIQUE,
  kind         TEXT NOT NULL CHECK (kind IN ('transcribe', 'translate', 'rewrite')),
  created_at   INTEGER NOT NULL,
  status       TEXT NOT NULL DEFAULT 'done' CHECK (status IN ('done', 'failed', 'cancelled')),
  error        TEXT NOT NULL DEFAULT '',
  app_bundle   TEXT NOT NULL DEFAULT '',
  app_name     TEXT NOT NULL DEFAULT '',
  url_host     TEXT NOT NULL DEFAULT '',
  rule_id      TEXT NOT NULL DEFAULT '',
  rule_name    TEXT NOT NULL DEFAULT '',
  raw_text     TEXT NOT NULL DEFAULT '',
  source_text  TEXT NOT NULL DEFAULT '',
  prompt_text  TEXT NOT NULL DEFAULT '',
  result       TEXT NOT NULL DEFAULT '',
  target_lang  TEXT NOT NULL DEFAULT '',
  delivery     TEXT NOT NULL DEFAULT '',
  asr_provider TEXT NOT NULL DEFAULT '',
  asr_model    TEXT NOT NULL DEFAULT '',
  llm_provider TEXT NOT NULL DEFAULT '',
  llm_model    TEXT NOT NULL DEFAULT '',
  audio_ms     INTEGER NOT NULL DEFAULT 0,
  asr_ms       INTEGER NOT NULL DEFAULT 0,
  llm_ms       INTEGER NOT NULL DEFAULT 0,
  total_ms     INTEGER NOT NULL DEFAULT 0,
  audio        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX history_created ON history(created_at DESC);
CREATE INDEX history_kind ON history(kind, created_at DESC);
CREATE VIRTUAL TABLE history_fts USING fts5(raw_text, source_text, prompt_text, result, content='history', content_rowid='seq', tokenize='trigram');
CREATE TRIGGER history_fts_insert AFTER INSERT ON history BEGIN
  INSERT INTO history_fts(rowid, raw_text, source_text, prompt_text, result) VALUES (new.seq, new.raw_text, new.source_text, new.prompt_text, new.result);
END;
CREATE TRIGGER history_fts_delete AFTER DELETE ON history BEGIN
  INSERT INTO history_fts(history_fts, rowid, raw_text, source_text, prompt_text, result) VALUES ('delete', old.seq, old.raw_text, old.source_text, old.prompt_text, old.result);
END;
CREATE TRIGGER history_fts_update AFTER UPDATE OF raw_text, source_text, prompt_text, result ON history BEGIN
  INSERT INTO history_fts(history_fts, rowid, raw_text, source_text, prompt_text, result) VALUES ('delete', old.seq, old.raw_text, old.source_text, old.prompt_text, old.result);
  INSERT INTO history_fts(rowid, raw_text, source_text, prompt_text, result) VALUES (new.seq, new.raw_text, new.source_text, new.prompt_text, new.result);
END;

-- The user's dictionary (store.Term). term is unique ignoring case;
-- aliases is a JSON array of strings.
CREATE TABLE dictionary_terms (
  id         TEXT PRIMARY KEY,
  term       TEXT NOT NULL COLLATE NOCASE UNIQUE CHECK (term != ''),
  aliases    TEXT NOT NULL DEFAULT '[]',
  note       TEXT NOT NULL DEFAULT '',
  enabled    INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
