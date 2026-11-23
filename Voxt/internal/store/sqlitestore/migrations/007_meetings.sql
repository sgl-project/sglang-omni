-- VoxtNew database, migration 7 (C1-1b-store): meeting records and segments.
CREATE TABLE meetings (
  seq          INTEGER PRIMARY KEY,
  id           TEXT NOT NULL UNIQUE,
  created_at   INTEGER NOT NULL,
  ended_at     INTEGER,
  status       TEXT NOT NULL CHECK (status IN ('recording','interrupted','asking','finishing','done','failed')),
  mode         TEXT NOT NULL CHECK (mode IN ('meeting','subtitles','recording')),
  title        TEXT NOT NULL DEFAULT '',
  duration_ms  INTEGER NOT NULL DEFAULT 0,
  asr_model    TEXT NOT NULL DEFAULT '',
  diar_model   TEXT NOT NULL DEFAULT '',
  llm_model    TEXT NOT NULL DEFAULT '',
  speakers     TEXT NOT NULL DEFAULT '{}',
  summary      TEXT NOT NULL DEFAULT '',
  chat         TEXT NOT NULL DEFAULT '[]',
  audio        TEXT NOT NULL DEFAULT '',
  audio_until  INTEGER
);
CREATE INDEX meetings_created ON meetings(created_at DESC);
CREATE TABLE meeting_segments (
  meeting_seq  INTEGER NOT NULL REFERENCES meetings(seq) ON DELETE CASCADE,
  idx          INTEGER NOT NULL,
  source       TEXT NOT NULL CHECK (source IN ('mic','system')),
  speaker      TEXT NOT NULL DEFAULT '',
  start_ms     INTEGER NOT NULL,
  end_ms       INTEGER NOT NULL,
  text         TEXT NOT NULL,
  translation  TEXT NOT NULL DEFAULT '',
  final        INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (meeting_seq, idx)
);
CREATE INDEX meeting_segments_meeting ON meeting_segments(meeting_seq, idx);
CREATE VIRTUAL TABLE meeting_segments_fts USING fts5(text, translation, content='meeting_segments', content_rowid='rowid', tokenize='trigram');
CREATE TRIGGER meeting_segments_fts_insert AFTER INSERT ON meeting_segments BEGIN
  INSERT INTO meeting_segments_fts(rowid, text, translation) VALUES (new.rowid, new.text, new.translation);
END;
CREATE TRIGGER meeting_segments_fts_delete AFTER DELETE ON meeting_segments BEGIN
  INSERT INTO meeting_segments_fts(meeting_segments_fts, rowid, text, translation) VALUES ('delete', old.rowid, old.text, old.translation);
END;
CREATE TRIGGER meeting_segments_fts_update AFTER UPDATE OF text, translation ON meeting_segments BEGIN
  INSERT INTO meeting_segments_fts(meeting_segments_fts, rowid, text, translation) VALUES ('delete', old.rowid, old.text, old.translation);
  INSERT INTO meeting_segments_fts(rowid, text, translation) VALUES (new.rowid, new.text, new.translation);
END;
