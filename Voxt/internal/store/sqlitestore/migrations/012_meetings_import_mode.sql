-- VoxtNew database, migration 12: a meeting can be an imported recording
-- (mode 'import'). The mode CHECK lives on the table, so the table is
-- rebuilt. meeting_segments references meetings with ON DELETE CASCADE
-- and DROP TABLE deletes the parent's rows through it, so the lines are
-- parked while the swap happens; their triggers rebuild the FTS index as
-- they come back.
CREATE TABLE meetings_new (
  seq          INTEGER PRIMARY KEY,
  id           TEXT NOT NULL UNIQUE,
  created_at   INTEGER NOT NULL,
  ended_at     INTEGER,
  status       TEXT NOT NULL CHECK (status IN ('recording','interrupted','asking','finishing','done','failed')),
  mode         TEXT NOT NULL CHECK (mode IN ('meeting','subtitles','recording','import')),
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
INSERT INTO meetings_new SELECT * FROM meetings;
CREATE TABLE meeting_segments_park AS SELECT * FROM meeting_segments;
DELETE FROM meeting_segments;
DROP TABLE meetings;
ALTER TABLE meetings_new RENAME TO meetings;
INSERT INTO meeting_segments (meeting_seq, idx, source, speaker, start_ms, end_ms, text, translation, final)
SELECT meeting_seq, idx, source, speaker, start_ms, end_ms, text, translation, final FROM meeting_segments_park;
DROP TABLE meeting_segments_park;
CREATE INDEX meetings_created ON meetings(created_at DESC);
