-- VoxtNew database, migration 6 (11-07): the user's notes (store.Note),
-- what they said after the note shortcut; not part of the history.
CREATE TABLE notes (
  id         TEXT PRIMARY KEY,
  text       TEXT NOT NULL CHECK (text != ''),
  created_at INTEGER NOT NULL
);
CREATE INDEX notes_created ON notes (created_at DESC);
