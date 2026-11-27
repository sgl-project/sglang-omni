-- VoxtNew database, migration 11: named dictionaries. Existing terms move
-- into the active Default profile. (Migrations 9 and 10 are the notes app's
-- name and the history reopen, from the notes and history changes; the
-- three must not share a number.)
CREATE TABLE dictionary_profiles (
  id     TEXT PRIMARY KEY,
  name   TEXT NOT NULL COLLATE NOCASE UNIQUE CHECK (name != ''),
  active INTEGER NOT NULL DEFAULT 0 CHECK (active IN (0, 1))
);
CREATE UNIQUE INDEX dictionary_one_active ON dictionary_profiles(active) WHERE active = 1;
INSERT INTO dictionary_profiles (id, name, active) VALUES ('default', 'Default', 1);

CREATE TABLE dictionary_terms_profiled (
  profile_id TEXT NOT NULL REFERENCES dictionary_profiles(id) ON DELETE CASCADE,
  id         TEXT NOT NULL,
  term       TEXT NOT NULL COLLATE NOCASE CHECK (term != ''),
  aliases    TEXT NOT NULL DEFAULT '[]',
  note       TEXT NOT NULL DEFAULT '',
  enabled    INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (profile_id, id),
  UNIQUE (profile_id, term)
);
INSERT INTO dictionary_terms_profiled
  SELECT 'default', id, term, aliases, note, enabled, created_at, updated_at FROM dictionary_terms;
DROP TABLE dictionary_terms;
ALTER TABLE dictionary_terms_profiled RENAME TO dictionary_terms;
