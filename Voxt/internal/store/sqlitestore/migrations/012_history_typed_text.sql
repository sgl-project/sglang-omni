-- VoxtNew database, migration 12: the Rewrite input box's typed text,
-- restored for another thinking level and follow-ups after a restart.
ALTER TABLE history ADD COLUMN typed_text TEXT NOT NULL DEFAULT '';
