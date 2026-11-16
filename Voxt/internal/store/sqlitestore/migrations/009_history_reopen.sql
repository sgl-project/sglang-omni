-- VoxtNew database, migration 9: reopening a card needs its answer titles
-- and which earlier answer each redo replaced.
ALTER TABLE history ADD COLUMN title TEXT NOT NULL DEFAULT '';
ALTER TABLE history ADD COLUMN replaces TEXT NOT NULL DEFAULT '';
