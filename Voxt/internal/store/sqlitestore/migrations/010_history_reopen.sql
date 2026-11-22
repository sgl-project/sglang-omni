-- VoxtNew database, migration 10: reopening a card needs its answer titles
-- and which earlier answer each redo replaced. (Migration 9 is the notes
-- app's name, from the notes change; the two must not share a number.)
ALTER TABLE history ADD COLUMN title TEXT NOT NULL DEFAULT '';
ALTER TABLE history ADD COLUMN replaces TEXT NOT NULL DEFAULT '';
