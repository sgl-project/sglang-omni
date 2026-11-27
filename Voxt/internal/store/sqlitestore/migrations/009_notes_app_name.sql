-- VoxtNew database, migration 9 (11-15): the app a note came from;
-- old notes have no source app to show.
ALTER TABLE notes ADD COLUMN app_name TEXT NOT NULL DEFAULT '';
