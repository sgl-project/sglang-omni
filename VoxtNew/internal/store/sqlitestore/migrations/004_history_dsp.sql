-- VoxtNew database, migration 4 (10-16): the sound processing the
-- speech service's copy of a recording went through (store.Entry.DSP:
-- "highpass+level", "" for none or unknown) and the gain it got
-- (store.Entry.GainDB); the recording kept is always the original.
ALTER TABLE history ADD COLUMN dsp TEXT NOT NULL DEFAULT '';
ALTER TABLE history ADD COLUMN gain_db REAL NOT NULL DEFAULT 0;
