-- VoxtNew database, migration 5 (10-17): a recording kept only for a
-- while (store.Entry.AudioUntil, Unix ms; 0: kept as long as the entry):
-- recognition failed or found no text while the settings keep no
-- recordings, so it stays for a retry.
ALTER TABLE history ADD COLUMN audio_until INTEGER NOT NULL DEFAULT 0;
