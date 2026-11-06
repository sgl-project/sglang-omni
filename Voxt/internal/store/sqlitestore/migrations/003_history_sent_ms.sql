-- VoxtNew database, migration 3 (10-15): how much of a recording the
-- speech service got (store.Entry.SentMS), once the local voice activity
-- detection left its silence out; 0 for older entries (unknown) and for
-- sessions that sent nothing.
ALTER TABLE history ADD COLUMN sent_ms INTEGER NOT NULL DEFAULT 0;
