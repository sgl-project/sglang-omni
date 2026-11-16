-- VoxtNew database, migration 8: the answer card's conversation
-- a session belongs to (store.Entry.Conversation), the ID of its first
-- session; '' for older entries.
ALTER TABLE history ADD COLUMN conversation TEXT NOT NULL DEFAULT '';
