-- VoxtNew database, migration 2 (M5): how a history entry's app
-- enhancement group matched (store.Entry.RuleSource): "url" (a URL rule
-- of the browser tab), "browserApp" (the browser is a member), "app"
-- (the app in front is a member); '' when no group matched.
ALTER TABLE history ADD COLUMN rule_source TEXT NOT NULL DEFAULT '';
