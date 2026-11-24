-- VoxtNew database, migration 14: a fresh Default dictionary starts with
-- Voxt and SGLang. A database that already holds a term is left alone.
INSERT INTO dictionary_terms (profile_id, id, term, aliases, note, enabled, created_at, updated_at)
  SELECT 'default', id, term, '[]', '', 1, strftime('%s', 'now') * 1000, strftime('%s', 'now') * 1000
  FROM (SELECT 'voxt' AS id, 'Voxt' AS term UNION ALL SELECT 'sglang', 'SGLang')
  WHERE NOT EXISTS (SELECT 1 FROM dictionary_terms) AND EXISTS (SELECT 1 FROM dictionary_profiles WHERE id = 'default');
