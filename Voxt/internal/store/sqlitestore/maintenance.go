package sqlitestore

// Maintenance: counts for Settings → Storage and the self-check, and, for
// moving the data folder (done while the database is closed, at the
// app's next start: the plugin refuses VACUUM INTO), the path rewrite of a
// copied database and an integrity check. (From AudioSmith's
// sqlitestore/maintenance.go; its usage statistics are not VoxtNew's.)

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/egoist/mygo/plugins/sqlite"
)

// Stats counts the records of the library.
type Stats struct {
	History    int // entries of every kind
	Dictionary int // terms
}

// Stats counts the records.
func (s *Store) Stats(ctx context.Context) (Stats, error) {
	var st Stats
	rows, err := s.db.Query(ctx, `SELECT (SELECT COUNT(*) FROM history), (SELECT COUNT(*) FROM dictionary_terms)`)
	if err != nil {
		return st, wrap("stats", err)
	}
	if len(rows.Values) == 1 {
		st.History, st.Dictionary = int(asInt(rows.Values[0][0])), int(asInt(rows.Values[0][1]))
	}
	return st, nil
}

// CheckFile opens the database at path, runs SQLite's integrity check,
// makes sure it is a Voxt library (the first migration's tables, columns
// and keys) and reports the schema version it has.
func CheckFile(ctx context.Context, path string) (int, error) {
	db, err := sqlite.Open(ctx, path, sqlite.OpenOptions{})
	if err != nil {
		return 0, err
	}
	defer db.Close()
	rows, err := db.Query(ctx, "PRAGMA integrity_check")
	if err != nil {
		return 0, err
	}
	if len(rows.Values) == 0 || asString(rows.Values[0][0]) != "ok" {
		var msgs []string
		for _, r := range rows.Values {
			msgs = append(msgs, asString(r[0]))
		}
		return 0, fmt.Errorf("sqlitestore: %s: integrity check: %s", path, strings.Join(msgs, "; "))
	}
	v, err := userVersion(ctx, db)
	if err != nil {
		return 0, err
	}
	for _, want := range librarySchema {
		rows, err := db.Query(ctx, fmt.Sprintf("PRAGMA table_info(%q)", want.table))
		if err != nil {
			return 0, err
		}
		have := make(map[string]bool, len(rows.Values))
		pkPos := make(map[int]string, len(want.pk))
		for _, r := range rows.Values {
			name := asString(r[1])
			have[name] = true
			if n := int(asInt(r[5])); n > 0 {
				pkPos[n] = name
			}
		}
		if len(have) == 0 {
			return 0, fmt.Errorf("sqlitestore: %s is not a Voxt library (no table %s)", path, want.table)
		}
		for _, col := range want.columns {
			if !have[col] {
				return 0, fmt.Errorf("sqlitestore: %s is not a Voxt library (table %s has no column %s)", path, want.table, col)
			}
		}
		pk := make([]string, 0, len(pkPos))
		for i := 1; i <= len(pkPos); i++ {
			pk = append(pk, pkPos[i])
		}
		wantPK := want.pk
		if want.table == "dictionary_terms" && v >= 11 {
			wantPK = []string{"profile_id", "id"}
		}
		if !sameColumns(pk, wantPK) {
			return 0, fmt.Errorf("sqlitestore: %s is not a Voxt library (table %s key %s)", path, want.table, strings.Join(pk, ", "))
		}
	}
	// The store upserts history by id (ON CONFLICT(id)), a unique column
	// from migration 1 rather than the seq primary key.
	if ok, err := hasUniqueColumn(ctx, db, "history", "id"); err != nil {
		return 0, err
	} else if !ok {
		return 0, fmt.Errorf("sqlitestore: %s is not a Voxt library (table history has no unique id)", path)
	}
	return v, nil
}

func sameColumns(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// hasUniqueColumn reports whether table has a unique index on column
// alone.
func hasUniqueColumn(ctx context.Context, db *sqlite.DB, table, column string) (bool, error) {
	indexes, err := db.Query(ctx, fmt.Sprintf("PRAGMA index_list(%q)", table))
	if err != nil {
		return false, err
	}
	for _, r := range indexes.Values {
		if asInt(r[2]) != 1 { // 1 = unique
			continue
		}
		cols, err := db.Query(ctx, fmt.Sprintf("PRAGMA index_info(%q)", asString(r[1])))
		if err != nil {
			return false, err
		}
		if len(cols.Values) == 1 && asString(cols.Values[0][2]) == column {
			return true, nil
		}
	}
	return false, nil
}

// librarySchema holds the first migration's columns and primary key,
// which every Voxt library has: later migrations only add columns.
// dictionary_terms gains profile_id in its key in migration 11. A foreign
// database can reuse the table names, so their shape tells them apart.
var librarySchema = []struct {
	table   string
	pk      []string
	columns []string
}{
	{"documents", []string{"key"}, []string{"key", "doc", "updated_at"}},
	{"history", []string{"seq"}, []string{
		"seq", "id", "kind", "created_at", "status", "error", "app_bundle", "app_name", "url_host",
		"rule_id", "rule_name", "raw_text", "source_text", "prompt_text", "result", "target_lang",
		"delivery", "asr_provider", "asr_model", "llm_provider", "llm_model", "audio_ms", "asr_ms",
		"llm_ms", "total_ms", "audio",
	}},
	{"dictionary_terms", []string{"id"}, []string{"id", "term", "aliases", "note", "enabled", "created_at", "updated_at"}},
}

// RewritePaths replaces the folder oldDir by newDir at the start of the
// paths the database at path holds in its text columns (as plain text,
// or inside JSON strings), so that a copied data folder points at its
// own files. Full-text indexes follow through their triggers. It
// returns how many rows changed.
func RewritePaths(ctx context.Context, path, oldDir, newDir string) (int64, error) {
	oldDir, newDir = strings.TrimRight(oldDir, "/"), strings.TrimRight(newDir, "/")
	if oldDir == "" || newDir == "" || oldDir == newDir {
		return 0, errors.New("sqlitestore: RewritePaths needs two different folders")
	}
	db, err := sqlite.Open(ctx, path, sqlite.OpenOptions{})
	if err != nil {
		return 0, err
	}
	defer db.Close()
	tables, err := db.Query(ctx, `SELECT name FROM sqlite_master WHERE type = 'table' AND sql NOT LIKE 'CREATE VIRTUAL%'
		AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%'`)
	if err != nil {
		return 0, err
	}
	// The forms a folder takes in the database: as it is, and escaped
	// in JSON (Go escapes <, > and &; "/" stays as it is).
	jsonForm := func(s string) string {
		b, _ := json.Marshal(s)
		return string(b[1 : len(b)-1])
	}
	pairs := [][2]string{{oldDir + "/", newDir + "/"}}
	if j, k := jsonForm(oldDir), jsonForm(newDir); j != oldDir || k != newDir {
		pairs = append(pairs, [2]string{j + "/", k + "/"})
	}
	var stmts []sqlite.Statement
	for _, t := range tables.Values {
		table := asString(t[0])
		cols, err := db.Query(ctx, fmt.Sprintf("PRAGMA table_info(%q)", table))
		if err != nil {
			return 0, err
		}
		for _, c := range cols.Values {
			col, typ := asString(c[1]), strings.ToUpper(asString(c[2]))
			if typ != "TEXT" && typ != "" {
				continue
			}
			for _, p := range pairs {
				stmts = append(stmts, sqlite.Statement{
					SQL:  fmt.Sprintf("UPDATE %q SET %q = replace(%q, ?, ?) WHERE instr(%q, ?) > 0", table, col, col, col),
					Args: []any{p[0], p[1], p[0]},
				})
				// A column holding exactly the folder (no trailing slash).
				stmts = append(stmts, sqlite.Statement{
					SQL:  fmt.Sprintf("UPDATE %q SET %q = ? WHERE %q = ?", table, col, col),
					Args: []any{strings.TrimSuffix(p[1], "/"), strings.TrimSuffix(p[0], "/")},
				})
			}
		}
	}
	if len(stmts) == 0 {
		return 0, nil
	}
	res, err := db.Transaction(ctx, stmts)
	if err != nil {
		return 0, err
	}
	var n int64
	for _, r := range res {
		n += r.Changes
	}
	return n, nil
}
