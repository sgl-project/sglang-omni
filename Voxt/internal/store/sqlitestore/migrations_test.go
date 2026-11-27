package sqlitestore

import (
	"slices"
	"strconv"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestMigrationNumbering guards the numbering contract the migrator
// enforces: the embedded files are consecutive from 1. Migrations 9 and 10
// are the notes app name and the history reopen, shared with the notes and
// history changes byte for byte; 11 is the dictionary profiles, 12 is
// the Rewrite input box's typed text and 13 indexes that text for search.
func TestMigrationNumbering(t *testing.T) {
	for i, m := range migrations {
		if m.version != i+1 {
			t.Fatalf("migration %d is missing", i+1)
		}
	}
	if Version() != 13 {
		t.Fatalf("schema version %d, want 13", Version())
	}
}

// TestCombinedUpgrade opens a database at each schema the merged tree can
// meet — 8 (before the notes, history and profile changes), 9 (notes app
// name), 10 (plus history reopen), 11 (dictionary profiles) — and migrates
// it to 13: every change's columns land and the old dictionary's terms move
// into the active Default profile. A typed Rewrite saved at 12 joins the
// search index.
func TestCombinedUpgrade(t *testing.T) {
	for _, from := range []int{8, 9, 10, 11, 12} {
		t.Run(strconv.Itoa(from), func(t *testing.T) {
			saved := migrations
			t.Cleanup(func() { migrations = saved })

			dir := t.TempDir()
			migrations = slices.Clone(saved[:from])
			s := openTest(t, dir)
			if v, _ := userVersion(ctx, s.db); v != from {
				t.Fatalf("user_version %d, want %d", v, from)
			}
			insertTerm := `INSERT INTO dictionary_terms (id, term, aliases, created_at, updated_at) VALUES ('t1', 'voxt', '["voxted"]', 1, 1)`
			if from >= 11 {
				insertTerm = `INSERT INTO dictionary_terms (profile_id, id, term, aliases, created_at, updated_at) VALUES ('default', 't1', 'voxt', '["voxted"]', 1, 1)`
			}
			if _, err := s.db.Execute(ctx, insertTerm); err != nil {
				t.Fatal(err)
			}
			if from == 12 {
				if _, err := s.db.Execute(ctx, `INSERT INTO history (id, kind, created_at, typed_text) VALUES ('typed-upgrade', 'rewrite', 1, 'upgrade-only-typed')`); err != nil {
					t.Fatal(err)
				}
			}
			s.Close()

			migrations = saved
			s = openTest(t, dir)
			if v, _ := userVersion(ctx, s.db); v != Version() {
				t.Fatalf("user_version %d, want %d", v, Version())
			}
			for table, cols := range map[string][]string{
				"notes":   {"app_name"},
				"history": {"title", "replaces", "typed_text"},
			} {
				rows, err := s.db.Query(ctx, "PRAGMA table_info("+table+")")
				if err != nil {
					t.Fatal(err)
				}
				var have []string
				for _, r := range rows.Values {
					have = append(have, asString(r[1]))
				}
				for _, col := range cols {
					if !slices.Contains(have, col) {
						t.Errorf("%s has no %s column: %v", table, col, have)
					}
				}
			}
			terms, err := s.Dictionary().List(ctx, "")
			if err != nil {
				t.Fatal(err)
			}
			if len(terms) != 1 || terms[0].Term != "voxt" || !slices.Equal(terms[0].Aliases, []string{"voxted"}) {
				t.Errorf("terms after upgrade: %+v", terms)
			}
			profiles, err := s.Dictionary().Profiles(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if len(profiles) != 1 || profiles[0].Name != "Default" || !profiles[0].Active {
				t.Errorf("profiles after upgrade: %+v", profiles)
			}
			if from == 12 {
				entries, err := s.History().List(ctx, store.HistoryQuery{Search: "upgrade-only-typed"})
				if err != nil || len(entries) != 1 || entries[0].ID != "typed-upgrade" {
					t.Errorf("typed text after upgrade: entries %+v, error %v", entries, err)
				}
			}
		})
	}
}
