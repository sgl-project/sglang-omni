package sqlitestore

import (
	"slices"
	"strconv"
	"testing"
)

// TestMigrationNumbering guards the numbering contract the migrator
// enforces: the embedded files are consecutive from 1. Migrations 9 and 10
// are the notes app name and the history reopen, shared with the notes and
// history changes byte for byte; 11 is dictionary profiles and 12 is meeting
// import mode.
func TestMigrationNumbering(t *testing.T) {
	for i, m := range migrations {
		if m.version != i+1 {
			t.Fatalf("migration %d is missing", i+1)
		}
	}
	if Version() != 12 {
		t.Fatalf("schema version %d, want 12", Version())
	}
}

// TestCombinedUpgrade opens a database at each schema the merged tree can
// meet — 8 (before the notes, history and profile changes), 9 (notes app
// name), 10 (plus history reopen), 11 (plus dictionary profiles) — and
// migrates it to 12: every change's columns land, the old dictionary's terms
// move into the active Default profile, and imported meetings are accepted.
func TestCombinedUpgrade(t *testing.T) {
	for _, from := range []int{8, 9, 10, 11} {
		t.Run(strconv.Itoa(from), func(t *testing.T) {
			saved := migrations
			t.Cleanup(func() { migrations = saved })

			dir := t.TempDir()
			migrations = slices.Clone(saved[:from])
			s := openTest(t, dir)
			if v, _ := userVersion(ctx, s.db); v != from {
				t.Fatalf("user_version %d, want %d", v, from)
			}
			termInsert := `INSERT INTO dictionary_terms (id, term, aliases, created_at, updated_at) VALUES ('t1', 'voxt', '["voxted"]', 1, 1)`
			if from == 11 {
				termInsert = `INSERT INTO dictionary_terms (profile_id, id, term, aliases, created_at, updated_at) VALUES ('default', 't1', 'voxt', '["voxted"]', 1, 1)`
			}
			if _, err := s.db.Execute(ctx, termInsert); err != nil {
				t.Fatal(err)
			}
			s.Close()

			migrations = saved
			s = openTest(t, dir)
			if v, _ := userVersion(ctx, s.db); v != 12 {
				t.Fatalf("user_version %d, want 12", v)
			}
			for table, cols := range map[string][]string{
				"notes":   {"app_name"},
				"history": {"title", "replaces"},
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
			if _, err := s.db.Execute(ctx, `INSERT INTO meetings (id, created_at, status, mode) VALUES ('imported', 1, 'done', 'import')`); err != nil {
				t.Errorf("import mode after upgrade: %v", err)
			}
		})
	}
}
