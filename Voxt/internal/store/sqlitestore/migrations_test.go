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

func TestCombinedUpgradeLegacyImportVersion9(t *testing.T) {
	saved := migrations
	t.Cleanup(func() { migrations = saved })

	dir := t.TempDir()
	migrations = slices.Clone(saved[:8])
	migrations = append(migrations, migration{version: 9, statements: saved[11].statements})
	s := openTest(t, dir)
	if v, _ := userVersion(ctx, s.db); v != 9 {
		t.Fatalf("user_version %d, want the old import migration's 9", v)
	}
	if _, err := s.db.Execute(ctx, `INSERT INTO notes (id, text, created_at) VALUES ('n1', 'keep me', 1)`); err != nil {
		t.Fatal(err)
	}
	if _, err := s.db.Execute(ctx, `INSERT INTO meetings (id, created_at, status, mode) VALUES ('m1', 1, 'done', 'import')`); err != nil {
		t.Fatal(err)
	}
	s.Close()

	migrations = saved
	s = openTest(t, dir)
	rows, err := s.db.Query(ctx, "PRAGMA table_info(notes)")
	if err != nil {
		t.Fatal(err)
	}
	hasAppName := false
	for _, row := range rows.Values {
		hasAppName = hasAppName || asString(row[1]) == "app_name"
	}
	if !hasAppName {
		t.Fatal("the old import version 9 skipped the notes app_name migration")
	}
	notes, err := s.Notes().List(ctx)
	if err != nil || len(notes) != 1 || notes[0].Text != "keep me" {
		t.Errorf("notes after upgrade: %+v, %v", notes, err)
	}
	_, segments, err := s.Meetings().Meeting(ctx, "m1")
	if err != nil || len(segments) != 0 {
		t.Errorf("meeting after upgrade: %+v, %v", segments, err)
	}
}

func TestCombinedUpgradeLegacyDictionaryVersion9(t *testing.T) {
	saved := migrations
	t.Cleanup(func() { migrations = saved })

	dir := t.TempDir()
	migrations = slices.Clone(saved[:8])
	migrations = append(migrations, migration{version: 9, statements: saved[10].statements})
	s := openTest(t, dir)
	if v, _ := userVersion(ctx, s.db); v != 9 {
		t.Fatalf("user_version %d, want the old dictionary migration's 9", v)
	}
	if _, err := s.db.Execute(ctx, `INSERT INTO dictionary_profiles (id, name, active) VALUES ('work', 'Work', 0)`); err != nil {
		t.Fatal(err)
	}
	if _, err := s.db.Execute(ctx, `INSERT INTO dictionary_terms (profile_id, id, term, aliases, note, enabled, created_at, updated_at) VALUES ('work', 't1', 'voxt', '["voxted"]', 'keep', 1, 1, 1)`); err != nil {
		t.Fatal(err)
	}
	s.Close()

	migrations = saved
	s = openTest(t, dir)
	if v, _ := userVersion(ctx, s.db); v != 12 {
		t.Fatalf("user_version %d, want 12", v)
	}
	rows, err := s.db.Query(ctx, "PRAGMA table_info(notes)")
	if err != nil {
		t.Fatal(err)
	}
	hasAppName := false
	for _, row := range rows.Values {
		hasAppName = hasAppName || asString(row[1]) == "app_name"
	}
	if !hasAppName {
		t.Fatal("the old dictionary version 9 skipped the notes app_name migration")
	}
	if err := s.Dictionary().ActivateProfile(ctx, "work"); err != nil {
		t.Fatal(err)
	}
	terms, err := s.Dictionary().List(ctx, "")
	if err != nil || len(terms) != 1 || terms[0].Term != "voxt" || terms[0].Note != "keep" || !slices.Equal(terms[0].Aliases, []string{"voxted"}) {
		t.Errorf("work terms after upgrade: %+v, %v", terms, err)
	}
	profiles, err := s.Dictionary().Profiles(ctx)
	if err != nil || len(profiles) != 2 {
		t.Errorf("profiles after upgrade: %+v, %v", profiles, err)
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
