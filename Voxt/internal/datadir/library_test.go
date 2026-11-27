package datadir

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

func putDoc(t *testing.T, st *sqlitestore.Store, key, doc string) {
	t.Helper()
	if err := st.Documents().Put(context.Background(), key, []byte(doc)); err != nil {
		t.Fatal(err)
	}
}

func docs(t *testing.T, dir string, keys ...string) []string {
	t.Helper()
	st, err := sqlitestore.Open(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	var out []string
	for _, k := range keys {
		doc, _, err := st.Documents().Get(context.Background(), k)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, string(doc))
	}
	return out
}

// One Mac exports its library while Voxt runs; the other imports it at
// its next start, with the paths pointing at its own data folder and its
// own library kept beside.
func TestLibraryExportImport(t *testing.T) {
	ctx := context.Background()
	macA, macB := t.TempDir(), t.TempDir()
	st, err := sqlitestore.Open(ctx, macA)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	putDoc(t, st, "settings:app", `{"from":"A"}`)
	putDoc(t, st, "test:path", `{"audio":"`+macA+`/audio/a.wav"}`)
	file := filepath.Join(t.TempDir(), "voxt.voxtlibrary")
	if err := ExportLibrary(ctx, macA, file); err != nil {
		t.Fatal(err)
	}
	putDoc(t, st, "settings:app", `{"from":"A, after the export"}`)

	other, err := sqlitestore.Open(ctx, macB)
	if err != nil {
		t.Fatal(err)
	}
	putDoc(t, other, "settings:app", `{"from":"B"}`)
	other.Close()
	if err := StageImport(ctx, macB, file); err != nil {
		t.Fatal(err)
	}
	if got := docs(t, macB, "settings:app")[0]; got != `{"from":"B"}` {
		t.Fatalf("staging changed the library before the restart: %s", got)
	}
	if got, err := Resolve(macB); err != nil || got != macB {
		t.Fatalf("Resolve = %q, %v", got, err)
	}
	got := docs(t, macB, "settings:app", "test:path")
	if got[0] != `{"from":"A"}` || got[1] != `{"audio":"`+macB+`/audio/a.wav"}` {
		t.Errorf("imported %q", got)
	}
	for _, name := range []string{ImportFile, importedDB} {
		if _, err := os.Stat(filepath.Join(macB, name)); !os.IsNotExist(err) {
			t.Errorf("%s is still there", name)
		}
	}
	kept := t.TempDir()
	if err := os.Rename(filepath.Join(macB, BeforeImport), filepath.Join(kept, sqlitestore.FileName)); err != nil {
		t.Fatal(err)
	}
	if got := docs(t, kept, "settings:app")[0]; got != `{"from":"B"}` {
		t.Errorf("the replaced library = %s", got)
	}
}

func TestStageImportRefusesOtherFiles(t *testing.T) {
	dir := t.TempDir()
	openDB(t, dir)
	notes := filepath.Join(t.TempDir(), "notes.voxtlibrary")
	write(t, filepath.Dir(notes), map[string]string{"notes.voxtlibrary": "not a database"})
	if err := StageImport(context.Background(), dir, notes); err == nil {
		t.Fatal("StageImport accepted a text file")
	}
	for _, name := range []string{ImportFile, importedDB} {
		if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
			t.Errorf("%s was left after a refused import", name)
		}
	}
}

// A sound SQLite file of another app, at a schema version Voxt reads, is
// refused too: the library stays as it was after the restart. Table names
// alone do not make it a Voxt library.
func TestStageImportRefusesForeignSQLite(t *testing.T) {
	for _, tc := range []struct {
		name   string
		schema string
	}{
		{"missing Voxt tables", "CREATE TABLE notes (x);"},
		{"same table names with wrong columns", "CREATE TABLE documents (x); CREATE TABLE history (x); CREATE TABLE dictionary_terms (x);"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			dir := t.TempDir()
			st, err := sqlitestore.Open(ctx, dir)
			if err != nil {
				t.Fatal(err)
			}
			putDoc(t, st, "settings:app", `{"from":"mine"}`)
			st.Close()
			db := filepath.Join(dir, sqlitestore.FileName)
			before, err := os.ReadFile(db)
			if err != nil {
				t.Fatal(err)
			}
			foreign := filepath.Join(t.TempDir(), "other.voxtlibrary")
			out, err := exec.Command(sqlite3, foreign, tc.schema+fmt.Sprintf("PRAGMA user_version = %d;", sqlitestore.Version())).CombinedOutput()
			if err != nil {
				t.Fatalf("%v: %s", err, out)
			}
			if err := StageImport(ctx, dir, foreign); err == nil {
				t.Fatal("StageImport accepted another app's SQLite file")
			}
			assertNoImport := func() {
				t.Helper()
				for _, name := range []string{ImportFile, importedDB, BeforeImport} {
					if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
						t.Errorf("%s was left after a refused import", name)
					}
				}
			}
			assertNoImport()

			// Resolve checks a staged file again before replacing the library.
			if _, err := copyVerified(foreign, filepath.Join(dir, importedDB)); err != nil {
				t.Fatal(err)
			}
			if err := WriteJSON(filepath.Join(dir, ImportFile), struct{}{}); err != nil {
				t.Fatal(err)
			}
			if _, err := Resolve(dir); err != nil {
				t.Fatal(err)
			}
			assertNoImport()
			after, err := os.ReadFile(db)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(after, before) {
				t.Error("the refused import changed the current library")
			}
			if got := docs(t, dir, "settings:app")[0]; got != `{"from":"mine"}` {
				t.Errorf("the library after the refused import: %s", got)
			}
		})
	}
}
