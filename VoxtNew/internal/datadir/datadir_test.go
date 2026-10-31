package datadir

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store/sqlitestore"
)

// write makes the files of tree under dir: path → contents.
func write(t *testing.T, dir string, tree map[string]string) {
	t.Helper()
	for p, data := range tree {
		full := filepath.Join(dir, p)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// read returns the regular files under dir: path → contents.
func read(t *testing.T, dir string) map[string]string {
	t.Helper()
	out := map[string]string{}
	filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		data, _ := os.ReadFile(p)
		out[filepath.ToSlash(rel)] = string(data)
		return nil
	})
	return out
}

func TestResolveDefault(t *testing.T) {
	def := t.TempDir()
	got, err := Resolve(def)
	if err != nil || got != def {
		t.Fatalf("Resolve = %q, %v; want the default folder", got, err)
	}
}

func TestResolveMovedFolder(t *testing.T) {
	def, moved := t.TempDir(), t.TempDir()
	if err := WriteJSON(filepath.Join(def, LocationFile), Location{Dir: moved}); err != nil {
		t.Fatal(err)
	}
	if got, err := Resolve(def); err != nil || got != moved {
		t.Fatalf("Resolve = %q, %v; want %q", got, err, moved)
	}
	// Its drive gone: an error that says what to do, not an empty library.
	os.RemoveAll(moved)
	if _, err := Resolve(def); err == nil || !strings.Contains(err.Error(), "not available") {
		t.Fatalf("a missing data folder: %v", err)
	}
}

func TestCopyTreeAndSizes(t *testing.T) {
	src, dst := t.TempDir(), filepath.Join(t.TempDir(), "VoxtNew")
	write(t, src, map[string]string{
		"audio/a.wav": "RIFF a", "audio/b.wav": "RIFF bb",
		sqlitestore.FileName: "db", sqlitestore.FileName + "-wal": "wal",
		LocationFile: "{}", "notes.txt": "n",
	})
	if n, err := TreeSize(context.Background(), src, nil); err != nil || n != int64(len("RIFF a")+len("RIFF bb")+2+3+2+1) {
		t.Fatalf("TreeSize = %d, %v", n, err)
	}
	skip := func(rel string) bool { return SkipData(rel, true) }
	var last [2]int64
	if err := CopyTree(context.Background(), src, dst, skip, func(done, total int64) { last = [2]int64{done, total} }); err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"audio/a.wav": "RIFF a", "audio/b.wav": "RIFF bb", "notes.txt": "n"}
	if got := read(t, dst); len(got) != len(want) || got["audio/b.wav"] != "RIFF bb" || got["notes.txt"] != "n" {
		t.Fatalf("copied %v, want %v (no database, no location files)", got, want)
	}
	if last[0] != last[1] || last[1] != int64(len("RIFF a")+len("RIFF bb")+1) {
		t.Errorf("progress ended at %d of %d", last[0], last[1])
	}
}

func TestCheckMoveTarget(t *testing.T) {
	from := t.TempDir()
	empty := t.TempDir()
	full := t.TempDir()
	write(t, full, map[string]string{"x.txt": "x"})
	withLocation := t.TempDir()
	write(t, withLocation, map[string]string{LocationFile: "{}", ".DS_Store": ""})
	for _, c := range []struct {
		to string
		ok bool
	}{
		{from, false}, {filepath.Join(from, "inside"), false}, {full, false},
		{empty, true}, {withLocation, true}, {filepath.Join(empty, "new"), true},
	} {
		if err := CheckMoveTarget(from, c.to); (err == nil) != c.ok {
			t.Errorf("CheckMoveTarget(%s) = %v, want ok %v", c.to, err, c.ok)
		}
	}
}

func TestRemoveExcept(t *testing.T) {
	dir := t.TempDir()
	write(t, dir, map[string]string{LocationFile: "{}", MoveFile: "{}", ResetFile: "{}", "a/b.wav": "b", "c.txt": "c"})
	if err := RemoveExcept(dir); err != nil {
		t.Fatal(err)
	}
	got := read(t, dir)
	if len(got) != 3 || got[LocationFile] == "" || got[MoveFile] == "" || got[ResetFile] == "" {
		t.Fatalf("left %v; want only the location files", got)
	}
}

func TestEraseEverything(t *testing.T) {
	def, moved, previous, logs := t.TempDir(), t.TempDir(), t.TempDir(), t.TempDir()
	write(t, def, map[string]string{"old.txt": "o"})
	write(t, moved, map[string]string{"audio/a.wav": "a"})
	write(t, previous, map[string]string{"left.txt": "l"})
	write(t, logs, map[string]string{"voxtnew.log": "log"})
	if err := WriteJSON(filepath.Join(def, LocationFile), Location{Dir: moved, Previous: previous}); err != nil {
		t.Fatal(err)
	}
	if err := WriteJSON(filepath.Join(def, ResetFile), PendingReset{Everything: true, Also: []string{logs}}); err != nil {
		t.Fatal(err)
	}
	got, err := Resolve(def)
	if err != nil || got != def {
		t.Fatalf("Resolve after an erase = %q, %v; want the default folder", got, err)
	}
	if left := read(t, def); len(left) != 0 {
		t.Errorf("the default folder keeps %v", left)
	}
	for _, d := range []string{moved, previous, logs} {
		if _, err := os.Stat(d); !os.IsNotExist(err) {
			t.Errorf("%s is still there", d)
		}
	}
}

// openDB makes a library database in dir, as the app leaves it closed.
func openDB(t *testing.T, dir string) {
	t.Helper()
	st, err := sqlitestore.Open(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestFinishMove(t *testing.T) {
	def := t.TempDir()
	to := filepath.Join(t.TempDir(), "VoxtNew")
	openDB(t, def)
	write(t, def, map[string]string{"audio/a.wav": "a"})
	// While the app runs: everything but the database is copied, then the
	// move is recorded; a file written after the copy comes at the start.
	skip := func(rel string) bool { return SkipData(rel, true) }
	if err := CopyTree(context.Background(), def, to, skip, nil); err != nil {
		t.Fatal(err)
	}
	write(t, def, map[string]string{"audio/b.wav": "b"})
	if err := WriteJSON(filepath.Join(def, MoveFile), PendingMove{From: def, To: to}); err != nil {
		t.Fatal(err)
	}
	got, err := Resolve(def)
	if err != nil || got != to {
		t.Fatalf("Resolve after a move = %q, %v; want %q", got, err, to)
	}
	files := read(t, to)
	if files["audio/a.wav"] != "a" || files["audio/b.wav"] != "b" {
		t.Errorf("moved %v", files)
	}
	if _, err := sqlitestore.CheckFile(context.Background(), filepath.Join(to, sqlitestore.FileName)); err != nil {
		t.Errorf("the moved database: %v", err)
	}
	loc := ReadLocation(def)
	if loc.Dir != to || loc.Previous != def || loc.MovedAt.IsZero() || loc.Failed != "" {
		t.Errorf("location.json = %+v", loc)
	}
	if _, err := os.Stat(filepath.Join(def, MoveFile)); !os.IsNotExist(err) {
		t.Error("move.json is still there")
	}
}

func TestFinishMoveFails(t *testing.T) {
	def := t.TempDir()
	to := filepath.Join(t.TempDir(), "VoxtNew")
	openDB(t, def)
	// The source of the move is gone: the copy is removed, the default
	// folder kept, and location.json says why.
	if err := WriteJSON(filepath.Join(def, MoveFile), PendingMove{From: filepath.Join(def, "gone"), To: to}); err != nil {
		t.Fatal(err)
	}
	got, err := Resolve(def)
	if err != nil || got != def {
		t.Fatalf("Resolve after a failed move = %q, %v; want the default folder", got, err)
	}
	if loc := ReadLocation(def); loc.Dir != "" || loc.Failed == "" {
		t.Errorf("location.json = %+v; want no folder and the failure", loc)
	}
	if _, err := os.Stat(to); !os.IsNotExist(err) {
		t.Error("the failed copy is still there")
	}
}
