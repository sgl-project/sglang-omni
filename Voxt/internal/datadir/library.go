package datadir

// Taking the library to another Mac: Settings → Storage exports the
// database (settings, history, meetings, dictionary, snippets) as one
// file, and imports one by replacing the database at the next start,
// while it is closed, keeping the replaced one beside it. Recordings,
// models and the Keychain's API keys stay on each Mac.

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

const (
	// ImportFile in the data folder asks the next start to swap in the
	// staged library (importedDB).
	ImportFile = "import.json"
	importedDB = "import.voxtlibrary"
	// BeforeImport is the replaced database, kept in the data folder.
	BeforeImport = sqlitestore.FileName + ".before-import"
	// exportedRoot stands for the data folder in an exported file: the
	// import rewrites it to the folder the file lands in.
	exportedRoot = "/voxt-exported-library"
)

// sqlite3 is macOS's: its .backup copies a consistent snapshot while the
// app keeps writing (the sqlite plugin refuses VACUUM INTO).
var sqlite3 = "/usr/bin/sqlite3"

// ExportLibrary writes the database of dataDir to dst as one file.
func ExportLibrary(ctx context.Context, dataDir, dst string) error {
	tmp, err := os.MkdirTemp("", "voxt-export-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	snap := filepath.Join(tmp, "library.db")
	out, err := exec.CommandContext(ctx, sqlite3, filepath.Join(dataDir, sqlitestore.FileName), ".backup '"+snap+"'").CombinedOutput()
	if err != nil {
		return fmt.Errorf("copying the library: %v: %s", err, bytes.TrimSpace(out))
	}
	if _, err := sqlitestore.RewritePaths(ctx, snap, dataDir, exportedRoot); err != nil {
		return err
	}
	if _, err := sqlitestore.CheckFile(ctx, snap); err != nil {
		return err
	}
	_, err = copyVerified(snap, dst)
	return err
}

// StageImport checks the library file src and leaves a copy in dataDir
// for the next start to swap in.
func StageImport(ctx context.Context, dataDir, src string) error {
	staged := filepath.Join(dataDir, importedDB)
	if _, err := copyVerified(src, staged); err != nil {
		os.Remove(staged)
		return err
	}
	v, err := sqlitestore.CheckFile(ctx, staged)
	if err == nil && (v < 1 || v > sqlitestore.Version()) {
		err = fmt.Errorf("not a Voxt library this version can read (schema %d, this Voxt reads 1 to %d)", v, sqlitestore.Version())
	}
	if err == nil {
		err = WriteJSON(filepath.Join(dataDir, ImportFile), struct{}{})
	}
	if err != nil {
		removeDB(staged)
		return err
	}
	return nil
}

// finishImport swaps the staged library in for dir's database, keeping
// the database it replaces as BeforeImport. Any failure keeps the
// current database.
func finishImport(ctx context.Context, dir string) {
	os.Remove(filepath.Join(dir, ImportFile))
	staged := filepath.Join(dir, importedDB)
	defer removeDB(staged)
	fail := func(err error) {
		log.Printf("data folder: the library import failed, keeping the current one: %v", err)
	}
	if _, err := sqlitestore.RewritePaths(ctx, staged, exportedRoot, dir); err != nil {
		fail(err)
		return
	}
	if _, err := sqlitestore.CheckFile(ctx, staged); err != nil {
		fail(err)
		return
	}
	db, before := filepath.Join(dir, sqlitestore.FileName), filepath.Join(dir, BeforeImport)
	removeDB(before)
	if err := renameDB(db, before); err != nil {
		renameDB(before, db)
		fail(err)
		return
	}
	if err := renameDB(staged, db); err != nil {
		removeDB(db)
		renameDB(before, db)
		fail(err)
		return
	}
	log.Printf("data folder: imported a library into %s, the replaced one is %s", dir, before)
}

// renameDB moves a closed database with its write-ahead log; a missing
// log is fine.
func renameDB(from, to string) error {
	os.Remove(from + "-shm")
	if err := os.Rename(from+"-wal", to+"-wal"); err != nil && !os.IsNotExist(err) {
		return err
	}
	return os.Rename(from, to)
}

func removeDB(path string) {
	for _, suffix := range []string{"", "-wal", "-shm"} {
		os.Remove(path + suffix)
	}
}
