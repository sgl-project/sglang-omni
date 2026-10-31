// Package sqlitestore implements store.Store on SQLite, through the MyGo
// SQLite plugin (github.com/egoist/mygo/plugins/sqlite: purego, no cgo,
// FTS5 and JSON1 built in).
//
// Layout of a store directory (the app's mygo.PathUserData, unless the
// user moved it):
//
//	voxtnew.db             the database (WAL journal)
//	voxtnew.db.bak-<v>     copy taken before migrating from version v
//	audio/                  recordings kept with history entries (store.History.AudioDir)
//
// The schema lives in migrations/NNN_*.sql; PRAGMA user_version records
// the last one applied. Open migrates in one transaction per migration,
// after backing the file up, and refuses a database from a newer app.
//
// The plugin serializes operations on its one connection, and runs a
// group of writes atomically only through DB.Transaction, which takes a
// fixed list of statements: repositories therefore write each change as
// one batch and keep derived data (full-text indexes, job events) in
// triggers rather than read-modify-write code.
package sqlitestore

import (
	"context"
	"embed"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/egoist/mygo/plugins/sqlite"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// audioDir holds the recordings kept with history entries.
const audioDir = "audio"

// FileName is the database's name in the store directory.
const FileName = "voxtnew.db"

//go:embed migrations/*.sql
var migrationFiles embed.FS

// migration is one schema step: the statements that take user_version
// from version-1 to version.
type migration struct {
	version    int
	statements []string
}

// migrations are the embedded steps in order; tests may replace them.
var migrations = mustLoadMigrations()

func mustLoadMigrations() []migration {
	entries, err := fs.ReadDir(migrationFiles, "migrations")
	if err != nil {
		panic(err)
	}
	var out []migration
	for _, e := range entries {
		num, _, ok := strings.Cut(e.Name(), "_")
		v, err := strconv.Atoi(num)
		if !ok || err != nil || !strings.HasSuffix(e.Name(), ".sql") {
			panic("sqlitestore: bad migration file name " + e.Name())
		}
		data, err := migrationFiles.ReadFile("migrations/" + e.Name())
		if err != nil {
			panic(err)
		}
		out = append(out, migration{version: v, statements: splitSQL(string(data))})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].version < out[j].version })
	for i, m := range out {
		if m.version != i+1 {
			panic(fmt.Sprintf("sqlitestore: migration %d is missing", i+1))
		}
	}
	return out
}

// splitSQL splits a migration file into statements: whole-line comments
// are dropped, a statement ends at a line ending in ";", and a CREATE
// TRIGGER statement at a line "END;".
func splitSQL(src string) []string {
	var out []string
	var cur []string
	for line := range strings.Lines(src) {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "--") {
			continue
		}
		cur = append(cur, strings.TrimRight(line, "\n"))
		trigger := strings.HasPrefix(strings.ToUpper(strings.TrimSpace(cur[0])), "CREATE TRIGGER")
		if (trigger && strings.EqualFold(trimmed, "END;")) || (!trigger && strings.HasSuffix(trimmed, ";")) {
			out = append(out, strings.Join(cur, "\n"))
			cur = nil
		}
	}
	if len(cur) > 0 {
		out = append(out, strings.Join(cur, "\n"))
	}
	return out
}

// Version is the schema version this build migrates to.
func Version() int { return len(migrations) }

// Store is the SQLite store. Goroutine-safe.
type Store struct {
	db   *sqlite.DB
	dir  string
	path string

	documents  *documentRepo
	history    *historyRepo
	dictionary *dictionaryRepo
}

var _ store.Store = (*Store)(nil)

// Open opens (creating or migrating) the store in dir.
func Open(ctx context.Context, dir string) (*Store, error) {
	for _, d := range []string{dir, filepath.Join(dir, audioDir)} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return nil, err
		}
	}
	path := filepath.Join(dir, FileName)
	db, err := openDB(ctx, path)
	if err != nil {
		return nil, err
	}
	v, err := userVersion(ctx, db)
	if err != nil {
		db.Close()
		return nil, err
	}
	latest := Version()
	if v > latest {
		db.Close()
		return nil, fmt.Errorf("sqlitestore: %s has schema version %d, newer than this app's %d: %w", path, v, latest, core.ErrUnsupported)
	}
	if v < latest && v > 0 {
		// Close first, so the WAL is checkpointed into the file we copy.
		if err := db.Close(); err != nil {
			return nil, err
		}
		if err := copyFile(path, fmt.Sprintf("%s.bak-%d", path, v)); err != nil {
			return nil, fmt.Errorf("sqlitestore: backup before migrating: %w", err)
		}
		if db, err = openDB(ctx, path); err != nil {
			return nil, err
		}
	}
	for _, m := range migrations[v:] {
		stmts := make([]sqlite.Statement, 0, len(m.statements)+1)
		for _, s := range m.statements {
			stmts = append(stmts, sqlite.Statement{SQL: s})
		}
		stmts = append(stmts, sqlite.Statement{SQL: fmt.Sprintf("PRAGMA user_version = %d", m.version)})
		if _, err := db.Transaction(ctx, stmts); err != nil {
			db.Close()
			return nil, fmt.Errorf("sqlitestore: migration %d: %w", m.version, err)
		}
	}
	s := &Store{db: db, dir: dir, path: path}
	s.documents = &documentRepo{db: db}
	s.history = &historyRepo{db: db, dir: filepath.Join(dir, audioDir)}
	s.dictionary = &dictionaryRepo{db: db}
	return s, nil
}

func openDB(ctx context.Context, path string) (*sqlite.DB, error) {
	db, err := sqlite.Open(ctx, path, sqlite.OpenOptions{})
	if err != nil {
		return nil, err
	}
	for _, pragma := range []string{"PRAGMA journal_mode = WAL", "PRAGMA synchronous = NORMAL", "PRAGMA foreign_keys = ON"} {
		if _, err := db.Query(ctx, pragma); err != nil {
			db.Close()
			return nil, fmt.Errorf("sqlitestore: %s: %w", pragma, err)
		}
	}
	return db, nil
}

func userVersion(ctx context.Context, db *sqlite.DB) (int, error) {
	rows, err := db.Query(ctx, "PRAGMA user_version")
	if err != nil {
		return 0, err
	}
	if len(rows.Values) != 1 {
		return 0, errors.New("sqlitestore: no user_version")
	}
	return int(asInt(rows.Values[0][0])), nil
}

// copyFile copies src to dst through a temporary file, synced before the rename.
func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp := dst + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, err = io.Copy(out, in)
	if err == nil {
		err = out.Sync()
	}
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dst)
}

// Path is the database file.
func (s *Store) Path() string { return s.path }

// Dir is the store directory.
func (s *Store) Dir() string { return s.dir }

func (s *Store) Documents() store.Documents   { return s.documents }
func (s *Store) History() store.History       { return s.history }
func (s *Store) Dictionary() store.Dictionary { return s.dictionary }

// Close closes the database; operations in flight are cancelled.
func (s *Store) Close() error { return s.db.Close() }

// --- value helpers ----------------------------------------------------

// The plugin returns nil, int64, float64, string or []byte.

func asInt(v any) int64 {
	switch x := v.(type) {
	case int64:
		return x
	case float64:
		return int64(x)
	case string:
		n, _ := strconv.ParseInt(x, 10, 64)
		return n
	}
	return 0
}

func asFloat(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int64:
		return float64(x)
	case string:
		f, _ := strconv.ParseFloat(x, 64)
		return f
	}
	return 0
}

func asString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case []byte:
		return string(x)
	case int64:
		return strconv.FormatInt(x, 10)
	case float64:
		return strconv.FormatFloat(x, 'g', -1, 64)
	}
	return ""
}

func asBytes(v any) []byte {
	switch x := v.(type) {
	case string:
		return []byte(x)
	case []byte:
		return x
	}
	return nil
}

// ms converts a time to the schema's Unix milliseconds; the zero time is NULL.
func ms(t time.Time) any {
	if t.IsZero() {
		return nil
	}
	return t.UnixMilli()
}

// fromMS is the inverse of ms; NULL is the zero time.
func fromMS(v any) time.Time {
	if v == nil {
		return time.Time{}
	}
	return time.UnixMilli(asInt(v))
}

// wrap maps SQLite constraint errors to core sentinels.
func wrap(what string, err error) error {
	if err == nil {
		return nil
	}
	var se *sqlite.Error
	if errors.As(err, &se) {
		switch se.ExtendedCode {
		case 1555, 2067: // SQLITE_CONSTRAINT_PRIMARYKEY, _UNIQUE
			return fmt.Errorf("%s: %w: %w", what, core.ErrExists, err)
		case 787, 275, 1299, 19: // _FOREIGNKEY, _CHECK, _NOTNULL, CONSTRAINT
			return fmt.Errorf("%s: %w: %w", what, core.ErrInvalid, err)
		}
	}
	return fmt.Errorf("%s: %w", what, err)
}

// validID rejects ids that could escape a directory when used as a file name.
func validID(id string) error {
	if id == "" || id == "." || id == ".." || strings.ContainsAny(id, `/\`+"\x00") {
		return fmt.Errorf("id %q: %w", id, core.ErrInvalid)
	}
	return nil
}

// likePattern turns a search into a LIKE pattern (ESCAPE '\') matching it anywhere.
func likePattern(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`)
	return "%" + r.Replace(s) + "%"
}

// ftsPhrase quotes a search as one FTS5 phrase.
func ftsPhrase(s string) string {
	return `"` + strings.ReplaceAll(s, `"`, `""`) + `"`
}

// useFTS reports whether a search is long enough for the trigram index;
// shorter ones use LIKE on the column itself.
func useFTS(s string) bool { return len([]rune(s)) >= 3 }

// page adds LIMIT and OFFSET to a query when asked for.
func page(sql string, args []any, limit, offset int) (string, []any) {
	if limit <= 0 && offset <= 0 {
		return sql, args
	}
	if limit <= 0 {
		limit = -1
	}
	return sql + ` LIMIT ? OFFSET ?`, append(args, limit, max(offset, 0))
}
