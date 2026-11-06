// Package sqlite is MyGo's SQLite plugin. SQLite's C amalgamation and a
// small C shim are compiled with Zig and loaded through purego: no cgo,
// Zig, Bun or compiler is needed at application run time.
//
// Register the plugin only when the web frontend uses @mygo-plugins/sqlite:
//
//	mygo.Use(sqlite.Plugin)
//
// The frontend uses open from @mygo-plugins/sqlite. It opens named databases
// in the app's user-data directory, executes parameterized SQL, queries
// rows and runs batches in transactions. Go and native UI apps call Open
// directly and close their own connections; no mygo.Use is needed.
package sqlite

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/egoist/mygo"
)

// Options configure the page plugin. Go's Open accepts its own path and
// OpenOptions and does not use these options.
type Options struct {
	// Directory holds the page's database files, mygo.PathUserData when
	// empty. Pages may open file names here, or ":memory:", not paths.
	Directory string
	// BusyTimeout follows OpenOptions.BusyTimeout.
	BusyTimeout time.Duration
}

// Plugin is the page plugin with default options. It loads the native
// library only when a page opens its first database.
var Plugin = New(Options{})

// New returns a plugin with options. Connections belong to their page,
// close when it navigates away or its window closes, and close on app quit.
func New(opts Options) mygo.Plugin {
	s := &service{opts: opts, dbs: map[key]*connection{}}
	return mygo.Plugin{Name: "sqlite", Service: s, Setup: func() error {
		mygo.App.OnQuit(s.shutdown)
		return nil
	}}
}

// key identifies a connection: by its page (a *mygo.Page), as a window
// may show several, those of its web views.
type key struct {
	page any
	id   string
}
type connection struct {
	db   *DB
	stop func() bool
	page context.Context
}
type service struct {
	opts   Options
	mu     sync.Mutex
	dbs    map[key]*connection
	closed bool
}

type openOptions struct {
	ReadOnly bool `json:"readOnly"`
}

func (s *service) Open(ctx context.Context, name string, opts openOptions) (string, error) {
	return s.open(ctx, mygo.CallerPage(ctx), name, opts)
}

func (s *service) Execute(ctx context.Context, id, sql string, args []wireValue) (wireResult, error) {
	db, err := s.database(mygo.CallerPage(ctx), id)
	if err != nil {
		return wireResult{}, err
	}
	params, err := parameters(args)
	if err != nil {
		return wireResult{}, err
	}
	result, err := db.Execute(ctx, sql, params...)
	return resultWire(result), err
}

func (s *service) Query(ctx context.Context, id, sql string, args []wireValue) (wireRows, error) {
	db, err := s.database(mygo.CallerPage(ctx), id)
	if err != nil {
		return wireRows{}, err
	}
	params, err := parameters(args)
	if err != nil {
		return wireRows{}, err
	}
	rows, err := db.Query(ctx, sql, params...)
	if err != nil {
		return wireRows{}, err
	}
	return rowsWire(rows), nil
}

func (s *service) Transaction(ctx context.Context, id string, statements []wireStatement) ([]wireResult, error) {
	db, err := s.database(mygo.CallerPage(ctx), id)
	if err != nil {
		return nil, err
	}
	batch := make([]Statement, len(statements))
	for i, stmt := range statements {
		args, err := parameters(stmt.Args)
		if err != nil {
			return nil, err
		}
		batch[i] = Statement{SQL: stmt.SQL, Args: args}
	}
	results, err := db.Transaction(ctx, batch)
	if err != nil {
		return nil, err
	}
	wire := make([]wireResult, len(results))
	for i, result := range results {
		wire[i] = resultWire(result)
	}
	return wire, nil
}

func (s *service) Close(ctx context.Context, id string) error {
	return s.close(key{mygo.CallerPage(ctx), id})
}

func (s *service) open(ctx context.Context, page any, name string, opts openOptions) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	path := name
	if name != ":memory:" {
		if err := databaseName(name); err != nil {
			return "", err
		}
		dir := s.opts.Directory
		if dir == "" {
			var err error
			dir, err = mygo.App.Path(mygo.PathUserData)
			if err != nil {
				return "", err
			}
		}
		if !opts.ReadOnly {
			if err := os.MkdirAll(dir, 0o700); err != nil {
				return "", err
			}
		}
		path = filepath.Join(dir, name)
		if info, err := os.Lstat(path); err == nil && info.Mode()&os.ModeSymlink != 0 {
			return "", errors.New("sqlite: database file may not be a symbolic link")
		}
	}
	db, err := Open(ctx, path, OpenOptions{ReadOnly: opts.ReadOnly, BusyTimeout: s.opts.BusyTimeout})
	if err != nil {
		return "", err
	}
	id := rand.Text()
	k := key{page, id}
	s.mu.Lock()
	if s.closed || ctx.Err() != nil {
		s.mu.Unlock()
		db.Close()
		if err := ctx.Err(); err != nil {
			return "", err
		}
		return "", ErrClosed
	}
	c := &connection{db: db, page: ctx}
	s.dbs[k] = c
	c.stop = context.AfterFunc(ctx, func() { _ = s.close(k) })
	s.mu.Unlock()
	return id, nil
}

func databaseName(name string) error {
	if name == "" || name == "." || name == ".." || !utf8.ValidString(name) ||
		strings.ContainsAny(name, "/\\\x00:<>\"|?*") || strings.TrimSpace(name) != name || strings.HasSuffix(name, ".") {
		return errors.New("sqlite: use a database file name without directories, or :memory:")
	}
	// Windows device names remain reserved even with a file extension.
	base, _, _ := strings.Cut(strings.ToUpper(name), ".")
	if base == "CON" || base == "PRN" || base == "AUX" || base == "NUL" ||
		(len(base) == 4 && (strings.HasPrefix(base, "COM") || strings.HasPrefix(base, "LPT")) && base[3] >= '1' && base[3] <= '9') {
		return errors.New("sqlite: database file name is reserved on Windows")
	}
	return nil
}

func (s *service) database(page any, id string) (*DB, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	c := s.dbs[key{page, id}]
	if c == nil || c.page.Err() != nil {
		return nil, fmt.Errorf("sqlite: database is closed or belongs to another page")
	}
	return c.db, nil
}

func (s *service) close(k key) error {
	s.mu.Lock()
	c := s.dbs[k]
	delete(s.dbs, k)
	s.mu.Unlock()
	if c == nil {
		return nil
	}
	c.stop()
	return c.db.Close()
}

func (s *service) shutdown() {
	s.mu.Lock()
	s.closed = true
	dbs := s.dbs
	s.dbs = map[key]*connection{}
	s.mu.Unlock()
	for _, c := range dbs {
		c.stop()
		_ = c.db.Close()
	}
}
