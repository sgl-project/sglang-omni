// Package watch is the watch plugin of MyGo: it lets pages watch for
// changes to the files and directories the app names (see
// @mygo-plugins/watch). Go code and native UI call mygo.WatchFiles
// directly.
package watch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/egoist/mygo"
)

// Root is a directory, or a file, pages may watch.
type Root struct {
	// Path is the absolute path of the directory or file.
	Path string
	// Recursive lets pages watch the directories in it too, and what is
	// in them.
	Recursive bool
	// Ignore, when set, is asked about every file and directory a watch
	// of the root finds, by its path relative to the root, separated by
	// slashes, as mygo.FileWatchOptions.Ignore is.
	Ignore func(path string, isDir bool) bool
}

// Request is a watch a page asks for.
type Request struct {
	// Root is the name of the root.
	Root string
	// Path is what to watch in the root, relative to it and separated by
	// slashes: "." for the root itself.
	Path string
	// Recursive asks to watch the directories in it too.
	Recursive bool
}

// Options configures the plugin.
type Options struct {
	// Roots are the directories and files pages may watch, by the names
	// they use: lowercase letters, digits and dashes, starting with a
	// letter.
	Roots map[string]Root
	// Resolve, when set, is asked for the root of a name that is not in
	// Roots, with the call's context, so that the app can grant roots
	// while it runs, such as a folder the user chose in a dialog. It
	// returns false for names it doesn't know.
	Resolve func(ctx context.Context, name string) (Root, bool)
	// Allow, when set, is asked before each watch with the call's context
	// and the request, and can refuse it.
	Allow func(ctx context.Context, r Request) bool
	// MaxWatchesPerPage is how many watches a page may have at a time: 8
	// when zero.
	MaxWatchesPerPage int
	// MaxWatches is how many watches all pages may have at a time: 128
	// when zero.
	MaxWatches int
}

// Plugin is the plugin without roots, which lets pages watch nothing.
var Plugin = New(Options{})

type watcher interface {
	Next(context.Context) ([]mygo.FileEvent, error)
	Close() error
	Root() string
}

// active is a running watch, which counts against its page's quota while
// its call's context is not done: the page context ends with navigation,
// before the watch has stopped.
type active struct {
	page any
	ctx  context.Context
}

type service struct {
	opts       Options
	identities map[string]os.FileInfo
	factory    func(context.Context, string, mygo.FileWatchOptions) (watcher, error)
	mu         sync.Mutex
	active     map[*active]bool
}

// New returns the plugin with opts. mygo.Use panics when a root is not an
// absolute path to a directory or a file, after resolving its symlinks.
func New(opts Options) mygo.Plugin {
	roots := make(map[string]Root, len(opts.Roots))
	for k, v := range opts.Roots {
		roots[k] = v
	}
	opts.Roots = roots
	if opts.MaxWatchesPerPage == 0 {
		opts.MaxWatchesPerPage = 8
	}
	if opts.MaxWatches == 0 {
		opts.MaxWatches = 128
	}
	s := &service{opts: opts, identities: make(map[string]os.FileInfo), active: make(map[*active]bool), factory: func(ctx context.Context, path string, o mygo.FileWatchOptions) (watcher, error) {
		return mygo.WatchFiles(ctx, path, o)
	}}
	return mygo.Plugin{Name: "watch", Service: s, Setup: s.setup}
}

var alias = regexp.MustCompile(`^[a-z][a-z0-9-]{0,63}$`)

func (s *service) setup() error {
	if s.opts.MaxWatchesPerPage < 0 || s.opts.MaxWatches < 0 {
		return errors.New("watch: negative quota")
	}
	for name, root := range s.opts.Roots {
		if !alias.MatchString(name) {
			return fmt.Errorf("watch: invalid root alias %q", name)
		}
		root, info, err := canonical(root)
		if err != nil {
			return fmt.Errorf("watch: root %q: %w", name, err)
		}
		s.opts.Roots[name] = root
		s.identities[name] = info
	}
	return nil
}

// canonical resolves the symlinks of root's path, and returns what it is.
func canonical(root Root) (Root, os.FileInfo, error) {
	if !filepath.IsAbs(root.Path) {
		return root, nil, errors.New("root must be absolute")
	}
	resolved, err := filepath.EvalSymlinks(root.Path)
	if err != nil {
		return root, nil, err
	}
	root.Path = resolved
	info, err := validateRoot(root.Path)
	return root, info, err
}

// validateRoot checks that path is a file or a directory in ordinary
// directories, none a symlink or a Windows junction, and returns it.
func validateRoot(path string) (os.FileInfo, error) {
	if !filepath.IsAbs(path) {
		return nil, errors.New("root must be absolute")
	}
	path = filepath.Clean(path)
	for p := path; ; p = filepath.Dir(p) {
		info, err := os.Lstat(p)
		if err != nil {
			return nil, err
		}
		// Lstat doesn't take symlinks and junctions for directories.
		if !info.IsDir() && (p != path || !info.Mode().IsRegular()) {
			return nil, errors.New("root components must be ordinary directories")
		}
		if filepath.Dir(p) == p {
			break
		}
	}
	info, err := os.Lstat(path)
	return mygo.StatIdentity(info), err
}

// validPath reports whether p is a path relative to a root, separated by
// slashes, that stays in it.
func validPath(p string) bool {
	if p == "." {
		return true
	}
	if !fs.ValidPath(p) || strings.Contains(p, `\`) || runtime.GOOS == "windows" && strings.Contains(p, ":") {
		return false
	}
	return filepath.IsLocal(filepath.FromSlash(p))
}

// resolve checks what p names in root, through its directories without
// following symlinks, and returns its path and whether it is a directory.
func resolve(root Root, p string) (string, bool, error) {
	if p == "." {
		info, err := os.Lstat(root.Path)
		if err != nil {
			return "", false, err
		}
		return root.Path, info.IsDir(), nil
	}
	r, err := os.OpenRoot(root.Path)
	if err != nil {
		return "", false, err
	}
	defer r.Close()
	parts := strings.Split(p, "/")
	for i := range parts {
		info, err := r.Lstat(filepath.FromSlash(strings.Join(parts[:i+1], "/")))
		if err != nil {
			return "", false, err
		}
		if !info.IsDir() && (i < len(parts)-1 || !info.Mode().IsRegular()) {
			return "", false, errDenied
		}
		if i == len(parts)-1 {
			return filepath.Join(root.Path, filepath.FromSlash(p)), info.IsDir(), nil
		}
	}
	panic("unreachable")
}

// samePath reports whether a and b are the same path, which on Windows
// EvalSymlinks gives the case of the names on disk.
func samePath(a, b string) bool {
	if runtime.GOOS == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}

// request is what the page sends.
type request struct {
	Root       string `json:"root"`
	Path       string `json:"path"`
	Recursive  bool   `json:"recursive"`
	DebounceMS int    `json:"debounceMs"`
}

// part is what Go streams: "ready" once everything is watched, then
// "events", a part per batch.
type part struct {
	Type   string           `json:"type"`
	Events []mygo.FileEvent `json:"events,omitzero"`
}

// outcome is how a watch ended, the result of Watch: the code of what
// failed or stopped it, or none.
type outcome struct {
	Code    string `json:"code,omitzero"`
	Message string `json:"message,omitzero"`
}

// failure is the code of a failure, as pages see it: never a path.
type failure string

func (f failure) Error() string { return "filesystem watch " + strings.ReplaceAll(string(f), "-", " ") }

const (
	errDenied      failure = "denied"
	errLimit       failure = "limit"
	errNotFound    failure = "not-found"
	errUnsupported failure = "unsupported"
	errEncoding    failure = "path-encoding"
	errIO          failure = "io"
)

func codeFor(err error) failure {
	if f, ok := errors.AsType[failure](err); ok {
		return f
	}
	switch {
	case errors.Is(err, mygo.ErrWatchLimit):
		return errLimit
	case errors.Is(err, errors.ErrUnsupported):
		return errUnsupported
	case errors.Is(err, fs.ErrNotExist):
		return errNotFound
	default:
		return errIO
	}
}

// Watch watches what the request names for the calling page, streaming
// "ready" once everything is watched, then the batches of events, until
// the page closes parts or goes away, or the watch ends. It returns what
// failed, if anything.
func (s *service) Watch(ctx context.Context, req request, parts *mygo.Channel[part]) outcome {
	page := mygo.CallerPage(ctx)
	if page == nil {
		return outcome{Code: string(errDenied), Message: errDenied.Error()}
	}
	return s.run(ctx, page, req, parts.Send, parts.Close)
}

func (s *service) run(ctx context.Context, page any, req request, send func(part) error, unblock func()) outcome {
	err := s.watch(ctx, page, req, send, unblock)
	if err == nil || ctx.Err() != nil {
		return outcome{}
	}
	f := codeFor(err)
	return outcome{Code: string(f), Message: f.Error()}
}

// root returns the root named name, which it checks is still what it was.
func (s *service) root(ctx context.Context, name string) (Root, error) {
	if !alias.MatchString(name) {
		return Root{}, errDenied
	}
	if root, ok := s.opts.Roots[name]; ok {
		info, err := validateRoot(root.Path)
		if err != nil {
			return Root{}, err
		}
		// A directory replaced since setup is another one; a file is
		// replaced whenever an editor saves it.
		if known := s.identities[name]; known.IsDir() && !os.SameFile(info, known) {
			return Root{}, errIO
		}
		return root, nil
	}
	if s.opts.Resolve == nil {
		return Root{}, errDenied
	}
	root, ok := s.opts.Resolve(ctx, name)
	if !ok {
		return Root{}, errDenied
	}
	root, _, err := canonical(root)
	return root, err
}

func (s *service) watch(ctx context.Context, page any, req request, send func(part) error, unblock func()) error {
	if req.Path == "" {
		req.Path = "."
	}
	if !validPath(req.Path) || !alias.MatchString(req.Root) {
		return errDenied
	}
	if req.DebounceMS > 60000 {
		return errLimit
	}
	root, err := s.root(ctx, req.Root)
	if err != nil {
		return err
	}
	target, isDir, err := resolve(root, req.Path)
	if err != nil {
		return err
	}
	// A root that isn't Recursive lets pages watch itself and the files
	// in it, not the directories.
	if req.Recursive && !root.Recursive || !root.Recursive && req.Path != "." && (isDir || strings.Contains(req.Path, "/")) {
		return errDenied
	}
	if s.opts.Allow != nil && !s.opts.Allow(ctx, Request{Root: req.Root, Path: req.Path, Recursive: req.Recursive}) {
		return errDenied
	}
	if ctx.Err() != nil {
		return nil
	}
	entry := &active{page: page, ctx: ctx}
	s.mu.Lock()
	count := 0
	for a := range s.active {
		if a.page == page && a.ctx.Err() == nil {
			count++
		}
	}
	if len(s.active) >= s.opts.MaxWatches || count >= s.opts.MaxWatchesPerPage {
		s.mu.Unlock()
		return errLimit
	}
	s.active[entry] = true
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		delete(s.active, entry)
		s.mu.Unlock()
	}()
	opts := mygo.FileWatchOptions{Recursive: req.Recursive, Debounce: time.Duration(req.DebounceMS) * time.Millisecond}
	if ignore := root.Ignore; ignore != nil {
		opts.Ignore = func(p string, isDir bool) bool { return ignore(path.Join(req.Path, p), isDir) }
	}
	life, cancel := context.WithCancel(ctx)
	defer cancel()
	w, err := s.factory(life, target, opts)
	if err != nil {
		if ctx.Err() != nil {
			return nil
		}
		return err
	}
	defer w.Close()
	if !samePath(w.Root(), target) {
		return errDenied // a symlink appeared since
	}
	// The page going away unblocks a send waiting for it.
	stop := context.AfterFunc(ctx, func() { w.Close(); unblock() })
	defer stop()
	if send(part{Type: "ready"}) != nil {
		return nil
	}
	for {
		batch, err := w.Next(life)
		if err != nil {
			if errors.Is(err, io.EOF) || ctx.Err() != nil {
				return nil // closed, or the root went away
			}
			return err
		}
		for _, e := range batch {
			if !utf8.ValidString(e.Path) || !utf8.ValidString(e.OldPath) {
				return errEncoding
			}
		}
		if send(part{Type: "events", Events: batch}) != nil {
			return nil
		}
	}
}
