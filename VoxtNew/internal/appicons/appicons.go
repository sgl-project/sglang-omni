// Package appicons keeps apps' icons for the lists that name apps (App
// enhancement's members, the Add apps and Browsers dialogs, the preview):
// which ones are wanted, asked for, there or missing (Book), and their
// PNGs on disk (Cache), so the helper draws each icon once a week rather
// than every time a list shows.
//
// The lists ask for the icons of the rows they show (lazily, as they
// scroll into view); the app asks the helper (app.icons) for a batch at a
// time, never for hundreds at once.
package appicons

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// State is where an app's icon is.
type State int

const (
	Unknown State = iota // not asked for
	Wanted               // a list showed the app: in the next batch
	Loading              // in a batch on its way
	Ready                // there
	Missing              // the helper found no bundle: the generic icon
	Failed               // no helper (or it failed): the generic icon until a helper is ready
)

// Key is the book's key of a bundle id: bundle ids ignore case.
func Key(bundleID string) string { return strings.ToLower(strings.TrimSpace(bundleID)) }

// Book is the state of each app's icon, and the apps wanted, in the order
// the lists asked for them. Not safe for concurrent use: the app's main
// thread owns it.
type Book struct {
	states map[string]State
	wanted []proto.AppRef
}

// State is the state of bundleID's icon.
func (b *Book) State(bundleID string) State { return b.states[Key(bundleID)] }

// Want asks for ref's icon unless it is asked for or known already.
func (b *Book) Want(ref proto.AppRef) {
	k := Key(ref.BundleID)
	if k == "" || b.states[k] != Unknown {
		return
	}
	if b.states == nil {
		b.states = map[string]State{}
	}
	b.states[k] = Wanted
	b.wanted = append(b.wanted, ref)
}

// Take returns up to n of the wanted apps, first asked first, and marks
// them Loading.
func (b *Book) Take(n int) []proto.AppRef {
	n = min(n, len(b.wanted))
	out := b.wanted[:n:n]
	b.wanted = b.wanted[n:]
	for _, r := range out {
		b.states[Key(r.BundleID)] = Loading
	}
	return out
}

// Pending is how many apps wait for a batch.
func (b *Book) Pending() int { return len(b.wanted) }

// Set records bundleID's icon state (Ready, Missing or Failed).
func (b *Book) Set(bundleID string, s State) {
	if b.states == nil {
		b.states = map[string]State{}
	}
	b.states[Key(bundleID)] = s
}

// Retry forgets the icons that failed, so the lists ask for them again
// (a helper became ready).
func (b *Book) Retry() {
	for k, s := range b.states {
		if s == Failed {
			delete(b.states, k)
		}
	}
}

// TTL is how long an icon on disk is fresh: an app's update may bring a
// new one.
const TTL = 7 * 24 * time.Hour

// Cache keeps icons as PNGs in Dir, one file a bundle id and size. A zero
// Cache (no Dir) keeps nothing.
type Cache struct {
	Dir string
	Now func() time.Time // time.Now when nil
}

// Path is the file of bundleID's icon of size pixels: the bundle id in
// lower case, other characters than letters, digits, '.', '-' and '_'
// replaced by '_'.
func (c Cache) Path(bundleID string, size int) string {
	safe := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '.', r == '-', r == '_':
			return r
		}
		return '_'
	}, Key(bundleID))
	return filepath.Join(c.Dir, safe+"-"+strconv.Itoa(size)+".png")
}

// Read returns bundleID's icon and whether it is fresh (younger than
// TTL); ok is false when there is none.
func (c Cache) Read(bundleID string, size int) (png []byte, fresh, ok bool) {
	if c.Dir == "" || Key(bundleID) == "" {
		return nil, false, false
	}
	p := c.Path(bundleID, size)
	st, err := os.Stat(p)
	if err != nil {
		return nil, false, false
	}
	data, err := os.ReadFile(p)
	if err != nil || len(data) == 0 {
		return nil, false, false
	}
	now := time.Now
	if c.Now != nil {
		now = c.Now
	}
	return data, now().Sub(st.ModTime()) < TTL, true
}

// Write keeps bundleID's icon (through a temporary file, so a reader
// never sees half of one).
func (c Cache) Write(bundleID string, size int, png []byte) error {
	if c.Dir == "" || Key(bundleID) == "" {
		return nil
	}
	if err := os.MkdirAll(c.Dir, 0o755); err != nil {
		return err
	}
	p := c.Path(bundleID, size)
	f, err := os.CreateTemp(c.Dir, ".icon-*")
	if err != nil {
		return err
	}
	_, err = f.Write(png)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = os.Rename(f.Name(), p)
	}
	if err != nil {
		os.Remove(f.Name())
	}
	return err
}

// Result is an app's icon as Fetch found it: PNG with Ready, else Missing
// or Failed.
type Result struct {
	BundleID string
	PNG      []byte
	State    State
}

// Ask asks the helper for icons (app.icons); nil when there is no helper
// that draws them.
type Ask func(refs []proto.AppRef) ([]proto.AppIcon, error)

// Fetch finds the icons of refs, of size pixels: fresh ones on disk, the
// others from ask (kept on disk), and a stale one from disk when ask
// gives none. It does I/O: run it on a goroutine.
func Fetch(c Cache, refs []proto.AppRef, size int, ask Ask) []Result {
	out := make([]Result, len(refs))
	var miss []int
	for i, r := range refs {
		out[i] = Result{BundleID: r.BundleID, State: Failed}
		png, fresh, ok := c.Read(r.BundleID, size)
		if ok {
			out[i].PNG, out[i].State = png, Ready
		}
		if !fresh {
			miss = append(miss, i)
		}
	}
	if len(miss) == 0 || ask == nil {
		return out
	}
	ask1 := make([]proto.AppRef, len(miss))
	for j, i := range miss {
		ask1[j] = refs[i]
	}
	icons, err := ask(ask1)
	if err != nil {
		return out // the stale ones, the others Failed
	}
	by := map[string]proto.AppIcon{}
	for _, ic := range icons {
		by[Key(ic.BundleID)] = ic
	}
	for _, i := range miss {
		ic, ok := by[Key(refs[i].BundleID)]
		switch {
		case !ok:
			// Not answered: as a failure.
		case len(ic.PNG) > 0:
			out[i].PNG, out[i].State = ic.PNG, Ready
			c.Write(refs[i].BundleID, size, ic.PNG)
		case out[i].State != Ready:
			out[i].State = Missing
		}
	}
	return out
}
