package ui

import (
	"net/url"
	"path"
	"runtime"
	"slices"
	"strings"
	"time"
)

// Router keeps the history of the pages of a window, or of a part of one,
// as a browser does for a tab: the page shown, those before it that Back
// returns to, and those after it that Forward goes to again. A page is a
// path, as "/notes/42?tab=info", and View builds the one shown. Make a
// router once, in a field of the app's state, and build its pages in the
// view:
//
//	type notesApp struct{ router *ui.Router }
//
//	app := &notesApp{router: ui.NewRouter("/notes")}
//
//	func (app *notesApp) view(c *ui.Context) {
//		app.router.View(c, func(r *ui.Route) {
//			switch {
//			case r.Match("/notes"):
//				r.Title("Notes")
//				app.notes(c)
//			case r.Match("/notes/{id}"):
//				r.Title("Note")
//				app.note(c, r.Param("id"))
//			default:
//				ui.Text(c, "Not found")
//			}
//		})
//	}
//
// Push goes to a page and Back comes back, as do Cmd+[ and Cmd+] on macOS,
// Alt+Left and Alt+Right elsewhere, the back and forward buttons of a
// mouse, BackButton and ForwardButton, and Links to paths in its pages.
//
// A page can be a layout around the pages of the rest of the path, which
// Route.View builds: the layout stays, keeping its state, as they change
// inside it.
//
// A page keeps the state of its elements while it is in the history:
// going back to one finds it scrolled where it was, with what was typed,
// and the keyboard focus where it was. Going to a page moves the focus
// into it, unless the focus is outside the page that changes, as in a
// sidebar choosing the pages, where it stays: screen readers then hear
// the title of the page. Pages slide or fade in (Transition).
//
// In a browser, the first router a window shows follows the history of
// the browser's tab, in the URL's fragment, as "#/notes/42": going to a
// page adds an entry to it, the browser's back and forward buttons go
// through the router's pages, and a URL with a fragment opens the app at
// its page.
//
// The zero Router shows "/". As the rest of the app's state, it is read
// and changed on the main thread: from another goroutine, change it in
// Window.Update.
type Router struct {
	Handle

	// Transition is how pages replace each other, and Animation, if set,
	// how they move then: as a spring (Snappy, …) or a curve, rather than
	// easing out over 250 ms as they slide, 150 ms as they fade.
	Transition Transition
	Animation  *Animation

	entries []*routeEntry
	at      int
	// base numbers entries[0] among all the entries the history had, as
	// the oldest go: a browser's history refers to them by number.
	base  int
	pages uint64
	// rt is the window showing the router, which a change asks for a
	// frame.
	rt *engine
	// view is the state of View; the views of Route.View keep theirs in
	// their elements.
	view routeView
}

// routeEntry is an entry of the history: a location of a page.
type routeEntry struct {
	loc string // the path, escaped, and the query
	// pages are the pages showing the entry: View's, then those of each
	// Route.View inside, made as they are first built. An entry pushed
	// shares the pages of the layouts whose part of the path it keeps,
	// and all of them for the same path with another query.
	pages []uint64
	// layouts are, by level, how many parts of the path a page holding a
	// Route.View takes, -1 for other pages and those not built yet.
	layouts []int
	title   string
	// focus is the element that had the keyboard focus as the entry went
	// out of sight.
	focus uint64
}

// routeView is the state of a view of the pages at a level: the entry
// it showed last, and the one going away since start, toward deeper pages
// (dir 1), back up (-1) or across. focusing moves the focus into the page
// shown, and announcing tells screen readers its title.
type routeView struct {
	shown, leaving       *routeEntry
	dir                  int
	start                time.Time
	focusing, announcing bool
	// progress goes from 0 to 1 as the page comes in.
	progress channel
}

// Transition is how a Router's pages replace each other.
type Transition uint8

const (
	// TransitionAuto slides a page in from the right going deeper, as from
	// "/notes" to "/notes/42", and from the left back up; pages across, as
	// from "/notes" to "/settings", fade in. Where the desktop asks for less
	// motion, they all fade.
	TransitionAuto Transition = iota
	// TransitionFade fades pages in.
	TransitionFade
	// TransitionNone shows pages at once.
	TransitionNone
)

const (
	// maxHistory is how many entries a history holds, and keptPages how
	// many pages of it besides the one shown keep their state.
	maxHistory = 100
	keptPages  = 10
	slideTime  = 250 * time.Millisecond
	fadeTime   = 150 * time.Millisecond
)

// NewRouter returns a router showing the page at path.
func NewRouter(path string) *Router {
	r := &Router{}
	r.Replace(path)
	return r
}

// current returns the entry shown, "/" for the zero Router.
func (r *Router) current() *routeEntry {
	if len(r.entries) == 0 {
		r.entries, r.at = []*routeEntry{{loc: "/"}}, 0
	}
	return r.entries[r.at]
}

// pageAt returns the page showing entry e at a level, making it the first
// time.
func (r *Router) pageAt(e *routeEntry, level int) uint64 {
	for len(e.pages) <= level {
		r.pages++
		e.pages = append(e.pages, r.pages)
	}
	return e.pages[level]
}

// resolve returns the location of target, relative to the page shown as
// a link in a web page is: its path made absolute and clean, and its
// query.
func (r *Router) resolve(target string) (loc, p string, ok bool) {
	ref, err := url.Parse(target)
	if err != nil {
		return "", "", false
	}
	base := &url.URL{Path: "/"}
	if len(r.entries) > 0 {
		base, _ = url.Parse(r.entries[r.at].loc)
	}
	u := base.ResolveReference(ref)
	p = path.Clean("/" + u.Path)
	return (&url.URL{Path: p, RawQuery: u.RawQuery}).String(), p, true
}

// shared returns the pages of entry e that an entry of path p shares, with
// what they take of the path: all of them for the same path, else those
// of the layouts whose part of the path p keeps.
func shared(e *routeEntry, p string) ([]uint64, []int) {
	if p == pathOf(e.loc) {
		return slices.Clone(e.pages), slices.Clone(e.layouts)
	}
	parts, from := pathParts(p), pathParts(pathOf(e.loc))
	n := 0
	for n < len(e.pages) && n < len(e.layouts) {
		k := e.layouts[n]
		if k < 0 || k > len(parts) || k > len(from) || !slices.Equal(parts[:k], from[:k]) {
			break
		}
		n++
	}
	return slices.Clone(e.pages[:n]), slices.Clone(e.layouts[:n])
}

// Push goes to the page at target, after the page shown, whose later
// pages it forgets: an absolute path, as "/notes/42", or one relative to
// the page shown, as a link in a web page, as "?tab=info" or "../". Going
// to the page shown does nothing. The same path with another query is
// the same page, keeping its state, in another entry of the history, and
// layouts whose part of the path stays stay too.
func (r *Router) Push(target string) {
	cur := r.current()
	loc, p, ok := r.resolve(target)
	if !ok || loc == cur.loc {
		return
	}
	e := &routeEntry{loc: loc}
	e.pages, e.layouts = shared(cur, p)
	r.entries = append(r.entries[:r.at+1], e)
	if n := len(r.entries) - maxHistory; n > 0 {
		r.entries = r.entries[n:]
		r.base += n
	}
	r.at = len(r.entries) - 1
	r.changed()
	if browserHistory != nil {
		browserHistory.pushed(r)
	}
}

// Replace shows the page at target in place of the page shown, in its
// entry of the history, as after a page moved: Back does not return to
// the page replaced. The same path with another query is the same page,
// keeping its state, as a search page following what is typed.
func (r *Router) Replace(target string) {
	loc, p, ok := r.resolve(target)
	if !ok || len(r.entries) > 0 && r.entries[r.at].loc == loc {
		return
	}
	e := &routeEntry{loc: loc}
	if len(r.entries) == 0 {
		r.entries = []*routeEntry{e}
	} else {
		old := r.entries[r.at]
		e.pages, e.layouts = shared(old, p)
		if p == pathOf(old.loc) {
			e.title = old.title
		}
		r.entries[r.at] = e
	}
	r.changed()
	if browserHistory != nil {
		browserHistory.replaced(r)
	}
}

// Back shows the page before, if any.
func (r *Router) Back() { r.Go(-1) }

// Forward shows the page after, which Back left, if any.
func (r *Router) Forward() { r.Go(1) }

// Go moves n pages through the history: back for a negative n, forward
// for a positive one, as far as it goes.
func (r *Router) Go(n int) {
	r.current()
	at := max(0, min(r.at+n, len(r.entries)-1))
	if at != r.at {
		moved := at - r.at
		r.at = at
		r.changed()
		if browserHistory != nil {
			browserHistory.went(r, moved)
		}
	}
}

// browserHistory is the history of the browser's tab the app runs in,
// which the first router a window shows follows (router_js.go): it hears
// of the router's changes. It is nil elsewhere.
var browserHistory interface {
	// claim makes r the router following the history, unless another
	// one does.
	claim(r *Router)
	pushed(r *Router)
	replaced(r *Router)
	went(r *Router, n int)
}

// CanGoBack reports whether there is a page to go back to, and
// CanGoForward one to go forward to.
func (r *Router) CanGoBack() bool    { return r.at > 0 }
func (r *Router) CanGoForward() bool { return r.at < len(r.entries)-1 }

// Path returns the path of the page shown, as "/notes/42", without its
// query.
func (r *Router) Path() string { return pathOf(r.current().loc) }

// Location returns the path and the query of the page shown, as
// "/notes/42?tab=info", which Push and Replace take, to show the page
// again when the app starts.
func (r *Router) Location() string { return r.current().loc }

// Query returns the value of the query parameter name of the page shown,
// "" if it has none.
func (r *Router) Query(name string) string { return queryOf(r.current().loc, name) }

// Title returns the title of the page shown, as its Route set it in the
// last frame.
func (r *Router) Title() string { return r.current().title }

// changed asks for a frame showing the page now current: another pass of
// the frame being built, or a frame.
func (r *Router) changed() {
	if rt := r.rt; rt != nil {
		if rt.inFrame {
			rt.consumed = true
		} else {
			rt.requestFrame()
		}
	}
}

// isPath reports whether a link's target is a path within the app, as
// "/notes/42", "edit" or "?tab=info", rather than a URL with a scheme.
// Router returns the router whose page is being built, which a Link to a
// path without a scheme goes through, or nil outside one.
func (c *context) Router() *Router { return c.router }

func isPath(target string) bool {
	u, err := url.Parse(target)
	return err == nil && u.Scheme == "" && u.Host == "" && u.Opaque == ""
}

// pathOf returns the path of a location, unescaped.
func pathOf(loc string) string {
	if u, err := url.Parse(loc); err == nil {
		return u.Path
	}
	return loc
}

// segmentsOf returns the parts of the path of a location, each unescaped,
// so that an escaped slash stays inside its part.
func segmentsOf(loc string) []string {
	u, err := url.Parse(loc)
	if err != nil {
		return nil
	}
	parts := pathParts(u.EscapedPath())
	for i, s := range parts {
		if p, err := url.PathUnescape(s); err == nil {
			parts[i] = p
		}
	}
	return parts
}

// queryOf returns the value of a query parameter of a location.
func queryOf(loc, name string) string {
	if u, err := url.Parse(loc); err == nil {
		return u.Query().Get(name)
	}
	return ""
}

// pathParts returns the parts of a path between slashes.
func pathParts(p string) []string {
	p = strings.Trim(p, "/")
	if p == "" {
		return nil
	}
	return strings.Split(p, "/")
}

// Route is a page a Router's View builds: Match tells its path apart and
// Param reads the parts of the path the pattern matched.
type Route struct {
	router    *Router
	entry     *routeEntry
	page      *node
	pageEpoch uint64
	level     int
	// segments are the parts of the path, unescaped, from part offset of
	// the whole path for the pages of a Route.View. params are what the
	// last pattern matched took, base what the patterns of the layouts
	// around took, and rest where its {name...} started, -1 without one.
	segments     []string
	offset       int
	params, base [][2]string
	rest         int
}

// Path returns the path of the page, as "/notes/42"; in a Route.View, the
// part of the path its layout leaves, as "/fonts" of "/settings/fonts".
func (r *Route) Path() string { return "/" + strings.Join(r.segments, "/") }

// Match reports whether the page's path matches pattern, whose parts
// between slashes are either the same as the path's or wildcards: {name}
// takes any one part, and {name...}, at the end, the rest of the path,
// which may be empty. Param returns what they took. A slash at the end of
// the path or the pattern makes no difference.
//
//	case r.Match("/notes/{id}"):        // "/notes/42"
//	case r.Match("/files/{path...}"):   // "/files", "/files/docs/a.txt"
func (r *Route) Match(pattern string) bool {
	parts := pathParts(pattern)
	var params [][2]string
	for i, p := range parts {
		if len(p) < 2 || p[0] != '{' || p[len(p)-1] != '}' {
			if i >= len(r.segments) || r.segments[i] != p {
				return false
			}
			continue
		}
		name := p[1 : len(p)-1]
		if rest, ok := strings.CutSuffix(name, "..."); ok {
			if i != len(parts)-1 {
				panic("ui: {" + name + "} does not end the pattern " + pattern)
			}
			params = append(params, [2]string{rest, strings.Join(r.segments[min(i, len(r.segments)):], "/")})
			r.params, r.rest = params, min(i, len(r.segments))
			return true
		}
		if i >= len(r.segments) {
			return false
		}
		params = append(params, [2]string{name, r.segments[i]})
	}
	if len(parts) != len(r.segments) {
		return false
	}
	r.params, r.rest = params, -1
	return true
}

// Param returns what the wildcard name of the pattern Match last matched
// took, unescaped, or of a layout's around, "" if none has it.
func (r *Route) Param(name string) string {
	for _, ps := range [2][][2]string{r.params, r.base} {
		for i := len(ps) - 1; i >= 0; i-- {
			if ps[i][0] == name {
				return ps[i][1]
			}
		}
	}
	return ""
}

// Query returns the value of the query parameter name, "" if the page has
// none.
func (r *Route) Query(name string) string { return queryOf(r.entry.loc, name) }

// Title names the page: screen readers read it as the page shows, and
// BackButton and ForwardButton list it. Inside a layout, the title of the
// page inside names the entry of the history.
func (r *Route) Title(title string) { r.entry.title = title }

// Page returns the element holding the page, a column filling the router,
// for its style.
func (r *Route) corePage() *node {
	if r == nil || r.router == nil || r.router.rt == nil || r.router.rt.closed || r.pageEpoch != r.router.rt.epoch {
		return nil
	}
	return r.page
}

// View builds the pages of the rest of the path, which the {name...}
// wildcard of the pattern Match matched last took, with fn, in a column
// taking the room it is given, and returns the column: the page holding
// it is a layout, as a page of settings with a list of its sections
// around the section shown. The layout stays as the pages inside change,
// keeping its state, and they slide or fade in its place, keep their state
// in the history and take the focus as the router's pages do. Their
// patterns match the rest of the path, and Param returns the wildcards of
// the layout's pattern too:
//
//	case r.Match("/settings/{section...}"):
//		app.settingsSidebar(c)
//		r.View(c, func(r *ui.Route) {
//			switch {
//			case r.Match("/general"):
//				app.general(c)
//			case r.Match("/fonts"):
//				app.fonts(c)
//			}
//		})
//
// View panics when the pattern Match matched last has no {name...}.
func (r *Route) coreView(c *context, fn func(r *Route)) *node {
	if r.rest < 0 {
		panic("ui: Route.View with no {name...} in the pattern matched")
	}
	box := coreColumn(c).Grow(1).AlignSelf(Stretch).MinWidth(0).MinHeight(0)
	box.widget = "Router"
	e := r.entry
	for len(e.layouts) <= r.level {
		e.layouts = append(e.layouts, -1)
	}
	e.layouts[r.level] = r.offset + r.rest
	v := coreLocal(box, "view", func() routeView { return routeView{} })
	r.router.build(c, box, v, r, fn)
	return box
}

// View builds the page shown with fn, in a column taking the room it is
// given, and returns the column. As a page slides in, it builds the one
// going away too, which takes neither the pointer nor the keyboard.
func (r *Router) coreView(c *context, fn func(r *Route)) *node {
	r.rt = c.rt
	if browserHistory != nil {
		browserHistory.claim(r)
	}
	box := coreColumn(c).Grow(1).AlignSelf(Stretch).MinWidth(0).MinHeight(0)
	box.widget = "Router"
	wrapElement(box).Bind(&r.Handle)
	r.keys(c, box)
	r.build(c, box, &r.view, nil, fn)
	return box
}

// build builds the pages of view v in box: the router's, or those inside
// the page of route parent, a layout.
func (r *Router) build(c *context, box *node, v *routeView, parent *Route, fn func(*Route)) {
	// The entry to show: the router's, or the layout's, which is going
	// away while it slides out.
	cur, level := r.current(), 0
	if parent != nil {
		cur, level = parent.entry, parent.level+1
	}
	switch {
	case v.shown == nil:
		v.shown = cur // the first page: nothing changes
	case r.pageAt(v.shown, level) != r.pageAt(cur, level):
		r.show(c, box, v, parent, cur)
	default:
		v.shown = cur
	}
	progress := float32(1)
	if v.leaving != nil {
		v.progress.advance(c.now)
		if v.progress.moving || v.progress.pending {
			progress = max(0, min(float32(v.progress.value), 1))
			c.AnimationFrame()
		} else {
			v.leaving = nil
		}
	}
	box.Children(func() {
		var page *node
		switch {
		case v.leaving == nil:
			page = r.page(c, parent, cur, fn, false)
		case v.dir == 0:
			// Across, the page fades in in place of the other, which goes
			// at once: the two would blur into each other.
			page = r.page(c, parent, cur, fn, false).Opacity(progress)
		default:
			// The page going away and the one coming: the one in front in
			// the history slides over the other, which moves less, dimmed.
			box.Clip()
			bg := backdrop(box)
			old := r.page(c, parent, v.leaving, fn, true)
			page = r.page(c, parent, cur, fn, false)
			upper, lower, shown := old, page, 1-progress
			if v.dir > 0 {
				upper, lower, shown = page, old, progress
			}
			// From the end: the left right to left.
			upper.Absolute().Top(0).Fill().Background(bg).StartPercent(100*(1-shown)).Shadow(0, 0, 16, 0, RGBA(0, 0, 0, 0.15))
			lower.StartPercent(-30 * shown).Background(bg).DrawOver(func(p *Painter, rc Rect) {
				p.Fill(rc, RGBA(0, 0, 0, 0.06*shown), 0)
			})
		}
		r.keep(c, box, v, level, cur)
		if v.focusing {
			v.focusing = false
			r.focusPage(c, page, cur)
		}
	})
	if v.announcing {
		v.announcing = false
		c.Announce(cur.title)
	}
}

// keys goes back and forward for the keys that do, with the keyboard
// focus in the router, or anywhere for the first router of the window.
func (r *Router) keys(c *context, box *node) {
	first := c.routers == 0
	c.routers++
	// Alt and the arrow pointing back, as the Back button's arrow does:
	// Right right to left, as in browsers.
	left, right := box.arrows()
	back, forward := [2]shortcut{{0, KeyBack}, {Alt, left}}, [2]shortcut{{0, KeyForward}, {Alt, right}}
	if runtime.GOOS == "darwin" {
		back[1], forward[1] = shortcut{Super, KeyBracketLeft}, shortcut{Super, KeyBracketRight}
	}
	pressed := func(keys [2]shortcut) bool {
		hit := false
		for _, k := range keys {
			if box.Shortcut(k.mods, k.key) || first && c.Shortcut(k.mods, k.key) {
				hit = true
			}
		}
		return hit
	}
	if pressed(back) {
		r.Back()
	}
	if pressed(forward) {
		r.Forward()
	}
}

// show starts view v showing entry cur in place of the entry it shows: the
// transition, and where the keyboard focus goes.
func (r *Router) show(c *context, box *node, v *routeView, parent *Route, cur *routeEntry) {
	rt := c.rt
	old, level, offset := v.shown, 0, 0
	if parent != nil {
		level, offset = parent.level+1, parent.offset+parent.rest
	}
	v.shown = cur
	if rt.within(rt.focused, keyedID(box.id, r.pageAt(old, level))) {
		old.focus = rt.focused
	}
	// The focus in the page going away, or in a popup that may have
	// chosen the page, goes into the new page; outside, as in a sidebar
	// choosing the pages, it stays.
	if rt.states[rt.focused] == nil || rt.within(rt.focused, box.id) || rt.within(rt.focused, overlayID) {
		v.focusing, v.announcing = true, false
		rt.focused = 0
	} else {
		v.focusing, v.announcing = false, true
	}
	v.leaving, v.dir, v.start = nil, 0, c.now
	if r.Transition == TransitionNone || rt.instant() {
		return
	}
	v.leaving = old
	defer func() {
		v.progress = channel{at: c.now}
		headFor(&v.progress, 1, r.animation(v.dir), c.now)
	}()
	if r.Transition == TransitionAuto && !rt.preferences().ReduceMotion {
		from, to := segmentsOf(old.loc), segmentsOf(cur.loc)
		from, to = from[min(offset, len(from)):], to[min(offset, len(to)):]
		switch {
		case len(to) > len(from) && slices.Equal(to[:len(from)], from):
			v.dir = 1
		case len(from) > len(to) && slices.Equal(from[:len(to)], to):
			v.dir = -1
		}
	}
}

// animation returns how pages go in direction dir (0 fading).
func (r *Router) animation(dir int) Animation {
	if r.Animation != nil {
		return *r.Animation
	}
	if dir == 0 {
		return Animation{Duration: fadeTime, Ease: EaseOut}
	}
	return Animation{Duration: slideTime, Ease: EaseOut}
}

// page builds the page of entry e with fn, in a view of the router or
// inside the page of route parent: inert while it goes away.
func (r *Router) page(c *context, parent *Route, e *routeEntry, fn func(*Route), leaving bool) *node {
	route := &Route{router: r, entry: e, rest: -1}
	if parent != nil {
		route.level, route.offset = parent.level+1, parent.offset+parent.rest
		route.base = append(slices.Clip(parent.base), parent.params...)
	}
	segments := segmentsOf(e.loc)
	route.segments = segments[min(route.offset, len(segments)):]
	pg := coreColumn(c).Key(r.pageAt(e, route.level)).Grow(1).MinWidth(0).MinHeight(0)
	route.page = pg
	route.pageEpoch = c.rt.epoch
	pg.flags |= flagPage
	if leaving {
		pg.flags |= flagInert
	} else {
		// The focus goes to it, but Tab goes into it, past it.
		pg.flags |= flagFocusable | flagFocusTarget | flagOwnRing
		pg.role = RoleGroup
	}
	saved, inert := c.router, c.inert
	if leaving {
		c.inert = true
	} else {
		c.router = r
	}
	pg.Children(func() { fn(route) })
	c.router, c.inert = saved, inert
	pg.Label(e.title)
	return pg
}

// keep keeps the state of the pages at a level of the entries nearest the
// one shown, but those built: of a layout's view, those inside the same
// layout, which keeps them when it goes out of sight.
func (r *Router) keep(c *context, box *node, v *routeView, level int, cur *routeEntry) {
	rt := c.rt
	if rt.kept == nil {
		rt.kept = map[uint64]bool{}
	}
	n := 0
	for d := 1; n < keptPages && (r.at-d >= 0 || r.at+d < len(r.entries)); d++ {
		for _, i := range [2]int{r.at - d, r.at + d} {
			if i < 0 || i >= len(r.entries) || n == keptPages {
				continue
			}
			e := r.entries[i]
			if len(e.pages) <= level || level > 0 && e.pages[level-1] != cur.pages[level-1] {
				continue
			}
			page := e.pages[level]
			id := keyedID(box.id, page)
			if page == cur.pages[level] || v.leaving != nil && v.dir != 0 && page == r.pageAt(v.leaving, level) || rt.kept[id] {
				continue
			}
			rt.kept[id] = true
			n++
		}
	}
}

// focusPage gives the keyboard focus to the element of the page that had
// it as the page went out of sight, else to the page, unless an element
// of the page took it as it was built (AutoFocus).
func (r *Router) focusPage(c *context, page *node, cur *routeEntry) {
	rt := c.rt
	if rt.focused != 0 {
		return
	}
	if s := rt.states[cur.focus]; cur.focus != 0 && s != nil && s.seen == rt.frame && s.pass == rt.pass {
		rt.focused = cur.focus
		rt.blinkStart = rt.now()
		return
	}
	page.Focus()
}

// backdrop returns the color behind an element: the background of the
// nearest element around it that has one.
func backdrop(e *node) Color {
	for p := e.parent; p != nil; p = p.parent {
		if p.bg.A > 0 {
			return p.bg
		}
	}
	return e.c.theme.Background
}

// within reports whether element id is ancestor or inside it, in the last
// frame.
func (rt *engine) within(id, ancestor uint64) bool {
	for s, n := rt.states[id], 0; s != nil && n < 1024; s, n = rt.states[s.parent], n+1 {
		if s.id == ancestor {
			return true
		}
		if s.parent == 0 {
			break
		}
	}
	return false
}

// BackButton creates a button going back in r's history, disabled at its
// start, as in the toolbar of Finder or a browser; a right click lists
// the pages before, to go back several. ForwardButton goes forward again.
func coreBackButton(c *context, r *Router) *node { return historyButton(c, r, -1) }

// ForwardButton creates a button going forward in r's history, as
// BackButton goes back.
func coreForwardButton(c *context, r *Router) *node { return historyButton(c, r, 1) }

func historyButton(c *context, r *Router, d int) *node {
	t := c.theme
	r.current()
	label, can := "Back", r.CanGoBack()
	if d > 0 {
		label, can = "Forward", r.CanGoForward()
	}
	b := button(c, "", false).Label(label).Tooltip(label).Disabled(!can)
	b.Padding(t.Space(1.5), t.Space(2))
	b.Children(func() {
		color := t.Text
		if !can {
			color = t.TextMuted
		}
		arrow := coreBox(c).Size(t.Space(4), t.Space(4)).Shrink(0).Role(RoleNone)
		// Back points the way pages came from: right right to left.
		dir := float32(d)
		if arrow.rtl() {
			dir = -dir
		}
		arrow.Draw(func(p *Painter, rc Rect) {
			cx, cy, s := rc.X+rc.W/2, rc.Y+rc.H/2, rc.W*0.22
			dx := s * dir
			var path Path
			path.MoveTo(cx-dx/2, cy-s*1.6).LineTo(cx+dx, cy).LineTo(cx-dx/2, cy+s*1.6)
			p.StrokePath(&path, 1.6, color)
		})
	})
	if b.Clicked() {
		r.Go(d)
	}
	b.ContextMenu(func(m *Menu) {
		for i, n := r.at+d, 1; i >= 0 && i < len(r.entries) && n <= 15; i, n = i+d, n+1 {
			e := r.entries[i]
			title := e.title
			if title == "" {
				title = e.loc
			}
			if m.Item(title).Chosen() {
				r.Go(d * n)
			}
		}
	})
	return b
}
