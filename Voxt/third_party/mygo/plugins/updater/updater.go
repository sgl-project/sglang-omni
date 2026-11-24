// Package updater is the update window of MyGo apps, in the manner of
// Sparkle on macOS. It checks for updates in the background, once a day by
// default, and when a new version is out it shows its release notes and
// offers to install it, skip it or remind the user later. Installing
// downloads the update with a progress bar, then offers to relaunch the
// app into it.
//
//	mygo.Use(updater.Plugin)
//
// Add "Check for Updates…" to the app's menu, after About on macOS:
//
//	{Label: "My App", Submenu: []*mygo.MenuItem{
//		{Role: mygo.RoleAbout},
//		updater.MenuItem(),
//		...
//	}},
//
// It is built on mygo.Updater, so the app must be built with updates (see
// the updates guide). Builds that cannot update themselves, such as
// development builds and apps installed by a package manager, never check
// in the background, and say why when the user checks.
//
// The window speaks the user's language when the plugin has it, and
// Options.Strings changes its texts or adds languages (see Strings).
//
// The user's choices are kept in updater.json in the app's user data
// directory: whether to check automatically (AutomaticChecks), whether to
// install updates without asking (AutomaticDownloads, the checkbox of the
// update window), the version they skipped and when the app last checked.
// OnChange tells the app when they changed, for a preferences page.
package updater

import (
	"context"
	"encoding/json/v2"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/plugins/updater/internal/frontend"
)

// Options configure the plugin.
type Options struct {
	// Interval between automatic checks. Zero means a day.
	Interval time.Duration
	// DisableAutomaticChecks turns automatic checks off until
	// SetAutomaticChecks turns them on, for apps that ask the user first or
	// only check from the menu.
	DisableAutomaticChecks bool
	// Icon is the PNG image shown in the update window. Nil means icon.png
	// among the app's resources, the default icon of `mygo build`, when
	// there is one.
	Icon []byte
	// Language of the window, a language tag such as "fr" or "zh-Hant".
	// Empty means the user's (App.Locale): the plugin shows the language
	// that matches it best among its own and those of Strings, else
	// English.
	Language string
	// Strings change the texts of the window by language tag, or add
	// languages. Their empty fields keep the plugin's texts, else the
	// English ones:
	//
	//	Strings: map[string]updater.Strings{
	//		"en": {Install: "Update Now"},
	//		"sv": {Title: "Programuppdatering", ...},
	//	}
	Strings map[string]Strings
}

// Plugin is the plugin with the default options.
var Plugin = New(Options{})

// New returns the plugin with options. Use one of them only.
func New(opts Options) mygo.Plugin { return newPlugin(opts, openWindow) }

func init() {
	// Package native shows the window in native UI.
	frontend.Plugin = func(opts any, open frontend.Open) mygo.Plugin {
		return newPlugin(opts.(Options), func(s *session) { open(s) })
	}
}

// newPlugin returns the plugin whose windows open creates.
func newPlugin(opts Options, open func(*session)) mygo.Plugin {
	u := newUpdater(opts, open)
	return mygo.Plugin{
		Name:    "updater",
		Service: &service{u},
		Setup: func() error {
			if err := checkStrings(opts.Strings); err != nil {
				return err
			}
			if !active.CompareAndSwap(nil, u) {
				return errors.New("another updater plugin is used")
			}
			mygo.App.WhenReady(func() { go u.start() })
			return nil
		},
	}
}

// CheckForUpdates checks for updates as the user asked, from a menu item
// or a button: the update window shows right away, says when the app is up
// to date or the check failed, and shows updates the user skipped. It
// returns at once; while a check is under way it brings its window to the
// front.
func CheckForUpdates() {
	u := used()
	go u.begin(true)
}

// MenuItem returns a "Check for Updates…" item that calls CheckForUpdates.
func MenuItem() *mygo.MenuItem {
	return &mygo.MenuItem{
		Label: used().text().MenuItem,
		Click: func(*mygo.MenuItem, *mygo.Window) { CheckForUpdates() },
	}
}

// AutomaticChecks reports whether the app checks for updates in the
// background.
func AutomaticChecks() bool {
	u := used()
	u.load()
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.automaticChecks()
}

// SetAutomaticChecks turns checking for updates in the background on or
// off, for a preference of the app.
func SetAutomaticChecks(on bool) {
	u := used()
	u.update(func(s *state) { s.AutomaticChecks = &on })
	u.schedule()
}

// AutomaticDownloads reports whether updates found in the background are
// installed without asking: they run the next time the app starts. The
// update window offers to turn it on.
func AutomaticDownloads() bool {
	u := used()
	u.load()
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.state.AutomaticDownloads
}

// SetAutomaticDownloads turns installing updates found in the background
// without asking on or off.
func SetAutomaticDownloads(on bool) {
	used().update(func(s *state) { s.AutomaticDownloads = on })
}

// LastCheck returns when the app last checked for updates successfully, or
// the zero time.
func LastCheck() time.Time {
	u := used()
	u.load()
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.state.LastCheck
}

// OnChange calls fn on the main thread after the choices of the user or the
// time of the last check changed: a check succeeded, the user answered the
// update window (its checkbox, Skip This Version), or the app called
// SetAutomaticChecks or SetAutomaticDownloads. A preferences page that shows
// AutomaticChecks, AutomaticDownloads or LastCheck reads them again. It
// returns a function that removes fn.
func OnChange(fn func()) (off func()) { return used().onChange(fn) }

// active is the plugin that was used.
var active atomic.Pointer[updater]

func used() *updater {
	u := active.Load()
	if u == nil {
		panic("updater: the plugin is not used: call mygo.Use(updater.Plugin) before App.Run")
	}
	return u
}

// Replaced by tests.
var (
	enabled        = mygo.Updater.Enabled
	check          = checkRelease
	updaterCheck   = mygo.Updater.Check
	relaunch       = mygo.App.Relaunch
	appVersion     = mygo.App.Version
	locale         = mygo.App.Locale
	now            = time.Now
	stateDir       = func() (string, error) { return mygo.App.Path(mygo.PathUserData) }
	resourcesDir   = func() (string, error) { return mygo.App.Path(mygo.PathResources) }
	runOnMain      = mygo.RunOnMain
	firstCheckWait = 10 * time.Second
)

// release is an update that a check found.
type release struct {
	version string
	notes   string
	install func(ctx context.Context, progress func(downloaded, total int64)) error
}

func checkRelease(ctx context.Context) (*release, error) {
	up, err := updaterCheck(ctx)
	if errors.Is(err, mygo.ErrNotPublished) {
		// Nothing is published on the feed yet: there is nothing newer.
		return nil, nil
	}
	if err != nil || up == nil {
		return nil, err
	}
	return &release{version: up.Version, notes: up.Notes, install: up.Install}, nil
}

// state is what updater.json keeps.
type state struct {
	// AutomaticChecks is nil until the app or the user chose.
	AutomaticChecks    *bool     `json:"automaticChecks,omitempty"`
	AutomaticDownloads bool      `json:"automaticDownloads,omitzero"`
	SkippedVersion     string    `json:"skippedVersion,omitzero"`
	LastCheck          time.Time `json:"lastCheck,omitzero"`
}

// same reports whether s keeps what o keeps.
func (s state) same(o state) bool {
	if (s.AutomaticChecks == nil) != (o.AutomaticChecks == nil) ||
		s.AutomaticChecks != nil && *s.AutomaticChecks != *o.AutomaticChecks {
		return false
	}
	return s.AutomaticDownloads == o.AutomaticDownloads &&
		s.SkippedVersion == o.SkippedVersion &&
		s.LastCheck.Equal(o.LastCheck)
}

const stateFile = "updater.json"

type updater struct {
	opts Options
	// open creates the window of a session and shows it.
	open func(*session)

	textOnce sync.Once
	txt      *text

	loadOnce sync.Once
	mu       sync.Mutex
	file     string // "" when the state cannot be saved
	state    state
	// changes are the functions of OnChange, in the order they were added.
	changes []*func()
	// running is set once the app is ready and can update itself: only
	// then are checks scheduled.
	running bool
	timer   *time.Timer
	// failed is when the last automatic check failed, which retries sooner
	// than the interval.
	failed time.Time
	// session is the check in progress; installed the update installed
	// while the app runs, which runs at the next launch.
	session   *session
	installed *release
}

func newUpdater(opts Options, open func(*session)) *updater {
	if opts.Interval <= 0 {
		opts.Interval = 24 * time.Hour
	}
	return &updater{opts: opts, open: open}
}

// text returns the texts of the window, in the language chosen once.
func (u *updater) text() *text {
	u.textOnce.Do(func() {
		lang := u.opts.Language
		if lang == "" {
			lang = locale()
		}
		u.txt = newText(lang, u.opts.Strings)
	})
	return u.txt
}

// start schedules the automatic checks, once the app is ready.
func (u *updater) start() {
	if !enabled() {
		return
	}
	u.load()
	u.mu.Lock()
	u.running = true
	u.mu.Unlock()
	// Timers stop while the computer sleeps: check what is due on waking.
	mygo.Power.OnResume(func() { go u.schedule() })
	u.schedule()
}

// load reads the state file, once.
func (u *updater) load() {
	u.loadOnce.Do(func() {
		dir, err := stateDir()
		if err != nil {
			return
		}
		u.mu.Lock()
		defer u.mu.Unlock()
		u.file = filepath.Join(dir, stateFile)
		if data, err := os.ReadFile(u.file); err == nil {
			_ = json.Unmarshal(data, &u.state) // a damaged file is ignored
		}
	})
}

// update changes the state, saves it, and calls the functions of OnChange
// when it changed.
func (u *updater) update(fn func(*state)) {
	u.load()
	u.mu.Lock()
	before := u.state
	fn(&u.state)
	u.save()
	var changes []*func()
	if !u.state.same(before) {
		changes = slices.Clone(u.changes)
	}
	u.mu.Unlock()
	if len(changes) > 0 {
		// From a goroutine of its own: no caller waits for the main thread.
		go runOnMain(func() {
			for _, fn := range changes {
				(*fn)()
			}
		})
	}
}

// save writes the state file. u.mu is held.
func (u *updater) save() {
	if u.file == "" {
		return
	}
	data, err := json.Marshal(u.state, json.Deterministic(true))
	if err != nil {
		return
	}
	if err := os.WriteFile(u.file+".tmp", data, 0o644); err == nil {
		_ = os.Rename(u.file+".tmp", u.file)
	}
}

func (u *updater) onChange(fn func()) (off func()) {
	p := &fn
	u.mu.Lock()
	u.changes = append(u.changes, p)
	u.mu.Unlock()
	return func() {
		u.mu.Lock()
		defer u.mu.Unlock()
		u.changes = slices.DeleteFunc(u.changes, func(q *func()) bool { return q == p })
	}
}

func (u *updater) automaticChecks() bool {
	if u.state.AutomaticChecks != nil {
		return *u.state.AutomaticChecks
	}
	return !u.opts.DisableAutomaticChecks
}

// schedule arms the timer of the next automatic check.
func (u *updater) schedule() {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.timer != nil {
		u.timer.Stop()
		u.timer = nil
	}
	if !u.running || u.session != nil || u.installed != nil || !u.automaticChecks() {
		return
	}
	wait := max(nextCheck(u.state.LastCheck, u.failed, u.opts.Interval, now()).Sub(now()), firstCheckWait)
	u.timer = time.AfterFunc(wait, func() { u.begin(false) })
}

// nextCheck returns when the next automatic check is due: an interval
// after the last successful one, and when a check failed since, an hour
// after that at most.
func nextCheck(last, failed time.Time, interval time.Duration, now time.Time) time.Time {
	if last.After(now) { // the clock was set back
		last = now
	}
	due := last.Add(interval)
	if retry := failed.Add(min(interval, time.Hour)); !failed.IsZero() && retry.After(due) {
		due = retry
	}
	return due
}

// checked records the outcome of a check.
func (u *updater) checked(err error) {
	u.mu.Lock()
	if err != nil {
		u.failed = now()
		u.mu.Unlock()
		return
	}
	u.failed = time.Time{}
	u.mu.Unlock()
	t := now()
	u.update(func(s *state) { s.LastCheck = t })
}

// begin starts a check, or makes the one under way show its window when
// the user asked for it.
func (u *updater) begin(user bool) {
	u.load()
	u.mu.Lock()
	if s := u.session; s != nil {
		u.mu.Unlock()
		if user {
			s.promote()
		}
		return
	}
	if !user && u.installed != nil {
		u.mu.Unlock()
		return
	}
	s := newSession(u, user)
	u.session = s
	if u.timer != nil {
		u.timer.Stop()
		u.timer = nil
	}
	u.mu.Unlock()
	u.run(s)
}

// end closes the window of a session and schedules the next check.
func (u *updater) end(s *session) {
	u.mu.Lock()
	if u.session == s {
		u.session = nil
	}
	u.mu.Unlock()
	s.close()
	u.schedule()
}

// run checks for updates and drives the update window.
func (u *updater) run(s *session) {
	defer u.end(s)
	t := u.text()
	if !enabled() {
		s.set(t.unavailableView())
		s.show()
		s.wait()
		return
	}
	u.mu.Lock()
	installed := u.installed
	u.mu.Unlock()
	if installed != nil {
		u.offerRelaunch(s, installed)
		return
	}

	s.set(t.checkingView())
	if s.User() {
		s.show()
	}
	r, err := check(s.ctx)
	if s.ctx.Err() != nil {
		return // canceled
	}
	u.checked(err)
	switch {
	case err != nil:
		if s.User() {
			s.set(t.errorView(t.CheckError, err))
			s.wait()
		}
		return
	case r == nil:
		if s.User() {
			s.set(t.upToDateView())
			s.wait()
		}
		return
	}

	u.mu.Lock()
	skipped := u.state.SkippedVersion == r.version
	automatic := u.state.AutomaticDownloads
	u.mu.Unlock()
	if !s.User() {
		if skipped {
			return
		}
		if automatic {
			u.install(s, r)
			return
		}
	}
	s.set(t.availableView(r, automatic))
	s.show()
	resp := s.wait()
	if resp.action == actionClose {
		return
	}
	u.update(func(st *state) {
		st.AutomaticDownloads = resp.automaticDownloads
		if resp.action == actionSkip {
			st.SkippedVersion = r.version
		}
	})
	if resp.action == actionInstall {
		u.install(s, r)
	}
}

// install downloads and installs r, showing its progress when the window
// shows, then offers to relaunch.
func (u *updater) install(s *session, r *release) {
	t := u.text()
	s.set(t.downloadingView(0, 0))
	err := r.install(s.ctx, func(downloaded, total int64) {
		if downloaded < total {
			s.set(t.downloadingView(downloaded, total))
		} else {
			s.set(t.installingView())
		}
	})
	if s.ctx.Err() != nil {
		return // canceled
	}
	if err != nil {
		if s.User() || s.isShown() {
			s.set(t.errorView(t.InstallError, err))
			s.show()
			s.wait()
		}
		return
	}
	u.mu.Lock()
	u.installed = r
	u.mu.Unlock()
	if s.User() || s.isShown() {
		u.offerRelaunch(s, r)
	}
}

func (u *updater) offerRelaunch(s *session, r *release) {
	s.set(u.text().readyView(r))
	s.show()
	if s.wait().action == actionRelaunch {
		relaunch()
	}
}
