package updater

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/plugins/updater/internal/frontend"
	"github.com/egoist/mygo/plugins/updater/internal/markdown"
)

// session is a check for updates and what follows it: the update window,
// the download, the offer to relaunch. The window shows the session's
// current view, which the session keeps up to date even while the window
// is not shown, so that it can show at any point. It is the
// frontend.Session of the window, which the updater's open function
// creates.
type session struct {
	u *updater
	// ctx is canceled when the session ends, the user cancels or the
	// window closes.
	ctx       context.Context
	cancel    context.CancelFunc
	responses chan response

	// showMu is held while the window is being created.
	showMu sync.Mutex

	mu sync.Mutex
	// user is set when the user asked for the check: the window shows
	// from the start and reports that the app is up to date.
	user    bool
	shown   bool
	ended   bool
	view    view
	changed chan struct{} // closed when the view changes
	win     *mygo.Window
	release bool // the window has the size of a release view
}

func newSession(u *updater, user bool) *session {
	ctx, cancel := context.WithCancel(context.Background())
	return &session{
		u: u, user: user, ctx: ctx, cancel: cancel,
		responses: make(chan response, 1),
		changed:   make(chan struct{}),
	}
}

// User reports whether the user asked for the check.
func (s *session) User() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.user
}

func (s *session) isShown() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.shown
}

// Current returns the view and a channel closed when it changes.
func (s *session) Current() (view, <-chan struct{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.view, s.changed
}

// Done is closed when the session ends or its window closes.
func (s *session) Done() <-chan struct{} { return s.ctx.Done() }

// set changes the view, and the window's size when the view needs another.
func (s *session) set(v view) {
	s.mu.Lock()
	v.Prompt = s.view.Prompt
	if !slices.Equal(v.Buttons, s.view.Buttons) {
		v.Prompt++
	}
	s.view = v
	close(s.changed)
	s.changed = make(chan struct{})
	win := s.win
	resize := win != nil && s.release != v.Release
	if resize {
		s.release = v.Release
	}
	s.mu.Unlock()
	if resize {
		fit(win, v.Release)
	}
}

// NewWindow creates the window of the session, hidden, showing content or,
// when it is nil, a web page.
func (s *session) NewWindow(content mygo.Content) *mygo.Window {
	v, _ := s.Current()
	width, height := frontend.Size(v.Release)
	light, dark := frontend.Background()
	win := mygo.NewWindow(mygo.WindowOptions{
		Title:             s.u.text().Title,
		Width:             width,
		Height:            height,
		UseContentSize:    true,
		Hidden:            true,
		DisableResize:     true,
		DisableMaximize:   true,
		DisableFullScreen: true,
		BackgroundColor:   "light-dark(" + light + ", " + dark + ")",
		Content:           content,
	})
	// A menu of its own, empty, rather than the app's menu bar (Linux,
	// Windows).
	win.SetMenu(mygo.NewMenu(nil))
	s.mu.Lock()
	s.win = win
	s.release = s.view.Release // the view may have changed since
	resize := s.release != v.Release
	release := s.release
	s.mu.Unlock()
	if resize {
		fit(win, release)
	}
	win.OnClosed(s.cancel)
	return win
}

// fit gives the window the size of a status or release view.
func fit(win *mygo.Window, release bool) {
	win.SetContentSize(frontend.Size(release))
	win.Center()
}

// Fit gives the window the width its buttons need, when they need more
// than the usual width, and a status view the height of its content.
func (s *session) Fit(width, height int) {
	s.mu.Lock()
	win, release := s.win, s.release
	s.mu.Unlock()
	if win == nil {
		return
	}
	w, h := frontend.Size(release)
	w = max(w, min(width, 1000))
	if !release {
		h = min(max(height, 80), 600)
	}
	win.SetContentSize(w, h)
}

// Texts returns the texts of the window around its views.
func (s *session) Texts() frontend.Texts { return s.u.text().texts() }

// Icon returns the icon of the window, or nil.
func (s *session) Icon() []byte { return s.u.icon() }

// icon returns the PNG image of the update window: Options.Icon, else
// icon.png among the app's resources, or nil.
func (u *updater) icon() []byte {
	if u.opts.Icon != nil {
		return u.opts.Icon
	}
	dir, err := resourcesDir()
	if err != nil {
		return nil
	}
	png, err := os.ReadFile(filepath.Join(dir, "icon.png"))
	if err != nil {
		return nil
	}
	return png
}

// show shows the window, bringing it to the front when the user asked for
// the check.
func (s *session) show() {
	s.showMu.Lock()
	defer s.showMu.Unlock()
	s.mu.Lock()
	win, user, ended, shown := s.win, s.user, s.ended, s.shown
	s.shown = shown || !ended
	s.mu.Unlock()
	switch {
	case ended:
	case !shown:
		s.u.open(s)
	case user && win != nil:
		win.Show()
		win.Focus()
	}
}

// promote makes a check running in the background one the user asked for.
func (s *session) promote() {
	s.mu.Lock()
	s.user = true
	s.mu.Unlock()
	s.show()
}

// wait waits for the user to respond to the view, or the window to close.
func (s *session) wait() response {
	select {
	case r := <-s.responses:
		return r
	case <-s.ctx.Done():
		return response{action: actionClose}
	}
}

// Respond takes the user's response to the prompt of a view. Only one
// response to a prompt counts, and only with one of its buttons.
func (s *session) Respond(prompt int, a action, automaticDownloads bool) {
	s.mu.Lock()
	ok := prompt == s.view.Prompt && slices.ContainsFunc(s.view.Buttons, func(b button) bool {
		return b.Action == a && !b.Disabled
	})
	if ok {
		s.view.Prompt++
	}
	s.mu.Unlock()
	switch {
	case !ok:
	case a == actionCancel:
		s.cancel()
	default:
		select {
		case s.responses <- response{a, automaticDownloads}:
		default: // the session ended
		}
	}
}

// close ends the session and closes its window.
func (s *session) close() {
	s.cancel()
	s.showMu.Lock()
	s.mu.Lock()
	s.ended = true
	win := s.win
	s.mu.Unlock()
	s.showMu.Unlock()
	if win != nil {
		win.Close()
	}
}

// The views of the update window and their buttons, as frontend has them.
type (
	view   = frontend.View
	button = frontend.Button
	action = frontend.Action
)

const (
	actionOK       = frontend.OK
	actionCancel   = frontend.Cancel
	actionSkip     = frontend.Skip
	actionLater    = frontend.Later
	actionInstall  = frontend.Install
	actionRelaunch = frontend.Relaunch
	actionClose    = frontend.Close
)

type response struct {
	action             action
	automaticDownloads bool
}

func (t *text) ok() []button {
	return []button{{Action: actionOK, Label: t.OK, Default: true, Cancel: true}}
}

func (t *text) cancel(disabled bool) []button {
	return []button{{Action: actionCancel, Label: t.Cancel, Cancel: true, Disabled: disabled}}
}

func (t *text) checkingView() view {
	return view{Title: t.Checking, Bar: true, Progress: -1, Buttons: t.cancel(false)}
}

func (t *text) upToDateView() view {
	return view{
		Title:   t.UpToDate,
		Message: fmt.Sprintf(t.UpToDateMessage, mygo.App.Name(), appVersion()),
		Buttons: t.ok(),
	}
}

func (t *text) unavailableView() view {
	msg := fmt.Sprintf(t.UnavailableMessage, mygo.App.Name())
	if mygo.IsDev() {
		msg = t.DevelopmentBuild
	}
	return view{Title: t.Unavailable, Message: msg, Buttons: t.ok()}
}

func (t *text) errorView(msg string, err error) view {
	return view{Title: t.Error, Message: msg, Detail: err.Error(), Buttons: t.ok()}
}

func (t *text) availableView(r *release, automaticDownloads bool) view {
	name := mygo.App.Name()
	notes := markdown.Parse(r.notes)
	return view{
		Release:  len(notes) > 0,
		Title:    fmt.Sprintf(t.Available, name),
		Message:  fmt.Sprintf(t.AvailableMessage, name, r.version, appVersion()),
		Notes:    notes,
		Checkbox: true,
		Checked:  automaticDownloads,
		Buttons: []button{
			{Action: actionSkip, Label: t.Skip, Aside: true},
			{Action: actionLater, Label: t.RemindLater, Cancel: true},
			{Action: actionInstall, Label: t.Install, Default: true},
		},
	}
}

func (t *text) downloadingView(downloaded, total int64) view {
	v := view{Title: t.Downloading, Bar: true, Progress: -1, Buttons: t.cancel(false)}
	if total > 0 {
		v.Progress = float64(downloaded) / float64(total)
		v.Message = fmt.Sprintf(t.Progress, t.megabytes(downloaded), t.megabytes(total))
	}
	return v
}

func (t *text) installingView() view {
	return view{Title: t.Installing, Bar: true, Progress: -1, Buttons: t.cancel(true)}
}

func (t *text) readyView(r *release) view {
	return view{
		Title:   t.Ready,
		Message: fmt.Sprintf(t.ReadyMessage, mygo.App.Name(), r.version),
		Buttons: []button{
			{Action: actionLater, Label: t.Later, Cancel: true},
			{Action: actionRelaunch, Label: t.Relaunch, Default: true},
		},
	}
}
