package updater

import (
	"context"
	"crypto/rand"
	_ "embed"
	"encoding/base64"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"html"
	"strings"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/plugins/updater/internal/frontend"
	"github.com/egoist/mygo/plugins/updater/internal/markdown"
)

//go:embed page.html
var pageHTML string

// openWindow creates the update window of s as a web page. It shows once
// its page is ready, in front when the user asked for the check.
func openWindow(s *session) {
	win := s.NewWindow(nil)
	win.Page().OnWillNavigate(func(e *mygo.NavigateEvent) {
		// Links of the release notes open in the browser.
		e.PreventDefault()
		if e.UserInitiated && markdown.SafeURL(e.URL) {
			go mygo.Shell.OpenExternal(e.URL)
		}
	})
	user := s.User()
	win.OnReadyToShow(func() { frontend.Show(win, user) })
	v, _ := s.Current()
	win.Page().LoadHTML(page(s.Texts(), s.u.iconURL(), v), "")
}

// pageView is a view as the page gets it, with the notes in HTML.
type pageView struct {
	view
	Notes string `json:"notes,omitzero"`
}

func newPageView(v view) pageView { return pageView{v, markdown.HTML(v.Notes)} }

// page returns the page of the update window with the texts t, showing v
// until the page watches the session.
func page(t frontend.Texts, icon string, v view) string {
	nonce := make([]byte, 16)
	rand.Read(nonce)
	initial, _ := json.Marshal(newPageView(v), jsontext.EscapeForHTML(true), jsontext.EscapeForJS(true))
	dir := "ltr"
	if t.RTL {
		dir = "rtl"
	}
	light, dark := frontend.Background()
	return strings.NewReplacer(
		"{{lang}}", html.EscapeString(t.Lang),
		"{{dir}}", dir,
		"{{background}}", light,
		"{{darkBackground}}", dark,
		"{{title}}", html.EscapeString(t.Title),
		"{{releaseNotes}}", html.EscapeString(t.ReleaseNotes),
		"{{automaticDownloads}}", html.EscapeString(t.AutomaticDownloads),
		"{{nonce}}", base64.StdEncoding.EncodeToString(nonce),
		"{{icon}}", icon,
		"{{view}}", string(initial),
	).Replace(pageHTML)
}

// iconURL returns the icon of the update window as a data URL, or "".
func (u *updater) iconURL() string {
	png := u.icon()
	if png == nil {
		return ""
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(png)
}

// service is what the update window's page calls.
type service struct{ u *updater }

var errNotUpdateWindow = errors.New("only the update window may call the updater")

// session returns the session whose window made the call.
func (sv *service) session(ctx context.Context) (*session, error) {
	win := mygo.CallerWindow(ctx)
	sv.u.mu.Lock()
	s := sv.u.session
	sv.u.mu.Unlock()
	if s == nil || win == nil {
		return nil, errNotUpdateWindow
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.win != win {
		return nil, errNotUpdateWindow
	}
	return s, nil
}

// Watch streams the views of the session to the update window.
func (sv *service) Watch(ctx context.Context, views *mygo.Channel[pageView]) error {
	s, err := sv.session(ctx)
	if err != nil {
		return err
	}
	for {
		v, changed := s.Current()
		if err := views.Send(newPageView(v)); err != nil {
			return nil // the page went away
		}
		select {
		case <-changed:
		case <-ctx.Done():
			return nil
		}
	}
}

// Fit gives the window the width its buttons need, when they need more
// than the usual width, and a status view the height of its text, which
// both depend on the language and the platform's fonts.
func (sv *service) Fit(ctx context.Context, width, height int) error {
	s, err := sv.session(ctx)
	if err != nil {
		return err
	}
	s.Fit(width, height)
	return nil
}

// Respond answers the prompt of a view with a button, and tells whether
// updates may be installed automatically from now on.
func (sv *service) Respond(ctx context.Context, prompt int, a action, automaticDownloads bool) error {
	s, err := sv.session(ctx)
	if err != nil {
		return err
	}
	s.Respond(prompt, a, automaticDownloads)
	return nil
}
