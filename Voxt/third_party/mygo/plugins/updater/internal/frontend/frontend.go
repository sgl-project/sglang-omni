// Package frontend is what the updater shares with the windows that show
// its sessions: package updater shows them in a web page, package native
// in native UI.
package frontend

import (
	"runtime"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/plugins/updater/internal/markdown"
)

// Session is a check for updates and what follows it, as its window sees
// it.
type Session interface {
	// NewWindow creates the session's window, hidden, showing content, or
	// a web page when content is nil. The session resizes it for its
	// views and closes it when it ends.
	NewWindow(content mygo.Content) *mygo.Window
	// Current returns the view to show and a channel closed when it
	// changes.
	Current() (View, <-chan struct{})
	// Done is closed when the session ends or its window closes.
	Done() <-chan struct{}
	// Respond answers the prompt of a view with a button, and tells
	// whether updates may be installed automatically from now on. Only
	// the first response to a prompt counts.
	Respond(prompt int, a Action, automaticDownloads bool)
	// Fit gives the window the width its buttons need, when they need
	// more than the usual width, and a status view the height of its
	// content, in DIPs.
	Fit(width, height int)
	// Texts returns the texts of the window that views leave out.
	Texts() Texts
	// Icon returns the PNG image of the window, or nil.
	Icon() []byte
	// User reports whether the user asked for the check: the window shows
	// in front.
	User() bool
}

// Open creates the window of a session and shows it, at once or once it
// is ready to.
type Open func(s Session)

// Plugin returns the updater plugin with opts, an updater.Options, whose
// windows open creates. Package updater sets it.
var Plugin func(opts any, open Open) mygo.Plugin

// Show shows win, in front when the user asked for the check.
func Show(win *mygo.Window, user bool) {
	if user {
		win.Show()
		win.Focus()
	} else {
		win.ShowInactive()
	}
}

// Action is a button of the update window.
type Action string

const (
	OK       Action = "ok"
	Cancel   Action = "cancel"
	Skip     Action = "skip"
	Later    Action = "later"
	Install  Action = "install"
	Relaunch Action = "relaunch"
	// Close is the window closing, not a button.
	Close Action = "close"
)

// View is what the update window shows.
type View struct {
	// Prompt changes with the buttons: responses name the prompt they
	// answer, so that none answers a later one.
	Prompt int `json:"prompt"`
	// Release views show release notes, in a larger window.
	Release bool   `json:"release,omitzero"`
	Title   string `json:"title"`
	Message string `json:"message,omitzero"`
	// Detail is the error of a failure.
	Detail string `json:"detail,omitzero"`
	// Notes are the release notes.
	Notes []markdown.Block `json:"-"`
	// Bar shows a progress bar with Progress, between 0 and 1, or -1 when
	// the progress is unknown.
	Bar      bool    `json:"bar,omitzero"`
	Progress float64 `json:"progress,omitzero"`
	// Checkbox offers to download and install updates automatically,
	// Checked or not.
	Checkbox bool     `json:"checkbox,omitzero"`
	Checked  bool     `json:"checked,omitzero"`
	Buttons  []Button `json:"buttons"`
}

// Button is a button of a view.
type Button struct {
	Action Action `json:"action"`
	Label  string `json:"label"`
	// Default is the button of Enter, Cancel the one of Escape.
	Default bool `json:"default,omitzero"`
	Cancel  bool `json:"cancel,omitzero"`
	// Aside sets the button apart, on the left.
	Aside    bool `json:"aside,omitzero"`
	Disabled bool `json:"disabled,omitzero"`
}

// Texts are the texts of the window around its views, in the language of
// the window.
type Texts struct {
	// Title is the title of the window, ReleaseNotes the heading of the
	// notes and AutomaticDownloads the label of the checkbox.
	Title, ReleaseNotes, AutomaticDownloads string
	// Lang is the language, such as "zh-Hans", and RTL is set for those
	// written from right to left.
	Lang string
	RTL  bool
}

// The size of the update window's content, for status views and for views
// with release notes.
const (
	StatusWidth, StatusHeight   = 480, 148
	ReleaseWidth, ReleaseHeight = 620, 440
)

// Size returns the size of the window's content for a status view or a
// release view.
func Size(release bool) (width, height int) {
	if release {
		return ReleaseWidth, ReleaseHeight
	}
	return StatusWidth, StatusHeight
}

// Background returns the light and dark colors of the update window: the
// gray of dialogs on macOS, and white elsewhere, where the title bar above
// it is light.
func Background() (light, dark string) {
	if runtime.GOOS == "darwin" {
		return "#ececec", "#1e1e1e"
	}
	return "#fff", "#1e1e1e"
}
