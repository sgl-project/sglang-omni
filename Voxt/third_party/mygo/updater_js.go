//go:build js

package mygo

import (
	"context"
	"errors"
	"time"
)

// Browsers load the app's newest version with the page: they neither
// update nor relaunch it, and builds for them leave out the updater, with
// net/http and the signatures' cryptography.

// UpdaterModule installs new versions of the app. In browsers updates are
// disabled.
type UpdaterModule struct{}

// Updater installs new versions of the app.
var Updater = &UpdaterModule{}

// ErrUpdatesDisabled is returned by Check for builds without updates, as
// those for browsers.
var ErrUpdatesDisabled = errors.New("mygo: updates are not enabled for this build")

// ErrNotPublished is wrapped by the error of Check when the feed has no
// version yet.
var ErrNotPublished = errors.New("no release is published")

// Update is a newer version of the app, found by Updater.Check.
type Update struct {
	// Version of the update, and its release notes in Markdown.
	Version string
	Notes   string
	// Date is when the update was built.
	Date time.Time
}

// Enabled reports whether the app can update itself: not in browsers.
func (u *UpdaterModule) Enabled() bool { return false }

// Check returns ErrUpdatesDisabled.
func (u *UpdaterModule) Check(ctx context.Context) (*Update, error) {
	return nil, ErrUpdatesDisabled
}

// SetPrereleases does nothing: browsers do not update.
func (u *UpdaterModule) SetPrereleases(on bool) {}

// Prereleases reports false.
func (u *UpdaterModule) Prereleases() bool { return false }

// Install returns ErrUpdatesDisabled.
func (up *Update) Install(ctx context.Context, progress func(downloaded, total int64)) error {
	return ErrUpdatesDisabled
}

func relaunchNow() {}

func cleanUpdateLeftovers() {}
