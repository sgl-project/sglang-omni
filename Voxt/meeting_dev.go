package main

import "github.com/egoist/mygo"

// meetingDevHook overrides meetingEnabled in tests (nil: use mygo.IsDev()).
var meetingDevHook func() bool

// meetingEnabled is whether meeting capture UI, menu items and shortcuts
// are wired (development builds only until C1-10).
func (a *app) meetingEnabled() bool {
	if meetingDevHook != nil {
		return meetingDevHook()
	}
	return mygo.IsDev()
}

// navPageVisible hides dev-only routes in release builds.
func (a *app) navPageVisible(path string) bool {
	if path == "/meetings" {
		return a.meetingEnabled()
	}
	return true
}
