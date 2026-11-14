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
