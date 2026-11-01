package main

// The helper as the UI sees it: main makes its supervisor
// (newHelperHost, connectHelper), the window's start starts it, and the
// sidebar's cards, Home, Settings → Permissions and Diagnostics read the snapshot of
// its status kept here from helper.StatusChanged. Requests go from a
// goroutine (helperCall); notifications that belong to no request
// (hotkey.event) arrive in helperNotified and start sessions
// (session_link.go).

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appbundle"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// helperHost holds the supervisor: newHelperHost finds the executable,
// connectHelper makes the supervisor (it needs the app's bus and log).
// Tests set sup to a fake before connectHelper.
type helperHost struct {
	bin, logDir string
	env         []string
	sup         helper.Supervisor
	// stale reports that an update replaced the app on disk
	// (helper.Config.Stale, updates.go); nil in tests.
	stale func() bool
}

func newHelperHost(resources, logDir string) *helperHost {
	return &helperHost{bin: helper.Find(resources), logDir: logDir,
		stale: func() bool { return appbundle.Updated(resources, mygo.App.Version()) }}
}

// helperLink is the helper's view state.
type helperLink struct {
	host    *helperHost
	hstatus helper.Status // as StatusChanged last said
	// hotkeys is the event tap's state as hotkey.state (or the last
	// hotkey.configure) said; front the app in front as app.activated
	// last said: the target of a session started from the menu bar.
	hotkeys proto.HotkeyStatus
	front   proto.App
}

// connectHelper makes the supervisor and follows its status.
func (a *app) connectHelper() {
	h := a.host
	core.On(a.bus, func(ev helper.StatusChanged) { a.post(func() { a.helperChanged(ev.Status) }) })
	if h.sup == nil {
		version := ""
		if a.hasMenu || a.win != nil {
			version = mygo.App.Version()
		}
		h.sup = helper.New(helper.Config{
			Bin: h.bin, AppVersion: version, LogDir: h.logDir, Bus: a.bus, Env: h.env, Stale: h.stale,
			Log:      a.logs.Append,
			OnNotify: func(m proto.Message) { a.post(func() { a.helperNotified(m) }) },
		})
	}
	a.hstatus = h.sup.Status()
	a.openSessions() // session_link.go
	a.openMute()     // mute_link.go
}

// startHelper starts the helper; it runs when the window opens.
func (a *app) startHelper() {
	if err := a.host.sup.Start(context.Background()); err != nil {
		log.Printf("helper: %v", err)
	}
}

// stopHelper stops it on quit.
func (a *app) stopHelper() {
	if a.host != nil && a.host.sup != nil {
		a.host.sup.Stop()
	}
}

// restartHelper restarts it (its card, Home, the palette).
func (a *app) restartHelper() {
	sup := a.host.sup
	a.async(func() { sup.Restart() })
}

func (a *app) helperStatus() helper.Status { return a.hstatus }

// helperChanged keeps the snapshot; a helper that just became ready is
// asked what Settings → Permissions shows.
func (a *app) helperChanged(st helper.Status) {
	was := a.hstatus.State
	a.hstatus = st
	a.muteHelperChanged(was, st) // mute_link.go: a new helper undoes an earlier one's mutes first
	if st.State == helper.Ready && was != helper.Ready {
		a.recheckPermissions() // permissions.go
		a.hotkeysSent = ""     // a new helper has no shortcuts yet
		a.syncHotkeys()        // session_hotkeys.go
		a.treesSent = ""
		a.syncTrees()        // appenhance.go
		a.iconsHelperReady() // appicons.go: the icons that failed, again
		a.loadLast()
		a.audioShown() // settings_audio.go: the microphones, if their section shows
		// The app in front until the first app.activated: the target of a
		// session started from the menu bar.
		helperCall(a, proto.MethodAppFrontmost, nil, 5*time.Second, func(front proto.App, err error) {
			if err == nil && a.front.PID == 0 {
				a.front = front
				a.frontChanged(front) // appenhance.go
			}
		})
	}
	if st.State != helper.Ready && was == helper.Ready {
		a.hotkeys = proto.HotkeyStatus{} // its tap went with it
		a.syncHotkeys()
	}
	a.checkHelper() // shell_notices.go: a card while it does not run
}

// helperNotified handles a notification of no request: a shortcut
// (session_link.go), the event tap's state and the app in front.
func (a *app) helperNotified(m proto.Message) {
	switch m.Method {
	case proto.MethodHotkeyEvent:
		var e proto.HotkeyEventParams
		m.DecodeParams(&e)
		msg := "hotkey " + e.Binding + " " + e.Phase
		if e.Front != nil {
			msg += " in " + e.Front.BundleID
		}
		a.logs.Append(core.LogEntry{Time: time.Now(), Level: core.SeverityDebug, Source: helper.LogSource, Message: msg})
		a.hotkeyFired(e)
	case proto.MethodHotkeyState:
		var st proto.HotkeyStatus
		m.DecodeParams(&st)
		if st.Accessibility != a.hotkeys.Accessibility {
			a.recheckPermissions() // Accessibility was granted or taken away
		}
		a.hotkeys = st
		a.syncCarbon(a.hotkeyPlan()) // the Carbon fallback while the tap is missing
	case proto.MethodOutputChanged, proto.MethodOutputReleased:
		a.muteNotified(m) // mute_link.go
	case proto.MethodAppActivated:
		var front proto.App
		if m.DecodeParams(&front) == nil {
			a.front = front
			a.frontChanged(front) // appenhance.go
			a.closeLangMenu()     // overlay_lang.go: the user went elsewhere
		}
	}
}

// helperCall sends a request on a goroutine of its own and hands the
// result to then on the main thread (decoded into a new T), with the
// error. Without a helper that implements method, then gets the error at
// once.
func helperCall[T any](a *app, method string, params any, timeout time.Duration, then func(T, error)) {
	sup := a.host.sup
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		var out T
		raw, err := sup.Call(ctx, "", method, params, nil)
		if err == nil {
			err = json.Unmarshal(raw, &out)
		}
		a.post(func() { then(out, err) })
	})
}
