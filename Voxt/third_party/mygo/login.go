package mygo

import (
	"os"
	"slices"
)

// loginArg is passed by the command that starts the app at login (Linux,
// Windows, and macOS 12). It is removed from os.Args before main sees it.
const loginArg = "--mygo-opened-at-login"

// openedAtLogin reports whether os.Args had loginArg.
var openedAtLogin = func() bool {
	var ok bool
	os.Args, ok = takeLoginArg(os.Args)
	return ok
}()

// takeLoginArg removes loginArg from a command line.
func takeLoginArg(args []string) ([]string, bool) {
	if i := slices.Index(args, loginArg); i > 0 {
		return slices.Delete(slices.Clone(args), i, i+1), true
	}
	return args, false
}

// SetOpenAtLogin makes the app start when the user logs in, or stops it
// from doing so. Apps usually offer it as a setting:
//
//	if err := mygo.App.SetOpenAtLogin(enabled); err != nil { … }
//
// On macOS 13 and later the app becomes a login item of the user (System
// Settings > General > Login Items), which needs an app bundle; on macOS 12
// a launch agent starts it. On Linux it gets an XDG autostart entry, on
// Windows a Run entry of the user, which Task Manager can disable.
func (a *Application) SetOpenAtLogin(open bool) error {
	id, name := appID(), a.Name()
	return onMainValue(func() error { return backend().App().SetOpenAtLogin(open, id, name, loginArg) })
}

// OpenAtLogin reports whether the app starts when the user logs in.
func (a *Application) OpenAtLogin() bool {
	id, name := appID(), a.Name()
	return onMainValue(func() bool { return backend().App().OpenAtLogin(id, name, loginArg) })
}

// WasOpenedAtLogin reports whether the system started the app because the
// user logged in, e.g. to start it in the background:
//
//	mygo.NewWindow(mygo.WindowOptions{URL: "/", Hidden: mygo.App.WasOpenedAtLogin()})
func (a *Application) WasOpenedAtLogin() bool {
	return openedAtLogin || onMainValue(func() bool { return backend().App().OpenedAtLogin() })
}

// appID identifies the app to the system: its identifier, else its name.
func appID() string {
	if info, ok := packageInfo(); ok && info.Identifier != "" {
		return info.Identifier
	}
	return App.Name()
}

// devRelaunchCode is the exit code with which a development build asks
// `mygo dev` to start it again (Relaunch).
const devRelaunchCode = 75

// startDir is the working directory the app started in.
var startDir, _ = os.Getwd()

// Relaunch quits the app like Quit, then starts it again with the same
// arguments, e.g. after changing a setting that needs a restart. Nothing
// happens when OnBeforeQuit, OnWillQuit or a window cancels the quit.
// Under `mygo dev` the build exits once it has quit and mygo dev starts it
// again.
func (a *Application) Relaunch() {
	postMain(func() {
		a.relaunch = true
		if a.prepareQuit() {
			backend().Quit()
		} else {
			a.relaunch = false
		}
	})
}
