//go:build darwin

package darwin

import (
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/ebitengine/purego/objc"
)

// Starting at login: on macOS 13 and later the app is a login item of the
// user through SMAppService (System Settings > General > Login Items); on
// macOS 12 a launch agent opens it. Both need an app bundle.

// SMAppService statuses.
const (
	smNotRegistered    = 0
	smEnabled          = 1
	smRequiresApproval = 2
)

// mainAppService returns the SMAppService of the app, or 0 before macOS 13.
func mainAppService() id {
	c := id(objc.GetClass("SMAppService"))
	if c == 0 {
		return 0
	}
	return send(c, "mainAppService")
}

func bundlePath() (string, error) {
	path := goString(send(send(class("NSBundle"), "mainBundle"), "bundlePath"))
	if !strings.HasSuffix(path, ".app") {
		return "", errors.New("mygo: starting at login needs an app bundle; run the app with mygo dev or build it with mygo build")
	}
	return path, nil
}

func (a appController) SetOpenAtLogin(open bool, appID, name, arg string) error {
	var err error
	withPool(func() {
		var app string
		if app, err = bundlePath(); err != nil {
			return
		}
		svc := mainAppService()
		if svc == 0 {
			err = setLaunchAgent(open, appID, app, arg)
			return
		}
		if !open && sendInt(svc, "status") == smNotRegistered {
			return
		}
		selector := "unregisterAndReturnError:"
		if open {
			selector = "registerAndReturnError:"
		}
		var nserr id
		if !sendBool(svc, selector, uintptr(unsafe.Pointer(&nserr))) {
			err = fmt.Errorf("mygo: %s", goString(send(nserr, "localizedDescription")))
			return
		}
		if open && sendInt(svc, "status") == smRequiresApproval {
			err = fmt.Errorf("mygo: allow %s to open at login in System Settings > General > Login Items", name)
		}
	})
	return err
}

func (a appController) OpenAtLogin(appID, name, arg string) bool {
	ok := false
	withPool(func() {
		if _, err := bundlePath(); err != nil {
			return
		}
		if svc := mainAppService(); svc != 0 {
			ok = sendInt(svc, "status") == smEnabled
			return
		}
		_, err := os.Stat(launchAgentPath(appID))
		ok = err == nil
	})
	return ok
}

// OpenedAtLogin reports whether the launch Apple Event said the app was
// started as a login item, recorded in applicationWillFinishLaunching:.
func (a appController) OpenedAtLogin() bool { return a.b.openedAtLogin }

// launchedAsLoginItem reports whether the Apple Event being handled, the
// one that launched the app, marks it as started at login.
func launchedAsLoginItem() bool {
	const keyAEPropData, keyAELaunchedAsLogInItem = 'p'<<24 | 'r'<<16 | 'd'<<8 | 't', 'l'<<24 | 'g'<<16 | 'i'<<8 | 't'
	ev := send(send(class("NSAppleEventManager"), "sharedAppleEventManager"), "currentAppleEvent")
	if ev == 0 {
		return false
	}
	prop := send(ev, "paramDescriptorForKeyword:", keyAEPropData)
	return prop != 0 && uint32(send(prop, "enumCodeValue")) == keyAELaunchedAsLogInItem
}

func launchAgentPath(appID string) string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "Library", "LaunchAgents", appID+".plist")
}

// setLaunchAgent adds or removes a launch agent that opens the app at
// login.
func setLaunchAgent(open bool, appID, app, arg string) error {
	path := launchAgentPath(appID)
	if !open {
		if err := os.Remove(path); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return err
		}
		return nil
	}
	esc := func(s string) string {
		var b bytes.Buffer
		_ = xml.EscapeText(&b, []byte(s))
		return b.String()
	}
	plist := `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>` + esc(appID) + `</string>
	<key>ProgramArguments</key>
	<array>
		<string>/usr/bin/open</string>
		<string>-a</string>
		<string>` + esc(app) + `</string>
		<string>--args</string>
		<string>` + esc(arg) + `</string>
	</array>
	<key>RunAtLoad</key>
	<true/>
</dict>
</plist>
`
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(plist), 0o644)
}
