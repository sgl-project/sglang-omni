//go:build linux && (amd64 || arm64)

package linux

import (
	"os"
	"path/filepath"
	"strings"
	"time"
)

type power struct{ b *Backend }

// Watch listens to logind for sleep, on the system bus, and to the screen
// savers of GNOME and of the freedesktop specification (KDE and others),
// which lock the screen, on the session bus.
func (p power) Watch() {
	var gerr ptr
	if system := gBusGetSync(1, 0, &gerr); system != 0 { // G_BUS_TYPE_SYSTEM
		gDBusConnectionSignalSubscribe(system, cs("org.freedesktop.login1"), cs("org.freedesktop.login1.Manager"),
			cs("PrepareForSleep"), cs("/org/freedesktop/login1"), nil, 0, cbPowerSignal, 0, 0)
	} else {
		_ = gErr(gerr)
	}
	if session, err := bus(); err == nil {
		for _, iface := range []string{"org.gnome.ScreenSaver", "org.freedesktop.ScreenSaver"} {
			gDBusConnectionSignalSubscribe(session, nil, cs(iface), cs("ActiveChanged"), nil, nil, 0, cbPowerSignal, 0, 0)
		}
	}
}

// KeepAwake asks the XDG desktop portal to inhibit suspending (and, with
// display, the session going idle), else the freedesktop screen saver to
// inhibit idleness.
func (p power) KeepAwake(display bool, reason string) func() {
	const inhibitSuspend, inhibitIdle = 4, 8
	flags := uint32(inhibitSuspend)
	if display {
		flags |= inhibitIdle
	}
	params := tuple(gVariantNewString(cs("")), gVariantNewUint32(flags), vardict(vardictEntry("reason", gVariantNewString(cs(reason)))))
	if res, err := portalCall(portalPath, "org.freedesktop.portal.Inhibit", "Inhibit", params, "(o)"); err == nil {
		child := gVariantGetChildValue(res, 0)
		handle := goStr(gVariantGetString(child, nil))
		gVariantUnref(child)
		gVariantUnref(res)
		return func() {
			if res, err := portalCall(handle, "org.freedesktop.portal.Request", "Close", 0, ""); err == nil {
				gVariantUnref(res)
			}
		}
	}
	res, err := dbusCall("org.freedesktop.ScreenSaver", "/org/freedesktop/ScreenSaver", "org.freedesktop.ScreenSaver", "Inhibit",
		tuple(gVariantNewString(cs(appName())), gVariantNewString(cs(reason))), "(u)")
	if err != nil {
		return nil
	}
	child := gVariantGetChildValue(res, 0)
	cookie := gVariantGetUint32(child)
	gVariantUnref(child)
	gVariantUnref(res)
	return func() {
		if res, err := dbusCall("org.freedesktop.ScreenSaver", "/org/freedesktop/ScreenSaver", "org.freedesktop.ScreenSaver", "UnInhibit",
			tuple(gVariantNewUint32(cookie)), ""); err == nil {
			gVariantUnref(res)
		}
	}
}

// OnBattery reports whether a battery discharges while no power supply is
// online, from /sys/class/power_supply.
func (p power) OnBattery() bool {
	read := func(dir, name string) string {
		b, _ := os.ReadFile(filepath.Join(dir, name))
		return strings.TrimSpace(string(b))
	}
	supplies, _ := filepath.Glob("/sys/class/power_supply/*")
	discharging := false
	for _, s := range supplies {
		switch read(s, "type") {
		case "Mains", "USB":
			if read(s, "online") == "1" {
				return false
			}
		case "Battery":
			if read(s, "scope") != "Device" && read(s, "status") == "Discharging" {
				discharging = true
			}
		}
	}
	return discharging
}

// IdleTime asks GNOME's idle monitor, else the freedesktop screen saver
// (KDE).
func (p power) IdleTime() time.Duration {
	if res, err := dbusCall("org.gnome.Mutter.IdleMonitor", "/org/gnome/Mutter/IdleMonitor/Core", "org.gnome.Mutter.IdleMonitor", "GetIdletime", 0, "(t)"); err == nil {
		child := gVariantGetChildValue(res, 0)
		ms := gVariantGetUint64(child)
		gVariantUnref(child)
		gVariantUnref(res)
		return time.Duration(ms) * time.Millisecond
	}
	if res, err := dbusCall("org.freedesktop.ScreenSaver", "/org/freedesktop/ScreenSaver", "org.freedesktop.ScreenSaver", "GetSessionIdleTime", 0, "(u)"); err == nil {
		child := gVariantGetChildValue(res, 0)
		s := gVariantGetUint32(child)
		gVariantUnref(child)
		gVariantUnref(res)
		return time.Duration(s) * time.Second
	}
	return 0
}
