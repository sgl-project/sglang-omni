//go:build linux && (amd64 || arm64)

package linux

import (
	"fmt"
	"log"
	"slices"
	"strings"
	"unsafe"

	"github.com/egoist/mygo/internal/accelerator"
	"github.com/egoist/mygo/internal/platform"
)

// Wayland lets apps grab no keys: the desktop binds global shortcuts for
// the apps that ask its GlobalShortcuts portal, in a session. A session
// binds its shortcuts once, so a change replaces it with a new session
// binding them all. Desktops remember the keys the user picks for each
// shortcut id, the canonical accelerator, and may ask the user to confirm
// new shortcuts.

const shortcutsPortal = "org.freedesktop.portal.GlobalShortcuts"

// portalShortcuts is the state of the shortcuts bound through the portal.
type portalShortcuts struct {
	available bool
	shortcuts map[int]accelerator.Accelerator
	session   string // the session the shortcuts are bound in
	busy      bool   // a session is being created and bound
	stale     bool   // the shortcuts changed while busy
}

// portalShortcutsOnly makes X11 sessions bind global shortcuts through the
// portal too (TestUsePortalShortcuts).
var portalShortcutsOnly bool

var wayland struct {
	loaded       bool
	displayType  func() uintptr
	setStartupID func(display ptr, id *byte)
}

func (b *Backend) bindShortcut(id int, a accelerator.Accelerator) error {
	p := &b.portal
	if !p.available {
		if portalVersion(shortcutsPortal) == 0 {
			return fmt.Errorf("mygo: global shortcuts need an X11 session or a desktop with the GlobalShortcuts portal: %w", platform.ErrUnsupported)
		}
		p.available = true
		conn, _ := bus()
		gDBusConnectionSignalSubscribe(conn, cs(portalName), cs(shortcutsPortal), cs("Activated"), cs(portalPath), nil, 0, cbPortalSignal, 0, 0)
	}
	if p.shortcuts == nil {
		p.shortcuts = map[int]accelerator.Accelerator{}
	}
	p.shortcuts[id] = a
	if err := b.syncShortcuts(); err != nil {
		delete(p.shortcuts, id)
		return err
	}
	return nil
}

func (b *Backend) unbindShortcut(id int) {
	if _, ok := b.portal.shortcuts[id]; ok {
		delete(b.portal.shortcuts, id)
		if err := b.syncShortcuts(); err != nil {
			log.Printf("mygo: global shortcuts: %v", err)
		}
	}
}

// syncShortcuts replaces the session with one binding the shortcuts. A
// session on its way binds them once it is created.
func (b *Backend) syncShortcuts() error {
	p := &b.portal
	if p.busy {
		p.stale = true
		return nil
	}
	if p.session != "" {
		if res, err := portalCall(p.session, "org.freedesktop.portal.Session", "Close", 0, ""); err == nil {
			gVariantUnref(res)
		}
		p.session = ""
	}
	if len(p.shortcuts) == 0 {
		return nil
	}
	_, list := p.list()
	options := []ptr{
		vardictEntry("session_handle_token", gVariantNewString(cs(shortcutSessionToken()))),
		// KDE Plasma 5 binds the shortcuts given here, as drafts of the
		// portal had it; portals before 1.17 pass them on.
		vardictEntry("shortcuts", list),
	}
	err := portalRequest(shortcutsPortal, "CreateSession", nil, options, func(code uint32, results ptr) {
		session := vardictString(results, "session_handle")
		if code != 0 || session == "" {
			p.busy = false
			log.Printf("mygo: global shortcuts: the desktop portal did not create a session (response %d)", code)
			return
		}
		p.stale = false
		if len(p.shortcuts) == 0 {
			p.busy = false
			if res, err := portalCall(session, "org.freedesktop.portal.Session", "Close", 0, ""); err == nil {
				gVariantUnref(res)
			}
			return
		}
		p.session = session
		if err := b.bindShortcuts(); err != nil {
			p.busy = false
			log.Printf("mygo: global shortcuts: %v", err)
		}
	})
	if err != nil {
		if strings.Contains(err.Error(), "Error.NotAllowed") {
			return fmt.Errorf("mygo: global shortcuts on Wayland need the app installed, with its desktop entry %s: %w", desktopEntryID(), err)
		}
		return fmt.Errorf("mygo: global shortcuts: %w", err)
	}
	p.busy, p.stale = true, false
	return nil
}

// bindShortcuts binds the shortcuts in the new session.
func (b *Backend) bindShortcuts() error {
	p := &b.portal
	accs, list := p.list()
	args := []ptr{gVariantNewObjectPath(cs(p.session)), list, gVariantNewString(cs(""))} // no parent window
	return portalRequest(shortcutsPortal, "BindShortcuts", args, nil, func(code uint32, results ptr) {
		p.busy = false
		if code != 0 {
			log.Printf("mygo: global shortcuts: the desktop did not bind them (response %d)", code)
		} else {
			reportUnbound(accs, results)
		}
		if p.stale {
			if err := b.syncShortcuts(); err != nil {
				log.Printf("mygo: global shortcuts: %v", err)
			}
		}
	})
}

// list returns the shortcuts in the order they were registered, and the
// a(sa{sv}) that describes them to the portal.
func (p *portalShortcuts) list() ([]accelerator.Accelerator, ptr) {
	ids := make([]int, 0, len(p.shortcuts))
	for id := range p.shortcuts {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	accs := make([]accelerator.Accelerator, len(ids))
	items := make([]ptr, len(ids))
	for i, id := range ids {
		a := p.shortcuts[id]
		accs[i] = a
		items[i] = tuple(gVariantNewString(cs(a.String())), vardict(
			vardictEntry("description", gVariantNewString(cs(a.String()))),
			vardictEntry("preferred_trigger", gVariantNewString(cs(a.XDGTrigger()))),
		))
	}
	return accs, gVariantNewArray(0, unsafe.Pointer(&items[0]), uintptr(len(items)))
}

// reportUnbound logs the shortcuts the desktop bound to no keys: users may
// decline them, and KDE Plasma 5 binds none with portals from 1.17 on.
func reportUnbound(accs []accelerator.Accelerator, results ptr) {
	keys := map[string]string{}
	if list := gVariantLookupValue(results, cs("shortcuts"), 0); list != 0 {
		if goStr(gVariantGetTypeString(list)) == "a(sa{sv})" {
			for i := uintptr(0); i < gVariantNChildren(list); i++ {
				item := gVariantGetChildValue(list, i)
				id, props := gVariantGetChildValue(item, 0), gVariantGetChildValue(item, 1)
				keys[goStr(gVariantGetString(id, nil))] = vardictString(props, "trigger_description")
				gVariantUnref(id)
				gVariantUnref(props)
				gVariantUnref(item)
			}
		}
		gVariantUnref(list)
	}
	for _, a := range accs {
		if keys[a.String()] == "" {
			log.Printf("mygo: global shortcuts: the desktop bound no keys to %s", a)
		}
	}
}

// shortcutActivated handles the portal's Activated signal: (session
// handle, shortcut id, timestamp, options).
func (b *Backend) shortcutActivated(params ptr) {
	str := func(i uintptr) string {
		v := gVariantGetChildValue(params, i)
		defer gVariantUnref(v)
		return goStr(gVariantGetString(v, nil))
	}
	if str(0) != b.portal.session {
		return
	}
	name := str(1)
	options := gVariantGetChildValue(params, 3)
	token := vardictString(options, "activation_token")
	gVariantUnref(options)
	for id, a := range b.portal.shortcuts {
		if a.String() == name {
			withActivationToken(token, func() { b.h.HotkeyPressed(id) })
			return
		}
	}
}

// shortcutSessionToken names the session after the app: KDE names the
// shortcuts of apps without an ID after it, and keeps the keys the user
// picked for them.
func shortcutSessionToken() string {
	return "mygo_" + strings.Map(func(r rune) rune {
		if r < 0x80 && (r >= '0' && r <= '9' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z') {
			return r
		}
		return '_'
	}, strings.TrimSuffix(desktopEntryID(), ".desktop"))
}

// withActivationToken runs fn with the XDG activation token of the key
// press that caused it: the window fn shows or focuses first uses it, which
// lets it take the focus on Wayland.
func withActivationToken(token string, fn func()) {
	if !wayland.loaded {
		wayland.loaded = true
		if !bind(libGDK, &wayland.displayType, "gdk_wayland_display_get_type") ||
			!bind(libGDK, &wayland.setStartupID, "gdk_wayland_display_set_startup_notification_id") {
			wayland.setStartupID = nil
		}
	}
	display := gdkDisplayGetDefault()
	if token == "" || wayland.setStartupID == nil || !gTypeCheckInstanceIsA(display, wayland.displayType()) {
		fn()
		return
	}
	wayland.setStartupID(display, cs(token))
	defer wayland.setStartupID(display, nil)
	fn()
}
