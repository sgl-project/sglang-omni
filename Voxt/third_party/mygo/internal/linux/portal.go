//go:build linux && (amd64 || arm64)

package linux

import (
	"strconv"
	"strings"
	"unsafe"
)

// The XDG desktop portal offers what the desktop decides on: settings,
// inhibiting sleep, global shortcuts. Methods that may involve the user
// return a request object at once and answer later with its Response
// signal, which cbPortalSignal routes by the object's path.

const (
	portalName = "org.freedesktop.portal.Desktop"
	portalPath = "/org/freedesktop/portal/desktop"
)

var (
	cbPortalSignal   ptr
	portalRegistered bool
	portalListening  bool
	portalRequests   = map[string]func(code uint32, results ptr){}
	portalTokens     int
	vardictType      ptr
)

// portalSignal handles the signals of the portal the backend subscribes
// to.
func portalSignal(conn, sender, path, iface, signal, params, data ptr) {
	switch goStr(signal) {
	case "Response":
		done := portalRequests[goStr(path)]
		if done == nil {
			return
		}
		delete(portalRequests, goStr(path))
		code := gVariantGetChildValue(params, 0)
		results := gVariantGetChildValue(params, 1)
		done(gVariantGetUint32(code), results)
		gVariantUnref(code)
		gVariantUnref(results)
	case "Activated":
		theBackend.shortcutActivated(params)
	case "SettingChanged":
		// A preference, or the color scheme, changed.
		ns := gVariantGetChildValue(params, 0)
		name := goStr(gVariantGetString(ns, nil))
		gVariantUnref(ns)
		if name == appearanceSettings || name == gnomeInterfaceSettings {
			theBackend.h.ThemeChanged()
		}
	}
}

// The namespaces of the settings portal that preferences come from.
const (
	appearanceSettings     = "org.freedesktop.appearance"
	gnomeInterfaceSettings = "org.gnome.desktop.interface"
)

// watchPortalSettings follows the changes of the settings the desktop
// portal gives.
func watchPortalSettings() {
	if conn, err := bus(); err == nil {
		gDBusConnectionSignalSubscribe(conn, cs(portalName), cs("org.freedesktop.portal.Settings"), cs("SettingChanged"), cs(portalPath), nil, 0, cbPortalSignal, 0, 0)
	}
}

// portalSettings reads the settings of namespaces from the desktop portal,
// as a{sa{sv}}, which the caller unrefs.
func portalSettings(namespaces ...string) (ptr, bool) {
	names := make([]ptr, len(namespaces))
	for i, n := range namespaces {
		names[i] = gVariantNewString(cs(n))
	}
	typ := gVariantTypeNew(cs("s"))
	arr := gVariantNewArray(typ, unsafe.Pointer(&names[0]), uintptr(len(names)))
	res, err := portalCall(portalPath, "org.freedesktop.portal.Settings", "ReadAll", tuple(arr), "(a{sa{sv}})")
	if err != nil {
		return 0, false
	}
	all := gVariantGetChildValue(res, 0)
	gVariantUnref(res)
	return all, true
}

// portalSetting returns the value of key in namespace of the settings
// portalSettings read, if it is of type typ, which the caller unrefs.
func portalSetting(all ptr, namespace, key, typ string) ptr {
	ns := gVariantLookupValue(all, cs(namespace), gVariantTypeNew(cs("a{sv}")))
	if ns == 0 {
		return 0
	}
	defer gVariantUnref(ns)
	v := gVariantLookupValue(ns, cs(key), 0)
	if v == 0 {
		return 0
	}
	if goStr(gVariantGetTypeString(v)) != typ {
		gVariantUnref(v)
		return 0
	}
	return v
}

// portalCall calls a method of the portal, after registering the app with
// it, which must come first.
func portalCall(path, iface, method string, params ptr, replyType string) (ptr, error) {
	registerWithPortal()
	return dbusCall(portalName, path, iface, method, params, replyType)
}

// registerWithPortal tells the portal the app's ID, the name of its
// desktop entry: portals know it only for sandboxed apps and apps the
// desktop launched, and global shortcuts need it. The portal accepts it
// before any other call, when the desktop entry is installed.
func registerWithPortal() {
	if portalRegistered {
		return
	}
	portalRegistered = true
	id := strings.TrimSuffix(desktopEntryID(), ".desktop")
	params := tuple(gVariantNewString(cs(id)), vardict())
	if res, err := dbusCall(portalName, portalPath, "org.freedesktop.host.portal.Registry", "Register", params, ""); err == nil {
		gVariantUnref(res)
	}
}

// portalRequest calls a portal method whose last argument is a vardict of
// options, and which answers with a Response signal: done then gets the
// response code, 0 when the request succeeded, and the results, which it
// must not keep.
func portalRequest(iface, method string, args []ptr, options []ptr, done func(code uint32, results ptr)) error {
	conn, err := bus()
	if err != nil {
		return err
	}
	if !portalListening {
		portalListening = true
		gDBusConnectionSignalSubscribe(conn, cs(portalName), cs("org.freedesktop.portal.Request"), cs("Response"), nil, nil, 0, cbPortalSignal, 0, 0)
	}
	portalTokens++
	token := "mygo" + strconv.Itoa(portalTokens)
	sender := strings.ReplaceAll(strings.TrimPrefix(goStr(gDBusConnectionGetUniqueName(conn)), ":"), ".", "_")
	path := portalPath + "/request/" + sender + "/" + token
	portalRequests[path] = done
	options = append(options, vardictEntry("handle_token", gVariantNewString(cs(token))))
	res, err := portalCall(portalPath, iface, method, tuple(append(args, vardict(options...))...), "(o)")
	if err != nil {
		delete(portalRequests, path)
		return err
	}
	handle := gVariantGetChildValue(res, 0)
	if p := goStr(gVariantGetString(handle, nil)); p != path {
		delete(portalRequests, path) // portals before 0.9 pick the path
		portalRequests[p] = done
	}
	gVariantUnref(handle)
	gVariantUnref(res)
	return nil
}

// portalVersion returns the version of a portal interface, 0 when the
// desktop does not offer it.
func portalVersion(iface string) uint32 {
	res, err := portalCall(portalPath, "org.freedesktop.DBus.Properties", "Get",
		tuple(gVariantNewString(cs(iface)), gVariantNewString(cs("version"))), "(v)")
	if err != nil {
		return 0
	}
	defer gVariantUnref(res)
	v := gVariantGetChildValue(res, 0)
	inner := gVariantGetVariant(v)
	gVariantUnref(v)
	defer gVariantUnref(inner)
	if goStr(gVariantGetTypeString(inner)) != "u" {
		return 0
	}
	return gVariantGetUint32(inner)
}

// vardict builds an a{sv} from entries made by vardictEntry.
func vardict(entries ...ptr) ptr {
	if len(entries) == 0 {
		if vardictType == 0 {
			vardictType = gVariantTypeNew(cs("{sv}"))
		}
		return gVariantNewArray(vardictType, nil, 0)
	}
	return gVariantNewArray(0, unsafe.Pointer(&entries[0]), uintptr(len(entries)))
}

func vardictEntry(key string, v ptr) ptr {
	return gVariantNewDictEntry(gVariantNewString(cs(key)), gVariantNewVariant(v))
}

// vardictString returns the string (or object path) under key in an
// a{sv}, "" when there is none.
func vardictString(dict ptr, key string) string {
	v := gVariantLookupValue(dict, cs(key), 0)
	if v == 0 {
		return ""
	}
	defer gVariantUnref(v)
	if t := goStr(gVariantGetTypeString(v)); t != "s" && t != "o" {
		return ""
	}
	return goStr(gVariantGetString(v, nil))
}
