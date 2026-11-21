//go:build linux && (amd64 || arm64)

package linux

import (
	"errors"
	"fmt"
	"hash/fnv"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

var (
	cbThemeChanged    ptr
	cbMonitorsChanged ptr
	cbNotification    ptr
	cbPowerSignal     ptr
	cbNetworkChanged  ptr
)

func initSystemCallbacks() {
	cbThemeChanged = purego.NewCallback(func(a, b, data ptr) { theBackend.h.ThemeChanged() })
	cbMonitorsChanged = purego.NewCallback(func(display, monitor, data ptr) { theBackend.h.DisplaysChanged() })
	cbPortalSignal = purego.NewCallback(portalSignal)
	cbPowerSignal = purego.NewCallback(func(conn, sender, path, iface, signal, params, data ptr) {
		child := gVariantGetChildValue(params, 0)
		on := gVariantGetBoolean(child)
		gVariantUnref(child)
		event := map[bool]string{true: "lock-screen", false: "unlock-screen"}[on]
		if goStr(signal) == "PrepareForSleep" {
			event = map[bool]string{true: "suspend", false: "resume"}[on]
		}
		theBackend.h.PowerEvent(event)
	})
	cbNotification = purego.NewCallback(func(conn, sender, path, iface, signal, params, data ptr) {
		if goStr(signal) != "ActionInvoked" {
			return
		}
		child := gVariantGetChildValue(params, 0)
		dbusID := gVariantGetUint32(child)
		gVariantUnref(child)
		if id, ok := notifications[dbusID]; ok {
			theBackend.h.NotificationClicked(id)
		}
	})
	// The network-changed signal of GNetworkMonitor (network.go).
	cbNetworkChanged = purego.NewCallback(func(monitor, available, data ptr) {
		theBackend.h.NetworkChanged(theBackend.Network().Status())
	})

	settings := gtkSettingsGetDefault()
	connect(settings, "notify::gtk-theme-name", cbThemeChanged, 0)
	connect(settings, "notify::gtk-font-name", cbThemeChanged, 0) // Theme.UIFont
	connect(settings, "notify::gtk-decoration-layout", cbDecorationLayout, 0)
	connect(settings, "notify::gtk-application-prefer-dark-theme", cbThemeChanged, 0)
	connect(settings, "notify::gtk-enable-animations", cbThemeChanged, 0) // Theme.Preferences
	watchPortalSettings()
	if d := gdkDisplayGetDefault(); d != 0 {
		connect(d, "monitor-added", cbMonitorsChanged, 0)
		connect(d, "monitor-removed", cbMonitorsChanged, 0)
	}
}

// Pixbuf helpers.

func pixbufFromPNG(png []byte) (ptr, error) {
	if len(png) == 0 {
		return 0, errors.New("mygo: empty image")
	}
	loader := gdkPixbufLoaderNew()
	defer gObjectUnref(loader)
	var gerr ptr
	if !gdkPixbufLoaderWrite(loader, unsafe.Pointer(&png[0]), uintptr(len(png)), &gerr) {
		gdkPixbufLoaderClose(loader, nil)
		return 0, gErr(gerr)
	}
	if !gdkPixbufLoaderClose(loader, &gerr) {
		return 0, gErr(gerr)
	}
	pix := gdkPixbufLoaderGetPixbuf(loader)
	if pix == 0 {
		return 0, errors.New("mygo: invalid image")
	}
	return gObjectRef(pix), nil
}

func pngFromPixbuf(pix ptr) []byte {
	var buf ptr
	var size uintptr
	if !gdkPixbufSaveToBufferv(pix, &buf, &size, cs("png"), 0, 0, nil) {
		return nil
	}
	out := append([]byte(nil), unsafe.Slice(*(**byte)(unsafe.Pointer(&buf)), size)...)
	gFree(buf)
	return out
}

// Clipboard.

type clipboard struct{}

func clip() ptr {
	cb := gtkClipboardGet(gdkAtomIntern(cs("CLIPBOARD"), false))
	if !clipboardConnected {
		connect(cb, "owner-change", cbClipboardOwner, 0)
		clipboardConnected = true
	}
	return cb
}

func (clipboard) ReadText() string        { return takeStr(gtkClipboardWaitForText(clip())) }
func (c clipboard) WriteText(text string) { _ = c.WriteData(transfer.TextData(text), nil) }

func (clipboard) ReadHTML() string {
	sd := gtkClipboardWaitForContents(clip(), gdkAtomIntern(cs("text/html"), false))
	if sd == 0 {
		return ""
	}
	defer gtkSelectionDataFree(sd)
	n := gtkSelectionDataGetLength(sd)
	data := gtkSelectionDataGetData(sd)
	if n <= 0 || data == 0 {
		return ""
	}
	return string(unsafe.Slice(*(**byte)(unsafe.Pointer(&data)), n))
}

func (c clipboard) WriteHTML(markup string) {
	_ = c.WriteData(transfer.New(transfer.NewItem(transfer.Bytes(transfer.HTML, []byte(markup)), transfer.Bytes(transfer.Text, []byte(markup)))), nil)
}

func (clipboard) ReadImage() []byte {
	pix := gtkClipboardWaitForImage(clip())
	if pix == 0 {
		return nil
	}
	defer gObjectUnref(pix)
	return pngFromPixbuf(pix)
}

func (c clipboard) WriteImage(png []byte) error {
	pix, err := pixbufFromPNG(png)
	if err != nil {
		return err
	}
	defer gObjectUnref(pix)
	return c.WriteData(transfer.New(transfer.NewItem(transfer.Bytes(transfer.PNG, png))), nil)
}

func (clipboard) Clear() { gtkClipboardClear(clip()) }

func (clipboard) AvailableFormats() []string {
	var atoms ptr
	var n int32
	if !gtkClipboardWaitForTargets(clip(), &atoms, &n) || atoms == 0 {
		return nil
	}
	defer gFree(atoms)
	out := make([]string, 0, n)
	for i := range int(n) {
		out = append(out, takeStr(gdkAtomName(field[ptr](atoms, uintptr(i)*8))))
	}
	return out
}

// Shell.

type shell struct{}

func launch(uri string) error {
	var gerr ptr
	if !gAppInfoLaunchDefaultForURI(cs(uri), 0, &gerr) {
		return gErr(gerr)
	}
	return nil
}

func fileURI(path string) string {
	f := gFileNewForPath(cs(path))
	defer gObjectUnref(f)
	return takeStr(gFileGetURI(f))
}

func (shell) OpenExternal(url string) error { return launch(url) }
func (shell) OpenPath(path string) error    { return launch(fileURI(path)) }

func (s shell) ShowItemInFolder(path string) {
	uri := fileURI(path)
	uris := gVariantNewArray(gVariantTypeNew(cs("s")), unsafe.Pointer(&[]ptr{gVariantNewString(cs(uri))}[0]), 1)
	params := tuple(uris, gVariantNewString(cs("")))
	if _, err := dbusCall("org.freedesktop.FileManager1", "/org/freedesktop/FileManager1", "org.freedesktop.FileManager1", "ShowItems", params, ""); err != nil {
		_ = s.OpenPath(filepath.Dir(path))
	}
}

func (shell) TrashItem(path string) error {
	f := gFileNewForPath(cs(path))
	defer gObjectUnref(f)
	var gerr ptr
	if !gFileTrash(f, 0, &gerr) {
		return gErr(gerr)
	}
	return nil
}

func (shell) Beep() { gdkDisplayBeep(gdkDisplayGetDefault()) }

// Screen.

type screen struct{}

func (screen) Displays() []platform.Display {
	d := gdkDisplayGetDefault()
	if d == 0 {
		return nil
	}
	primary := gdkDisplayGetPrimaryMonitor(d)
	n := int(gdkDisplayGetNMonitors(d))
	out := make([]platform.Display, 0, n)
	for i := range n {
		m := gdkDisplayGetMonitor(d, int32(i))
		var geo, work gdkRectangle
		gdkMonitorGetGeometry(m, &geo)
		gdkMonitorGetWorkarea(m, &work)
		out = append(out, platform.Display{
			ID:          int64(i),
			Label:       goStr(gdkMonitorGetModel(m)),
			Bounds:      platform.Rect{X: int(geo.X), Y: int(geo.Y), Width: int(geo.Width), Height: int(geo.Height)},
			WorkArea:    platform.Rect{X: int(work.X), Y: int(work.Y), Width: int(work.Width), Height: int(work.Height)},
			ScaleFactor: float64(gdkMonitorGetScaleFactor(m)),
			Primary:     m == primary || (primary == 0 && i == 0),
		})
	}
	// Primary first, like on other platforms.
	for i, disp := range out {
		if disp.Primary && i > 0 {
			out[0], out[i] = out[i], out[0]
			break
		}
	}
	return out
}

func (screen) CursorPoint() platform.Point {
	d := gdkDisplayGetDefault()
	if d == 0 {
		return platform.Point{}
	}
	var x, y int32
	gdkDeviceGetPosition(gdkSeatGetPointer(gdkDisplayGetDefaultSeat(d)), nil, &x, &y)
	return platform.Point{X: int(x), Y: int(y)}
}

// Theme.

type theme struct{ b *Backend }

// FontRendering returns GTK's gtk-xft settings, which the desktop's
// settings daemon gives (XSETTINGS) or GDK's defaults, as GTK makes them
// the screen's font options: no hinting where hinting is off, and subpixel
// antialiasing where the screen's subpixels have an order.
func (theme) FontRendering() platform.FontRendering {
	settings := gtkSettingsGetDefault()
	if settings == 0 {
		return platform.FontRendering{}
	}
	var antialias, hinting int32
	var style, rgba ptr
	gObjectGetPtr(settings, cs("gtk-xft-antialias"), unsafe.Pointer(&antialias), 0)
	gObjectGetPtr(settings, cs("gtk-xft-hinting"), unsafe.Pointer(&hinting), 0)
	gObjectGetPtr(settings, cs("gtk-xft-hintstyle"), unsafe.Pointer(&style), 0)
	gObjectGetPtr(settings, cs("gtk-xft-rgba"), unsafe.Pointer(&rgba), 0)
	var r platform.FontRendering
	switch hintStyle := takeStr(style); {
	case hinting == 0:
		r.Hinting = "none"
	case hinting == 1 && strings.HasPrefix(hintStyle, "hint"):
		switch h := strings.TrimPrefix(hintStyle, "hint"); h {
		case "none", "slight", "medium", "full":
			r.Hinting = h
		}
	}
	switch order := takeStr(rgba); order {
	case "rgb", "bgr", "vrgb", "vbgr":
		r.Subpixels = order
	}
	switch {
	case antialias == 0:
		r.Antialias = "none"
	case antialias == 1 && r.Subpixels != "":
		r.Antialias = "subpixel"
	case antialias == 1:
		r.Antialias = "gray"
	}
	return r
}

// UIFont returns the family of the font GTK apps show their interface
// in, gtk-font-name, such as "Cantarell 11" on GNOME, which the desktop's
// settings daemon or settings.ini gives; fontconfig knows only a default
// sans-serif.
func (theme) UIFont() string {
	var name ptr
	gObjectGetPtr(gtkSettingsGetDefault(), cs("gtk-font-name"), unsafe.Pointer(&name), 0)
	spec := takeStr(name)
	if spec == "" {
		return ""
	}
	var (
		fromString func(s *byte) ptr
		getFamily  func(d ptr) ptr
		free       func(d ptr)
	)
	pango, err := open("libpango-1.0.so.0")
	if err != nil || !bind(pango, &fromString, "pango_font_description_from_string") ||
		!bind(pango, &getFamily, "pango_font_description_get_family") || !bind(pango, &free, "pango_font_description_free") {
		return ""
	}
	desc := fromString(cs(spec))
	defer free(desc)
	return goStr(getFamily(desc))
}

// portalColorScheme reads the desktop wide preference (1 = dark, 2 =
// light, 0 = no preference) from the settings portal.
func portalColorScheme() (uint32, bool) {
	params := tuple(gVariantNewString(cs("org.freedesktop.appearance")), gVariantNewString(cs("color-scheme")))
	res, err := portalCall(portalPath, "org.freedesktop.portal.Settings", "Read", params, "(v)")
	if err != nil {
		return 0, false
	}
	defer gVariantUnref(res)
	v := gVariantGetChildValue(res, 0) // v
	inner := gVariantGetVariant(v)     // v (the portal wraps twice)
	gVariantUnref(v)
	for strings.HasPrefix(goStr(gVariantGetTypeString(inner)), "v") {
		next := gVariantGetVariant(inner)
		gVariantUnref(inner)
		inner = next
	}
	defer gVariantUnref(inner)
	if goStr(gVariantGetTypeString(inner)) != "u" {
		return 0, false
	}
	return gVariantGetUint32(inner), true
}

// doubleClick reads the double click as GTK's settings have it from the
// desktop.
func doubleClick() platform.DoubleClick {
	ms, distance := int32(400), int32(5) // GTK's defaults
	if settings := gtkSettingsGetDefault(); settings != 0 {
		gObjectGetPtr(settings, cs("gtk-double-click-time"), unsafe.Pointer(&ms), 0)
		gObjectGetPtr(settings, cs("gtk-double-click-distance"), unsafe.Pointer(&distance), 0)
	}
	return platform.DoubleClick{Time: time.Duration(ms) * time.Millisecond, DX: float64(distance), DY: float64(distance)}
}

// Preferences reads GTK's animations setting, and the accent, contrast
// and text size the desktop portal gives, as GNOME's and KDE's do.
func (t theme) Preferences() platform.Preferences {
	var p platform.Preferences
	settings := gtkSettingsGetDefault()
	animations := int32(1)
	gObjectGetPtr(settings, cs("gtk-enable-animations"), unsafe.Pointer(&animations), 0)
	p.ReduceMotion = animations == 0
	var name ptr
	gObjectGetPtr(settings, cs("gtk-theme-name"), unsafe.Pointer(&name), 0)
	p.HighContrast = strings.Contains(strings.ToLower(takeStr(name)), "highcontrast")
	all, ok := portalSettings(appearanceSettings, gnomeInterfaceSettings)
	if !ok {
		return p
	}
	defer gVariantUnref(all)
	// The accent: red, green and blue between 0 and 1, others for none.
	if v := portalSetting(all, appearanceSettings, "accent-color", "(ddd)"); v != 0 {
		var rgb [3]float64
		in := true
		for i := range rgb {
			c := gVariantGetChildValue(v, uintptr(i))
			rgb[i] = gVariantGetDouble(c)
			gVariantUnref(c)
			in = in && rgb[i] >= 0 && rgb[i] <= 1
		}
		gVariantUnref(v)
		if in {
			b := func(f float64) uint8 { return uint8(f*255 + 0.5) }
			p.Accent = platform.Color{R: b(rgb[0]), G: b(rgb[1]), B: b(rgb[2]), A: 255}
		}
	}
	if v := portalSetting(all, appearanceSettings, "contrast", "u"); v != 0 {
		p.HighContrast = p.HighContrast || gVariantGetUint32(v) == 1
		gVariantUnref(v)
	}
	if v := portalSetting(all, gnomeInterfaceSettings, "text-scaling-factor", "d"); v != 0 {
		if s := gVariantGetDouble(v); s >= 0.5 && s <= 5 {
			p.TextScale = s
		}
		gVariantUnref(v)
	}
	return p
}

func (t theme) IsDark() bool {
	settings := gtkSettingsGetDefault()
	var prefer int32
	gObjectGetPtr(settings, cs("gtk-application-prefer-dark-theme"), unsafe.Pointer(&prefer), 0)
	if prefer != 0 {
		return true
	}
	if scheme, ok := portalColorScheme(); ok && scheme != 0 {
		return scheme == 1
	}
	var name ptr
	gObjectGetPtr(settings, cs("gtk-theme-name"), unsafe.Pointer(&name), 0)
	return strings.Contains(strings.ToLower(takeStr(name)), "dark")
}

func (t theme) SetSource(source string) {
	dark := source == "dark"
	if source == "system" {
		scheme, _ := portalColorScheme()
		dark = scheme == 1
	}
	gObjectSetBool(gtkSettingsGetDefault(), cs("gtk-application-prefer-dark-theme"), dark, 0)
}

// D-Bus helpers.

var sessionBus ptr

func bus() (ptr, error) {
	if sessionBus != 0 {
		return sessionBus, nil
	}
	var gerr ptr
	conn := gBusGetSync(2, 0, &gerr) // G_BUS_TYPE_SESSION
	if conn == 0 {
		return 0, gErr(gerr)
	}
	sessionBus = conn
	return conn, nil
}

func tuple(children ...ptr) ptr {
	return gVariantNewTuple(unsafe.Pointer(&children[0]), uintptr(len(children)))
}

func dbusCall(name, path, iface, method string, params ptr, replyType string) (ptr, error) {
	conn, err := bus()
	if err != nil {
		return 0, err
	}
	var rt ptr
	if replyType != "" {
		rt = gVariantTypeNew(cs(replyType))
	}
	var gerr ptr
	res := gDBusConnectionCallSync(conn, cs(name), cs(path), cs(iface), cs(method), params, rt, 0, 3000, 0, &gerr)
	if res == 0 {
		return 0, gErr(gerr)
	}
	return res, nil
}

// updateLauncherEntry shows the progress and badge count on the app's
// launcher entry with the Unity launcher API, which the docks of KDE Plasma
// and Ubuntu, Dash to Dock and Plank implement. They find the app by its
// desktop entry.
func (b *Backend) updateLauncherEntry() {
	conn, err := bus()
	if err != nil {
		return
	}
	l := &b.launcher
	count, _ := strconv.Atoi(l.badge)
	entry := func(key string, v ptr) ptr {
		return gVariantNewDictEntry(gVariantNewString(cs(key)), gVariantNewVariant(v))
	}
	props := []ptr{
		entry("progress", gVariantNewDouble(l.progress)),
		entry("progress-visible", gVariantNewBoolean(l.showing)),
		entry("count", gVariantNewInt64(int64(count))),
		entry("count-visible", gVariantNewBoolean(count > 0)),
	}
	id := desktopEntryID()
	params := tuple(gVariantNewString(cs("application://"+id)), gVariantNewArray(0, unsafe.Pointer(&props[0]), uintptr(len(props))))
	h := fnv.New32a()
	h.Write([]byte(id))
	path := fmt.Sprintf("/com/canonical/unity/launcherentry/%d", h.Sum32())
	var gerr ptr
	if !gDBusConnectionEmitSignal(conn, nil, cs(path), cs("com.canonical.Unity.LauncherEntry"), cs("Update"), params, &gerr) {
		_ = gErr(gerr)
	}
}

// desktopEntryID is the file name of the app's desktop entry: the one it
// was launched from, else the one `mygo build` writes, named after the
// executable.
func desktopEntryID() string {
	for _, env := range []string{"GIO_LAUNCHED_DESKTOP_FILE", "BAMF_DESKTOP_FILE_HINT"} {
		if p := os.Getenv(env); p != "" {
			return filepath.Base(p)
		}
	}
	exe, _ := os.Executable()
	return filepath.Base(exe) + ".desktop"
}

// Notifications (org.freedesktop.Notifications).

var (
	notifications    = map[uint32]string{}
	notifyDBusIDs    = map[string]uint32{}
	notifySubscribed bool
)

func (b *Backend) NotificationsSupported() bool {
	_, err := bus()
	return err == nil
}

// WebViewAvailable reports whether a window showing a web page can have
// one: WebKitGTK, JavaScriptCore and libsoup, which it needs. Windows
// showing native UI need GTK alone, so an app whose windows all do runs
// without them. Loading the libraries is as much as can be known before a
// page loads.
func (b *Backend) WebViewAvailable() bool { return webKit() == nil }

func (b *Backend) ShowNotification(n *platform.Notification, done func(error)) {
	done(b.showNotification(n))
}

func (b *Backend) showNotification(n *platform.Notification) error {
	conn, err := bus()
	if err != nil {
		return fmt.Errorf("mygo: notifications need a D-Bus session: %w", err)
	}
	if !notifySubscribed {
		notifySubscribed = true
		gDBusConnectionSignalSubscribe(conn, cs("org.freedesktop.Notifications"), cs("org.freedesktop.Notifications"),
			cs("ActionInvoked"), cs("/org/freedesktop/Notifications"), nil, 0, cbNotification, 0, 0)
	}
	actions := []ptr{gVariantNewString(cs("default")), gVariantNewString(cs("Open"))}
	var hints []ptr
	if n.Silent {
		hints = append(hints, gVariantNewDictEntry(gVariantNewString(cs("suppress-sound")), gVariantNewVariant(gVariantNewBoolean(true))))
	}
	hintArray := gVariantNewArray(gVariantTypeNew(cs("{sv}")), nil, 0)
	if len(hints) > 0 {
		hintArray = gVariantNewArray(0, unsafe.Pointer(&hints[0]), uintptr(len(hints)))
	}
	body := n.Body
	if n.Subtitle != "" {
		body = n.Subtitle + "\n" + body
	}
	params := tuple(
		gVariantNewString(cs(appName())),
		gVariantNewUint32(notifyDBusIDs[n.ID]),
		gVariantNewString(cs("")),
		gVariantNewString(cs(n.Title)),
		gVariantNewString(cs(body)),
		gVariantNewArray(0, unsafe.Pointer(&actions[0]), uintptr(len(actions))),
		hintArray,
		gVariantNewInt32(-1),
	)
	res, err := dbusCall("org.freedesktop.Notifications", "/org/freedesktop/Notifications", "org.freedesktop.Notifications", "Notify", params, "(u)")
	if err != nil {
		return err
	}
	child := gVariantGetChildValue(res, 0)
	dbusID := gVariantGetUint32(child)
	gVariantUnref(child)
	gVariantUnref(res)
	notifications[dbusID] = n.ID
	notifyDBusIDs[n.ID] = dbusID
	return nil
}

func (b *Backend) RemoveNotification(id string) {
	dbusID, ok := notifyDBusIDs[id]
	if !ok {
		return
	}
	delete(notifyDBusIDs, id)
	delete(notifications, dbusID)
	if res, err := dbusCall("org.freedesktop.Notifications", "/org/freedesktop/Notifications", "org.freedesktop.Notifications", "CloseNotification", tuple(gVariantNewUint32(dbusID)), ""); err == nil {
		gVariantUnref(res)
	}
}

func (b *Backend) RemoveAllNotifications() {
	for id := range notifyDBusIDs {
		b.RemoveNotification(id)
	}
}

func appName() string {
	if theBackend.name != "" {
		return theBackend.name
	}
	return "MyGo"
}
