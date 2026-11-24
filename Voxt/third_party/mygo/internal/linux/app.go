//go:build linux && (amd64 || arm64)

package linux

import (
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"syscall"

	"github.com/ebitengine/purego"

	"github.com/egoist/mygo/internal/platform"
)

// Backend is the Linux (GTK 3 + WebKitGTK) implementation of
// platform.Backend.
type Backend struct {
	h       platform.AppHandler
	name    string
	running bool

	windows map[int]*window
	// flyouts are the flyouts that show on Wayland, in the order they
	// showed: each popup's parent is the one before or a toplevel.
	flyouts   []*window
	byWebView map[ptr]*window
	// webViews are the web views under surfaces (webview.go), by id.
	webViews map[int]*window
	nextID   int
	// frames are the window manager's frames seen last around a decorated
	// window in its normal state that the user cannot resize and one they
	// can, which windows that have not shown yet expect (see
	// window.frameExtents).
	frames [2]extents

	appMenu *platform.Menu
	// languages are those SetLanguages told, which the web context
	// takes as the first web view is made (webLanguages).
	languages    []string
	webLanguages bool
	schemes      map[string]bool
	signaled     atomic.Bool
	quitLoops    []ptr
	popups       []ptr // context menus being shown
	modals       []modal
	trays        map[int]*tray
	nextTray     int
	grabs        map[int]grab    // global shortcuts on X11
	portal       portalShortcuts // global shortcuts elsewhere
	themeHooks   bool
	// onX11 reports an X11 display, where windows have positions, and
	// wmName names its window manager.
	onX11  bool
	wmName func(screen ptr) ptr
	// compositorDecorates reports a Wayland compositor that decorates
	// windows itself, whose title bar shows no buttons of GTK's (titlebar.go).
	compositorDecorates bool
	// announceCSD tells a Wayland compositor that a window decorates
	// itself, so that it draws no title bar (gdk_wayland_window_announce_csd).
	announceCSD func(window ptr)

	// What the launcher entry shows (Window.SetProgressBar, badges).
	launcher struct {
		badge    string
		progress float64
		showing  bool // progress
	}
}

var theBackend *Backend

// New creates the Linux backend. Libraries are loaded by Init so that
// programs still start (and report a clear error) on machines without GTK.
func New() *Backend {
	return &Backend{
		windows:   map[int]*window{},
		byWebView: map[ptr]*window{},
		webViews:  map[int]*window{},
		schemes:   map[string]bool{},
		trays:     map[int]*tray{},
	}
}

func (b *Backend) Name() string { return "linux/webkitgtk" }

// windowManager reports an X11 window manager, which places windows once
// they show; Wayland compositors do not tell where windows are.
func (b *Backend) windowManager(win ptr) bool {
	return b.onX11 && goStr(b.wmName(gtkWidgetGetScreen(win))) != "unknown"
}

// IsMainThread reports whether the caller runs on the thread the process
// started with, whose thread id is the process id. Package mygo locks the
// main goroutine to it, so this holds before Init too.
func (b *Backend) IsMainThread() bool {
	return syscall.Gettid() == syscall.Getpid()
}

func (b *Backend) Init(h platform.AppHandler, opts platform.AppOptions) error {
	b.h = h
	b.name = opts.Name
	theBackend = b
	if err := load(); err != nil {
		return err
	}
	if os.Getenv("WEBKIT_DISABLE_DMABUF_RENDERER") == "" {
		// Avoid blank windows on setups where WebKitGTK's DMA-BUF renderer
		// fails (NVIDIA, VMs, containers).
		_ = os.Setenv("WEBKIT_DISABLE_DMABUF_RENDERER", "1")
	}
	if !gtkInitCheck(0, 0) {
		return errors.New("mygo: cannot open display (is DISPLAY or WAYLAND_DISPLAY set?)")
	}
	var x11Type func() uintptr
	b.onX11 = bind(libGDK, &x11Type, "gdk_x11_display_get_type") && gTypeCheckInstanceIsA(gdkDisplayGetDefault(), x11Type()) &&
		bind(libGDK, &b.wmName, "gdk_x11_screen_get_window_manager_name")
	// GTK has the compositor's default decoration mode once the display is
	// open, and decides with it which windows it decorates.
	var waylandType func() uintptr
	var prefersSSD func(display ptr) bool
	onWayland := bind(libGDK, &waylandType, "gdk_wayland_display_get_type") &&
		gTypeCheckInstanceIsA(gdkDisplayGetDefault(), waylandType())
	b.compositorDecorates = onWayland &&
		bind(libGDK, &prefersSSD, "gdk_wayland_display_prefers_ssd") && prefersSSD(gdkDisplayGetDefault())
	if onWayland {
		bind(libGDK, &b.announceCSD, "gdk_wayland_window_announce_csd")
	}
	initCallbacks()
	return nil
}

func (b *Backend) Run() error {
	b.running = true
	gIdleAddFull(0, readyCallback, 0, 0)
	gtkMain()
	b.running = false
	return nil
}

func (b *Backend) Quit() {
	// Dialogs and popups run nested main loops, which gtk_main_quit does
	// not end: dismiss them so gtk_main can return.
	for _, m := range b.modals {
		if m.native {
			gtkNativeDialogHide(m.dialog)
		} else {
			gtkWidgetHide(m.dialog)
		}
	}
	for _, l := range b.quitLoops {
		gMainLoopQuit(l)
	}
	if b.running {
		gtkMainQuit()
	}
}

// modal is a dialog running its own main loop.
type modal struct {
	dialog ptr
	native bool // a GtkNativeDialog, else a GtkDialog
}

// runDialog runs a GtkDialog (native false) or a GtkNativeDialog until it
// is answered, or dismissed by Quit.
func (b *Backend) runDialog(d ptr, native bool) int32 {
	b.modals = append(b.modals, modal{d, native})
	defer func() { b.modals = b.modals[:len(b.modals)-1] }()
	if native {
		return gtkNativeDialogRun(d)
	}
	return gtkDialogRun(d)
}

func (b *Backend) Signal() {
	if gIdleAddFull == nil || !b.signaled.CompareAndSwap(false, true) {
		return
	}
	gIdleAddFull(0, dispatchCallback, 0, 0)
}

func (b *Backend) Step() { gMainContextIteration(0, true) }

func (b *Backend) Wake() {
	if gMainContextWakeup != nil {
		gMainContextWakeup(0)
	}
}

func (b *Backend) App() platform.AppController { return appController{b} }
func (b *Backend) Dialogs() platform.Dialogs   { return dialogs{b} }
func (b *Backend) Clipboard() platform.Clipboard {
	return clipboard{}
}
func (b *Backend) Shell() platform.Shell     { return shell{} }
func (b *Backend) Screen() platform.Screen   { return screen{} }
func (b *Backend) Theme() platform.Theme     { return theme{b} }
func (b *Backend) Power() platform.Power     { return power{b} }
func (b *Backend) Secrets() platform.Secrets { return secrets{} }

// Callbacks are created once; purego callbacks are never freed.
var (
	callbacksOnce    sync.Once
	readyCallback    ptr
	dispatchCallback ptr
)

func initCallbacks() {
	callbacksOnce.Do(func() {
		readyCallback = purego.NewCallback(func(data ptr) int32 {
			theBackend.h.Ready()
			return 0 // G_SOURCE_REMOVE
		})
		dispatchCallback = purego.NewCallback(func(data ptr) int32 {
			theBackend.signaled.Store(false)
			theBackend.h.Dispatch()
			return 0
		})
		initWindowCallbacks()
		initFlyoutCallbacks()
		initSurfaceCallbacks()
		initDownloadCallbacks()
		initMenuCallbacks()
		initSystemCallbacks()
		initClipboardCallbacks()
	})
}

type appController struct{ b *Backend }

func (a appController) SetActivationPolicy(string) {}

func (a appController) Activate() {
	for _, w := range a.b.windows {
		if gtkWidgetGetVisible(w.win) {
			gtkWindowPresent(w.win)
			return
		}
	}
}

func (a appController) Hide() {
	for _, w := range a.b.windows {
		gtkWidgetHide(w.win)
	}
}

func (a appController) Unhide() {
	for _, w := range a.b.windows {
		gtkWidgetShow(w.win)
	}
}

func (a appController) IsHidden() bool { return false }
func (a appController) SetBadge(label string) {
	a.b.launcher.badge = label
	a.b.updateLauncherEntry()
}

func (a appController) Badge() string { return a.b.launcher.badge }

func (a appController) Bounce(critical bool) int {
	for _, w := range a.b.windows {
		gtkWindowSetUrgencyHint(w.win, true)
	}
	return 1
}

func (a appController) CancelBounce(int) {
	for _, w := range a.b.windows {
		gtkWindowSetUrgencyHint(w.win, false)
	}
}

// SetDockMenu does nothing: there is no Dock.
func (a appController) SetDockMenu(*platform.Menu) {}

func (a appController) SetDockIcon(png []byte) error {
	pix, err := pixbufFromPNG(png)
	if err != nil {
		return err
	}
	defer gObjectUnref(pix)
	gtkWindowSetDefaultIcon(pix)
	return nil
}

func (a appController) ShowAboutPanel(o platform.AboutPanelOptions) {
	d := gtkAboutDialogNew()
	gtkAboutDialogSetProgramName(d, cs(o.ApplicationName))
	version := o.ApplicationVersion
	if o.Version != "" {
		version += " (" + o.Version + ")"
	}
	gtkAboutDialogSetVersion(d, optCS(version))
	gtkAboutDialogSetCopyright(d, optCS(o.Copyright))
	gtkAboutDialogSetComments(d, optCS(o.Credits))
	a.b.runDialog(d, false)
	gtkWidgetDestroy(d)
}

func (a appController) Package() (platform.PackageInfo, bool) { return platform.PackageInfo{}, false }

func (a appController) ClearBrowsingData(done func(error)) {
	// Web views of earlier runs kept their data on disk: WebKitGTK clears
	// it, loaded for that when no window has shown a web page yet.
	if webKit() != nil {
		done(nil) // no web view kept any data
		return
	}
	const all = 1<<14 - 1 // WEBKIT_WEBSITE_DATA_ALL
	manager := webkitWebContextGetWebsiteDataManager(webkitWebContextGetDefault())
	id := pending.add(func(source, res ptr) {
		var gerr ptr
		webkitWebsiteDataManagerClearFinish(source, res, &gerr)
		done(gErr(gerr))
	})
	webkitWebsiteDataManagerClear(manager, all, 0, 0, cbAsyncReady, id)
}

func (a appController) AddRecentDocument(path string) error {
	uri := fileURI(path)
	if uri == "" {
		return fmt.Errorf("mygo: cannot record %q as a recent document", path)
	}
	// GTK records the item asynchronously, under the application name, so
	// a failure to write the list is not reported here.
	if !gtkRecentManagerAddItem(gtkRecentManagerGetDefault(), cs(uri)) {
		return fmt.Errorf("mygo: the recent files list refused %q", path)
	}
	return nil
}

// ClearRecentDocuments removes the items of the shared list that this
// application registered, including those of earlier runs, and leaves the
// other applications' items alone.
func (a appController) ClearRecentDocuments() error {
	m := gtkRecentManagerGetDefault()
	name := gGetApplicationName()
	if name == 0 {
		return nil
	}
	var uris []string
	list := gtkRecentManagerGetItems(m)
	for node := list; node != 0; node = field[ptr](node, 8) {
		info := field[ptr](node, 0)
		if gtkRecentInfoHasApplication(info, name) {
			uris = append(uris, goStr(gtkRecentInfoGetURI(info)))
		}
		gtkRecentInfoUnref(info)
	}
	gListFree(list)
	for _, uri := range uris {
		gtkRecentManagerRemoveItem(m, cs(uri), 0)
	}
	return nil
}

func (appController) SetJumpList(platform.JumpList) error { return nil }
