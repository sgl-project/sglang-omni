package mygo

import (
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/egoist/mygo/internal/platform"
)

// ActivationPolicy controls how a macOS application appears in the Dock and
// app switcher.
type ActivationPolicy string

// Activation policies.
const (
	// ActivationPolicyRegular is an ordinary app with a Dock icon and menu bar.
	ActivationPolicyRegular ActivationPolicy = "regular"
	// ActivationPolicyAccessory hides the Dock icon; windows can still be
	// shown. Typical for menu bar (tray) apps.
	ActivationPolicyAccessory ActivationPolicy = "accessory"
	// ActivationPolicyProhibited hides the app entirely.
	ActivationPolicyProhibited ActivationPolicy = "prohibited"
)

// Application controls the event loop and the lifecycle of the process.
// Use the App singleton.
type Application struct {
	mu        sync.Mutex
	name      string
	version   string
	policy    ActivationPolicy
	running   bool
	ready     bool
	readyCh   chan struct{}
	whenReady []func()
	menu      *Menu
	menuSet   bool
	paths     map[PathName]string
	// stateStore is the WindowStateStore, or nil for the file.
	stateStore WindowStateStore
	finished   sync.Once

	// quitting is only touched on the main thread.
	quitting    bool
	relaunch    bool // Relaunch: start again once quit (main thread)
	initialized bool // Run initialized the backend (main thread)

	// Dock controls the Dock icon on macOS.
	Dock *Dock
	// JumpList controls the jump list of the taskbar button on Windows.
	JumpList *JumpList

	onReady           listeners[func()]
	onWindowAllClosed listeners[func()]
	onBeforeQuit      listeners[func(*QuitEvent)]
	onWillQuit        listeners[func(*QuitEvent)]
	onQuit            listeners[func()]
	onActivate        listeners[func(bool)]
	onDidBecomeActive listeners[func()]
	onDidResignActive listeners[func()]
	onOpenURL         listeners[func(string)]
	onOpenFile        listeners[func(string)]
	onWindowCreated   listeners[func(*Window)]

	onNotificationClick listeners[func(string)]
}

// App is the application singleton.
var App = &Application{readyCh: make(chan struct{}), Dock: &Dock{}, JumpList: &JumpList{}}

// Run starts the native event loop and blocks until the application quits.
// It must be called from the main goroutine, usually as the last statement
// of main.
//
// SIGINT and SIGTERM quit the application like Quit; a second signal exits
// immediately.
//
// When the MYGO_GENERATE environment variable is set (see `mygo generate`),
// Run writes the TypeScript client for the bound services to that path, or
// nothing when the app binds none, and returns without starting the event
// loop.
func (a *Application) Run() error {
	if out := os.Getenv("MYGO_GENERATE"); out != "" {
		return writeClient(out)
	}
	if !isMainThread() {
		return errors.New("mygo: App.Run must be called from the main goroutine")
	}
	a.mu.Lock()
	if a.running {
		a.mu.Unlock()
		return errors.New("mygo: App.Run called twice")
	}
	a.running = true
	a.mu.Unlock()
	go cleanUpdateLeftovers()

	b := backend()
	if err := b.Init(appHandler{}, platform.AppOptions{
		Name:             a.Name(),
		ActivationPolicy: string(a.activationPolicy()),
	}); err != nil {
		return err
	}
	a.initialized = true
	// Settings made before Run, which the backend could not take yet.
	Theme.apply()
	a.startLocale()
	if m := a.Dock.Menu(); m != nil {
		b.App().SetDockMenu(m.snapshot())
	}
	quitOnSignals()
	// Run whatever was scheduled before the event loop existed.
	b.Signal()
	err := b.Run()
	a.finish()
	return err
}

// needsApp panics when main calls something that needs the running app
// before Run: the backend is not initialized yet (on Linux its libraries
// are not even loaded), and main cannot wait for it. Other goroutines
// wait for the app to start instead. call names the method.
func needsApp(call string) {
	if isMainThread() && !App.initialized {
		panic("mygo: " + call + " called before App.Run; call it once the application is ready (App.WhenReady)")
	}
}

// WhenReady calls fn on the main thread once the application has finished
// launching. Windows can only be created after that point. If the
// application is already ready, fn is scheduled right away.
func (a *Application) WhenReady(fn func()) {
	a.mu.Lock()
	if !a.ready {
		a.whenReady = append(a.whenReady, fn)
		a.mu.Unlock()
		return
	}
	a.mu.Unlock()
	postMain(fn)
}

// IsReady reports whether the application has finished launching.
func (a *Application) IsReady() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.ready
}

// Quit closes all windows and then quits. OnBeforeQuit and OnWillQuit
// listeners, as well as the OnClose listeners of every window, can cancel
// it.
func (a *Application) Quit() {
	postMain(func() {
		if a.prepareQuit() {
			backend().Quit()
		}
	})
}

// Exit terminates the process immediately with the given exit code,
// without emitting quit events or asking windows.
func (a *Application) Exit(code int) {
	onMain(func() {
		for _, w := range Windows() {
			w.destroy()
		}
		saveWindowStates()
	})
	os.Exit(code)
}

// prepareQuit runs the quit sequence on the main thread and reports whether
// the application may quit.
func (a *Application) prepareQuit() bool {
	if a.quitting {
		return false
	}
	before := &QuitEvent{}
	fire1(&a.onBeforeQuit, before)
	if before.prevented {
		return false
	}
	a.quitting = true
	for _, w := range Windows() {
		if !w.close() {
			a.quitting = false
			return false
		}
	}
	will := &QuitEvent{}
	fire1(&a.onWillQuit, will)
	if will.prevented {
		a.quitting = false
		return false
	}
	// Providers are application-owned, independent of the windows just
	// closed. Persist them before the native loop ends; callers needing
	// error handling may Flush explicitly from a quit listener.
	_ = Clipboard.Flush()
	stopCookieCalls()
	return true
}

// finish runs once after the event loop has stopped.
func (a *Application) finish() {
	a.finished.Do(func() {
		stopCookieCalls()
		stopFileWatches()
		clipboardStopped = true
		backend().Clipboard().Close()
		saveWindowStates()
		fire(&a.onQuit)
		if a.relaunch {
			relaunchNow() // after OnQuit, which releases the single instance lock
		}
		loop.shutdown()
	})
}

// lastWindowClosed is called on the main thread when the last window has
// been closed.
func (a *Application) lastWindowClosed() {
	if a.quitting {
		return
	}
	if a.onWindowAllClosed.len() > 0 {
		fire(&a.onWindowAllClosed)
		return
	}
	a.Quit()
}

func (a *Application) handleReady() {
	a.mu.Lock()
	menuSet := a.menuSet
	a.mu.Unlock()
	var def *Menu
	if !menuSet {
		// Built outside the lock: role labels need App.Name.
		def = defaultMenu()
	}

	a.mu.Lock()
	a.ready = true
	fns := a.whenReady
	a.whenReady = nil
	if !a.menuSet {
		a.menu = def
	}
	menu := a.menu
	a.mu.Unlock()
	close(a.readyCh)

	backend().SetApplicationMenu(menu.snapshot())
	fire(&a.onReady)
	for _, fn := range fns {
		fn()
	}
	// After the windows the app opens when ready, which URLs may target.
	postMain(launchArgs)
}

// waitReady blocks a goroutine other than the main one until the
// application is ready.
func (a *Application) waitReady() { <-a.readyCh }

// Name returns the application name. It defaults to the bundle name of a
// packaged app, else to the executable name.
func (a *Application) Name() string {
	a.mu.Lock()
	name := a.name
	a.mu.Unlock()
	if name != "" {
		return name
	}
	if info, ok := packageInfo(); ok && info.Name != "" {
		return info.Name
	}
	exe, err := os.Executable()
	if err != nil {
		return "MyGo"
	}
	return strings.TrimSuffix(filepath.Base(exe), ".exe")
}

// SetName overrides the application name. Call it before Run.
func (a *Application) SetName(name string) {
	a.mu.Lock()
	a.name = name
	a.mu.Unlock()
}

// Version returns the application version, which defaults to the version of
// the packaged app.
func (a *Application) Version() string {
	a.mu.Lock()
	v := a.version
	a.mu.Unlock()
	if v == "" {
		if info, ok := packageInfo(); ok {
			return info.Version
		}
	}
	return v
}

// SetVersion overrides the application version.
func (a *Application) SetVersion(v string) {
	a.mu.Lock()
	a.version = v
	a.mu.Unlock()
}

// IsPackaged reports whether the application runs from an app bundle
// (created by `mygo dev` and `mygo build`) rather than a plain executable.
func (a *Application) IsPackaged() bool {
	_, ok := packageInfo()
	return ok
}

// SetMenu sets the application menu: the menu bar on macOS and the menu bar
// of windows without their own menu on Linux and Windows. nil removes it.
// Without a call to SetMenu a default menu with the usual App, Edit, View
// and Window items is installed.
func (a *Application) SetMenu(m *Menu) {
	a.mu.Lock()
	a.menu = m
	a.menuSet = true
	ready := a.ready
	a.mu.Unlock()
	if ready {
		onMain(func() { backend().SetApplicationMenu(m.snapshot()) })
	}
}

// Menu returns the application menu.
func (a *Application) Menu() *Menu {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.menu
}

func (a *Application) activationPolicy() ActivationPolicy {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.policy == "" {
		return ActivationPolicyRegular
	}
	return a.policy
}

// SetActivationPolicy changes how the app appears in the Dock (macOS).
func (a *Application) SetActivationPolicy(p ActivationPolicy) {
	a.mu.Lock()
	a.policy = p
	running := a.running
	a.mu.Unlock()
	if running {
		onMain(func() { backend().App().SetActivationPolicy(string(p)) })
	}
}

// Focus brings the application to the foreground.
func (a *Application) Focus() {
	needsApp("App.Focus")
	onMain(func() { backend().App().Activate() })
}

// Hide hides all windows of the application (macOS).
func (a *Application) Hide() {
	needsApp("App.Hide")
	onMain(func() { backend().App().Hide() })
}

// Show shows the application after Hide (macOS).
func (a *Application) Show() {
	needsApp("App.Show")
	onMain(func() { backend().App().Unhide() })
}

// IsHidden reports whether the application is hidden (macOS).
func (a *Application) IsHidden() bool {
	needsApp("App.IsHidden")
	return onMainValue(func() bool { return backend().App().IsHidden() })
}

// SetBadgeCount shows a counter on the application icon. Zero clears it.
func (a *Application) SetBadgeCount(n int) {
	needsApp("App.SetBadgeCount")
	label := ""
	if n > 0 {
		label = strconv.Itoa(n)
	}
	a.Dock.SetBadge(label)
}

// AboutPanelOptions customizes the standard about panel.
type AboutPanelOptions struct {
	ApplicationName    string
	ApplicationVersion string
	// Version is the build version shown next to ApplicationVersion.
	Version   string
	Copyright string
	Credits   string
}

// ShowAboutPanel shows the standard about panel (macOS).
func (a *Application) ShowAboutPanel(opts AboutPanelOptions) {
	needsApp("App.ShowAboutPanel")
	onMain(func() { backend().App().ShowAboutPanel(platform.AboutPanelOptions(opts)) })
}

// OnReady is called once the application has finished launching.
func (a *Application) OnReady(fn func()) (off func()) { return a.onReady.add(fn, false) }

// OnWindowAllClosed is called when the last window has been closed, unless
// the application is quitting. Without listeners the application quits;
// registering one keeps it running (the common choice on macOS).
func (a *Application) OnWindowAllClosed(fn func()) (off func()) {
	return a.onWindowAllClosed.add(fn, false)
}

// OnBeforeQuit is called when a quit starts, before windows are closed.
func (a *Application) OnBeforeQuit(fn func(e *QuitEvent)) (off func()) {
	return a.onBeforeQuit.add(fn, false)
}

// OnWillQuit is called after all windows have been closed, right before the
// event loop stops.
func (a *Application) OnWillQuit(fn func(e *QuitEvent)) (off func()) {
	return a.onWillQuit.add(fn, false)
}

// OnQuit is called after the event loop has stopped.
func (a *Application) OnQuit(fn func()) (off func()) { return a.onQuit.add(fn, false) }

// OnActivate is called when the application is re-activated, for example by
// clicking its Dock icon (macOS). Apps usually open a window when
// hasVisibleWindows is false.
func (a *Application) OnActivate(fn func(hasVisibleWindows bool)) (off func()) {
	return a.onActivate.add(fn, false)
}

// OnDidBecomeActive is called when the application becomes the active one.
func (a *Application) OnDidBecomeActive(fn func()) (off func()) {
	return a.onDidBecomeActive.add(fn, false)
}

// OnDidResignActive is called when the application stops being active.
func (a *Application) OnDidResignActive(fn func()) (off func()) {
	return a.onDidResignActive.add(fn, false)
}

// OnOpenURL is called when the application is asked to open a URL of a
// scheme it handles: one listed in urlSchemes in mygo.config.ts or registered
// with RegisterURLScheme. Register it before Run to receive the URL the app
// was launched with. On Windows and Linux, where a URL starts a new
// instance of the app, use RequestSingleInstanceLock so that the first
// instance gets the URLs of later ones.
func (a *Application) OnOpenURL(fn func(url string)) (off func()) {
	return a.onOpenURL.add(fn, false)
}

// OnOpenFile is called when a file is opened with the application: one of
// the types of fileAssociations in mygo.config.ts opened from the file manager,
// a file dropped on the Dock icon (macOS), or one passed on the command
// line. Register it before Run to receive the files the app was launched
// with; on Windows and Linux, RequestSingleInstanceLock makes the first
// instance get those of later ones.
func (a *Application) OnOpenFile(fn func(path string)) (off func()) {
	return a.onOpenFile.add(fn, false)
}

// OnNotificationClick is called with the ID of every notification of the
// app that the user clicks, after its OnClick listeners: also one of an
// earlier run, which no Notification of this run holds, such as the one
// whose click launched the app (macOS). Register it before Run to receive
// that one. Give notifications IDs of the app's own, such as the
// conversation they are about, to know what to open.
func (a *Application) OnNotificationClick(fn func(id string)) (off func()) {
	return a.onNotificationClick.add(fn, false)
}

// OnWindowCreated is called for every new window.
func (a *Application) OnWindowCreated(fn func(w *Window)) (off func()) {
	return a.onWindowCreated.add(fn, false)
}

// ClearBrowsingData deletes what the app's pages stored: cookies, local
// and session storage, IndexedDB, service workers and caches, e.g. when the
// user signs out. Open pages keep what they hold in memory until they are
// reloaded.
func (a *Application) ClearBrowsingData() error {
	needsApp("App.ClearBrowsingData")
	ch := make(chan error, 1)
	onMain(func() { backend().App().ClearBrowsingData(func(err error) { deliver(ch, err) }) })
	return await(ch)
}

// Dock controls the application's Dock icon (macOS). Its methods do
// nothing on other platforms.
type Dock struct {
	mu   sync.Mutex
	menu *Menu
}

// SetMenu sets the menu the Dock icon shows above the standard items, e.g.
// to open a new window; nil removes it. It may be called before App.Run.
func (d *Dock) SetMenu(m *Menu) {
	d.mu.Lock()
	d.menu = m // its items must stay reachable for clicks
	d.mu.Unlock()
	onMain(func() {
		if App.initialized { // else Run installs it
			// The menu set last: calls may reach the main thread in
			// another order.
			backend().App().SetDockMenu(d.Menu().snapshot())
		}
	})
}

// Menu returns the menu set with SetMenu.
func (d *Dock) Menu() *Menu {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.menu
}

// SetBadge shows text on the Dock icon; "" clears it.
func (d *Dock) SetBadge(text string) {
	needsApp("App.Dock.SetBadge")
	onMain(func() { backend().App().SetBadge(text) })
}

// Badge returns the text shown on the Dock icon.
func (d *Dock) Badge() string {
	needsApp("App.Dock.Badge")
	return onMainValue(func() string { return backend().App().Badge() })
}

// Bounce bounces the Dock icon to request attention and returns an id for
// CancelBounce. Critical bounces continue until the app is activated.
func (d *Dock) Bounce(critical bool) int {
	needsApp("App.Dock.Bounce")
	return onMainValue(func() int { return backend().App().Bounce(critical) })
}

// CancelBounce stops a bounce started with Bounce.
func (d *Dock) CancelBounce(id int) {
	needsApp("App.Dock.CancelBounce")
	onMain(func() { backend().App().CancelBounce(id) })
}

// SetIcon replaces the Dock icon with a PNG image.
func (d *Dock) SetIcon(png []byte) error {
	needsApp("App.Dock.SetIcon")
	return onMainValue(func() error { return backend().App().SetDockIcon(png) })
}

// Hide removes the Dock icon (accessory activation policy).
func (d *Dock) Hide() { App.SetActivationPolicy(ActivationPolicyAccessory) }

// Show restores the Dock icon (regular activation policy).
func (d *Dock) Show() { App.SetActivationPolicy(ActivationPolicyRegular) }

// appHandler receives application events from the backend.
type appHandler struct{}

func (appHandler) Ready()              { App.handleReady() }
func (appHandler) Dispatch()           { loop.drain() }
func (appHandler) QuitRequested() bool { return App.prepareQuit() }
func (appHandler) Terminating()        { App.finish() }
func (appHandler) Activated(visible bool) {
	fire1(&App.onActivate, visible)
}
func (appHandler) DidBecomeActive() { fire(&App.onDidBecomeActive) }
func (appHandler) DidResignActive() { fire(&App.onDidResignActive) }
func (appHandler) OpenURLs(urls []string) {
	for _, u := range urls {
		fire1(&App.onOpenURL, u)
	}
}
func (appHandler) OpenFiles(paths []string) {
	for _, p := range paths {
		fire1(&App.onOpenFile, p)
	}
}
func (appHandler) MenuItemClicked(id int) { menuItemClicked(id, FocusedWindow()) }
func (appHandler) ThemeChanged() {
	updateBackgrounds()
	contentThemeChanged()
	Theme.changed()
	// Pages hear the accent and the settings of accessibility, which the
	// media queries do not tell them.
	sendPreferences()
}
func (appHandler) DisplaysChanged()              { Screen.changed() }
func (appHandler) LocaleChanged()                { App.localeChangedFromSystem() }
func (appHandler) PowerEvent(event string)       { Power.event(event) }
func (appHandler) HotkeyPressed(id int)          { GlobalShortcut.pressed(id) }
func (appHandler) NotificationClicked(id string) { notificationClicked(id) }

func (appHandler) NetworkChanged(s platform.NetworkStatus) { Network.changed(networkStatus(s)) }
