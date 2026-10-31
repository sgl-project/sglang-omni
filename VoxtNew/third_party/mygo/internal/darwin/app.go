//go:build darwin

package darwin

import (
	"strings"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"

	"github.com/egoist/mygo/internal/platform"
)

// Backend is the macOS implementation of platform.Backend.
type Backend struct {
	h        platform.AppHandler
	app      id
	delegate id
	source   uintptr
	mainLoop uintptr

	// Lookup tables for Objective-C callbacks. Main thread only.
	byDelegate map[id]*window
	byWebView  map[id]*window
	byNSWindow map[id]*window
	bySurface  map[id]*surface
	byAccess   map[id]*accessElement

	menuTarget id
	menuItems  map[int][]id
	appMenu    []int

	trays       map[id]*tray
	hotkeys     map[int]uintptr
	hkInstalled bool

	notifyDelegate id
	// notifyAnswered records that the user has answered whether the app
	// may show notifications; notifyWaiting holds those shown meanwhile.
	notifyAnswered bool
	notifyWaiting  []waitingNotification

	stepping       int
	quitAfterModal bool

	dockProgress dockProgress
	// openedAtLogin records whether the system started the app as a login
	// item (WasOpenedAtLogin).
	openedAtLogin bool
	// dockMenu is returned by applicationDockMenu: (retained).
	dockMenu      id
	dockMenuOwner int
}

// New creates the macOS backend.
func New() *Backend {
	load()
	return &Backend{
		byDelegate: map[id]*window{},
		byWebView:  map[id]*window{},
		byNSWindow: map[id]*window{},
		bySurface:  map[id]*surface{},
		byAccess:   map[id]*accessElement{},
		menuItems:  map[int][]id{},
		trays:      map[id]*tray{},
		hotkeys:    map[int]uintptr{},
	}
}

var theBackend *Backend

func (b *Backend) Name() string { return "darwin/wkwebview" }

func (b *Backend) IsMainThread() bool {
	r, _, _ := purego.SyscallN(mainNPFn)
	return int32(r) != 0
}

const (
	nsApplicationActivationPolicyRegular    = 0
	nsApplicationActivationPolicyAccessory  = 1
	nsApplicationActivationPolicyProhibited = 2
)

func activationPolicy(p string) uintptr {
	switch p {
	case "accessory":
		return nsApplicationActivationPolicyAccessory
	case "prohibited":
		return nsApplicationActivationPolicyProhibited
	}
	return nsApplicationActivationPolicyRegular
}

func (b *Backend) Init(h platform.AppHandler, opts platform.AppOptions) error {
	b.h = h
	theBackend = b
	registerClasses()

	b.app = send(class("NSApplication"), "sharedApplication")
	send(b.app, "setActivationPolicy:", activationPolicy(opts.ActivationPolicy))
	// No automatic "Show Tab Bar" items and window tabbing.
	send(class("NSWindow"), "setAllowsAutomaticWindowTabbing:", 0)

	b.delegate = alloc("MyGoAppDelegate")
	send(b.app, "setDelegate:", uintptr(b.delegate))
	b.menuTarget = alloc("MyGoMenuTarget")
	b.attachNotificationDelegate()

	// A run loop source drives the queue of functions posted from other
	// goroutines. It fires in every run loop mode, including while menus
	// are tracked, windows are resized and modal dialogs run.
	b.mainLoop = cfRunLoopGetMain()
	var ctx [10]uintptr // CFRunLoopSourceContext; perform is the last field
	ctx[9] = dispatchCallback
	b.source = cfRunLoopSourceCreate(0, 0, unsafe.Pointer(&ctx[0]))
	cfRunLoopAddSource(b.mainLoop, b.source, kCFRunLoopCommonModes)

	// Appearance (dark mode) and display configuration changes.
	send(b.app, "addObserver:forKeyPath:options:context:", uintptr(b.delegate), uintptr(nsString("effectiveAppearance")), 1, 0)
	center := send(class("NSNotificationCenter"), "defaultCenter")
	send(center, "addObserver:selector:name:object:", uintptr(b.delegate), uintptr(sel("screensChanged:")),
		uintptr(nsString("NSApplicationDidChangeScreenParametersNotification")), 0)
	// The accent color, and the display settings of accessibility.
	send(center, "addObserver:selector:name:object:", uintptr(b.delegate), uintptr(sel("preferencesChanged:")),
		uintptr(nsString("NSSystemColorsDidChangeNotification")), 0)
	send(send(workspace(), "notificationCenter"), "addObserver:selector:name:object:", uintptr(b.delegate), uintptr(sel("preferencesChanged:")),
		uintptr(nsString("NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification")), 0)
	// The languages and regional formats of the user: Foundation tells of
	// a new current locale, and System Settings of new languages, a new
	// time and date format, which the current locale may not change.
	send(center, "addObserver:selector:name:object:", uintptr(b.delegate), uintptr(sel("localeChanged:")),
		uintptr(nsString("NSCurrentLocaleDidChangeNotification")), 0)
	distributed := send(class("NSDistributedNotificationCenter"), "defaultCenter")
	for _, name := range []string{"AppleLanguagePreferencesChangedNotification", "AppleTimePreferencesChangedNotification", "AppleDatePreferencesChangedNotification"} {
		send(distributed, "addObserver:selector:name:object:", uintptr(b.delegate), uintptr(sel("localeChanged:")), uintptr(nsString(name)), 0)
	}
	return nil
}

var dispatchCallback = purego.NewCallback(func(info uintptr) {
	withPool(func() {
		drainMainQueue()
		theBackend.h.Dispatch()
	})
})

func (b *Backend) Run() error {
	send(b.app, "run")
	return nil
}

func (b *Backend) Quit() {
	if send(b.app, "modalWindow") != 0 {
		// stop: would end the modal session (an app-modal dialog) rather
		// than the event loop: abort the session, and stop once it has
		// unwound (see runPanel).
		b.quitAfterModal = true
		send(b.app, "abortModal")
		return
	}
	send(b.app, "stop:", 0)
	// stop: takes effect after the next event; make sure there is one.
	b.postEvent(0)
}

func (b *Backend) Signal() {
	if b.source != 0 {
		purego.SyscallN(cfRunLoopSourceSignalFn, b.source)
		purego.SyscallN(cfRunLoopWakeUpFn, b.mainLoop)
	}
}

const (
	nsEventTypeApplicationDefined = 15
	wakeSubtype                   = 0x6d67 // 'mg'
)

func (b *Backend) postEvent(subtype int16) {
	withPool(func() {
		ev := msgOtherEvent(class("NSEvent"), sel("otherEventWithType:location:modifierFlags:timestamp:windowNumber:context:subtype:data1:data2:"),
			nsEventTypeApplicationDefined, NSPoint{}, 0, 0, 0, 0, subtype, 0, 0)
		send(b.app, "postEvent:atStart:", uintptr(ev), 1)
	})
}

func (b *Backend) Step() {
	b.stepping++
	defer func() { b.stepping-- }()
	withPool(func() {
		until := msgDateSince(class("NSDate"), sel("dateWithTimeIntervalSinceNow:"), 0.5)
		mode := nsString("kCFRunLoopDefaultMode")
		ev := msgNextEvent(b.app, sel("nextEventMatchingMask:untilDate:inMode:dequeue:"), ^uint64(0), until, mode, true)
		if ev == 0 {
			return
		}
		if sendInt(ev, "type") == nsEventTypeApplicationDefined && int16(sendInt(ev, "subtype")) == wakeSubtype {
			return
		}
		send(b.app, "sendEvent:", uintptr(ev))
	})
}

// Wake makes a pending Step return. It is called from any goroutine.
func (b *Backend) Wake() {
	if b.app == 0 {
		return
	}
	b.postEvent(wakeSubtype)
}

func (b *Backend) App() platform.AppController   { return appController{b} }
func (b *Backend) Dialogs() platform.Dialogs     { return dialogs{b} }
func (b *Backend) Clipboard() platform.Clipboard { return clipboard{} }
func (b *Backend) Shell() platform.Shell         { return shell{} }
func (b *Backend) Screen() platform.Screen       { return screen{} }
func (b *Backend) Theme() platform.Theme         { return theme{b} }
func (b *Backend) Power() platform.Power         { return power{b} }
func (b *Backend) Secrets() platform.Secrets     { return secrets{} }

var classesOnce bool

func registerClasses() {
	if classesOnce {
		return
	}
	classesOnce = true
	registerAppDelegate()
	registerWindowClasses()
	registerSurfaceClass()
	registerClipboardClass()
	registerAccessClass()
	registerMenuTarget()
	registerSchemeHandler()
	registerTrayTarget()
	registerNotificationDelegate()
}

const (
	nsTerminateCancel = 0
	nsTerminateNow    = 1

	kCoreEventClass    = 0x61657674 // 'aevt'
	kAEQuitApplication = 0x71756974 // 'quit'
	keyErrorNumber     = 0x6572726E // 'errn'
	userCanceledErr    = -128
)

func registerAppDelegate() {
	classDef("MyGoAppDelegate", "NSObject", []string{"NSApplicationDelegate"}, []objc.MethodDef{
		// The end of a PrintToPDF job (printJobs). WebKit's print operations
		// may finish on a background thread.
		method("mygoPrintOperationDidRun:success:contextInfo:", func(self id, _ objc.SEL, op id, success bool, job uintptr) {
			theBackend.runOnMain(func() {
				if j := printJobs[job]; j != nil {
					delete(printJobs, job)
					j.done(success)
				}
			})
		}),
		method("applicationWillFinishLaunching:", func(self id, _ objc.SEL, n id) {
			// Handle the quit Apple Event (Dock > Quit, AppleScript, logout)
			// ourselves so the sender gets a proper reply while Run still
			// returns normally. AppKit installs its default handlers right
			// before this notification.
			aem := send(class("NSAppleEventManager"), "sharedAppleEventManager")
			send(aem, "setEventHandler:andSelector:forEventClass:andEventID:", uintptr(self),
				uintptr(sel("mygoHandleQuitEvent:withReplyEvent:")), kCoreEventClass, kAEQuitApplication)
			theBackend.openedAtLogin = launchedAsLoginItem()
		}),
		method("applicationDidFinishLaunching:", func(self id, _ objc.SEL, n id) {
			b := theBackend
			if send(b.app, "activationPolicy") == nsApplicationActivationPolicyRegular {
				// Plain executables (go run) do not come to the front on
				// their own.
				send(b.app, "activateIgnoringOtherApps:", 1)
			}
			b.h.Ready()
		}),
		// terminate: (Cmd+Q and the Quit menu item) ends up here. Rather
		// than letting AppKit exit the process, stop the run loop so Run
		// returns and deferred cleanup in main runs.
		method("applicationShouldTerminate:", func(self id, _ objc.SEL, app id) uint {
			b := theBackend
			if b.h.QuitRequested() {
				b.Quit()
			}
			return nsTerminateCancel
		}),
		// Power and session notifications (Backend.Power().Watch).
		method("mygoPowerNotification:", func(self id, _ objc.SEL, n id) {
			if event := powerEvents[goString(send(n, "name"))]; event != "" {
				theBackend.h.PowerEvent(event)
			}
		}),
		method("mygoHandleQuitEvent:withReplyEvent:", func(self id, _ objc.SEL, event, reply id) {
			b := theBackend
			if b.h.QuitRequested() {
				b.Quit()
				return
			}
			if reply != 0 {
				canceled := int32(userCanceledErr)
				code := send(class("NSAppleEventDescriptor"), "descriptorWithInt32:", uintptr(uint32(canceled)))
				send(reply, "setParamDescriptor:forKeyword:", uintptr(code), keyErrorNumber)
			}
		}),
		method("applicationWillTerminate:", func(self id, _ objc.SEL, n id) {
			theBackend.h.Terminating()
		}),
		method("applicationDockMenu:", func(self id, _ objc.SEL, app id) id {
			return theBackend.dockMenu
		}),
		method("applicationShouldHandleReopen:hasVisibleWindows:", func(self id, _ objc.SEL, app id, visible bool) bool {
			theBackend.h.Activated(visible)
			return true
		}),
		method("applicationDidBecomeActive:", func(self id, _ objc.SEL, n id) {
			theBackend.h.DidBecomeActive()
		}),
		method("applicationDidResignActive:", func(self id, _ objc.SEL, n id) {
			theBackend.h.DidResignActive()
		}),
		method("application:openURLs:", func(self id, _ objc.SEL, app id, urls id) {
			var files, links []string
			for _, u := range arrayItems(urls) {
				if sendBool(u, "isFileURL") {
					files = append(files, goString(send(u, "path")))
				} else {
					links = append(links, goString(send(u, "absoluteString")))
				}
			}
			if len(files) > 0 {
				theBackend.h.OpenFiles(files)
			}
			if len(links) > 0 {
				theBackend.h.OpenURLs(links)
			}
		}),
		method("applicationSupportsSecureRestorableState:", func(self id, _ objc.SEL, app id) bool { return true }),
		method("applicationShouldTerminateAfterLastWindowClosed:", func(self id, _ objc.SEL, app id) bool { return false }),
		method("observeValueForKeyPath:ofObject:change:context:", func(self id, _ objc.SEL, keyPath, object, change id, ctx uintptr) {
			if goString(keyPath) == "effectiveAppearance" {
				theBackend.h.ThemeChanged()
			}
		}),
		method("screensChanged:", func(self id, _ objc.SEL, n id) {
			theBackend.h.DisplaysChanged()
		}),
		method("preferencesChanged:", func(self id, _ objc.SEL, n id) {
			theBackend.h.ThemeChanged()
		}),
		method("localeChanged:", func(self id, _ objc.SEL, n id) {
			theBackend.h.LocaleChanged()
		}),
	})
}

type appController struct{ b *Backend }

func (a appController) SetActivationPolicy(p string) {
	send(a.b.app, "setActivationPolicy:", activationPolicy(p))
	if p == "regular" {
		send(a.b.app, "activateIgnoringOtherApps:", 1)
	}
}

func (a appController) Activate()      { send(a.b.app, "activateIgnoringOtherApps:", 1) }
func (a appController) Hide()          { send(a.b.app, "hide:", 0) }
func (a appController) Unhide()        { send(a.b.app, "unhide:", 0) }
func (a appController) IsHidden() bool { return sendBool(a.b.app, "isHidden") }

func (a appController) SetBadge(label string) {
	tile := send(a.b.app, "dockTile")
	var s id
	if label != "" {
		s = nsString(label)
	}
	send(tile, "setBadgeLabel:", uintptr(s))
}

func (a appController) Badge() string {
	return goString(send(send(a.b.app, "dockTile"), "badgeLabel"))
}

func (a appController) Bounce(critical bool) int {
	kind := 10 // NSInformationalRequest
	if critical {
		kind = 0 // NSCriticalRequest
	}
	return sendInt(a.b.app, "requestUserAttention:", uintptr(kind))
}

func (a appController) CancelBounce(requestID int) {
	send(a.b.app, "cancelUserAttentionRequest:", uintptr(requestID))
}

func (a appController) SetDockMenu(m *platform.Menu) {
	b := a.b
	if b.dockMenuOwner == 0 {
		b.dockMenuOwner = newOwner()
	}
	withPool(func() {
		dropOwner(b.dockMenuOwner)
		release(b.dockMenu)
		b.dockMenu = 0
		if m != nil {
			b.dockMenu = retain(b.buildMenu(m, "", b.dockMenuOwner))
		}
	})
}

func (a appController) SetDockIcon(png []byte) error {
	var img id
	if len(png) > 0 {
		img = send(send(class("NSImage"), "alloc"), "initWithData:", uintptr(nsData(png)))
		if img == 0 {
			return errInvalidImage
		}
		defer release(img)
	}
	send(a.b.app, "setApplicationIconImage:", uintptr(img))
	return nil
}

func (a appController) ShowAboutPanel(o platform.AboutPanelOptions) {
	dict := send(class("NSMutableDictionary"), "dictionary")
	set := func(key, value string) {
		if value != "" {
			send(dict, "setObject:forKey:", uintptr(nsString(value)), uintptr(nsString(key)))
		}
	}
	set("ApplicationName", o.ApplicationName)
	set("ApplicationVersion", o.ApplicationVersion)
	set("Version", o.Version)
	set("Copyright", o.Copyright)
	if o.Credits != "" {
		credits := autorelease(send(send(class("NSAttributedString"), "alloc"), "initWithString:", uintptr(nsString(o.Credits))))
		send(dict, "setObject:forKey:", uintptr(credits), uintptr(nsString("Credits")))
	}
	send(a.b.app, "activateIgnoringOtherApps:", 1)
	send(a.b.app, "orderFrontStandardAboutPanelWithOptions:", uintptr(dict))
}

func (a appController) Package() (platform.PackageInfo, bool) {
	var info platform.PackageInfo
	ok := false
	withPool(func() {
		bundle := send(class("NSBundle"), "mainBundle")
		path := goString(send(bundle, "bundlePath"))
		if !strings.HasSuffix(path, ".app") {
			return
		}
		ok = true
		get := func(key string) string {
			return goString(send(bundle, "objectForInfoDictionaryKey:", uintptr(nsString(key))))
		}
		info.Name = get("CFBundleDisplayName")
		if info.Name == "" {
			info.Name = get("CFBundleName")
		}
		info.Version = get("CFBundleShortVersionString")
		info.Identifier = goString(send(bundle, "bundleIdentifier"))
	})
	return info, ok
}

func (a appController) ClearBrowsingData(done func(error)) {
	withPool(func() {
		store := send(class("WKWebsiteDataStore"), "defaultDataStore")
		types := send(class("WKWebsiteDataStore"), "allWebsiteDataTypes")
		blk := newBlock(func(_ objc.Block) { a.b.runOnMain(func() { done(nil) }) })
		send(store, "removeDataOfTypes:modifiedSince:completionHandler:", uintptr(types), uintptr(send(class("NSDate"), "distantPast")), uintptr(blk))
		blk.Release()
	})
}

// AddRecentDocument and ClearRecentDocuments use the shared document
// controller, as Electron does: AppKit lists its recent documents in the
// Dock menu, and in an Open Recent menu.
func (a appController) AddRecentDocument(path string) error {
	withPool(func() {
		send(send(class("NSDocumentController"), "sharedDocumentController"), "noteNewRecentDocumentURL:", uintptr(fileURL(path)))
	})
	return nil
}

func (a appController) ClearRecentDocuments() error {
	withPool(func() {
		send(send(class("NSDocumentController"), "sharedDocumentController"), "clearRecentDocuments:", 0)
	})
	return nil
}

func (appController) SetJumpList(platform.JumpList) error { return nil }

// WebViewAvailable reports whether a window showing a web page can have
// one. WKWebView is part of macOS, so this holds wherever the app runs.
func (b *Backend) WebViewAvailable() bool { return hasClass("WKWebView") }
