package main

// The window shell's wiring (shell.go draws it): the shell's state, the
// library's services it owns, navigation, the window, and the frame's
// composition of the shell's view state with every area's dialogs.

import (
	"context"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync/atomic"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/logs"
	"github.com/sgl-project/sglang-omni/Voxt/internal/notify"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// shellState is the shell's view state and the services it owns.
type shellState struct {
	router         *uiRouter
	openingStarted time.Time
	// nav is the sidebar's choice; navShown the router path it was last
	// synced from (see syncNav).
	nav, navShown string
	lastSettings  string // settings section shown last
	// The settings dialog (0.3): open, and the router of its sections
	// ("/general"), which navigate drives from "/settings/…" paths.
	settingsOpen bool
	setRouter    *uiRouter
	// mainUI is the main window's services, kept by its build: read
	// between frames (textFocused).
	mainUI uiServices

	bus      *core.Bus
	settings settings.Service
	prefs    settings.Settings // the settings, as this thread last saw them
	notes    *notify.Center
	logs     *logs.Ring

	noticesState // shell_notices_controller.go

	version string // the app's, beside Settings in the sidebar; "" in tests

	paletteOpen  bool
	paletteQuery string
	paletteSel   int
	sheetOpen    bool
	noticesOpen  bool // the rail's popover of notifications
	logsPanel    bool
	logsSource   string
	logsQuery    string
	logsHeight   float32
	logsDirty    atomic.Bool // a redraw for new lines is on its way

	logsState     // shell_logs_controller.go
	navState      // shell_nav_controller.go
	commandsState // shell_commands_controller.go
	shellUI       shellLocalUI
	// shellViewModel is the frame's aggregate, reused between frames
	// (shellModel).
	shellViewModel shellViewState

	menuLocale string // locale the native menu was built in
	hasMenu    bool   // the app runs (not a test): it has a menu bar to build
}

func newShellState() shellState {
	return shellState{
		router:       newShellRouter("/home", true),
		setRouter:    newShellRouter("/general", false),
		lastSettings: "general",
		logsHeight:   220,
		prefs:        settings.Defaults(),
	}
}

// openShell wires the shell to the library: its bus, the settings it
// keeps in the database, and the notification centre, which keeps info
// and warnings there too (document "notifications"). Logs stay in memory
// until main's wiring (connectShell) adds the files.
func (a *app) openShell(lib *library) {
	a.bus = lib.bus
	a.settings = lib.settings
	a.prefs = lib.settings.Get()
	notes, err := notify.Open(context.Background(), lib.store.Documents(), a.bus)
	if err != nil {
		log.Printf("notifications: %v", err)
	}
	a.notes = notes
	a.notices, _ = notes.List(context.Background())
	a.logs = logs.New(logs.Options{Bus: a.bus})
	a.loadSettingsForm()

	core.On(a.bus, func(ev core.NotificationPosted) {
		a.post(func() { a.noticePosted(ev.Notification) })
	})
	core.On(a.bus, func(ev settings.Changed) {
		a.post(func() { a.settingsChanged(ev.Old, ev.New) })
	})
	core.On(a.bus, func(core.LogAppended) {
		// Many lines may come at once: one redraw for all of them.
		if a.logsDirty.CompareAndSwap(false, true) {
			a.post(func() { a.logsDirty.Store(false) })
		}
	})
	a.applyLocale()
}

// connectShell adds what only the real app has: log files in logDir (the
// Go log package writes there too), the app's version, the system
// language and deep links. main calls it before App.Run.
func (a *app) connectShell(logDir string) {
	a.logDir = logDir // Settings → Storage, Diagnostics
	if logDir != "" {
		a.logs.Close()
		a.logs = logs.New(logs.Options{Dir: logDir, Bus: a.bus})
		log.SetOutput(io.MultiWriter(os.Stderr, a.logs.Writer("app")))
		keepCrashes(logDir)
	}
	mygo.App.OnOpenURL(func(u string) { a.post(func() { a.openURL(u) }) })
}

// keepCrashes has the Go runtime write a crash (a panic, a fatal error)
// to crash.log in logDir: macOS keeps no report of a Go crash, and an app
// opened from the Finder has no stderr to read. The crash of the run
// before moves to crash.log.1 and the log says so. The diagnostic bundle
// takes both (writeBundle).
func keepCrashes(logDir string) {
	path := filepath.Join(logDir, "crash.log")
	if fi, err := os.Stat(path); err == nil && fi.Size() > 0 {
		os.Rename(path, path+".1")
		log.Printf("warning: Voxt crashed last time; what it was is in %s", path+".1")
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		log.Printf("crash log: %v", err)
		return
	}
	defer f.Close() // the runtime keeps its own copy
	if err := debug.SetCrashOutput(f, debug.CrashOptions{}); err != nil {
		log.Printf("crash log: %v", err)
	}
}

// closeShell waits for the notifications to be saved and flushes the
// logs; it runs on quit.
func (a *app) closeShell() {
	a.notes.Close()
	a.logs.Close()
}

// navigate goes to a router path ("/transcribe"); it does nothing when that
// page shows already. "/settings/general" opens the settings dialog at
// that section, over the page, and "/settings" at the section shown last;
// any other path closes it.
func (a *app) navigate(path string) {
	if to, ok := movedPaths[path]; ok {
		path = to
	}
	if path == "/settings" || path == "/settings/" {
		path = "/settings/" + a.lastSettings
	}
	if rest, ok := strings.CutPrefix(path, "/settings/"); ok && rest != "" {
		a.lastSettings, _, _ = strings.Cut(rest, "/")
		a.openSettings("/" + rest)
		return
	}
	a.settingsOpen = false
	if path == a.router.Path() {
		return
	}
	a.router.Push(path)
	a.pageShown(path) // pages.go
}

// currentPage is the name of the page shown: the first part of its path.
func (a *app) currentPage() string { return section(a.router.Path())[1:] }

func (a *app) startOpening() {
	if a.prefs.Setup.Done || a.prefs.Setup.OpeningShown {
		return
	}
	a.openingStarted = time.Now()
	a.updateSettings(func(state *settings.Settings) { state.Setup.OpeningShown = true })
}

func (a *app) openingModel(now time.Time, systemReducedMotion bool) (openingViewState, bool) {
	if a.openingStarted.IsZero() {
		return openingViewState{}, false
	}
	elapsed := now.Sub(a.openingStarted)
	if elapsed >= openingDuration || a.prefs.Appearance.ReduceMotion || systemReducedMotion {
		a.openingStarted = time.Time{}
		return openingViewState{}, false
	}
	return buildOpeningViewState(elapsed), true
}

// openWindow opens the main window; it runs in App.WhenReady. Opened
// at login, with the menu bar icon and "Keep running when the window
// closes" on, VoxtNew starts in the menu bar: the window waits hidden.
func (a *app) openWindow() {
	a.version = mygo.App.Version()
	a.systemLocale = systemLanguage() // settings_locale.go
	a.hasMenu = true
	a.applyLocale() // the system's language, unless the settings name one
	a.buildMenu()
	a.startOpening()
	a.win = mygo.NewWindow(mygo.WindowOptions{
		Title:          "Voxt",
		Width:          windowWidth,
		Height:         windowHeight,
		MinWidth:       windowMinWidth,
		MinHeight:      windowMinHeight,
		StateKey:       "main",
		TitleBarStyle:  mygo.TitleBarHiddenInset,
		TitleBarHeight: titleBarHeight,
		Transparent:    true,
		Content:        shellContent(a.view),
		Hidden:         startHidden(mygo.App.WasOpenedAtLogin(), a.prefs.General),
	})
	a.life = a.win
	a.maybeOnboard()        // onboarding.go: the setup guide, on the first start
	a.offerLegacy()         // legacy_import.go: the previous Voxt's data, for who set up without the guide's card
	a.legacyModelsAtStart() // legacy_models.go: the previous Voxt's speech models this version added
	a.legacyDockAtStart()   // legacy_settings_general.go: Voxt 1.x's Show in Dock, which 2.0.0 and 2.0.1 did not read
	a.syncOpenAtLogin()     // shell_settings.go
}

// startHidden says whether the window waits hidden at start: opened at
// login, when VoxtNew stays in the menu bar.
func startHidden(atLogin bool, g settings.General) bool {
	return atLogin && g.MenuBar && g.CloseToTray
}

// shellModel is the frame's shell view state: the theme, the drained
// toasts, and the cached sub-states (the sidebar, the notifications, the
// logs panel, the palette, the sheet). The palette's and the sheet's
// models build only while their dialog shows: closed, DialogBase and
// Modal never look at them, and the registry and the sheet's rows stay
// off the frame. The aggregate is reused, so a quiet frame allocates
// nothing here.
func (a *app) shellModel(now time.Time) *shellViewState {
	model := &a.shellViewModel
	*model = shellViewState{
		Theme: a.prefs.Appearance.Theme, AppearancePalette: paletteSnapshot(a.prefs.Appearance.Palette), Scale: a.prefs.Appearance.Scale,
		Still:  a.win == nil || a.prefs.Appearance.ReduceMotion,
		Toasts: a.toasts,
		Nav:    a.navModel(), Notices: a.noticesModel(now),
		Palette: &a.paletteView, Sheet: &a.sheetView,
		LogsHeight:   a.logsHeight,
		SettingsOpen: a.settingsOpen,
	}
	a.toasts = nil
	if a.logsPanel && section(a.router.Path()) != "/logs" {
		model.Logs = a.logsModel()
	}
	if a.paletteOpen {
		model.Palette = a.paletteModel()
	}
	if a.sheetOpen {
		model.Sheet = a.sheetModel()
	}
	return model
}

// shellLocal returns the shell views' local state, wired to the app's on
// first use.
func (a *app) shellLocal() *shellLocalUI {
	if a.shellUI.nav == nil {
		a.shellUI = shellLocalUI{
			nav: a.navLocal(), logs: a.logsLocal(), palette: a.paletteLocal(),
			sheetOpen: &a.sheetOpen, announcementOpen: &a.announceShownOpen,
		}
	}
	return &a.shellUI
}

// view builds a frame: the state handoffs (the window's services, the
// keys, the sidebar's sync), the shell's view over its view state, and
// each area's dialogs.
func (a *app) view(c *uiContext) {
	a.mainUI = c.Services()
	if opening, shown := a.openingModel(time.Now(), c.Preferences().ReduceMotion); shown {
		openingView(c, opening)
		return
	}
	shellShortcuts(c, a.navModel(), a, &a.sheetOpen)
	a.syncNav()
	now := time.Now()
	model := a.shellModel(now)
	shellView(c, model, a, a.shellLocal(), func() { a.pages(c) }, func() {
		settingsDialog(c, a.settingsDialogModel(now), a, settingsDialogLocalUI{
			Open: &a.settingsOpen, Router: a.setRouter, Form: &a.settingsForm,
			HistoryAccess: &a.histAccessUI, Storage: &a.storageState.settingsStorageLocalUI,
			Updates: &a.updUI, Models: &a.modelsLocalUI,
			Remote: remoteLocalUI{wait: &a.llmWaitLocalUI, dialog: &a.provDlg.providerDialogLocalUI},
		})
		// The Speech and Language models sections' dialogs, over Settings.
		models := a.modelsPageModel()
		modelDeleteDialogs(c, &models.Delete, a, &a.modelsLocalUI) // models_delete.go
		networkDialog(c, &models.Network, a, &a.modelsLocalUI)     // models_network.go
		if dialog := a.remoteDialogModel(); dialog != nil {
			remoteProviderDialogView(c, dialog, a, &a.provDlg.providerDialogLocalUI) // remote_dialog.go
		}
		if a.ob.open {
			obState := a.onboardingViewState()
			onboardingView(c, obState, a, &onboardingLocalUI{open: obState.Open})
		}
		shortcutPromptDialog(c, a.shortcutPromptModel(), a) // settings_shortcuts_family.go
		announcementDialog(c, a.announcementModel(now), &a.announceShownOpen)
		m, u := a.releaseDialogModel(), &a.updUI
		releaseDialog(c, m, a, u) // updates_view.go
		a.legacyReportDialog(c)   // legacy_report.go
	})
	a.saveClosedSections()
	// A shortcut field took the focus or lost it.
	if a.keysHeard != a.keysListening {
		a.keysListening = a.keysHeard
		a.syncHotkeys()
	}
	a.keysHeard = false
}

// syncNav makes the sidebar follow the router (Back, deep links, ⌘1…),
// but not undo a choice in the sidebar the router has not seen yet.
func (a *app) syncNav() {
	if p := section(a.router.Path()); p != a.navShown {
		a.nav, a.navShown = p, p
	}
}

// showFatal explains, in a dialog, why VoxtNew cannot start (its library
// did not open), then quits. It runs instead of the app.
func showFatal(err error) {
	log.Print(err)
	mygo.App.WhenReady(func() {
		showLanguage(i18n.SetLocale(systemLanguage())) // settings_locale.go
		go func() {
			dataDir, _ := mygo.App.Path(mygo.PathUserData)
			res, _ := mygo.Dialog.Message(mygo.MessageOptions{
				Type:    mygo.MessageError,
				Title:   "Voxt",
				Message: i18n.T("Voxt cannot open its library"),
				Detail: i18n.T("Your history, dictionary and settings are kept in a database in %s. It could not be opened: %v", dataDir, err) +
					"\n\n" + i18n.T("If a newer Voxt created it, use that version. Nothing was changed."),
				Buttons: []string{i18n.T("Quit"), i18n.T("Show folder")},
			})
			if res.Button == 1 {
				mygo.Shell.OpenPath(dataDir)
			}
			mygo.App.Quit()
		}()
	})
	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}
