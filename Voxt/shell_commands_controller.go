package main

// Commands: one registry behind the ⌘K palette, the native menu and the
// window's shortcuts, plus voxtnew:// deep links. (From AudioSmith.) The
// palette and the "?" sheet are shell_commands.go's views.

import (
	"net/url"
	"slices"
	"strings"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// command is something the user can do from anywhere.
type command struct {
	id      string
	group   string // translated
	title   string // translated
	en      string // English title, so English words find it in any language
	keys    string // shortcut as shown, "⌘1"
	run     func()
	enabled bool
}

// commandsState is the commands' part of the app (main thread): the
// registry's and the view states' caches.
type commandsState struct {
	paletteUI paletteLocalUI // the palette's local state

	// list is the last built registry, listKey its inputs (commandList).
	list    []command
	listKey commandsCacheKey
	// paletteView is the last built palette state, paletteViewKey its
	// inputs (paletteModel); sheetView and sheetViewKey the sheet's
	// (sheetModel).
	paletteView    paletteViewState
	paletteViewKey paletteCacheKey
	sheetView      sheetViewState
	sheetViewKey   sheetCacheKey
}

// commandsCacheKey is every input of the command registry: the settings
// (hidden pages, theme, language), the language of the titles, and the
// updater's presence.
type commandsCacheKey struct {
	settings uint64
	locale   string
	hasUpd   bool
}

// paletteCacheKey is every input of a paletteViewState.
type paletteCacheKey struct {
	list  commandsCacheKey
	query string
}

// sheetCacheKey is every input of a sheetViewState.
type sheetCacheKey struct {
	settings uint64
	locale   string
}

// commandList is the registry, rebuilt when the settings, the language
// or the updater's presence changed.
func (a *app) commandList() []command {
	key := commandsCacheKey{settings: a.settingsRevision, locale: i18n.Locale(), hasUpd: a.upd != nil}
	if key != a.listKey {
		a.list = a.commands()
		a.listKey = key
	}
	return a.list
}

// commands lists the commands, in the palette's order.
func (a *app) commands() []command {
	nav := i18n.T("Go to")
	var cmds []command
	add := func(id, group, title, en, keys string, enabled bool, run func()) {
		cmds = append(cmds, command{id: id, group: group, title: title, en: en, keys: keys, run: run, enabled: enabled})
	}
	for _, p := range pages { // every sidebar page (shell_pages.go navGroups)
		if !a.pageHidden(p.path) {
			add("go"+p.path, nav, pageTitle(p.path), p.title, pageKeys(p.path), true, func() { a.navigate(p.path) })
		}
	}
	add("go/settings", nav, i18n.T("Settings"), "Settings", "⌘,", true, func() { a.navigate("/settings") })
	for _, s := range settingsSections() {
		id := "go/settings/" + s.id
		switch s.id {
		case "models":
			id = "go/models"
		case "language-models":
			id = "go/language-models"
		}
		add(id, i18n.T("Settings"), s.title, s.en, "", true, func() { a.navigate("/settings/" + s.id) })
	}

	view := i18n.T("View")
	add("palette", view, i18n.T("Command palette"), "Command palette", "⌘K", true, a.openPalette)
	add("logs-panel", view, i18n.T("Show or hide the logs panel"), "Show or hide the logs panel", "⇧⌘L", true, a.toggleLogsPanel)
	add("shortcuts", view, i18n.T("Keyboard shortcuts"), "Keyboard shortcuts", "?", true, func() { a.sheetOpen = true })
	add("helper-restart", view, i18n.T("Restart helper"), "Restart helper", "", true, a.restartHelper)
	if a.upd != nil {
		add("check-updates", view, i18n.T("Check for Updates…"), "Check for Updates", "", true, a.checkForUpdates)
	}
	add("llm-debug", view, i18n.T("Language model debug"), "Language model debug prompts", "", true, func() { a.navigate(llmDebugPath) })
	add("setup-guide", view, i18n.T("Open setup guide"), "Open setup guide onboarding", "", true, a.openOnboarding)

	look := i18n.T("Appearance")
	for _, th := range []struct{ id, title, en string }{
		{settings.ThemeSystem, i18n.T("Theme: follow the system"), "Theme: follow the system"},
		{settings.ThemeDark, i18n.T("Theme: dark"), "Theme: dark"},
		{settings.ThemeLight, i18n.T("Theme: light"), "Theme: light"},
		{settings.ThemeTransparent, i18n.T("Theme: transparent"), "Theme: transparent"},
	} {
		add("theme/"+th.id, look, th.title, th.en, "", a.prefs.Appearance.Theme != th.id, func() {
			a.setTheme(th.id)
		})
	}
	lang := i18n.T("Language")
	for _, tag := range append([]string{""}, i18n.Supported()...) {
		title := i18n.T("Follow the system language")
		if tag != "" {
			title = i18n.Name(tag)
		}
		add("lang/"+tag, lang, title, "Language "+tag, "", a.prefs.General.Language != tag, func() {
			a.updateSettings(func(s *settings.Settings) { s.General.Language = tag })
		})
	}
	return cmds
}

// matches reports whether every word of q is in the command's words.
func (cmd command) matches(q string) bool {
	hay := strings.ToLower(cmd.title + " " + cmd.group + " " + cmd.en + " " + cmd.id)
	for _, w := range strings.Fields(strings.ToLower(q)) {
		if !strings.Contains(hay, w) {
			return false
		}
	}
	return true
}

func (a *app) paletteItems() []command {
	var out []command
	for _, cmd := range a.commands() {
		if cmd.enabled && cmd.matches(a.paletteQuery) {
			out = append(out, cmd)
		}
	}
	return out
}

func (a *app) openPalette() {
	if !a.paletteOpen {
		a.paletteOpen, a.paletteQuery, a.paletteSel = true, "", 0
		a.paletteView = buildPaletteViewState(a.commandList(), "")
		a.paletteViewKey = paletteCacheKey{list: a.listKey}
	}
}

// toggleLogsPanel shows or hides the logs panel under the page (⇧⌘L,
// the View menu, the palette).
func (a *app) toggleLogsPanel() { a.logsPanel = !a.logsPanel }

// runCommand closes the overlays and runs cmd.
func (a *app) runCommand(cmd command) {
	a.paletteOpen, a.sheetOpen = false, false
	cmd.run()
}

// runCommandID runs the command with this id (the palette's rows and
// Return).
func (a *app) runCommandID(id string) {
	for _, cmd := range a.commandList() {
		if cmd.id == id {
			a.runCommand(cmd)
			return
		}
	}
}

// paletteModel is the palette's view state, rebuilt when the registry,
// the query or the language changed.
func (a *app) paletteModel() *paletteViewState {
	commands := a.commandList()
	key := paletteCacheKey{list: a.listKey, query: a.paletteQuery}
	if key != a.paletteViewKey {
		a.paletteView = buildPaletteViewState(commands, a.paletteQuery)
		a.paletteViewKey = key
	}
	return &a.paletteView
}

// paletteLocal returns the palette's local state, wired to the app's on
// first use.
func (a *app) paletteLocal() *paletteLocalUI {
	if a.paletteUI.open == nil {
		a.paletteUI.open, a.paletteUI.query, a.paletteUI.sel = &a.paletteOpen, &a.paletteQuery, &a.paletteSel
	}
	return &a.paletteUI
}

// sheetModel is the "?" sheet's view state, rebuilt when the settings or
// the language changed.
func (a *app) sheetModel() *sheetViewState {
	key := sheetCacheKey{settings: a.settingsRevision, locale: i18n.Locale()}
	if key != a.sheetViewKey {
		a.sheetView = buildSheetViewState(i18n.CurrentTranslator(), a.prefs.Features)
		a.sheetViewKey = key
	}
	return &a.sheetView
}

// buildMenu sets the native menu bar in the current language; the
// language switch calls it again. Items with shortcuts open what the
// window's own shortcuts open (shell_commands.go's shellShortcuts).
func (a *app) buildMenu() {
	a.menuLocale = i18n.Locale()
	if !a.hasMenu {
		return
	}
	act := func(fn func()) func(*mygo.MenuItem, *mygo.Window) {
		return func(*mygo.MenuItem, *mygo.Window) { a.post(fn) }
	}
	goItems := []*mygo.MenuItem{}
	for gi, g := range navGroups {
		if gi > 0 {
			goItems = append(goItems, mygo.Separator())
		}
		for _, p := range g.pages {
			if p.path == settingsPath {
				continue // the app menu's Settings… (⌘,)
			}
			acc := ""
			if k := pageKeys(p.path); k != "" {
				acc = "CmdOrCtrl+" + k[len("⌘"):]
			}
			goItems = append(goItems, &mygo.MenuItem{Label: pageTitle(p.path), Accelerator: acc, Click: act(func() { a.navigate(p.path) })})
		}
	}
	// Check for Updates… goes after About (updates.go).
	appItems := append([]*mygo.MenuItem{{Role: mygo.RoleAbout, Label: i18n.T("About Voxt")}}, a.updateMenuItems(act)...)
	appItems = append(appItems, &mygo.MenuItem{Label: i18n.T("Start meeting"), Click: act(func() {
		if !a.meetingActive() {
			a.startMeeting()
		}
	})})
	mygo.App.SetMenu(mygo.NewMenu([]*mygo.MenuItem{
		{Label: "Voxt", Submenu: append(appItems,
			mygo.Separator(),
			&mygo.MenuItem{Label: i18n.T("Settings…"), Accelerator: "CmdOrCtrl+,", Click: act(func() { a.navigate("/settings") })},
			mygo.Separator(),
			&mygo.MenuItem{Role: mygo.RoleServices, Label: i18n.T("Services")},
			mygo.Separator(),
			&mygo.MenuItem{Role: mygo.RoleHide, Label: i18n.T("Hide Voxt")},
			&mygo.MenuItem{Role: mygo.RoleHideOthers, Label: i18n.T("Hide Others")},
			&mygo.MenuItem{Role: mygo.RoleUnhide, Label: i18n.T("Show All")},
			mygo.Separator(),
			&mygo.MenuItem{Role: mygo.RoleQuit, Label: i18n.T("Quit Voxt"), Accelerator: "CmdOrCtrl+Q"},
		)},
		{Label: i18n.T("Edit"), Submenu: []*mygo.MenuItem{
			{Role: mygo.RoleUndo, Label: i18n.T("Undo")},
			{Role: mygo.RoleRedo, Label: i18n.T("Redo")},
			mygo.Separator(),
			{Role: mygo.RoleCut, Label: i18n.T("Cut")},
			{Role: mygo.RoleCopy, Label: i18n.T("Copy")},
			{Role: mygo.RolePaste, Label: i18n.T("Paste")},
			{Role: mygo.RoleSelectAll, Label: i18n.T("Select All")},
		}},
		{Label: i18n.T("View"), Submenu: []*mygo.MenuItem{
			{Label: i18n.T("Command palette"), Accelerator: "CmdOrCtrl+K", Click: act(a.openPalette)},
			{Label: i18n.T("Show or hide the logs panel"), Click: act(a.toggleLogsPanel)},
			mygo.Separator(),
			{Role: mygo.RoleToggleFullScreen, Label: i18n.T("Enter Full Screen")},
		}},
		{Label: i18n.T("Go"), Submenu: append(goItems,
			mygo.Separator(),
			&mygo.MenuItem{Label: i18n.T("Settings"), Click: act(func() { a.navigate("/settings") })},
		)},
		{Label: i18n.T("Window"), Submenu: []*mygo.MenuItem{
			{Label: i18n.T("Close Window"), Accelerator: "CmdOrCtrl+W", Click: act(a.closeWindow)}, // as the red ×: tray.go
			mygo.Separator(),
			{Role: mygo.RoleMinimize, Label: i18n.T("Minimize")},
			{Role: mygo.RoleZoom, Label: i18n.T("Zoom")},
			mygo.Separator(),
			{Role: mygo.RoleFront, Label: i18n.T("Bring All to Front")},
		}},
		{Label: i18n.T("Help"), Submenu: []*mygo.MenuItem{
			{Label: i18n.T("Keyboard shortcuts"), Click: act(func() { a.sheetOpen = true })},
			{Label: i18n.T("Logs"), Click: act(func() { a.navigate("/logs") })},
		}},
	}))
}

// openURL follows an voxtnew:// deep link: voxtnew://history,
// voxtnew://settings/audio, voxtnew:///home, voxtnew://models/network, or
// voxtnew://dictionary/profile?name=Work. Router paths in notifications go
// through it too.
func (a *app) openURL(raw string) {
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "" && !strings.EqualFold(u.Scheme, "voxtnew")) {
		a.toast(i18n.T("Unknown link: %s", raw))
		return
	}
	path := u.Path
	if u.Scheme != "" && u.Host != "" {
		path = "/" + u.Host + u.Path
	}
	path = "/" + strings.Trim(path, "/")
	if to, ok := movedPaths[path]; ok {
		path = to
	}
	switch sec := section(path); {
	case path == "/dictionary/profile":
		a.switchDictionaryProfileByName(u.Query().Get("name"))
	case path == networkPath: // the Speech models page's Network dialog (models_network.go)
		a.navigate(modelsPath)
		a.openNetwork()
	case slices.ContainsFunc(pages, func(p navPage) bool { return p.path == sec }): // every sidebar page
		a.navigate(sec)
	case sec == "/settings":
		if path == "/settings" {
			path = "/settings/" + a.lastSettings
		}
		a.navigate(path)
	default:
		a.toast(i18n.T("Unknown link: %s", raw))
	}
}
