package main

// Commands: one registry behind the ⌘K palette, the native menu and the
// window's shortcuts, plus the "?" sheet listing the shortcuts, and
// voxtnew:// deep links. (From AudioSmith.)

import (
	"net/url"
	"slices"
	"strings"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
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

// commands lists the commands, in the palette's order.
func (a *app) commands() []command {
	nav := i18n.T("Go to")
	var cmds []command
	add := func(id, group, title, en, keys string, enabled bool, run func()) {
		cmds = append(cmds, command{id: id, group: group, title: title, en: en, keys: keys, run: run, enabled: enabled})
	}
	for _, p := range pages { // every sidebar page (shell_pages.go navGroups)
		add("go"+p.path, nav, pageTitle(p.path), p.title, pageKeys(p.path), true, func() { a.navigate(p.path) })
	}
	add("go/settings", nav, i18n.T("Settings"), "Settings", "⌘,", true, func() { a.navigate("/settings") })
	for _, s := range settingsSections() {
		add("go/settings/"+s.id, i18n.T("Settings"), s.title, s.en, "", true, func() { a.navigate("/settings/" + s.id) })
	}

	view := i18n.T("View")
	add("palette", view, i18n.T("Command palette"), "Command palette", "⌘K", true, a.openPalette)
	add("logs-panel", view, i18n.T("Show or hide the logs panel"), "Show or hide the logs panel", "⇧⌘L", true, func() { a.logsPanel = !a.logsPanel })
	add("shortcuts", view, i18n.T("Keyboard shortcuts"), "Keyboard shortcuts", "?", true, func() { a.sheetOpen = true })
	add("helper-restart", view, i18n.T("Restart the helper"), "Restart the helper", "", true, a.restartHelper)
	add("check-updates", view, i18n.T("Check for Updates…"), "Check for Updates", "", a.upd != nil, a.checkForUpdates)
	add("setup-guide", view, i18n.T("Open the setup guide"), "Open the setup guide onboarding", "", true, a.openOnboarding)

	look := i18n.T("Appearance")
	for _, th := range []struct{ id, title, en string }{
		{"system", i18n.T("Theme: follow the system"), "Theme: follow the system"},
		{"light", i18n.T("Theme: light"), "Theme: light"},
		{"dark", i18n.T("Theme: dark"), "Theme: dark"},
	} {
		add("theme/"+th.id, look, th.title, th.en, "", a.prefs.Appearance.Theme != th.id, func() {
			a.updateSettings(func(s *settings.Settings) { s.Appearance.Theme = th.id })
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
	}
}

// runCommand closes the overlays and runs cmd.
func (a *app) runCommand(cmd command) {
	a.paletteOpen, a.sheetOpen = false, false
	cmd.run()
}

// shortcuts handles the window's keys. Each opens rather than toggles
// what the native menu's items also open, so a key that reaches both
// does the same thing once.
func (a *app) shortcuts(c *ui.Context) {
	if c.Shortcut(ui.Cmd, ui.KeyK) {
		a.openPalette()
	}
	if c.Shortcut(ui.Cmd, ui.KeyComma) {
		a.navigate("/settings")
	}
	for i, path := range shortcutPages {
		if c.Shortcut(ui.Cmd, ui.Key1+ui.Key(i)) {
			a.navigate(path)
		}
	}
	if c.Shortcut(ui.Cmd|ui.Shift, ui.KeyL) {
		a.logsPanel = !a.logsPanel
	}
	if c.Shortcut(ui.Shift, ui.KeySlash) {
		a.sheetOpen = true
	}
}

// commandIcon is a command's icon in the palette: its page's for "Go
// to" commands, a settings section's, or nil.
func commandIcon(id string) *icons.Icon {
	if rest, ok := strings.CutPrefix(id, "go/settings"); ok {
		ic := icons.Settings
		if sec := strings.TrimPrefix(rest, "/"); sec != "" {
			ic = settingsSectionIcon(sec)
		}
		return &ic
	}
	if rest, ok := strings.CutPrefix(id, "go"); ok {
		for _, p := range pages {
			if p.path == rest {
				ic := p.icon
				return &ic
			}
		}
	}
	return nil
}

// kbd shows a shortcut as a key cap.
func kbd(c *ui.Context, keys string) ui.Element {
	t := c.Theme()
	k := tok(t)
	return ui.Text(c, keys).FontSize(px(t, fsMicro)).TextColor(k.Text2).Padding(px(t, 1), px(t, 5)).Radius(px(t, 4)).
		Background(k.Subtle).Border(1, k.Line).SingleLine().Shrink(0)
}

// palette is the ⌘K command palette.
func (a *app) palette(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	_, h := c.Size()
	ui.DialogBase(c, &a.paletteOpen, func(backdrop, panel ui.Element) {
		backdrop.Background(ui.RGBA(0, 0, 0, 0.2)).Justify(ui.Start).Padding(min(px(t, 90), h*0.12), px(t, sp6), 0)
		panel.Width(px(t, 560)).MaxWidth(px(t, 560)).Radius(px(t, radiusCard)).Background(k.Bg).Border(1, k.Line).Clip().
			Shadow(0, 16, 40, 0, ui.RGBA(0, 0, 0, 0.22)).Label(i18n.T("Command palette"))
		items := a.paletteItems()
		a.paletteSel = max(0, min(a.paletteSel, len(items)-1))
		ui.Row(c).Padding(px(t, 10), px(t, 14)).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			ui.Icon(c, searchIcon.Linear).Size(px(t, 18), px(t, 18)).TextColor(k.Text3)
			// A text input rather than ui.SearchField, whose placeholder
			// is not translated; Escape closes the palette.
			search := ui.TextInputBase(c.Key("palette-search"), &a.paletteQuery).AutoFocus().Grow(1).FontSize(px(t, 15)).
				Placeholder(i18n.T("Type a command or a page…")).Label(i18n.T("Search commands"))
			if search.Changed() {
				a.paletteSel = 0
			}
			if search.Shortcut(0, ui.KeyDown) && len(items) > 0 {
				a.paletteSel = (a.paletteSel + 1) % len(items)
			}
			if search.Shortcut(0, ui.KeyUp) && len(items) > 0 {
				a.paletteSel = (a.paletteSel - 1 + len(items)) % len(items)
			}
			if search.Submitted() && len(items) > 0 {
				a.runCommand(items[a.paletteSel])
			}
			kbd(c, "Esc")
		})
		ui.Divider(c)
		ui.Scroll(c).MaxHeight(min(px(t, 380), h*0.6)).Children(func() {
			ui.Column(c).Padding(px(t, 6)).Children(func() {
				if len(items) == 0 {
					ui.Text(c, i18n.T("No matching commands")).TextColor(k.Text2).Padding(px(t, 10))
				}
				group := ""
				for i, cmd := range items {
					if cmd.group != group {
						group = cmd.group
						ui.Text(c, group).FontSize(px(t, fsMicro)).FontWeight(600).TextColor(k.Text3).Padding(px(t, sp2), px(t, 10), px(t, 4))
					}
					row := ui.ButtonBase(c.Key("cmd-"+cmd.id)).Padding(px(t, 7), px(t, 10)).Radius(px(t, radiusControl)).Label(cmd.title)
					if i == a.paletteSel {
						row.Background(k.Pressed)
					} else if row.Hovered() {
						row.Background(k.Hover)
					}
					row.Children(func() {
						ui.Row(c).Grow(1).Gap(px(t, 10)).AlignItems(ui.Center).Children(func() {
							ic := icons.Command
							if p := commandIcon(cmd.id); p != nil {
								ic = *p
							}
							ui.Icon(c, ic.Linear).Size(px(t, 16), px(t, 16)).TextColor(k.Text2)
							ui.Text(c, cmd.title).Grow(1).SingleLine().Ellipsis("…").MinWidth(0)
							if cmd.keys != "" {
								kbd(c, cmd.keys)
							}
						})
					})
					if row.Clicked() {
						a.runCommand(cmd)
					}
				}
			})
		})
	})
}

// shortcutSheet lists the keyboard shortcuts ("?").
func (a *app) shortcutSheet(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	ui.Modal(c, &a.sheetOpen, func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			ui.Icon(c, icons.Keyboard.Linear).Size(px(t, 18), px(t, 18)).TextColor(k.Text2)
			sectionHeading(c, i18n.T("Keyboard shortcuts")).Grow(1)
		})
		f := a.prefs.Features
		first := func(hs []settings.Hotkey) string {
			if len(hs) == 0 {
				return "—"
			}
			return uitext.HotkeyGlyphs(hs[0].Keys)
		}
		groups := []struct {
			title string
			rows  [][2]string
		}{
			{i18n.T("Anywhere on the Mac"), [][2]string{
				{first(f.Transcribe.Hotkeys), i18n.T("Transcribe")},
				{first(f.Translate.Hotkeys), i18n.T("Translate")},
				{first(f.Rewrite.Hotkeys), i18n.T("Rewrite")},
				{uitext.HotkeyGlyphs(f.PasteLast.Keys), i18n.T("Paste the last result")},
			}},
			{i18n.T("Navigation"), [][2]string{
				{"⌘K", i18n.T("Command palette")},
				{"⌘1  ⌘2", i18n.T("Home, History")},
				{"⌘3 … ⌘6", i18n.T("Transcribe, Translate, Rewrite, App enhancement")},
				{"⌘,", i18n.T("Settings")},
				{"⌘[  ⌘]", i18n.T("Back, forward")},
				{"⇧⌘L", i18n.T("Show or hide the logs panel")},
				{"?", i18n.T("Keyboard shortcuts")},
				{"Esc", i18n.T("Close a dialog or popover")},
			}},
		}
		// The list scrolls in a short window (680×440): Close stays in it.
		_, h := c.Size()
		ui.Scroll(c.Key("sheet-scroll")).MaxHeight(max(px(t, 160), h-px(t, 180))).Children(func() {
			ui.Column(c).Gap(px(t, sp3)).Width(px(t, 440)).MaxWidth(px(t, 440)).Children(func() {
				for _, g := range groups {
					sectionTitle(c, g.title)
					ui.Grid(c).ColumnTracks(ui.Fixed(px(t, 100)), ui.Fr(1)).GapX(px(t, sp3)).GapY(px(t, 6)).AlignItems(ui.Center).Children(func() {
						for _, r := range g.rows {
							ui.Row(c).Children(func() { kbd(c, r[0]) })
							ui.Text(c, r[1])
						}
					})
				}
			})
		})
		ui.Row(c).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("sheet-close"), i18n.T("Close")).Clicked() {
				a.sheetOpen = false
			}
		})
	})
}

// buildMenu sets the native menu bar in the current language; the
// language switch calls it again. Items with shortcuts open what the
// window's own shortcuts open (see shortcuts).
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
			acc := ""
			if k := pageKeys(p.path); k != "" {
				acc = "CmdOrCtrl+" + k[len("⌘"):]
			}
			goItems = append(goItems, &mygo.MenuItem{Label: pageTitle(p.path), Accelerator: acc, Click: act(func() { a.navigate(p.path) })})
		}
	}
	// Check for Updates… goes after About (updates.go).
	appItems := append([]*mygo.MenuItem{{Role: mygo.RoleAbout, Label: i18n.T("About VoxtNew")}}, a.updateMenuItems(act)...)
	mygo.App.SetMenu(mygo.NewMenu([]*mygo.MenuItem{
		{Label: "VoxtNew", Submenu: append(appItems,
			mygo.Separator(),
			&mygo.MenuItem{Label: i18n.T("Settings…"), Accelerator: "CmdOrCtrl+,", Click: act(func() { a.navigate("/settings") })},
			mygo.Separator(),
			&mygo.MenuItem{Role: mygo.RoleServices, Label: i18n.T("Services")},
			mygo.Separator(),
			&mygo.MenuItem{Role: mygo.RoleHide, Label: i18n.T("Hide VoxtNew")},
			&mygo.MenuItem{Role: mygo.RoleHideOthers, Label: i18n.T("Hide Others")},
			&mygo.MenuItem{Role: mygo.RoleUnhide, Label: i18n.T("Show All")},
			mygo.Separator(),
			&mygo.MenuItem{Role: mygo.RoleQuit, Label: i18n.T("Quit VoxtNew")},
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
			{Label: i18n.T("Show or hide the logs panel"), Click: act(func() { a.logsPanel = !a.logsPanel })},
			mygo.Separator(),
			{Role: mygo.RoleToggleFullScreen, Label: i18n.T("Enter Full Screen")},
		}},
		{Label: i18n.T("Go"), Submenu: append(goItems,
			mygo.Separator(),
			&mygo.MenuItem{Label: i18n.T("Settings"), Click: act(func() { a.navigate("/settings") })},
		)},
		{Label: i18n.T("Window"), Submenu: []*mygo.MenuItem{
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
// voxtnew://settings/audio, voxtnew:///home, voxtnew://models/network. Router paths in
// notifications go through it too.
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
	case path == networkPath: // the Local models page's Network dialog (models_network.go)
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
