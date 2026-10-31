package main

// The App enhancement page (/app-enhancement): the
// switch, the groups in a list on the left (a drawer when the window is
// narrow), and the group shown on the right: its name, whether it is on,
// the features it applies to, its scene prompt (layered on top of the
// feature's own clean-up), its apps, its website rules, and the key
// pressed after inserting. A preview at the top says which app is in
// front (the last one before VoxtNew), its address and the group it
// falls in, live. Browsers (built in and the user's) are listed in a
// dialog that says which ones gave an address at the last shortcut.
//
// Every change goes through setRules (appenhance.go): the sessions see it
// at once, the store a moment later.

import (
	"slices"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// Widths (DIPs at 100 %): the group list, and the least the group's
// column keeps beside it (else the list is a drawer): both fit beside
// the sidebar in the default window (800 wide) and beside the rail in the
// smallest (680).
const (
	aeListW    = 200
	aeMainMinW = 380
)

// aePageState is the page's view state.
type aePageState struct {
	sel  string // the group shown
	list panelState
	// The fields of the group editFor, as typed.
	editFor string
	name    string
	prompt  string
	// The website rule being added, and why it is not valid.
	urlIn  string
	urlErr string
	// The key after inserting: the select's label, the typed keys.
	keySel    string
	keyCustom bool
	keyText   string
	keyErr    string
	// The Add apps dialog.
	appsOpen    bool
	appsQuery   string
	apps        []proto.AppInfo
	appsLoading bool
	appsErr     string
	bundleIn    string
	// The Browsers dialog.
	browsersOpen bool
	brID, brName string
	// The Delete dialog.
	delAsk bool
}

// aeShown runs when the page shows: the preview reads the app in front.
func (a *app) aeShown() { a.refreshPreview(true) }

// aeGroup is the group shown, its index (-1: none).
func (a *app) aeGroup() (appctx.Group, int) {
	p := &a.aePage
	for i, g := range a.aeRules.Groups {
		if g.ID == p.sel {
			return g, i
		}
	}
	if len(a.aeRules.Groups) > 0 {
		p.sel = a.aeRules.Groups[0].ID
		return a.aeRules.Groups[0], 0
	}
	return appctx.Group{}, -1
}

// updateGroup changes the group id with fn.
func (a *app) updateGroup(id string, fn func(g *appctx.Group)) {
	r := cloneRules(a.aeRules)
	for i := range r.Groups {
		if r.Groups[i].ID == id {
			fn(&r.Groups[i])
		}
	}
	a.setRules(r)
}

// addGroup adds a group and shows it.
func (a *app) addGroup() {
	r := cloneRules(a.aeRules)
	name := i18n.T("Group %d", len(r.Groups)+1)
	g := appctx.NewGroup(newGroupID(), name)
	r.Groups = append(r.Groups, g)
	a.setRules(r)
	a.aePage.sel = g.ID
	a.aePage.editFor = ""
	a.aePage.list.open = false // the drawer gives way to the new group
}

// deleteGroup deletes the group shown.
func (a *app) deleteGroup(id string) {
	r := cloneRules(a.aeRules)
	r.Groups = slices.DeleteFunc(r.Groups, func(g appctx.Group) bool { return g.ID == id })
	a.setRules(r)
	a.aePage.sel, a.aePage.editFor = "", ""
	a.toast(i18n.T("Group deleted"))
}

func (a *app) appEnhancementPage(c *ui.Context) {
	t := c.Theme()
	p := &a.aePage
	inline, shown := p.list.layout(a, c, px(t, aeListW), px(t, aeMainMinW))
	pageHeader(c, i18n.T("App enhancement"), func() {
		if !inline {
			p.list.toggle(c.Key("ae-list-toggle"), icons.SidebarMinimalistic, shown, i18n.T("Show the groups"), i18n.T("Hide the groups"))
		}
		on := &a.prefs.AppEnhancement.Enabled
		state := i18n.T("Off")
		if *on {
			state = i18n.T("On")
		}
		caption(c, state).SingleLine()
		if ui.Switch(c.Key("ae-on"), on).Label(i18n.T("Use App enhancement")).Tooltip(i18n.T("Use App enhancement")).Changed() {
			v := *on
			a.updateSettings(func(s *settings.Settings) { s.AppEnhancement.Enabled = v })
		}
		if primaryIconButton(c.Key("ae-new"), icons.Add, i18n.T("New group")).Clicked() {
			a.addGroup()
		}
		// The browsers: a setting of the page, not a group, so not in
		// the groups' list.
		if iconButton(c.Key("ae-browsers"), icons.Global, i18n.T("Browsers")).
			Tooltip(i18n.T("Browsers: which apps are browsers, read for their tab's address")).Clicked() {
			a.openBrowsers()
		}
	})
	ui.Row(c).Grow(1).MinHeight(0).AlignItems(ui.Stretch).Children(func() {
		p.list.view(c, "ae-groups", i18n.T("Groups"), inline, shown, true, px(t, aeListW), func() { a.aeGroupList(c) })
		ui.Scroll(c.Key("ae-main")).Grow(1).MinWidth(0).Children(func() {
			ui.Column(c).Padding(0, px(t, sp6), px(t, sp6)).Gap(px(t, sp3)).MaxWidth(px(t, 760)).Children(func() {
				a.aePreviewCard(c)
				if !a.prefs.AppEnhancement.Enabled {
					notice(c, "ae-off", noticeInfo, i18n.T("App enhancement is off: every app gets only the features' own rules."), "", nil, nil)
				}
				if a.aeErr != "" {
					notice(c, "ae-err", noticeError, i18n.T("Could not read the groups."), a.aeErr, nil, nil)
				}
				g, i := a.aeGroup()
				if i < 0 {
					a.aeEmpty(c)
					return
				}
				a.aeDetail(c, g)
			})
		})
	})
	a.aeAppsDialog(c)
	a.aeBrowsersDialog(c)
	a.aeDeleteDialog(c)
	a.flushIcons() // appicons.go: the icons the rows asked for
}

// openBrowsers opens the Browsers dialog, with the apps on this Mac.
func (a *app) openBrowsers() {
	p := &a.aePage
	p.browsersOpen = true
	if len(p.apps) == 0 {
		a.loadApps()
	}
}

// aeEmpty is the page without groups.
func (a *app) aeEmpty(c *ui.Context) {
	emptyState(c, appEnhancementIcon, i18n.T("Rules for the app or website in front"),
		i18n.T("Group apps and website addresses, and give each group a scene prompt (a chat app, your mail, code review). It is added on top of the features' own rules when you dictate there."), func() {
			if ghostIconButton(c.Key("ae-empty-new"), icons.Add, i18n.T("New group")).Clicked() {
				a.addGroup()
			}
		})
}

// ---- The list ----

// groupSummary is a group's second line: its apps and sites.
func groupSummary(g appctx.Group) string {
	var parts []string
	if n := len(g.Apps); n > 0 {
		parts = append(parts, i18n.N(n, "%d app", "%d apps", n))
	}
	if n := len(g.URLs); n > 0 {
		parts = append(parts, i18n.N(n, "%d website", "%d websites", n))
	}
	if len(parts) == 0 {
		return i18n.T("Empty")
	}
	return strings.Join(parts, " · ")
}

func (a *app) aeGroupList(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	p := &a.aePage
	_, _ = a.aeGroup()
	match := a.previewMatch()
	ui.Column(c).Padding(px(t, sp2), px(t, sp2), px(t, sp4)).Gap(px(t, 2)).Children(func() {
		for _, g := range a.aeRules.Groups {
			row := ui.ButtonBase(c.Key("ae-group-"+g.ID)).Padding(px(t, 6), px(t, sp2)).Radius(px(t, radiusControl)).
				Label(g.Name).Gap(px(t, sp2)).AlignItems(ui.Center)
			switch {
			case p.sel == g.ID:
				row.Background(k.Pressed)
			case row.Hovered():
				row.Background(k.Hover)
			}
			if row.Clicked() {
				p.sel = g.ID
				p.list.open = false
			}
			row.Children(func() {
				col := k.Success
				if !g.Enabled {
					col = k.G[5]
				}
				statusDot(c, col)
				ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 1)).Children(func() {
					name := g.Name
					if name == "" {
						name = i18n.T("Untitled")
					}
					tx := ui.Text(c, name).SingleLine().Ellipsis("…").MinWidth(0)
					if !g.Enabled {
						tx.TextColor(k.Text2)
					}
					ui.Text(c, groupSummary(g)).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine().Ellipsis("…").MinWidth(0)
				})
				if match.GroupID() == g.ID {
					iconMark(c, icons.Target, k.Accent, i18n.T("Matches the app in front"))
				}
			})
		}
	})
}
