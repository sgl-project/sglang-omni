package main

// The App enhancement page's dialogs (appenhance_page.go): adding apps,
// the browsers whose addresses are read, and deleting a group.

import (
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// ---- Dialogs ----

// loadApps asks the helper for the running and installed apps.
func (a *app) loadApps() {
	p := &a.aePage
	if !a.helperReady() {
		p.appsErr = i18n.T("The helper is not running: add an app by its bundle ID.")
		return
	}
	p.appsLoading, p.appsErr = true, ""
	helperCall(a, proto.MethodAppList, proto.AppListParams{Installed: true}, 10*time.Second, func(res proto.AppListResult, err error) {
		p := &a.aePage
		p.appsLoading = false
		if err != nil {
			p.appsErr = err.Error()
			return
		}
		p.apps = slices.DeleteFunc(res.Apps, func(x proto.AppInfo) bool { return isSelf(proto.App{PID: x.PID, BundleID: x.BundleID}) })
	})
}

func (a *app) aeAppsDialog(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	p := &a.aePage
	if !p.appsOpen {
		return
	}
	g, i := a.aeGroup()
	if i < 0 {
		p.appsOpen = false
		return
	}
	dialogFrame(c, "ae-apps-dlg", &p.appsOpen, i18n.T("Add apps to %s", g.Name), i18n.T("Running apps first, then the ones in Applications."), 480, func() {
		ui.Row(c).Children(func() { searchField(c, "ae-apps-search", &p.appsQuery, i18n.T("Search apps"), 0) })
		switch {
		case p.appsLoading:
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ui.Spinner(c).Size(px(t, 16), px(t, 16))
				caption(c, i18n.T("Looking for apps…"))
			})
		case p.appsErr != "":
			warnLine(c, p.appsErr)
		}
		q := strings.ToLower(strings.TrimSpace(p.appsQuery))
		apps := slices.Clone(p.apps)
		slices.SortStableFunc(apps, func(x, y proto.AppInfo) int {
			switch {
			case x.Running == y.Running:
				return 0
			case x.Running:
				return -1
			}
			return 1
		})
		shown := 0
		ui.Column(c.Key("ae-apps-list")).Gap(px(t, 2)).Children(func() {
			// Icons load as rows show: the first screenful at once, the
			// others as they scroll into view.
			const eager = 12
			for _, app := range apps {
				if q != "" && !strings.Contains(strings.ToLower(app.Name), q) && !strings.Contains(strings.ToLower(app.BundleID), q) {
					continue
				}
				shown++
				member := g.HasApp(app.BundleID)
				row := ui.ButtonBase(c.Key("ae-pick-"+app.BundleID)).Padding(px(t, 6), px(t, sp2)).Radius(px(t, radiusControl)).
					Label(app.Name).Gap(px(t, sp2)).AlignItems(ui.Center).Checked(member)
				if row.Hovered() {
					row.Background(k.Hover)
				}
				if row.Clicked() {
					info := app
					a.updateGroup(g.ID, func(g *appctx.Group) {
						if g.HasApp(info.BundleID) {
							g.Apps = slices.DeleteFunc(g.Apps, func(m appctx.App) bool { return strings.EqualFold(m.BundleID, info.BundleID) })
						} else {
							g.Apps = append(g.Apps, appctx.App{BundleID: info.BundleID, Name: info.Name})
						}
					})
				}
				visible := shown <= eager || row.InView()
				row.Children(func() {
					a.appIcon(c, proto.AppRef{BundleID: app.BundleID, Path: app.Path}, 20, visible)
					ui.Text(c, app.Name).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
					truncText(c, app.BundleID).FontSize(px(t, fsCaption)).TextColor(k.Text3).Grow(1)
					if app.Running {
						tagChip(c, i18n.T("Running"))
					}
					// In the group: a check at the end (the room kept for
					// the others, so the rows line up).
					mark := ui.Box(c).Size(px(t, 16), px(t, 16)).Center().Shrink(0)
					if member {
						mark.Children(func() {
							ui.Icon(c, icons.CheckCircle.Bold).Size(px(t, 16), px(t, 16)).TextColor(k.Success)
						})
					}
				})
			}
		})
		if shown == 0 && !p.appsLoading && len(p.apps) > 0 {
			caption(c, i18n.T("No app matches."))
		}
		ui.Box(c).Height(1).Background(k.Line).Shrink(0).Margin(px(t, sp1), 0)
		caption(c, i18n.T("Not listed? Add it by its bundle ID:"))
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			in := ui.TextInput(c.Key("ae-bundle-in"), &p.bundleIn).Label(i18n.T("Bundle ID")).Placeholder("com.example.app").Font("monospace").Grow(1).MinWidth(0)
			id := strings.TrimSpace(p.bundleIn)
			if (ui.Button(c.Key("ae-bundle-add"), i18n.T("Add")).Label(i18n.T("Add the bundle ID")).Disabled(id == "" || strings.ContainsAny(id, " /")).Clicked() || in.Submitted()) && id != "" && !strings.ContainsAny(id, " /") {
				a.updateGroup(g.ID, func(g *appctx.Group) {
					if !g.HasApp(id) {
						g.Apps = append(g.Apps, appctx.App{BundleID: id})
					}
				})
				p.bundleIn = ""
			}
		})
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("ae-apps-done"), i18n.T("Done")).Clicked() {
			p.appsOpen = false
		}
	})
}

// appPath is the bundle of bundleID as app.list said, "" when it did not
// list it (the helper finds the app by its id).
func (a *app) appPath(bundleID string) string {
	for _, x := range a.aePage.apps {
		if strings.EqualFold(x.BundleID, bundleID) {
			return x.Path
		}
	}
	return ""
}

// browserState says what the last shortcut in a browser told of its
// address.
func (a *app) browserState(b appctx.Browser) (string, noticeKind) {
	if r, ok := a.urlReads[b.BundleID]; ok {
		if r.ok {
			return i18n.T("Address read"), noticeOK
		}
		return i18n.T("No address: app only"), noticeWarn
	}
	if b.Engine == appctx.Gecko || b.Engine == "" {
		return i18n.T("Not tried; may be app only"), noticeInfo
	}
	return i18n.T("Not tried yet"), noticeInfo
}

func (a *app) aeBrowsersDialog(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	p := &a.aePage
	if !p.browsersOpen {
		return
	}
	custom := a.prefs.AppEnhancement.Browsers
	dialogFrame(c, "ae-browsers-dlg", &p.browsersOpen, i18n.T("Browsers"),
		i18n.T("Read through Accessibility only: no browser permission to grant."), 520, func() {
			// The browsers on this Mac (once the helper listed the apps),
			// the others in a line.
			installed := map[string]bool{}
			for _, x := range p.apps {
				installed[strings.ToLower(x.BundleID)] = true
			}
			var shown []appctx.Browser
			var others []string
			for _, b := range appctx.Browsers(custom) {
				if len(p.apps) == 0 || installed[strings.ToLower(b.BundleID)] || !appctx.IsBuiltin(b.BundleID) {
					shown = append(shown, b)
				} else {
					others = append(others, b.Name)
				}
			}
			caption(c, i18n.T("At each shortcut in a browser, VoxtNew reads its tab's address. A browser that shows none matches only as an app."))
			ui.Column(c.Key("ae-browser-list")).Gap(px(t, 2)).Children(func() {
				for _, b := range shown {
					ui.Row(c.Key("ae-browser-" + b.BundleID)).Gap(px(t, sp2)).AlignItems(ui.Center).MinHeight(px(t, 30)).Children(func() {
						text, kind := a.browserState(b)
						col := k.Text3
						switch kind {
						case noticeOK:
							col = k.Success
						case noticeWarn:
							col = k.Warning
						}
						a.appIcon(c, proto.AppRef{BundleID: b.BundleID, Path: a.appPath(b.BundleID)}, 24, true)
						ui.Column(c).Grow(1).MinWidth(0).Children(func() {
							ui.Text(c, b.Name).SingleLine().Ellipsis("…").MinWidth(0)
							truncText(c, b.BundleID).FontSize(px(t, fsMicro)).TextColor(k.Text3)
						})
						statusDot(c, col).Tooltip(text)
						truncText(c, text).FontSize(px(t, fsCaption)).TextColor(k.Text2).MaxWidth(px(t, 200))
						if !appctx.IsBuiltin(b.BundleID) {
							if iconButton(c.Key("remove"), icons.Close, i18n.T("Remove %s", b.Name)).Clicked() {
								id := b.BundleID
								a.updateSettings(func(s *settings.Settings) {
									s.AppEnhancement.Browsers = slices.DeleteFunc(slices.Clone(s.AppEnhancement.Browsers), func(x settings.Browser) bool {
										return strings.EqualFold(x.BundleID, id)
									})
								})
							}
						}
					})
				}
			})
			if len(others) > 0 {
				caption(c, i18n.T("Also known when installed: %s.", strings.Join(others, i18n.T(", ")))).TextColor(k.Text3)
			}
			ui.Box(c).Height(1).Background(k.Line).Shrink(0).Margin(px(t, sp1), 0)
			caption(c, i18n.T("Another browser: its bundle ID (and a name)."))
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
				ui.TextInput(c.Key("ae-br-id"), &p.brID).Label(i18n.T("Bundle ID")).Placeholder("com.example.browser").Font("monospace").Grow(1).Basis(px(t, 180)).MinWidth(0)
				ui.TextInput(c.Key("ae-br-name"), &p.brName).Label(i18n.T("Name")).Placeholder(i18n.T("Name")).Width(px(t, 140))
				id := strings.TrimSpace(p.brID)
				if ui.Button(c.Key("ae-br-add"), i18n.T("Add")).Label(i18n.T("Add the browser")).Disabled(id == "" || strings.ContainsAny(id, " /") || appctx.IsBrowser(id, custom)).Clicked() {
					b := settings.Browser{BundleID: id, Name: strings.TrimSpace(p.brName)}
					a.updateSettings(func(s *settings.Settings) {
						s.AppEnhancement.Browsers = append(slices.Clone(s.AppEnhancement.Browsers), b)
					})
					p.brID, p.brName = "", ""
				}
			})
		}, func() {
			ui.Spacer(c)
			if ui.Button(c.Key("ae-browsers-done"), i18n.T("Done")).Clicked() {
				p.browsersOpen = false
			}
		})
}

func (a *app) aeDeleteDialog(c *ui.Context) {
	p := &a.aePage
	if !p.delAsk {
		return
	}
	g, i := a.aeGroup()
	if i < 0 {
		p.delAsk = false
		return
	}
	dialogFrame(c, "ae-del", &p.delAsk, i18n.T("Delete “%s”?", g.Name), "", 420, func() {
		ui.Text(c, i18n.T("Its apps and websites go back to the features' own rules. This cannot be undone."))
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("ae-del-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
			p.delAsk = false
		}
		if recordButton(c.Key("ae-del-ok"), i18n.T("Delete")).Clicked() {
			p.delAsk = false
			a.deleteGroup(g.ID)
		}
	})
}
