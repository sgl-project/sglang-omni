package main

// App enhancement's preview (appenhance_page.go): which group the app in
// front (or the address) would use, and where a history entry's group came
// from.

import (
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

func (a *app) aePreviewCard(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	app := a.previewApp()
	pv := a.aePrev
	browser := appctx.IsBrowser(app.BundleID, a.prefs.AppEnhancement.Browsers)
	fillCard(c.Key("ae-preview")).Padding(px(t, sp3), px(t, sp4)).Gap(px(t, sp2)).Children(func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			if app.BundleID == "" {
				ui.Icon(c, icons.WindowFrame.Linear).Size(px(t, 16), px(t, 16)).TextColor(k.Text2).Shrink(0)
			} else {
				a.appIcon(c, proto.AppRef{BundleID: app.BundleID}, 20, true)
			}
			name := app.Name
			if name == "" {
				name = app.BundleID
			}
			if name == "" {
				ui.Text(c, i18n.T("Switch to another app to see which group it falls in.")).TextColor(k.Text2).Grow(1).MinWidth(0)
			} else {
				ui.Text(c, i18n.T("In front: %s", name)).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
				if browser {
					tagChip(c, i18n.T("Browser"))
				}
				ui.Spacer(c)
			}
			if iconButton(c.Key("ae-preview-refresh"), icons.Refresh, i18n.T("Read again")).Disabled(app.BundleID == "").Clicked() {
				a.refreshPreview(true)
			}
		})
		if app.BundleID == "" {
			return
		}
		if browser {
			addr := ""
			switch {
			case pv.app != app || pv.reading:
				addr = i18n.T("Reading the address…")
			case pv.url != "":
				if ad, ok := appctx.ParseAddress(pv.url); ok {
					addr = ad.String()
				} else {
					addr = pv.url
				}
			case !a.helperReady():
				addr = i18n.T("The helper is not running: the address cannot be read.")
			default:
				addr = i18n.T("No address: %s does not show it to Accessibility now. It is read again when you press a shortcut there; if it never shows, this browser can only match as an app.", app.Name)
			}
			truncText(c, addr).FontSize(px(t, fsCaption)).TextColor(k.Text2)
		}
		m := a.previewMatch()
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			if m.Matched() {
				ui.Icon(c, icons.Target.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Accent).Shrink(0)
				ui.Text(c, i18n.T("Matches %s, %s", m.GroupName(), uitext.SourceText(m.Source, m.Pattern))).FontSize(px(t, fsCaption)).Grow(1).MinWidth(0)
			} else {
				ui.Text(c, i18n.T("No group matches: only the features' own rules apply.")).FontSize(px(t, fsCaption)).TextColor(k.Text2).Grow(1).MinWidth(0)
			}
		})
		g, i := a.aeGroup()
		if i < 0 {
			return
		}
		addApp := !g.HasApp(app.BundleID)
		ad, okAddr := appctx.ParseAddress(pv.url)
		addSite := okAddr && browser && pv.app == app && !slices.ContainsFunc(g.URLs, func(u string) bool { return appctx.MatchPattern(u, ad) })
		if !addApp && !addSite {
			return
		}
		ui.Row(c).Gap(px(t, sp1)).AlignItems(ui.Center).Wrap().Margin(0, 0, 0, -px(t, 4)).Children(func() {
			if addApp && quietButton(c.Key("ae-preview-add-app"), icons.Add, i18n.T("Add %s to %s", app.Name, g.Name)).Clicked() {
				a.updateGroup(g.ID, func(g *appctx.Group) {
					g.Apps = append(g.Apps, appctx.App{BundleID: app.BundleID, Name: app.Name})
				})
			}
			if addSite && quietButton(c.Key("ae-preview-add-site"), icons.Add, i18n.T("Add %s to %s", ad.Host, g.Name)).Clicked() {
				host := ad.Host
				a.updateGroup(g.ID, func(g *appctx.Group) { g.URLs = append(g.URLs, host) })
			}
		})
	})
}
