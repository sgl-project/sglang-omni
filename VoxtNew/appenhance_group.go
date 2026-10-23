package main

// A group's details on the App enhancement page (appenhance_page.go): its
// name and scene prompt, its apps, its addresses and the key pressed after a
// paste.

import (
	"fmt"
	"slices"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// ---- The group ----

// aeSyncFields sets the fields from the group when another group shows,
// or the group changed elsewhere and nothing typed waits.
func (a *app) aeSyncFields(g appctx.Group) {
	p := &a.aePage
	if p.editFor == g.ID {
		return
	}
	p.editFor = g.ID
	p.name, p.prompt = g.Name, g.Prompt
	p.urlIn, p.urlErr = "", ""
	p.keyCustom, p.keyErr = false, ""
	p.keyText = g.AutoKey
	p.keySel = uitext.AutoKeyLabel(g.AutoKey)
}

func (a *app) aeDetail(c *ui.Context, g appctx.Group) {
	t := c.Theme()
	p := &a.aePage
	a.aeSyncFields(g)
	card(c.Key("ae-group")).Children(func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			in := ui.TextInput(c.Key("ae-name"), &p.name).Label(i18n.T("Group name")).Placeholder(i18n.T("Group name")).Grow(1).MinWidth(0)
			if in.Changed() {
				name := p.name
				a.updateGroup(g.ID, func(g *appctx.Group) { g.Name = name })
			}
			on := g.Enabled // derived: committed in OnChange, while it holds the new value
			ui.Switch(c.Key("ae-enabled"), &on).Label(i18n.T("Group on")).Tooltip(i18n.T("Group on")).OnChange(func() {
				v := on
				a.updateGroup(g.ID, func(g *appctx.Group) { g.Enabled = v })
			})
			moreMenu(c.Key("ae-more"), "", func(m *ui.Menu) {
				if m.Item(i18n.T("Delete the group…")).Chosen() {
					p.delAsk = true
				}
			})
		})
		// The features it applies to.
		ui.Row(c).Wrap().GapX(px(t, sp4)).GapY(px(t, sp1)).AlignItems(ui.Center).Children(func() {
			ui.Row(c).Gap(px(t, 6)).AlignItems(ui.Center).Children(func() {
				caption(c, i18n.T("Applies to"))
				infoTip(c, i18n.T("Transcribe: the scene shapes the clean-up. Rewrite: it helps read your spoken instruction. Translate: off by default, since a scene's tone rarely belongs in a translation."))
			})
			for _, f := range []struct {
				key, label string
				get        func(appctx.Features) bool
				set        func(*appctx.Features, bool)
			}{
				{"ae-f-transcribe", i18n.T("Transcribe"), func(f appctx.Features) bool { return f.Transcribe }, func(f *appctx.Features, v bool) { f.Transcribe = v }},
				{"ae-f-translate", i18n.T("Translate"), func(f appctx.Features) bool { return f.Translate }, func(f *appctx.Features, v bool) { f.Translate = v }},
				{"ae-f-rewrite", i18n.T("Rewrite"), func(f appctx.Features) bool { return f.Rewrite }, func(f *appctx.Features, v bool) { f.Rewrite = v }},
			} {
				v := f.get(g.Features)
				set := f.set
				ui.Checkbox(c.Key(f.key), &v, f.label).Label(i18n.T("Use the group for %s", f.label)).OnChange(func() {
					on := v
					a.updateGroup(g.ID, func(g *appctx.Group) { set(&g.Features, on) })
				})
			}
		})
	})
	a.aeScene(c, g)
	a.aeApps(c, g)
	a.aeURLs(c, g)
	a.aeAutoKey(c, g)
}

// aeScene is the scene prompt.
func (a *app) aeScene(c *ui.Context, g appctx.Group) {
	t := c.Theme()
	k := tok(t)
	p := &a.aePage
	sectionTitleTip(c, i18n.T("Scene prompt"), i18n.T("Added on top of the features' basic clean-up and your rules, never instead of them: say what is special here (tone, format, names)."))
	in := ui.TextArea(c.Key("ae-prompt"), &p.prompt).Label(i18n.T("Scene prompt of the group")).
		Placeholder(i18n.T("For example: this is a team chat; keep it short and casual, no greetings or sign-offs.")).MinHeight(px(t, 88))
	if in.Changed() {
		v := p.prompt
		a.updateGroup(g.ID, func(g *appctx.Group) { g.Prompt = v })
	}
	ui.Row(c.Key("ae-vars")).Wrap().Gap(px(t, sp1)).AlignItems(ui.Center).Children(func() {
		caption(c, i18n.T("Variables:"))
		for _, name := range compose.VariableNames {
			token := "{{" + name + "}}"
			b := ui.ButtonBase(c.Key("ae-var-"+name)).Padding(px(t, 2), px(t, 6)).Radius(px(t, 4)).Background(k.Hover).
				Label(token).Tooltip(variableInfo(name)).Shrink(0)
			if b.Hovered() {
				b.Background(k.Pressed)
			}
			b.Children(func() {
				ui.Text(c, token).FontSize(px(t, fsCaption)).Font("monospace").SingleLine()
			})
			if b.Clicked() {
				c.Services().WriteClipboard(token)
				a.toast(i18n.T("Copied"))
			}
		}
	})
}

// aeApps are the member apps.
func (a *app) aeApps(c *ui.Context, g appctx.Group) {
	t := c.Theme()
	k := tok(t)
	p := &a.aePage
	if sectionTitleAdd(c, "ae-add-app", i18n.T("Apps"), i18n.T("Dictating in one of these apps uses the group. A browser here covers the websites no website rule of any group matches."), i18n.T("Add apps…")) {
		p.appsOpen, p.appsQuery, p.bundleIn = true, "", ""
		a.loadApps()
	}
	card(c.Key("ae-apps")).Gap(px(t, sp1)).Children(func() {
		if len(g.Apps) == 0 {
			caption(c, i18n.T("No apps yet."))
		}
		for i, m := range g.Apps {
			ui.Row(c.Key("ae-app-" + m.BundleID)).Gap(px(t, sp2)).AlignItems(ui.Center).MinHeight(px(t, 28)).Children(func() {
				a.appIcon(c, proto.AppRef{BundleID: m.BundleID, Path: a.appPath(m.BundleID)}, 20, true)
				name := m.Name
				if name == "" {
					name = m.BundleID
				}
				ui.Text(c, name).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
				truncText(c, m.BundleID).FontSize(px(t, fsCaption)).TextColor(k.Text3).Grow(1)
				if appctx.IsBrowser(m.BundleID, a.prefs.AppEnhancement.Browsers) {
					tagChip(c, i18n.T("Browser"))
				}
				if iconButton(c.Key("remove"), icons.Close, i18n.T("Remove %s", name)).Clicked() {
					idx := i
					a.updateGroup(g.ID, func(g *appctx.Group) {
						if idx < len(g.Apps) {
							g.Apps = slices.Delete(g.Apps, idx, idx+1)
						}
					})
				}
			})
		}
	})
}

// aeURLs are the website rules.
func (a *app) aeURLs(c *ui.Context, g appctx.Group) {
	t := c.Theme()
	k := tok(t)
	p := &a.aePage
	sectionTitleTip(c, i18n.T("Websites"), i18n.T("In a browser, the address of the tab in front is matched against these rules: a site (github.com), a part of it (github.com/acme/*), any subdomain (*.notion.so). * stands for anything. A website rule wins over the browser's own group; the most specific rule wins."))
	card(c.Key("ae-urls")).Gap(px(t, sp1)).Children(func() {
		if len(g.URLs) == 0 {
			caption(c, i18n.T("No website rules yet."))
		}
		for i, u := range g.URLs {
			ui.Row(c.Key(fmt.Sprintf("ae-url-%d", i))).Gap(px(t, sp2)).AlignItems(ui.Center).MinHeight(px(t, 28)).Children(func() {
				ui.Icon(c, icons.LinkMinimalistic2.Linear).Size(px(t, 16), px(t, 16)).TextColor(k.Text2).Shrink(0)
				ui.Text(c, u).Font("monospace").FontSize(px(t, fsCaption)).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
				canon := appctx.CanonicalPattern(u)
				if err := appctx.CheckPattern(u); err != nil {
					iconMark(c, icons.DangerTriangle, k.Warning, uitext.PatternErrorText(err))
				} else if canon != strings.ToLower(strings.TrimSpace(u)) {
					truncText(c, i18n.T("matches %s", canon)).FontSize(px(t, fsCaption)).TextColor(k.Text3)
				}
				ui.Spacer(c)
				if iconButton(c.Key("remove"), icons.Close, i18n.T("Remove %s", u)).Clicked() {
					idx := i
					a.updateGroup(g.ID, func(g *appctx.Group) {
						if idx < len(g.URLs) {
							g.URLs = slices.Delete(g.URLs, idx, idx+1)
						}
					})
				}
			})
		}
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			in := ui.TextInput(c.Key("ae-url-in"), &p.urlIn).Label(i18n.T("New website rule")).Placeholder("github.com/acme/*").
				Font("monospace").Grow(1).MinWidth(0).MaxWidth(px(t, 360))
			if in.Changed() {
				p.urlErr = ""
			}
			add := ui.Button(c.Key("ae-url-add"), i18n.T("Add")).Label(i18n.T("Add the website rule")).Disabled(strings.TrimSpace(p.urlIn) == "")
			if add.Clicked() || in.Submitted() {
				a.aeAddURL(g)
			}
		})
		if p.urlErr != "" {
			warnLine(c, p.urlErr)
		}
	})
}

// aeAddURL adds the typed rule, if it is valid and new.
func (a *app) aeAddURL(g appctx.Group) {
	p := &a.aePage
	u := strings.TrimSpace(p.urlIn)
	if u == "" {
		return
	}
	if err := appctx.CheckPattern(u); err != nil {
		p.urlErr = uitext.PatternErrorText(err)
		return
	}
	if slices.ContainsFunc(g.URLs, func(v string) bool { return appctx.CanonicalPattern(v) == appctx.CanonicalPattern(u) }) {
		p.urlErr = i18n.T("This rule is in the group already.")
		return
	}
	a.updateGroup(g.ID, func(g *appctx.Group) { g.URLs = append(g.URLs, u) })
	p.urlIn, p.urlErr = "", ""
}

// autoKeys are the auto key choices besides None and Custom.
var autoKeys = []string{"Return", "Command+Return", "Shift+Return", "Control+Return", "Option+Return"}

// aeAutoKey is the key after inserting.
func (a *app) aeAutoKey(c *ui.Context, g appctx.Group) {
	t := c.Theme()
	p := &a.aePage
	sectionTitle(c, i18n.T("After inserting"))
	settingRow(c, "ae-key", i18n.T("Press a key"), i18n.T("Pressed right after the text is inserted, as when a chat app sends with Return. Not pressed when the result shows in the answer card."), func() {
		opts := []string{i18n.T("None")}
		byLabel := map[string]string{i18n.T("None"): ""}
		for _, k := range autoKeys {
			l := uitext.AutoKeyLabel(k)
			opts = append(opts, l)
			byLabel[l] = k
		}
		if cur := appctx.CanonicalAutoKey(g.AutoKey); cur != "" && !slices.Contains(autoKeys, cur) {
			l := uitext.AutoKeyLabel(cur)
			opts = append(opts, l)
			byLabel[l] = cur
		}
		opts = append(opts, i18n.T("Custom…"))
		if p.keyCustom {
			p.keySel = i18n.T("Custom…")
		} else if !slices.Contains(opts, p.keySel) {
			p.keySel = uitext.AutoKeyLabel(g.AutoKey)
		}
		if ui.Select(c.Key("ae-key-sel"), &p.keySel, opts).Label(i18n.T("Key after inserting")).Width(px(t, 180)).Changed() {
			if p.keySel == i18n.T("Custom…") {
				p.keyCustom, p.keyText, p.keyErr = true, g.AutoKey, ""
				return
			}
			keys := byLabel[p.keySel]
			p.keyCustom, p.keyErr = false, ""
			a.updateGroup(g.ID, func(g *appctx.Group) { g.AutoKey = keys })
		}
	})
	if p.keyCustom {
		in := ui.TextInput(c.Key("ae-key-custom"), &p.keyText).Placeholder("Command+Return").Label(i18n.T("Custom key after inserting")).Font("monospace").MaxWidth(px(t, 300))
		if in.Submitted() || in.Changed() && !in.Focused() {
			if _, err := appctx.ParseAutoKey(p.keyText); err != nil || strings.TrimSpace(p.keyText) == "" {
				p.keyErr = i18n.T("One key, with modifiers or not: Return, Command+Return, Tab.")
			} else {
				keys := appctx.CanonicalAutoKey(p.keyText)
				p.keyErr, p.keyCustom, p.keySel = "", false, uitext.AutoKeyLabel(keys)
				a.updateGroup(g.ID, func(g *appctx.Group) { g.AutoKey = keys })
			}
		}
		if p.keyErr != "" {
			warnLine(c, p.keyErr)
		} else {
			caption(c, i18n.T("Type the keys and press Return: Command+Return, Control+Return, Tab."))
		}
	}
}
