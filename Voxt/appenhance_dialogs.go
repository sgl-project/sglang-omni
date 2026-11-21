package main

// The App enhancement page's dialogs. The controller supplies filtered rows,
// status text, and validation results; local UI owns the open/close controls.

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

func appEnhancementAppsDialogView(c *ui.Context, state *appEnhancementAppsDialog, actions appEnhancementActions, local *appEnhancementLocalUI, iconSource appEnhancementIconSource) {
	if state == nil {
		return
	}
	theme := c.Theme()
	colors := tok(theme)
	dialogFrame(c, "ae-apps-dlg", &local.appsOpen, state.Title, "", 480, func() {
		ui.Row(c).Children(func() { searchField(c, "ae-apps-search", &local.appsQuery, i18n.T("Search apps"), 0) })
		switch {
		case state.Loading:
			ui.Row(c).Gap(px(theme, sp2)).AlignItems(ui.Center).Children(func() {
				ui.Spinner(c).Size(px(theme, 16), px(theme, 16))
				caption(c, i18n.T("Looking for apps…"))
			})
		case state.Error != "":
			warnLine(c, state.Error)
		}
		ui.Column(c.Key("ae-apps-list")).Gap(px(theme, 2)).Children(func() {
			// Icons load as rows show: the first screenful at once, the others as they scroll into view.
			const eager = 12
			for index, app := range state.Rows {
				row := ui.ButtonBase(c.Key("ae-pick-"+app.Ref.BundleID)).Padding(px(theme, 6), px(theme, sp2)).Radius(px(theme, radiusControl)).
					Label(app.Ref.Name).Gap(px(theme, sp2)).AlignItems(ui.Center).Checked(app.Member)
				if row.Hovered() {
					row.Background(colors.Hover)
				}
				if row.Clicked() {
					actions.toggleAppEnhancementApp(app.Ref.BundleID)
				}
				visible := index < eager || row.InView()
				row.Children(func() {
					appIcon(c, iconSource, app.Ref, 20, visible)
					ui.Text(c, app.Ref.Name).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
					truncText(c, app.Ref.BundleID).FontSize(px(theme, fsCaption)).TextColor(colors.Text3).Grow(1)
					if app.Running {
						tagChip(c, i18n.T("Running"))
					}
					mark := ui.Box(c).Size(px(theme, 16), px(theme, 16)).Center().Shrink(0)
					if app.Member {
						mark.Children(func() {
							ui.Icon(c, icons.CheckCircle.Bold).Size(px(theme, 16), px(theme, 16)).TextColor(colors.Success)
						})
					}
				})
			}
		})
		if state.NoMatches {
			caption(c, i18n.T("No app matches."))
		}
		ui.Box(c).Height(1).Background(colors.Line).Shrink(0).Margin(px(theme, sp1), 0)
		caption(c, i18n.T("Not listed? Add it by its bundle ID:"))
		ui.Row(c).Gap(px(theme, sp2)).AlignItems(ui.Center).Children(func() {
			input := ui.TextInput(c.Key("ae-bundle-in"), &local.bundleIn).Label(i18n.T("Bundle ID")).Placeholder("com.example.app").Font("monospace").Grow(1).MinWidth(0)
			bundleID := strings.TrimSpace(local.bundleIn)
			valid := bundleID != "" && !strings.ContainsAny(bundleID, " /")
			if (ui.Button(c.Key("ae-bundle-add"), i18n.T("Add")).Label(i18n.T("Add bundle ID")).Disabled(!valid).Clicked() || input.Submitted()) && valid {
				actions.addAppEnhancementBundle(state.GroupID, bundleID)
			}
		})
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("ae-apps-done"), i18n.T("Done")).Clicked() {
			local.appsOpen = false
		}
	})
}

func appEnhancementBrowsersDialogView(c *ui.Context, state *appEnhancementBrowsersDialog, actions appEnhancementActions, local *appEnhancementLocalUI, iconSource appEnhancementIconSource) {
	if state == nil {
		return
	}
	theme := c.Theme()
	colors := tok(theme)
	dialogFrame(c, "ae-browsers-dlg", &local.browsersOpen, i18n.T("Browsers"), "", 520, func() {
		ui.Column(c.Key("ae-browser-list")).Gap(px(theme, 2)).Children(func() {
			for _, browser := range state.Rows {
				ui.Row(c.Key("ae-browser-" + browser.Ref.BundleID)).Gap(px(theme, sp2)).AlignItems(ui.Center).MinHeight(px(theme, 30)).Children(func() {
					color := colors.Text3
					switch browser.StatusTone {
					case "success":
						color = colors.Success
					case "warning":
						color = colors.Warning
					}
					appIcon(c, iconSource, browser.Ref, 24, true)
					ui.Column(c).Grow(1).MinWidth(0).Children(func() {
						ui.Text(c, browser.Ref.Name).SingleLine().Ellipsis("…").MinWidth(0)
						truncText(c, browser.Ref.BundleID).FontSize(px(theme, fsMicro)).TextColor(colors.Text3)
					})
					statusDot(c, color).Tooltip(browser.StatusText)
					truncText(c, browser.StatusText).FontSize(px(theme, fsCaption)).TextColor(colors.Text2).MaxWidth(px(theme, 200))
					if browser.Custom {
						if iconButton(c.Key("remove"), icons.Close, i18n.T("Remove %s", browser.Ref.Name)).Clicked() {
							actions.removeAppEnhancementBrowser(browser.Ref.BundleID)
						}
					}
				})
			}
		})
		if state.OthersText != "" {
			caption(c, state.OthersText).TextColor(colors.Text3)
		}
		ui.Box(c).Height(1).Background(colors.Line).Shrink(0).Margin(px(theme, sp1), 0)
		ui.Row(c).Gap(px(theme, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			ui.TextInput(c.Key("ae-br-id"), &local.brID).Label(i18n.T("Bundle ID")).Placeholder("com.example.browser").Font("monospace").Grow(1).Basis(px(theme, 180)).MinWidth(0)
			ui.TextInput(c.Key("ae-br-name"), &local.brName).Label(i18n.T("Name")).Placeholder(i18n.T("Name")).Width(px(theme, 140))
			bundleID := strings.TrimSpace(local.brID)
			invalid := bundleID == "" || strings.ContainsAny(bundleID, " /")
			unchangedInvalid := state.AddDisabled && bundleID == strings.TrimSpace(state.BundleInput)
			if ui.Button(c.Key("ae-br-add"), i18n.T("Add")).Label(i18n.T("Add browser")).Disabled(invalid || unchangedInvalid).Clicked() {
				actions.addAppEnhancementBrowser(local.brID, local.brName)
			}
		})
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("ae-browsers-done"), i18n.T("Done")).Clicked() {
			local.browsersOpen = false
		}
	})
}

func appEnhancementDeleteDialogView(c *ui.Context, state *appEnhancementDeleteDialog, actions appEnhancementActions, local *appEnhancementLocalUI) {
	if state == nil {
		return
	}
	dialogFrame(c, "ae-del", &local.delAsk, state.Title, "", 420, func() {
		ui.Text(c, i18n.T("Its apps and websites go back to the features' own rules. This cannot be undone."))
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("ae-del-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
			local.delAsk = false
		}
		if recordButton(c.Key("ae-del-ok"), i18n.T("Delete")).Clicked() {
			local.delAsk = false
			actions.deleteAppEnhancementGroup(state.GroupID)
		}
	})
}
