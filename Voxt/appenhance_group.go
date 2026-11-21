package main

// A selected group's editor. Draft text and dialog choices stay in local UI
// state; committed values and display options come from view state.

import (
	"fmt"
	"slices"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

func appEnhancementSyncFields(state *appEnhancementGroupDetail, local *appEnhancementLocalUI) {
	if local.editFor == state.ID {
		return
	}
	local.editFor = state.ID
	local.name, local.prompt = state.Name, state.Prompt
	local.urlIn, local.urlErr = state.URLInput, state.URLError
	local.keyCustom, local.keyErr = state.AutoKey.Custom, state.AutoKey.Error
	local.keyText = state.AutoKey.Text
	local.keySel = state.AutoKey.CurrentLabel
}

func appEnhancementDetail(c *ui.Context, state *appEnhancementGroupDetail, actions appEnhancementActions, local *appEnhancementLocalUI, iconSource appEnhancementIconSource) {
	theme := c.Theme()
	appEnhancementSyncFields(state, local)
	card(c.Key("ae-group")).Children(func() {
		ui.Row(c).Gap(px(theme, sp2)).AlignItems(ui.Center).Children(func() {
			input := ui.TextInput(c.Key("ae-name"), &local.name).Label(i18n.T("Group name")).Placeholder(i18n.T("Group name")).Grow(1).MinWidth(0)
			if input.Changed() {
				actions.setAppEnhancementGroupName(state.ID, local.name)
			}
			on := state.Enabled
			ui.Switch(c.Key("ae-enabled"), &on).Label(i18n.T("Group enabled")).Tooltip(i18n.T("Group enabled")).OnChange(func() {
				actions.setAppEnhancementGroupEnabled(state.ID, on)
			})
			moreMenu(c.Key("ae-more"), "", func(menu *ui.Menu) {
				if menu.Item(i18n.T("Delete group…")).Chosen() {
					local.delAsk = true
				}
			})
		})
		ui.Row(c).Wrap().GapX(px(theme, sp4)).GapY(px(theme, sp1)).AlignItems(ui.Center).Children(func() {
			caption(c, i18n.T("Applies to"))
			for _, feature := range state.Features {
				on := feature.Checked
				ui.Checkbox(c.Key("ae-f-"+feature.ID), &on, feature.Label).Label(i18n.T("Use the group for %s", feature.Label)).OnChange(func() {
					actions.setAppEnhancementGroupFeature(state.ID, feature.ID, on)
				})
			}
		})
	})
	appEnhancementScene(c, state, actions, local)
	appEnhancementApps(c, state, actions, iconSource)
	appEnhancementURLs(c, state, actions, local)
	appEnhancementAutoKeyEditor(c, state, actions, local)
}

func appEnhancementScene(c *ui.Context, state *appEnhancementGroupDetail, actions appEnhancementActions, local *appEnhancementLocalUI) {
	theme := c.Theme()
	colors := tok(theme)
	sectionTitleTip(c, i18n.T("Group prompt"), i18n.T("Added on top of each feature's own rules."))
	input := ui.TextArea(c.Key("ae-prompt"), &local.prompt).Label(i18n.T("Prompt for this group")).
		Placeholder(i18n.T("For example: this is a team chat; keep it short and casual, no greetings or sign-offs.")).MinHeight(px(theme, 88))
	if input.Changed() {
		actions.setAppEnhancementGroupPrompt(state.ID, local.prompt)
	}
	ui.Row(c.Key("ae-vars")).Wrap().Gap(px(theme, sp1)).AlignItems(ui.Center).Children(func() {
		caption(c, i18n.T("Variables:"))
		for _, variable := range state.Variables {
			button := ui.ButtonBase(c.Key("ae-var-"+variable.Name)).Padding(px(theme, 2), px(theme, 6)).Radius(px(theme, 4)).Background(colors.Hover).
				Label(variable.Token).Tooltip(variable.Tooltip).Shrink(0)
			if button.Hovered() {
				button.Background(colors.Pressed)
			}
			button.Children(func() {
				ui.Text(c, variable.Token).FontSize(px(theme, fsCaption)).Font("monospace").SingleLine()
			})
			if button.Clicked() {
				actions.copyAppEnhancementVariable(variable.Token)
			}
		}
	})
}

func appEnhancementApps(c *ui.Context, state *appEnhancementGroupDetail, actions appEnhancementActions, iconSource appEnhancementIconSource) {
	theme := c.Theme()
	colors := tok(theme)
	if sectionTitleAdd(c, "ae-add-app", i18n.T("Apps"), i18n.T("A browser added here covers the websites no website rule matches."), i18n.T("Add apps…")) {
		actions.openAppEnhancementApps()
	}
	card(c.Key("ae-apps")).Gap(px(theme, sp1)).Children(func() {
		if len(state.Apps) == 0 {
			caption(c, i18n.T("No apps yet."))
		}
		for _, app := range state.Apps {
			ui.Row(c.Key("ae-app-" + app.Ref.BundleID)).Gap(px(theme, sp2)).AlignItems(ui.Center).MinHeight(px(theme, 28)).Children(func() {
				appIcon(c, iconSource, app.Ref, 20, true)
				ui.Text(c, app.Ref.Name).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
				truncText(c, app.Ref.BundleID).FontSize(px(theme, fsCaption)).TextColor(colors.Text3).Grow(1)
				if app.Browser {
					tagChip(c, i18n.T("Browser"))
				}
				if iconButton(c.Key("remove"), icons.Close, i18n.T("Remove %s", app.Ref.Name)).Clicked() {
					actions.removeAppEnhancementApp(state.ID, app.Ref.BundleID)
				}
			})
		}
	})
}

func appEnhancementURLs(c *ui.Context, state *appEnhancementGroupDetail, actions appEnhancementActions, local *appEnhancementLocalUI) {
	theme := c.Theme()
	colors := tok(theme)
	sectionTitleTip(c, i18n.T("Websites"), i18n.T("Match a site (github.com), part of it (github.com/acme/*) or its subdomains (*.notion.so). A website rule wins over the browser's group."))
	card(c.Key("ae-urls")).Gap(px(theme, sp1)).Children(func() {
		if len(state.URLs) == 0 {
			caption(c, i18n.T("No website rules yet."))
		}
		for index, pattern := range state.URLs {
			ui.Row(c.Key(fmt.Sprintf("ae-url-%d", index))).Gap(px(theme, sp2)).AlignItems(ui.Center).MinHeight(px(theme, 28)).Children(func() {
				ui.Icon(c, icons.LinkMinimalistic2.Linear).Size(px(theme, 16), px(theme, 16)).TextColor(colors.Text2).Shrink(0)
				ui.Text(c, pattern.Text).Font("monospace").FontSize(px(theme, fsCaption)).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
				if pattern.Warning != "" {
					iconMark(c, icons.DangerTriangle, colors.Warning, pattern.Warning)
				} else if pattern.Canonical != "" {
					truncText(c, pattern.Canonical).FontSize(px(theme, fsCaption)).TextColor(colors.Text3)
				}
				ui.Spacer(c)
				if iconButton(c.Key("remove"), icons.Close, i18n.T("Remove %s", pattern.Text)).Clicked() {
					actions.removeAppEnhancementURL(state.ID, index)
				}
			})
		}
		ui.Row(c).Gap(px(theme, sp2)).AlignItems(ui.Center).Children(func() {
			input := ui.TextInput(c.Key("ae-url-in"), &local.urlIn).Label(i18n.T("New website rule")).Placeholder("github.com/acme/*").
				Font("monospace").Grow(1).MinWidth(0).MaxWidth(px(theme, 360))
			if input.Changed() {
				local.urlErr = ""
			}
			add := ui.Button(c.Key("ae-url-add"), i18n.T("Add")).Label(i18n.T("Add website rule")).Disabled(strings.TrimSpace(local.urlIn) == "")
			if add.Clicked() || input.Submitted() {
				actions.addAppEnhancementURL(state.ID, local.urlIn)
			}
		})
		if local.urlErr != "" {
			warnLine(c, local.urlErr)
		}
	})
}

func appEnhancementAutoKeyEditor(c *ui.Context, state *appEnhancementGroupDetail, actions appEnhancementActions, local *appEnhancementLocalUI) {
	theme := c.Theme()
	sectionTitle(c, i18n.T("After inserting"))
	settingRow(c, "ae-key", i18n.T("Press a key"), i18n.T("For example, Return to send in a chat app."), func() {
		options := make([]string, len(state.AutoKey.Options))
		labels := make(map[string]string, len(state.AutoKey.Options))
		for index, option := range state.AutoKey.Options {
			options[index] = option.Label
			labels[option.Label] = option.Value
		}
		if local.keyCustom {
			local.keySel = state.AutoKey.CustomLabel
		} else if !slices.Contains(options, local.keySel) {
			local.keySel = state.AutoKey.CurrentLabel
		}
		if ui.Select(c.Key("ae-key-sel"), &local.keySel, options).Label(i18n.T("Key after inserting")).Width(px(theme, 180)).Changed() {
			if local.keySel == state.AutoKey.CustomLabel {
				local.keyCustom, local.keyText, local.keyErr = true, state.AutoKey.Value, ""
				return
			}
			actions.setAppEnhancementAutoKey(state.ID, labels[local.keySel])
		}
	})
	if local.keyCustom {
		input := ui.TextInput(c.Key("ae-key-custom"), &local.keyText).Placeholder("Command+Return").Label(i18n.T("Custom key after inserting")).Font("monospace").MaxWidth(px(theme, 300))
		if input.Submitted() || input.Changed() && !input.Focused() {
			actions.commitAppEnhancementCustomAutoKey(state.ID, local.keyText)
		}
		if local.keyErr != "" {
			warnLine(c, local.keyErr)
		}
	}
}
