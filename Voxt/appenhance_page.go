package main

// The App enhancement page (/app-enhancement): the switch, groups, the
// selected group's editor, live preview, and dialogs. Controllers own every
// rule, setting, helper, and store change; this file only renders view state.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

const (
	aeListW    = 200
	aeMainMinW = 380
)

type appEnhancementLocalUI struct {
	sel  string
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
	// The Add apps dialog's local controls.
	appsOpen  bool
	appsQuery string
	bundleIn  string
	// The Browsers dialog's local controls.
	browsersOpen bool
	brID, brName string
	// The Delete dialog.
	delAsk bool
}

func appEnhancementPage(c *ui.Context, state *appEnhancementViewState, actions appEnhancementActions, local *appEnhancementLocalUI, pageWidth float32, iconSource appEnhancementIconSource) {
	theme := c.Theme()
	inline, shown := local.list.layout(pageWidth, px(theme, aeListW), px(theme, aeMainMinW))
	widePageHeader(c, i18n.T("App enhancement"), func() {
		if !inline {
			local.list.toggle(c.Key("ae-list-toggle"), icons.SidebarMinimalistic, shown, i18n.T("Show groups"), i18n.T("Hide groups"))
		}
		caption(c, state.EnabledText).SingleLine()
		on := state.Enabled
		if ui.Switch(c.Key("ae-on"), &on).Label(i18n.T("Use App enhancement")).Tooltip(i18n.T("Use App enhancement")).Changed() {
			actions.setAppEnhancementEnabled(on)
		}
		if primaryIconButton(c.Key("ae-new"), icons.Add, i18n.T("New group")).Clicked() {
			actions.addAppEnhancementGroup()
		}
		if iconButton(c.Key("ae-browsers"), icons.Global, i18n.T("Browsers")).Tooltip(i18n.T("Choose which apps are browsers")).Clicked() {
			actions.openAppEnhancementBrowsers()
		}
	})
	pageBody(c).Children(func() {
		ui.Row(c).Grow(1).MinHeight(0).AlignItems(ui.Stretch).Children(func() {
			local.list.view(c, "ae-groups", i18n.T("Groups"), inline, shown, true, px(theme, aeListW), func() {
				appEnhancementGroupList(c, state, local)
			})
			ui.Scroll(c.Key("ae-main")).Grow(1).MinWidth(0).Children(func() {
				ui.Column(c).Padding(0, px(theme, sp6), px(theme, sp6)).Gap(px(theme, sp3)).Children(func() {
					appEnhancementPreviewCard(c, &state.Preview, actions, iconSource)
					if !state.Enabled {
						notice(c, "ae-off", noticeInfo, i18n.T("App enhancement is off, so every app uses only the features' own rules."), "", nil, nil)
					}
					if state.Error != "" {
						notice(c, "ae-err", noticeError, i18n.T("Could not read the groups."), state.Error, nil, nil)
					}
					if state.Detail == nil {
						appEnhancementEmpty(c, actions)
						return
					}
					appEnhancementDetail(c, state.Detail, actions, local, iconSource)
				})
			})
		})
	})
	appEnhancementAppsDialogView(c, state.Apps, actions, local, iconSource)
	appEnhancementBrowsersDialogView(c, state.Browsers, actions, local, iconSource)
	appEnhancementDeleteDialogView(c, state.Delete, actions, local)
}

func appEnhancementEmpty(c *ui.Context, actions appEnhancementActions) {
	emptyState(c, appEnhancementIcon, i18n.T("Rules for the frontmost app or website"),
		i18n.T("Give apps and websites their own prompt, such as a casual tone in chats."), func() {
			if ghostIconButton(c.Key("ae-empty-new"), icons.Add, i18n.T("New group")).Clicked() {
				actions.addAppEnhancementGroup()
			}
		})
}

func appEnhancementGroupList(c *ui.Context, state *appEnhancementViewState, local *appEnhancementLocalUI) {
	theme := c.Theme()
	colors := tok(theme)
	ui.Column(c).Padding(px(theme, sp2), px(theme, sp2), px(theme, sp4)).Gap(px(theme, 2)).Children(func() {
		for _, group := range state.Groups {
			row := ui.ButtonBase(c.Key("ae-group-"+group.ID)).Padding(px(theme, 6), px(theme, sp2)).Radius(px(theme, radiusControl)).
				Label(group.Name).Gap(px(theme, sp2)).AlignItems(ui.Center)
			switch {
			case group.Selected:
				row.Background(colors.Pressed)
			case row.Hovered():
				row.Background(colors.Hover)
			}
			if row.Clicked() {
				local.sel = group.ID
				local.list.open = false
			}
			row.Children(func() {
				color := colors.Success
				if !group.Enabled {
					color = colors.G[5]
				}
				statusDot(c, color)
				ui.Column(c).Grow(1).MinWidth(0).Gap(px(theme, 1)).Children(func() {
					text := ui.Text(c, group.Name).SingleLine().Ellipsis("…").MinWidth(0)
					if !group.Enabled {
						text.TextColor(colors.Text2)
					}
					ui.Text(c, group.Summary).FontSize(px(theme, fsCaption)).TextColor(colors.Text3).SingleLine().Ellipsis("…").MinWidth(0)
				})
				if group.Matches {
					iconMark(c, icons.Target, colors.Accent, i18n.T("Matches the frontmost app"))
				}
			})
		}
	})
}
