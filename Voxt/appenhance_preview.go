package main

// App enhancement's preview: the frontmost app, its address, the matching
// group, and shortcuts for adding that app or site to the selected group.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

func appEnhancementPreviewCard(c *ui.Context, state *appEnhancementPreview, actions appEnhancementActions, iconSource appEnhancementIconSource) {
	theme := c.Theme()
	colors := tok(theme)
	fillCard(c.Key("ae-preview")).Padding(px(theme, sp3), px(theme, sp4)).Gap(px(theme, sp2)).Children(func() {
		ui.Row(c).Gap(px(theme, sp2)).AlignItems(ui.Center).Children(func() {
			if state.Empty {
				ui.Icon(c, icons.WindowFrame.Linear).Size(px(theme, 16), px(theme, 16)).TextColor(colors.Text2).Shrink(0)
			} else {
				appIcon(c, iconSource, state.App, 20, true)
			}
			if state.Empty {
				ui.Text(c, state.FrontText).TextColor(colors.Text2).Grow(1).MinWidth(0)
			} else {
				ui.Text(c, state.FrontText).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
				if state.Browser {
					tagChip(c, i18n.T("Browser"))
				}
				ui.Spacer(c)
			}
			if iconButton(c.Key("ae-preview-refresh"), icons.Refresh, i18n.T("Read again")).Disabled(!state.CanRefresh).Clicked() {
				actions.refreshAppEnhancementPreview()
			}
		})
		if state.Empty {
			return
		}
		if state.Browser {
			truncText(c, state.Address).FontSize(px(theme, fsCaption)).TextColor(colors.Text2)
		}
		ui.Row(c).Gap(px(theme, sp2)).AlignItems(ui.Center).Children(func() {
			if state.Matched {
				ui.Icon(c, icons.Target.Linear).Size(px(theme, 14), px(theme, 14)).TextColor(colors.Accent).Shrink(0)
				ui.Text(c, state.MatchText).FontSize(px(theme, fsCaption)).Grow(1).MinWidth(0)
			} else {
				ui.Text(c, state.MatchText).FontSize(px(theme, fsCaption)).TextColor(colors.Text2).Grow(1).MinWidth(0)
			}
		})
		if !state.AddApp && !state.AddSite {
			return
		}
		ui.Row(c).Gap(px(theme, sp1)).AlignItems(ui.Center).Wrap().Margin(0, 0, 0, -px(theme, 4)).Children(func() {
			if state.AddApp && quietButton(c.Key("ae-preview-add-app"), icons.Add, state.AddAppLabel).Clicked() {
				actions.addAppEnhancementPreviewApp()
			}
			if state.AddSite && quietButton(c.Key("ae-preview-add-site"), icons.Add, state.AddSiteLabel).Clicked() {
				actions.addAppEnhancementPreviewSite()
			}
		})
	})
}
