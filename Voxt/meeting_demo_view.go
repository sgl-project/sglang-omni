package main

import (
	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func meetingDemoContent(host *meetingDemoHost, localize func()) mygo.Content {
	return ui.View(func(c *ui.Context) {
		if localize != nil {
			localize()
		}
		meetingDemoView(c, host)
	})
}

func meetingDemoView(c *ui.Context, host *meetingDemoHost) {
	fixture := host.current()
	ui.Column(c).Fill().Children(func() {
		ui.Row(c).Padding(8, 12).Gap(8).AlignItems(ui.Center).Shrink(0).Children(func() {
			if ghostButton(c.Key("demo-previous"), i18n.T("Back")).Clicked() {
				host.selectMeetingFixture(-1)
			}
			if ghostButton(c.Key("demo-next"), i18n.T("Continue")).Clicked() {
				host.selectMeetingFixture(1)
			}
			caption(c, fixture.name)
		})
		ui.Box(c).Grow(1).Children(func() {
			switch fixture.state.Kind {
			case "page":
				meetingsPage(c, fixture.state.Page, host, &host.pageUI, 800)
				meetingDeleteDialog(c, &host.pageUI, host)
			case "overlay":
				state := *fixture.state.Overlay
				state.Collapsed = host.overlayUI.collapsed
				state.ShowExpanded = state.ShowExpanded && !state.Collapsed
				if state.Visible {
					model := meetingOverlayLayoutModel(state, host.overlayUI.menu)
					meetingOverlayView(c, state, model, host, &host.overlayUI, meetingOverlayAnimation{}, host.faceStyle(model))
				}
			case "start":
				ui.Column(c).Padding(24).Children(func() {
					fillCard(c).Padding(16).Children(func() {
						ui.Text(c, fixture.state.Start.Text)
					})
				})
			case "history":
				historyPage(c, fixture.state.History, host, &host.historyUI, 800)
				historyDeleteDialog(c, &host.historyUI, host)
			case "history-settings":
				historySettingsDialog(c, fixture.state.HistorySettings, host, &host.historySettingsUI)
			case "history-access":
				ui.Scroll(c.Key("demo-history-access")).Grow(1).Children(func() {
					pageColumn(c).Children(func() {
						historyAccessSection(c, fixture.state.HistoryAccess, host, &host.historyAccessUI)
					})
				})
			case "appenhance":
				appEnhancementPage(c, appEnhancementDemoState(host), host, &host.appEnhancementUI, 800, host)
			}
		})
	})
}

func appEnhancementDemoState(host *meetingDemoHost) *appEnhancementViewState {
	fixture := host.current()
	state := *fixture.state.AppEnhancement
	local := &host.appEnhancementUI
	if local.sel != state.Selected {
		state.Selected = local.sel
		state.Groups = append([]appEnhancementGroupRow(nil), state.Groups...)
		for index := range state.Groups {
			state.Groups[index].Selected = state.Groups[index].ID == local.sel
		}
		for _, detail := range host.appEnhancementDetails[fixture.locale] {
			if detail.ID == local.sel {
				detail := detail
				state.Detail = &detail
				break
			}
		}
	}
	if !local.appsOpen {
		state.Apps = nil
	} else if state.Apps == nil && state.Detail != nil {
		state.Apps = &appEnhancementAppsDialog{
			GroupID: state.Detail.ID, Title: i18n.T("Add apps to %s", state.Detail.Name), BundleInput: local.bundleIn,
		}
	}
	if !local.browsersOpen {
		state.Browsers = nil
	} else if state.Browsers == nil {
		state.Browsers = &appEnhancementBrowsersDialog{BundleInput: local.brID, NameInput: local.brName, AddDisabled: true}
	}
	if !local.delAsk {
		state.Delete = nil
	} else if state.Delete == nil && state.Detail != nil {
		state.Delete = &appEnhancementDeleteDialog{GroupID: state.Detail.ID, Title: i18n.T("Delete “%s”?", state.Detail.Name)}
	}
	return &state
}
