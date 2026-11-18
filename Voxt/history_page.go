package main

// The History page (/history): a searchable list and an optional detail panel.

import (
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

const (
	historyPanelW   = 290
	historyListMinW = 300
)

type historyLocalUI struct {
	tab    int
	query  string
	find   finder
	sel    string
	listUI ui.ListState
	delAsk bool
	delID  string
}

func historyPage(c *ui.Context, state *historyViewState, actions historyActions, local *historyLocalUI, pageWidth float32) {
	t := c.Theme()
	query := strings.TrimSpace(local.query)
	hits := findHits(query, len(state.Rows), func(index int) []string {
		return []string{state.Rows[index].Title}
	})
	if local.find.advance(hits[len(state.Rows)]) {
		row, _ := rowOf(hits, local.find.cur)
		local.listUI.ScrollIntoView(row)
	}
	current, currentAt := rowOf(hits, local.find.cur)
	widePageHeader(c, i18n.T("History"), func() {
		if state.CountLabel != "" {
			caption(c, state.CountLabel).SingleLine()
		}
	})
	pageBody(c).Children(func() {
		ui.Row(c).Padding(px(t, sp1), px(t, sp6), px(t, sp3)).Gap(px(t, sp2)).AlignItems(ui.Center).Shrink(0).Children(func() {
			if ui.Segmented(c.Key("hist-tab"), &local.tab, state.Tabs...).Label(i18n.T("Feature")).Shrink(0).Changed() {
				actions.showHistoryTab(local.tab)
				state.Detail = nil
			}
			if local.find.field(c, "hist-search", &local.query, i18n.T("Search"), px(t, 360)).Label(i18n.T("Search history")).Changed() {
				actions.searchHistory(local.query)
			}
			if hits[len(state.Rows)] > 0 {
				local.find.count(c, query, hits[len(state.Rows)])
			}
			if state.SettingsAvailable && iconButton(c.Key("hist-settings"), icons.Tuning2, i18n.T("History settings")).Shrink(0).Clicked() {
				actions.openHistorySettingsDialog()
			}
		})
		if state.FolderNotice != "" {
			ui.Column(c).Padding(0, px(t, sp6), px(t, sp3)).Shrink(0).Children(func() {
				notice(c, "hist-missing", noticeWarn, state.FolderNotice, "", func() {
					if ghostButton(c.Key("hist-missing-open"), i18n.T("History settings")).Clicked() {
						actions.openHistorySettingsDialog()
					}
				}, nil)
			})
		}
		if !state.KeepHistory {
			ui.Column(c).Padding(0, px(t, sp6), px(t, sp3)).Shrink(0).Children(func() {
				notice(c, "hist-off", noticeInfo, i18n.T("History is off, so new results are not kept."), "", func() {
					if ghostButton(c.Key("hist-privacy"), i18n.T("Privacy settings")).Clicked() {
						actions.openHistoryPrivacy()
					}
				}, nil)
			})
		}
		if state.Error != "" {
			ui.Column(c).Padding(0, px(t, sp6), px(t, sp3)).Shrink(0).Children(func() {
				notice(c, "hist-err", noticeError, i18n.T("Could not read the history."), state.Error, func() {
					if ghostButton(c.Key("hist-retry"), i18n.T("Try again")).Clicked() {
						actions.loadHistory()
					}
				}, nil)
			})
		}
		width := px(t, historyPanelW)
		inline := pageWidth-width >= px(t, historyListMinW)
		ui.Row(c).Grow(1).MinHeight(0).AlignItems(ui.Stretch).Children(func() {
			ui.Column(c).Grow(1).MinWidth(0).Children(func() {
				switch {
				case !state.Read:
				case state.Empty != nil:
					historyEmpty(c, state.Empty, actions)
				default:
					c.After(time.Minute)
					ui.List(c.Key("hist-list"), &local.listUI, len(state.Rows)).Grow(1).MinHeight(0).Gap(px(t, 2)).Padding(0, 0, px(t, sp4)).
						ItemKey(func(index int) any { return state.Rows[index].ID }).Rows(func(row ui.ListRow) {
						at := -1
						if row.Index == current {
							at = currentAt
						}
						historyRow(c, &state.Rows[row.Index], query, at, actions, local)
					})
				}
			})
			if state.Detail != nil {
				closePanel := func() {
					if !local.delAsk {
						actions.closeHistoryDetail()
					}
				}
				sidePanel(c, "hist-detail", width, inline, i18n.T("Entry details"), closePanel, func() {
					historyDetail(c, state.Detail, actions, local)
				})
			}
		})
	})
}

func historyEmpty(c *ui.Context, state *historyEmptyViewState, actions historyActions) {
	if state.Search {
		emptyState(c, icons.Magnifer, state.Title, "", func() {
			if ghostButton(c.Key("hist-clear-search"), state.ActionLabel).Clicked() {
				actions.searchHistory("")
			}
		})
		return
	}
	icon := icons.History
	key := "hist-go-transcribe"
	switch state.Kind {
	case "translate":
		icon = translateIcon
		key = "hist-go-translate"
	case "rewrite":
		icon = rewriteIcon
		key = "hist-go-rewrite"
	}
	var action func()
	if state.ActionLabel != "" {
		action = func() {
			if ghostButton(c.Key(key), state.ActionLabel).Clicked() {
				actions.openHistoryFeature(state.Kind)
			}
		}
	}
	emptyState(c, icon, state.Title, state.Body, action)
}

func historyRow(c *ui.Context, state *historyRowViewState, query string, at int, actions historyActions, local *historyLocalUI) {
	t := c.Theme()
	colors := tok(t)
	row := ui.ButtonBase(c.Key("hist-row-"+state.ID)).Padding(px(t, sp2), px(t, sp2)).Margin(0, px(t, sp4)).Radius(px(t, radiusControl)).
		Label(state.Title).Gap(px(t, sp2)).AlignItems(ui.Start).Role(ui.RoleGroup)
	switch {
	case state.Selected:
		row.Background(colors.Pressed)
	case row.Hovered():
		row.Background(colors.Hover)
	}
	if row.Clicked() {
		if state.Selected {
			actions.closeHistoryDetail()
		} else {
			actions.openHistoryDetail(state.ID)
		}
	}
	entryMenu(c, row, state, actions, local, true, func() { actions.openHistoryDetail(state.ID) })
	row.Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 3)).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				text := markedText(c, state.Title, query, at).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
				if state.FailedTitle {
					text.TextColor(colors.Text2)
				}
				entryStatus(c, state, actions)
			})
			ui.Text(c, state.Meta).FontSize(px(t, fsCaption)).TextColor(colors.Text3).SingleLine().Ellipsis("…").MinWidth(0)
		})
	})
}

func entryStatus(c *ui.Context, state *historyRowViewState, actions historyActions) {
	t := c.Theme()
	colors := tok(t)
	switch {
	case state.Retry.Active:
		ui.Row(c.Key("hist-retrying")).Gap(px(t, 4)).AlignItems(ui.Center).Shrink(0).Children(func() {
			ui.Spinner(c).Size(px(t, 12), px(t, 12))
			ui.Text(c, state.Retry.StageText).FontSize(px(t, fsMicro)).TextColor(colors.Text2).SingleLine()
		})
	case state.Retry.Available && state.Retry.Wanted:
		button := ui.ButtonBase(c.Key("hist-retry-chip")).Padding(px(t, 1), px(t, 6)).Radius(px(t, 4)).Gap(px(t, 3)).AlignItems(ui.Center).Shrink(0).
			Label(i18n.T("Retry")).Tooltip(i18n.T("Recognise the recording again")).Disabled(state.Retry.Disabled)
		background := colors.Hover
		if button.Hovered() && !state.Retry.Disabled {
			background = colors.Pressed
		}
		button.Background(background).Children(func() {
			ui.Icon(c, icons.Refresh.Linear).Size(px(t, 11), px(t, 11)).TextColor(colors.Danger)
			ui.Text(c, i18n.T("Retry")).FontSize(px(t, fsMicro)).TextColor(colors.Danger).SingleLine()
		})
		if button.Clicked() {
			actions.retryHistoryEntry(state.ID)
		}
	default:
		entryMarks(c, state.Marks)
	}
}

func entryMarks(c *ui.Context, marks []historyMarkViewState) {
	colors := tok(c.Theme())
	for _, mark := range marks {
		chip := tagChip(c, mark.Label).Tooltip(mark.Tooltip)
		if mark.Tone == "danger" {
			chip.TextColor(colors.Danger)
		}
	}
}

func entryRetryButton(c *ui.Context, state *historyDetailViewState, actions historyActions) {
	t := c.Theme()
	if !state.Retry.Available {
		return
	}
	if state.Retry.Active {
		ui.Row(c.Key("hist-detail-retrying")).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			ui.Spinner(c).Size(px(t, 16), px(t, 16))
			caption(c, state.Retry.StageText)
		})
		return
	}
	button := secondaryIconButton(c.Key("hist-detail-retry"), icons.Refresh, state.Retry.Label).Label(state.Retry.Label).Disabled(state.Retry.Disabled)
	if state.Retry.Tooltip != "" {
		button.Tooltip(state.Retry.Tooltip)
	}
	if button.Clicked() {
		actions.retryHistoryEntry(state.ID)
	}
}
