package main

// The Meetings page: a searchable list and an optional detail panel.

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

const (
	meetingListW      = 300
	meetingDetailMinW = 520
)

type meetingsLocalUI struct {
	query   string
	page    int
	find    finder
	listUI  ui.ListState
	setOpen bool
	delID   string
	detail  *meetingDetailLocalUI
}

func meetingsPage(c *ui.Context, state *meetingsViewState, actions meetingsActions, local *meetingsLocalUI, pageWidth float32) {
	t := c.Theme()
	query := strings.TrimSpace(local.query)
	var hits []int
	if query == "" {
		hits = make([]int, len(state.Items)+1)
	} else {
		hits = findHits(query, len(state.Items), func(index int) []string {
			if row := state.Items[index].Meeting; row != nil {
				return []string{row.Title, row.Snippet}
			}
			return nil
		})
	}
	if local.find.advance(hits[len(state.Items)]) {
		row, _ := rowOf(hits, local.find.cur)
		local.listUI.ScrollIntoView(row)
	}
	current, currentAt := rowOf(hits, local.find.cur)
	widePageHeader(c, i18n.T("Meetings"), func() {
		if state.MeetingCount > 0 || local.query != "" || local.page > 0 {
			if local.find.field(c, "meetings-search", &local.query, i18n.T("Search"), px(t, 240)).
				Label(i18n.T("Search meetings")).Changed() {
				actions.searchMeetings(local.query)
			}
			if hits[len(state.Items)] > 0 {
				local.find.count(c, query, hits[len(state.Items)])
			}
		}
		if state.SettingsAvailable && iconButton(c.Key("mt-settings"), icons.Tuning2, i18n.T("Meeting settings")).Shrink(0).Clicked() {
			local.setOpen = true
		}
		if state.Active {
			if primaryButton(c.Key("mt-end"), i18n.T("End meeting")).Clicked() {
				actions.meetingFinish()
			}
		} else if primaryButton(c.Key("mt-start"), i18n.T("Start meeting")).Clicked() {
			actions.startMeeting()
		}
	})
	pageBody(c).Children(func() {
		if state.Error != "" {
			ui.Column(c).Padding(0, px(t, sp6), px(t, sp3)).Shrink(0).Children(func() {
				notice(c, "mt-err", noticeError, i18n.T("Could not read the meetings."), state.Error, func() {
					if ghostButton(c.Key("mt-retry"), i18n.T("Try again")).Clicked() {
						actions.loadMeetings()
					}
				}, nil)
			})
		}
		split := pageWidth-px(t, meetingListW) >= px(t, meetingDetailMinW)
		ui.Row(c).Grow(1).MinHeight(0).AlignItems(ui.Stretch).Children(func() {
			if state.Detail == nil || split {
				column := ui.Column(c).MinWidth(0)
				if state.Detail != nil {
					column = column.Width(px(t, meetingListW)).Shrink(0)
				} else {
					column = column.Grow(1)
				}
				column.Children(func() { meetingsList(c, state, actions, local, query, current, currentAt) })
			}
			if state.Detail != nil {
				if split {
					ui.Box(c).Width(1).Background(tok(t).Line).Shrink(0)
				}
				meetingDetailSection(c, *state.Detail, actions, local)
			}
		})
	})
}

func meetingsList(c *ui.Context, state *meetingsViewState, actions meetingsActions, local *meetingsLocalUI, query string, current, currentAt int) {
	t := c.Theme()
	switch {
	case !state.Read:
	case state.MeetingCount == 0:
		meetingsEmpty(c, local, actions)
	default:
		ui.List(c.Key("meetings-list"), &local.listUI, len(state.Items)).Grow(1).MinHeight(0).Gap(px(t, 2)).Padding(0, 0, px(t, sp2)).
			ItemKey(func(index int) any { return state.Items[index].Key }).Rows(func(row ui.ListRow) {
			item := state.Items[row.Index]
			if item.Meeting != nil {
				at := -1
				if row.Index == current {
					at = currentAt
				}
				meetingRow(c, item.Meeting, query, at, actions)
			} else {
				ui.Text(c, item.Heading).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(tok(t).Text3).
					Padding(px(t, sp4), px(t, sp6), px(t, sp1))
			}
		})
		if local.page > 0 || state.HasMore {
			meetingsPager(c, state, actions, local)
		}
	}
}

func meetingsEmpty(c *ui.Context, local *meetingsLocalUI, actions meetingsActions) {
	if query := strings.TrimSpace(local.query); query != "" {
		emptyState(c, icons.Magnifer, i18n.T("Nothing matches “%s”", query),
			i18n.T("Try another search, or clear the field to see every meeting."), func() {
				if ghostButton(c.Key("meetings-clear"), i18n.T("Clear search")).Clicked() {
					actions.searchMeetings("")
				}
			})
		return
	}
	emptyState(c, meetingIcon, i18n.T("No meetings yet."),
		i18n.T("Start a meeting to record your microphone and, when supported, audio from other apps."), nil)
}

func meetingRow(c *ui.Context, row *meetingRowViewState, query string, at int, actions meetingsActions) {
	t := c.Theme()
	colors := tok(t)
	button := ui.ButtonBase(c.Key("mt-row-"+row.ID)).Padding(px(t, sp3), px(t, sp3)).Margin(0, px(t, sp4)).Radius(px(t, radiusControl)).
		Label(row.Title).Gap(px(t, sp2)).AlignItems(ui.Start)
	switch {
	case row.Selected:
		button.Background(colors.Pressed)
	case button.Hovered():
		button.Background(colors.Hover)
	}
	if button.Clicked() {
		if row.Selected {
			actions.closeMeetingDetail()
		} else {
			actions.openMeetingDetail(row.ID)
		}
	}
	button.Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp1)).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				markedText(c, row.Title, query, curIn(at, 0)).FontSize(px(t, fsBody+1)).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
				if row.StatusLabel != "" {
					tagChip(c, row.StatusLabel)
				}
				ui.Text(c, row.ClockText).FontSize(px(t, fsCaption)).TextColor(colors.Text3).SingleLine().Shrink(0)
			})
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				markedText(c, row.Snippet, query, curIn(at, len(findFolded(row.Title, query)))).FontSize(px(t, fsCaption)).TextColor(colors.Text2).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
				if row.DurationText != "" {
					ui.Text(c, row.DurationText).FontSize(px(t, fsCaption)).TextColor(colors.Text3).SingleLine().Shrink(0)
				}
			})
		})
	})
}

func meetingsPager(c *ui.Context, state *meetingsViewState, actions meetingsActions, local *meetingsLocalUI) {
	t := c.Theme()
	ui.Row(c).Padding(0, px(t, sp6), px(t, sp3)).Gap(px(t, sp2)).AlignItems(ui.Center).Justify(ui.End).Shrink(0).Children(func() {
		if ghostButton(c.Key("mt-page-prev"), i18n.T("Previous page")).Disabled(local.page == 0).Clicked() {
			actions.showMeetingsPage(local.page - 1)
		}
		caption(c, i18n.T("Page %d", local.page+1))
		if ghostButton(c.Key("mt-page-next"), i18n.T("Next page")).Disabled(!state.HasMore).Clicked() {
			actions.showMeetingsPage(local.page + 1)
		}
	})
}

func meetingDeleteDialog(c *ui.Context, local *meetingsLocalUI, actions meetingsActions) {
	open := local.delID != ""
	if !open {
		return
	}
	dialogFrame(c, "mt-del-dlg", &open, i18n.T("Delete this meeting?"), "", 420, func() {
		ui.Text(c, i18n.T("Its transcript, summary and audio are deleted. This cannot be undone."))
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("mt-del-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
			open = false
		}
		if recordButton(c.Key("mt-del-ok"), i18n.T("Delete meeting")).Clicked() {
			open = false
			actions.deleteMeeting(local.delID)
		}
	})
	if !open {
		local.delID = ""
	}
}
