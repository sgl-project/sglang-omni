package main

// Meeting detail view: summary, follow-ups, export and transcript.

import (
	"fmt"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

const meetingDetailMaxW = 960

type meetingDetailLocalUI struct {
	question string
	segList  ui.ListState
	// loadingSince is when the detail began loading (loadingShows).
	loadingSince time.Time
}

func meetingDetailSection(c *ui.Context, state meetingDetailViewState, actions meetingsActions, local *meetingsLocalUI) {
	t := c.Theme()
	if local.detail == nil {
		local.detail = &meetingDetailLocalUI{}
	}
	if state.Loading || state.Error != "" {
		ui.Column(c).Grow(1).MinWidth(0).Padding(px(t, sp4), px(t, sp6)).Gap(px(t, sp3)).Children(func() {
			panelHeader(c, i18n.T("Meeting details"), actions.closeMeetingDetail)
			if state.Loading {
				if loadingShows(c, &local.detail.loadingSince) {
					caption(c, i18n.T("Loading…"))
				}
			} else {
				notice(c, "mt-detail-err", noticeError, i18n.T("Could not read the meeting."), state.Error, nil, nil)
			}
		})
		return
	}
	rows := 1 + max(len(state.Segments), 1)
	ui.List(c.Key("mt-detail"), &local.detail.segList, rows).Grow(1).MinWidth(0).Label(i18n.T("Meeting details")).
		ItemKey(func(index int) any {
			if index == 0 || len(state.Segments) == 0 {
				return index
			}
			return state.Segments[index-1].Key
		}).Rows(func(row ui.ListRow) {
		top, bottom := float32(0), float32(0)
		if row.Index == 0 {
			top = px(t, sp4)
		}
		if row.Index == rows-1 {
			bottom = px(t, sp6)
		}
		ui.Column(c).Padding(top, px(t, sp6), bottom).MaxWidth(px(t, meetingDetailMaxW)).Children(func() {
			switch {
			case row.Index == 0:
				ui.Column(c).Gap(px(t, sp5)).Padding(0, 0, px(t, sp1)).Children(func() {
					meetingDetailTop(c, state, actions, local)
				})
			case len(state.Segments) == 0:
				ui.Text(c, i18n.T("No transcript yet.")).TextColor(tok(t).Text3).Padding(px(t, sp2))
			default:
				meetingSegment(c, state.Segments[row.Index-1], state.Playback.Highlight, state.ID, actions)
			}
		})
	})
}

func meetingDetailTop(c *ui.Context, state meetingDetailViewState, actions meetingsActions, local *meetingsLocalUI) {
	t := c.Theme()
	colors := tok(t)
	meetingDetailHeader(c, state, actions, local)
	section := func(key, heading string, actionsView, body func()) {
		ui.Column(c.Key(key)).Gap(px(t, sp1)).Children(func() {
			ui.Row(c).Gap(px(t, sp1)).AlignItems(ui.Center).MinHeight(px(t, 28)).Children(func() {
				sectionTitleText(c, heading).Grow(1).MinWidth(0)
				if actionsView != nil {
					actionsView()
				}
			})
			if body != nil {
				body()
			}
		})
	}
	section("mt-sum", i18n.T("Summary"), func() {
		if state.CanUseLLM && iconButton(c.Key("mt-regen"), icons.Refresh, i18n.T("Regenerate summary")).Clicked() {
			actions.regenerateMeetingSummary(state.ID)
		}
	}, func() {
		fillCard(c).Padding(px(t, sp3)).Gap(px(t, sp2)).Children(func() {
			switch {
			case state.Summary.Present:
				if state.Summary.Title != "" {
					ui.Text(c, state.Summary.Title).FontWeight(600).Selectable()
				}
				if state.Summary.Body != "" {
					ui.Text(c, state.Summary.Body).TextColor(colors.Text2).Selectable()
				}
				for index, todo := range state.Summary.Todos {
					ui.Row(c.Key(fmt.Sprintf("mt-todo-%d", index))).Gap(px(t, sp2)).Children(func() {
						ui.Text(c, "•").TextColor(colors.Text3)
						ui.Text(c, todo).Grow(1).MinWidth(0).Selectable()
					})
				}
			case state.CanUseLLM:
				ui.Text(c, i18n.T("No summary yet.")).TextColor(colors.Text3)
			default:
				ui.Text(c, i18n.T("Set up a language model to get a summary and ask questions.")).TextColor(colors.Text3)
				ui.Row(c).Children(func() {
					if ui.Button(c.Key("mt-open-llm"), i18n.T("Open Language models")).Clicked() {
						actions.openMeetingLanguageModels()
					}
				})
			}
		})
	})
	if state.CanUseLLM {
		section("mt-ask", i18n.T("Ask about this meeting"), nil, func() {
			for index, message := range state.Chat {
				if message.User {
					ui.Text(c.Key(fmt.Sprintf("mt-chat-%d", index)), message.Text).FontWeight(600)
				} else {
					ui.Text(c.Key(fmt.Sprintf("mt-chat-%d", index)), message.Text).TextColor(colors.Text2).Selectable()
				}
			}
			if state.Answering {
				caption(c, i18n.T("Answering…"))
			}
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ui.TextInput(c.Key("mt-followup"), &local.detail.question).Placeholder(i18n.T("Follow-up question")).Grow(1).MinWidth(0)
				question := strings.TrimSpace(local.detail.question)
				if ui.Button(c.Key("mt-ask"), i18n.T("Send question")).Disabled(state.Answering || question == "").Clicked() {
					actions.askMeetingFollowUp(state.ID, question)
				}
			})
		})
	}
	section("mt-tr", i18n.T("Transcript"), func() {
		meetingPlaybackBar(c, state.Playback, state.ID, actions)
		if !state.HasTranscript {
			return
		}
		if iconButton(c.Key("mt-copy"), icons.Copy, i18n.T("Copy transcript")).Clicked() {
			actions.copyMeetingTranscript(state.ID)
		}
		iconButton(c.Key("mt-export"), icons.Export, i18n.T("Export transcript")).Menu(func(menu *ui.Menu) {
			if menu.Item(i18n.T("Export .txt…")).Chosen() {
				actions.saveMeetingExport(state.ID, meetingExportText)
			}
			if menu.Item(i18n.T("Export Markdown…")).Chosen() {
				actions.saveMeetingExport(state.ID, meetingExportMarkdown)
			}
		})
	}, func() {
		if state.SpeakerHelp {
			ui.Text(c, i18n.T("Download MOSS-Transcribe-Diarize on the Speech models page to label the other side as Speaker 1–4.")).TextColor(tok(t).Text3).Selectable()
			ui.Row(c).Children(func() {
				if ui.Button(c.Key("mt-open-speakers"), i18n.T("Open Speech models")).Clicked() {
					actions.openMeetingSpeechModels()
				}
			})
		}
	})
}

func meetingDetailHeader(c *ui.Context, state meetingDetailViewState, actions meetingsActions, local *meetingsLocalUI) {
	t := c.Theme()
	ui.Column(c).Gap(px(t, sp1)).Children(func() {
		ui.Row(c).Gap(px(t, sp1)).AlignItems(ui.Center).Children(func() {
			sectionHeading(c, state.Title).Grow(1).MinWidth(0).SingleLine().Ellipsis("…")
			if state.ShowMore {
				moreMenu(c.Key("mt-more"), "", func(menu *ui.Menu) {
					if state.CanReprocess && menu.Item(i18n.T("Reprocess transcript")).Chosen() {
						actions.meetingReprocess(state.ID)
					}
					if state.CanCorrect && menu.Item(i18n.T("Correct with the language model")).Chosen() {
						actions.meetingCorrect(state.ID)
					}
					if state.CanRevealAudio && menu.Item(i18n.T("Show in Finder")).Chosen() {
						actions.showMeetingInFinder(state.ID)
					}
				})
			}
			if iconButton(c.Key("mt-del"), icons.TrashBinMinimalistic, i18n.T("Delete meeting…")).Disabled(state.DeleteDisabled).Clicked() {
				local.delID = state.ID
			}
			if iconButton(c.Key("panel-close"), icons.Close, i18n.T("Close")).Clicked() {
				actions.closeMeetingDetail()
			}
		})
		caption(c, state.Meta)
		if state.Correcting {
			caption(c, i18n.T("Correcting the transcript with the language model…"))
		}
		if state.CanContinue {
			ui.Row(c).Margin(px(t, sp2), 0, 0, 0).Children(func() {
				if ui.Button(c.Key("mt-continue"), i18n.T("Continue processing")).Clicked() {
					actions.meetingReprocess(state.ID)
				}
			})
		}
	})
}
