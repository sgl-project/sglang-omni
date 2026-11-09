package main

// Meeting detail panel: summary, export, and regeneration (C1-4).

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type meetingDetail struct {
	id       string
	meeting  store.Meeting
	segments []store.MeetingSegment
	chat     []meeting.ChatMessage
	question string
	loading  bool
	busy     bool
	err      string
	playback meetingDetailPlayback
	segList  ui.ListState
}

func (a *app) openMeetingDetail(id string) {
	if id == "" {
		return
	}
	a.stopMeetingPlayback()
	n := &a.mv
	n.detail = &meetingDetail{id: id, loading: true}
	n.detail.playback.highlight = -1
	repo := a.lib.store.Meetings()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		mt, segs, err := repo.Meeting(ctx, id)
		a.post(func() {
			if n.detail == nil || n.detail.id != id {
				return
			}
			n.detail.loading = false
			if err != nil {
				n.detail.err = err.Error()
				return
			}
			n.detail.meeting = mt
			n.detail.segments = segs
			n.detail.chat, _ = meeting.ParseChatMessages(mt.Chat)
		})
	})
}

// meetingDetailSection is the details panel's content; close hides it.
func (a *app) meetingDetailSection(c *ui.Context, close func()) {
	t := c.Theme()
	d := a.mv.detail
	if d == nil {
		return
	}
	ui.Column(c).Padding(px(t, sp4)).Gap(px(t, sp3)).Children(func() { a.meetingDetailBody(c, d, close) })
}

func (a *app) meetingDetailBody(c *ui.Context, d *meetingDetail, close func()) {
	t := c.Theme()
	if d.loading || d.err != "" {
		panelHeader(c, i18n.T("Meeting detail"), close)
		if d.loading {
			caption(c, i18n.T("Loading…"))
		} else {
			notice(c, "mt-detail-err", noticeError, i18n.T("Could not read the meeting."), d.err, nil, nil)
		}
		return
	}
	panelHeader(c, meetingTitle(d.meeting), close)
	caption(c, meetingMeta(d.meeting))
	a.meetingActions(c, d.meeting)
	sum, hasSum := meeting.ParseStoredSummary(d.meeting.Summary)
	names := speakerNameMap(d.meeting.Speakers)
	canLLM := len(a.installed(local.LLM)) > 0
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		sectionHeading(c, i18n.T("Summary")).Grow(1)
		if ghostButton(c.Key("mt-regen"), i18n.T("Regenerate summary")).Disabled(!canLLM).Clicked() {
			a.regenerateMeetingSummary(d.id)
		}
	})
	switch {
	case hasSum:
		if sum.Title != "" {
			ui.Text(c, sum.Title).FontWeight(600)
		}
		if sum.Body != "" {
			ui.Text(c, sum.Body).TextColor(tok(t).Text2)
		}
		for _, todo := range sum.Todos {
			ui.Text(c, "• "+todo)
		}
	case canLLM:
		caption(c, i18n.T("No summary yet."))
	default:
		caption(c, i18n.T("Set up a language model on the Language models page to generate summaries."))
	}
	exportSegs := storeSegmentsToExport(d.segments)
	title := meetingTitle(d.meeting)
	file := strings.NewReplacer("/", "-", ":", "-").Replace(title)
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		sectionHeading(c, i18n.T("Transcript")).Grow(1)
		a.meetingPlaybackBar(c, d)
	})
	if len(d.segments) > 0 {
		ui.Row(c).Gap(px(t, sp2)).Wrap().Children(func() {
			if ghostButton(c.Key("mt-copy"), i18n.T("Copy transcript")).Clicked() {
				_ = mygo.Clipboard.Write(transfer.TextData(meeting.ExportTxt(exportSegs, names)))
				a.toast(i18n.T("Copied"))
			}
			if ghostButton(c.Key("mt-export-txt"), i18n.T("Export .txt…")).Clicked() {
				a.saveMeetingExport(file+".txt", meeting.ExportTxt(exportSegs, names))
			}
			if ghostButton(c.Key("mt-export-md"), i18n.T("Export Markdown…")).Clicked() {
				a.saveMeetingExport(file+".md", meeting.ExportMarkdown(title, exportSegs, names))
			}
		})
		a.meetingTranscriptList(c, d)
	} else {
		caption(c, i18n.T("No transcript yet."))
	}
	sectionHeading(c, i18n.T("Ask about this meeting"))
	for i, msg := range d.chat {
		line := msg.Content
		if msg.Role == "user" {
			line = i18n.T("You") + ": " + line
		}
		ui.Text(c.Key(fmt.Sprintf("mt-chat-%d", i)), line).TextColor(tok(t).Text2)
	}
	if d.busy {
		caption(c, i18n.T("Answering…"))
	}
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		ui.TextInput(c.Key("mt-followup"), &d.question).Placeholder(i18n.T("Follow-up question")).Grow(1).MinWidth(0)
		q := strings.TrimSpace(d.question)
		if primaryButton(c.Key("mt-ask"), i18n.T("Send question")).Disabled(!canLLM || d.busy || q == "").Clicked() {
			a.askMeetingFollowUp(d.id, q)
		}
	})
	if !canLLM {
		caption(c, i18n.T("Set up a language model on the Language models page to ask follow-up questions."))
	}
}

func (a *app) askMeetingFollowUp(id, question string) {
	if a.meetMgr == nil || id == "" || question == "" {
		return
	}
	n := &a.mv
	if n.detail == nil || n.detail.id != id {
		return
	}
	n.detail.busy = true
	a.async(func() {
		ctx := context.Background()
		if err := a.meetMgr.AnswerMeetingFollowUp(ctx, id, question); err != nil {
			a.post(func() {
				if n.detail != nil && n.detail.id == id {
					n.detail.busy = false
				}
				a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: "follow-up: " + err.Error()})
			})
			return
		}
		a.post(func() {
			if n.detail != nil && n.detail.id == id {
				n.detail.question = ""
				n.detail.busy = false
			}
			a.openMeetingDetail(id)
		})
	})
}

func (a *app) regenerateMeetingSummary(id string) {
	if a.meetMgr == nil {
		return
	}
	a.async(func() {
		ctx := context.Background()
		if err := a.meetMgr.GenerateMeetingSummary(ctx, id); err != nil {
			a.post(func() {
				a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: "summary: " + err.Error()})
			})
			return
		}
		a.post(func() {
			a.openMeetingDetail(id)
			a.loadMeetings()
		})
	})
}

// saveMeetingExport writes text to a file the user picks, name by default.
func (a *app) saveMeetingExport(name, text string) {
	chooseExportFile(a, i18n.T("Export transcript"), name, func(dst string) {
		a.async(func() {
			err := os.WriteFile(dst, []byte(text), 0o644)
			a.post(func() {
				if err != nil {
					a.toast(i18n.T("Could not export: %v", err))
					return
				}
				a.toast(i18n.T("Exported to %s", filepath.Base(dst)))
			})
		})
	})
}

func speakerNameMap(raw string) map[string]string {
	out := map[string]string{}
	_ = json.Unmarshal([]byte(raw), &out)
	return out
}

func storeSegmentsToExport(rows []store.MeetingSegment) []meeting.ExportSegment {
	out := make([]meeting.ExportSegment, 0, len(rows))
	for _, row := range rows {
		out = append(out, meeting.ExportSegment{
			Start:       time.Duration(row.StartMS) * time.Millisecond,
			Speaker:     row.Speaker,
			Text:        row.Text,
			Translation: row.Translation,
		})
	}
	return out
}
