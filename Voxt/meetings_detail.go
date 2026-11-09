package main

// Meeting detail panel: summary, export, and regeneration (C1-4).

import (
	"context"
	"encoding/json"
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
	loading  bool
	err      string
	playback meetingDetailPlayback
}

func (a *app) openMeetingDetail(id string) {
	if id == "" {
		return
	}
	a.stopMeetingPlayback()
	n := &a.mv
	n.detail = &meetingDetail{id: id, loading: true}
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
		})
	})
}

func (a *app) meetingDetailSection(c *ui.Context) {
	t := c.Theme()
	d := a.mv.detail
	if d == nil {
		return
	}
	sectionTitle(c, i18n.T("Meeting detail"))
	if d.loading {
		ui.Text(c, i18n.T("Loading…"))
		return
	}
	if d.err != "" {
		ui.Text(c, d.err)
		return
	}
	sum, hasSum := meeting.ParseStoredSummary(d.meeting.Summary)
	names := speakerNameMap(d.meeting.Speakers)
	if hasSum {
		if sum.Title != "" {
			ui.Text(c, sum.Title).FontWeight(600)
		}
		if sum.Body != "" {
			ui.Text(c, sum.Body).TextColor(tok(t).Text2)
		}
		for _, todo := range sum.Todos {
			ui.Text(c, "• "+todo)
		}
	} else {
		ui.Text(c, i18n.T("No summary yet.")).TextColor(tok(t).Text2)
	}
	canLLM := len(a.installed(local.LLM)) > 0
	if primaryButton(c.Key("mt-regen"), i18n.T("Regenerate summary")).Disabled(!canLLM).Clicked() {
		a.regenerateMeetingSummary(d.id)
	}
	if !canLLM {
		ui.Text(c, i18n.T("Set up a language model on the Language models page to generate summaries.")).
			FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
	}
	exportSegs := storeSegmentsToExport(d.segments)
	if ghostButton(c.Key("mt-export-txt"), i18n.T("Copy transcript (.txt)")).Clicked() {
		_ = mygo.Clipboard.Write(transfer.TextData(meeting.ExportTxt(exportSegs, names)))
		a.toast(i18n.T("Copied"))
	}
	if ghostButton(c.Key("mt-export-md"), i18n.T("Copy transcript (Markdown)")).Clicked() {
		title := d.meeting.Title
		if title == "" {
			title = d.meeting.CreatedAt.Format("2006-01-02 15:04")
		}
		_ = mygo.Clipboard.Write(transfer.TextData(meeting.ExportMarkdown(title, exportSegs, names)))
		a.toast(i18n.T("Copied"))
	}
	a.meetingPlaybackBar(c, d)
	a.meetingTranscriptList(c, d)
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
