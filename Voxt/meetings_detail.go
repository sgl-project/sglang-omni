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
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
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
	// correcting: the language model is correcting the transcript.
	correcting bool
	err        string
	playback   meetingDetailPlayback
	segList    ui.ListState // the details' one list (meetingDetailSection)
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

// meetingDetailMaxW caps the details' lines (DIPs at 100 %), so that a
// wide window does not make them too long to read.
const meetingDetailMaxW = 960

// meetingDetailSection is an open meeting's details, one list that
// scrolls them all (#4019): its top (the title, summary, questions and
// the transcript's heading), then a row for each transcript line, laid
// out only in view however long the meeting.
func (a *app) meetingDetailSection(c *ui.Context, close func()) {
	t := c.Theme()
	d := a.mv.detail
	if d == nil {
		return
	}
	if d.loading || d.err != "" {
		ui.Column(c).Grow(1).MinWidth(0).Padding(px(t, sp4), px(t, sp6)).Gap(px(t, sp3)).Children(func() {
			panelHeader(c, i18n.T("Meeting details"), close)
			if d.loading {
				caption(c, i18n.T("Loading…"))
			} else {
				notice(c, "mt-detail-err", noticeError, i18n.T("Could not read the meeting."), d.err, nil, nil)
			}
		})
		return
	}
	segs := d.segments
	names := speakerNameMap(d.meeting.Speakers)
	rows := 1 + max(len(segs), 1) // the top, then the lines or the empty one
	ui.List(c.Key("mt-detail"), &d.segList, rows).Grow(1).MinWidth(0).Label(i18n.T("Meeting details")).
		ItemKey(func(i int) any {
			if i == 0 || len(segs) == 0 {
				return i
			}
			return "seg/" + itoa(segs[i-1].Idx)
		}).Rows(func(r ui.ListRow) {
		top, bottom := float32(0), float32(0)
		if r.Index == 0 {
			top = px(t, sp4)
		}
		if r.Index == rows-1 {
			bottom = px(t, sp6)
		}
		ui.Column(c).Padding(top, px(t, sp6), bottom).MaxWidth(px(t, meetingDetailMaxW)).Children(func() {
			switch {
			case r.Index == 0:
				ui.Column(c).Gap(px(t, sp5)).Padding(0, 0, px(t, sp1)).Children(func() { a.meetingDetailTop(c, d, close) })
			case len(segs) == 0:
				ui.Text(c, i18n.T("No transcript yet.")).TextColor(tok(t).Text3).Padding(px(t, sp2))
			default:
				a.meetingSegment(c, d, names, segs[r.Index-1])
			}
		})
	})
}

// meetingDetailTop is the details above the transcript's lines.
func (a *app) meetingDetailTop(c *ui.Context, d *meetingDetail, close func()) {
	t := c.Theme()
	k := tok(t)
	mt := d.meeting
	a.meetingDetailHeader(c, mt, close)
	sum, hasSum := meeting.ParseStoredSummary(mt.Summary)
	names := speakerNameMap(mt.Speakers)
	canLLM := len(a.installed(local.LLM)) > 0
	exportSegs := storeSegmentsToExport(d.segments)
	title := meetingTitle(mt)
	file := strings.NewReplacer("/", "-", ":", "-").Replace(title)

	// Each section: a title row with its actions, then its body.
	section := func(key, heading string, actions, body func()) {
		ui.Column(c.Key(key)).Gap(px(t, sp1)).Children(func() {
			ui.Row(c).Gap(px(t, sp1)).AlignItems(ui.Center).MinHeight(px(t, 28)).Children(func() {
				sectionTitleText(c, heading).Grow(1).MinWidth(0)
				if actions != nil {
					actions()
				}
			})
			if body != nil {
				body()
			}
		})
	}
	section("mt-sum", i18n.T("Summary"), func() {
		if canLLM && iconButton(c.Key("mt-regen"), icons.Refresh, i18n.T("Regenerate summary")).Clicked() {
			a.regenerateMeetingSummary(d.id)
		}
	}, func() {
		fillCard(c).Padding(px(t, sp3)).Gap(px(t, sp2)).Children(func() {
			switch {
			case hasSum:
				if sum.Title != "" {
					ui.Text(c, sum.Title).FontWeight(600).Selectable()
				}
				if sum.Body != "" {
					ui.Text(c, sum.Body).TextColor(k.Text2).Selectable()
				}
				for i, todo := range sum.Todos {
					ui.Row(c.Key(fmt.Sprintf("mt-todo-%d", i))).Gap(px(t, sp2)).Children(func() {
						ui.Text(c, "•").TextColor(k.Text3)
						ui.Text(c, todo).Grow(1).MinWidth(0).Selectable()
					})
				}
			case canLLM:
				ui.Text(c, i18n.T("No summary yet.")).TextColor(k.Text3)
			default:
				ui.Text(c, i18n.T("Set up a language model to get a summary and ask questions.")).TextColor(k.Text3)
				ui.Row(c).Children(func() {
					if ui.Button(c.Key("mt-open-llm"), i18n.T("Open Language models")).Clicked() {
						a.navigate(remotePath)
					}
				})
			}
		})
	})
	if canLLM {
		section("mt-ask", i18n.T("Ask about this meeting"), nil, func() {
			for i, msg := range d.chat {
				if msg.Role == "user" {
					ui.Text(c.Key(fmt.Sprintf("mt-chat-%d", i)), msg.Content).FontWeight(600)
				} else {
					ui.Text(c.Key(fmt.Sprintf("mt-chat-%d", i)), msg.Content).TextColor(k.Text2).Selectable()
				}
			}
			if d.busy {
				caption(c, i18n.T("Answering…"))
			}
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ui.TextInput(c.Key("mt-followup"), &d.question).Placeholder(i18n.T("Follow-up question")).Grow(1).MinWidth(0)
				q := strings.TrimSpace(d.question)
				if ui.Button(c.Key("mt-ask"), i18n.T("Send question")).Disabled(d.busy || q == "").Clicked() {
					a.askMeetingFollowUp(d.id, q)
				}
			})
		})
	}
	section("mt-tr", i18n.T("Transcript"), func() {
		a.meetingPlaybackBar(c, d)
		if len(d.segments) == 0 {
			return
		}
		if iconButton(c.Key("mt-copy"), icons.Copy, i18n.T("Copy transcript")).Clicked() {
			_ = mygo.Clipboard.Write(transfer.TextData(meeting.ExportTxt(exportSegs, names)))
			a.toast(i18n.T("Copied"))
		}
		iconButton(c.Key("mt-export"), icons.Export, i18n.T("Export transcript")).Menu(func(m *ui.Menu) {
			if m.Item(i18n.T("Export .txt…")).Chosen() {
				a.saveMeetingExport(file+".txt", meeting.ExportTxt(exportSegs, names))
			}
			if m.Item(i18n.T("Export Markdown…")).Chosen() {
				a.saveMeetingExport(file+".md", meeting.ExportMarkdown(title, exportSegs, names))
			}
		})
	}, nil)
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
