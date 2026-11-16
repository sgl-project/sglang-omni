package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type meetingDetail struct {
	id           string
	meeting      store.Meeting
	segments     []store.MeetingSegment
	chat         []meeting.ChatMessage
	loading      bool
	busy         bool
	correcting   bool
	err          string
	playback     meetingDetailPlayback
	audioReady   bool
	audioProbing bool
	audioProbe   int
}

func (a *app) openMeetingDetail(id string) {
	if id == "" {
		return
	}
	a.stopMeetingPlayback()
	a.mv.detail = &meetingDetail{id: id, loading: true, audioProbe: 1}
	a.mv.detail.playback.highlight = -1
	a.mv.detailView = nil
	a.mvUI.detail = &meetingDetailLocalUI{}
	a.loadMeetingDetail(id)
}

// loadMeetingDetail preserves the open detail's scroll, question and playback.
func (a *app) loadMeetingDetail(id string) {
	repo := a.lib.store.Meetings()
	version := 0
	if detail := a.mv.detail; detail != nil && detail.id == id {
		detail.audioProbe++
		detail.audioProbing = false
		version = detail.audioProbe
	}
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		stored, segments, err := repo.Meeting(ctx, id)
		audioReady := false
		if err == nil {
			audioReady = meeting.AudioFolderReady(filepath.Join(a.dataDir(), "meetings"), stored.Audio)
		}
		a.post(func() {
			detail := a.mv.detail
			if detail == nil || detail.id != id || detail.audioProbe != version {
				return
			}
			detail.loading = false
			if err != nil {
				detail.err = err.Error()
				a.rebuildMeetingDetail()
				return
			}
			detail.meeting = stored
			detail.segments = segments
			detail.chat, _ = meeting.ParseChatMessages(stored.Chat)
			detail.audioReady = audioReady
			a.rebuildMeetingDetail()
		})
	})
}

func (a *app) followMeeting(snapshot, previous meeting.Snapshot) {
	detail := a.mv.detail
	if detail == nil {
		return
	}
	if detail.id == previous.ID && snapshot.ID != previous.ID {
		a.loadMeetingDetail(detail.id)
		return
	}
	if detail.id != snapshot.ID || detail.loading {
		return
	}
	if snapshot.State != previous.State {
		a.loadMeetingDetail(snapshot.ID)
		return
	}
	at := make(map[int]int, len(detail.segments))
	for index, row := range detail.segments {
		at[row.Idx] = index
	}
	changed := false
	for _, segment := range snapshot.Segments {
		if index, ok := at[segment.StoreIdx]; ok {
			if detail.segments[index].Translation != segment.Translation {
				detail.segments[index].Translation = segment.Translation
				changed = true
			}
		} else {
			detail.segments = append(detail.segments, segment.Row())
			changed = true
		}
	}
	if changed {
		a.rebuildMeetingDetail()
	}
}

func (a *app) meetingUpdated(id string) {
	if a.mv.read {
		a.loadMeetings()
	}
	if detail := a.mv.detail; detail != nil && detail.id == id {
		a.loadMeetingDetail(id)
	}
}

func (a *app) rebuildMeetingDetail() {
	detail := a.mv.detail
	if detail == nil {
		a.mv.detailView = nil
		return
	}
	a.mv.detailView = buildMeetingDetailViewState(meetingDetailStateInput{
		Meeting: detail.meeting, Segments: detail.segments, Chat: detail.chat,
		Loading: detail.loading, Answering: detail.busy, Correcting: detail.correcting,
		Error: detail.err, CanUseLLM: len(a.installed(local.LLM)) > 0,
		MeetingActive: a.meetingActive(), AudioReady: detail.audioReady,
		Playback: meetingPlaybackStateInput{
			Position: detail.playback.pos, Playing: detail.playback.playing, Highlight: detail.playback.highlight,
		},
	})
	a.mv.locale = i18n.Locale()
}

func (a *app) refreshMeetingPlaybackState() {
	detail, view := a.mv.detail, a.mv.detailView
	if detail == nil || view == nil {
		return
	}
	view.Playback.Available = detail.audioReady
	view.Playback.Playing = detail.playback.playing
	view.Playback.Highlight = detail.playback.highlight
	view.Playback.PositionText = ""
	if detail.playback.pos > 0 || detail.playback.playing {
		view.Playback.PositionText = meeting.FormatPlaybackTime(detail.playback.pos)
	}
}

func (a *app) askMeetingFollowUp(id, question string) {
	if a.meetMgr == nil || id == "" || question == "" || a.mv.detail == nil || a.mv.detail.id != id {
		return
	}
	a.mv.detail.busy = true
	a.rebuildMeetingDetail()
	a.async(func() {
		if err := a.meetMgr.AnswerMeetingFollowUp(context.Background(), id, question); err != nil {
			a.post(func() {
				if a.mv.detail != nil && a.mv.detail.id == id {
					a.mv.detail.busy = false
					a.rebuildMeetingDetail()
				}
				a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: "follow-up: " + err.Error()})
			})
			return
		}
		a.post(func() {
			if a.mv.detail != nil && a.mv.detail.id == id {
				a.mv.detail.busy = false
				if a.mvUI.detail != nil {
					a.mvUI.detail.question = ""
				}
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
		if err := a.meetMgr.GenerateMeetingSummary(context.Background(), id); err != nil {
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

func (a *app) openMeetingLanguageModels() {
	a.navigate(remotePath)
}

func (a *app) showMeetingInFinder(id string) {
	if detail := a.mv.detail; detail != nil && detail.id == id && detail.meeting.Audio != "" {
		revealInFinder(filepath.Join(a.dataDir(), "meetings", detail.meeting.Audio))
	}
}

func (a *app) copyMeetingTranscript(id string) {
	detail := a.mv.detail
	if detail == nil || detail.id != id {
		return
	}
	segments := storeSegmentsToExport(detail.segments)
	_ = mygo.Clipboard.Write(transfer.TextData(meeting.ExportTxt(segments, meetingViewSpeakerNameMap(detail.meeting.Speakers))))
	a.toast(i18n.T("Copied"))
}

func (a *app) saveMeetingExport(id string, format meetingExportFormat) {
	detail := a.mv.detail
	if detail == nil || detail.id != id {
		return
	}
	segments := storeSegmentsToExport(detail.segments)
	names := meetingViewSpeakerNameMap(detail.meeting.Speakers)
	title := meetingTitle(detail.meeting)
	filename := strings.NewReplacer("/", "-", ":", "-").Replace(title)
	var name, text string
	switch format {
	case meetingExportText:
		name, text = filename+".txt", meeting.ExportTxt(segments, names)
	case meetingExportMarkdown:
		name, text = filename+".md", meeting.ExportMarkdown(title, segments, names)
	default:
		return
	}
	a.saveMeetingExportFile(name, text)
}

func (a *app) saveMeetingExportFile(name, text string) {
	chooseExportFile(a, i18n.T("Export transcript"), name, func(destination string) {
		a.async(func() {
			err := os.WriteFile(destination, []byte(text), 0o644)
			a.post(func() {
				if err != nil {
					a.toast(i18n.T("Could not export: %v", err))
					return
				}
				a.toast(i18n.T("Exported to %s", filepath.Base(destination)))
			})
		})
	})
}

func storeSegmentsToExport(rows []store.MeetingSegment) []meeting.ExportSegment {
	segments := make([]meeting.ExportSegment, 0, len(rows))
	for _, row := range rows {
		segments = append(segments, meeting.ExportSegment{
			Start: time.Duration(row.StartMS) * time.Millisecond, Speaker: row.Speaker,
			Text: row.Text, Translation: row.Translation,
		})
	}
	return segments
}
