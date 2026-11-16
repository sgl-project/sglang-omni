package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const meetingsPageSize = 20

type meetingListCacheKey struct {
	seq            int
	listLen        int
	firstID        string
	lastID         string
	selectedID     string
	locale         string
	liveID         string
	liveState      string
	elapsedMinutes int64
	year           int
	month          time.Month
	day            int
	read           bool
	loading        bool
}

type meetingsView struct {
	list       []store.Meeting
	read       bool
	loading    bool
	err        string
	seq        int
	lines      map[string]string
	hasMore    bool
	detail     *meetingDetail
	detailView *meetingDetailViewState
	locale     string
	translator *i18n.Translator
	listItems  []meetingItemViewState
	listKey    meetingListCacheKey
}

type meetingsState struct {
	mv       meetingsView
	mvUI     meetingsLocalUI
	mtPlayer meetingPlayer
}

func (a *app) meetingsWant(query string, page int) string {
	return strings.TrimSpace(query) + "\x00" + string(rune('0'+page))
}

func (a *app) loadMeetings() {
	n := &a.mv
	n.seq++
	n.loading = true
	seq := n.seq
	query, page := a.mvUI.query, a.mvUI.page
	want := a.meetingsWant(query, page)
	request := store.MeetingQuery{Search: strings.TrimSpace(query), Limit: meetingsPageSize + 1, Offset: page * meetingsPageSize}
	repo := a.lib.store.Meetings()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		list, err := repo.ListMeetings(ctx, request)
		var lines map[string]string
		if request.Search != "" && err == nil {
			lines = map[string]string{}
			for _, stored := range list {
				if _, segments, err := repo.Meeting(ctx, stored.ID); err == nil {
					lines[stored.ID] = meetingLine(segments, request.Search)
				}
			}
		}
		a.post(func() {
			n := &a.mv
			if seq != n.seq || want != a.meetingsWant(a.mvUI.query, a.mvUI.page) {
				return
			}
			n.loading, n.read = false, true
			if err != nil {
				n.err = err.Error()
				return
			}
			n.hasMore = len(list) > meetingsPageSize
			if n.hasMore {
				list = list[:meetingsPageSize]
			}
			n.list, n.lines, n.err = list, lines, ""
		})
	})
}

func (a *app) searchMeetings(query string) {
	a.mvUI.query, a.mvUI.page = query, 0
	a.loadMeetings()
}

func (a *app) showMeetingsPage(page int) {
	if page < 0 {
		return
	}
	a.mvUI.page = page
	a.loadMeetings()
}

// meetingLine is the first transcript or translation line holding query.
func meetingLine(segments []store.MeetingSegment, query string) string {
	for _, segment := range segments {
		for _, text := range []string{segment.Text, segment.Translation} {
			for line := range strings.SplitSeq(text, "\n") {
				if line = strings.TrimSpace(line); len(findFolded(line, query)) > 0 {
					return line
				}
			}
		}
	}
	return ""
}

func (a *app) prepareMeetingsPage() {
	if !a.mv.read && !a.mv.loading {
		a.loadMeetings()
	}
	a.refreshMeetingAudio()
	if a.mv.detail != nil {
		canUseLLM := len(a.installed(local.LLM)) > 0
		if a.mv.detailView == nil || a.mv.detailView.CanUseLLM != canUseLLM || a.mv.locale != i18n.Locale() {
			a.rebuildMeetingDetail()
		}
		if a.mvUI.detail == nil {
			a.mvUI.detail = &meetingDetailLocalUI{}
		}
	}
}

func (a *app) meetingsPageModel(now time.Time) meetingsViewState {
	selectedID := ""
	if a.mv.detail != nil {
		selectedID = a.mv.detail.id
	}
	year, month, day := now.Date()
	key := meetingListCacheKey{
		seq: a.mv.seq, listLen: len(a.mv.list), selectedID: selectedID, locale: i18n.Locale(),
		liveID: a.meetSnap.ID, liveState: string(a.meetSnap.State), year: year, month: month, day: day,
		read: a.mv.read, loading: a.mv.loading,
	}
	if len(a.mv.list) > 0 {
		key.firstID, key.lastID = a.mv.list[0].ID, a.mv.list[len(a.mv.list)-1].ID
	}
	if a.meetSnap.State.Active() {
		key.elapsedMinutes = int64((a.meetSnap.Elapsed + 30*time.Second) / time.Minute)
	}
	if key != a.mv.listKey {
		built := buildMeetingsViewState(i18n.NewTranslator(i18n.Locale()), meetingsViewStateInput{
			List: a.mv.list, Lines: a.mv.lines, Read: a.mv.read, Loading: a.mv.loading,
			Error: a.mv.err, Query: a.mvUI.query, Page: a.mvUI.page, HasMore: a.mv.hasMore,
			Active: a.meetingActive(), SelectedID: selectedID, Live: a.meetSnap, Now: now,
		})
		a.mv.listItems, a.mv.listKey = built.Items, key
	}
	state := meetingsViewState{
		Read: a.mv.read, Loading: a.mv.loading, Error: a.mv.err, Query: a.mvUI.query,
		Page: a.mvUI.page, HasMore: a.mv.hasMore, Active: a.meetingActive(), SettingsAvailable: true,
		MeetingCount: len(a.mv.list), Items: a.mv.listItems, Detail: a.mv.detailView,
	}
	return state
}

func (a *app) closeMeetingDetail() {
	a.stopMeetingPlayback()
	a.stopMeetingAudioWatch(a.mv.detail)
	a.mv.detail, a.mv.detailView = nil, nil
	a.mvUI.detail = nil
}

func (a *app) deleteMeeting(id string) {
	audio := ""
	for _, stored := range a.mv.list {
		if stored.ID == id {
			audio = stored.Audio
		}
	}
	if a.mv.detail != nil && a.mv.detail.id == id {
		a.closeMeetingDetail()
	}
	a.async(func() {
		_ = a.lib.store.Meetings().DeleteMeeting(context.Background(), id)
		if audio != "" {
			_ = os.RemoveAll(filepath.Join(a.dataDir(), "meetings", audio))
		}
		a.post(a.loadMeetings)
	})
}

func (a *app) refreshMeetingAudio() {
	detail := a.mv.detail
	if detail == nil || detail.meeting.Audio == "" || detail.audioProbing {
		return
	}
	detail.audioProbe++
	probe := detail.audioProbe
	detail.audioProbing = true
	id, audio := detail.id, detail.meeting.Audio
	done := make(chan struct{})
	detail.audioDone = done
	previous := detail.audioReady
	a.async(func() {
		for {
			ready := meeting.AudioFolderReady(filepath.Join(a.dataDir(), "meetings"), audio)
			if ready != previous {
				a.post(func() {
					current := a.mv.detail
					if current == nil || current.id != id || current.meeting.Audio != audio || current.audioProbe != probe {
						return
					}
					current.audioReady = ready
					a.refreshMeetingPlaybackState()
				})
				previous = ready
			}
			if a.win == nil {
				return
			}
			select {
			case <-done:
				return
			case <-time.After(500 * time.Millisecond):
			}
		}
	})
}

func (a *app) stopMeetingAudioWatch(detail *meetingDetail) {
	if detail == nil {
		return
	}
	detail.audioProbing = false
	if detail.audioDone != nil {
		close(detail.audioDone)
		detail.audioDone = nil
	}
}
