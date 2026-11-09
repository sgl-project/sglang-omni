package main

// Meetings as the UI sees them: internal/meeting.Manager wired to the
// helper's dual capture, the store, and the meetings folder under the
// data directory; snapshots drive the meeting overlay and block dictation
// shortcuts while a meeting is active.

import (
	"context"
	"path/filepath"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

const meetingLogSource = "meeting"

type meetingState struct {
	meetMgr        *meeting.Manager
	meetSnap       meeting.Snapshot
	meetingUI      *meetingSettingsUI // settings_meeting.go
	meetingAbandon bool               // abandon while recording
}

func (a *app) openMeetings() {
	if a.host == nil || a.host.sup == nil {
		return
	}
	dir := filepath.Join(a.dataDir(), "meetings")
	var modelHold local.Holder
	if h, ok := a.local.(local.Holder); ok {
		modelHold = h
	}
	hasSystem := a.host.sup != nil && a.host.sup.Status().Has(proto.MethodCaptureSystem)
	a.meetMgr = meeting.NewManager(meeting.Config{
		Mic:       session.HelperAudio{H: a.host.sup},
		System:    session.HelperAudio{H: a.host.sup, Method: proto.MethodCaptureSystem},
		Store:     a.lib.store.Meetings(),
		Dir:       dir,
		Settings:  func() settings.Meeting { return a.prefs.Features.Meeting },
		Privacy:   func() settings.Privacy { return a.prefs.Privacy },
		Now:       time.Now,
		FreeSpace: func(path string) (int64, error) { return diskFree(path), nil },
		ModelHold: modelHold,
		ASR:       func(ctx context.Context, src meeting.Source) (meeting.TrackASR, error) { return a.meetingTrackASR(ctx, src) },
		VAD:       a.meetingVAD(),
		LLM: func(ctx context.Context, id string) (meeting.SummaryChat, error) {
			chat, _, err := a.local.LLM(ctx, id)
			return chat, err
		},
		MainLanguage: func() string {
			return uitext.LangLabel(a.prefs.General.MainLanguage)
		},
		SystemCaptureOK: func() bool { return hasSystem },
	})
	a.meetMgr.Subscribe(func(s meeting.Snapshot) {
		a.post(func() { a.meetingChanged(s) })
	})
	a.meetSnap = a.meetMgr.Current()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, _ = a.meetMgr.Recover(ctx)
	})
}

func (a *app) meetingActive() bool {
	return a.meetMgr != nil && a.meetSnap.State.Active()
}

func (a *app) meetingChanged(s meeting.Snapshot) {
	a.meetSnap = s
	a.syncMeetingOverlay()
	a.syncHotkeys()
	if a.tray != nil {
		a.buildTrayMenu()
	}
}

func meetingModeFrom(s settings.Meeting) meeting.Mode {
	switch s.Mode {
	case "subtitles":
		return meeting.ModeSubtitles
	case "recording":
		return meeting.ModeRecording
	default:
		return meeting.ModeMeeting
	}
}

func (a *app) startMeeting() {
	if !a.meetingEnabled() || a.meetMgr == nil {
		return
	}
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		_, err := a.meetMgr.Start(ctx, meeting.StartRequest{Mode: meetingModeFrom(a.prefs.Features.Meeting)})
		if err != nil {
			a.post(func() {
				a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: "start: " + err.Error()})
			})
		}
	})
}

func (a *app) meetingPauseResume() {
	if a.meetMgr == nil || a.meetSnap.ID == "" {
		return
	}
	id := a.meetSnap.ID
	a.async(func() {
		ctx := context.Background()
		var err error
		switch a.meetSnap.State {
		case meeting.StateRecording:
			err = a.meetMgr.Pause(ctx, id)
		case meeting.StatePaused:
			err = a.meetMgr.Resume(ctx, id)
		}
		if err != nil {
			a.post(func() {
				a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: err.Error()})
			})
		}
	})
}

func (a *app) meetingFinish() {
	if a.meetMgr == nil || !a.meetingActive() {
		return
	}
	id := a.meetSnap.ID
	a.async(func() {
		ctx := context.Background()
		if err := a.meetMgr.Finish(ctx, id); err != nil {
			a.post(func() {
				a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: "finish: " + err.Error()})
			})
		}
	})
}

func (a *app) meetingSave() {
	if a.meetMgr == nil || a.meetSnap.State != meeting.StateAsking {
		return
	}
	id := a.meetSnap.ID
	a.async(func() {
		if err := a.meetMgr.Save(context.Background(), id); err != nil {
			a.post(func() {
				a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: "save: " + err.Error()})
			})
			return
		}
		a.post(a.loadMeetings)
	})
}

func (a *app) meetingReprocess(id string) {
	if a.meetMgr == nil || id == "" {
		return
	}
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Hour)
		defer cancel()
		if err := a.meetMgr.Reprocess(ctx, id); err != nil {
			a.post(func() {
				a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: "reprocess: " + err.Error()})
			})
			return
		}
		a.post(a.loadMeetings)
	})
}

func (a *app) meetingSetMode(mode meeting.Mode) {
	if a.meetMgr == nil || a.meetSnap.ID == "" {
		return
	}
	id, mode := a.meetSnap.ID, mode
	a.async(func() {
		if err := a.meetMgr.SetMode(context.Background(), id, mode); err != nil {
			a.post(func() {
				a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: err.Error()})
			})
		}
	})
}

func (a *app) meetingSetTranslate(target string) {
	if a.meetMgr == nil || a.meetSnap.ID == "" {
		return
	}
	id := a.meetSnap.ID
	a.async(func() {
		if err := a.meetMgr.SetTranslate(id, target); err != nil {
			a.post(func() {
				a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: err.Error()})
			})
		}
	})
}

func (a *app) meetingDiscard() {
	if a.meetMgr == nil {
		return
	}
	id := a.meetSnap.ID
	a.async(func() {
		if err := a.meetMgr.Discard(context.Background(), id); err != nil {
			a.post(func() {
				a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: "discard: " + err.Error()})
			})
		}
	})
}
