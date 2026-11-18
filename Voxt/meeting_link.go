package main

// Meetings as the UI sees them: internal/meeting.Manager wired to the
// helper's dual capture, the store, and the meetings folder under the
// data directory; snapshots drive the meeting overlay and block dictation
// shortcuts while a meeting is active.

import (
	"context"
	"errors"
	"path/filepath"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
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
		ASR: func(ctx context.Context, src meeting.Source) (meeting.TrackASR, error) {
			return a.meetingTrackASR(ctx, src)
		},
		VAD: a.meetingVAD(),
		LLM: func(ctx context.Context, id string) (meeting.SummaryChat, error) {
			chat, _, err := a.local.LLM(ctx, id)
			return chat, err
		},
		MainLanguage: func() string {
			return uitext.LangLabel(a.prefs.General.MainLanguage)
		},
		Context:         a.meetingTranscriptContext,
		SystemCaptureOK: a.systemCaptureOK,
		Warn: func(msg string) {
			a.post(func() {
				a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: msg})
			})
		},
		Updated: func(id string) { a.post(func() { a.meetingUpdated(id) }) },
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

// systemCaptureOK is asked at each meeting start: openMeetings runs before
// the helper's handshake, when its methods are not known yet.
func (a *app) systemCaptureOK() bool {
	return a.host != nil && a.host.sup != nil && a.host.sup.Status().Has(proto.MethodCaptureSystem)
}

func (a *app) meetingActive() bool {
	return a.meetMgr != nil && a.meetSnap.State.Active()
}

func (a *app) meetingChanged(s meeting.Snapshot) {
	prev := a.meetSnap
	a.meetSnap = s
	if s.ID != prev.ID || s.State != prev.State {
		a.meetOvMenu = meetingMenuNone // a menu for the state it opened in
	}
	if a.mv.read && (s.ID != prev.ID || s.State != prev.State) {
		a.loadMeetings() // the meeting's row was written or its status changed (#3990)
	}
	a.followMeeting(s, prev)
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
	if a.meetMgr == nil {
		return
	}
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		_, err := a.meetMgr.Start(ctx, meeting.StartRequest{Mode: meetingModeFrom(a.prefs.Features.Meeting)})
		if err != nil {
			a.post(func() {
				a.meetingFailed("start", meetingStartProblem(err), err)
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
				a.meetingFailed("pause", i18n.T("Could not pause or resume the meeting: %v", err), err)
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
				a.meetingFailed("finish", i18n.T("Could not end the meeting: %v", err), err)
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
				a.meetingFailed("save", i18n.T("Could not save the meeting: %v", err), err)
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
				a.meetingFailed("reprocess", i18n.T("Could not transcribe the meeting again: %v", err), err)
			})
			return
		}
		a.post(a.loadMeetings)
	})
}

// meetingTranscriptContext is the dictionary and speech context a
// transcript's correction gets, as dictation's clean-up has them.
func (a *app) meetingTranscriptContext(ctx context.Context) meeting.TranscriptContext {
	d := a.prefs.Dictionary
	tc := meeting.TranscriptContext{Notes: session.PromptDictionary(a.prefs), Speech: d.SpeechContext}
	if d.Enabled && d.Glossary {
		terms, _ := a.lib.store.Dictionary().List(ctx, "")
		for _, t := range terms {
			if t.Enabled {
				tc.Terms = append(tc.Terms, compose.Term{Term: t.Term, Aliases: t.Aliases})
			}
		}
	}
	return tc
}

// meetingCorrect corrects a stored meeting's transcript with the language
// model, then summarizes it again.
func (a *app) meetingCorrect(id string) {
	if a.meetMgr == nil || id == "" {
		return
	}
	n := &a.mv
	if n.detail != nil && n.detail.id == id {
		n.detail.correcting = true
	}
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
		defer cancel()
		err := a.meetMgr.CorrectTranscript(ctx, id)
		if err == nil {
			err = a.meetMgr.GenerateMeetingSummary(ctx, id)
		}
		a.post(func() {
			if err != nil {
				a.toast(i18n.T("Could not correct the transcript: %v", err))
				a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: "correct: " + err.Error()})
			}
			if n.detail != nil && n.detail.id == id {
				a.openMeetingDetail(id)
			}
			a.loadMeetings()
		})
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
				a.meetingFailed("mode", i18n.T("Could not change the meeting mode: %v", err), err)
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
				a.meetingFailed("discard", i18n.T("Could not discard the meeting: %v", err), err)
			})
		}
	})
}

// meetingFailed tells the user an action on the meeting failed, and logs
// it: a failure only in the log looked like a button that did nothing
// (11-15, Start meeting without the speech model).
func (a *app) meetingFailed(what, msg string, err error) {
	a.toast(msg)
	a.logs.Append(core.LogEntry{Time: time.Now(), Source: meetingLogSource, Level: core.SeverityWarning, Message: what + ": " + err.Error()})
}

// meetingStartProblem is what the user reads when a meeting does not start.
func meetingStartProblem(err error) string {
	var se *meeting.StartError
	switch {
	case errors.Is(err, local.ErrNotInstalled):
		return i18n.T("The speech model for meetings is not downloaded. Download it in Settings → Meetings.")
	case errors.Is(err, meeting.ErrBusy):
		return i18n.T("A meeting is already recording.")
	case errors.As(err, &se):
		if se.Guidance == meeting.GuidanceModel {
			return i18n.T("Could not load the speech model for meetings: %v", se.Err)
		}
		denied := errors.Is(se.Err, &proto.Error{Code: proto.CodePermissionDenied})
		switch se.Source {
		case meeting.SourceSystem:
			if denied {
				return i18n.T("Could not record system audio. Allow system audio recording for Voxt in System Settings → Privacy & Security → Screen & System Audio Recording, or record the microphone only.")
			}
			return i18n.T("Could not record system audio. Try again, or record the microphone only.")
		case meeting.SourceMic:
			if denied {
				return i18n.T("Could not record the microphone. Allow Voxt in System Settings → Privacy & Security → Microphone.")
			}
			return i18n.T("Could not record the microphone. Check your input device and try again.")
		}
	}
	return i18n.T("Could not start the meeting: %v", err)
}
