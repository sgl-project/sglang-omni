package main

// Meetings as the UI sees them: internal/meeting.Manager wired to the
// helper's dual capture, the store, and the meetings folder under the
// data directory; snapshots drive the meeting overlay and block dictation
// shortcuts while a meeting is active.

import (
	"context"
	"net/url"
	"path/filepath"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

const meetingLogSource = "meeting"

type meetingState struct {
	meetMgr            *meeting.Manager
	meetSnap           meeting.Snapshot
	meetingUI          *meetingSettingsUI // settings_meeting.go
	meetingDetectState                    // meeting_detect.go
	// savedNotice is the meeting "Meeting saved" was last shown for: a
	// meeting kept from its question says it once, after its final pass.
	savedNotice string
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
		Diarizer: a.meetingDiarizer,
		VAD:      a.meetingVAD(),
		LLM: func(ctx context.Context, id string) (meeting.SummaryChat, error) {
			chat, _, err := a.local.LLM(ctx, id)
			return chat, err
		},
		LLMIsLocal: a.languageModelIsLocal,
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

// languageModelIsLocal says whether the language model id is served on
// this Mac (a loopback address): its meeting correction does not overlap
// the final pass, which needs the GPU and memory for recognition.
func (a *app) languageModelIsLocal(id string) bool {
	endpoint := ""
	for _, p := range a.prefs.Providers.LLM {
		if p.ID == id {
			endpoint = p.Endpoint
			break
		}
	}
	preset, err := llm.ProviderConfig{ID: id, BaseURL: endpoint}.Resolve()
	if err != nil {
		return false
	}
	u, err := url.Parse(preset.BaseURL)
	return err == nil && core.IsLoopback(u.Hostname())
}

func (a *app) meetingActive() bool {
	return a.meetMgr != nil && a.meetSnap.State.Active()
}

func (a *app) meetingChanged(s meeting.Snapshot) {
	prev := a.meetSnap
	a.meetSnap = s
	if s.ID != prev.ID || s.State != prev.State {
		a.meetOvUI.menu = meetingMenuNone // a menu for the state it opened in
		a.meetOvUI.endAsk = false         // and End's question
	}
	if a.mv.read && (s.ID != prev.ID || s.State != prev.State) {
		a.loadMeetings() // the meeting's row was written or its status changed (#3990)
	}
	a.followMeeting(s, prev)
	if prev.State == meeting.StateFinishing && prev.ID != "" && (s.State == meeting.StateDone || s.State == meeting.StateIdle) {
		// The overlay goes once the meeting is filed: say where it went.
		a.noticeMeetingSaved(prev.ID)
	}
	if s.State.Active() != prev.State.Active() {
		a.closeMeetingAsk() // meeting_detect.go: it records now
		a.ask.snoozed = ""  // and is not asked about again later
		if !s.State.Active() {
			a.recorded = meetingRecorded{quietAsk: a.recorded.quietAsk} // meeting_detect_end.go
		}
		a.syncMeetingDetect() // an app that records is settled while Voxt does
	}
	a.followRecordedMeeting() // meeting_detect_end.go: quiet, a sound, or ended in Voxt
	if s.State == meeting.StateAsking && s.Mistake && (prev.State != meeting.StateAsking || prev.ID != s.ID) {
		a.keepUnanswered(s.ID)
	}
	a.syncMeetingOverlay()
	a.syncHotkeys()
	if a.tray != nil {
		a.buildTrayMenu()
	}
}

// meetingMistakeWait is how long "Started by mistake?" waits for an
// answer. A meeting that asks blocks dictation and the next meeting, so a
// question nobody answers is answered Keep: nothing is lost.
const meetingMistakeWait = 30 * time.Second

// keepUnanswered keeps meeting id when its question is still open after
// meetingMistakeWait. With history off it waits for the user, as "Save
// this meeting?" does: kept by itself, it would keep what they chose not
// to.
func (a *app) keepUnanswered(id string) {
	a.later(meetingMistakeWait, func() {
		if s := a.meetSnap; s.ID == id && s.State == meeting.StateAsking && s.Mistake && a.prefs.Privacy.KeepHistory {
			a.meetingSave()
		}
	})
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

func (a *app) startMeeting() { a.startMeetingThen(nil) }

// startMeetingThen starts a meeting; failed, when not nil, also hears why
// one did not start (the toast is the main window's, which may be closed).
func (a *app) startMeetingThen(failed func(msg string, err error)) {
	if a.meetMgr == nil {
		return
	}
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		_, err := a.meetMgr.Start(ctx, meeting.StartRequest{Mode: meetingModeFrom(a.prefs.Features.Meeting)})
		if err != nil {
			a.post(func() {
				msg := meetingStartProblem(err)
				a.meetingFailed("start", msg, err)
				if failed != nil {
					failed(msg, err)
				}
			})
		}
	})
}

func (a *app) meetingPauseResume() {
	if a.meetMgr == nil || a.meetSnap.ID == "" || a.meetOvPending != "" {
		return // one at a time: the last press is still on its way
	}
	id, target := a.meetSnap.ID, meeting.StatePaused
	switch a.meetSnap.State {
	case meeting.StateRecording:
	case meeting.StatePaused:
		target = meeting.StateRecording
	default:
		return
	}
	a.meetOvPending = target // shown at once (meeting_overlay_window.go)
	a.syncMeetingOverlay()
	a.async(func() {
		ctx := context.Background()
		var err error
		if target == meeting.StatePaused {
			err = a.meetMgr.Pause(ctx, id)
		} else {
			err = a.meetMgr.Resume(ctx, id)
		}
		a.post(func() {
			a.meetOvPending = "" // the manager's state has come (or the press failed)
			a.syncMeetingOverlay()
			if err != nil {
				a.meetingFailed("pause", i18n.T("Could not pause or resume the meeting: %v", err), err)
			}
		})
	})
}

func (a *app) meetingFinish() {
	if a.meetMgr == nil || !a.meetingActive() || a.meetSnap.State == meeting.StateFinishing || a.meetSnap.State == meeting.StateAsking {
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
		a.post(func() {
			a.loadMeetings()
			if a.meetSnap.ID == id && a.meetSnap.State == meeting.StateFinishing {
				return // its final pass runs: the notice comes once it is filed
			}
			a.noticeMeetingSaved(id)
		})
	})
}

// noticeMeetingSaved says once where meeting id went: "Meeting saved",
// with View meeting.
func (a *app) noticeMeetingSaved(id string) {
	if a.savedNotice == id {
		return
	}
	a.savedNotice = id
	a.showNotice(sessionNotice{text: i18n.T("Meeting saved"), action: actionMeeting, entry: id, ok: true})
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
		a.rebuildMeetingDetail()
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
