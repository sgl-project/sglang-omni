package main

import (
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type tryNoticeViewState struct {
	Text        string `json:"text"`
	More        string `json:"more"`
	Action      string `json:"action"`
	ActionLabel string `json:"action_label"`
	Dismiss     bool   `json:"dismiss"`
}

type tryRecordingViewState struct {
	Kind        string    `json:"kind"`
	Levels      []float64 `json:"levels"`
	Heard       bool      `json:"heard"`
	ElapsedText string    `json:"elapsed_text"`
	LiveText    string    `json:"live_text"`
}

type tryEndedViewState struct {
	Cancelled   bool                `json:"cancelled"`
	FailureText string              `json:"failure_text"`
	Failure     *tryNoticeViewState `json:"failure,omitempty"`
	Warning     *tryNoticeViewState `json:"warning,omitempty"`
	Result      string              `json:"result"`
	Highlights  []dict.Span         `json:"highlights,omitempty"`
	RawTitle    string              `json:"raw_title"`
	Raw         string              `json:"raw"`
	TimingsText string              `json:"timings_text"`
}

type tryCardViewState struct {
	Prefix      string                 `json:"prefix"`
	Feature     string                 `json:"feature"`
	State       string                 `json:"state"`
	ProblemCode string                 `json:"problem_code"`
	Mode        string                 `json:"mode"`
	Intro       string                 `json:"intro"`
	Block       *tryNoticeViewState    `json:"block,omitempty"`
	StartLabel  string                 `json:"start_label"`
	Busy        bool                   `json:"busy"`
	BusyText    string                 `json:"busy_text"`
	Recording   *tryRecordingViewState `json:"recording,omitempty"`
	WorkingText string                 `json:"working_text"`
	Draft       string                 `json:"draft"`
	// SessionID keys the draft's and the result's preview (#4105).
	SessionID string             `json:"session_id,omitempty"`
	Ended     *tryEndedViewState `json:"ended,omitempty"`
}

type tryNoticeStateInput struct {
	Text    string
	More    string
	Action  string
	Dismiss bool
}

type tryCardStateInput struct {
	Prefix   string
	Feature  string
	Mine     bool
	Intro    string
	Block    *tryNoticeStateInput
	Snapshot session.Snapshot
	Now      time.Time
}

func newTryCardStateInput(prefix, feature string, mine bool, intro string, block *sessionNotice, snapshot session.Snapshot, now time.Time) tryCardStateInput {
	input := tryCardStateInput{Prefix: prefix, Feature: feature, Mine: mine, Intro: intro, Snapshot: snapshot, Now: now}
	if block != nil {
		input.Block = &tryNoticeStateInput{Text: block.text, More: block.more, Action: block.action, Dismiss: true}
	}
	return input
}

func buildTryIntro(translator *i18n.Translator, feature, keys string) string {
	switch feature {
	case hotkey.Translate:
		if keys != "" {
			return translator.T("Press %s or Start, then say a sentence or two. The translation shows here and is not typed anywhere.", keys)
		}
		return translator.T("Say a sentence or two. The translation shows here and is not typed anywhere.")
	case hotkey.Rewrite:
		if keys != "" {
			return translator.T("Press %s or Start, then say what to write (\"a short note saying I will be late\"). The answer shows here and is not typed anywhere.", keys)
		}
		return translator.T("Say what to write (\"a short note saying I will be late\"). The answer shows here and is not typed anywhere.")
	}
	if keys != "" {
		return translator.T("Press %s or Start, then say a sentence or two. The result shows here and is not typed anywhere.", keys)
	}
	return translator.T("Say a sentence or two. The result shows here and is not typed anywhere.")
}

func buildTryCardViewState(translator *i18n.Translator, input tryCardStateInput) tryCardViewState {
	snapshot := input.Snapshot
	state := tryCardViewState{
		Prefix: input.Prefix, Feature: input.Feature, State: string(snapshot.State), Mode: "idle", Intro: input.Intro,
		StartLabel: translator.T("Start"), Busy: snapshot.State.Live(), Block: buildTryNotice(translator, input.Block),
	}
	if snapshot.Problem != nil {
		state.ProblemCode = snapshot.Problem.Code
	}
	if state.Busy {
		state.BusyText = translator.T("Another task is running.")
		if snapshot.Kind == store.KindTranscribe {
			state.BusyText = translator.T("A transcription is running.")
		}
	}
	if input.Mine && snapshot.State.Ended() {
		state.StartLabel = translator.T("Try again")
	}
	if input.Mine && snapshot.State == session.Recording {
		state.Mode = "recording"
		state.Recording = &tryRecordingViewState{
			Kind: string(snapshot.Kind), Levels: snapshot.Levels,
			Heard: !snapshot.HeardAt.IsZero(), ElapsedText: tryElapsedText(snapshot.Elapsed(input.Now)),
			LiveText: strings.TrimSpace(snapshot.Partial),
		}
		return state
	}
	if input.Mine && snapshot.State.Live() {
		state.Mode = "working"
		state.WorkingText = uitext.WorkingTextFor(translator, snapshot)
		if snapshot.State == session.Composing {
			state.Draft = strings.TrimSpace(snapshot.Draft)
			state.SessionID = snapshot.ID
		}
		return state
	}
	if !input.Mine || !snapshot.State.Ended() {
		return state
	}
	state.Mode = "ended"
	state.SessionID = snapshot.ID
	ended := &tryEndedViewState{Result: snapshot.Result, Highlights: snapshot.Highlights, TimingsText: uitext.TimingsTextFor(translator, snapshot)}
	state.Ended = ended
	if snapshot.State == session.Cancelled {
		ended.Cancelled = true
		return state
	}
	if snapshot.Problem != nil {
		text, more, action := sessionProblemTextFor(translator, snapshot.Problem)
		notice := buildTryNotice(translator, &tryNoticeStateInput{Text: text, More: more, Action: action})
		if snapshot.State == session.Failed {
			if notice != nil && notice.Text != "" {
				ended.Failure = notice
				return state
			}
			ended.FailureText = translator.T("Nothing was heard.")
			return state
		}
		if notice != nil && notice.Text != "" {
			ended.Warning = notice
		}
	}
	if snapshot.State == session.Failed {
		ended.FailureText = translator.T("Nothing was heard.")
		return state
	}
	raw := strings.TrimSpace(snapshot.Raw)
	if raw != "" && raw != strings.TrimSpace(snapshot.Result) {
		ended.Raw, ended.RawTitle = raw, translator.T("As recognised")
		if snapshot.Kind == store.KindRewrite {
			ended.RawTitle = translator.T("Your instruction")
		}
	}
	return state
}

func buildTryNotice(translator *i18n.Translator, input *tryNoticeStateInput) *tryNoticeViewState {
	if input == nil {
		return nil
	}
	return &tryNoticeViewState{
		Text: input.Text, More: input.More, Action: input.Action,
		ActionLabel: tryNoticeActionLabel(translator, input.Action), Dismiss: input.Dismiss,
	}
}

func tryNoticeActionLabel(translator *i18n.Translator, action string) string {
	switch action {
	case actionSignIn:
		return translator.T("Sign in")
	case actionUsage:
		return translator.T("Manage usage")
	case actionModels:
		return translator.T("Open Speech models")
	case actionLanguage:
		return translator.T("Open Language models")
	case actionPermissions:
		return translator.T("Open Permissions")
	case actionHelper:
		return translator.T("Restart helper")
	case actionRetry:
		return translator.T("Retry")
	}
	return ""
}

func tryElapsedText(elapsed time.Duration) string {
	seconds := int(elapsed.Seconds())
	return itoa(seconds/60) + ":" + string(rune('0'+seconds%60/10)) + string(rune('0'+seconds%10))
}
