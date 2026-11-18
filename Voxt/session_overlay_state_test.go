package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestSessionOverlayProjectionLocaleAndTime(t *testing.T) {
	now := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	input := sessionOverlayProjectionInput{
		Snapshot: session.Snapshot{
			Rev: 7, ID: "s1", Kind: store.KindTranslate, State: session.Recording,
			HeardAt: now.Add(-75 * time.Second), MaxLen: 2 * time.Minute,
			Partial: "the limit warning takes precedence", Levels: []float64{0.1, 0.2, 0.3},
		},
		Mode: ovRecording, ShowTime: true,
	}
	english := buildSessionPillViewState(i18n.NewTranslator("en"), input, now)
	if english.ElapsedText != "1:15" || english.LiveText != "The length limit stops the recording in 0:45" || !english.LiveHint {
		t.Fatalf("English time and limit: %+v", english)
	}
	if english.Title != "Translate" || english.WorkingLabel != "Recognising…" {
		t.Errorf("English labels: %+v", english)
	}
	if english.Levels != ([session.Bars]float64{13: 0.1, 14: 0.2, 15: 0.3}) {
		t.Errorf("levels were not copied into their fixed slots: %v", english.Levels)
	}

	translator := i18n.NewTranslator("zh-Hans")
	chinese := buildSessionPillViewState(translator, input, now)
	if chinese.LiveText != translator.T("The length limit stops the recording in %s", "0:45") || chinese.Title != translator.T("Translate") || chinese.WorkingLabel != translator.T("Recognising…") {
		t.Errorf("Chinese labels: %+v", chinese)
	}
	if chinese.ElapsedText != "1:15" {
		t.Errorf("locale changed the clock: %+v", chinese)
	}
	if again := buildSessionPillViewState(translator, input, now); again != chinese {
		t.Errorf("same translator and time produced different states:\n%+v\n%+v", chinese, again)
	}
}

func TestSessionOverlayProjectionPrivacyAndPillNote(t *testing.T) {
	now := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	input := sessionOverlayProjectionInput{
		Snapshot: session.Snapshot{ID: "s1", State: session.Recording, Partial: "  private\nwords  "},
		Mode:     ovRecording, HideLiveText: true,
	}
	state := buildSessionPillViewState(i18n.NewTranslator("en"), input, now)
	if state.LiveText != "" {
		t.Fatalf("privacy exposed live text: %q", state.LiveText)
	}
	input.PillNoteID, input.PillNoteText = "s1", "Switched microphone"
	state = buildSessionPillViewState(i18n.NewTranslator("en"), input, now)
	if state.LiveText != "Switched microphone" {
		t.Errorf("the pill note did not replace private text: %q", state.LiveText)
	}
	input.PillNoteID = "another-session"
	if state := buildSessionPillViewState(i18n.NewTranslator("en"), input, now); state.LiveText != "" {
		t.Errorf("another session's note appeared: %q", state.LiveText)
	}
	input.HideLiveText = false
	input.PillNoteID, input.PillNoteText = "", ""
	if state := buildSessionPillViewState(i18n.NewTranslator("en"), input, now); state.LiveText != "private words" {
		t.Errorf("visible live text was not normalized: %q", state.LiveText)
	}
}

func TestSessionOverlayProjectionClearsPillOutsidePillModes(t *testing.T) {
	a := &app{}
	now := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	record := func() {
		a.sess = session.Snapshot{ID: "recording", State: session.Recording, Partial: "private words", Levels: []float64{0.5}}
		a.refreshSessionOverlay(now)
		if a.ov.projection.view.Pill.LiveText != "private words" || !a.ov.projection.pillReady {
			t.Fatalf("recording pill was not built: %+v", a.ov.projection.view.Pill)
		}
	}
	assertCleared := func(mode overlayMode) {
		t.Helper()
		if a.ov.projection.view.Mode != mode || a.ov.projection.view.Pill != (sessionPillViewState{}) || a.ov.projection.pillReady {
			t.Fatalf("%s retained pill data: %+v, ready %v", mode, a.ov.projection.view.Pill, a.ov.projection.pillReady)
		}
	}
	record()
	a.answer.open = true
	a.sess = session.Snapshot{ID: "answer", State: session.Done, Card: true}
	a.refreshSessionOverlay(now)
	assertCleared(ovAnswer)
	record()
	a.answer.open = false
	a.notice = &sessionNotice{text: "failed"}
	a.sess = session.Snapshot{ID: "failed", State: session.Failed}
	a.refreshSessionOverlay(now)
	assertCleared(ovNotice)
	a.notice = nil
	record()
}

func TestSessionOverlayProjectionNoticeToneAndAction(t *testing.T) {
	translator := i18n.NewTranslator("en")
	for _, test := range []struct {
		name       string
		notice     sessionNotice
		tone       string
		action     string
		actionText string
	}{
		{name: "error", notice: sessionNotice{text: "Recognition failed", action: actionModels}, tone: sessionNoticeError, action: actionModels, actionText: "Open Speech models"},
		{name: "warning", notice: sessionNotice{text: "Nothing was heard", warn: true, action: actionRetry}, tone: sessionNoticeWarning, action: actionRetry, actionText: "Retry"},
		{name: "success", notice: sessionNotice{text: "Saved", ok: true, warn: true}, tone: sessionNoticeSuccess},
	} {
		t.Run(test.name, func(t *testing.T) {
			state := buildSessionNoticeViewState(translator, test.notice)
			if state.Text != test.notice.text || state.Tone != test.tone || state.Action != test.action || state.ActionLabel != test.actionText {
				t.Errorf("notice projection: %+v", state)
			}
		})
	}
	chinese := i18n.NewTranslator("zh-Hans")
	if state := buildSessionNoticeViewState(chinese, sessionNotice{action: actionRetry}); state.ActionLabel != chinese.T("Retry") {
		t.Errorf("notice action used the global locale: %q", state.ActionLabel)
	}
}

func TestSessionOverlayViewStateJSONOmitsDomain(t *testing.T) {
	hidden, err := json.Marshal(ovHidden)
	if err != nil || string(hidden) != `"hidden"` {
		t.Fatalf("hidden mode JSON: %s, %v", hidden, err)
	}
	var mode overlayMode
	if err := json.Unmarshal(hidden, &mode); err != nil || mode != ovHidden {
		t.Fatalf("hidden mode round trip: %q, %v", mode, err)
	}
	notice := sessionNotice{
		text: "Readable notice", more: "private detail", action: actionRetry,
		entry: "domain-entry", then: func() { panic("must not serialize or run") },
	}
	input := sessionOverlayProjectionInput{
		Snapshot: session.Snapshot{
			Rev: 9, ID: "s1", State: session.Failed,
			Problem: &session.Problem{Code: session.ProblemInternal, Err: errors.New("private error")},
		},
		Mode: ovNotice, Notice: &notice,
	}
	state := buildSessionOverlayViewState(i18n.NewTranslator("en"), input, time.Time{})
	encoded, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]any
	if err := json.Unmarshal(encoded, &fields); err != nil {
		t.Fatal(err)
	}
	if fields["state"] != "failed" || fields["mode"] != "notice" || fields["problem_code"] != session.ProblemInternal {
		t.Errorf("session codes missing from JSON: %s", encoded)
	}
	for _, key := range []string{"snapshot", "settings", "translator", "now", "error", "err", "then", "entry", "more"} {
		if _, present := fields[key]; present {
			t.Errorf("domain field %q crossed into JSON: %s", key, encoded)
		}
	}
	for _, private := range []string{"private detail", "private error", "domain-entry"} {
		if bytes.Contains(encoded, []byte(private)) {
			t.Errorf("domain value %q crossed into JSON: %s", private, encoded)
		}
	}
}
