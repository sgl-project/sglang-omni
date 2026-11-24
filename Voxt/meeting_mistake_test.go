package main

import (
	"context"
	"slices"
	"testing"

	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// The overlay's question says why it asks: a meeting that ended with
// hardly a word is "Started by mistake?", with a line under it.
func TestMeetingMistakeOverlay(t *testing.T) {
	asking := meeting.Snapshot{ID: "m1", State: meeting.StateAsking}
	plain := buildMeetingOverlayViewState(i18n.NewTranslator("en"), asking, false, 1)
	if plain.AskingText != "Save this meeting?" || plain.KeepLabel != "Save" || plain.DiscardLabel != "Discard" || plain.AskingHint != "" {
		t.Fatalf("history off: %+v", plain)
	}
	asking.Mistake = true
	mistake := buildMeetingOverlayViewState(i18n.NewTranslator("en"), asking, false, 1)
	if mistake.AskingText != "Started by mistake?" || mistake.AskingHint != "Only a few words were captured." || mistake.KeepLabel != "Keep" || mistake.DiscardLabel != "Discard" {
		t.Fatalf("few words: %+v", mistake)
	}
	zh := buildMeetingOverlayViewState(i18n.NewTranslator("zh-Hans"), asking, false, 1)
	if zh.AskingText != "是误开的吗？" || zh.AskingHint != "只录到几个字。" || zh.KeepLabel != "保留" || zh.DiscardLabel != "丢弃" {
		t.Fatalf("few words, Chinese: %+v", zh)
	}
	short, tall := meetingOverlayLayoutModel(plain, meetingMenuNone).Layout, meetingOverlayLayoutModel(mistake, meetingMenuNone).Layout
	if tall.Height != short.Height+meetingOvAskHintH || tall.Width != short.Width {
		t.Fatalf("the question with its line: %+v, without %+v", tall, short)
	}
}

// meetingOfWords is the app over the fake helper and a speech model that
// hears text in every meeting.
func meetingOfWords(t *testing.T, text string) *sessionApp {
	t.Helper()
	s := newSessionApp(t, "", func(st *settings.Settings) {
		st.Features.Meeting.NoFinalPass, st.Features.Meeting.NoAutoSummary = true, true
	}, &fakeLocal{ASRP: fakeStreamer{final: text, partial: text}})
	// A meeting plays no cue; were one played, it would be silent.
	useTestSounds(t, s.app)
	s.playSound = func(sounds.Target) {}
	return s
}

// holdMeeting records a meeting until it heard something, then ends it.
func holdMeeting(s *sessionApp) {
	s.t.Helper()
	s.startMeeting()
	s.pump("the meeting", func() bool { return s.meetSnap.State == meeting.StateRecording })
	s.pump("its live text", func() bool { return len(s.meetSnap.Live) > 0 || len(s.meetSnap.Segments) > 0 })
	s.meetingFinish()
	s.pump("its end", func() bool { return s.meetSnap.State != meeting.StateRecording })
}

func meetingRows(t *testing.T, s *sessionApp) []store.Meeting {
	t.Helper()
	list, err := s.lib.store.Meetings().ListMeetings(context.Background(), store.MeetingQuery{})
	if err != nil {
		t.Fatal(err)
	}
	return list
}

// A whole meeting in the app, over the fake helper and a speech model that
// hears one word: it ends with the question, and Discard, Keep or no
// answer each leave the app free for the next meeting.
func TestMeetingMistakeEndToEnd(t *testing.T) {
	t.Run("discard", func(t *testing.T) {
		s := meetingOfWords(t, "hi")
		holdMeeting(s)
		if s.meetSnap.State != meeting.StateAsking || !s.meetSnap.Mistake {
			t.Fatalf("one word: %s, mistake %v", s.meetSnap.State, s.meetSnap.Mistake)
		}
		if view := s.meetOvView; !view.ShowAsking || view.AskingText != "Started by mistake?" {
			t.Fatalf("the overlay: %+v", view)
		}
		s.meetingDiscard()
		s.pump("the discard", func() bool { return s.meetSnap.State == meeting.StateIdle })
		if rows := meetingRows(t, s); len(rows) != 0 || s.meetingActive() {
			t.Fatalf("discarded: %d rows, active %v", len(rows), s.meetingActive())
		}
		// The app is free again: the next meeting starts.
		s.startMeeting()
		s.pump("the next meeting", func() bool { return s.meetSnap.State == meeting.StateRecording })
		s.meetingDiscard()
		s.pump("its discard", func() bool { return s.meetSnap.State == meeting.StateIdle })
	})
	t.Run("keep", func(t *testing.T) {
		s := meetingOfWords(t, "hi")
		holdMeeting(s)
		s.meetingSave()
		s.pump("the save", func() bool { return s.meetSnap.State == meeting.StateIdle })
		if rows := meetingRows(t, s); len(rows) != 1 || rows[0].Status != store.MeetingDone {
			t.Fatalf("kept: %+v", rows)
		}
	})
	t.Run("no answer", func(t *testing.T) {
		s := meetingOfWords(t, "hi")
		timers := (&fakeTimers{}).install(s.app)
		holdMeeting(s)
		timers.advance(meetingMistakeWait - 1)
		if s.meetSnap.State != meeting.StateAsking {
			t.Fatalf("answered before its time: %s", s.meetSnap.State)
		}
		timers.advance(1)
		s.pump("the keep", func() bool { return s.meetSnap.State == meeting.StateIdle })
		if rows := meetingRows(t, s); len(rows) != 1 || rows[0].Status != store.MeetingDone {
			t.Fatalf("unanswered: %+v", rows)
		}
	})
	t.Run("a real meeting", func(t *testing.T) {
		s := meetingOfWords(t, "let us begin the weekly meeting now")
		holdMeeting(s)
		s.pump("its filing", func() bool { return s.meetSnap.State == meeting.StateIdle })
		if rows := meetingRows(t, s); len(rows) != 1 || rows[0].Status != store.MeetingDone {
			t.Fatalf("seven words: %+v", rows)
		}
	})
}

// The question's buttons call the overlay's actions.
func TestMeetingMistakeDemoControls(t *testing.T) {
	for label, want := range map[string]string{"Keep": "meetingSave", "Discard": "meetingDiscard"} {
		host, err := loadMeetingDemoHost("testdata/meeting")
		if err != nil {
			t.Fatal(err)
		}
		selectMeetingDemoFixture(t, host, "en/state-asking-mistake")
		tester := newMeetingDemoTester(t, host)
		if !tester.HasText("Started by mistake?") || !tester.HasText("Only a few words were captured.") {
			t.Fatalf("the question: %q", tester.Texts())
		}
		if err := tester.ClickControl(label); err != nil {
			t.Fatalf("%s: %v (controls %q)", label, err, tester.AccessLabels())
		}
		frames(tester)
		if !slices.Equal(host.calls, []string{want}) {
			t.Fatalf("%s: calls %q", label, host.calls)
		}
	}
}
