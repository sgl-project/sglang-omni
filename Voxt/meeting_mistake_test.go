package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
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
	if tall.Height != meetingOvAskCardH || tall.Height <= short.Height || tall.Width != meetingOvAskCardW || tall.Width <= short.Width {
		t.Fatalf("the question as a card: %+v, as a row %+v", tall, short)
	}
	card, row := meetingOverlayLayoutModel(mistake, meetingMenuNone), meetingOverlayLayoutModel(plain, meetingMenuNone)
	if _, height := meetingOverlayFace(card); height != 2*meetingOvAskRadius {
		t.Fatalf("the card's corners come from a height of %d", height)
	}
	if _, height := meetingOverlayFace(row); height != row.Layout.Height-2*layout.Pad {
		t.Fatalf("the row keeps the pill's ends: %d", height)
	}
}

// meetingOfWords is the app over the fake helper and a speech model that
// hears text in every meeting.
func meetingOfWords(t *testing.T, text string, more ...func(*settings.Settings)) *sessionApp {
	t.Helper()
	return meetingOfWordsWith(t, text, "", more...)
}

// meetingOfWordsWith is meetingOfWords with the fake helper's options
// (opts: its audio, for one).
func meetingOfWordsWith(t *testing.T, text, opts string, more ...func(*settings.Settings)) *sessionApp {
	t.Helper()
	s := newSessionApp(t, opts, func(st *settings.Settings) {
		st.Features.Meeting.NoFinalPass, st.Features.Meeting.NoAutoSummary = true, true
		for _, f := range more {
			f(st)
		}
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
	// End shows "finishing" at once; the question or the filing follows.
	s.pump("its end", func() bool {
		return s.meetSnap.State != meeting.StateRecording && s.meetSnap.State != meeting.StateFinishing
	})
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
		if s.meetSnap.State != meeting.StateAsking || !s.meetSnap.Mistake || s.meetSnap.Stage != "" {
			t.Fatalf("one word: %s (stage %q), mistake %v", s.meetSnap.State, s.meetSnap.Stage, s.meetSnap.Mistake)
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
	t.Run("no answer, history off", func(t *testing.T) {
		// The question waits for the user, as "Save this meeting?" does.
		s := meetingOfWords(t, "hi", func(st *settings.Settings) { st.Privacy.KeepHistory = false })
		timers := (&fakeTimers{}).install(s.app)
		holdMeeting(s)
		timers.advance(2 * meetingMistakeWait)
		if s.meetSnap.State != meeting.StateAsking || !s.meetSnap.Mistake {
			t.Fatalf("answered by itself with history off: %s, mistake %v", s.meetSnap.State, s.meetSnap.Mistake)
		}
		s.meetingDiscard()
		s.pump("the discard", func() bool { return s.meetSnap.State == meeting.StateIdle })
		if rows := meetingRows(t, s); len(rows) != 0 {
			t.Fatalf("discarded: %+v", rows)
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

// A quiet room where the speech model writes words for nothing (11-25 on a
// Mac: Whisper Turbo wrote "Продолжение следует...", "you", "Gracias" on
// both tracks, 7 lines in a 58 s empty meeting, so it was filed without the
// question). Lines with no speech under them are dropped, so a short quiet
// meeting still asks "Started by mistake?".
func TestMeetingMistakeOverMadeUpWords(t *testing.T) {
	quiet := filepath.Join(t.TempDir(), "quiet.pcm")
	if err := os.WriteFile(quiet, make([]byte, 2), 0o644); err != nil {
		t.Fatal(err)
	}
	made := "Продолжение следует you Gracias thank you for watching"
	s := meetingOfWordsWith(t, made, helpertest.Opt("pcm", quiet)+","+helpertest.Opt("system", quiet))
	s.startMeeting()
	s.pump("the meeting", func() bool { return s.meetSnap.State == meeting.StateRecording })
	s.drain(time.Second) // the model writes; nobody speaks
	s.meetingFinish()
	s.pump("its end", func() bool {
		return s.meetSnap.State != meeting.StateRecording && s.meetSnap.State != meeting.StateFinishing
	})
	if s.meetSnap.State != meeting.StateAsking || !s.meetSnap.Mistake {
		t.Fatalf("made-up words in a quiet room: %s, mistake %v, live %q, %d segments", s.meetSnap.State, s.meetSnap.Mistake, s.meetSnap.Live, len(s.meetSnap.Segments))
	}
	s.meetingDiscard()
	s.pump("discarded", func() bool { return s.meetSnap.State == meeting.StateIdle })
}

// streamAndFile is a speech model that streams (live text) and takes the
// recording whole (the final pass), slowly enough to see the pass run.
type streamAndFile struct {
	fakeStreamer
	file *fakeTranscriptions
}

func (streamAndFile) Caps() asr.Caps { return asr.Caps{Stream: true, File: true} }
func (s streamAndFile) Transcribe(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	time.Sleep(100 * time.Millisecond)
	return s.file.Transcribe(ctx, req)
}

// Keep on "Started by mistake?" with the final pass on files the meeting
// through the pass: one "Meeting saved", once it is filed, not at the
// click while the overlay still says the transcript is being finalised.
func TestMeetingMistakeKeepWithFinalPass(t *testing.T) {
	s := newSessionApp(t, "", func(st *settings.Settings) {
		st.Features.Meeting.NoAutoSummary = true // the final pass runs
	}, &fakeLocal{ASRP: streamAndFile{fakeStreamer{final: "hi", partial: "hi"}, &fakeTranscriptions{text: "hi"}}})
	useTestSounds(t, s.app)
	s.playSound = func(sounds.Target) {}
	holdMeeting(s)
	if s.meetSnap.State != meeting.StateAsking || !s.meetSnap.Mistake {
		t.Fatalf("one word: %s, mistake %v", s.meetSnap.State, s.meetSnap.Mistake)
	}
	s.meetingSave()
	var early bool
	saved := 0
	last := 0
	s.pump("the final pass and its filing", func() bool {
		if s.notice != nil && s.notice.seq != last {
			last = s.notice.seq
			if s.notice.text == "Meeting saved" {
				saved++
				early = early || s.meetSnap.State == meeting.StateFinishing
			}
		}
		return !s.meetingActive()
	})
	s.drain(200 * time.Millisecond)
	if s.notice != nil && s.notice.seq != last && s.notice.text == "Meeting saved" {
		saved++
	}
	if saved != 1 || early {
		t.Fatalf("\"Meeting saved\" %d time(s), one while still finishing: %v", saved, early)
	}
	if rows := meetingRows(t, s); len(rows) != 1 || rows[0].Status != store.MeetingDone {
		t.Fatalf("kept: %+v", rows)
	}
}
