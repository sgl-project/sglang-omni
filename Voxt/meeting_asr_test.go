package main

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// A meeting gives its speech model the language hint dictation gives it
// (the main language, which says Simplified or Traditional Chinese), on
// the live path and the final pass alike (#3387).
func TestMeetingTrackASRUsesDictationLanguage(t *testing.T) {
	for _, tc := range []struct{ main, want string }{
		{"zh-Hant", "zh-Hant"},
		{"zh-Hans", "zh-Hans"},
		{"", ""}, // English UI, no main language: no hint (#3398)
	} {
		a, _, _ := newTestApp(t)
		a.local = &fakeLocal{ASRP: &fakeTranscriptions{}}
		a.prefs.General.MainLanguage = tc.main
		ta, err := a.meetingTrackASR(context.Background(), meeting.SourceMic)
		if err != nil {
			t.Fatal(err)
		}
		if ta.Lang != tc.want || ta.StreamReq.Lang != tc.want {
			t.Errorf("main %q: file Lang %q, stream Lang %q; want %q", tc.main, ta.Lang, ta.StreamReq.Lang, tc.want)
		}
	}
}

// A meeting gives its speech model the dictionary terms and the speech
// context dictation gives it, on the live path and the final pass alike;
// with the dictionary or its speech-model hints off it gives no terms
// (#3988).
func TestMeetingTrackASRUsesDictionary(t *testing.T) {
	for _, tc := range []struct {
		name          string
		dict, asrHint bool
		want          []string
	}{
		{"on", true, true, []string{"Voxt", "Typeless"}},
		{"dictionary off", false, true, nil},
		{"hints off", true, false, nil},
	} {
		a, _, _ := newTestApp(t)
		a.local = &fakeLocal{ASRP: &fakeTranscriptions{}}
		a.prefs.Dictionary.Enabled, a.prefs.Dictionary.ASRHints = tc.dict, tc.asrHint
		a.prefs.Dictionary.SpeechContext = "A Zoom call about Voxt"
		now := time.Now()
		for _, tm := range []store.Term{
			{ID: "1", Term: "Typeless", Enabled: true, UpdatedAt: now},
			{ID: "2", Term: "Voxt", Enabled: true, UpdatedAt: now.Add(time.Second)},
			{ID: "3", Term: "Disabled", Enabled: false, UpdatedAt: now},
		} {
			if err := a.lib.store.Dictionary().Put(context.Background(), tm); err != nil {
				t.Fatal(err)
			}
		}
		ta, err := a.meetingTrackASR(context.Background(), meeting.SourceSystem)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(ta.Hints, tc.want) || !slices.Equal(ta.StreamReq.Hints, tc.want) {
			t.Errorf("%s: file Hints %q, stream Hints %q; want %q", tc.name, ta.Hints, ta.StreamReq.Hints, tc.want)
		}
		if !strings.Contains(ta.Prompt, "A Zoom call about Voxt") || ta.StreamReq.Prompt != ta.Prompt {
			t.Errorf("%s: file Prompt %q, stream Prompt %q; want the speech context", tc.name, ta.Prompt, ta.StreamReq.Prompt)
		}
	}
}
