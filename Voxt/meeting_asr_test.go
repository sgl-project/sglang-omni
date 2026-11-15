package main

import (
	"context"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
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
