package main

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type meetingDiarizerFake struct{ fakeTranscriptions }

func (f *meetingDiarizerFake) Diarize(context.Context, asr.FileRequest) (asr.Result, error) {
	return asr.Result{}, nil
}

type meetingMTDLocal struct {
	local.NotInstalled
	provider asr.Provider
}

func (l *meetingMTDLocal) Installed(kind local.Kind) []local.Model {
	if kind == local.ASR {
		return []local.Model{{Kind: local.ASR, ID: models.ASRMoss}}
	}
	return nil
}

func (l *meetingMTDLocal) ASR(_ context.Context, id string) (asr.Provider, local.Model, error) {
	if id != models.ASRMoss {
		return nil, local.Model{}, local.NotInstalledError(local.ASR, id)
	}
	return l.provider, local.Model{Kind: local.ASR, ID: models.ASRMoss}, nil
}

func TestMeetingDiarizerRequiresInstalledMTD(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.local = &fakeLocal{ASRP: &fakeTranscriptions{}}
	d, err := a.meetingDiarizer(context.Background())
	if err != nil || d != nil {
		t.Fatalf("without MTD: diarizer %v, err %v", d, err)
	}
	provider := &meetingDiarizerFake{}
	a.local = &meetingMTDLocal{provider: provider}
	a.prefs.General.MainLanguage = "en"
	d, err = a.meetingDiarizer(context.Background())
	if err != nil || d != provider {
		t.Fatalf("with MTD: diarizer %v, err %v", d, err)
	}
}

func TestMeetingDiarizerLanguageGate(t *testing.T) {
	for _, language := range []string{"", "en", "zh-Hans", "ko"} {
		t.Run(language, func(t *testing.T) {
			provider := &meetingDiarizerFake{}
			a, _, _ := newTestApp(t)
			a.local = &meetingMTDLocal{provider: provider}
			a.prefs.General.MainLanguage = language
			d, err := a.meetingDiarizer(context.Background())
			want := language == "en" || language == "zh-Hans"
			if err != nil || (d != nil) != want {
				t.Fatalf("language %q: diarizer %v, err %v; want provider %v", language, d, err, want)
			}
		})
	}
}

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
