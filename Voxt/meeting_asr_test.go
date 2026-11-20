package main

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
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
	a.prefs.General.MainLanguage = "en" // an unknown language keeps the selected model
	d, err = a.meetingDiarizer(context.Background())
	if err != nil || d != provider {
		t.Fatalf("with MTD: diarizer %v, err %v", d, err)
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

// MTD knows only zh and en: a meeting in another language keeps the
// selected model's transcription instead of a forced wrong-language one.
// An unknown language (English UI, no main language: no hint, #3398) is
// not zh or en either.
func TestMeetingDiarizerLanguageGate(t *testing.T) {
	provider := &meetingDiarizerFake{}
	for _, tc := range []struct {
		main string
		want bool
	}{
		{"zh-Hans", true},
		{"en", true},
		{"ko", false},
		{"ja", false},
		{"", false},
	} {
		t.Run(tc.main, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			a.local = &meetingMTDLocal{provider: provider}
			a.prefs.General.MainLanguage = tc.main
			d, err := a.meetingDiarizer(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if (d != nil) != tc.want {
				t.Errorf("%s: diarizer %v", tc.main, d)
			}
		})
	}
}

// meetingGateLocal lists MTD and Qwen3-ASR 1.7B (and System dictation
// with withSystem) and keeps the id it opened, for the Automatic
// language-gate tests.
type meetingGateLocal struct {
	local.NotInstalled
	opened     string
	withSystem bool
}

func (l *meetingGateLocal) Installed(kind local.Kind) []local.Model {
	if kind == local.ASR {
		out := []local.Model{
			{Kind: local.ASR, ID: models.ASRMoss, Name: "MOSS-Transcribe-Diarize", Stream: true},
			{Kind: local.ASR, ID: models.ASRAccurate, Name: "Qwen3-ASR 1.7B (8bit)", Stream: true},
		}
		if l.withSystem {
			out = append(out, local.Model{Kind: local.ASR, ID: dictation.ModelID, Name: "System dictation", Stream: true})
		}
		return out
	}
	return nil
}

func (l *meetingGateLocal) ASR(_ context.Context, id string) (asr.Provider, local.Model, error) {
	m, ok := local.Choose(l.Installed(local.ASR), id)
	if !ok {
		return nil, local.Model{}, local.NotInstalledError(local.ASR, id)
	}
	l.opened = m.ID
	return &fakeTranscriptions{}, m, nil
}

// A meeting on Automatic follows the features' language gate (#4067):
// MTD knows only zh and en, so a Japanese meeting opens Qwen3-ASR and a
// meeting in a language no installed model knows opens nothing instead
// of the wrong-language model. An explicit choice is untouched, and an
// unknown language keeps the first installed, as before.
func TestMeetingTrackASRLanguageGate(t *testing.T) {
	for _, tc := range []struct {
		name, main, meetingLang, explicit, want string
	}{
		{"automatic picks the model that knows the language", "ja", "", "", models.ASRAccurate},
		{"automatic keeps MTD for Chinese", "zh", "", "", models.ASRMoss},
		{"the meeting's own language wins over the main one", "zh", "ja", "", models.ASRAccurate},
		{"unknown language keeps the first installed", "", "", "", models.ASRMoss},
		{"explicit MTD is untouched", "ja", "", models.ASRMoss, models.ASRMoss},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l := &meetingGateLocal{}
			a, _, _ := newTestApp(t)
			a.local = l
			a.prefs.General.MainLanguage = tc.main
			a.prefs.Features.Meeting.Lang = tc.meetingLang
			a.prefs.Features.Meeting.ASR = tc.explicit
			if _, err := a.meetingTrackASR(context.Background(), meeting.SourceMic); err != nil {
				t.Fatal(err)
			}
			if l.opened != tc.want {
				t.Errorf("opened %s, want %s", l.opened, tc.want)
			}
			if tc.explicit == "" {
				if id, _ := a.firstOmniASR(); id != tc.want {
					t.Errorf("firstOmniASR %s, want %s", id, tc.want)
				}
			}
		})
	}
	t.Run("a language no installed model knows opens nothing", func(t *testing.T) {
		a, _, _ := newTestApp(t)
		a.local = &meetingGateLocal{}
		a.prefs.General.MainLanguage = "zz"
		if _, err := a.meetingTrackASR(context.Background(), meeting.SourceMic); !errors.Is(err, local.ErrNotInstalled) {
			t.Fatalf("err %v, want not installed", err)
		}
		if id, _ := a.firstOmniASR(); id != "" {
			t.Errorf("firstOmniASR %s, want none", id)
		}
	})
	t.Run("System dictation is the fallback nothing else knows", func(t *testing.T) {
		l := &meetingGateLocal{withSystem: true}
		a, _, _ := newTestApp(t)
		a.local = l
		a.prefs.General.MainLanguage = "zz"
		if _, err := a.meetingTrackASR(context.Background(), meeting.SourceMic); err != nil {
			t.Fatal(err)
		}
		if l.opened != dictation.ModelID {
			t.Errorf("opened %s, want System dictation", l.opened)
		}
	})
}
