package main

import (
	"context"
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func TestMeetingSettingsModeKey(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.prefs.Features.Meeting.Mode = "subtitles"
	if a.meetingSettingsKey() != "subtitles|0" {
		t.Fatalf("mode key %q", a.meetingSettingsKey())
	}
}

func TestMeetingFeatureLabels(t *testing.T) {
	if uitext.FeaturePrefix(hotkey.Meeting) != "mt" {
		t.Fatal("prefix")
	}
	if uitext.FeatureTitle(hotkey.Meeting) == "" {
		t.Fatal("title")
	}
}

// TestMeetingSpeechSettings: meeting settings choose the speech model
// (Automatic by default, as Transcribe's menu) and the language it
// listens for, which the meeting's live and final requests then carry;
// Automatic and Same as the main language go back to the defaults (#3991).
func TestMeetingSpeechSettings(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400)
	a.navigate("/meetings")
	a.mvUI.setOpen = true
	frames(tt)
	const speakerGuidance = "The language the speech model listens for. Choose one when the whole meeting is in it. For Speaker 1–4 labels, install MOSS-Transcribe-Diarize and choose Chinese or English here."
	if !has(tt, speakerGuidance) {
		t.Fatalf("spoken-language help does not explain speaker labels: %q", tt.Texts())
	}
	if !slices.Contains(tt.AccessLabels(), "Speech model for meetings") {
		t.Fatalf("no speech model menu: %q", tt.AccessLabels())
	}
	readyModels(a)
	a.updateSettings(func(s *settings.Settings) { s.General.MainLanguage = "zh" })
	frames(tt)

	auto := "Automatic (now: " + fakeASRModel.Name + ")"
	click(t, tt, "Speech model for meetings")
	if texts := tt.Texts(); !slices.Contains(texts, auto) || !slices.Contains(texts, fakeASRModel.Name) {
		t.Fatalf("menu: %q", texts)
	}
	click(t, tt, fakeASRModel.Name)
	if id := a.lib.settings.Get().Features.Meeting.ASR; id != fakeASRModel.ID {
		t.Fatalf("speech model %q, want %q", id, fakeASRModel.ID)
	}

	ja := uitext.LangLabel("ja")
	click(t, tt, "Spoken language in meetings")
	click(t, tt, ja)
	if l := a.lib.settings.Get().Features.Meeting.Lang; l != "ja" {
		t.Fatalf("spoken language %q", l)
	}
	ta, err := a.meetingTrackASR(context.Background(), meeting.SourceSystem)
	if err != nil {
		t.Fatal(err)
	}
	if ta.Lang != "ja" || ta.StreamReq.Lang != "ja" {
		t.Errorf("meeting hint: file %q, stream %q; want ja over the main language", ta.Lang, ta.StreamReq.Lang)
	}

	click(t, tt, "Speech model for meetings")
	click(t, tt, auto)
	click(t, tt, "Spoken language in meetings")
	click(t, tt, "Same as the main language")
	m := a.lib.settings.Get().Features.Meeting
	if m.ASR != "" || m.Lang != "" {
		t.Fatalf("not back to the defaults: %+v", m)
	}
	if ta, _ := a.meetingTrackASR(context.Background(), meeting.SourceSystem); ta.Lang != "zh" {
		t.Errorf("default hint %q, want the main language", ta.Lang)
	}
}

// #4134: Meeting settings edit, save and reset the correction style.
func TestMeetingCorrectionStyleSettings(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400)
	a.navigate("/meetings")
	a.mvUI.setOpen = true
	frames(tt)
	const label = "Correction style for meeting transcripts"
	if !slices.Contains(tt.AccessLabels(), label) {
		t.Fatalf("no correction style editor: %q", tt.AccessLabels())
	}
	custom := "Keep the conversation's tone. 保留英文产品名。"
	click(t, tt, label)
	tt.Command("selectAll")
	tt.Type(custom)
	frames(tt)
	a.saveMeetingCorrectionStyle()
	if got := a.lib.settings.Get().Features.Meeting.CorrectionStyle; got != custom {
		t.Fatalf("saved style %q, want %q", got, custom)
	}
	a.meetingUI.correction.text = custom + " \n"
	a.saveMeetingCorrectionStyle()
	if got := a.lib.settings.Get().Features.Meeting.CorrectionStyle; got != custom {
		t.Fatalf("trimmed saved style %q, want %q", got, custom)
	}
	if got := a.meetingUI.correction.text; got != custom+" \n" {
		t.Fatalf("autosave changed the focused editor to %q", got)
	}
	frames(tt)
	click(t, tt, "Reset to default prompt")
	if got := a.lib.settings.Get().Features.Meeting.CorrectionStyle; got != "" || a.meetingUI.correction.text != meeting.DefaultCorrectionStyle {
		t.Fatalf("reset saved %q, editor %q", got, a.meetingUI.correction.text)
	}
	a.meetingUI.correction.text = " \n"
	a.saveMeetingCorrectionStyle()
	if got := a.lib.settings.Get().Features.Meeting.CorrectionStyle; got != "" || a.meetingUI.correction.text != meeting.DefaultCorrectionStyle {
		t.Fatalf("blank saved %q, editor %q", got, a.meetingUI.correction.text)
	}
}

// TestMeetingSpeechSettingsNoModel: with no speech model installed the
// menu reads Automatic and cannot open.
func TestMeetingSpeechSettingsNoModel(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400)
	a.navigate("/meetings")
	a.mvUI.setOpen = true
	frames(tt)
	_ = tt.ClickControl("Speech model for meetings")
	frames(tt)
	if slices.Contains(tt.Texts(), "Automatic (now: ") || a.lib.settings.Get().Features.Meeting.ASR != "" {
		t.Fatalf("menu opened with no model: %q", tt.Texts())
	}
	if !has(tt, "Automatic") {
		t.Fatalf("no Automatic label: %q", tt.Texts())
	}
}
