package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

const speechContextNote = "The speech models in use do not take it while they recognise. The language model still gets it when it corrects a meeting's transcript, and your dictionary still guides its clean-up."

// TestSpeechContextNoteWithoutTaker: with Cohere Transcribe, which takes
// no context, as Transcribe, Translate, Rewrite and Meetings' speech
// model, the Speech context says so and keeps its field, which meeting
// correction reads; one feature on Qwen3-ASR drops the note (#4067).
// MOSS takes the dictionary since #4418.
func TestSpeechContextNoteWithoutTaker(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 4800)
	seedModels(t, a, models.ASRCohere, models.ASRAccurate, models.VADSilero)
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Transcribe.ASR, s.Features.Translate.ASR, s.Features.Rewrite.ASR = models.ASRCohere, models.ASRCohere, models.ASRCohere
		s.Features.Meeting.ASR = models.ASRCohere
	})
	a.navigate(modelsPath)
	frames(tt)
	if !has(tt, speechContextNote) {
		t.Errorf("Cohere everywhere: no note: %q", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Rewrite.ASR = models.ASRAccurate })
	frames(tt)
	if has(tt, speechContextNote) {
		t.Errorf("Rewrite on Qwen3-ASR still has the note: %q", tt.Texts())
	}
}

// TestSpeechContextNoteMeetingTaker: a meeting's speech model reads the
// speech context too, so the note hides while Meetings uses a model that
// takes it, even with Transcribe, Translate and Rewrite on Cohere (#4069).
// Meetings on Cohere keeps the note.
func TestSpeechContextNoteMeetingTaker(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 4800)
	seedModels(t, a, models.ASRCohere, models.ASRAccurate, models.VADSilero)
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Transcribe.ASR, s.Features.Translate.ASR, s.Features.Rewrite.ASR = models.ASRCohere, models.ASRCohere, models.ASRCohere
		s.Features.Meeting.ASR = models.ASRAccurate
	})
	a.navigate(modelsPath)
	frames(tt)
	if has(tt, speechContextNote) {
		t.Errorf("Meetings on Qwen3-ASR still has the note: %q", tt.Texts())
	}
	// Automatic takes the pick for the meeting's language: Cohere does not
	// know Thai, Qwen3-ASR does, so Qwen3-ASR runs the meeting.
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Meeting.ASR = ""
		s.Features.Meeting.Lang = "th"
	})
	frames(tt)
	if has(tt, speechContextNote) {
		t.Errorf("Meetings on Automatic for Thai still has the note: %q", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Meeting.ASR = models.ASRCohere
		s.Features.Meeting.Lang = ""
	})
	frames(tt)
	if !has(tt, speechContextNote) {
		t.Errorf("Cohere in every feature and Meetings: no note: %q", tt.Texts())
	}
}

// TestSpeechContextNoteAutomaticDiarizer: with Cohere in every feature
// and Meetings, the installed MOSS-Transcribe-Diarize still reads the
// Speech context: the final pass of a full-mode Chinese or English
// meeting diarizes the system track with it (#4687 review). The final
// pass off, a non-diarizing mode or a language MOSS does not know keeps
// the note.
func TestSpeechContextNoteAutomaticDiarizer(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 4800)
	seedModels(t, a, models.ASRCohere, models.ASRMoss, models.VADSilero)
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Transcribe.ASR, s.Features.Translate.ASR, s.Features.Rewrite.ASR = models.ASRCohere, models.ASRCohere, models.ASRCohere
		s.Features.Meeting.ASR = models.ASRCohere
		s.Features.Meeting.Lang = "en"
	})
	a.navigate(modelsPath)
	frames(tt)
	if has(tt, speechContextNote) {
		t.Errorf("Cohere everywhere but the MTD diarizer runs: note shows: %q", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Meeting.NoFinalPass = true })
	frames(tt)
	if !has(tt, speechContextNote) {
		t.Errorf("final pass off: no note: %q", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Meeting.NoFinalPass = false
		s.Features.Meeting.Mode = "subtitles"
	})
	frames(tt)
	if !has(tt, speechContextNote) {
		t.Errorf("subtitles mode: no note: %q", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Meeting.Mode = ""
		s.Features.Meeting.Lang = "fr"
	})
	frames(tt)
	if !has(tt, speechContextNote) {
		t.Errorf("a meeting language MOSS does not know: no note: %q", tt.Texts())
	}
}

// TestSpeechContextNoteDeletedMeetingModel: Meetings' saved selection
// deleted, a meeting falls back to an installed model the way
// meetingTrackASR opens it, and the note follows that fallback, not the
// deleted selection (#4687 review): with Qwen3-ASR and Cohere installed
// and Meetings still on a deleted Whisper Large v3 Turbo, the meeting
// runs Qwen3-ASR, which takes the context, so the note hides. The
// selection installed again brings the note back.
func TestSpeechContextNoteDeletedMeetingModel(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 4800)
	seedModels(t, a, models.ASRCohere, models.ASRAccurate, models.VADSilero)
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Transcribe.ASR, s.Features.Translate.ASR, s.Features.Rewrite.ASR = models.ASRCohere, models.ASRCohere, models.ASRCohere
		s.Features.Meeting.ASR = models.ASRWhisperTurbo
	})
	if m, ok := a.meetingASRModel(); !ok || m.ID != models.ASRAccurate {
		t.Errorf("deleted meeting selection: the meeting's model is %q, %v; want the Qwen3-ASR fallback", m.ID, ok)
	}
	a.navigate(modelsPath)
	frames(tt)
	if has(tt, speechContextNote) {
		t.Errorf("the meeting falls back to Qwen3-ASR, which takes the context: note shows: %q", tt.Texts())
	}
	seedModels(t, a, models.ASRWhisperTurbo)
	frames(tt)
	if !has(tt, speechContextNote) {
		t.Errorf("the selection installed again: no note: %q", tt.Texts())
	}
}
