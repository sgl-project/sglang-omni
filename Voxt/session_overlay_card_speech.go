package main

// A card-enabled Transcribe or Translate session shows in the answer card
// from its first snapshot (#4167).

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// buildSessionSpeechCardViewState is a card-enabled Transcribe or Translate
// session before it ends: the waveform while it records, the selected text
// and what was recognised, then the answer as the model writes it.
func buildSessionSpeechCardViewState(translator *i18n.Translator, input sessionCardProjectionInput, snapshot session.Snapshot) sessionCardViewState {
	state := sessionCardViewState{
		ID: snapshot.ID, Kind: string(snapshot.Kind), Title: sessionCardTitleFor(translator, snapshot.Kind, snapshot.Title),
		TargetName: snapshot.Target.Name, Language: snapshot.TargetLanguage,
		LanguageLabel: compose.LanguageName(snapshot.TargetLanguage, translator.Locale()),
		SizeStep:      input.SizeStep, Busy: true, Speech: true,
		Recording: snapshot.State == session.Recording, Writing: snapshot.State == session.Composing,
		CanStopAnswer: snapshot.CanStop(), CanUseTranscript: snapshot.CanSkip, Heard: !snapshot.HeardAt.IsZero(),
		WorkingText: uitext.WorkingTextFor(translator, snapshot),
	}
	state.CanStop = state.Recording
	if state.Recording {
		state.Message = input.Note
	}
	copySessionLevels(&state.Levels, snapshot.Levels)
	if state.Writing && strings.TrimSpace(snapshot.Draft) != "" {
		state.Speech, state.WorkingText = false, "" // the answer streams in as a writing card's does
		state.Live[0], state.LiveCount = sessionCardWritingBlock(input.ShownDraft, snapshot.Kind), 1
		return state
	}
	if selection := strings.TrimSpace(snapshot.Selection); selection != "" {
		state.Blocks = append(state.Blocks, sessionCardBlock{Key: "selection", Kind: sessionCardSource, Text: selection, Selectable: true})
	}
	if text := strings.TrimSpace(cmpOr(snapshot.Raw, snapshot.Partial)); text != "" || len(state.Blocks) == 0 {
		state.Blocks = append(state.Blocks, sessionCardBlock{Key: "speech", Kind: sessionCardBody, Text: text, Selectable: text != ""})
	}
	return state
}

// liveCard says whether a card-enabled Transcribe or Translate session
// shows in the card from its first snapshot; Rewrite keeps its own flow.
func (a *app) liveCard() bool {
	snapshot := a.sess
	return !a.answer.open && snapshot.State.Live() && snapshot.Card && !snapshot.Try && !snapshot.Note &&
		(snapshot.Kind == store.KindTranscribe || snapshot.Kind == store.KindTranslate)
}
