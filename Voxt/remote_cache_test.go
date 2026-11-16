package main

import (
	"reflect"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestRemotePageViewCache(t *testing.T) {
	a, _, _ := newTestApp(t)
	first := a.remotePageModel()
	second := a.remotePageModel()
	if !reflect.DeepEqual(first, second) {
		t.Fatal("identical inputs produced different remote page state")
	}
	if &first.Wait.Options[0] != &second.Wait.Options[0] || &first.Cards.Groups[0] != &second.Cards.Groups[0] {
		t.Fatal("stable inputs rebuilt the remote page DTOs")
	}
	speechFirst, speechSecond := a.remoteCardsModel(local.ASR), a.remoteCardsModel(local.ASR)
	if &speechFirst.Groups[0] != &speechSecond.Groups[0] {
		t.Fatal("stable inputs rebuilt the speech card DTOs")
	}

	a.prefs.Models.LLMWait = settings.Wait5Min
	if got := a.remotePageModel().Wait.Value; got != settings.Wait5Min {
		t.Fatalf("wait cache was not invalidated: %q", got)
	}
	a.ob.awayTo = remotePath
	if got := a.remotePageModel().BackBar; got == nil {
		t.Fatal("onboarding cache was not invalidated")
	}

	a.updateSettings(func(s *settings.Settings) {
		s.Providers.LLM = []settings.Provider{{ID: llm.Custom, Endpoint: "http://localhost/v1", Model: "fixture-model"}}
	})
	a.keyMasks["llm.custom"] = "masked"
	cards := a.remotePageModel().Cards
	if len(cards.Groups) == 0 || cards.Groups[0].Title != "Set up" || !cards.Groups[0].Cards[0].Configured {
		t.Fatalf("provider cache was not invalidated: %+v", cards.Groups)
	}
	a.keySpaced["llm.custom"] = true
	if got := a.remotePageModel().Cards.Groups[0].Cards[0].KeyText; got != "Key has a space" {
		t.Fatalf("key cache was not invalidated: %q", got)
	}

	a.openProviderDialog(remoteKind("llm"), llm.Custom)
	if got := a.remotePageModel().Dialog; got == nil {
		t.Fatal("open dialog reused the closed-page cache")
	}
	a.provDlg = providerDialog{}
	if got := a.remotePageModel().Dialog; got != nil {
		t.Fatal("closed dialog did not restore the page cache")
	}
}

func TestLLMDebugViewCache(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{"llm." + llm.Custom: "sk-mock-123"}})
	a, _, _ := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) {
		s.Providers.LLM = []settings.Provider{{ID: llm.Custom, Endpoint: "http://localhost/v1", Model: "fixture-model"}}
		s.Features.Transcribe.Prompt = "Transcribe saved"
		s.Features.Rewrite.Prompt = "Rewrite saved"
	})
	a.readUnreadKeys()
	first := a.llmDebugModel()
	second := a.llmDebugModel()
	if !reflect.DeepEqual(first, second) || &first.PresetStates[0] != &second.PresetStates[0] {
		t.Fatal("stable inputs rebuilt the debug DTOs")
	}

	a.llmDbg.input = "new input"
	if got := a.llmDebugModel(); !got.CanRun {
		t.Fatal("debug input cache was not invalidated")
	}
	a.llmDbg.prompt = "Edited"
	if got := a.llmDebugModel(); !got.Edited {
		t.Fatal("prompt edit cache was not invalidated")
	}
	a.llmDbg.preset = "Rewrite"
	if got := a.llmDebugModel(); !got.Instruction || got.SavedPrompt != "Rewrite saved" {
		t.Fatalf("preset cache was not invalidated: %+v", got)
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Rewrite.Prompt = "Rewrite changed" })
	if got := a.llmDebugModel(); got.SavedPrompt != "Rewrite changed" {
		t.Fatalf("saved-prompt cache was not invalidated: %q", got.SavedPrompt)
	}
	a.updateSettings(func(s *settings.Settings) { s.Providers.LLM = nil })
	if got := a.llmDebugModel(); got.Ready {
		t.Fatal("provider removal left a stale debug model")
	}
}
