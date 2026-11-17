package session

import (
	"errors"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestLocalDictationFormatsWithoutLanguageModel(t *testing.T) {
	model := &fakeLLM{text: "unexpected language model result"}
	h := newHarness(t, &fakeStreamer{final: "第一，准备 Atlas。第二，发布。"}, model, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.LocalStructure = true
		s.Features.Transcribe.Enhance = true
		s.Dictionary.Enabled = false
		s.Dictionary.Replacements = "Atlas → 项目 A"
	})
	id := h.start()
	h.speak(id, 5)
	end := h.rec.wait(t)
	if end.Result != "1. 准备 项目 A。\n2. 发布。" || len(model.requests()) != 0 {
		t.Fatalf("result %q, language model calls %d", end.Result, len(model.requests()))
	}
}

func TestReplacementsMaskLanguageModelInputAndFallback(t *testing.T) {
	model := &fakeLLM{err: errors.New("offline")}
	h := newHarness(t, &fakeStreamer{final: "Contact Project Atlas."}, model, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
		s.Features.Transcribe.Prompt = "Check Project Atlas."
		s.Dictionary.Notes = "Project Atlas is a private project."
		s.Dictionary.Replacements = "Project Atlas → Project A\nProject A → Public"
	})
	id := h.start()
	h.speak(id, 5)
	end := h.rec.wait(t)
	if end.Result != "Contact Project A." {
		t.Fatalf("result %q", end.Result)
	}
	requests := model.requests()
	if len(requests) != 1 {
		t.Fatalf("model requests = %d", len(requests))
	}
	var messages strings.Builder
	for _, message := range requests[0].Messages {
		messages.WriteString(message.Content)
	}
	if prompt := messages.String(); strings.Contains(prompt, "Project Atlas") || !strings.Contains(prompt, "Contact Project A.") || !strings.Contains(prompt, "Check Project A.") {
		t.Fatalf("unexpected request %q", prompt)
	}
}

func TestReplacementMasksCopiesOfSharedComposeFields(t *testing.T) {
	r := &run{}
	r.settings.Dictionary.Replacements = "Atlas → Project A"
	in := compose.Input{
		Text: "Atlas", Instruction: "Summarise Atlas",
		Scene:    &compose.Scene{AppName: "Atlas", Prompt: "Atlas"},
		Glossary: []compose.Term{{Term: "Atlas", Aliases: []string{"Atlas"}}},
		Previous: []compose.Turn{{Instruction: "Atlas", Selection: "Atlas", Answer: compose.Answer{Title: "Atlas", Content: "Atlas"}}},
	}
	masked := r.replaceInput(in)
	if masked.Text != "Project A" || masked.Instruction != "Summarise Project A" ||
		masked.Scene.Prompt != "Project A" || masked.Glossary[0].Aliases[0] != "Project A" ||
		masked.Previous[0].Answer.Content != "Project A" {
		t.Fatalf("replacement missed a prompt field: %+v", masked)
	}
	if in.Scene.Prompt != "Atlas" || in.Glossary[0].Aliases[0] != "Atlas" || in.Previous[0].Answer.Content != "Atlas" {
		t.Fatalf("replacement changed original fields: %+v", in)
	}
}
