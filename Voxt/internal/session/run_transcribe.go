package session

import (
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func (r *run) warmLLM() {
	r.m.mu.Lock()
	client, ctx := r.svc.LLM, r.llmCtx
	r.m.mu.Unlock()
	prefiller, ok := client.(llm.Prefiller)
	if !ok {
		return
	}
	if capability, ok := client.(llm.PrefillCapability); ok && !capability.CanPrefill() {
		return
	}
	var input compose.Input
	var skill llm.Skill
	switch r.req.Kind {
	case store.KindTranscribe:
		if !r.transcribeCleanup() {
			return
		}
		input, skill = r.transcribeInput(""), llm.SkillTranscribeEnhance
	case store.KindTranslate:
		input, skill = r.writeInput("", ""), llm.SkillTranslateText
	default:
		return
	}
	output, err := compose.Build(input)
	if err != nil {
		return
	}
	request := llm.Request{Skill: skill, Messages: llmMessages(output)}
	start := r.now()
	if err := prefiller.Prefill(ctx, request); err != nil && r.ctx.Err() == nil {
		r.logf("info", "warming the language model up: %v", err)
	} else if err == nil {
		r.logf("debug", "warmed the language model up in %v", r.now().Sub(start).Round(time.Millisecond))
	}
}

// transcribeInput is Transcribe's request to compose for raw, its
// snippets' triggers marked for the model to keep (expandSnippets).
func (r *run) transcribeInput(raw string) compose.Input {
	feature := r.settings.Features.Transcribe
	text, marked := snippet.Mark(raw, r.snippets)
	return compose.Input{
		Task: compose.Transcribe, Cleanup: feature.Cleanup, Structure: feature.Structure, TaskPrompt: feature.Prompt,
		MainLanguage: MainLanguage(r.settings, r.m.cfg.Locale()), OtherLanguages: r.settings.General.OtherLanguages, Text: text, Snippets: len(marked) > 0,
		Glossary: r.glossary(), Dictionary: PromptDictionary(r.settings), Scene: r.composeScene(),
	}
}

func (r *run) glossary() []compose.Term {
	if !r.settings.Dictionary.Enabled || !r.settings.Dictionary.Glossary {
		return nil
	}
	var terms []compose.Term
	for _, term := range r.terms {
		if term.Enabled {
			terms = append(terms, compose.Term{Term: term.Term, Aliases: term.Aliases})
		}
	}
	return terms
}

func (r *run) typedOnly() bool {
	if !r.req.Box || r.m.cfg.BoxInput == nil {
		return false
	}
	typed, _, err := r.m.cfg.BoxInput(r.ctx)
	return err == nil && strings.TrimSpace(typed) != ""
}
