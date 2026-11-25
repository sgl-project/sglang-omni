package session

import (
	"context"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func (r *run) warmLLM() {
	if r.req.Kind == store.KindTranscribe && !r.transcribeCleanup() {
		return
	}
	r.m.mu.Lock()
	client, llmContext, remote := r.svc.LLM, r.llmCtx, r.svc.LLMInfo.Remote
	r.m.mu.Unlock()
	if client == nil || llmContext == nil {
		return
	}
	if connector, ok := client.(llm.Connector); ok && remote {
		go func() {
			start := r.now()
			if err := connector.Connect(llmContext); err != nil {
				if llmContext.Err() == nil {
					r.logf("info", "warming the language model up: %v", err)
				}
				return
			}
			r.logf("debug", "connected to the language model in %v", r.now().Sub(start).Round(time.Millisecond))
		}()
	}
	prefiller, prefillContext, request, ok := r.prepareWarmup()
	if !ok {
		return
	}
	go func() {
		start := r.now()
		if err := prefiller.Prefill(prefillContext, request); err != nil {
			if prefillContext.Err() == nil {
				r.logf("info", "warming the language model up: %v", err)
			}
			return
		}
		r.logf("debug", "warmed the language model up in %v", r.now().Sub(start).Round(time.Millisecond))
	}()
}

func (r *run) prepareWarmup() (llm.Prefiller, context.Context, llm.Request, bool) {
	r.m.mu.Lock()
	client, ctx := r.svc.LLM, r.llmCtx
	r.m.mu.Unlock()
	prefiller, ok := client.(llm.Prefiller)
	if !ok || ctx == nil {
		return nil, nil, llm.Request{}, false
	}
	if capability, ok := client.(llm.PrefillCapability); ok && !capability.CanPrefill() {
		return nil, nil, llm.Request{}, false
	}
	var input compose.Input
	var skill llm.Skill
	switch r.req.Kind {
	case store.KindTranscribe:
		if !r.transcribeCleanup() {
			return nil, nil, llm.Request{}, false
		}
		input, skill = r.transcribeInput(""), llm.SkillTranscribeEnhance
	case store.KindTranslate:
		input, skill = r.writeInput("", ""), llm.SkillTranslateText
	default:
		return nil, nil, llm.Request{}, false
	}
	output, err := compose.Build(input)
	if err != nil {
		return nil, nil, llm.Request{}, false
	}
	request := llm.Request{Skill: skill, Messages: llmMessages(output)}
	return prefiller, ctx, request, true
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
