package main

// What the previous Voxt sent to OpenAI's Responses API, for its OpenAI
// setups: the import keeps what it sent (legacy_settings.go), as this app
// speaks chat completions to OpenAI.

import (
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// legacyModelIs reports whether model is id or a dated snapshot of it, the
// previous Voxt's way of matching OpenAI's models.
func legacyModelIs(model, id string) bool {
	m := strings.ToLower(strings.TrimSpace(model))
	return m == id || strings.HasPrefix(m, id+"-")
}

// legacyOpenAIEfforts are the reasoning efforts the previous Voxt sent for
// an OpenAI model (its OpenAIReasoningEffort.supportedCases).
func legacyOpenAIEfforts(model string) []string {
	m := strings.ToLower(strings.TrimSpace(model))
	switch {
	case legacyModelIs(m, "gpt-5.2-pro"):
		return []string{"medium", "high", "xhigh"}
	case legacyModelIs(m, "gpt-5-pro"):
		return []string{"high"}
	case legacyModelIs(m, "gpt-5.2-codex"):
		return []string{"low", "medium", "high", "xhigh"}
	case legacyModelIs(m, "gpt-5.3-codex-spark"):
		return []string{"none", "low", "medium", "high"}
	case legacyModelIs(m, "gpt-5.1-codex-max"):
		return []string{"none", "medium", "high", "xhigh"}
	case legacyModelIs(m, "gpt-5.2"):
		return []string{"none", "low", "medium", "high", "xhigh"}
	case legacyModelIs(m, "gpt-5.1"):
		return []string{"none", "low", "medium", "high"}
	case strings.HasPrefix(m, "gpt-5"):
		return []string{"minimal", "low", "medium", "high"}
	case strings.HasPrefix(m, "o1"), strings.HasPrefix(m, "o3"), strings.HasPrefix(m, "o4"):
		return []string{"low", "medium", "high"}
	}
	return nil
}

// legacyOpenAIGeneration is g as far as the previous Voxt sent it to OpenAI:
// its Responses request carried the output limit, the response format, the
// effort and verbosity for models that take them, and Extra Body.
func legacyOpenAIGeneration(g settings.Generation, model string) settings.Generation {
	out := settings.Generation{MaxOutputTokens: g.MaxOutputTokens, ResponseFormat: g.ResponseFormat, JSONSchema: g.JSONSchema, ExtraBodyJSON: g.ExtraBodyJSON}
	if g.Thinking == llm.ThinkEffort && slices.Contains(legacyOpenAIEfforts(model), g.ThinkingEffort) {
		out.Thinking, out.ThinkingEffort = g.Thinking, g.ThinkingEffort
	}
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(model)), "gpt-5") {
		out.Verbosity = g.Verbosity
	}
	return out
}
