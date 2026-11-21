package providers

import (
	"reflect"
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestLegacyOpenAIHelpers pins what the import of the previous Voxt's OpenAI
// setups returned before these helpers left package main.
func TestLegacyOpenAIHelpers(t *testing.T) {
	for _, c := range []struct {
		model, id string
		want      bool
	}{
		{"gpt-5.2", "gpt-5.2", true},
		{" GPT-5.2-2025-12-11 ", "gpt-5.2", true},
		{"gpt-5.2-pro", "gpt-5.2", true},
		{"gpt-5.20", "gpt-5.2", false},
		{"gpt-5", "gpt-5.2", false},
		{"", "gpt-5", false},
	} {
		if got := LegacyModelIs(c.model, c.id); got != c.want {
			t.Errorf("LegacyModelIs(%q, %q) = %v, want %v", c.model, c.id, got, c.want)
		}
	}

	for _, c := range []struct {
		model string
		want  []string
	}{
		{"gpt-5.2-pro", []string{"medium", "high", "xhigh"}},
		{"gpt-5-pro-2025-10-06", []string{"high"}},
		{"gpt-5.2-codex", []string{"low", "medium", "high", "xhigh"}},
		{"gpt-5.3-codex-spark", []string{"none", "low", "medium", "high"}},
		{"gpt-5.1-codex-max", []string{"none", "medium", "high", "xhigh"}},
		{" GPT-5.2 ", []string{"none", "low", "medium", "high", "xhigh"}},
		{"gpt-5.1-mini", []string{"none", "low", "medium", "high"}},
		{"gpt-5-mini", []string{"minimal", "low", "medium", "high"}},
		{"o1", []string{"low", "medium", "high"}},
		{"o3-mini", []string{"low", "medium", "high"}},
		{"o4-mini", []string{"low", "medium", "high"}},
		{"gpt-4.1", nil},
		{"", nil},
	} {
		if got := legacyOpenAIEfforts(c.model); !slices.Equal(got, c.want) {
			t.Errorf("legacyOpenAIEfforts(%q) = %q, want %q", c.model, got, c.want)
		}
	}

	temp, topP, maxOut := 0.4, 0.9, 512
	full := settings.Generation{
		Temperature: &temp, TopP: &topP, MaxOutputTokens: &maxOut, Stop: []string{"\n"}, ResponseFormat: "json_schema", JSONSchema: `{"type":"object"}`,
		Thinking: llm.ThinkEffort, ThinkingEffort: "xhigh", ThinkingBudget: 2048, Verbosity: "low", ExtraBodyJSON: `{"a":1}`,
	}
	kept := settings.Generation{MaxOutputTokens: &maxOut, ResponseFormat: "json_schema", JSONSchema: `{"type":"object"}`, ExtraBodyJSON: `{"a":1}`}
	withEffort := kept
	withEffort.Thinking, withEffort.ThinkingEffort = llm.ThinkEffort, "xhigh"
	withVerbosity := kept
	withVerbosity.Verbosity = "low"
	withBoth := withEffort
	withBoth.Verbosity = "low"
	budget := full
	budget.Thinking = "budget"
	for _, c := range []struct {
		name  string
		g     settings.Generation
		model string
		want  settings.Generation
	}{
		{"gpt-5.2 keeps a supported effort and verbosity", full, "gpt-5.2", withBoth},
		{"gpt-5-mini drops an unsupported effort", full, " GPT-5-mini ", withVerbosity},
		{"o3 keeps no xhigh and no verbosity", full, "o3", kept},
		{"gpt-5.2-pro keeps xhigh, verbosity", full, "gpt-5.2-pro-2025-12-11", withBoth},
		{"budget thinking is dropped", budget, "gpt-5.2", withVerbosity},
		{"gpt-4.1 keeps only the request fields", full, "gpt-4.1", kept},
		{"empty stays empty", settings.Generation{}, "gpt-5.2", settings.Generation{}},
	} {
		if got := LegacyOpenAIGeneration(c.g, c.model); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: LegacyOpenAIGeneration = %+v, want %+v", c.name, got, c.want)
		}
	}

	for _, c := range []struct {
		format, endpoint string
		want             bool
	}{
		{"responses", "https://example.com/v1", true},
		{"chatCompletions", "https://api.openai.com/v1", false},
		{"", "", true},
		{"automatic", "https://api.openai.com/v1", true},
		{"unknown", " https://API.OpenAI.com:443/v1 ", true},
		{"", "https://example.com/v1", false},
		{"", "/v1", true},
		{"", "http://[::1", false},
	} {
		if got := LegacyResponsesAPI(c.format, c.endpoint); got != c.want {
			t.Errorf("LegacyResponsesAPI(%q, %q) = %v, want %v", c.format, c.endpoint, got, c.want)
		}
	}
}
