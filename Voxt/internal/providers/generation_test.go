package providers

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestCapabilities: each service shows the fields Voxt 1.x showed for it.
func TestCapabilities(t *testing.T) {
	for _, c := range []struct {
		id, model string
		modes     []string
		efforts   []string
		check     func(GenCaps) bool
	}{
		{"openai", "", []string{"", "effort"}, effortsOpenAI, func(c GenCaps) bool { return c.Verbosity && c.Logprobs && c.ResponseFormat && !c.TopK }},
		{"custom", "", []string{"", "effort"}, effortsCustom, func(c GenCaps) bool { return c.Verbosity && slices.Contains(c.Efforts, "max") }},
		{"anthropic", "", []string{"", "budget"}, nil, func(c GenCaps) bool { return c.TopK && !c.ResponseFormat }},
		{"ollama", "", []string{"", "off", "on", "effort"}, []string{"low", "medium", "high"}, func(c GenCaps) bool {
			return c.KeepAlive && c.Options && c.JSONSchema && c.MinP && c.Seed && c.Penalties
		}},
		{"omlx", "", nil, nil, func(c GenCaps) bool { return c.IncludeUsage && c.JSONSchema && c.TopK }},
		{"lmstudio", "", nil, nil, func(c GenCaps) bool { return !c.IncludeUsage && !c.JSONSchema && c.TopK }},
		{"deepseek", "", []string{"", "off", "on", "effort"}, []string{"none", "low", "high", "max"}, nil},
		{"dashscope", "", []string{"", "off", "on", "effort", "budget"}, effortsOpenAI, func(c GenCaps) bool { return c.Penalties }},
		{"moonshot", "", []string{"", "effort", "budget"}, effortsOpenAI, nil},
		{"stepfun", "step-2", []string{"off"}, nil, func(c GenCaps) bool { return c.FrequencyOnly && !c.Penalties }},
		{"stepfun", "step-3.5-flash-2603", []string{"off", "effort"}, []string{"low", "high"}, nil},
		{"minimax", "", nil, nil, func(c GenCaps) bool { return c.Penalties && c.MaxOutputTokens && c.ExtraBody }},
	} {
		got := Capabilities(c.id, c.model)
		if !slices.Equal(got.ThinkModes, c.modes) || !slices.Equal(got.Efforts, c.efforts) || c.check != nil && !c.check(got) {
			t.Errorf("%s %s: %+v", c.id, c.model, got)
		}
	}
}

func ptr[T any](v T) *T { return &v }

// TestGenerationSettings: a setup's settings reach its request (through
// Config), JSON fields parsed; one that is not a JSON object is refused
// on save; unset settings stay out of settings.json.
func TestGenerationSettings(t *testing.T) {
	g := settings.Generation{MaxOutputTokens: ptr(131072), Temperature: ptr(0.0), TopP: ptr(0.9),
		Thinking: llm.ThinkEffort, ThinkingEffort: "none", ExtraBodyJSON: `{"user":"me"}`, OptionsJSON: " "}
	p, err := Config(settings.Provider{ID: llm.Custom, Endpoint: "https://example.com/v1", Model: "m", Generation: g}).Resolve()
	if err != nil {
		t.Fatal(err)
	}
	pg := p.Generation
	if *pg.MaxOutputTokens != 131072 || *pg.Temperature != 0 || *pg.TopP != 0.9 || pg.Thinking.Mode != llm.ThinkEffort ||
		pg.Thinking.Effort != "none" || pg.ExtraBody["user"] != "me" || pg.Options != nil {
		t.Errorf("generation %+v", pg)
	}
	if err := CheckGeneration(g); err != nil {
		t.Error(err)
	}
	for _, bad := range []settings.Generation{{ExtraBodyJSON: "[1]"}, {OptionsJSON: "{"}, {JSONSchema: "null"}} {
		if CheckGeneration(bad) == nil {
			t.Errorf("accepted %+v", bad)
		}
	}
	raw, _ := json.Marshal(settings.Provider{ID: "openai"})
	if strings.Contains(string(raw), "generation") {
		t.Errorf("unset settings saved: %s", raw)
	}
}

// TestAPIFormat: only OpenAI and custom services offer the Responses API,
// a setup passes its choice on, and one saved before the option existed
// stays on chat completions.
func TestAPIFormat(t *testing.T) {
	for _, pr := range llm.Presets() {
		if got, want := Capabilities(pr.ID, "").APIFormat, pr.ID == "openai" || pr.ID == llm.Custom; got != want {
			t.Errorf("%s: APIFormat %v", pr.ID, got)
		}
	}
	if api := Config(settings.Provider{ID: "openai", APIFormat: llm.APIResponses}).API; api != llm.APIResponses {
		t.Errorf("API %q", api)
	}
	var old settings.Provider
	if err := json.Unmarshal([]byte(`{"id":"openai","model":"gpt-4.1-mini"}`), &old); err != nil {
		t.Fatal(err)
	}
	if api := Config(old).API; api != "" {
		t.Errorf("an old setup: API %q", api)
	}
	raw, _ := json.Marshal(old)
	if strings.Contains(string(raw), "apiFormat") {
		t.Errorf("the default is written: %s", raw)
	}
}
