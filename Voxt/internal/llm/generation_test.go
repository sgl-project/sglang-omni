package llm

import (
	"encoding/json"
	"testing"
)

func ptr[T any](v T) *T { return &v }

// bodyFor is the streamed request body for a setup of preset id with g.
func bodyFor(t *testing.T, id, model string, g Generation, req Request) map[string]any {
	t.Helper()
	p, err := ProviderConfig{ID: id, BaseURL: "https://example.com/v1", Model: model, Generation: g}.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	b := NewOpenAI(p, "k", nil).body(req, model, true, true, true)
	// As the server reads it: numbers and maps as JSON has them.
	raw, _ := json.Marshal(b)
	var out map[string]any
	_ = json.Unmarshal(raw, &out)
	return out
}

func js(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// TestGenerationFields: the settings of Voxt 1.x's dialog go into the
// request as it sent them, over Voxt's own values; Extra Body JSON goes
// last and wins.
func TestGenerationFields(t *testing.T) {
	// The setup the team uses: a custom server, thinking effort none.
	g := Generation{MaxOutputTokens: ptr(131072), Temperature: ptr(0.0), TopP: ptr(0.9),
		Thinking: Thinking{Mode: ThinkEffort, Effort: "none"}}
	b := bodyFor(t, Custom, "m", g, Request{Messages: msgs("hi"), Temperature: ptr(0.3)})
	if b["max_tokens"] != 131072.0 || b["temperature"] != 0.0 || b["top_p"] != 0.9 || b["reasoning_effort"] != "none" {
		t.Errorf("custom: %s", js(b))
	}

	g = Generation{TopK: ptr(40), MinP: ptr(0.05), Seed: ptr(int64(7)), Stop: []string{"###"}, PresencePenalty: ptr(0.1),
		FrequencyPenalty: ptr(0.2), RepetitionPenalty: ptr(1.1), Logprobs: true, TopLogprobs: ptr(5), ResponseFormat: ResponseJSON,
		Verbosity: "low", ExtraBody: map[string]any{"temperature": 1.5, "user": "me"}}
	b = bodyFor(t, "openai", "gpt-5-mini", g, Request{Messages: msgs("hi"), MaxTokens: 100})
	want := map[string]any{"top_k": 40.0, "min_p": 0.05, "seed": 7.0, "presence_penalty": 0.1, "frequency_penalty": 0.2,
		"repetition_penalty": 1.1, "logprobs": true, "top_logprobs": 5.0, "verbosity": "low", "temperature": 1.5, "user": "me",
		"max_completion_tokens": 100.0}
	for k, v := range want {
		if b[k] != v {
			t.Errorf("openai %s = %v, want %v", k, b[k], v)
		}
	}
	if js(b["stop"]) != `["###"]` || js(b["response_format"]) != `{"type":"json_object"}` {
		t.Errorf("openai stop %v, format %v", b["stop"], b["response_format"])
	}

	// A feature's JSON format wins over the setting; StepFun gets no
	// presence penalty; oMLX can leave usage out of the stream; Ollama's
	// options are merged.
	b = bodyFor(t, "openai", "gpt-5-mini", Generation{ResponseFormat: ResponseJSON}, Request{Messages: msgs("hi"), Format: FormatJSON, Schema: &Schema{Name: "x"}})
	if f, _ := b["response_format"].(map[string]any); f["type"] != "json_schema" {
		t.Errorf("feature format: %v", b["response_format"])
	}
	if b = bodyFor(t, "stepfun", "step-2", Generation{PresencePenalty: ptr(0.1), FrequencyPenalty: ptr(0.2)}, Request{Messages: msgs("hi")}); b["presence_penalty"] != nil || b["frequency_penalty"] != 0.2 {
		t.Errorf("stepfun: %s", js(b))
	}
	if b = bodyFor(t, "omlx", "m", Generation{IncludeUsage: ptr(false)}, Request{Messages: msgs("hi")}); b["stream_options"] != nil {
		t.Errorf("omlx usage: %v", b["stream_options"])
	}
	if b = bodyFor(t, "ollama", "m", Generation{KeepAlive: "5m", ExtraBody: map[string]any{"options": map[string]any{"num_ctx": 8192}, "keep_alive": nil}}, Request{Messages: msgs("hi")}); b["keep_alive"] != nil || js(b["options"]) != `{"num_ctx":8192}` {
		t.Errorf("ollama: %s", js(b))
	}
}

// TestGenerationThinking: each service's thinking fields, as Voxt 1.x
// wrote them; Default keeps Voxt's own (the preset's ExtraBody).
func TestGenerationThinking(t *testing.T) {
	for _, c := range []struct {
		id, model string
		t         Thinking
		want      map[string]string // field → its JSON; "" for absent
	}{
		{"moonshot", "kimi-k2.5", Thinking{}, map[string]string{"thinking": `{"type":"disabled"}`}},
		{"moonshot", "kimi-k2.5", Thinking{Mode: ThinkBudget, Budget: 2048}, map[string]string{"thinking": `{"budget_tokens":2048,"type":"enabled"}`}},
		{"moonshot", "kimi-k2.5", Thinking{Mode: ThinkEffort, Effort: "low"}, map[string]string{"thinking": "", "reasoning_effort": `"low"`}},
		{"deepseek", "deepseek-chat", Thinking{Mode: ThinkOn}, map[string]string{"thinking": `{"type":"enabled"}`}},
		{"deepseek", "deepseek-chat", Thinking{Mode: ThinkEffort, Effort: "max"}, map[string]string{"reasoning_effort": `"max"`}},
		{"dashscope", "qwen-plus", Thinking{Mode: ThinkBudget, Budget: 500}, map[string]string{"enable_thinking": "true", "thinking_budget": "500"}},
		{"dashscope", "qwen-plus", Thinking{Mode: ThinkOff}, map[string]string{"enable_thinking": "false", "thinking": `{"type":"disabled"}`}},
		{"zhipu", "glm-4.6", Thinking{Mode: ThinkOn}, map[string]string{"thinking": `{"type":"enabled"}`}},
		{"mimo", "mimo-v2-flash", Thinking{Mode: ThinkOn}, map[string]string{"thinking": `{"type":"enabled"}`}},
		{"openrouter", "x/y", Thinking{Mode: ThinkEffort, Effort: "high"}, map[string]string{"reasoning": `{"effort":"high","exclude":true}`}},
		{"openrouter", "x/y", Thinking{Mode: ThinkOff, ExposeReasoning: true}, map[string]string{"reasoning": `{"enabled":false,"exclude":false}`}},
		{"xai", "grok-4", Thinking{Mode: ThinkEffort, Effort: "low"}, map[string]string{"reasoning_effort": `"low"`}},
		{"stepfun", "step-3.5-flash-2603", Thinking{Mode: ThinkEffort, Effort: "high"}, map[string]string{"reasoning_effort": `"high"`}},
		{"stepfun", "step-2", Thinking{Mode: ThinkEffort, Effort: "high"}, map[string]string{"reasoning_effort": ""}},
		{"anthropic", "claude-sonnet-4-5", Thinking{Mode: ThinkBudget, Budget: 1024}, map[string]string{"thinking": `{"budget_tokens":1024,"type":"enabled"}`}},
		{"gemini", "gemini-2.5-flash", Thinking{}, map[string]string{"reasoning_effort": `"none"`}},
		{"gemini", "gemini-2.5-flash", Thinking{Mode: ThinkBudget, Budget: 800}, map[string]string{"reasoning_effort": "", "extra_body": `{"google":{"thinking_config":{"thinking_budget":800}}}`}},
		{"ollama", "qwen3", Thinking{Mode: ThinkOff}, map[string]string{"reasoning_effort": `"none"`}},
	} {
		b := bodyFor(t, c.id, c.model, Generation{Thinking: c.t}, Request{Messages: msgs("hi")})
		for k, w := range c.want {
			got := ""
			if v, ok := b[k]; ok {
				got = js(v)
			}
			if got != w {
				t.Errorf("%s %+v: %s = %s, want %s", c.id, c.t, k, got, w)
			}
		}
	}
}

// TestAnthropicBudgetRaisesMaxTokens: Voxt 1.x kept max_tokens above the
// budget; below it the request would be refused.
func TestAnthropicBudgetRaisesMaxTokens(t *testing.T) {
	budget := Thinking{Mode: ThinkBudget, Budget: 8192}
	for _, c := range []struct {
		name string
		g    Generation
		req  Request
		want any
	}{
		{"below", Generation{MaxOutputTokens: ptr(4096), Thinking: budget}, Request{}, 8193.0},
		{"equal", Generation{MaxOutputTokens: ptr(8192), Thinking: budget}, Request{}, 8193.0},
		{"above", Generation{MaxOutputTokens: ptr(16384), Thinking: budget}, Request{}, 16384.0},
		{"request's own limit", Generation{Thinking: budget}, Request{MaxTokens: 512}, 8193.0},
		{"none sent", Generation{Thinking: budget}, Request{}, nil},
		{"another mode", Generation{MaxOutputTokens: ptr(4096), Thinking: Thinking{Mode: ThinkOff}}, Request{}, 4096.0},
		{"no budget", Generation{MaxOutputTokens: ptr(4096), Thinking: Thinking{Mode: ThinkBudget}}, Request{}, 4096.0},
	} {
		c.req.Messages = msgs("hi")
		if got := bodyFor(t, "anthropic", "m", c.g, c.req)["max_tokens"]; got != c.want {
			t.Errorf("%s: max_tokens %v, want %v", c.name, got, c.want)
		}
	}
	// Another service with a budget is left alone.
	g := Generation{MaxOutputTokens: ptr(4096), Thinking: budget}
	if got := bodyFor(t, "zhipu", "m", g, Request{Messages: msgs("hi")})["max_tokens"]; got != 4096.0 {
		t.Errorf("zhipu max_tokens %v", got)
	}
}

// TestRequestThinkingOverrides: a request's Thinking (the answer card's
// menu) replaces the setup's for that request only, and a Default one
// brings the preset's own fields back.
func TestRequestThinkingOverrides(t *testing.T) {
	off := Generation{Thinking: Thinking{Mode: ThinkOff}}
	b := bodyFor(t, "deepseek", "deepseek-chat", off, Request{Messages: msgs("hi"), Thinking: &Thinking{Mode: ThinkEffort, Effort: "high"}})
	if b["reasoning_effort"] != "high" || b["thinking"] != nil {
		t.Errorf("effort over off: %s", js(b))
	}
	b = bodyFor(t, "deepseek", "deepseek-chat", Generation{Thinking: Thinking{Mode: ThinkOn}}, Request{Messages: msgs("hi"), Thinking: &Thinking{}})
	if js(b["thinking"]) != `{"type":"disabled"}` {
		t.Errorf("default over on: the preset's extra body, got %s", js(b))
	}
	b = bodyFor(t, "anthropic", "m", Generation{MaxOutputTokens: ptr(512)}, Request{Messages: msgs("hi"), Thinking: &Thinking{Mode: ThinkBudget, Budget: 4096}})
	if js(b["thinking"]) != `{"budget_tokens":4096,"type":"enabled"}` || b["max_tokens"] != 4097.0 {
		t.Errorf("budget: %s", js(b))
	}
	if b = bodyFor(t, "deepseek", "deepseek-chat", off, Request{Messages: msgs("hi")}); js(b["thinking"]) != `{"type":"disabled"}` || b["reasoning_effort"] != nil {
		t.Errorf("no override: %s", js(b))
	}
}

// TestThinkOffUnder (#3951): with ThinkOffUnder set, a short input goes
// out with Voxt's default thinking (off), a long one at the service's
// level, and the one-input override wins over both.
func TestThinkOffUnder(t *testing.T) {
	g := Generation{Thinking: Thinking{Mode: ThinkOn}, ThinkOffUnder: 200}
	on := &Thinking{Mode: ThinkOn}
	for _, c := range []struct {
		name string
		req  Request
		want string
	}{
		{"short", Request{InputChars: 20}, `{"type":"disabled"}`},
		{"long", Request{InputChars: 500}, `{"type":"enabled"}`},
		{"at the limit", Request{InputChars: 200}, `{"type":"enabled"}`},
		{"unknown length", Request{}, `{"type":"enabled"}`},
		{"short, a level chosen for it", Request{InputChars: 20, Thinking: on}, `{"type":"enabled"}`},
	} {
		c.req.Messages = msgs("hi")
		if got := js(bodyFor(t, "zhipu", "glm-4.6", g, c.req)["thinking"]); got != c.want {
			t.Errorf("%s: thinking = %s, want %s", c.name, got, c.want)
		}
	}
	if got := js(bodyFor(t, "zhipu", "glm-4.6", Generation{Thinking: Thinking{Mode: ThinkOn}}, Request{Messages: msgs("hi"), InputChars: 20})["thinking"]); got != `{"type":"enabled"}` {
		t.Errorf("option off, short input: thinking = %s", got)
	}
}
