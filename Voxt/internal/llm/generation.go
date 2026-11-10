package llm

import "maps"

// Generation is a remote model's request settings, field for field as
// Voxt 1.x's service dialog had them (Generation Settings, Advanced,
// Expert, and a few services' own options). They are written over Voxt's
// own request (applyGeneration): a nil, "" or false field leaves Voxt's
// value, or the service's default.
type Generation struct {
	MaxOutputTokens   *int
	Temperature       *float64
	TopP              *float64
	TopK              *int
	MinP              *float64
	Seed              *int64
	Stop              []string
	PresencePenalty   *float64
	FrequencyPenalty  *float64
	RepetitionPenalty *float64
	Logprobs          bool
	TopLogprobs       *int
	// ResponseFormat is "" (plain text), ResponseJSON or ResponseJSONSchema
	// (JSONSchema: Ollama's and oMLX's).
	ResponseFormat string
	JSONSchema     map[string]any
	Thinking       Thinking
	Verbosity      string // OpenAI's text verbosity: "low", "medium", "high"
	KeepAlive      string // Ollama's keep_alive ("5m")
	IncludeUsage   *bool  // oMLX's stream_options.include_usage; nil: Voxt's (on)
	// ExtraBody is applied to the body last as a JSON merge patch (RFC
	// 7386): its keys win, an object merges into the body's, null removes.
	ExtraBody map[string]any
}

// Thinking is whether, and how much, a model thinks before it answers.
type Thinking struct {
	Mode   string // ThinkDefault, ThinkOff, ThinkOn, ThinkEffort or ThinkBudget
	Effort string // ThinkEffort's level: "none", "low", "high"…
	Budget int    // ThinkBudget's tokens; 0: the service's
	// ExposeReasoning: OpenRouter returns the reasoning, which it
	// otherwise leaves out.
	ExposeReasoning bool
}

// Thinking modes and response formats.
const (
	// ThinkDefault is Voxt's: the preset's ExtraBody, which turns thinking
	// off where a service lets it, for answers without the wait.
	ThinkDefault = ""
	ThinkOff     = "off"
	ThinkOn      = "on"
	ThinkEffort  = "effort"
	ThinkBudget  = "budget"

	ResponseJSON       = "json"
	ResponseJSONSchema = "jsonSchema"
)

// applyGeneration writes p's Generation over the body b of a request to
// model, as Voxt 1.x wrote its settings for an OpenAI-compatible API.
func applyGeneration(b map[string]any, p Preset, model string, stream bool) {
	g := p.Generation
	if g.MaxOutputTokens != nil {
		f := p.MaxTokensField
		if f == "" {
			f = "max_tokens"
		}
		b[f] = max(1, *g.MaxOutputTokens)
	}
	float := func(key string, v *float64) {
		if v != nil {
			b[key] = *v
		}
	}
	float("temperature", g.Temperature)
	float("top_p", g.TopP)
	if g.TopK != nil {
		b["top_k"] = *g.TopK
	}
	float("min_p", g.MinP)
	if g.Seed != nil {
		b["seed"] = *g.Seed
	}
	if len(g.Stop) > 0 {
		b["stop"] = g.Stop
	}
	if p.ID != "stepfun" { // StepFun refuses it
		float("presence_penalty", g.PresencePenalty)
	}
	float("frequency_penalty", g.FrequencyPenalty)
	float("repetition_penalty", g.RepetitionPenalty)
	if g.Logprobs {
		b["logprobs"] = true
		if g.TopLogprobs != nil {
			b["top_logprobs"] = *g.TopLogprobs
		}
	}
	// A feature that asks for JSON has set the format already.
	if _, set := b["response_format"]; !set {
		switch {
		case g.ResponseFormat == ResponseJSONSchema && len(g.JSONSchema) > 0:
			b["response_format"] = map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "result", "schema": g.JSONSchema}}
		case g.ResponseFormat != "":
			b["response_format"] = map[string]any{"type": "json_object"}
		}
	}
	applyThinking(b, p.ID, model, g.Thinking)
	if g.Verbosity != "" {
		b["verbosity"] = g.Verbosity
	}
	if g.KeepAlive != "" {
		b["keep_alive"] = g.KeepAlive
	}
	if stream && g.IncludeUsage != nil && !*g.IncludeUsage {
		delete(b, "stream_options")
	}
	mergePatch(b, g.ExtraBody)
}

// mergePatch applies patch to doc as a JSON merge patch (RFC 7386): an
// object merges into a copy of the object doc holds under its key (the
// preset's are shared), null removes the key, any other value replaces it.
func mergePatch(doc, patch map[string]any) {
	for k, v := range patch {
		switch v := v.(type) {
		case nil:
			delete(doc, k)
		case map[string]any:
			sub, _ := doc[k].(map[string]any)
			sub = maps.Clone(sub)
			if sub == nil {
				sub = map[string]any{}
			}
			mergePatch(sub, v)
			doc[k] = sub
		default:
			doc[k] = v
		}
	}
}

// applyThinking writes t in the fields service id reads (Voxt 1.x's
// mapping; a mode a service has no field for sends nothing).
func applyThinking(b map[string]any, id, model string, t Thinking) {
	enabled := func(budget int) map[string]any {
		m := map[string]any{"type": "enabled"}
		if budget > 0 {
			m["budget_tokens"] = budget
		}
		return m
	}
	disabled := map[string]any{"type": "disabled"}
	effort := func() {
		if t.Mode == ThinkEffort && t.Effort != "" {
			b["reasoning_effort"] = t.Effort
		}
	}
	switch id {
	case "openai", Custom, "xai":
		effort()
	case "openrouter":
		r := map[string]any{"exclude": !t.ExposeReasoning}
		switch t.Mode {
		case ThinkEffort:
			if t.Effort != "" {
				r["effort"] = t.Effort
			}
		case ThinkBudget:
			if t.Budget > 0 {
				r["max_tokens"] = t.Budget
			}
		case ThinkOff:
			r["enabled"] = false
		case ThinkOn:
			r["enabled"] = true
		}
		b["reasoning"] = r
	case "deepseek":
		switch t.Mode {
		case ThinkOff:
			b["thinking"] = disabled
		case ThinkOn, ThinkBudget: // DeepSeek has no budget: thinking on
			b["thinking"] = enabled(0)
		}
		effort()
	case "zhipu", "volcengine", "moonshot":
		switch t.Mode {
		case ThinkOff:
			b["thinking"] = disabled
		case ThinkOn:
			b["thinking"] = enabled(0)
		case ThinkBudget:
			b["thinking"] = enabled(t.Budget)
		}
		effort()
	case "dashscope":
		switch t.Mode {
		case ThinkOff:
			b["thinking"], b["enable_thinking"] = disabled, false
		case ThinkOn:
			b["thinking"], b["enable_thinking"] = enabled(0), true
		case ThinkBudget:
			b["enable_thinking"] = true
			if t.Budget > 0 {
				b["thinking_budget"] = t.Budget
			}
		}
		effort()
	case "mimo":
		switch t.Mode {
		case ThinkOff:
			b["thinking"] = disabled
		case ThinkOn:
			b["thinking"] = enabled(0)
		}
	case "stepfun":
		// Only step-3.5-flash-2603 takes an effort, low or high.
		if model == "step-3.5-flash-2603" && (t.Effort == "low" || t.Effort == "high") {
			effort()
		}
	case "anthropic":
		switch t.Mode {
		case ThinkOff:
			b["thinking"] = disabled
		case ThinkBudget:
			b["thinking"] = enabled(t.Budget)
			if n, ok := b["max_tokens"].(int); ok && n <= t.Budget { // Voxt 1.x kept it above the budget
				b["max_tokens"] = t.Budget + 1
			}
		}
	case "gemini":
		switch t.Mode {
		case ThinkOff:
			b["reasoning_effort"] = "none"
		case ThinkBudget:
			b["extra_body"] = map[string]any{"google": map[string]any{"thinking_config": map[string]any{"thinking_budget": t.Budget}}}
		}
	case "ollama":
		switch t.Mode {
		case ThinkOff:
			b["reasoning_effort"] = "none"
		case ThinkEffort:
			effort()
		}
	}
}
