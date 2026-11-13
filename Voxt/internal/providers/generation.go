package providers

import (
	"encoding/json"
	"errors"
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// GenCaps are the request settings a language model service takes: the
// fields its dialog shows, as Voxt 1.x's capability registry had them.
type GenCaps struct {
	// ThinkModes are the Think menu's choices (llm.ThinkDefault…); none:
	// no Think row. Efforts are ThinkEffort's levels; none: typed.
	ThinkModes []string
	Efforts    []string

	MaxOutputTokens, Temperature, TopP bool // Generation Settings
	TopK, MinP, Seed, Logprobs, Stop   bool // Advanced
	// Penalties: presence, frequency and repetition; FrequencyOnly:
	// StepFun's frequency penalty alone.
	Penalties, FrequencyOnly bool
	// ResponseFormat: plain text or JSON; JSONSchema adds a schema.
	ResponseFormat, JSONSchema bool
	ExtraBody                  bool // Advanced: Request parameters
	// A service's own options: OpenAI's verbosity, Ollama's keep alive,
	// oMLX's usage in stream events.
	Verbosity, KeepAlive, IncludeUsage bool
	// APIFormat: Chat Completions or the Responses API.
	APIFormat bool
}

// The effort levels of Voxt 1.x's menus.
var (
	effortsOpenAI = []string{"none", "minimal", "low", "medium", "high", "xhigh"}
	effortsCustom = append(slices.Clone(effortsOpenAI), "max")
)

// Capabilities are the request settings language model service id takes,
// for model (StepFun's effort depends on it).
func Capabilities(id, model string) GenCaps {
	if id == ChatGPT {
		return GenCaps{} // the ChatGPT backend takes none of them (Voxt 1.x's)
	}
	c := GenCaps{MaxOutputTokens: true, Temperature: true, TopP: true, Stop: true, ExtraBody: true}
	var toggle, effort, budget bool
	switch id {
	case "openai", llm.Custom:
		effort, c.Logprobs, c.ResponseFormat, c.Verbosity, c.APIFormat = true, true, true, true, true
		c.Efforts = effortsOpenAI
		if id == llm.Custom {
			c.Efforts = effortsCustom
		}
	case "anthropic":
		budget, c.TopK = true, true
	case "gemini":
		budget, c.TopK, c.ResponseFormat = true, true, true
	case "ollama":
		toggle, effort, c.TopK, c.MinP, c.Seed, c.Penalties, c.Logprobs, c.ResponseFormat, c.JSONSchema = true, true, true, true, true, true, true, true, true
		c.KeepAlive = true
		c.Efforts = []string{"low", "medium", "high"}
	case "omlx", "lmstudio":
		c.TopK, c.MinP, c.Seed, c.Penalties, c.Logprobs, c.ResponseFormat = true, true, true, true, true, true
		c.JSONSchema, c.IncludeUsage = id == "omlx", id == "omlx"
	case "deepseek":
		toggle, effort, c.Logprobs, c.ResponseFormat = true, true, true, true
		c.Efforts = []string{"none", "low", "high", "max"}
	case "zhipu", "volcengine", "dashscope":
		toggle, effort, budget, c.Penalties, c.Logprobs, c.ResponseFormat = true, true, true, true, true, true
		c.Efforts = effortsOpenAI
	case "mimo":
		toggle, c.Penalties, c.ResponseFormat = true, true, true
	case "stepfun":
		c.FrequencyOnly, c.ResponseFormat = true, true
		c.ThinkModes = []string{llm.ThinkOff}
		if strings.TrimSpace(strings.ToLower(model)) == "step-3.5-flash-2603" {
			c.ThinkModes = append(c.ThinkModes, llm.ThinkEffort)
			c.Efforts = []string{"low", "high"}
		}
		return c
	case "openrouter", "xai", "moonshot":
		effort, budget, c.Penalties, c.Logprobs, c.ResponseFormat = true, true, true, true, true
		c.Efforts = effortsOpenAI
	case "minimax":
		c.Penalties, c.ResponseFormat = true, true
	}
	if toggle || effort || budget {
		c.ThinkModes = []string{llm.ThinkDefault}
		if toggle {
			c.ThinkModes = append(c.ThinkModes, llm.ThinkOff, llm.ThinkOn)
		}
		if effort {
			c.ThinkModes = append(c.ThinkModes, llm.ThinkEffort)
		}
		if budget {
			c.ThinkModes = append(c.ThinkModes, llm.ThinkBudget)
		}
	}
	return c
}

// Generation is g as the request's settings. A JSON field that is not a
// JSON object is an error (CheckGeneration), and left out here.
func Generation(g settings.Generation) llm.Generation {
	out := llm.Generation{
		MaxOutputTokens: g.MaxOutputTokens, Temperature: g.Temperature, TopP: g.TopP, TopK: g.TopK, MinP: g.MinP,
		Seed: g.Seed, Stop: g.Stop, PresencePenalty: g.PresencePenalty, FrequencyPenalty: g.FrequencyPenalty,
		RepetitionPenalty: g.RepetitionPenalty, Logprobs: g.Logprobs, TopLogprobs: g.TopLogprobs,
		ResponseFormat: g.ResponseFormat, Verbosity: g.Verbosity, KeepAlive: g.KeepAlive, IncludeUsage: g.IncludeUsage,
		Thinking: llm.Thinking{Mode: g.Thinking, Effort: g.ThinkingEffort, Budget: g.ThinkingBudget},
	}
	out.JSONSchema, _ = jsonObject(g.JSONSchema)
	out.ExtraBody, _ = jsonObject(g.ExtraBodyJSON)
	return out
}

// Errors of CheckGeneration: the field that is not a JSON object.
var (
	ErrJSONSchema = errors.New("the JSON schema is not a JSON object")
	ErrExtraBody  = errors.New("the request parameters are not a JSON object")
)

// CheckGeneration checks g's JSON fields, before a setup is saved.
func CheckGeneration(g settings.Generation) error {
	for _, f := range []struct {
		s   string
		err error
	}{{g.JSONSchema, ErrJSONSchema}, {g.ExtraBodyJSON, ErrExtraBody}} {
		if _, err := jsonObject(f.s); err != nil {
			return f.err
		}
	}
	return nil
}

// jsonObject parses s as a JSON object; "" is none.
func jsonObject(s string) (map[string]any, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil || m == nil {
		return nil, errors.New("not a JSON object")
	}
	return m, nil
}
