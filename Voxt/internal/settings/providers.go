package settings

import (
	"encoding/json"
	"strings"
)

// Providers are the remote services the user set up, in the order they
// were added (Auralink's): the language models of the Remote models page
// and the speech recognition services; a service nobody set up has no
// entry.
type Providers struct {
	LLM []Provider `json:"llm,omitempty"`
	ASR []Provider `json:"asr,omitempty"`
}

// Provider is one remote service the user set up. Its key is in the
// Keychain (internal/secrets: "llm.<ID>", "asr.<ID>"), never here. ""
// fields use its preset's (internal/llm's presets.json, internal/
// providers' speech presets).
type Provider struct {
	// ID is its preset's id ("openai", "deepseek", "doubao"), one setup
	// each per list; a feature's LLM or ASR holds it to use the service.
	ID string `json:"id"`
	// Endpoint is a language model's API root ("https://…/v1"), a speech
	// service's URL (wss://…, https://…/audio/transcriptions).
	Endpoint string `json:"endpoint,omitempty"`
	Model    string `json:"model,omitempty"`
	// AppID is Doubao's App ID (its older console: the key is then the
	// Access Token); "" for its newer console's single API key.
	AppID string `json:"appID,omitempty"`
	// RecognitionPrompt is a remote speech service's own recognition hint
	// template (Voxt 1.x's per-service ASR prompt). Session assembly puts
	// the global speech context first, then this text with
	// {{USER_MAIN_LANGUAGE}}, {{USER_OTHER_LANGUAGES}} and
	// {{DICTIONARY_TERMS}} filled in. Services that take no prompt
	// ignore it.
	RecognitionPrompt string `json:"recognitionPrompt,omitempty"`
	// Generation is a language model's request settings (unset: Voxt's).
	Generation Generation `json:"generation,omitzero"`
	// APIFormat is how a language model is called (OpenAI and custom
	// services only): "" chat completions, "responses" the Responses API.
	APIFormat string `json:"apiFormat,omitempty"`
}

// Generation is a language model setup's request settings, the fields of
// Voxt 1.x's service dialog (Generation Settings, Advanced, Expert, a
// few services' own options); internal/providers turns them into the
// request's (llm.Generation). An unset field leaves Voxt's value, or the
// service's default.
type Generation struct {
	MaxOutputTokens   *int     `json:"maxOutputTokens,omitempty"`
	Temperature       *float64 `json:"temperature,omitempty"`
	TopP              *float64 `json:"topP,omitempty"`
	TopK              *int     `json:"topK,omitempty"`
	MinP              *float64 `json:"minP,omitempty"`
	Seed              *int64   `json:"seed,omitempty"`
	Stop              []string `json:"stop,omitempty"`
	PresencePenalty   *float64 `json:"presencePenalty,omitempty"`
	FrequencyPenalty  *float64 `json:"frequencyPenalty,omitempty"`
	RepetitionPenalty *float64 `json:"repetitionPenalty,omitempty"`
	Logprobs          bool     `json:"logprobs,omitempty"`
	TopLogprobs       *int     `json:"topLogprobs,omitempty"`
	// ResponseFormat is "" (plain text), "json" or "jsonSchema" (with
	// JSONSchema).
	ResponseFormat string `json:"responseFormat,omitempty"`
	JSONSchema     string `json:"jsonSchema,omitempty"`
	// Thinking is "" (Voxt's default), "off", "on", "effort" (with
	// ThinkingEffort) or "budget" (with ThinkingBudget tokens).
	Thinking       string `json:"thinking,omitempty"`
	ThinkingEffort string `json:"thinkingEffort,omitempty"`
	ThinkingBudget int    `json:"thinkingBudget,omitempty"`
	Verbosity      string `json:"verbosity,omitempty"`    // OpenAI's
	KeepAlive      string `json:"keepAlive,omitempty"`    // Ollama's
	IncludeUsage   *bool  `json:"includeUsage,omitempty"` // oMLX's
	// ExtraBodyJSON is the dialog's Request parameters: a JSON object
	// applied to the request body last as a JSON merge patch (RFC 7386),
	// so its keys win, an object merges into the one Voxt sends (Ollama's
	// "options") and null removes a field.
	ExtraBodyJSON string `json:"extraBodyJSON,omitempty"`
}

// WithOptions is the Request parameters extra with options, Ollama's
// options as a JSON object (a field of its own in Voxt 1.x and before
// 2.0.3), as their "options"; extra's own "options" wins, as it did.
func WithOptions(extra, options string) string {
	var opts map[string]any
	if json.Unmarshal([]byte(options), &opts) != nil || len(opts) == 0 {
		return extra
	}
	body := map[string]any{}
	if strings.TrimSpace(extra) != "" && json.Unmarshal([]byte(extra), &body) != nil || body == nil {
		return extra
	}
	if _, ok := body["options"]; ok {
		return extra
	}
	body["options"] = opts
	b, _ := json.MarshalIndent(body, "", "  ")
	return string(b)
}

// migrateOptionsJSON folds a stored setup's Options JSON into its
// Request parameters.
func migrateOptionsJSON(data []byte, p *Providers) {
	var doc struct {
		Providers struct {
			LLM []struct {
				Generation struct {
					OptionsJSON string `json:"optionsJSON"`
				} `json:"generation"`
			} `json:"llm"`
		} `json:"providers"`
	}
	if json.Unmarshal(data, &doc) != nil {
		return
	}
	for i, old := range doc.Providers.LLM {
		if i < len(p.LLM) && old.Generation.OptionsJSON != "" {
			g := &p.LLM[i].Generation
			g.ExtraBodyJSON = WithOptions(g.ExtraBodyJSON, old.Generation.OptionsJSON)
		}
	}
}
