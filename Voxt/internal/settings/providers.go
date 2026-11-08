package settings

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
	// Generation is a language model's request settings (unset: Voxt's).
	Generation Generation `json:"generation,omitzero"`
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
	OptionsJSON    string `json:"optionsJSON,omitempty"`  // Ollama's options: a JSON object
	// ExtraBodyJSON is a JSON object merged into the request body last.
	ExtraBodyJSON string `json:"extraBodyJSON,omitempty"`
}
