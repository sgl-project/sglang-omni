package settings

// Providers are the remote language models the user set up on the Remote
// models page, in the order they were added (Auralink's, language models
// only); a service nobody set up has no entry.
type Providers struct {
	LLM []Provider `json:"llm,omitempty"`
}

// Provider is one remote service the user set up. Its key is in the
// Keychain (internal/secrets: "llm.<ID>"), never here. "" fields use its
// preset's (internal/llm, presets.json).
type Provider struct {
	// ID is its preset's id ("openai", "deepseek", "custom"), one setup
	// each; a feature's LLM holds it to use the service.
	ID       string `json:"id"`
	Endpoint string `json:"endpoint,omitempty"` // the API root ("https://…/v1")
	Model    string `json:"model,omitempty"`
}
