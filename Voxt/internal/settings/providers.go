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
}
