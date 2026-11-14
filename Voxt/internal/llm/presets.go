package llm

import (
	"context"
	_ "embed"
	"encoding/json"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
)

// JSONMode is how a remote service is asked for JSON.
type JSONMode string

const (
	JSONSchemaMode JSONMode = "schema" // response_format: json_schema (OpenAI structured outputs)
	JSONObjectMode JSONMode = "object" // response_format: json_object; the schema goes in the prompt
	JSONPromptMode JSONMode = "prompt" // no API support; the schema goes in the prompt
)

// Preset is a remote service's fixed facts (presets.json, Auralink's,
// OpenAI-compatible ones only).
type Preset struct {
	ID      string `json:"id"`
	Name    string `json:"name"`    // brand name, not translated
	BaseURL string `json:"baseURL"` // the API root ("…/v1"); "" for custom: the user supplies it
	// KeyOptional: it works without a key (a server on this Mac).
	KeyOptional bool `json:"keyOptional,omitempty"`
	// SignIn: reached with the user's sign-in (OpenAI.WithTokens) rather
	// than a key, by streamed requests with only the fields bareBody keeps.
	SignIn bool `json:"signIn,omitempty"`
	// Models are the suggested models, for the model menu (any id may be
	// typed).
	Models       []string          `json:"models,omitempty"`
	JSONMode     JSONMode          `json:"jsonMode"`
	DefaultModel string            `json:"defaultModel,omitempty"`
	Headers      map[string]string `json:"headers,omitempty"`
	// MaxTokensField is the request field for Request.MaxTokens
	// (OpenAI: "max_completion_tokens"); default "max_tokens".
	MaxTokensField string `json:"maxTokensField,omitempty"`
	// ExtraBody is merged into every request body (top-level keys;
	// request fields win): how a preset turns a service's thinking off,
	// which otherwise spends a short answer's time before any text
	// (Voxt's lesson). A server that rejects one of its keys is asked
	// again without ExtraBody.
	ExtraBody map[string]any `json:"extraBody,omitempty"`
	// KeyURL is where to get a key.
	KeyURL string `json:"keyURL,omitempty"`
	// Generation is the user's request settings (ProviderConfig), not
	// the preset's.
	Generation Generation `json:"-"`
	// API is the API format: "" chat completions, APIResponses; the
	// user's ProviderConfig.API replaces it.
	API string `json:"api,omitempty"`
}

// Custom is the preset of a server the user names (any OpenAI-compatible
// API).
const Custom = "custom"

//go:embed presets.json
var presetsJSON []byte

var presets = sync.OnceValue(func() []Preset {
	var doc struct {
		Presets []Preset `json:"presets"`
	}
	if err := json.Unmarshal(presetsJSON, &doc); err != nil {
		panic("llm: presets.json: " + err.Error())
	}
	return doc.Presets
})

// Presets returns the remote services' presets, in display order. The
// slice is a copy.
func Presets() []Preset { return slices.Clone(presets()) }

// LookupPreset returns the preset with id.
func LookupPreset(id string) (Preset, bool) {
	for _, p := range presets() {
		if p.ID == id {
			return p, true
		}
	}
	return Preset{}, false
}

// ProviderConfig is a user's setup of one remote service (settings keep
// it; never its key).
type ProviderConfig struct {
	ID      string // the preset's id ("openai", "custom")
	BaseURL string // overrides the preset's
	Model   string // overrides the preset's default
	// Generation is the user's request settings (Voxt 1.x's fields).
	Generation Generation
	API        string // Preset.API
}

// Resolve merges cfg over its preset and checks the address: keys go
// over https only, plain http only to this Mac (a local server), and
// never with a user or password in the URL. A full chat completions or
// Responses URL is taken for its API root.
func (cfg ProviderConfig) Resolve() (Preset, error) {
	p, ok := LookupPreset(cfg.ID)
	if !ok {
		return Preset{}, newErr(CodeUnknown, "provider", cfg.ID)
	}
	if b := strings.TrimSpace(cfg.BaseURL); b != "" {
		p.BaseURL = b
	}
	if m := strings.TrimSpace(cfg.Model); m != "" {
		p.DefaultModel = m
	}
	p.Generation = cfg.Generation
	if cfg.API != "" {
		p.API = cfg.API
	}
	p.BaseURL = strings.TrimSuffix(strings.TrimSuffix(strings.TrimRight(p.BaseURL, "/"), "/chat/completions"), "/responses")
	if p.BaseURL == "" {
		return Preset{}, newErr(CodeConfig, "provider", p.ID, "field", "baseURL")
	}
	u, err := url.Parse(p.BaseURL)
	if err != nil || u.Host == "" || u.User != nil || (u.Scheme != "https" && !(u.Scheme == "http" && core.IsLoopback(u.Hostname()))) {
		return Preset{}, newErr(CodeConfig, "provider", p.ID, "field", "baseURL")
	}
	return p, nil
}

// ProbeTimeout bounds a Probe.
const ProbeTimeout = 20 * time.Second

// Probe checks that b answers ("Test connection"): one tiny, non-streamed
// request, no retries, at most a few tokens out, so that it costs next to
// nothing and returns fast. An answer cut short by the token limit, even
// an empty one, counts as working; a refused request (400, 422: a bad
// model or parameter) does not and carries the server's detail. It
// returns the request's latency.
func Probe(ctx context.Context, b Backend) (time.Duration, error) {
	ctx, cancel := context.WithTimeout(ctx, ProbeTimeout)
	defer cancel()
	req := Request{
		Messages:  []Message{{Role: User, Content: "Reply with: ok"}},
		MaxTokens: 8,
		Timeout:   ProbeTimeout,
	}
	start := time.Now()
	if _, err := b.Chat(ctx, req); err != nil {
		return 0, ctxErr(ctx, b.Info().ID, err, ProbeTimeout)
	}
	return time.Since(start), nil
}
