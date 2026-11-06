// Package llm is VoxtNew's chat-model layer, the part of AudioSmith's
// internal/llm (@f171c86, by way of Auralink) that does not depend on a
// model runtime:
//
//   - the request and response types a session sends (Request, Response,
//     Message, Format, Schema) and the skills that tag them;
//   - Backend: one model, one attempt per call. VoxtNew's language models
//     are local (internal/local: mlx-swift-lm and a few GGUF models in
//     the voxtnew-llm process, internal/llmrt); there is no backend here;
//   - Client: a Backend plus timeouts, retries with backoff, usage
//     accounting, thinking stripped from the text and JSON / structured
//     output (schema validation and a repair retry).
//
// Everything is goroutine-safe and blocks: call it off the UI thread.
// Errors are *Error values with a stable Code and Params, never
// user-visible text.
package llm

import (
	"context"
	"encoding/json"
	"time"
)

// Role is a chat message role.
type Role string

const (
	System    Role = "system"
	User      Role = "user"
	Assistant Role = "assistant"
)

// Message is one chat turn. Text only: VoxtNew sends no images.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

// Format selects the output format.
type Format uint8

const (
	FormatText Format = iota // free text
	FormatJSON               // a JSON value; Schema (if set) constrains it
)

// Request is one chat completion.
type Request struct {
	// Model is the backend's model id. Empty: the backend's default.
	Model    string
	Messages []Message

	Temperature *float64 // nil: the backend's default
	TopP        *float64
	MaxTokens   int // 0: the backend's default
	Stop        []string
	Seed        *int64

	// Format and Schema ask for JSON. Backends use the model's native
	// mode when it has one (a grammar, json_schema) and the prompt
	// otherwise; Client.JSON validates and repairs either way.
	Format Format
	Schema *Schema

	// OnDelta, when set, streams: it is called with each text delta, in
	// order, on the calling goroutine, before Chat returns. It must not
	// block for long. The Response still carries the whole text.
	OnDelta func(delta string)

	// Timeout bounds one attempt (0: the Client's default).
	Timeout time.Duration
	// FirstTokenTimeout and StallTimeout end an attempt that makes no
	// progress (0: no bound): no delta within FirstTokenTimeout of its
	// start, or none for StallTimeout after the one before. Every delta
	// counts, a reasoning block's too; the request streams for them
	// (OnDelta or not). Such a timeout is not retried: a model that is
	// slow would be slow again.
	FirstTokenTimeout time.Duration
	StallTimeout      time.Duration

	// Skill tags the call for usage accounting and its default
	// temperature (DefaultTemperature).
	Skill Skill

	// Translate describes a translation (SkillTranslateText) for a model
	// that takes a prompt of its own instead of Messages (a translation
	// model: internal/llmrt's Hy-MT2). nil for other requests; chat
	// models ignore it.
	Translate *TranslateHint
}

// TranslateHint is what a translation model needs: the text and the
// target language.
type TranslateHint struct {
	Text   string // the text to translate: a selection as it is, speech without its fillers (compose.StripFillers)
	Target string // the target language's code (internal/lang)
}

// F64 returns a pointer to v, for Request.Temperature and TopP.
func F64(v float64) *float64 { return &v }

// Usage is the token count of a call. Estimated is true when the
// backend reported none and the counts come from EstimateTokens.
type Usage struct {
	PromptTokens     int  `json:"promptTokens"`
	CompletionTokens int  `json:"completionTokens"`
	TotalTokens      int  `json:"totalTokens"`
	Estimated        bool `json:"estimated,omitempty"`
}

// Add returns u + v.
func (u Usage) Add(v Usage) Usage {
	return Usage{
		PromptTokens:     u.PromptTokens + v.PromptTokens,
		CompletionTokens: u.CompletionTokens + v.CompletionTokens,
		TotalTokens:      u.TotalTokens + v.TotalTokens,
		Estimated:        u.Estimated || v.Estimated,
	}
}

// Finish reasons, normalised across backends.
const (
	FinishStop   = "stop"
	FinishLength = "length" // hit MaxTokens
	FinishFilter = "content_filter"
	FinishOther  = "other"
)

// Response is a finished completion.
type Response struct {
	Text         string
	FinishReason string
	Model        string // as reported by the backend (may differ from the request)
	Provider     string // backend id
	Usage        Usage
	Latency      time.Duration // of the successful attempt
	Attempts     int           // 1 + retries (Client)
	Repairs      int           // JSON repair turns (Client.JSON)
	// JSON is the validated value for FormatJSON requests made through
	// Client.JSON (the extracted JSON text otherwise empty).
	JSON json.RawMessage
}

// BackendInfo describes a backend.
type BackendInfo struct {
	ID string // the model's id in internal/local ("" for none)
}

// Backend is one model. Chat makes a single attempt (Client adds
// retries); it honours ctx cancellation and deadlines and returns *Error.
// Implementations are goroutine-safe.
type Backend interface {
	Info() BackendInfo
	Chat(ctx context.Context, req Request) (Response, error)
}

// Prefiller is a model that keeps a prompt cache (a local model):
// Prefill evaluates the shared part of req's prompt, its messages before
// the last, so that a chat starting the same way answers sooner. The last
// message's text does not matter.
type Prefiller interface {
	Prefill(ctx context.Context, req Request) error
}
