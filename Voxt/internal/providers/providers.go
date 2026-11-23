// Package providers joins the user's setup of the remote language models
// and speech services (settings.Providers) to the code that talks to them
// (internal/llm's OpenAI transport, internal/asr's clients: speech.go):
// which setups are ready, building a client from a setup and its key, the
// connection test, and Services, which offers the ready ones to the
// features beside the local models. It is Auralink's internal/providers
// (@748cf55).
//
// It never reads the Keychain itself: callers pass keys in (read off the
// main thread with internal/secrets) and a hasKey function for the
// decisions that only need to know whether one is saved.
//
// Imports: asr (+ openai, doubao, aliyun, gemini, mimo), codexauth, llm, local,
// settings.
package providers

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/codexauth"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// Account is the Keychain account of a setup's key (internal/secrets):
// "llm.<id>", "asr.<id>". The two lists share ids ("openai").
func Account(kind local.Kind, id string) string { return string(kind) + "." + id }

// Setup is the user's setup of service id, and whether there is one.
func Setup(s settings.Settings, id string) (settings.Provider, bool) {
	return setupIn(s.Providers.LLM, id)
}

func setupIn(list []settings.Provider, id string) (settings.Provider, bool) {
	for _, p := range list {
		if p.ID == id {
			return p, true
		}
	}
	return settings.Provider{ID: id}, false
}

// Put sets p in s, replacing the setup with its ID or appending it. It
// writes a new list: copies of the settings share the old one's array.
func Put(s *settings.Settings, p settings.Provider) { s.Providers.LLM = put(s.Providers.LLM, p) }

func put(list []settings.Provider, p settings.Provider) []settings.Provider {
	l := slices.Clone(list)
	if i := slices.IndexFunc(l, func(x settings.Provider) bool { return x.ID == p.ID }); i >= 0 {
		l[i] = p
	} else {
		l = append(l, p)
	}
	return l
}

// Remove takes setup id out of s (a new list, as Put), and out of the
// features that use it (they go back to automatic).
func Remove(s *settings.Settings, id string) {
	s.Providers.LLM = slices.DeleteFunc(slices.Clone(s.Providers.LLM), func(p settings.Provider) bool { return p.ID == id })
	for _, f := range []*settings.Feature{&s.Features.Transcribe.Feature, &s.Features.Translate.Feature, &s.Features.Rewrite.Feature} {
		if f.LLM == id {
			f.LLM = ""
		}
	}
}

// Config is the llm package's configuration of a setup.
func Config(p settings.Provider) llm.ProviderConfig {
	return llm.ProviderConfig{ID: p.ID, BaseURL: p.Endpoint, Model: p.Model, Generation: Generation(p.Generation), API: p.APIFormat}
}

// Model is the model a setup uses: its own, else its preset's default.
func Model(p settings.Provider) string {
	if m := strings.TrimSpace(p.Model); m != "" {
		return m
	}
	pr, _ := llm.LookupPreset(p.ID)
	return pr.DefaultModel
}

// What a setup can lack (Missing).
const (
	MissingKey      = "key"
	MissingEndpoint = "endpoint"
	MissingModel    = "model"
	MissingAddress  = "address" // saved, but Resolve refuses it
)

// Missing says what a setup lacks before it can be used: "" when it is
// ready.
func Missing(p settings.Provider, hasKey bool) string {
	pr, ok := llm.LookupPreset(p.ID)
	switch {
	case !ok || pr.BaseURL == "" && strings.TrimSpace(p.Endpoint) == "":
		return MissingEndpoint
	case Model(p) == "":
		return MissingModel
	case !hasKey && !pr.KeyOptional:
		return MissingKey
	}
	if _, err := Config(p).Resolve(); err != nil {
		return MissingAddress
	}
	return ""
}

// Ready lists the setups that can be used now, in the presets' order.
func Ready(s settings.Settings, hasKey func(id string) bool) []settings.Provider {
	var out []settings.Provider
	for _, pr := range llm.Presets() {
		if p, ok := Setup(s, pr.ID); ok && Missing(p, hasKey(p.ID)) == "" {
			out = append(out, p)
		}
	}
	return out
}

// NewLLM is the backend of a setup with its key, over client (nil:
// http.DefaultClient). It fails when the setup's address is not allowed
// (llm.ProviderConfig.Resolve).
func NewLLM(p settings.Provider, key string, client *http.Client) (*llm.OpenAI, error) {
	pr, err := Config(p).Resolve()
	if err != nil {
		return nil, err
	}
	return llm.NewOpenAI(pr, key, client), nil
}

// ChatGPT is the id of the setup reached by signing in to ChatGPT.
const ChatGPT = "codex"

// ChatGPTTokens is sess as the transport's token source.
func ChatGPTTokens(sess *codexauth.Session) llm.TokenSource {
	return func(ctx context.Context, stale string) (llm.Credential, error) {
		c, err := sess.Token(ctx, stale)
		switch {
		case errors.Is(err, codexauth.ErrSignedOut):
			return llm.Credential{}, llm.SignInError(ChatGPT)
		case err != nil:
			return llm.Credential{}, err
		}
		return llm.Credential{Token: c.Access, AccountID: c.AccountID}, nil
	}
}

// NewSignedIn is the backend of a setup that signs in instead of having a
// key.
func NewSignedIn(p settings.Provider, tokens llm.TokenSource, client *http.Client) (*llm.OpenAI, error) {
	b, err := NewLLM(p, "", client)
	if err != nil {
		return nil, err
	}
	return b.WithTokens(tokens), nil
}

// OpenSignedIn is Open for a setup that signs in.
func OpenSignedIn(p settings.Provider, tokens llm.TokenSource, client *http.Client) (local.Chat, error) {
	b, err := NewSignedIn(p, tokens, client)
	if err != nil {
		return nil, err
	}
	return &llm.Client{Backend: b, Retry: llm.RetryPolicy{MaxAttempts: 2}}, nil
}

// TestSignedIn is Test for a setup that signs in.
func TestSignedIn(ctx context.Context, p settings.Provider, tokens llm.TokenSource, client *http.Client) TestResult {
	res := TestResult{At: time.Now()}
	b, err := NewSignedIn(p, tokens, client)
	if err == nil {
		res.Latency, err = llm.Probe(ctx, b)
	}
	res.Err = err
	return res
}

// Open is a setup as a session uses it: an llm.Client (two attempts,
// JSON repair, thinking stripped) over its backend.
func Open(p settings.Provider, key string, client *http.Client) (local.Chat, error) {
	b, err := NewLLM(p, key, client)
	if err != nil {
		return nil, err
	}
	return &llm.Client{Backend: b, Retry: llm.RetryPolicy{MaxAttempts: 2}}, nil
}

// TestResult is the outcome of a connection test.
type TestResult struct {
	Latency time.Duration
	Err     error // nil: it works
	At      time.Time
	Models  []string // when the server lacks the model: the ones it has, if it lists them
}

// Test checks that a setup works: one tiny request (llm.Probe). When the
// server says it lacks the model, it also asks which models it has.
func Test(ctx context.Context, p settings.Provider, key string, client *http.Client) TestResult {
	res := TestResult{At: time.Now()}
	b, err := NewLLM(p, key, client)
	if err == nil {
		res.Latency, err = llm.Probe(ctx, b)
		if llm.Code(err) == llm.CodeModelNotFound {
			res.Models, _ = b.Models(ctx)
		}
	}
	res.Err = err
	return res
}

// Models lists the models a setup's server has (llm.OpenAI.Models).
func Models(ctx context.Context, p settings.Provider, key string, client *http.Client) ([]string, error) {
	b, err := NewLLM(p, key, client)
	if err != nil {
		return nil, err
	}
	return b.Models(ctx)
}

// TestLog is the log line of a connection test: the service, its model,
// the outcome, how long it took and, for a failure, the error (its code,
// HTTP status and the service's message, clipped). Never the key
// (removed should an error quote it).
func TestLog(p settings.Provider, res TestResult, key string) (level, msg string) {
	return testLog(fmt.Sprintf("connection test of remote language model %s (model %s)", p.ID, Model(p)), res, key)
}

func testLog(head string, res TestResult, key string) (level, msg string) {
	if res.Err == nil {
		return "info", fmt.Sprintf("%s: ok in %v", head, res.Latency.Round(time.Millisecond))
	}
	return "warning", fmt.Sprintf("%s: failed: %s", head, ErrorLog(res.Err, key))
}

// ErrorLog is err for the log: clipped to about 400 bytes, with key (when
// not empty) blanked out.
func ErrorLog(err error, key string) string {
	if err == nil {
		return ""
	}
	s := err.Error()
	if key = strings.TrimSpace(key); len(key) >= 4 {
		s = strings.ReplaceAll(s, key, "[key]")
	}
	if len(s) > 400 {
		s = s[:400] + "…"
	}
	return s
}
