package voxtcore

import (
	"cmp"
	"context"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/openai"
	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// Language is a language the pickers offer.
type Language struct {
	Code string `json:"code"`
	// Name is its name in English and Chinese is its name in Simplified
	// Chinese (lang.Language.NameIn); Native in the language itself ("" when
	// unknown).
	Name    string `json:"name"`
	Chinese string `json:"chinese"`
	Native  string `json:"native,omitempty"`
}

// Languages are the languages speech models commonly recognise (Whisper's
// list, as internal/lang has it), sorted by code.
func Languages() []Language {
	var out []Language
	for _, code := range lang.WhisperCodes() {
		l, ok := lang.Lookup(code)
		if !ok {
			continue
		}
		out = append(out, Language{Code: l.Code, Name: l.Name, Chinese: l.NameIn("zh-Hans"), Native: l.Native})
	}
	return out
}

// Presets are the language model services the iOS settings offer: the
// macOS app's presets but those reached by signing in (ChatGPT), whose
// browser sign-in the iOS app does not have yet.
func Presets() []Preset {
	var out []Preset
	for _, p := range llm.Presets() {
		if p.SignIn {
			continue
		}
		out = append(out, Preset{
			ID: p.ID, Name: p.Name, BaseURL: p.BaseURL, KeyOptional: p.KeyOptional,
			Models: p.Models, DefaultModel: p.DefaultModel, KeyURL: p.KeyURL,
		})
	}
	return out
}

// openBackend is p's transport, as the macOS app opens a setup: what it
// lacks first (providers.Missing: an address, a model, a key), then
// providers.NewLLM. A setup the app cannot use is a config Error.
func openBackend(p Provider, client *http.Client) (*llm.OpenAI, error) {
	setup := settings.Provider{ID: p.Preset, Endpoint: p.BaseURL, Model: p.Model}
	preset, ok := llm.LookupPreset(p.Preset)
	lacks := func(field, what string) error {
		return &Error{Code: CodeConfig, Field: field, Message: fmt.Sprintf("%s %s", cmp.Or(preset.Name, strconv.Quote(p.Preset)), what)}
	}
	switch {
	case !ok:
		return nil, lacks(FieldPreset, "is not a language model service this app has")
	case preset.SignIn:
		return nil, lacks(FieldSignIn, "needs a sign-in, which the iOS app does not offer yet")
	}
	switch providers.Missing(setup, strings.TrimSpace(p.Key) != "") {
	case providers.MissingEndpoint:
		return nil, lacks(FieldAddress, "needs an API address")
	case providers.MissingModel:
		return nil, lacks(FieldModel, "needs a model")
	case providers.MissingKey:
		return nil, lacks(FieldKey, "needs an API key")
	}
	return providers.NewLLM(setup, p.Key, client) // MissingAddress: Resolve's own error says why
}

// TestProvider is Settings' "Test connection": one tiny request
// (llm.Probe).
func TestProvider(ctx context.Context, p Provider, client *http.Client) (TestResult, error) {
	backend, err := openBackend(p, client)
	if err != nil {
		return TestResult{}, err
	}
	ctx, cancel := context.WithTimeout(ctx, llm.ProbeTimeout)
	defer cancel()
	latency, err := llm.Probe(ctx, backend)
	return TestResult{LatencyMillis: latency.Milliseconds()}, err
}

// Transcribe sends a WAV recording to a remote OpenAI-compatible speech
// service (internal/asr/openai: https, or http on loopback).
func Transcribe(ctx context.Context, req SpeechRequest, client *http.Client) (SpeechResult, error) {
	audio, err := os.ReadFile(req.Path)
	if err != nil {
		return SpeechResult{}, err
	}
	c, err := openai.New(asr.Config{Endpoint: req.Endpoint, Model: req.Model, Key: req.Key, HTTPClient: client})
	if err != nil {
		return SpeechResult{}, err
	}
	hints := make([]string, 0, len(req.Terms))
	for _, t := range req.Terms {
		hints = append(hints, t.Term)
	}
	start := time.Now()
	res, err := c.Transcribe(ctx, asr.FileRequest{Audio: audio, Format: "wav", Lang: req.Language, Hints: hints, Prompt: req.Prompt})
	if err != nil {
		return SpeechResult{}, err
	}
	return SpeechResult{Text: res.Text, Language: res.Lang, Model: res.Model, LatencyMillis: time.Since(start).Milliseconds()}, nil
}
