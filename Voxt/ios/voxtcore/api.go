// Package voxtcore is the part of Voxt that the iOS app shares with the
// macOS one: the prompt composer (internal/compose), the language models'
// client and presets (internal/llm), snippets (internal/snippet) and the
// OpenAI-compatible speech client (internal/asr/openai), behind one JSON
// call per operation (Call). The app reaches it through the C ABI in
// cabi/; this package itself is pure Go and tested on any platform.
//
// The steps follow the macOS session (internal/session): Transcribe is
// cleaned up in parts (compose.BuildParts, CleanOutput, JoinParts) with
// its snippets' triggers marked and restored after the model; Translate
// asks again, strictly, when the first answer looks untranslated;
// Rewrite asks for a {title, content} answer when it writes from
// scratch. Recording, the speech models on the device, the Keychain and
// the screens are the app's (VoxtKit).
package voxtcore

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
)

// Provider is the language model a request goes to: a preset's id
// (llm.Presets) with the user's overrides and the key the app read from
// its Keychain. The key is never logged or returned.
type Provider struct {
	Preset  string `json:"preset"`
	BaseURL string `json:"baseURL,omitempty"`
	Model   string `json:"model,omitempty"`
	Key     string `json:"key,omitempty"`
}

// Term is a dictionary entry: the term as it must be written and what
// speech recognition tends to make of it.
type Term struct {
	Term    string   `json:"term"`
	Aliases []string `json:"aliases,omitempty"`
}

// Snippet is a spoken trigger and the text it stands for.
type Snippet struct {
	Trigger string `json:"trigger"`
	Text    string `json:"text"`
}

// Feature is a feature's settings (settings.Feature's on macOS).
type Feature struct {
	Cleanup   bool   `json:"cleanup"`
	Structure bool   `json:"structure,omitempty"`
	Prompt    string `json:"prompt,omitempty"`
}

// Turn is an earlier request of a Rewrite conversation and its answer.
type Turn struct {
	Instruction string `json:"instruction"`
	Source      string `json:"source,omitempty"`
	Title       string `json:"title,omitempty"`
	Answer      string `json:"answer"`
}

// ProcessRequest is one feature's language-model step.
type ProcessRequest struct {
	// Task is "transcribe", "translate" or "rewrite".
	Task string `json:"task"`
	// Text is the recognised speech (Transcribe, Translate) or the text
	// to rewrite (Rewrite; "" writes from scratch).
	Text string `json:"text"`
	// Typed says Translate's Text was typed or pasted, not spoken: it is
	// translated as it is, without the speech clean-up.
	Typed bool `json:"typed,omitempty"`
	// Instruction is Rewrite's spoken instruction.
	Instruction string `json:"instruction,omitempty"`
	// TargetLanguage is Translate's (a lang code, "en").
	TargetLanguage string `json:"targetLanguage,omitempty"`
	// MainLanguage and OtherLanguages are the user's (lang codes).
	MainLanguage   string   `json:"mainLanguage"`
	OtherLanguages []string `json:"otherLanguages,omitempty"`
	Feature        Feature  `json:"feature"`
	// Notes are the personal dictionary's Markdown notes; Terms its
	// glossary.
	Notes    string    `json:"notes,omitempty"`
	Terms    []Term    `json:"terms,omitempty"`
	Snippets []Snippet `json:"snippets,omitempty"`
	Previous []Turn    `json:"previous,omitempty"`
	Provider Provider  `json:"provider"`
}

// ProcessResult is the step's answer.
type ProcessResult struct {
	Text  string `json:"text"`
	Title string `json:"title,omitempty"`
	// Profile names the prompt's layers ("transcribe:cleanup+structure").
	Profile string `json:"profile,omitempty"`
	// Degraded: some of the speech could not be cleaned up and is kept as
	// recognised; Warning says why.
	Degraded bool   `json:"degraded,omitempty"`
	Warning  string `json:"warning,omitempty"`
	// Retried: Translate asked again, strictly.
	Retried          bool  `json:"retried,omitempty"`
	PromptTokens     int   `json:"promptTokens,omitempty"`
	CompletionTokens int   `json:"completionTokens,omitempty"`
	LatencyMillis    int64 `json:"latencyMillis"`
}

// SpeechRequest is one recording for a remote, OpenAI-compatible speech
// service (/v1/audio/transcriptions).
type SpeechRequest struct {
	// Path is a WAV file the app wrote.
	Path     string `json:"path"`
	Endpoint string `json:"endpoint,omitempty"`
	Model    string `json:"model,omitempty"`
	Key      string `json:"key,omitempty"`
	Language string `json:"language,omitempty"`
	Prompt   string `json:"prompt,omitempty"`
	// Terms become the request's hints.
	Terms []Term `json:"terms,omitempty"`
}

// SpeechResult is the recognised text.
type SpeechResult struct {
	Text          string `json:"text"`
	Language      string `json:"language,omitempty"`
	Model         string `json:"model,omitempty"`
	LatencyMillis int64  `json:"latencyMillis"`
}

// Preset is a language model service the settings offer.
type Preset struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	BaseURL      string   `json:"baseURL,omitempty"`
	KeyOptional  bool     `json:"keyOptional,omitempty"`
	Models       []string `json:"models,omitempty"`
	DefaultModel string   `json:"defaultModel,omitempty"`
	KeyURL       string   `json:"keyURL,omitempty"`
}

// TestResult is a connection test's.
type TestResult struct {
	LatencyMillis int64 `json:"latencyMillis"`
}

func terms(list []Term) []compose.Term {
	out := make([]compose.Term, 0, len(list))
	for _, t := range list {
		out = append(out, compose.Term{Term: t.Term, Aliases: t.Aliases})
	}
	return out
}

func snippets(list []Snippet) []snippet.Snippet {
	out := make([]snippet.Snippet, 0, len(list))
	for _, s := range list {
		out = append(out, snippet.Snippet{Trigger: s.Trigger, Text: s.Text})
	}
	return out
}
