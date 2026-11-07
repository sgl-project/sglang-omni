package main

import (
	"bytes"
	"encoding/json"
	"encoding/xml"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// The previous Voxt's language model setups as it saved them
// (remoteLLMProviderConfigurations): an OpenAI-compatible server at a
// custom address, an Ollama on another Mac at its native address, three at
// the previous Voxt's own addresses (one of them Gemini's native API), a
// Codex sign-in and a service never set up.
const legacySetupsJSON = `[
 {"providerID": "openAI", "model": "kimi-k3", "endpoint": "https://inference.example.com/v1/chat/completions",
  "storedCredentialPresence": {"apiKey": true, "appID": false, "accessToken": false}, "openAITextVerbosity": "automatic",
  "generationSettings": {"maxOutputTokens": 131072, "temperature": 0, "stop": [], "logprobs": false, "responseFormat": "plain",
   "thinking": {"mode": "effort", "effort": "none", "exposeReasoning": false}, "extraBodyJSON": "", "extraOptionsJSON": ""}},
 {"providerID": "ollama", "model": "qwen3", "endpoint": "http://192.168.1.20:11434/api/chat", "ollamaKeepAlive": "10m",
  "ollamaJSONSchema": "{\"type\": \"object\"}",
  "generationSettings": {"responseFormat": "jsonSchema", "thinking": {"mode": "off"}, "extraOptionsJSON": "{\"num_ctx\": 8192}"}},
 {"providerID": "deepseek", "model": "deepseek-chat", "endpoint": "https://api.deepseek.com", "storedCredentialPresence": {"apiKey": true}},
 {"providerID": "google", "model": "gemini-2.5-flash", "endpoint": "https://example.com/v1beta/models", "storedCredentialPresence": {"apiKey": true}},
 {"providerID": "omlx", "model": "m", "endpoint": "http://localhost:8000/v1", "omlxIncludeUsageStreamOptions": false},
 {"providerID": "codex", "model": "gpt-5-codex"},
 {"providerID": "kimi", "model": "", "endpoint": ""}
]`

// legacyFeaturesJSON is the previous Voxt's featureSettings: Transcribe on
// the OpenAI setup with the user's own prompt, Translate on a local model,
// Rewrite on Codex with its built-in prompt (testdata/legacy/prompts: the
// last version of its English rewrite prompt), which it saved when the
// preset was picked.
func legacyFeaturesJSON(t *testing.T, rewritePrompt string) string {
	t.Helper()
	b, err := json.Marshal(map[string]any{
		"transcription": map[string]string{"llmSelectionID": "remote-llm:openAI", "prompt": "Keep my English terms.\n{{RAW_TRANSCRIPTION}}", "promptPresetID": "precise"},
		"translation":   map[string]string{"modelSelectionID": "local-llm:mlx-community/Qwen3.5-4B-OptiQ-4bit", "prompt": "", "promptPresetID": "precise"},
		"rewrite":       map[string]string{"llmSelectionID": "remote-llm:codex", "prompt": rewritePrompt, "promptPresetID": "strict"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

const legacyTuningJSON = `{"qwen3ASR": {"preset": "balanced", "qwenContextBias": "Leo works on LLM serving.\n{{DICTIONARY_TERMS}}"}}`

func builtInRewritePrompt(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(legacyFixtures, "prompts", "en-rewrite.txt"))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// The previous Voxt's settings as they come over: each setup at an API
// root here ("" for its own addresses), with its model and generation
// settings, and without the services that have no place here; the setup a
// feature used when it came over too; its own prompts without its
// transcript variable, and none of its built-in ones; its Qwen3-ASR context
// without its terms variable.
func TestParseLegacySettings(t *testing.T) {
	builtIn := builtInRewritePrompt(t)
	l := parseLegacySettings(legacySetupsJSON, legacyFeaturesJSON(t, builtIn), legacyTuningJSON)
	var ids []string
	by := map[string]settings.Provider{}
	for _, p := range l.llm {
		ids = append(ids, p.ID)
		by[p.ID] = p
	}
	if got := strings.Join(ids, ","); got != "openai,ollama,deepseek,gemini,omlx" {
		t.Fatalf("setups %s", got)
	}
	o := by["openai"]
	if g := o.Generation; o.Endpoint != "https://inference.example.com/v1" || o.Model != "kimi-k3" || *g.MaxOutputTokens != 131072 ||
		g.Temperature == nil || *g.Temperature != 0 || g.Thinking != llm.ThinkEffort || g.ThinkingEffort != "none" || g.Verbosity != "" {
		t.Errorf("openai %+v", o)
	}
	ol := by["ollama"]
	if g := ol.Generation; ol.Endpoint != "http://192.168.1.20:11434/v1" || g.KeepAlive != "10m" || g.Thinking != llm.ThinkOff ||
		g.ResponseFormat != llm.ResponseJSONSchema || g.JSONSchema != `{"type": "object"}` || g.OptionsJSON != `{"num_ctx": 8192}` {
		t.Errorf("ollama %+v", ol)
	}
	for _, id := range []string{"deepseek", "gemini", "omlx"} {
		if by[id].Endpoint != "" {
			t.Errorf("%s at %q, want the preset's address", id, by[id].Endpoint)
		}
	}
	if u := by["omlx"].Generation.IncludeUsage; u == nil || *u {
		t.Errorf("oMLX usage %v, want off as the previous Voxt had it", u)
	}
	if l.featureLLM != [3]string{"openai", "", ""} {
		t.Errorf("features' setups %q", l.featureLLM)
	}
	if l.prompts != [3]string{"Keep my English terms.", "", ""} {
		t.Errorf("prompts %q", l.prompts)
	}
	if l.speechContext != "Leo works on LLM serving." {
		t.Errorf("speech context %q", l.speechContext)
	}
	if got := legacyOwnPrompt("rewrite", builtIn+"\nSign mails with Leo."); !strings.HasSuffix(got, "Sign mails with Leo.") {
		t.Errorf("the built-in rewrite prompt changed by the user was dropped: %q", got)
	}
	for _, context := range []string{"{{DICTIONARY_TERMS}}", "说话者的主要语言是中文。其他常用语言：英语。请将识别偏向于正确拼写。词典词汇：Voxt"} {
		if got := legacyOwnContext(context); got != "" {
			t.Errorf("a built-in context came over: %q", got)
		}
	}
}

// What the user set here already stays: a service set up here keeps its
// setup, a feature keeps its model and its rules; the rest comes over.
func TestLegacySettingsApply(t *testing.T) {
	l := parseLegacySettings(legacySetupsJSON, legacyFeaturesJSON(t, ""), legacyTuningJSON)
	s := settings.Defaults()
	s.Providers.LLM = []settings.Provider{{ID: "openai", Model: "mine"}}
	s.Features.Transcribe.Prompt = "My rules."
	added, prompts := l.apply(&s)
	var ids []string
	for _, p := range added {
		ids = append(ids, p.ID)
	}
	if strings.Join(ids, ",") != "ollama,deepseek,gemini,omlx" || len(s.Providers.LLM) != 5 || s.Providers.LLM[0].Model != "mine" {
		t.Errorf("added %v, setups %+v", ids, s.Providers.LLM)
	}
	if s.Features.Transcribe.Prompt != "My rules." || s.Features.Transcribe.LLM != "openai" || prompts != 1 || s.Dictionary.SpeechContext != "Leo works on LLM serving." {
		t.Errorf("prompts %d, transcribe %+v, speech context %q", prompts, s.Features.Transcribe.Feature, s.Dictionary.SpeechContext)
	}
}

// The import reads those settings from the previous Voxt's binary
// preferences beside its database (plutil), and says what came over.
func TestImportLegacySettings(t *testing.T) {
	if _, err := os.Stat("/usr/bin/plutil"); err != nil {
		t.Skip("no plutil: the previous Voxt's preferences are read on macOS only")
	}
	lib := copyFixture(t, "real")
	var plist bytes.Buffer
	plist.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n" + `<plist version="1.0"><dict>`)
	for _, kv := range [][2]string{
		{"remoteLLMProviderConfigurations", legacySetupsJSON},
		{"featureSettings", legacyFeaturesJSON(t, "")},
		{"mlxLocalASRTuningSettings", legacyTuningJSON},
	} {
		plist.WriteString("<key>" + kv[0] + "</key><string>")
		xml.EscapeText(&plist, []byte(kv[1]))
		plist.WriteString("</string>")
	}
	plist.WriteString("</dict></plist>")
	prefs := filepath.Join(lib, "Preferences", legacyBundleID+".plist")
	if err := os.MkdirAll(filepath.Dir(prefs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(prefs, plist.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	a, _, _ := newTestApp(t)
	a.importLegacyFrom(lib)
	s := a.lib.settings.Get()
	if len(s.Providers.LLM) != 5 || s.Providers.LLM[0].Endpoint != "https://inference.example.com/v1" ||
		s.Features.Transcribe.Prompt != "Keep my English terms." || s.Dictionary.SpeechContext != "Leo works on LLM serving." {
		t.Fatalf("settings: setups %+v, transcribe %+v, speech context %q (message %q)", s.Providers.LLM, s.Features.Transcribe.Feature, s.Dictionary.SpeechContext, a.ob.legacyMsg)
	}
	for _, want := range []string{"5 language model services came over", "2 prompts came over"} {
		if !strings.Contains(a.ob.legacyMsg, want) {
			t.Errorf("message %q lacks %q", a.ob.legacyMsg, want)
		}
	}
}
