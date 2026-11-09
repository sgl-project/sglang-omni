package main

import (
	"bytes"
	"encoding/json"
	"encoding/xml"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
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

// The previous Voxt's speech service setups (remoteASRProviderConfigurations):
// Doubao at its 2.0 address with its App ID and token, Aliyun at its
// Singapore address with a model this app has, StepFun at its own SSE
// address with a model this app lacks, a local Whisper server, Doubao's
// flash file recognition as a second Doubao (one setup per service: the
// first comes over), and a service never set up.
const legacySpeechJSON = `[
 {"providerID": "doubaoASR", "model": "volc.bigasr.sauc.duration", "endpoint": "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
  "storedCredentialPresence": {"apiKey": false, "appID": true, "accessToken": true}, "doubaoDictionaryMode": "requestScoped"},
 {"providerID": "aliyunBailianASR", "model": "qwen3-asr-flash-realtime", "endpoint": "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
  "storedCredentialPresence": {"apiKey": true}, "aliyunASRSettings": {"serverVADThreshold": 0.35}},
 {"providerID": "stepFunASR", "model": "stepaudio-2.5-asr", "endpoint": "https://api.stepfun.com/v1/audio/asr/sse", "storedCredentialPresence": {"apiKey": true}},
 {"providerID": "openAIWhisper", "model": "large-v3", "endpoint": "http://127.0.0.1:9000/v1/audio/transcriptions"},
 {"providerID": "doubaoASR", "model": "volc.bigasr.auc_turbo", "endpoint": "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"},
 {"providerID": "glmASR", "model": "", "endpoint": ""}
]`

// legacyFeaturesJSON is the previous Voxt's featureSettings: Transcribe on
// the OpenAI setup with the user's own prompt and Doubao, Translate on
// local models, Rewrite on Codex with its built-in prompt and Gemini (never
// set up) (testdata/legacy/prompts: the
// last version of its English rewrite prompt), which it saved when the
// preset was picked.
func legacyFeaturesJSON(t *testing.T, rewritePrompt string) string {
	t.Helper()
	b, err := json.Marshal(map[string]any{
		"transcription": map[string]string{"asrSelectionID": "remote-asr:doubaoASR", "llmSelectionID": "remote-llm:openAI", "prompt": "Keep my English terms.\n{{RAW_TRANSCRIPTION}}", "promptPresetID": "precise"},
		"translation": map[string]string{"asrSelectionID": "mlx:mlx-community/Qwen3-ASR-0.6B-4bit", "modelSelectionID": "local-llm:mlx-community/Qwen3.5-4B-OptiQ-4bit",
			"prompt": "", "promptPresetID": "precise"},
		"rewrite": map[string]string{"asrSelectionID": "remote-asr:googleGeminiASR", "llmSelectionID": "remote-llm:codex", "prompt": rewritePrompt, "promptPresetID": "strict"},
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
// root here ("" for its own addresses; Gemini's native address on its
// OpenAI-compatible path, to check), with its model and generation
// settings, and without the services that have no place here; the setups a
// feature used when they came over too; its own prompts without its
// transcript variable, and none of its built-in ones; its Qwen3-ASR context
// without its terms variable.
func TestParseLegacySettings(t *testing.T) {
	builtIn := builtInRewritePrompt(t)
	l := parseLegacySettings(legacySetupsJSON, legacySpeechJSON, legacyFeaturesJSON(t, builtIn), legacyTuningJSON)
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
		g.ResponseFormat != llm.ResponseJSONSchema || g.JSONSchema != `{"type": "object"}` || !strings.Contains(g.ExtraBodyJSON, `"num_ctx": 8192`) {
		t.Errorf("ollama %+v", ol)
	}
	for _, id := range []string{"deepseek", "omlx"} {
		if by[id].Endpoint != "" {
			t.Errorf("%s at %q, want the preset's address", id, by[id].Endpoint)
		}
	}
	if g := by["gemini"].Endpoint; g != "https://example.com/v1beta/openai" || strings.Join(l.checkLLM, ",") != "gemini" {
		t.Errorf("a custom Gemini address %q, to check %v", g, l.checkLLM)
	}
	if u := by["omlx"].Generation.IncludeUsage; u == nil || *u {
		t.Errorf("oMLX usage %v, want off as the previous Voxt had it", u)
	}
	if l.featureLLM != [3]string{"openai", "", ""} || l.featureSpeech != [3]string{"doubao", "", ""} || l.featureLocal != [3]string{"", models.ASRFast, ""} {
		t.Errorf("features' setups %q, speech %q, local speech %q", l.featureLLM, l.featureSpeech, l.featureLocal)
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

// A language model setup's address is replaced by the preset's only when
// it is its own service's default in the previous Voxt: an OpenAI setup at
// StepFun's address went to OpenAI, with the StepFun key the user pastes.
// An address of a service's own API this app does not speak stays, to
// check, instead of becoming the preset's.
func TestLegacyApplyGeneral(t *testing.T) {
	if _, err := os.Stat("/usr/bin/plutil"); err != nil {
		t.Skip("no plutil: the previous Voxt's preferences are read on macOS only")
	}
	lib := t.TempDir()
	prefs := filepath.Join(lib, "Preferences", legacyBundleID+".plist")
	if err := os.MkdirAll(filepath.Dir(prefs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(prefs, []byte(`<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict><key>showInDock</key><false/></dict></plist>`), 0o644); err != nil {
		t.Fatal(err)
	}
	voxtDir := filepath.Join(lib, "Application Support", "Voxt")
	if err := os.MkdirAll(voxtDir, 0o755); err != nil {
		t.Fatal(err)
	}
	s := settings.Defaults()
	if !legacyApplyGeneral(voxtDir, &s) || !s.General.HideDock || !s.General.MenuBar {
		t.Fatalf("hide dock: %+v", s.General)
	}
	if err := os.WriteFile(prefs, []byte(`<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict><key>showInDock</key><true/></dict></plist>`), 0o644); err != nil {
		t.Fatal(err)
	}
	s = settings.Defaults()
	if !legacyApplyGeneral(voxtDir, &s) || s.General.HideDock {
		t.Fatalf("show dock: %+v", s.General)
	}
}

func TestLegacyEndpoint(t *testing.T) {
	for _, c := range []struct {
		service, endpoint, want string
		check                   bool
	}{
		{"openAI", "https://api.stepfun.com/v1/chat/completions", "https://api.stepfun.com/v1", false},
		{"stepFun", "https://api.stepfun.com/v1/chat/completions", "", false},
		{"anthropic", "https://claude-proxy.example.com/v1/messages", "https://claude-proxy.example.com/v1", true},
		{"google", "https://gemini-proxy.example.com/gemini", "https://gemini-proxy.example.com/gemini", true},
	} {
		if got, check := legacyEndpoint(c.service, c.endpoint); got != c.want || check != c.check {
			t.Errorf("%s at %s: %q, check %v; want %q, %v", c.service, c.endpoint, got, check, c.want, c.check)
		}
	}
}

// The previous Voxt's speech setups as they come over: the service's own
// address for its own default (Doubao's 2.0, StepFun's SSE API), a regional
// address kept; a model this app has kept, one it lacks back to the
// service's default, to check, any model for OpenAI-compatible servers; one
// setup per service, the first; never-made setups left out. Another
// service's address and one this app's client cannot use are kept, never
// replaced by the service's own; the second is to check.
func TestParseLegacySpeech(t *testing.T) {
	l := parseLegacySettings("", legacySpeechJSON, "", "")
	want := []settings.Provider{
		{ID: "doubao", Model: "volc.bigasr.sauc.duration"},
		{ID: "aliyun", Endpoint: "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime", Model: "qwen3-asr-flash-realtime"},
		{ID: providers.StepFunASR},
		{ID: "openai", Endpoint: "http://127.0.0.1:9000/v1/audio/transcriptions", Model: "large-v3"},
	}
	if len(l.speech) != len(want) {
		t.Fatalf("speech setups %+v", l.speech)
	}
	for i, p := range l.speech {
		if p.ID != want[i].ID || p.Endpoint != want[i].Endpoint || p.Model != want[i].Model {
			t.Errorf("setup %d: %+v, want %+v", i, p, want[i])
		}
	}
	if strings.Join(l.checkSpeech, ",") != providers.StepFunASR {
		t.Errorf("to check %v, want StepFun (its model is not here)", l.checkSpeech)
	}
	speech := func(id string) providers.Speech { sp, _ := providers.LookupSpeech(id); return sp }
	for _, c := range []struct {
		service, id, model, endpoint, want string
		check                              bool
	}{
		{"openAIWhisper", "openai", "", "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions", "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions", false},
		{"doubaoASR", "doubao", "", "https://doubao-proxy.example.com/asr", "https://doubao-proxy.example.com/asr", true},
		{"doubaoASR", "doubao", "volc.bigasr.auc_turbo", "https://doubao-proxy.example.com/flash", "https://doubao-proxy.example.com/flash", false},
		{"stepFunASR", providers.StepFunASR, "", "https://step.example.com/v1/audio/asr/sse", "https://step.example.com/v1/audio/transcriptions", true},
	} {
		if got, check := legacySpeechEndpoint(c.service, speech(c.id), c.model, c.endpoint); got != c.want || check != c.check {
			t.Errorf("%s at %s: %q, check %v; want %q, %v", c.service, c.endpoint, got, check, c.want, c.check)
		}
	}
}

// Every speech service of the previous Voxt maps to one here.
func TestLegacySpeechServicesKnown(t *testing.T) {
	for old, id := range legacySpeechServices {
		if _, ok := providers.LookupSpeech(id); !ok {
			t.Errorf("%s maps to %q, which is not a speech service here", old, id)
		}
	}
}

// What the user set here already stays: a service set up here keeps its
// setup, a feature keeps its models and its rules; the rest comes over.
func TestLegacySettingsApply(t *testing.T) {
	l := parseLegacySettings(legacySetupsJSON, legacySpeechJSON, legacyFeaturesJSON(t, ""), legacyTuningJSON)
	s := settings.Defaults()
	s.Providers.LLM = []settings.Provider{{ID: "openai", Model: "mine"}}
	s.Providers.ASR = []settings.Provider{{ID: "aliyun", Model: "fun-asr-realtime"}}
	s.Features.Transcribe.Prompt = "My rules."
	llmAdded, speechAdded, prompts := l.apply(&s, "")
	ids := func(ps []settings.Provider) string {
		var out []string
		for _, p := range ps {
			out = append(out, p.ID)
		}
		return strings.Join(out, ",")
	}
	if ids(llmAdded) != "ollama,deepseek,gemini,omlx" || len(s.Providers.LLM) != 5 || s.Providers.LLM[0].Model != "mine" {
		t.Errorf("added %v, setups %+v", ids(llmAdded), s.Providers.LLM)
	}
	if ids(speechAdded) != "doubao,stepfun,openai" || len(s.Providers.ASR) != 4 || s.Providers.ASR[0].Model != "fun-asr-realtime" || s.Features.Transcribe.ASR != "doubao" {
		t.Errorf("speech added %v, setups %+v, transcribe %+v", ids(speechAdded), s.Providers.ASR, s.Features.Transcribe.Feature)
	}
	if s.Features.Transcribe.Prompt != "My rules." || s.Features.Transcribe.LLM != "openai" || prompts != 1 || s.Dictionary.SpeechContext != "Leo works on LLM serving." {
		t.Errorf("prompts %d, transcribe %+v, speech context %q", prompts, s.Features.Transcribe.Feature, s.Dictionary.SpeechContext)
	}
	if s.Features.Translate.ASR != models.ASRFast { // Voxt 1.x's local Qwen3-ASR 0.6B
		t.Errorf("translate %+v", s.Features.Translate.Feature)
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
		{"remoteASRProviderConfigurations", legacySpeechJSON},
		{"featureSettings", legacyFeaturesJSON(t, "")},
		{"mlxLocalASRTuningSettings", legacyTuningJSON},
		{"showInDock", "false"},
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
	if len(s.Providers.LLM) != 5 || s.Providers.LLM[0].Endpoint != "https://inference.example.com/v1" || len(s.Providers.ASR) != 4 || s.Features.Transcribe.ASR != "doubao" ||
		s.Features.Transcribe.Prompt != "Keep my English terms." || s.Dictionary.SpeechContext != "Leo works on LLM serving." ||
		!s.General.HideDock || !s.General.MenuBar {
		t.Fatalf("settings: setups %+v, transcribe %+v, speech context %q general %+v (report %q)", s.Providers.LLM, s.Features.Transcribe.Feature, s.Dictionary.SpeechContext, s.General, a.ob.legacy.text())
	}
	r := a.ob.legacy
	for _, want := range []string{"5 language model services", "4 speech services", "2 prompts"} {
		if !slices.Contains(r.imported, want) {
			t.Errorf("imported %q lacks %q", r.imported, want)
		}
	}
	// The services' keys stay behind: those that need one are to
	// complete (not Ollama, oMLX or the Whisper server on this Mac); so
	// are the setups to check.
	var todo []string
	for _, d := range r.todo {
		todo = append(todo, string(d.kind)+"/"+d.id)
	}
	for _, want := range []string{"llm/openai", "llm/deepseek", "llm/gemini", "asr/doubao", "asr/stepfun"} {
		if !slices.Contains(todo, want) {
			t.Errorf("to do %v lacks %s", todo, want)
		}
	}
	if slices.Contains(todo, "llm/ollama") || slices.Contains(todo, "llm/omlx") || !strings.Contains(r.text(), "Check ") {
		t.Errorf("to do %v (%q)", todo, r.text())
	}
}
