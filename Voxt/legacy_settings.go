package main

// The previous Voxt's settings that have a place here, read from its
// preferences (Library/Preferences/com.voxt.Voxt.plist) when its data is
// imported: its language model and speech service setups
// (settings.Providers), the setups each feature used, its features' own
// prompts (Your rules) and its Qwen3-ASR context (the speech context). A
// setup's key (and Doubao's App ID) stays behind: the
// previous Voxt kept it in the Data Protection Keychain under its own
// keychain group, which this app's signature cannot read, so the user
// pastes it again. What the user set here already stays.

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/url"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// legacyPreference is the previous Voxt's preference key as text, from
// its preferences beside voxtDir (its Application Support/Voxt): "" without
// plutil (the binary plist's reader), the file or the key.
func legacyPreference(voxtDir, key string) string {
	prefs := filepath.Join(voxtDir, "..", "..", "Preferences", legacyBundleID+".plist")
	out, err := exec.Command("/usr/bin/plutil", "-extract", key, "raw", "-o", "-", prefs).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// legacyLLMServices are this app's presets for the previous Voxt's
// language model services (its RemoteLLMProvider); its Codex, a ChatGPT
// sign-in, has none.
var legacyLLMServices = map[string]string{
	"anthropic": "anthropic", "google": "gemini", "openAI": "openai", "ollama": "ollama", "omlx": "omlx",
	"deepseek": "deepseek", "openrouter": "openrouter", "grok": "xai", "zai": "zhipu", "volcengine": "volcengine",
	"kimi": "moonshot", "lmStudio": "lmstudio", "minimax": "minimax", "aliyunBailian": "dashscope",
	"stepFun": "stepfun", "xiaomiMiMo": "mimo",
}

// legacyDefaultEndpoints are the previous Voxt's own address of each of
// its language model services, by its service id, as it saved it. Only a
// setup's own service's becomes the preset's address here: a setup of one
// service may point at another's (an OpenAI setup at StepFun).
var legacyDefaultEndpoints = map[string]string{
	"anthropic": "https://api.anthropic.com/v1/messages", "google": "https://generativelanguage.googleapis.com/v1beta/models",
	"openAI": "https://api.openai.com/v1/responses", "ollama": "http://localhost:11434", "omlx": "http://localhost:8000/v1",
	"deepseek": "https://api.deepseek.com", "openrouter": "https://openrouter.ai/api/v1/models", "grok": "https://api.x.ai/v1/models",
	"zai": "https://open.bigmodel.cn/api/paas/v4/models", "volcengine": "https://ark.cn-beijing.volces.com/api/v3/responses",
	"kimi": "https://api.moonshot.cn/v1/models", "lmStudio": "http://127.0.0.1:1234/v1/models",
	"minimax": "https://api.minimax.chat/v1/text/chatcompletion_v2", "aliyunBailian": "https://dashscope.aliyuncs.com/compatible-mode/v1/responses",
	"stepFun": "https://api.stepfun.com/v1/chat/completions", "xiaomiMiMo": "https://api.xiaomimimo.com/v1/chat/completions",
}

// legacyEndpoint is the address of a setup of the previous Voxt's language
// model service (its service id) as an API root here, and whether the user
// should check it. The service's own default is the preset's (""); any
// other address keeps its host, never replaced by the preset's, which
// would send the requests elsewhere. It loses the request path it ended
// in; Ollama's native API becomes its OpenAI-compatible /v1. The previous
// Voxt spoke Anthropic's, Gemini's and MiniMax's own APIs, which this app
// does not: their other addresses are kept (Gemini's native /v1beta/models
// becomes /v1beta/openai on the same host), to check.
func legacyEndpoint(service, endpoint string) (root string, check bool) {
	id := legacyLLMServices[service]
	e := strings.TrimRight(strings.TrimSpace(endpoint), "/")
	if e == "" || e == legacyDefaultEndpoints[service] {
		return "", false
	}
	paths := []string{"/chat/completions", "/responses", "/models", "/messages", "/text/chatcompletion_v2"}
	switch id {
	case "gemini":
		paths = nil
		if base, ok := strings.CutSuffix(e, "/models"); ok {
			e = base + "/openai"
		}
	case "ollama":
		paths = []string{"/chat/completions", "/api/chat", "/api/generate", "/api/tags", "/api"}
	}
	for _, path := range paths {
		if base, ok := strings.CutSuffix(e, path); ok {
			e = base
			break
		}
	}
	if id == "ollama" && !strings.HasSuffix(e, "/v1") {
		e += "/v1"
	}
	if p, ok := llm.LookupPreset(id); ok && strings.TrimRight(p.BaseURL, "/") == e {
		return "", false
	}
	return e, id == "anthropic" || id == "gemini" || id == "minimax"
}

// legacyLLMSetup is a language model setup as the previous Voxt saved it
// (remoteLLMProviderConfigurations, a JSON array in a string), without
// its key.
type legacyLLMSetup struct {
	ProviderID, Model, Endpoint   string
	StoredCredentialPresence      struct{ APIKey, AccessToken bool }
	OpenAITextVerbosity           string
	OllamaKeepAlive               string
	OllamaJSONSchema              string
	OmlxJSONSchema                string
	OmlxIncludeUsageStreamOptions bool
	Generation                    struct {
		MaxOutputTokens, TopK, TopLogprobs                   *int
		Temperature, TopP, MinP                              *float64
		PresencePenalty, FrequencyPenalty, RepetitionPenalty *float64
		Seed                                                 *int64
		Stop                                                 []string
		Logprobs                                             bool
		ResponseFormat                                       string
		Thinking                                             struct {
			Mode, Effort string
			BudgetTokens int
		}
		ExtraBodyJSON, ExtraOptionsJSON string
	} `json:"generationSettings"`
}

// provider is s as a setup here, and whether the user should check its
// address (legacyEndpoint); ok is false for a service without a preset
// here and for a setup never made (no model, address or key).
func (s legacyLLMSetup) provider() (p settings.Provider, check, ok bool) {
	id, ok := legacyLLMServices[s.ProviderID]
	keyed := s.StoredCredentialPresence.APIKey || s.StoredCredentialPresence.AccessToken
	if !ok || strings.TrimSpace(s.Model) == "" && strings.TrimSpace(s.Endpoint) == "" && !keyed {
		return settings.Provider{}, false, false
	}
	g := s.Generation
	gen := settings.Generation{
		MaxOutputTokens: g.MaxOutputTokens, Temperature: g.Temperature, TopP: g.TopP, TopK: g.TopK, MinP: g.MinP, Seed: g.Seed,
		Stop: g.Stop, PresencePenalty: g.PresencePenalty, FrequencyPenalty: g.FrequencyPenalty, RepetitionPenalty: g.RepetitionPenalty,
		Logprobs: g.Logprobs, TopLogprobs: g.TopLogprobs, KeepAlive: strings.TrimSpace(s.OllamaKeepAlive),
		OptionsJSON: strings.TrimSpace(g.ExtraOptionsJSON), ExtraBodyJSON: strings.TrimSpace(g.ExtraBodyJSON),
	}
	switch g.ResponseFormat {
	case llm.ResponseJSON:
		gen.ResponseFormat = g.ResponseFormat
	case llm.ResponseJSONSchema:
		gen.ResponseFormat, gen.JSONSchema = g.ResponseFormat, cmp.Or(s.OllamaJSONSchema, s.OmlxJSONSchema)
	}
	switch t := g.Thinking; t.Mode {
	case llm.ThinkOff, llm.ThinkOn:
		gen.Thinking = t.Mode
	case llm.ThinkEffort:
		gen.Thinking, gen.ThinkingEffort = t.Mode, t.Effort
	case llm.ThinkBudget:
		gen.Thinking, gen.ThinkingBudget = t.Mode, t.BudgetTokens
	}
	if s.OpenAITextVerbosity != "automatic" {
		gen.Verbosity = s.OpenAITextVerbosity
	}
	if id == "omlx" { // off was its default; on is this app's
		usage := s.OmlxIncludeUsageStreamOptions
		gen.IncludeUsage = &usage
	}
	endpoint, check := legacyEndpoint(s.ProviderID, s.Endpoint)
	return settings.Provider{ID: id, Endpoint: endpoint, Model: strings.TrimSpace(s.Model), Generation: gen}, check, true
}

// legacySpeechServices are this app's speech services for the previous
// Voxt's (its RemoteASRProvider): all seven are still here.
var legacySpeechServices = map[string]string{
	"openAIWhisper": "openai", "doubaoASR": "doubao", "glmASR": providers.ZhipuASR, "aliyunBailianASR": "aliyun",
	"stepFunASR": providers.StepFunASR, "xiaomiMiMoASR": "mimo", "googleGeminiASR": "gemini",
}

// legacyDefaultSpeechEndpoints are the previous Voxt's own addresses of
// each of its speech services, by its service id (Doubao's 1.0, 2.0 and
// file recognition ones, Aliyun's for each protocol): only a setup's own
// service's becomes the service's own address here, which picks it by
// model.
var legacyDefaultSpeechEndpoints = map[string][]string{
	"openAIWhisper": {"https://api.openai.com/v1/audio/transcriptions"},
	"doubaoASR": {"wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async", "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel",
		"https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"},
	"glmASR":           {"https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"},
	"aliyunBailianASR": {"wss://dashscope.aliyuncs.com/api-ws/v1/inference", "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"},
	"stepFunASR":       {"https://api.stepfun.com/v1/audio/asr/sse"},
	"xiaomiMiMoASR":    {"https://api.xiaomimimo.com/v1/chat/completions"},
	"googleGeminiASR":  {"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"},
}

// legacySpeechEndpoint is the address of a setup of the previous Voxt's
// speech service (its service id; sp here) for model as one here, and
// whether the user should check it. The service's own default is its own here ("");
// any other address is kept, never replaced by the service's, which would
// send the audio elsewhere: to check when this app's client cannot use it
// (a streaming service's https://, a file service's wss://). StepFun's
// was its SSE API; here its transcriptions on the same host, to check.
func legacySpeechEndpoint(service string, sp providers.Speech, model, endpoint string) (address string, check bool) {
	e := strings.TrimRight(strings.TrimSpace(endpoint), "/")
	if e == "" || slices.Contains(legacyDefaultSpeechEndpoints[service], e) {
		return "", false
	}
	if root, ok := strings.CutSuffix(e, "/audio/asr/sse"); ok && sp.ID == providers.StepFunASR {
		e, check = root+"/audio/transcriptions", true
	}
	if e == sp.Endpoint {
		return "", false
	}
	u, err := url.Parse(e)
	schemes := []string{"https", "http"}
	if sp.Streams(model) {
		schemes = []string{"wss", "ws"}
	}
	return e, check || err != nil || u.Host == "" || !slices.Contains(schemes, strings.ToLower(u.Scheme))
}

// legacySpeechSetup is a speech service setup as the previous Voxt saved
// it (remoteASRProviderConfigurations), without its key and App ID. Its
// services' own options (Aliyun's VAD, Doubao's dictionary modes, OpenAI's
// pseudo-realtime chunks) have no place here.
type legacySpeechSetup struct {
	ProviderID, Model, Endpoint string
	StoredCredentialPresence    struct{ APIKey, AppID, AccessToken bool }
}

// provider is s as a setup here, and whether the user should check it
// (its address, legacySpeechEndpoint, or its model); ok is false for a
// setup never made (no model, address or credential). Its model comes over
// when this app's client of the service takes it (OpenAI's takes any: a
// compatible server names its own models), else the service's default
// does, to check.
func (s legacySpeechSetup) provider() (p settings.Provider, check, ok bool) {
	sp, ok := providers.LookupSpeech(legacySpeechServices[s.ProviderID])
	c := s.StoredCredentialPresence
	model := strings.TrimSpace(s.Model)
	if !ok || model == "" && strings.TrimSpace(s.Endpoint) == "" && !c.APIKey && !c.AppID && !c.AccessToken {
		return settings.Provider{}, false, false
	}
	endpoint, check := legacySpeechEndpoint(s.ProviderID, sp, model, s.Endpoint)
	if sp.ID != "openai" && model != "" && !slices.Contains(sp.Models, model) {
		model, check = "", true
	}
	return settings.Provider{ID: sp.ID, Endpoint: endpoint, Model: model}, check, true
}

// legacyBuiltInPrompts are the SHA-256 digests of the previous Voxt's
// built-in prompts, by kind: every version of its prompt files in each of
// its languages, and the older defaults it recognised itself. It saved a
// preset's text as the feature's prompt when one was picked, so a saved
// prompt that is one of them is not the user's own.
var legacyBuiltInPrompts = map[string][]string{
	"transcription": {
		"02741c386fa9ced8d2158e7cf4c6104b15013b7be272a061da55343ba2e69acd", "1653d67b2661988289508482c4d9ecbdef404251308fe38140e1a737248893c5",
		"21c438a9a05a34699dbb5199f715e62086f2746858b846fdbe2e6f51e735cfe0", "27ad68f2641b5c74a86d790cd641819004d59002d82f33e5d17b4d8e272997b3",
		"30fc6631aeeedb02325d1a2dbb0f0d2d93cc24c7b5654970c688d119c452cf6e", "360bd4e764fb6bd1ea4080e170655e31d240b2beff600189446c6316a4ddc02c",
		"55fa7253c76d272e011c40d7775cfa1bd62c99ff6edd56a2cc04e2930fc32008", "57b7358c481764cc653b82b38034d5bc51425464437e126704dfc94a2a37f459",
		"597df6bba09df1a90f6e89513f2f85c82f453b17c9bc4cbbe894c1296521f00c", "63ac6dc353ce8e7e5902f3fa5383c7d23df2b4460461dbc9b749218d42795abf",
		"6ae4ab890cc3fc5b9d0c147953c2067bada9934ef80df8de2aa498e0f41095a7", "74fcfe75cd2ea3721d8c7030485a7ec8d393f168acac3a9e4f4d0a68942434a3",
		"a6791751db82f240ceb9fa7f05c65d06a0b74ca26965a4b4fa3040ea518d220c", "b9f9680a04c0bcb363170cde0eccbd7e84f0f5a3a5f7e19066ce5eac0b1349ad",
		"d0426c874bf54ab9d4cc6d4947d348b78fc158eed8564c7fbe60496c11f25f3b", "efdff73d5d2cb1c31b8bdf4dcddd2f43c0c0652c646ccf557f3427eb897a192b",
	},
	"translation": {
		"0a00a63561f0670abbdd5dc57efd96829124fd68cb1f691539c93cfa68f3373f", "18f5d0be6a61366d16508f5a31f3b3576141e4feba9b18d0a96d007d1958e416",
		"3cfccb8fc1e8f591adf4d1da8e149da1ab871e462302f058e5a56033e638c10b", "509f3be5d697a01104b691fd99d23be6db463e99da3e0c0492fe822962a68b9b",
		"585d0f4bf8a21d55aa669c5b16aa7c7ef77adb2417acbc88c85c36a85e8b3aec", "7949566e9e8848e1989800a593c719eedea2a7934d015dc0244aadc63cdca7f4",
		"7bbe13f7c2b445e0cca4420947e2b8e3b709bb9f596095ee72cee8c51c31d680", "998a0a8a67786a9ab76ca5ff81b46b43b984f32cbfce5b5c22fa938da9f89cdc",
		"a0ceeb27b2352b08c98065c24bdd68b8127913aa9a2516de3c290de02ffbc32d", "bd4babdeec2dcadff63778f1ebadf385e85a94dc7104fb74cb7bbae13d3dbcbb",
		"cd193cee3027a3ba4a6f24dd548c268fb177b76c86ae6abca27c5e70f70cb9fd", "d25406d98c9c6862f8193a9f892dd0425c16c43ef980653c607c6f955a626703",
		"e0319e193a842e00609a9e9ba4212013bbe7a909a2974f2fbe5426e21448f566", "e43c49c82fdce5f0190dfcc5d0859ff7f23d18a73d46f8f0e2d3437455e295a0",
		"f59232f69bf85ac67b7deb0ff30d5e529363a4589c662fec4184efd921e9d0ca",
	},
	"rewrite": {
		"09f35f472123a67b863f54e0d8f9429a097047060cbe82b97145efa85b4bb03d", "5b3bb1cd6779e3ca43b727fb8e91066c749ac4c9b87dce66f1da9606b8ad3bb5",
		"5ed3a58145fa218cac67277b95042d84a4d153eb30de7e411668555d4378972f", "821bb73ad9372d231078bee8ac4d8ab8223c5c4010cce0d8fc25510ca12696cb",
		"896482a35261df59cd0e37f50265e7ca9b28b942796f250381bf26d5bbbf0b74", "a7b3cfe434a9eae896d63e2c51f56bc77402170e3398bb4138acab45864f85d2",
		"bc2ffc14b264b3ef5db47990855e7a07ecb4c133c69f7ed3011629a1fdb72ec9", "c125f2c3a51684fa9c593b448ffbcadb3e807f12316e0e50930d24362d3303ca",
		"d662ecb5d34cada98d06f112cd51b7bd299b5114f5531a5b8d21a0a18d08c4d1",
	},
	"context": {
		"857db2aa4d9011bf501af532934cc572084de0c2473f8be15de6d7ef094ae963", "cf09f95b46e85461983705807b793be659b6d277a1d58d117a4dda3425e8122f",
		"faa073ae8ad5b56d7ae66b78eee186ebc198f0490235cea6ad7e7f6a1dcdb8de", "fe4f1c8dd5529ae266e3f4df3d5773ec1d799855eb8736bb3c6e0f5422377497",
	},
}

// legacyOwnPrompt is a prompt the previous Voxt saved, as rules here: ""
// for none and for one of its built-in ones of kind. Its transcript
// variable goes: the input is never in the rules here.
func legacyOwnPrompt(kind, prompt string) string {
	p := strings.TrimSpace(prompt)
	sum := sha256.Sum256([]byte(p))
	if p == "" || slices.Contains(legacyBuiltInPrompts[kind], hex.EncodeToString(sum[:])) {
		return ""
	}
	return strings.TrimSpace(strings.ReplaceAll(p, "{{RAW_TRANSCRIPTION}}", ""))
}

// legacyContextMarkers are phrases that together mark an older built-in
// Qwen3-ASR context of the previous Voxt, one set per language (its
// matchesLegacyQwenASRContextBiasText): those were saved with the user's
// languages filled in, so no digest knows them.
var legacyContextMarkers = [][]string{
	{"The speaker's primary language is", "Other commonly used languages", "Bias recognition toward correct spelling", "Prefer these dictionary terms"},
	{"说话者的主要语言", "其他常用语言", "请将识别偏向于", "词典词汇"},
	{"話者の主要言語", "その他のよく使う言語", "認識を寄せてください", "辞書語"},
}

// legacyOwnContext is the previous Voxt's Qwen3-ASR context as the speech
// context here: "" for none and for a built-in one. Its terms variable
// goes: the dictionary's terms follow the context here anyway.
func legacyOwnContext(context string) string {
	c := legacyOwnPrompt("context", context)
	for _, markers := range legacyContextMarkers {
		if !slices.ContainsFunc(markers, func(m string) bool { return !strings.Contains(c, m) }) {
			return ""
		}
	}
	return strings.TrimSpace(strings.ReplaceAll(c, "{{DICTIONARY_TERMS}}", ""))
}

// legacySettings are the previous Voxt's settings the import brings.
type legacySettings struct {
	llm, speech []settings.Provider
	// featureLLM, featureSpeech and prompts are by feature: Transcribe,
	// Translate, Rewrite. featureLLM and featureSpeech are the setups the
	// feature used ("": a local model or none), prompts its own prompt.
	featureLLM, featureSpeech, prompts [3]string
	speechContext                      string
	// checkLLM and checkSpeech are the setups whose address or model may
	// not work here as it did (by id): the result says so.
	checkLLM, checkSpeech []string
}

// parseLegacySettings reads the previous Voxt's settings from its
// preferences remoteLLMProviderConfigurations (llmSetups),
// remoteASRProviderConfigurations (speechSetups), featureSettings
// (features) and mlxLocalASRTuningSettings (tuning): each a JSON document
// in a string, "" when it lacks one.
func parseLegacySettings(llmSetups, speechSetups, features, tuning string) legacySettings {
	var l legacySettings
	has := func(list []settings.Provider, id string) bool {
		return slices.ContainsFunc(list, func(p settings.Provider) bool { return p.ID == id })
	}
	var savedLLM []legacyLLMSetup
	json.Unmarshal([]byte(llmSetups), &savedLLM) // not JSON: no setups
	for _, s := range savedLLM {
		if p, check, ok := s.provider(); ok && !has(l.llm, p.ID) {
			l.llm = append(l.llm, p)
			if check {
				l.checkLLM = append(l.checkLLM, p.ID)
			}
		}
	}
	var savedSpeech []legacySpeechSetup
	json.Unmarshal([]byte(speechSetups), &savedSpeech)
	for _, s := range savedSpeech {
		if p, check, ok := s.provider(); ok && !has(l.speech, p.ID) {
			l.speech = append(l.speech, p)
			if check {
				l.checkSpeech = append(l.checkSpeech, p.ID)
			}
		}
	}
	var f struct {
		Transcription struct{ ASRSelectionID, LLMSelectionID, Prompt string }
		Translation   struct{ ASRSelectionID, ModelSelectionID, Prompt string }
		Rewrite       struct{ ASRSelectionID, LLMSelectionID, Prompt string }
	}
	json.Unmarshal([]byte(features), &f)
	// used is the setup here of a feature's remote selection ("remote-llm:openAI").
	used := func(selection, prefix string, services map[string]string, list []settings.Provider) string {
		service, remote := strings.CutPrefix(selection, prefix)
		if id := services[service]; remote && has(list, id) {
			return id
		}
		return ""
	}
	l.featureLLM = [3]string{
		used(f.Transcription.LLMSelectionID, "remote-llm:", legacyLLMServices, l.llm),
		used(f.Translation.ModelSelectionID, "remote-llm:", legacyLLMServices, l.llm),
		used(f.Rewrite.LLMSelectionID, "remote-llm:", legacyLLMServices, l.llm),
	}
	l.featureSpeech = [3]string{
		used(f.Transcription.ASRSelectionID, "remote-asr:", legacySpeechServices, l.speech),
		used(f.Translation.ASRSelectionID, "remote-asr:", legacySpeechServices, l.speech),
		used(f.Rewrite.ASRSelectionID, "remote-asr:", legacySpeechServices, l.speech),
	}
	l.prompts = [3]string{
		legacyOwnPrompt("transcription", f.Transcription.Prompt),
		legacyOwnPrompt("translation", f.Translation.Prompt),
		legacyOwnPrompt("rewrite", f.Rewrite.Prompt),
	}
	var t struct {
		Qwen3ASR struct{ QwenContextBias string }
	}
	json.Unmarshal([]byte(tuning), &t)
	l.speechContext = legacyOwnContext(t.Qwen3ASR.QwenContextBias)
	return l
}

// apply puts l into s where s has nothing of its own: a service set up
// here, a feature's chosen models and rules, and the speech context stay.
// It returns the language model and speech setups it added and the
// prompts it filled.
func (l legacySettings) apply(s *settings.Settings) (llmAdded, speechAdded []settings.Provider, prompts int) {
	add := func(list *[]settings.Provider, from []settings.Provider) (added []settings.Provider) {
		for _, p := range from {
			if !slices.ContainsFunc(*list, func(q settings.Provider) bool { return q.ID == p.ID }) {
				*list = append(*list, p)
				added = append(added, p)
			}
		}
		return added
	}
	llmAdded, speechAdded = add(&s.Providers.LLM, l.llm), add(&s.Providers.ASR, l.speech)
	for i, f := range []*settings.Feature{&s.Features.Transcribe.Feature, &s.Features.Translate.Feature, &s.Features.Rewrite.Feature} {
		if f.LLM == "" {
			f.LLM = l.featureLLM[i]
		}
		if f.ASR == "" {
			f.ASR = l.featureSpeech[i]
		}
		if f.Prompt == "" && l.prompts[i] != "" {
			f.Prompt = l.prompts[i]
			prompts++
		}
	}
	if s.Dictionary.SpeechContext == "" && l.speechContext != "" {
		s.Dictionary.SpeechContext = l.speechContext
		prompts++
	}
	return llmAdded, speechAdded, prompts
}

// toCheck names the setups added (apply's llmAdded, speechAdded) whose
// address or model the user should check, for the result.
func (l legacySettings) toCheck(llmAdded, speechAdded []settings.Provider) (llmCheck, speechCheck []settings.Provider) {
	for _, p := range llmAdded {
		if slices.Contains(l.checkLLM, p.ID) {
			llmCheck = append(llmCheck, p)
		}
	}
	for _, p := range speechAdded {
		if slices.Contains(l.checkSpeech, p.ID) {
			speechCheck = append(speechCheck, p)
		}
	}
	return llmCheck, speechCheck
}
