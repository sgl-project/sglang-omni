// Package credentials shares remote-service credentials between family
// apps (spec §8): one login-Keychain item per vendor account (service
// "ActNow", account "vendor.<id>", written with /usr/bin/security), and
// credentials.json for what is not secret (a hint, a fingerprint, which
// apps use it). An app reads a shared secret only after the user agreed
// in that app ("Use this key"), which adds it to the vendor's users.
package credentials

import (
	"net/url"
	"strings"
)

// Auth kinds of a vendor's credential.
const (
	AuthAPIKey           = "apiKey"
	AuthAppIDAccessToken = "appId+accessToken" // Doubao speech: the App ID is not secret
	AuthToken            = "token"             // Hugging Face access token
)

// Legacy is a secret an app kept under its own service before the spec
// ("AudioSmith" / "llm.deepseek"). Migration copies it to the shared item
// once the user agrees.
type Legacy struct {
	App     string // app id
	Service string // Keychain service: the app's name
	Account string
}

// Vendor is a row of the vendor table (spec §8.1): one account on one
// platform, whatever an app uses it for (ASR, LLM).
type Vendor struct {
	ID     string
	Name   string // English display name
	NameZH string // Chinese display name ("" = Name)
	Auth   string
	Legacy []Legacy
	Note   string
}

// Label is the Keychain item's label: "ActNow DeepSeek".
func (v Vendor) Label() string { return "ActNow " + v.Name }

func llm(app, id string) Legacy {
	svc := map[string]string{"audiosmith": "AudioSmith", "auralink": "Auralink", "dicta": "Dicta"}[app]
	return Legacy{App: app, Service: svc, Account: id}
}

// Vendors is the vendor table. New vendors only add rows.
var Vendors = []Vendor{
	{ID: "openai", Name: "OpenAI", Auth: AuthAPIKey, Legacy: []Legacy{llm("audiosmith", "llm.openai"), llm("auralink", "llm.openai"), llm("auralink", "asr.openai")},
		Note: "Auralink's OpenAI-compatible ASR with a custom endpoint is not this vendor."},
	{ID: "anthropic", Name: "Anthropic", Auth: AuthAPIKey, Legacy: []Legacy{llm("audiosmith", "llm.anthropic"), llm("auralink", "llm.anthropic")}},
	{ID: "deepseek", Name: "DeepSeek", Auth: AuthAPIKey, Legacy: []Legacy{llm("audiosmith", "llm.deepseek"), llm("auralink", "llm.deepseek")}},
	{ID: "dashscope", Name: "Alibaba Cloud Model Studio (China)", NameZH: "阿里云百炼（中国站）", Auth: AuthAPIKey,
		Legacy: []Legacy{llm("audiosmith", "llm.dashscope"), llm("auralink", "llm.dashscope"), llm("auralink", "asr.aliyun")},
		Note:   "One DashScope key serves both ASR and LLM."},
	{ID: "dashscope-intl", Name: "Alibaba Cloud Model Studio (International)", NameZH: "阿里云百炼（国际站）", Auth: AuthAPIKey,
		Legacy: []Legacy{llm("audiosmith", "llm.dashscope-intl"), llm("auralink", "llm.dashscope-intl")}},
	{ID: "stepfun", Name: "StepFun", NameZH: "阶跃星辰", Auth: AuthAPIKey, Legacy: []Legacy{llm("auralink", "llm.stepfun"), llm("auralink", "asr.stepfun")}},
	{ID: "volcengine-ark", Name: "Volcano Engine Ark", NameZH: "火山方舟", Auth: AuthAPIKey, Legacy: []Legacy{llm("auralink", "llm.volcengine")},
		Note: "Not the same credential as Doubao speech."},
	{ID: "volcengine-speech", Name: "Doubao Speech", NameZH: "豆包语音", Auth: AuthAPIKey, Legacy: []Legacy{llm("auralink", "asr.doubao")},
		Note: "Either an API key, or an App ID (not secret, kept in credentials.json) and an access token."},
	{ID: "moonshot", Name: "Kimi (moonshot.cn)", NameZH: "Kimi（中国站）", Auth: AuthAPIKey, Legacy: []Legacy{llm("audiosmith", "llm.moonshot"), llm("auralink", "llm.moonshot")},
		Note: "AudioSmith's llm.moonshot served both sites: its base URL decides (VendorForLegacy)."},
	{ID: "moonshot-intl", Name: "Kimi (moonshot.ai)", NameZH: "Kimi（国际站）", Auth: AuthAPIKey, Legacy: []Legacy{llm("auralink", "llm.moonshot-intl")}},
	{ID: "siliconflow", Name: "SiliconFlow (siliconflow.cn)", NameZH: "硅基流动（中国站）", Auth: AuthAPIKey, Legacy: []Legacy{llm("audiosmith", "llm.siliconflow"), llm("auralink", "llm.siliconflow")},
		Note: "AudioSmith's llm.siliconflow served both sites: its base URL decides (VendorForLegacy)."},
	{ID: "siliconflow-intl", Name: "SiliconFlow (siliconflow.com)", NameZH: "硅基流动（国际站）", Auth: AuthAPIKey, Legacy: []Legacy{llm("auralink", "llm.siliconflow-intl")}},
	{ID: "zhipu", Name: "Zhipu AI", NameZH: "智谱 AI", Auth: AuthAPIKey, Legacy: both("llm.zhipu")},
	{ID: "minimax", Name: "MiniMax", Auth: AuthAPIKey, Legacy: both("llm.minimax")},
	{ID: "mimo", Name: "Xiaomi MiMo", NameZH: "小米 MiMo", Auth: AuthAPIKey, Legacy: both("llm.mimo")},
	{ID: "gemini", Name: "Google Gemini", Auth: AuthAPIKey, Legacy: both("llm.gemini")},
	{ID: "xai", Name: "xAI", Auth: AuthAPIKey, Legacy: both("llm.xai")},
	{ID: "groq", Name: "Groq", Auth: AuthAPIKey, Legacy: both("llm.groq")},
	{ID: "mistral", Name: "Mistral AI", Auth: AuthAPIKey, Legacy: both("llm.mistral")},
	{ID: "together", Name: "Together AI", Auth: AuthAPIKey, Legacy: both("llm.together")},
	{ID: "fireworks", Name: "Fireworks AI", Auth: AuthAPIKey, Legacy: both("llm.fireworks")},
	{ID: "cerebras", Name: "Cerebras", Auth: AuthAPIKey, Legacy: both("llm.cerebras")},
	{ID: "nvidia", Name: "NVIDIA NIM", Auth: AuthAPIKey, Legacy: both("llm.nvidia")},
	{ID: "openrouter", Name: "OpenRouter", Auth: AuthAPIKey, Legacy: both("llm.openrouter")},
	{ID: "deepl", Name: "DeepL", Auth: AuthAPIKey, Legacy: []Legacy{llm("audiosmith", "mt.deepl")}},
	{ID: "google-translate", Name: "Google Cloud Translation", NameZH: "Google 翻译", Auth: AuthAPIKey, Legacy: []Legacy{llm("audiosmith", "mt.google")},
		Note: "mt.google-v3 is an OAuth token and is not shared."},
	{ID: "microsoft-translator", Name: "Microsoft Translator", NameZH: "微软翻译", Auth: AuthAPIKey, Legacy: []Legacy{llm("audiosmith", "mt.microsoft")}},
	{ID: "huggingface", Name: "Hugging Face", Auth: AuthToken, Legacy: []Legacy{llm("audiosmith", "huggingface-token"), llm("dicta", "huggingface-token")}},
}

func both(account string) []Legacy {
	return []Legacy{llm("audiosmith", account), llm("auralink", account)}
}

// VendorByID finds a vendor.
func VendorByID(id string) (Vendor, bool) {
	for _, v := range Vendors {
		if v.ID == id {
			return v, true
		}
	}
	return Vendor{}, false
}

// VendorForLegacy finds the vendor of an app's old Keychain account
// ("audiosmith", "llm.deepseek"). baseURL is the provider's endpoint, which
// decides the site for accounts that served two (AudioSmith's
// llm.moonshot, llm.siliconflow); "" picks the Chinese site. An OpenAI
// account with a custom (non-api.openai.com) endpoint is not shared.
func VendorForLegacy(appID, account, baseURL string) (Vendor, bool) {
	host := ""
	if u, err := url.Parse(strings.TrimSpace(baseURL)); err == nil {
		host = strings.ToLower(u.Hostname())
	}
	for _, v := range Vendors {
		for _, l := range v.Legacy {
			if l.App != appID || l.Account != account {
				continue
			}
			switch {
			case v.ID == "moonshot" && strings.HasSuffix(host, "moonshot.ai"):
				return VendorByID("moonshot-intl")
			case v.ID == "siliconflow" && strings.HasSuffix(host, "siliconflow.com"):
				return VendorByID("siliconflow-intl")
			case v.ID == "openai" && host != "" && host != "api.openai.com":
				return Vendor{}, false
			}
			return v, true
		}
	}
	return Vendor{}, false
}
