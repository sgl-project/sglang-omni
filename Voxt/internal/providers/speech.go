package providers

// The remote speech services (Voxt 1.x's, the clients Auralink's and
// internal/asr's): their presets, the user's setups (settings.Providers.ASR,
// keys in the Keychain as "asr.<id>"), building a client and the
// connection test.

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/aliyun"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/doubao"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/gemini"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/mimo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/openai"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// Speech is a remote speech service's fixed facts.
type Speech struct {
	ID   string
	Name string // brand name, not translated
	// Stream: it recognises while the user speaks (text shows as it
	// comes: "Realtime"); else it gets the recording once it stops.
	Stream   bool
	Endpoint string   // the default URL
	Model    string   // the default model
	Models   []string // the model menu's (any id may be typed)
	// AppID: the key may come with an App ID (Doubao's older console;
	// its newer one gives a single API key).
	AppID  bool
	KeyURL string // where to get a key
}

// Speech services' ids that have no client package of their own.
const (
	StepFunASR = "stepfun"
	ZhipuASR   = "zhipu"
)

// speechPresets are Voxt 1.x's remote speech services, in its order.
var speechPresets = []Speech{
	{ID: openai.ID, Name: "OpenAI Transcribe", Endpoint: openai.DefaultEndpoint, Model: openai.DefaultModel, Models: openai.Models,
		KeyURL: "https://platform.openai.com/api-keys"},
	{ID: doubao.ID, Name: "Doubao (Volcengine)", Stream: true, Endpoint: doubao.DefaultEndpoint, Model: doubao.DefaultModel, Models: doubao.Models, AppID: true,
		KeyURL: "https://console.volcengine.com/speech/app"},
	{ID: ZhipuASR, Name: "Zhipu GLM-ASR", Endpoint: "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions", Model: "glm-asr-2512", Models: []string{"glm-asr-2512", "glm-asr-1"},
		KeyURL: "https://bigmodel.cn/usercenter/proj-mgmt/apikeys"},
	{ID: aliyun.ID, Name: "Aliyun Bailian ASR", Stream: true, Endpoint: aliyun.InferenceEndpoint, Model: aliyun.DefaultModel, Models: aliyun.Models,
		KeyURL: "https://bailian.console.aliyun.com/?tab=model#/api-key"},
	// StepFun's transcriptions are OpenAI-compatible (multipart, Bearer),
	// checked by hand with step-asr for Auralink on 2026-10-15.
	{ID: StepFunASR, Name: "StepFun", Endpoint: "https://api.stepfun.com/v1/audio/transcriptions", Model: "step-asr", Models: []string{"step-asr", "step-asr-1.1"},
		KeyURL: "https://platform.stepfun.com/interface-key"},
	{ID: mimo.ID, Name: "Xiaomi MiMo", Endpoint: mimo.DefaultEndpoint, Model: mimo.DefaultModel, Models: mimo.Models,
		KeyURL: "https://platform.xiaomimimo.com/"},
	{ID: gemini.ID, Name: "Google Gemini", Stream: true, Endpoint: gemini.DefaultEndpoint, Model: gemini.DefaultModel, Models: gemini.Models,
		KeyURL: "https://aistudio.google.com/apikey"},
}

// SpeechPresets returns the remote speech services, in display order (a
// copy).
func SpeechPresets() []Speech { return slices.Clone(speechPresets) }

// LookupSpeech returns the speech service id.
func LookupSpeech(id string) (Speech, bool) {
	i := slices.IndexFunc(speechPresets, func(p Speech) bool { return p.ID == id })
	if i < 0 {
		return Speech{}, false
	}
	return speechPresets[i], true
}

// SpeechSetup is the user's setup of speech service id, and whether there
// is one.
func SpeechSetup(s settings.Settings, id string) (settings.Provider, bool) {
	return setupIn(s.Providers.ASR, id)
}

// PutSpeech sets p in s's speech setups (a new list, as Put).
func PutSpeech(s *settings.Settings, p settings.Provider) { s.Providers.ASR = put(s.Providers.ASR, p) }

// RemoveSpeech takes speech setup id out of s, and out of the features
// that use it (they go back to automatic).
func RemoveSpeech(s *settings.Settings, id string) {
	s.Providers.ASR = slices.DeleteFunc(slices.Clone(s.Providers.ASR), func(p settings.Provider) bool { return p.ID == id })
	for _, f := range []*settings.Feature{&s.Features.Transcribe.Feature, &s.Features.Translate.Feature, &s.Features.Rewrite.Feature} {
		if f.ASR == id {
			f.ASR = ""
		}
	}
}

// SpeechModel is the model a speech setup uses: its own, else its
// preset's default.
func SpeechModel(p settings.Provider) string {
	if m := strings.TrimSpace(p.Model); m != "" {
		return m
	}
	pr, _ := LookupSpeech(p.ID)
	return pr.Model
}

// SpeechReady lists the speech setups that can be used now (a known
// service with a key, or one that needs none: SpeechKeyOptional), in the
// presets' order.
func SpeechReady(s settings.Settings, hasKey func(id string) bool) []settings.Provider {
	var out []settings.Provider
	for _, pr := range speechPresets {
		if p, ok := SpeechSetup(s, pr.ID); ok && (hasKey(p.ID) || SpeechKeyOptional(p)) {
			out = append(out, p)
		}
	}
	return out
}

// SpeechKeyOptional says whether a speech setup works without a key: an
// OpenAI-compatible transcription server on this Mac (a local Whisper
// server), as the language models' Ollama.
func SpeechKeyOptional(p settings.Provider) bool {
	switch p.ID {
	case openai.ID, StepFunASR, ZhipuASR:
		u, err := url.Parse(strings.TrimSpace(p.Endpoint))
		return err == nil && u.Host != "" && asr.IsLoopback(u.Hostname())
	}
	return false
}

// NewASR is the client of a speech setup with its key, over client (nil:
// a direct one). It fails when the setup's endpoint is not allowed (wss://
// for streams, https:// for uploads; plain only on loopback) or it has no
// key.
func NewASR(p settings.Provider, key string, client *http.Client) (asr.Provider, error) {
	pr, ok := LookupSpeech(p.ID)
	if !ok {
		return nil, asr.Errorf(p.ID, asr.KindConfig, "unknown speech service %q", p.ID)
	}
	cfg := asr.Config{Endpoint: p.Endpoint, Model: SpeechModel(p), Key: key, AppID: p.AppID, HTTPClient: client}
	switch pr.ID {
	case openai.ID:
		return openai.New(cfg)
	case StepFunASR, ZhipuASR:
		if strings.TrimSpace(cfg.Endpoint) == "" {
			cfg.Endpoint = pr.Endpoint
		}
		c, err := openai.New(cfg)
		if err == nil {
			c.Basic = pr.ID == ZhipuASR
		}
		return c, err
	case doubao.ID:
		return doubao.New(cfg)
	case aliyun.ID:
		return aliyun.New(cfg)
	case mimo.ID:
		return mimo.New(cfg)
	case gemini.ID:
		return gemini.New(cfg)
	}
	return nil, asr.Errorf(p.ID, asr.KindConfig, "unknown speech service %q", p.ID)
}

// SpeechTestTimeout bounds a speech service's connection test.
const SpeechTestTimeout = 30 * time.Second

// TestSpeech checks that a speech setup works: a second of silence
// recognised (streamed or uploaded), which proves the key, the endpoint
// and the protocol (Auralink's test). Silence's text is usually "".
func TestSpeech(ctx context.Context, p settings.Provider, key string, client *http.Client) TestResult {
	ctx, cancel := context.WithTimeout(ctx, SpeechTestTimeout)
	defer cancel()
	res := TestResult{At: time.Now()}
	c, err := NewASR(p, key, client)
	if err == nil {
		_, err = asr.Recognize(ctx, c, asr.Silence(time.Second), "", nil, "", asr.Feed{})
	}
	res.Latency, res.Err = time.Since(res.At), err
	return res
}

// SpeechTestLog is TestLog's line for a speech service.
func SpeechTestLog(p settings.Provider, res TestResult, key string) (level, msg string) {
	return testLog(fmt.Sprintf("connection test of remote speech service %s (model %s)", p.ID, SpeechModel(p)), res, key)
}
