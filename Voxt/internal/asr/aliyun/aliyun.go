// Package aliyun is the streamer of Alibaba Cloud Model Studio (百炼,
// DashScope): recognition over a WebSocket while the user speaks. The
// service speaks two protocols, chosen by the model (ProtocolFor):
//
//   - Inference (wss://dashscope.aliyuncs.com/api-ws/v1/inference), the
//     DashScope duplex task protocol of fun-asr-realtime, paraformer-realtime
//     and qwen-audio-3.x-asr-flash-streaming: run-task → task-started →
//     binary PCM frames → result-generated… → finish-task → task-finished
//     (or task-failed at any point).
//   - Realtime (wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=…),
//     the OpenAI-Realtime-like protocol of qwen3-asr-flash-realtime:
//     session.update → session.updated → input_audio_buffer.append (base64)…
//     → session.finish → …transcription.completed → session.finished.
//
// The international site (dashscope-intl.aliyuncs.com) and the workspace
// hosts ({WorkspaceId}.cn-beijing.maas.aliyuncs.com) take the same
// protocols: the user sets the endpoint, and Endpoint fits its path to the
// model's protocol.
//
// Message building and parsing are pure functions (inference.go,
// realtime.go) with tests of their own; stream.go owns the connection.
package aliyun

import (
	"context"
	"net/http"
	"net/url"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// ID is the provider's preset id.
const ID = "aliyun"

// DefaultModel is the model of a new setup.
const DefaultModel = "fun-asr-realtime"

// Default endpoints, Beijing region (the international site is
// dashscope-intl.aliyuncs.com with the same paths).
const (
	InferenceEndpoint = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
	RealtimeEndpoint  = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
)

const (
	inferencePath = "/api-ws/v1/inference"
	realtimePath  = "/api-ws/v1/realtime"
)

// Models are the models offered in the model menu, grouped as Voxt groups
// them (any id may be typed). The 8 kHz telephone models are left out:
// they take 8 kHz audio only and Voxt records at 16 kHz (New refuses
// them).
var Models = []string{
	// Qwen-Audio ASR streaming (inference protocol, inline hot words).
	"qwen-audio-3.0-asr-flash-streaming",
	// Qwen3 ASR realtime (realtime protocol).
	"qwen3-asr-flash-realtime",
	"qwen3-asr-flash-realtime-2026-02-10",
	"qwen3-asr-flash-realtime-2025-10-27",
	// Fun-ASR realtime (inference protocol).
	"fun-asr-realtime",
	"fun-asr-realtime-2026-02-28",
	"fun-asr-realtime-2025-11-07",
	"fun-asr-realtime-2025-09-15",
	// Paraformer realtime (inference protocol).
	"paraformer-realtime-v2",
	"paraformer-realtime-v1",
}

// Protocol is the wire protocol a model speaks.
type Protocol uint8

const (
	// Inference is DashScope's duplex task protocol (/api-ws/v1/inference).
	Inference Protocol = iota + 1
	// Realtime is the Realtime API (/api-ws/v1/realtime?model=…).
	Realtime
)

func (p Protocol) String() string {
	switch p {
	case Inference:
		return "inference"
	case Realtime:
		return "realtime"
	}
	return "unknown"
}

// ProtocolFor returns the protocol of model: Realtime for the
// qwen3-asr-flash-realtime family, Inference for fun-asr*, paraformer-
// realtime* and qwen-audio-*-asr-flash-streaming, and Inference for an
// unknown model too (DashScope's ASR models mostly speak it).
func ProtocolFor(model string) Protocol {
	if strings.HasPrefix(norm(model), "qwen3-asr-flash-realtime") {
		return Realtime
	}
	return Inference
}

// model is what this package knows of a model.
type model struct {
	proto Protocol
	// context: it takes input.context (hot words and the prompt as text the
	// recogniser matches words against, ≤ 400 characters).
	context bool
	// vocabulary: it takes inline hot words (parameters.vocabulary).
	vocabulary bool
	// hints: how many language_hints it takes (0: none); langs: the
	// language codes it knows (nil: any).
	hints int
	langs []string
	// rate8k: it takes 8 kHz audio only.
	rate8k bool
}

// Language catalogues from Voxt's AliyunASRLanguageCatalog, checked
// against the official run-task / session.update references.
var (
	langsFunRealtime = []string{"zh", "en", "ja", "ko", "vi", "th", "id", "ms", "tl", "hi",
		"ar", "fr", "de", "es", "pt", "ru", "it", "nl", "sv", "da",
		"fi", "no", "el", "pl", "cs", "hu", "ro", "bg", "hr", "sk"}
	langsQwenRealtime = []string{"zh", "yue", "en", "ja", "de", "ko", "ru", "fr", "pt", "ar",
		"it", "es", "hi", "id", "th", "tr", "uk", "vi", "cs", "da",
		"fil", "fi", "is", "ms", "no", "pl", "sv"}
	langsParaformer = []string{"zh", "en", "ja", "ko", "de", "fr", "ru"}
)

func modelInfo(name string) model {
	n := norm(name)
	switch {
	case strings.HasPrefix(n, "qwen3-asr-flash-realtime"):
		return model{proto: Realtime, hints: 1, langs: langsQwenRealtime}
	case strings.HasPrefix(n, "qwen-audio-") && strings.Contains(n, "-asr-flash-streaming"):
		return model{proto: Inference, context: true, vocabulary: true, hints: 4, langs: langsFunRealtime}
	case strings.HasPrefix(n, "paraformer-realtime"):
		m := model{proto: Inference, hints: 1, langs: langsParaformer}
		if strings.Contains(n, "-8k-") {
			m.rate8k, m.langs = true, []string{"zh"}
		}
		return m
	case strings.HasPrefix(n, "fun-asr"):
		m := model{proto: Inference, hints: 1, langs: langsFunRealtime}
		switch {
		case strings.Contains(n, "-8k-"):
			m.rate8k, m.langs = true, []string{"zh"}
		case strings.HasPrefix(n, "fun-asr-realtime-2026-"):
			m.langs = []string{"zh", "en", "ja"}
		case strings.HasPrefix(n, "fun-asr-realtime-2025-09-"):
			m.langs = []string{"zh", "en"}
		}
		// The reference: "only fun-asr-realtime supports context"; Voxt
		// also sends it to the snapshot that alias pointed at.
		m.context = n == "fun-asr-realtime" || n == "fun-asr-realtime-2025-11-07"
		return m
	}
	return model{proto: ProtocolFor(n)}
}

func norm(s string) string { return strings.ToLower(strings.TrimSpace(s)) }

// Endpoint is the WebSocket URL for what the user typed and model:
//
//   - "" is the Beijing default of the model's protocol.
//   - https:// and http:// become wss:// and ws:// (a pasted console URL).
//   - A path ending in /api-ws/v1/inference or /api-ws/v1/realtime is
//     switched to the model's protocol (Voxt does the same: its docs tell
//     users to paste the realtime URL for every model), dropping or adding
//     the model query.
//   - A bare host, or one of DashScope's HTTP paths (/compatible-mode/…,
//     /api/v1/…), gets the model's protocol path.
//   - Any other path (a relay) is kept; the realtime protocol adds
//     ?model= when it has none.
func Endpoint(raw, modelName string) string {
	p := ProtocolFor(modelName)
	raw = strings.TrimSpace(raw)
	if raw == "" {
		if p == Realtime {
			return RealtimeEndpoint + "?model=" + url.QueryEscape(strings.TrimSpace(modelName))
		}
		return InferenceEndpoint
	}
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return raw // CheckEndpoint reports it
	}
	switch strings.ToLower(u.Scheme) {
	case "https":
		u.Scheme = "wss"
	case "http":
		u.Scheme = "ws"
	}
	want := inferencePath
	if p == Realtime {
		want = realtimePath
	}
	low := strings.ToLower(strings.TrimRight(u.Path, "/"))
	switch {
	case strings.HasSuffix(low, inferencePath), strings.HasSuffix(low, realtimePath):
		cut := len(realtimePath)
		if strings.HasSuffix(low, inferencePath) {
			cut = len(inferencePath)
		}
		u.Path = strings.TrimRight(u.Path, "/")
		u.Path = u.Path[:len(u.Path)-cut] + want
	case low == "", strings.HasPrefix(low, "/compatible-mode/"), strings.HasPrefix(low, "/api/v1/"), low == "/api-ws/v1":
		u.Path = want
	}
	u.RawPath = ""
	q := u.Query()
	if p == Realtime {
		if q.Get("model") == "" {
			q.Set("model", strings.TrimSpace(modelName))
		}
	} else {
		q.Del("model")
	}
	u.RawQuery = q.Encode()
	return u.String()
}

// Client is a configured streamer. Goroutine-safe.
type Client struct {
	endpoint string
	model    string
	info     model
	key      string
	http     *http.Client
}

var _ asr.Streamer = (*Client)(nil)

// New checks cfg and returns a client: cfg.Key (the DashScope API key) is
// required, cfg.Model defaults to DefaultModel, cfg.Endpoint (see
// Endpoint) must be wss (ws only on loopback).
func New(cfg asr.Config) (*Client, error) {
	key := strings.TrimSpace(cfg.Key)
	if key == "" {
		return nil, asr.Errorf(ID, asr.KindConfig, "no API key")
	}
	name := strings.TrimSpace(cfg.Model)
	if name == "" {
		name = DefaultModel
	}
	info := modelInfo(name)
	if info.rate8k {
		return nil, asr.Errorf(ID, asr.KindConfig, "model %s takes 8 kHz telephone audio only; Voxt records at 16 kHz", name)
	}
	ep := Endpoint(cfg.Endpoint, name)
	if _, err := asr.CheckEndpoint(ID, ep, "wss"); err != nil {
		return nil, err
	}
	return &Client{endpoint: ep, model: name, info: info, key: key, http: asr.WSClient(cfg.HTTPClient)}, nil
}

// Model is the model the client uses.
func (c *Client) Model() string { return c.model }

// Protocol is the protocol the client speaks.
func (c *Client) Protocol() Protocol { return c.info.proto }

// Caps reports a streamer; Hints and Prompt where the model takes them
// (fun-asr-realtime and qwen-audio streaming: input.context; the latter
// also inline hot words).
func (c *Client) Caps() asr.Caps {
	return asr.Caps{Stream: true, Hints: c.info.context || c.info.vocabulary, Prompt: c.info.context}
}

// Open connects, starts the task or session, and returns once the service
// accepted it (task-started, session.updated).
func (c *Client) Open(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	s, err := c.connect(ctx, req)
	if err != nil {
		return nil, asr.HideKey(err, c.key)
	}
	return s, nil
}

func (c *Client) connect(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	if req.SampleRate <= 0 {
		req.SampleRate = asr.SampleRate
	}
	h := http.Header{}
	h.Set("Authorization", "Bearer "+c.key)
	var p protocol
	if c.info.proto == Realtime {
		if req.SampleRate != 16000 && req.SampleRate != 8000 {
			return nil, asr.Errorf(ID, asr.KindAudio, "%s takes 16 kHz or 8 kHz audio, not %d Hz", c.model, req.SampleRate)
		}
		h.Set("OpenAI-Beta", "realtime=v1")
		p = &realtimeProto{model: c.model, info: c.info, req: req}
	} else {
		p = &inferenceProto{model: c.model, info: c.info, req: req, taskID: newID()}
	}
	s, err := open(ctx, c, h, p, req)
	if err != nil {
		return nil, err // not a nil *stream in an interface
	}
	return s, nil
}

// lang is the language code of req.Lang the model knows, "" when it does
// not know it (the service then detects the language).
func (m model) lang(l string) string {
	l = norm(l)
	if i := strings.IndexAny(l, "-_"); i >= 0 {
		l = l[:i]
	}
	if l == "" {
		return ""
	}
	if l == "fil" || l == "tl" { // Filipino: the two catalogues disagree
		for _, k := range m.langs {
			if k == "fil" || k == "tl" {
				return k
			}
		}
	}
	if m.langs == nil {
		return l
	}
	for _, k := range m.langs {
		if k == l {
			return l
		}
	}
	return ""
}
