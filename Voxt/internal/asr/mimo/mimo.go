// Package mimo is Xiaomi MiMo's speech recognition as an
// asr.FileTranscriber, as Voxt 1.x used it: one OpenAI-style chat
// completion whose user message is the recording (input_audio, a WAV data
// URI), with asr_options.language; the answer's message is the text. It
// takes no prompt or hot words.
package mimo

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// ID is the service's preset id.
const ID = "mimo"

// DefaultEndpoint and DefaultModel are Voxt 1.x's.
const (
	DefaultEndpoint = "https://api.xiaomimimo.com/v1/chat/completions"
	DefaultModel    = "mimo-v2.5-asr"
)

// Models are the model menu's (any id may be typed).
var Models = []string{DefaultModel}

// Client is a configured transcriber. Goroutine-safe.
type Client struct {
	endpoint, model, key string
	http                 *http.Client
}

var _ asr.FileTranscriber = (*Client)(nil)

// New checks cfg: a key, and an endpoint that is https:// (http:// only
// on loopback); an API root ("…/v1") gets /chat/completions.
func New(cfg asr.Config) (*Client, error) {
	key := strings.TrimSpace(cfg.Key)
	if key == "" {
		return nil, asr.Errorf(ID, asr.KindConfig, "no API key")
	}
	ep := strings.TrimRight(strings.TrimSpace(cfg.Endpoint), "/")
	switch {
	case ep == "":
		ep = DefaultEndpoint
	case !strings.HasSuffix(ep, "/chat/completions"):
		ep += "/chat/completions"
	}
	if _, err := asr.CheckEndpoint(ID, ep, "https"); err != nil {
		return nil, err
	}
	model := strings.TrimSpace(cfg.Model)
	if model == "" {
		model = DefaultModel
	}
	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &Client{endpoint: ep, model: model, key: key, http: client}, nil
}

// Caps: one upload, nothing else.
func (c *Client) Caps() asr.Caps { return asr.Caps{File: true} }

// language is asr_options.language: zh or en, else auto.
func language(lang string) string {
	lang = strings.ToLower(strings.TrimSpace(lang))
	if i := strings.IndexAny(lang, "-_"); i > 0 {
		lang = lang[:i]
	}
	if lang == "zh" || lang == "en" {
		return lang
	}
	return "auto"
}

// Transcribe sends req's audio in one request.
func (c *Client) Transcribe(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	res, err := c.transcribe(ctx, req)
	return res, asr.HideKey(err, c.key)
}

func (c *Client) transcribe(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	audio := req.Audio
	if req.Format == "pcm" {
		audio = asr.WAV(req.Audio, req.SampleRate)
	}
	body, _ := json.Marshal(map[string]any{
		"model": c.model,
		"messages": []any{map[string]any{"role": "user", "content": []any{map[string]any{
			"type":        "input_audio",
			"input_audio": map[string]string{"data": "data:audio/wav;base64," + base64.StdEncoding.EncodeToString(audio)},
		}}}},
		"asr_options": map[string]string{"language": language(req.Lang)},
	})
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 120*time.Second)
		defer cancel()
	}
	hr, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return asr.Result{}, &asr.Error{Kind: asr.KindConfig, Provider: ID, Detail: "bad endpoint", Err: err}
	}
	hr.Header.Set("Content-Type", "application/json")
	hr.Header.Set("Accept", "application/json")
	hr.Header.Set("Authorization", "Bearer "+c.key)
	start := time.Now()
	resp, err := c.http.Do(hr)
	if err != nil {
		return asr.Result{}, asr.NetError(ctx, ID, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return asr.Result{}, asr.NetError(ctx, ID, err)
	}
	if resp.StatusCode/100 != 2 {
		code, msg := asr.ErrorBody(raw)
		if msg == "" {
			msg = resp.Status
		}
		if resp.StatusCode/100 == 4 && asr.NoSpeech(code, msg) {
			return asr.Result{Model: c.model}, nil
		}
		return asr.Result{}, asr.HTTPError(ID, resp.StatusCode, code, msg)
	}
	var v struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &v); err != nil || len(v.Choices) == 0 {
		return asr.Result{}, &asr.Error{Kind: asr.KindProtocol, Provider: ID, Detail: "the answer has no text: " + asr.Clip(string(raw), 200), Err: err}
	}
	return asr.Result{Text: strings.TrimSpace(v.Choices[0].Message.Content), Model: c.model, Latency: time.Since(start)}, nil
}
