// Package openai is the OpenAI-compatible file transcriber: one
// multipart upload to …/audio/transcriptions after the recording stops.
// OpenAI itself, and the services that speak its API (StepFun, Zhipu's
// GLM-ASR) with their endpoint and model. The ASR prompt goes in the
// "prompt" field, with the dictionary's terms folded into it.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// ID is the provider's preset id.
const ID = "openai"

// DefaultEndpoint and DefaultModel are OpenAI's.
const (
	DefaultEndpoint = "https://api.openai.com/v1/audio/transcriptions"
	DefaultModel    = "gpt-4o-mini-transcribe"
)

// Models are the models offered in the model menu (any id may be typed).
var Models = []string{"gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"}

// MaxBytes is OpenAI's upload limit (25 MB).
const MaxBytes = 25 << 20

// maxPrompt bounds the prompt: Whisper reads its last 224 tokens only.
const maxPrompt = 800

// Client is a configured transcriber. Goroutine-safe.
type Client struct {
	endpoint string
	model    string
	key      string
	http     *http.Client
	// Basic sends the file, model and prompt only: Zhipu's GLM-ASR
	// documents no response_format or language (it detects it).
	Basic bool
	// Sleep waits before a retry (tests replace it).
	Sleep func(time.Duration)
}

var _ asr.FileTranscriber = (*Client)(nil)

// New checks cfg and returns a client. The endpoint may be the full
// transcription URL or an API root ending in /v1 (…/audio/transcriptions
// is added); it must be https (http only on loopback). A server on this
// Mac may take no key (a local Whisper server); any other needs one.
func New(cfg asr.Config) (*Client, error) {
	key := strings.TrimSpace(cfg.Key)
	ep := Endpoint(cfg.Endpoint)
	u, err := asr.CheckEndpoint(ID, ep, "https")
	if err != nil {
		return nil, err
	}
	if key == "" && !asr.IsLoopback(u.Hostname()) {
		return nil, asr.Errorf(ID, asr.KindConfig, "no API key")
	}
	model := strings.TrimSpace(cfg.Model)
	if model == "" {
		model = DefaultModel
	}
	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &Client{endpoint: ep, model: model, key: key, http: client, Sleep: time.Sleep}, nil
}

// Endpoint is the transcription URL for what the user typed: "" is
// DefaultEndpoint, and an API root (…/v1, a path without /audio/) gets
// /audio/transcriptions appended.
func Endpoint(raw string) string {
	ep := strings.TrimSpace(raw)
	if ep == "" {
		return DefaultEndpoint
	}
	base, query, _ := strings.Cut(ep, "?")
	base = strings.TrimRight(base, "/")
	if !strings.Contains(base, "/audio/") {
		base += "/audio/transcriptions"
	}
	if query != "" {
		return base + "?" + query
	}
	return base
}

// Caps: an upload of at most 25 MB with a prompt; hints go into it.
func (c *Client) Caps() asr.Caps {
	return asr.Caps{File: true, Prompt: true, MaxBytes: MaxBytes}
}

// Model is the model the client asks for.
func (c *Client) Model() string { return c.model }

// Transcribe uploads req's audio. Network failures, 5xx and rate limits
// are retried once.
func (c *Client) Transcribe(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	audio, name := req.Audio, "audio.wav"
	switch req.Format {
	case "", "wav":
	case "pcm":
		audio = asr.WAV(req.Audio, req.SampleRate)
	default:
		name = "audio." + req.Format
	}
	if len(audio) > MaxBytes {
		return asr.Result{}, asr.Errorf(ID, asr.KindAudio, "the recording is %d MB, over the 25 MB limit", len(audio)>>20)
	}
	body, ctype, err := c.form(audio, name, req)
	if err != nil {
		return asr.Result{}, &asr.Error{Kind: asr.KindProtocol, Provider: ID, Detail: "multipart", Err: err}
	}
	start := time.Now()
	var res asr.Result
	for attempt := 1; ; attempt++ {
		res, err = c.post(ctx, body, ctype)
		e, _ := err.(*asr.Error)
		if err == nil || attempt >= 2 || e == nil || !e.Retryable || ctx.Err() != nil {
			break
		}
		c.Sleep(700 * time.Millisecond)
	}
	if err != nil {
		return asr.Result{}, err
	}
	res.Model = c.model
	res.Latency = time.Since(start)
	return res, nil
}

// form builds the multipart body: file, model, response_format, then
// language and prompt when there are any.
func (c *Client) form(audio []byte, name string, req asr.FileRequest) ([]byte, string, error) {
	var b bytes.Buffer
	w := multipart.NewWriter(&b)
	h := textproto.MIMEHeader{}
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename=%q`, name))
	h.Set("Content-Type", mimeType(name))
	fw, err := w.CreatePart(h)
	if err != nil {
		return nil, "", err
	}
	fw.Write(audio)
	fields := [][2]string{{"model", c.model}}
	if !c.Basic {
		fields = append(fields, [2]string{"response_format", "json"})
	}
	if lang := isoLanguage(req.Lang); lang != "" && !c.Basic {
		fields = append(fields, [2]string{"language", lang})
	}
	if p := Prompt(req.Prompt, req.Hints); p != "" && !strings.Contains(c.model, "diarize") {
		fields = append(fields, [2]string{"prompt", p})
	}
	for _, f := range fields {
		if err := w.WriteField(f[0], f[1]); err != nil {
			return nil, "", err
		}
	}
	if err := w.Close(); err != nil {
		return nil, "", err
	}
	return b.Bytes(), w.FormDataContentType(), nil
}

func (c *Client) post(ctx context.Context, body []byte, ctype string) (asr.Result, error) {
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 120*time.Second)
		defer cancel()
	}
	hr, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return asr.Result{}, &asr.Error{Kind: asr.KindConfig, Provider: ID, Detail: "bad endpoint", Err: err}
	}
	hr.Header.Set("Content-Type", ctype)
	hr.Header.Set("Accept", "application/json, text/plain")
	if c.key != "" {
		hr.Header.Set("Authorization", "Bearer "+c.key)
	}
	hr.Header.Set("User-Agent", "Voxt")
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
			return asr.Result{}, nil // silence: nothing was said (StepFun's 400 "no speech found")
		}
		return asr.Result{}, asr.HTTPError(ID, resp.StatusCode, code, msg)
	}
	return parse(raw)
}

// parse reads {"text": …} (with "language" and "duration" in the
// verbose format), or a plain text answer.
func parse(raw []byte) (asr.Result, error) {
	s := strings.TrimSpace(string(raw))
	if strings.HasPrefix(s, "{") {
		var v struct {
			Text     *string `json:"text"`
			Language string  `json:"language"`
			Duration float64 `json:"duration"`
			Segments []struct {
				Text string `json:"text"`
			} `json:"segments"`
			Error json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal(raw, &v); err != nil {
			return asr.Result{}, &asr.Error{Kind: asr.KindProtocol, Provider: ID, Detail: asr.Clip(s, 200), Err: err}
		}
		if len(v.Error) > 0 && string(v.Error) != "null" {
			code, msg := asr.ErrorBody(raw)
			return asr.Result{}, asr.HTTPError(ID, 500, code, msg)
		}
		text := ""
		switch {
		case v.Text != nil && strings.TrimSpace(*v.Text) != "" || v.Text != nil && len(v.Segments) == 0:
			text = *v.Text
		case len(v.Segments) > 0:
			var parts []string
			for _, sg := range v.Segments {
				parts = append(parts, strings.TrimSpace(sg.Text))
			}
			text = strings.Join(parts, " ")
		default:
			return asr.Result{}, &asr.Error{Kind: asr.KindProtocol, Provider: ID, Detail: "the answer has no text: " + asr.Clip(s, 200)}
		}
		return asr.Result{Text: strings.TrimSpace(text), Lang: v.Language, Duration: time.Duration(v.Duration * float64(time.Second))}, nil
	}
	return asr.Result{Text: s}, nil
}

// Prompt is the "prompt" field: the user's prompt, then the hints as a
// comma-separated list of terms (Whisper follows the spelling of what it
// was prompted with), at most 800 bytes, keeping the start.
func Prompt(prompt string, hints []string) string {
	prompt = strings.TrimSpace(prompt)
	var terms []string
	for _, h := range hints {
		if h = strings.TrimSpace(h); h != "" {
			terms = append(terms, h)
		}
	}
	if len(terms) > 0 {
		list := strings.Join(terms, ", ")
		if prompt == "" {
			prompt = list
		} else {
			prompt += "\n" + list
		}
	}
	if len(prompt) > maxPrompt {
		cut := maxPrompt
		for cut > 0 && prompt[cut]&0xC0 == 0x80 {
			cut--
		}
		prompt = prompt[:cut]
	}
	return prompt
}

// isoLanguage is the ISO-639-1 code OpenAI wants: "zh-Hans" → "zh".
func isoLanguage(lang string) string {
	lang = strings.ToLower(strings.TrimSpace(lang))
	if i := strings.IndexAny(lang, "-_"); i > 0 {
		lang = lang[:i]
	}
	if len(lang) != 2 {
		return ""
	}
	return lang
}

func mimeType(name string) string {
	switch {
	case strings.HasSuffix(name, ".wav"):
		return "audio/wav"
	case strings.HasSuffix(name, ".m4a"):
		return "audio/mp4"
	case strings.HasSuffix(name, ".mp3"):
		return "audio/mpeg"
	}
	return "application/octet-stream"
}
