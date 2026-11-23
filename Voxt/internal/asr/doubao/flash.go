package doubao

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// The flash file recognition (大模型录音文件极速版识别, docs/6561/1631584):
// the whole recording in one POST, the text in its answer. The same
// credentials as the streams; the resource is ModelFlash, which the
// account must have opened.
const (
	FlashEndpoint = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
	ModelFlash    = "volc.bigasr.auc_turbo"
	// FlashMaxBytes and FlashMaxDuration are the docs' limits: 100 MB
	// (about 52 minutes of 16 kHz mono WAV) and 2 hours.
	FlashMaxBytes    = 100_000_000
	FlashMaxDuration = 2 * time.Hour
)

// codeSilence is the flash recognition's answer to a recording without
// speech.
const codeSilence = 20000003

// Flash transcribes whole recordings. Goroutine-safe.
type Flash struct {
	endpoint     string
	appID, token string
	http         *http.Client
	// Sleep waits before a retry (tests replace it).
	Sleep func(time.Duration)
}

var _ asr.FileTranscriber = (*Flash)(nil)

// NewFlash checks cfg as New does; the endpoint, if set, must be https://
// (http:// only on loopback).
func NewFlash(cfg asr.Config) (*Flash, error) {
	f := &Flash{appID: strings.TrimSpace(cfg.AppID), token: strings.TrimSpace(cfg.Key), http: cfg.HTTPClient, Sleep: time.Sleep}
	if err := checkKey(f.appID, f.token); err != nil {
		return nil, err
	}
	ep := strings.TrimSpace(cfg.Endpoint)
	if ep == "" {
		ep = FlashEndpoint
	}
	u, err := asr.CheckEndpoint(ID, ep, "https")
	if err != nil {
		return nil, err
	}
	f.endpoint = u.String()
	if f.http == nil {
		f.http = http.DefaultClient
	}
	return f, nil
}

// Caps: an upload within the docs' limits, with hot words and a prompt
// (corpus.context, as the streams).
func (f *Flash) Caps() asr.Caps {
	return asr.Caps{File: true, Hints: true, Prompt: true, MaxBytes: FlashMaxBytes, MaxDuration: FlashMaxDuration}
}

// Model is the resource id.
func (f *Flash) Model() string { return ModelFlash }

// Endpoint is the URL recordings go to.
func (f *Flash) Endpoint() string { return f.endpoint }

// String is for logs: no credentials.
func (f *Flash) String() string {
	u, _ := url.Parse(f.endpoint)
	return fmt.Sprintf("doubao(%s%s, %s)", u.Host, u.Path, ModelFlash)
}

// Transcribe sends req's recording, in pieces under FlashMaxBytes when
// it is a longer PCM recording. Network failures and the service's
// busy and internal errors are retried once.
func (f *Flash) Transcribe(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	res, err := asr.TranscribeInPieces(ctx, req, FlashMaxBytes, f.transcribe)
	return res, asr.HideKey(err, f.token)
}

func (f *Flash) transcribe(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	audio, format := req.Audio, req.Format
	if format == "pcm" {
		audio, format = asr.WAV(req.Audio, req.SampleRate), "wav"
	}
	if len(audio) > FlashMaxBytes {
		return asr.Result{}, asr.Errorf(ID, asr.KindAudio, "the recording is %d MB, over the 100 MB limit", len(audio)/1_000_000)
	}
	body, err := flashRequest(req, format, audio)
	if err != nil {
		return asr.Result{}, asr.Errorf(ID, asr.KindConfig, "cannot build the request: %v", err)
	}
	start := time.Now()
	var res asr.Result
	for attempt := 1; ; attempt++ {
		res, err = f.post(ctx, body)
		e, _ := err.(*asr.Error)
		if err == nil || attempt >= 2 || e == nil || !e.Retryable || ctx.Err() != nil {
			break
		}
		f.Sleep(700 * time.Millisecond)
	}
	if err != nil {
		return asr.Result{}, err
	}
	res.Model = ModelFlash
	res.Latency = time.Since(start)
	return res, nil
}

func (f *Flash) post(ctx context.Context, body []byte) (asr.Result, error) {
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 120*time.Second)
		defer cancel()
	}
	hr, err := http.NewRequestWithContext(ctx, http.MethodPost, f.endpoint, bytes.NewReader(body))
	if err != nil {
		return asr.Result{}, &asr.Error{Kind: asr.KindConfig, Provider: ID, Detail: "bad endpoint", Err: err}
	}
	hr.Header = authHeader(f.appID, f.token)
	hr.Header.Set("X-Api-Resource-Id", ModelFlash)
	hr.Header.Set("X-Api-Request-Id", newUUID())
	hr.Header.Set("X-Api-Sequence", "-1")
	hr.Header.Set("Content-Type", "application/json")
	hr.Header.Set("Accept", "application/json")
	hr.Header.Set("User-Agent", "Voxt")
	resp, err := f.http.Do(hr)
	if err != nil {
		return asr.Result{}, asr.NetError(ctx, ID, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return asr.Result{}, asr.NetError(ctx, ID, err)
	}
	return flashAnswer(resp.StatusCode, resp.Header, raw)
}

// flashAnswer reads the service's answer. X-Api-Status-Code says how it
// went (the HTTP status only when that header is missing); the message
// is X-Api-Message, else the body.
func flashAnswer(status int, h http.Header, raw []byte) (asr.Result, error) {
	logID := h.Get("X-Tt-Logid")
	code, noCode := strconv.ParseUint(h.Get("X-Api-Status-Code"), 10, 32)
	switch {
	case noCode != nil && status/100 != 2:
		c, msg := asr.ErrorBody(raw)
		if msg == "" {
			msg = http.StatusText(status)
		}
		return asr.Result{}, withLog(handshakeError(asr.HTTPError(ID, status, c, msg)), logID)
	case noCode == nil && code == codeSilence:
		return asr.Result{}, nil
	case noCode == nil && code != codeSuccess:
		msg := []byte(h.Get("X-Api-Message"))
		if len(msg) == 0 {
			msg = raw
		}
		return asr.Result{}, withLog(codeError(uint32(code), msg), logID)
	}
	p, err := parseResponse(raw)
	if err != nil {
		return asr.Result{}, withLog(err, logID)
	}
	return asr.Result{Text: strings.TrimSpace(p.Text), Duration: time.Duration(p.DurationMS) * time.Millisecond}, nil
}
