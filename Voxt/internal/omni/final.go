package omni

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// loopback is the client of the servers: never through a proxy, no
// overall timeout (a Final takes as long as the audio needs; contexts and
// the stall watch bound it).
var loopback = &http.Client{Transport: &http.Transport{
	Proxy:               nil,
	MaxIdleConnsPerHost: 4,
	IdleConnTimeout:     30 * time.Second,
}}

// finalRequest is one request of a Final.
type finalRequest struct {
	pcm       []byte // PCM16 mono at 16 kHz
	language  string // "" detects
	context   string
	maxTokens int
	layout    string
	preview   bool // a provisional request with faster disconnect detection
	piece     bool // a long session's incremental Final (pieces.go)
}

// finalResult is a request's answer.
type finalResult struct {
	text     string
	tokens   int    // generated
	language string // detected (or the hint), "" when none
	length   bool   // the budget ended it
	draft    *asr.DraftStats
	segments []struct {
		Start   float64 `json:"start"`
		End     float64 `json:"end"`
		Speaker string  `json:"speaker"`
		Text    string  `json:"text"`
	}
	windowTexts []string
}

// body is the request's multipart form (Voxt's OmniMultipartBody for
// Qwen3-ASR: streamed, with the generation metadata).
func (r finalRequest) body(model string) (*bytes.Buffer, string) {
	var b bytes.Buffer
	w := multipart.NewWriter(&b)
	field := func(k, v string) { _ = w.WriteField(k, v) }
	field("model", model)
	field("stream", "true")
	field("response_format", "json")
	if r.language != "" {
		field("language", r.language)
	}
	if r.context != "" {
		field("prompt", r.context)
	}
	if r.maxTokens > 0 {
		field("max_new_tokens", itoa(r.maxTokens))
	}
	field("stop_at_end_of_text", "true")
	field("stop_on_token_loop", "true")
	field("include_generation_metadata", "true")
	if r.preview {
		field("preview", "true")
	}
	if r.layout != "" {
		field("audio_layout", r.layout)
	}
	fw, _ := w.CreateFormFile("file", "audio.wav")
	_, _ = fw.Write(asr.WAV(r.pcm, sampleRate))
	_ = w.Close()
	return &b, w.FormDataContentType()
}

// transcribe sends one request on l's server and reads its event stream.
// A stream silent for stall (the server's heartbeat is every 250 ms) has
// hung the server: it is killed.
func transcribe(ctx context.Context, l *lease, r finalRequest, stall time.Duration) (finalResult, error) {
	if p := decodeProbeOf(ctx); p != nil {
		kind := r.decodeKind()
		ahead := int(p.inflight.Add(1) - 1)
		submitted := time.Now()
		res, err := transcribeRound(ctx, l, r, stall)
		finished := time.Now()
		p.inflight.Add(-1)
		ev := decodeEvent{
			Kind:        kind,
			RequestID:   int(p.nextID.Add(1)),
			SubmittedMS: submitted.UnixMilli(),
			FinishedMS:  finished.UnixMilli(),
			Inflight:    ahead,
			TailSamples: len(r.pcm) / 2,
		}
		p.record(ev)
		return res, err
	}
	return transcribeRound(ctx, l, r, stall)
}

func transcribeRound(ctx context.Context, l *lease, r finalRequest, stall time.Duration) (finalResult, error) {
	body, ctype := r.body(l.endpoint().model)
	// The hint goes to the log, never the recognised text: a wrong hint
	// translated Chinese dictation (#3398) and nothing showed it.
	if !r.preview {
		l.s.rt.logf("info", "%s: Final request with language hint %q", l.s.spec.ID, r.language)
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.endpoint().base()+"/v1/audio/transcriptions", body)
	if err != nil {
		return finalResult{}, &asr.Error{Kind: asr.KindProtocol, Provider: Provider, Err: err}
	}
	req.Header.Set("Content-Type", ctype)
	var stalled atomic.Bool
	watch := time.AfterFunc(stall, func() { stalled.Store(true); cancel() })
	defer watch.Stop()
	resp, err := loopback.Do(req)
	if err != nil {
		return finalResult{}, stallOr(ctx, l, stalled.Load(), err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		code, msg := asr.ErrorBody(raw)
		e := asr.HTTPError(Provider, resp.StatusCode, code, msg)
		if resp.StatusCode == http.StatusBadRequest {
			e.Kind = asr.KindProtocol
		}
		return finalResult{}, e
	}
	rd := bufio.NewReaderSize(resp.Body, 64<<10)
	var done *finalResult
	for {
		watch.Reset(stall)
		line, err := rd.ReadString('\n')
		if err != nil {
			if done != nil && errors.Is(err, io.EOF) {
				return *done, nil
			}
			if errors.Is(err, io.EOF) {
				return finalResult{}, &asr.Error{Kind: asr.KindServer, Provider: Provider, Detail: "the transcription ended before its result", Retryable: true}
			}
			return finalResult{}, stallOr(ctx, l, stalled.Load(), err)
		}
		payload, ok := strings.CutPrefix(strings.TrimRight(line, "\r\n"), "data:")
		if !ok {
			continue // a heartbeat (":") or a blank line
		}
		payload = strings.TrimSpace(payload)
		if payload == "[DONE]" {
			if done == nil {
				return finalResult{}, &asr.Error{Kind: asr.KindServer, Provider: Provider, Detail: "the transcription ended before its result", Retryable: true}
			}
			return *done, nil
		}
		res, err := parseEvent(payload)
		if err != nil {
			return finalResult{}, err
		}
		if res != nil {
			done = res
		}
	}
}

// parseEvent reads one server-sent event: the result for
// transcript.text.done, nil for others, an error for "error". Only the
// error's code is kept: messages may echo the request.
func parseEvent(payload string) (*finalResult, error) {
	var ev struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		Segments []struct {
			Start   float64 `json:"start"`
			End     float64 `json:"end"`
			Speaker string  `json:"speaker"`
			Text    string  `json:"text"`
		} `json:"segments"`
		Metadata *struct {
			Tokens      *int     `json:"generated_token_count"`
			Language    *string  `json:"language"`
			Finish      string   `json:"finish_reason"`
			Verify      *int     `json:"verify_round_count"`
			Proposed    *int     `json:"proposed_draft_token_count"`
			Accepted    *int     `json:"accepted_draft_token_count"`
			WindowTexts []string `json:"window_texts"`
		} `json:"generation_metadata"`
		Error *struct {
			Type string `json:"type"`
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(payload), &ev); err != nil || ev.Type == "" {
		return nil, &asr.Error{Kind: asr.KindProtocol, Provider: Provider, Detail: "an unreadable transcription event"}
	}
	switch ev.Type {
	case "transcript.text.done":
		r := &finalResult{text: ev.Text, segments: ev.Segments}
		if m := ev.Metadata; m != nil {
			if m.Tokens == nil || m.Finish == "" {
				return nil, &asr.Error{Kind: asr.KindProtocol, Provider: Provider, Detail: "the generation metadata is incomplete"}
			}
			r.tokens, r.length = *m.Tokens, m.Finish == "length"
			r.windowTexts = m.WindowTexts
			if m.Language != nil {
				r.language = *m.Language
			}
			if m.Verify != nil && *m.Verify > 0 && m.Proposed != nil && m.Accepted != nil {
				r.draft = &asr.DraftStats{VerifyRounds: *m.Verify, Proposed: *m.Proposed, Accepted: *m.Accepted}
			}
		}
		return r, nil
	case "error":
		code := "error"
		if ev.Error != nil && ev.Error.Code != "" {
			code = ev.Error.Code
		}
		return nil, &asr.Error{Kind: asr.KindServer, Provider: Provider, Code: code, Detail: "the transcription failed"}
	}
	return nil, nil
}

// stallOr is a transport error's *asr.Error, killing a stalled server.
func stallOr(ctx context.Context, l *lease, stalled bool, err error) error {
	if stalled {
		l.broken("no answer")
		return &asr.Error{Kind: asr.KindServer, Provider: Provider, Detail: "the local speech model stopped answering", Retryable: true, Err: err}
	}
	// A server that crashed breaks the connection a moment before its
	// exit is seen.
	if ctx.Err() == nil && l.p.wait(500*time.Millisecond) {
		return &asr.Error{Kind: asr.KindServer, Provider: Provider, Detail: fmt.Sprintf("the local speech model exited (%v)", l.p.exitErr), Retryable: true, Err: err}
	}
	return asr.NetError(ctx, Provider, err)
}
