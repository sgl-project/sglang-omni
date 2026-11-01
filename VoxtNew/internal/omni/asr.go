package omni

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
)

// ASR is a curated speech model on the runtime: an asr.FileTranscriber
// and an asr.Streamer. It is cheap: the model's server starts on first
// use (or Warm) and is shared by every ASR of the same model.
type ASR struct {
	rt   *Runtime
	spec Spec
	dir  string
	vad  *VAD // the voice activity model beside it, nil for none
}

var (
	_ asr.FileTranscriber = (*ASR)(nil)
	_ asr.Streamer        = (*ASR)(nil)
)

// NewASR is speech model modelID (ASRAccurate, ASRFast), installed in
// modelDir (its config.json, weights and tokenizer), on rt. It checks the
// id and that the directory exists; nothing starts yet.
func NewASR(rt *Runtime, modelID, modelDir string) (*ASR, error) {
	spec, ok := Lookup(modelID)
	if !ok || spec.ASR == nil {
		return nil, &asr.Error{Kind: asr.KindConfig, Provider: Provider, Detail: fmt.Sprintf("%q is not a speech model", modelID)}
	}
	if err := checkDir(modelDir); err != nil {
		return nil, err
	}
	return &ASR{rt: rt, spec: spec, dir: modelDir}, nil
}

func checkDir(dir string) error {
	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
		return &asr.Error{Kind: asr.KindConfig, Provider: Provider, Detail: "the model's folder is missing: " + dir, Err: err}
	}
	return nil
}

// WithVAD is a copy of a that comes with v, Silero (nil: none), for the
// session's speech detection (VAD, local.VoiceActivity).
func (a *ASR) WithVAD(v *VAD) *ASR {
	c := *a
	c.vad = v
	return &c
}

// Pieces is the incremental Final's policy (ASRTuning's PiecesAfter,
// PieceMin, PieceSeconds and PieceMax): After 0 never cuts.
type Pieces struct {
	After, Min, Target, Max time.Duration
}

// WithPieces is a copy of a whose live sessions are recognised in pieces
// as p says (zero fields keep the model's), the eval's and tests' switch;
// p.After < 0: never.
func (a *ASR) WithPieces(p Pieces) *ASR {
	c := *a
	c.spec = a.spec.clone()
	t := c.spec.ASR
	set := func(d *time.Duration, v time.Duration) {
		if v != 0 {
			*d = v
		}
	}
	set(&t.PiecesAfter, p.After)
	set(&t.PieceMin, p.Min)
	set(&t.PieceSeconds, p.Target)
	set(&t.PieceMax, p.Max)
	if t.PiecesAfter < 0 {
		t.PiecesAfter = 0
	}
	return &c
}

// VAD is the voice activity model that comes with it, nil for none.
func (a *ASR) VAD() vad.Model {
	if a.vad == nil {
		return nil
	}
	return a.vad
}

// Spec is the model's.
func (a *ASR) Spec() Spec { return a.spec.clone() }

// Warm starts the model's server in the background (a session starting:
// the model loads while the user speaks).
func (a *ASR) Warm() { a.rt.Warm(a.spec, a.dir) }

// Caps: it streams (a live preview, then the Final) and transcribes
// files; it takes hints (the context bias) and a prompt.
func (a *ASR) Caps() asr.Caps {
	return asr.Caps{Stream: true, File: true, Prompt: true, Hints: true}
}

// Transcribe is a Final over a whole recording.
func (a *ASR) Transcribe(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	pcm, err := filePCM(req)
	if err != nil {
		return asr.Result{}, err
	}
	l, err := a.rt.acquire(ctx, a.spec, a.dir)
	if err != nil {
		return asr.Result{}, err
	}
	defer l.release()
	start := time.Now()
	text, language, err := a.final(ctx, l, pcm, a.spec.languageHint(req.Lang), a.spec.ASR.contextBias(req.Hints, req.Prompt))
	if err != nil {
		return asr.Result{}, err
	}
	return asr.Result{Text: text, Lang: language, Duration: asr.Duration(pcm, sampleRate), Model: a.spec.ID, Latency: time.Since(start)}, nil
}

// filePCM is a request's audio as PCM16 mono at 16 kHz.
func filePCM(req asr.FileRequest) ([]byte, error) {
	switch req.Format {
	case "wav":
		pcm, rate, ch, err := asr.ReadWAV(req.Audio)
		if err != nil {
			return nil, &asr.Error{Kind: asr.KindAudio, Provider: Provider, Detail: err.Error()}
		}
		if rate != sampleRate || ch != 1 {
			return nil, &asr.Error{Kind: asr.KindAudio, Provider: Provider, Detail: fmt.Sprintf("audio must be 16 kHz mono, not %d Hz × %d", rate, ch)}
		}
		return pcm, nil
	case "pcm", "":
		if req.SampleRate != 0 && req.SampleRate != sampleRate {
			return nil, &asr.Error{Kind: asr.KindAudio, Provider: Provider, Detail: fmt.Sprintf("audio must be 16 kHz, not %d Hz", req.SampleRate)}
		}
		return req.Audio, nil
	}
	return nil, &asr.Error{Kind: asr.KindAudio, Provider: Provider, Detail: "unknown audio format " + req.Format}
}

// final is Voxt's Qwen3-ASR Final (transcribeQwenFinal) with language,
// the model's ("" detects), and the context bias: the chunks share one
// token budget, and the first chunk's detected language is forced on the
// rest. It returns the text and the language.
func (a *ASR) final(ctx context.Context, l *lease, pcm []byte, language string, b bias) (string, string, error) {
	t := a.spec.ASR
	budget := t.tokenBudget(asr.Duration(pcm, sampleRate))
	var text strings.Builder
	for _, c := range t.splitChunks(pcm) {
		if budget <= 0 {
			break
		}
		req := finalRequest{pcm: c.pcm, language: language, context: b.context, maxTokens: budget, layout: t.AudioLayout}
		r, err := transcribe(ctx, l, req, a.rt.cfg.StallTimeout)
		if err != nil {
			return "", "", err
		}
		r.text = a.trimEcho(ctx, l, req, r.text, r.length, b.terms)
		budget -= r.tokens
		if language == "" {
			language = r.language
		}
		text.WriteString(r.text)
	}
	return strings.TrimSpace(text.String()), language, nil
}

// Open starts a live session: partial text while audio comes in, then
// at Finish the Final over everything sent.
func (a *ASR) Open(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	if req.SampleRate != 0 && req.SampleRate != sampleRate {
		return nil, &asr.Error{Kind: asr.KindAudio, Provider: Provider, Detail: fmt.Sprintf("audio must be 16 kHz, not %d Hz", req.SampleRate)}
	}
	l, err := a.rt.acquire(ctx, a.spec, a.dir)
	if err != nil {
		return nil, err
	}
	s, err := openLive(ctx, a, l, req)
	if err != nil {
		l.release()
		return nil, err
	}
	return s, nil
}
