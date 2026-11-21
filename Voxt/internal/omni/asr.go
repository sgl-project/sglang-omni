package omni

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

// ASR is a curated speech model on the runtime: an asr.FileTranscriber
// and an asr.Streamer. It is cheap: the model's server starts on first
// use (or Warm) and is shared by every ASR of the same model.
type ASR struct {
	rt        *Runtime
	spec      Spec
	dir       string
	draft     string // the speculative draft's folder, "" none (WithDraft)
	draftBits int    // 4, 6 or 8; Qwen3-ASR defaults to 8
	vad       *VAD   // the voice activity model beside it, nil for none
}

var (
	_ asr.FileTranscriber = (*ASR)(nil)
	_ asr.Streamer        = (*ASR)(nil)
	_ asr.Diarizer        = (*ASR)(nil)
)

// NewASR is speech model modelID (ASRAccurate, ASRFast, …), installed in
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

// WithDraft is a copy of a whose server verifies the speculative draft
// installed in dir (--draft-model-path), decoding several tokens a step.
// An empty dir disables it. Changing the draft restarts the model (Runtime.acquire).
func (a *ASR) WithDraft(dir string) *ASR {
	c := *a
	c.draft = dir
	if c.spec.Kind == kindQwen3ASR {
		c.draftBits = 8
	}
	return &c
}

// WithDraftBits is WithDraft with the draft's load-time quantisation.
// Only the native servers' supported 4, 6 and 8 bit drafts are accepted.
func (a *ASR) WithDraftBits(dir string, bits int) (*ASR, error) {
	if dir == "" || bits != 4 && bits != 6 && bits != 8 {
		return nil, &asr.Error{Kind: asr.KindConfig, Provider: Provider, Detail: "a draft needs a folder and 4, 6 or 8 bits"}
	}
	if a.spec.Kind != kindQwen3ASR && a.spec.Kind != kindMoss {
		return nil, &asr.Error{Kind: asr.KindConfig, Provider: Provider, Detail: "this speech model has no speculative draft support"}
	}
	c := *a
	c.draft = dir
	c.draftBits = bits
	return &c, nil
}

// Tuning overrides a model's existing request and server parameters.
// Zero fields keep the model's defaults; a nil CacheLimitMB keeps its
// cache policy, while a pointer to zero passes --cache-limit-mb 0.
type Tuning struct {
	DecodeInterval time.Duration
	CacheLimitMB   *int
}

// WithTuning returns a copy with the supported overrides applied. It
// rejects overrides the model's native server would otherwise ignore.
func (a *ASR) WithTuning(t Tuning) (*ASR, error) {
	if t.DecodeInterval < 0 || t.CacheLimitMB != nil && *t.CacheLimitMB < 0 {
		return nil, &asr.Error{Kind: asr.KindConfig, Provider: Provider, Detail: "decode interval and cache limit cannot be negative"}
	}
	if a.spec.Kind != kindQwen3ASR && (t.DecodeInterval != 0 || t.CacheLimitMB != nil) {
		return nil, &asr.Error{Kind: asr.KindConfig, Provider: Provider, Detail: "decode interval and cache limit are supported only by Qwen3-ASR"}
	}
	c := *a
	c.spec = a.spec.clone()
	if t.DecodeInterval != 0 {
		c.spec.ASR.DecodeInterval = t.DecodeInterval
	}
	if t.CacheLimitMB != nil {
		c.spec.ASR.CacheMB = *t.CacheLimitMB
		c.spec.ASR.CacheLimitSet = true
	}
	return &c, nil
}

// WithInstruction is a copy of a whose requests and live sessions carry
// instruction as the prompt instead of the model's own Instruction: the
// evals run a model that takes no context at another of its tasks
// (MOSS's diarization, "" for the server's built-in task). A model with
// a context ignores it (its prompt is the request's).
func (a *ASR) WithInstruction(instruction string) *ASR {
	c := *a
	c.spec = a.spec.clone()
	c.spec.ASR.Instruction = instruction
	return &c
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
func (a *ASR) Warm() { a.rt.warm(a.spec, a.dir, a.draft, a.draftBits) }

// Hold keeps this model's server loaded until release (meeting capture).
func (a *ASR) Hold(ctx context.Context) (func(), error) {
	return a.rt.Hold(ctx, HoldTarget{Spec: a.spec, Dir: a.dir, Draft: a.draft, DraftBits: a.draftBits})
}

// Caps: it streams (a live preview, then the Final) and transcribes
// files; a model with a context takes hints (the context bias) and a
// prompt.
func (a *ASR) Caps() asr.Caps {
	context := a.spec.ASR.ContextTerms > 0
	return asr.Caps{Stream: true, File: true, Prompt: context, Hints: context}
}

// Transcribe is a Final over a whole recording.
func (a *ASR) Transcribe(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	pcm, err := filePCM(req)
	if err != nil {
		return asr.Result{}, err
	}
	acquireStart := time.Now()
	l, err := a.rt.acquireDraft(ctx, a.spec, a.dir, a.draft, a.draftBits)
	acquire := time.Since(acquireStart)
	if err != nil {
		return asr.Result{}, err
	}
	defer l.release()
	start := time.Now()
	text, language, draft, err := a.final(ctx, l, pcm, a.spec.languageHint(req.Lang), a.spec.ASR.contextBias(req.Hints, req.Prompt))
	if err != nil {
		return asr.Result{}, err
	}
	return asr.Result{Text: text, Lang: language, Duration: asr.Duration(pcm, sampleRate), Model: a.spec.ID, Acquire: acquire, Latency: time.Since(start), Draft: draft}, nil
}

// Diarize transcribes a whole recording with MOSS's default speaker prompt.
// One request keeps speaker identities stable across Voxt's usual Final chunks.
func (a *ASR) Diarize(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	if a.spec.Kind != kindMoss {
		return asr.Result{}, &asr.Error{Kind: asr.KindConfig, Provider: Provider, Detail: "speaker diarization needs MOSS-Transcribe-Diarize"}
	}
	pcm, err := filePCM(req)
	if err != nil {
		return asr.Result{}, err
	}
	acquireStart := time.Now()
	l, err := a.rt.acquireDraft(ctx, a.spec, a.dir, a.draft, a.draftBits)
	acquire := time.Since(acquireStart)
	if err != nil {
		return asr.Result{}, err
	}
	defer l.release()
	start := time.Now()
	if a.noSpeech(ctx, pcm) {
		return asr.Result{Duration: asr.Duration(pcm, sampleRate), Model: a.spec.ID, Acquire: acquire, Latency: time.Since(start)}, nil
	}
	r, err := transcribe(ctx, l, finalRequest{pcm: pcm, language: a.spec.languageHint(req.Lang), context: req.Prompt, layout: a.spec.ASR.AudioLayout}, a.rt.cfg.StallTimeout)
	if err != nil {
		return asr.Result{}, err
	}
	if r.length {
		return asr.Result{}, &asr.Error{Kind: asr.KindServer, Provider: Provider, Detail: "speaker diarization reached the token limit", Retryable: true}
	}
	// The tags are in the segments' times and speakers; a text the server
	// could not split into segments still carries them.
	res := asr.Result{Text: stripTags(r.text), Lang: r.language, Duration: asr.Duration(pcm, sampleRate), Model: a.spec.ID, Acquire: acquire, Latency: time.Since(start), Draft: r.draft}
	for _, seg := range r.segments {
		res.Segments = append(res.Segments, asr.SpeakerSegment{
			Start: time.Duration(seg.Start * float64(time.Second)), End: time.Duration(seg.End * float64(time.Second)),
			Speaker: seg.Speaker, Text: stripTags(seg.Text),
		})
	}
	return res, nil
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

// final transcribes quiet-cut chunks in order, with a shared token budget
// for Qwen3-ASR and one budget per Whisper window. The first detected
// language, when available, is passed to subsequent requests. It also
// returns the accumulated draft counters.
func (a *ASR) final(ctx context.Context, l *lease, pcm []byte, language string, b bias) (string, string, *asr.DraftStats, error) {
	t := a.spec.ASR
	budget := t.tokenBudget(asr.Duration(pcm, sampleRate))
	shared := t.TokensPerSecond > 0 // else each request has the whole budget
	batchWhisper := a.spec.Kind == kindWhisper && a.draft == ""
	chunks := t.splitChunks(pcm)
	if a.spec.Kind == kindWhisper {
		chunks = trimWhisperSilenceTail(chunks)
	}
	var parts []string
	var draft *asr.DraftStats
	for len(chunks) > 0 {
		if shared && budget <= 0 {
			break
		}
		if batchWhisper {
			batchPCM, chunkWindows := packWhisperChunks(chunks)
			if len(chunkWindows) > 1 {
				req := finalRequest{pcm: batchPCM, language: language, maxTokens: budget}
				texts, detected, err := a.transcribeWhisperBatch(ctx, l, req, chunkWindows)
				if err != nil {
					return "", "", nil, err
				}
				if texts != nil {
					parts = append(parts, texts...)
					if language == "" {
						language = detected
					}
					chunks = chunks[len(chunkWindows):]
					continue
				}
				// A mismatched older runtime cannot split this first batch;
				// redo it per chunk and avoid batching the remaining chunks.
				batchWhisper = false
			}
		}
		c := chunks[0]
		chunks = chunks[1:]
		pcm, ok := a.speechOnly(ctx, c.pcm)
		if !ok {
			continue
		}
		req := finalRequest{pcm: pcm, language: language, context: b.context, maxTokens: budget, layout: t.AudioLayout}
		r, err := transcribe(ctx, l, req, a.rt.cfg.StallTimeout)
		if err != nil {
			return "", "", nil, err
		}
		r.text = a.trimEcho(ctx, l, req, a.plainText(r.text), r.length, b.terms)
		if shared {
			budget -= r.tokens
		}
		if language == "" {
			language = r.language
		}
		if r.draft != nil {
			if draft == nil {
				draft = &asr.DraftStats{}
			}
			draft.VerifyRounds += r.draft.VerifyRounds
			draft.Proposed += r.draft.Proposed
			draft.Accepted += r.draft.Accepted
		}
		parts = append(parts, r.text)
	}
	return joinParts(parts), language, draft, nil
}

// transcribeWhisperBatch reassembles each quiet-cut chunk before joinParts
// joins the chunks. The runtime uses spaces between windows of one chunk.
func (a *ASR) transcribeWhisperBatch(ctx context.Context, l *lease, req finalRequest, chunkWindows []int) ([]string, string, error) {
	r, err := transcribe(ctx, l, req, a.rt.cfg.StallTimeout)
	if err != nil || r.windowTexts == nil {
		return nil, "", err
	}
	if len(r.windowTexts) != len(req.pcm)/whisperWindowBytes {
		return nil, "", &asr.Error{Kind: asr.KindProtocol, Provider: Provider, Detail: "the Whisper batch returned the wrong number of windows"}
	}
	parts := make([]string, 0, len(chunkWindows))
	for _, count := range chunkWindows {
		var windows []string
		for _, text := range r.windowTexts[:count] {
			if text != "" {
				windows = append(windows, text)
			}
		}
		parts = append(parts, strings.Join(windows, " "))
		r.windowTexts = r.windowTexts[count:]
	}
	return parts, r.language, nil
}

// Open starts a live session: partial text while audio comes in, then
// at Finish the Final over everything sent.
func (a *ASR) Open(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	if req.SampleRate != 0 && req.SampleRate != sampleRate {
		return nil, &asr.Error{Kind: asr.KindAudio, Provider: Provider, Detail: fmt.Sprintf("audio must be 16 kHz, not %d Hz", req.SampleRate)}
	}
	l, err := a.rt.acquireDraft(ctx, a.spec, a.dir, a.draft, a.draftBits)
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
