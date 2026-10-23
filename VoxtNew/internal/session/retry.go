package session

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// RetryRequest asks for a history entry to be made again from its
// recording.
type RetryRequest struct {
	Entry store.Entry
	// OnStage is told when recognition (Recognizing) and the language
	// model (Composing) start, on Retry's goroutine; nil: nobody.
	OnStage func(State)
}

// ErrNoRecording: the entry kept no recording to retry.
var ErrNoRecording = errors.New("session: the entry kept no recording")

// Retry makes a history entry again from its recording (behavior.md
// 5.20a): recognition as a whole recording, then the feature's language
// model step with the settings as they are now (Transcribe's clean-up,
// Translate into the entry's target, Rewrite of the entry's selection),
// with the entry's app enhancement group. Nothing is inserted. The entry
// is updated in place: done with its new texts and times, or failed with
// the new error. A success drops a recording kept only for the retry
// (store.Entry.AudioUntil) when the settings keep no recordings. It
// runs beside any session (it never takes the Manager's), blocking; it
// returns the entry as written.
func (m *Manager) Retry(ctx context.Context, req RetryRequest) (store.Entry, error) {
	cfg := m.cfg
	e := req.Entry
	if cfg.History == nil {
		return e, errors.New("session: no history")
	}
	if e.Audio == "" {
		return e, ErrNoRecording
	}
	raw, err := os.ReadFile(cfg.History.AudioPath(e.Audio))
	if err != nil {
		return e, fmt.Errorf("%w: %v", ErrNoRecording, err)
	}
	pcm, rate, ch, err := asr.ReadWAV(raw)
	if err != nil || rate != asr.SampleRate || ch != 1 {
		return e, fmt.Errorf("%w: not a 16 kHz mono recording (%v)", ErrNoRecording, err)
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	r := &run{m: m, id: e.ID, req: StartRequest{Kind: e.Kind}, settings: cfg.Settings(), ctx: ctx, cancel: cancel,
		stop: make(chan struct{}), pcm: newPCMBuffer(), target: e.TargetLang}
	r.pcm.append(pcm)
	r.retryScene(e)
	stage := func(st State) {
		if req.OnStage != nil {
			req.OnStage(st)
		}
	}
	r.logf("info", "retry of a %s entry: %v of audio", e.Kind, asr.Duration(pcm, asr.SampleRate).Round(time.Millisecond))
	start := r.now()
	out, err := r.retry(e, pcm, stage)
	out.TotalMS = r.now().Sub(start).Milliseconds()
	if err != nil {
		r.logf("warning", "retry failed: %v", err)
		out.Status, out.Error = store.StatusFailed, err.Error()
	} else {
		out.Status = store.StatusDone // Error: a clean-up's failure, the text going on as recognised
		if !out.AudioUntil.IsZero() {
			if !r.settings.Privacy.KeepAudio {
				_ = os.Remove(cfg.History.AudioPath(out.Audio))
				out.Audio = ""
			}
			out.AudioUntil = time.Time{}
		}
		r.logf("info", "retry done in %v", time.Duration(out.TotalMS)*time.Millisecond)
	}
	pctx, pcancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer pcancel()
	if perr := cfg.History.Put(pctx, out); perr != nil {
		r.logf("error", "history: %v", perr)
		if err == nil {
			err = perr
		}
	}
	return out, err
}

// retryScene decides a retry's app enhancement group from what its entry
// recorded: the app, and the host of a browser's tab.
func (r *run) retryScene(e store.Entry) {
	r.sc.done = make(chan struct{})
	app := proto.App{BundleID: e.AppBundle, Name: e.AppName}
	r.req.Target = app
	global := appctx.Match{Source: appctx.SourceGlobal, App: app}
	cfg := r.m.cfg.Scene
	if cfg == nil || cfg.Rules == nil || e.AppBundle == "" {
		r.setScene(global, false)
		return
	}
	rules, on := cfg.Rules()
	if !on {
		r.setScene(global, false)
		return
	}
	in := appctx.Context{App: app}
	if e.URLHost != "" {
		in.Browser, in.URL = true, "https://"+e.URLHost
	}
	r.setScene(appctx.Resolve(rules, in, string(e.Kind)), false)
}

// retry is Retry's work: e with its new texts, models and times, or the
// first error (e as far as it got).
func (r *run) retry(e store.Entry, pcm []byte, stage func(State)) (store.Entry, error) {
	cfg := r.m.cfg
	svc, err := cfg.Resolve(r.ctx, e.Kind, r.settings)
	if err != nil {
		return e, err
	}
	r.svc = svc
	if r.settings.Dictionary.Enabled && cfg.Glossary != nil {
		r.terms, _ = cfg.Glossary(r.ctx)
	}
	if p := r.lacks(); p != nil {
		return e, p.Err
	}
	ft, ok := svc.ASR.(asr.FileTranscriber)
	if !ok {
		return e, &asr.Error{Kind: asr.KindConfig, Detail: "the speech model does not take a whole recording"}
	}
	stage(Recognizing)
	e.Error = ""
	e.ASRProvider, e.ASRModel = svc.ASRInfo.ID, svc.ASRInfo.Model
	e.LLMProvider, e.LLMModel, e.LLMMS = "", "", 0
	asrStart := r.now()
	upload, speech := r.prepare(pcm)
	e.DSP, e.GainDB = r.dsp, r.gainDB
	if !speech {
		e.SentMS = 0
		return e, errors.New("no speech was heard in the recording")
	}
	e.SentMS = asr.Duration(upload, asr.SampleRate).Milliseconds()
	r.asrLimit = cfg.asrLimit(asr.Duration(pcm, asr.SampleRate))
	actx, acancel := context.WithTimeout(r.ctx, r.asrLimit)
	res, err := ft.Transcribe(actx, asr.FileRequest{Audio: upload, Format: "pcm", SampleRate: asr.SampleRate, Lang: r.speechLang(), Hints: r.hints()})
	if err != nil && actx.Err() != nil {
		err = r.ctxErr(actx)
	}
	acancel()
	e.ASRMS = r.now().Sub(asrStart).Milliseconds()
	if err != nil {
		return e, err
	}
	raw := strings.TrimSpace(res.Text)
	r.logf("info", "retry: recognised %s in %v", r.logText(raw), time.Duration(e.ASRMS)*time.Millisecond)
	if raw == "" {
		return e, errors.New("nothing was recognised in the recording")
	}
	text := r.spoken(raw)
	if e.Kind == store.KindRewrite {
		e.PromptText = raw
	} else {
		e.RawText = raw
	}
	e.Result = text
	if e.Kind == store.KindTranscribe && (!r.settings.Features.Transcribe.Enhance || svc.LLM == nil) {
		return e, nil
	}
	stage(Composing)
	llmStart := r.now()
	var llmErr error
	if e.Kind == store.KindTranscribe {
		e.Result, _, llmErr = r.cleanUp(text)
		if llmErr != nil {
			// As a session: the text as recognised, with the warning.
			e.LLMMS = r.now().Sub(llmStart).Milliseconds()
			e.LLMProvider, e.LLMModel = svc.LLMInfo.ID, svc.LLMInfo.Model
			e.Error = llmErr.Error()
			return e, nil
		}
	} else {
		var ans compose.Answer
		ans, _, llmErr = r.ask(r.writeInput(text, e.SourceText))
		e.Result = strings.TrimSpace(ans.Content)
		if llmErr == nil && e.Result == "" {
			llmErr = errors.New("the language model answered nothing")
		}
	}
	e.LLMMS = r.now().Sub(llmStart).Milliseconds()
	e.LLMProvider, e.LLMModel = svc.LLMInfo.ID, svc.LLMInfo.Model
	return e, llmErr
}
