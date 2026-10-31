package session

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/deliver"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// run is one session. Its fields are the session goroutine's, but for
// ended (Manager.mu) and what is atomic or a channel.
type run struct {
	m        *Manager
	id       string
	req      StartRequest
	settings settings.Settings

	ctx       context.Context // the session's: cancelled by Cancel and at the end
	cancel    context.CancelFunc
	cancelled atomic.Bool
	stop      chan struct{}
	stopOnce  sync.Once
	pcm       *pcmBuffer
	ended     bool // guarded by m.mu

	svc   Services
	terms []store.Term
	// svcDone gets the services, resolved meanwhile; resolved once r.svc
	// holds them.
	svcDone  chan svcResult
	resolved bool
	sc       sceneState // app enhancement (scene.go)
	// target is Translate's target language (a code).
	target string
	// asrLimit is recognition's limit after Stop (Config.asrLimit).
	asrLimit time.Duration
	// recognitionFailed: the Final failed or found no text; the history
	// entry keeps the recording for a while (keep), for Retry.
	recognitionFailed bool
	// asrDegraded: the Final failed and the session goes on with the
	// text the stream had (listen); its notice wins over a clean-up's.
	asrDegraded *Problem
	// opening: the speech model is still starting (wait.go); waitDue:
	// Config.WaitNote has passed since the start.
	opening, waitDue atomic.Bool
	// dsp and gainDB are what the sound processing did (noteDSP), for a
	// retry's entry.
	dsp    string
	gainDB float64
}

type capResult struct {
	res proto.CaptureResult
	err error
}

type svcResult struct {
	svc   Services
	terms []store.Term
	err   error
}

type openResult struct {
	st  asr.Stream
	err error
}

// streamEnd is a stream's last event: its final result or its error;
// silent when the recording ended with no speech, so nothing was sent
// (the gate held it all) and no result will come.
type streamEnd struct {
	res    asr.Result
	err    error
	silent bool
}

func (r *run) logf(level, format string, args ...any) {
	r.m.cfg.Log(level, "session "+r.id+": "+fmt.Sprintf(format, args...))
}

// logText is a text for the log: itself when the settings allow it, else
// its length.
func (r *run) logText(s string) string {
	if r.settings.Privacy.LogText {
		return fmt.Sprintf("%q", s)
	}
	return fmt.Sprintf("%d chars", len([]rune(s)))
}

func (r *run) now() time.Time { return r.m.cfg.Now() }

// firstState is where a session starts: composing with its text,
// reading the selection (Translate), else recording.
func (r *run) firstState() State {
	switch {
	case r.req.Text != nil:
		return Composing
	case r.req.Kind == store.KindTranslate && r.canSelect():
		return Reading
	}
	return Recording
}

// canSelect says whether the session reads the selection: Translate and
// Rewrite with a selector, not a try, not a follow-up, in an app that is
// not VoxtNew.
func (r *run) canSelect() bool {
	cfg := r.m.cfg
	t := r.req.Target
	return cfg.Select != nil && !r.req.Try && len(r.req.Previous) == 0 && t.PID != 0 && (cfg.Self == 0 || t.PID != cfg.Self) &&
		(r.req.Kind == store.KindTranslate || r.req.Kind == store.KindRewrite)
}

// main is the session's goroutine: the selection, recording,
// recognition, composition, delivery, history.
func (r *run) main() {
	defer r.cancel()
	cfg := r.m.cfg
	kind := r.req.Kind
	switch kind {
	case store.KindTranscribe, store.KindTranslate, store.KindRewrite:
	default:
		r.fail(&Problem{Code: ProblemUnsupported, Err: fmt.Errorf("unknown feature %q", kind)})
		return
	}
	r.logf("info", "start %s in %s", kind, r.req.Target.BundleID)

	// The services and the dictionary, meanwhile.
	r.svcDone = make(chan svcResult, 1)
	go func() {
		svc, err := cfg.Resolve(r.ctx, kind, r.settings)
		var terms []store.Term
		if err == nil && r.settings.Dictionary.Enabled && cfg.Glossary != nil {
			terms, _ = cfg.Glossary(r.ctx)
		}
		r.svcDone <- svcResult{svc, terms, err}
	}()

	if t := r.req.Text; t != nil {
		if r.awaitServices() {
			r.write(r.spoken(strings.TrimSpace(t.Raw)), t.Selection)
		}
		return
	}
	var selDone chan string
	switch {
	case kind == store.KindTranslate && r.canSelect():
		sel := r.readSelection()
		if r.ctx.Err() != nil {
			r.endCancelled(nil)
			return
		}
		if sel != "" {
			if r.awaitServices() {
				r.write("", sel)
			}
			return
		}
		r.m.update(r, func(s *Snapshot) { s.State = Recording })
	case kind == store.KindRewrite && r.canSelect():
		selDone = make(chan string, 1)
		go func() { selDone <- r.readSelection() }()
	}

	raw, ok := r.listen()
	if !ok {
		return
	}
	text := r.spoken(raw)
	switch kind {
	case store.KindTranscribe:
		r.transcribe(text)
	case store.KindTranslate:
		r.write(text, "")
	case store.KindRewrite:
		sel := ""
		if selDone != nil {
			select {
			case sel = <-selDone:
			case <-r.ctx.Done():
				r.endCancelled(nil)
				return
			}
		}
		r.write(text, sel)
	}
}

// finishOpts say how a result is delivered: card, when set, sends it to
// the answer card without trying to insert it (the reason it shows
// there); replace replaces the selection, only in the app it was made in
// (deliver.Request.Replace).
type finishOpts struct {
	card    deliver.Reason
	replace bool
}

// finish delivers result (unless it is a try or empty), keeps the
// history entry and ends the session.
func (r *run) finish(st State, problem *Problem, result string, o finishOpts) {
	cfg := r.m.cfg
	var dres deliver.Result
	switch {
	case result == "" || r.req.Try:
	case o.card != "":
		dres.Reason = o.card
		r.m.update(r, func(s *Snapshot) { s.Result, s.Card = result, true })
	default:
		r.m.update(r, func(s *Snapshot) { s.State, s.Result = Delivering, result })
		press := r.press(o.replace) // app enhancement's auto key
		ctx, cancel := context.WithTimeout(context.Background(), cfg.DeliverTime)
		dres = deliver.Insert(ctx, cfg.Text, deliver.Request{Text: result, Target: r.req.Target, Press: press, Self: cfg.Self, Activate: cfg.Activate,
			Replace: o.replace, VerifyMS: deliver.DefaultVerifyMS})
		cancel()
		if dres.Err != nil {
			r.logf("warning", "insert: %v", dres.Err)
		}
		r.logf("info", "delivery: %s", dres.Diag())
		switch {
		case dres.Reason == deliver.ReasonUnconfirmed || dres.Reason == deliver.ReasonSecureInput:
			r.logf("warning", "pasted into %s, but the text did not show there (%s): the card shows it, the clipboard holds it", dres.To.BundleID, dres.Reason)
		case dres.Inserted && dres.Moved(r.req.Target):
			r.logf("info", "inserted into %s, not %s (the user switched apps; no auto key) in %v", dres.To.BundleID, r.req.Target.BundleID, dres.Elapsed.Round(time.Millisecond))
		case dres.Inserted:
			r.logf("info", "inserted in %v", dres.Elapsed.Round(time.Millisecond))
		default:
			r.logf("info", "not inserted (%s; %s in front) in %v", dres.Reason, dres.Front.BundleID, dres.Elapsed.Round(time.Millisecond))
		}
	}
	if r.asrDegraded != nil && (problem == nil || problem.Degraded) {
		problem = r.asrDegraded
	}
	snap := r.m.Current()
	total := r.now().Sub(snap.StartedAt)
	r.m.end(r, st, func(s *Snapshot) {
		s.Result, s.Problem = result, problem
		s.Recording = r.keepsRecording()
		s.Inserted = dres.Inserted
		if dres.Inserted {
			s.InsertedInto = dres.To
		}
		s.Card = result != "" && !r.req.Try && !dres.Inserted
		s.Delivery = dres.Reason
		s.Timings.Deliver, s.Timings.Total = dres.Elapsed, total
	}, func(s Snapshot) { r.keep(s, store.StatusDone) })
}

// fail ends the session failed before it recorded anything worth keeping.
func (r *run) fail(p *Problem) {
	r.cancel()
	r.logf("warning", "failed: %s: %v", p.Code, p.Err)
	r.m.end(r, Failed, func(s *Snapshot) { s.Problem = p }, nil)
}

// endFailed ends the session failed after it recorded: the entry keeps
// the recording when the settings say so.
func (r *run) endFailed(p *Problem, stream asr.Stream, more ...func(*Snapshot)) {
	if stream != nil {
		stream.Close()
	}
	r.cancel()
	r.logf("warning", "failed: %s: %v", p.Code, p.Err)
	total := r.now().Sub(r.m.Current().StartedAt)
	audio := asr.Duration(r.pcm.bytes(), asr.SampleRate)
	r.m.end(r, Failed, func(s *Snapshot) {
		s.Problem = p
		s.Timings.Total = total
		s.Timings.Audio = audio
		s.Recording = audio >= r.m.cfg.MinRecording && r.keepsRecording()
		for _, fn := range more {
			fn(s)
		}
	}, func(s Snapshot) {
		if audio >= r.m.cfg.MinRecording {
			r.keep(s, store.StatusFailed)
		}
	})
}

func (r *run) endCancelled(stream asr.Stream) {
	if stream != nil {
		stream.Close()
	}
	r.cancel()
	r.logf("info", "cancelled")
	r.m.end(r, Cancelled, nil, nil)
}

// asrProblem classifies a speech model's error.
func (r *run) asrProblem(err error) *Problem {
	if r.cancelled.Load() {
		return &Problem{Code: ProblemInternal, Err: context.Canceled}
	}
	return &Problem{Code: ProblemASR, Err: err}
}

func (r *run) ctxErr(ctx context.Context) error {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return &asr.Error{Kind: asr.KindTimeout, Detail: fmt.Sprintf("no result within %v (%v of audio)", r.asrLimit, asr.Duration(r.pcm.bytes(), asr.SampleRate).Round(time.Second)), Retryable: true}
	}
	return &asr.Error{Kind: asr.KindCancelled, Detail: "cancelled"}
}

func (r *run) timeoutProblem(ctx context.Context) *Problem {
	return &Problem{Code: ProblemASR, Err: r.ctxErr(ctx)}
}

// keep writes the history entry of an ended session (and its recording),
// when the settings keep history. Not for tries.
func (r *run) keep(s Snapshot, status store.Status) {
	cfg := r.m.cfg
	if r.req.Try || cfg.History == nil || !r.settings.Privacy.KeepHistory {
		return
	}
	e := store.Entry{
		ID: r.id, Kind: r.req.Kind, CreatedAt: s.StartedAt, Status: status,
		AppBundle: s.Target.BundleID, AppName: s.Target.Name,
		RawText: s.Raw, Result: s.Result, SourceText: s.Selection, TargetLang: s.TargetLanguage,
		ASRProvider: s.ASR.ID, ASRModel: s.ASR.Model, LLMProvider: s.LLM.ID, LLMModel: s.LLM.Model,
		AudioMS: s.Timings.Audio.Milliseconds(), SentMS: s.Timings.Sent.Milliseconds(), ASRMS: s.Timings.ASR.Milliseconds(), LLMMS: s.Timings.LLM.Milliseconds(),
		TotalMS: s.Timings.Total.Milliseconds(), DSP: s.DSP, GainDB: s.GainDB,
	}
	if r.req.Kind == store.KindRewrite {
		e.PromptText, e.RawText = s.Raw, "" // the spoken instruction
	}
	if s.Timings.LLM == 0 {
		e.LLMProvider, e.LLMModel = "", ""
	}
	if r.req.Text != nil || s.Timings.Audio == 0 {
		e.ASRProvider, e.ASRModel = "", "" // nothing was recognised
	}
	if m, ok := r.knownScene(); ok {
		e.URLHost = m.Host
		if m.Matched() {
			e.RuleID, e.RuleName, e.RuleSource = m.Group.ID, m.Group.Name, string(m.Source)
		}
	}
	switch {
	case s.Inserted:
		e.Delivery = "insert"
		if to := s.InsertedInto; to.PID != 0 && to.PID != s.Target.PID {
			e.InsertedBundle, e.InsertedName = to.BundleID, to.Name // where it went; the app and the rule stay the start app's
		}
	case s.Card:
		e.Delivery = "card"
	}
	if s.Problem != nil && s.Problem.Err != nil {
		e.Error = s.Problem.Err.Error()
	}
	if s.Recording {
		name := r.id + ".wav"
		dir := cfg.History.AudioDir()
		if err := os.MkdirAll(dir, 0o755); err == nil {
			if err := os.WriteFile(filepath.Join(dir, name), asr.WAV(r.pcm.bytes(), asr.SampleRate), 0o644); err == nil {
				e.Audio = name
				if !r.settings.Privacy.KeepAudio {
					e.AudioUntil = r.now().Add(cfg.AudioRetention)
					r.logf("info", "recognition failed: the recording is kept until %s, for Retry", e.AudioUntil.Format(time.DateTime))
				}
			} else {
				r.logf("warning", "keeping the recording: %v", err)
			}
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := cfg.History.Put(ctx, e); err != nil {
		r.logf("error", "history: %v", err)
	}
}

// keepsRecording reports whether the session's history entry keeps the
// recording: the settings keep recordings, or recognition failed (for
// a while: Config.AudioRetention). Not for tries.
func (r *run) keepsRecording() bool {
	cfg := r.m.cfg
	if r.req.Try || r.req.Text != nil || cfg.History == nil || !r.settings.Privacy.KeepHistory || len(r.pcm.bytes()) == 0 {
		return false
	}
	return r.settings.Privacy.KeepAudio || r.recognitionFailed
}

func isHelperErr(err error) bool { return errors.Is(err, core.ErrHelper) }
