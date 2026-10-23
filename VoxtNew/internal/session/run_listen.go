package session

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// awaitServices waits for the services; false when the session ended
// (cancelled, or the services failed).
func (r *run) awaitServices() bool {
	if r.resolved {
		return true
	}
	select {
	case s := <-r.svcDone:
		if s.err != nil {
			r.fail(ProblemOf(s.err))
			return false
		}
		r.useServices(s)
		return true
	case <-r.ctx.Done():
		r.endCancelled(nil)
		return false
	}
}

// useServices keeps the resolved services.
func (r *run) useServices(s svcResult) {
	r.resolved, r.svc, r.terms = true, s.svc, s.terms
	r.m.update(r, func(sn *Snapshot) { sn.ASR, sn.LLM = s.svc.ASRInfo, s.svc.LLMInfo })
}

// needs reports what the services lack for the session: a speech
// model, or the language model Translate and Rewrite need.
func (r *run) lacks() *Problem {
	if r.svc.ASR == nil {
		return ProblemOf(fmt.Errorf("%s: %w", r.req.Kind, ErrNoASR))
	}
	if r.svc.LLM == nil && r.req.Kind != store.KindTranscribe {
		return &Problem{Code: ProblemNoLLM, Err: errors.New("no language model is ready")}
	}
	return nil
}

// readSelection reads the selected text of the app in front (Translate,
// Rewrite) and puts it in the snapshot; "" for none, or when it could
// not be read (the session goes on without one).
func (r *run) readSelection() string {
	cfg := r.m.cfg
	ctx, cancel := context.WithTimeout(r.ctx, cfg.ReadTimeout)
	defer cancel()
	res, err := cfg.Select.Selection(ctx, proto.SelectionParams{AllowCopy: AllowCopy(r.req.Target.BundleID)})
	if err != nil {
		if r.ctx.Err() == nil {
			r.logf("warning", "reading the selection: %v", err)
		}
		return ""
	}
	text := res.Text
	if strings.TrimSpace(text) == "" {
		r.logf("info", "no selection (%s, %.0f ms)", res.Via, res.MS)
		return ""
	}
	r.logf("info", "selection: %s via %s in %.0f ms", r.logText(text), res.Via, res.MS)
	r.m.update(r, func(s *Snapshot) { s.Selection, s.SelectionVia = text, res.Via })
	return text
}

// listen records and recognises: the recognised text, or false when the
// session ended (cancelled, failed, too short, nothing heard).
func (r *run) listen() (string, bool) {
	cfg := r.m.cfg
	// The recording, at once: the user is speaking already.
	capCtx, capCancel := context.WithCancel(r.ctx)
	defer capCancel()
	capDone := make(chan capResult, 1)
	go func() {
		params := proto.CaptureParams{Devices: r.settings.Audio.Devices, SampleRate: asr.SampleRate, MaxSeconds: cfg.MaxRecording.Seconds()}
		res, err := cfg.Audio.Record(capCtx, r.id, params, r.onAudio)
		r.pcm.close()
		capDone <- capResult{res, err}
	}()
	svcDone := r.svcDone
	if r.resolved {
		svcDone = nil
	}
	open := func() chan openResult {
		st, ok := r.streamer()
		if !ok {
			r.setOpening(false)
			return nil
		}
		ch := make(chan openResult, 1)
		go func() {
			s, err := st.Open(r.ctx, asr.StreamRequest{SampleRate: asr.SampleRate, Lang: r.speechLang(), Hints: r.hints()})
			ch <- openResult{s, err}
		}()
		return ch
	}

	var (
		openDone   chan openResult
		stream     asr.Stream
		finals     chan streamEnd
		early      *streamEnd
		stopCh     = r.stop
		stopTimer  <-chan time.Time
		captured   capResult
		recordDone bool
	)
	// Until the speech model's stream is open, the session waits for it
	// (Snapshot.Waiting, after Config.WaitNote).
	r.opening.Store(true)
	waitNote := r.waitTimer(r.m.Current().StartedAt)
	if waitNote == nil {
		r.noteWait()
	}
	if r.resolved {
		if p := r.lacks(); p != nil {
			r.fail(p)
			return "", false
		}
		openDone = open()
	}
	for !recordDone {
		select {
		case <-r.ctx.Done():
			r.endCancelled(stream)
			return "", false
		case s := <-svcDone:
			svcDone = nil
			if s.err != nil {
				r.fail(ProblemOf(s.err))
				return "", false
			}
			r.useServices(s)
			r.warmLLM()
			if p := r.lacks(); p != nil {
				r.fail(p)
				return "", false
			}
			openDone = open()
		case o := <-openDone:
			openDone = nil
			if o.err != nil {
				r.fail(r.asrProblem(o.err))
				return "", false
			}
			r.setOpening(false)
			stream = o.st
			finals = r.feed(stream)
		case <-waitNote:
			waitNote = nil
			r.waitDue.Store(true)
			r.noteWait()
		case e := <-finals:
			finals = nil
			if e.err != nil {
				r.endFailed(r.asrProblem(e.err), stream)
				return "", false
			}
			early = &e // a final before the end: kept for the end
		case <-stopCh:
			stopCh = nil
			at := r.now()
			r.m.update(r, func(s *Snapshot) { s.State, s.StoppedAt = Recognizing, at })
			go func() {
				ctx, cancel := context.WithTimeout(r.ctx, cfg.StopGrace)
				defer cancel()
				if err := cfg.Audio.Stop(ctx, r.id); err != nil {
					r.logf("warning", "capture.stop: %v", err)
				}
			}()
			stopTimer = time.After(cfg.StopGrace)
		case <-stopTimer:
			r.logf("warning", "the recording did not end %v after Stop: discarding the capture job", cfg.StopGrace)
			capCancel()
		case c := <-capDone:
			recordDone, captured = true, c
		}
	}
	if r.cancelled.Load() {
		r.endCancelled(stream)
		return "", false
	}
	stopped := stopCh == nil
	if captured.err != nil && !(stopped && errors.Is(captured.err, context.Canceled)) && !isCancelled(captured.err, stopped) {
		r.endFailed(ProblemOf(captured.err), stream)
		return "", false
	}
	pcm := r.pcm.bytes()
	audioLen := asr.Duration(pcm, asr.SampleRate)
	r.m.update(r, func(s *Snapshot) {
		s.State = Recognizing
		if s.StoppedAt.IsZero() {
			s.StoppedAt = r.now()
		}
		s.Timings.Audio = audioLen
	})
	first := time.Duration(captured.res.FirstFrameMS * float64(time.Millisecond))
	r.logf("info", "recorded %v (%s, %d frames, %d restarts, first audio after %v from %s)", audioLen.Round(time.Millisecond), captured.res.Reason,
		captured.res.Frames, captured.res.Restarts, first.Round(time.Millisecond), r.m.Current().Device)
	if first > slowMic {
		r.logf("warning", "the microphone gave its first audio %v after the start: what was said before is not in the recording", first.Round(time.Millisecond))
	}
	if audioLen < cfg.MinRecording {
		if stream != nil {
			stream.Close()
		}
		r.m.end(r, Cancelled, func(s *Snapshot) { s.Problem = &Problem{Code: ProblemTooShort} }, nil)
		return "", false
	}
	asrStart := r.now()

	// The services, if they are still on their way.
	if !r.resolved {
		if !r.awaitServices() {
			return "", false
		}
		if p := r.lacks(); p != nil {
			r.fail(p)
			return "", false
		}
		openDone = open()
	}

	// Recognition.
	r.asrLimit = cfg.asrLimit(audioLen)
	actx, acancel := context.WithTimeout(r.ctx, r.asrLimit)
	defer acancel()
	var res asr.Result
	var err error
	salvaged := ""
	if _, ok := r.streamer(); ok {
		for openDone != nil {
			select {
			case o := <-openDone:
				openDone = nil
				if o.err != nil {
					r.recognitionFailed = true
					r.endFailed(r.asrProblem(o.err), nil)
					return "", false
				}
				r.setOpening(false)
				stream = o.st
				finals = r.feed(stream)
			case <-waitNote:
				waitNote = nil
				r.waitDue.Store(true)
				r.noteWait()
			case <-actx.Done():
				r.recognitionFailed = true
				r.endFailed(r.timeoutProblem(actx), nil)
				return "", false
			}
		}
		silent := false
		switch {
		case early != nil:
			res, silent = early.res, early.silent
		default:
			select {
			case e := <-finals:
				res, err, silent = e.res, e.err, e.silent
			case <-actx.Done():
				err = r.ctxErr(actx)
			}
		}
		if err != nil {
			salvaged = r.salvage(stream)
		}
		stream.Close()
		if silent {
			// The gate held the whole recording: no speech, nothing sent.
			r.logf("info", "no speech in %v: nothing sent to %s", audioLen.Round(time.Millisecond), r.svc.ASRInfo.ID)
			r.recognitionFailed = true
			r.m.update(r, func(s *Snapshot) { s.Timings.Sent = 0 })
			r.finish(Done, &Problem{Code: ProblemNoSpeech}, "", finishOpts{})
			return "", false
		}
	} else if ft, ok := r.svc.ASR.(asr.FileTranscriber); ok {
		upload, speech := r.prepare(pcm)
		if !speech {
			r.logf("info", "no speech in %v: not sent to %s", audioLen.Round(time.Millisecond), r.svc.ASRInfo.ID)
			r.recognitionFailed = true
			r.m.update(r, func(s *Snapshot) { s.Timings.Sent = 0 })
			r.finish(Done, &Problem{Code: ProblemNoSpeech}, "", finishOpts{})
			return "", false
		}
		sent := asr.Duration(upload, asr.SampleRate)
		r.m.update(r, func(s *Snapshot) { s.Timings.Sent = sent })
		res, err = ft.Transcribe(actx, asr.FileRequest{Audio: upload, Format: "pcm", SampleRate: asr.SampleRate, Lang: r.speechLang(), Hints: r.hints()})
		if err != nil && actx.Err() != nil {
			err = r.ctxErr(actx)
		}
	} else {
		err = &asr.Error{Kind: asr.KindConfig, Detail: "the speech model neither takes a whole recording nor streams"}
	}
	if r.cancelled.Load() {
		r.endCancelled(nil)
		return "", false
	}
	asrTime := r.now().Sub(asrStart)
	if err != nil {
		r.recognitionFailed = true
		// The error itself follows in endFailed's line (or the notice's).
		r.logf("warning", "speech model %s (%s) failed after %v (limit %v)", r.svc.ASRInfo.ID, r.svc.ASRInfo.Model, asrTime.Round(time.Millisecond), r.asrLimit.Round(time.Second))
		if salvaged == "" {
			r.endFailed(r.asrProblem(err), nil, func(s *Snapshot) { s.Timings.ASR = asrTime })
			return "", false
		}
		// The text the stream had, rather than nothing (behavior.md 5.5a).
		r.asrDegraded = &Problem{Code: ProblemASR, Err: err, Degraded: true}
		r.logf("warning", "going on with the text the live session had: %v", err)
		res = asr.Result{Text: salvaged}
	}
	raw := strings.TrimSpace(res.Text)
	r.logf("info", "recognised %s in %v", r.logText(raw), asrTime.Round(time.Millisecond))
	r.m.update(r, func(s *Snapshot) { s.Raw, s.Partial, s.Timings.ASR = raw, raw, asrTime })
	if raw == "" {
		r.recognitionFailed = true
		r.finish(Done, &Problem{Code: ProblemNoSpeech}, "", finishOpts{})
		return "", false
	}
	return raw, true
}

// salvage is the text a stream whose Final failed had (asr.Salvager:
// its pieces or its preview), else the preview the snapshot shows.
func (r *run) salvage(st asr.Stream) string {
	if sv, ok := st.(asr.Salvager); ok {
		if text := strings.TrimSpace(sv.Salvage()); text != "" {
			return text
		}
	}
	return strings.TrimSpace(r.m.Current().Partial)
}

// isCancelled reports whether a capture error is the end of a job that
// Stop discarded (cancelled after Config.StopGrace).
func isCancelled(err error, stopped bool) bool {
	var pe *proto.Error
	return stopped && errors.As(err, &pe) && pe.Code == proto.CodeCancelled
}

// warmLLM sends the language model the shared part of the prompt this
// session will send (the system prompt and the example turn) while the
// user speaks, when the model keeps a prompt cache (llm.Prefiller): the
// chat after the recording then evaluates little more than the
// transcript. Transcribe's clean-up and Translate of speech only:
// Rewrite's prompt depends on the selection, read meanwhile.
func (r *run) warmLLM() {
	p, ok := r.svc.LLM.(llm.Prefiller)
	if !ok {
		return
	}
	var in compose.Input
	var skill llm.Skill
	switch r.req.Kind {
	case store.KindTranscribe:
		if !r.settings.Features.Transcribe.Enhance {
			return
		}
		in, skill = r.transcribeInput(""), llm.SkillTranscribeEnhance
	case store.KindTranslate:
		in, skill = r.writeInput("", ""), llm.SkillTranslateText
	default:
		return
	}
	out, err := compose.Build(in)
	if err != nil {
		return
	}
	req := llm.Request{Skill: skill, Messages: llmMessages(out)}
	go func() {
		start := r.now()
		if err := p.Prefill(r.ctx, req); err != nil && r.ctx.Err() == nil {
			r.logf("info", "warming the language model up: %v", err)
			return
		}
		r.logf("debug", "warmed the language model up in %v", r.now().Sub(start).Round(time.Millisecond))
	}()
}

// transcribeInput is Transcribe's request to compose for raw.
func (r *run) transcribeInput(raw string) compose.Input {
	f := r.settings.Features.Transcribe
	return compose.Input{
		Task: compose.Transcribe, Cleanup: f.Cleanup, Structure: f.Structure, TaskPrompt: f.Prompt,
		MainLanguage: MainLanguage(r.settings, r.m.cfg.Locale()), OtherLanguages: r.settings.General.OtherLanguages, Text: raw,
		Glossary: r.glossary(),
		Scene:    r.composeScene(),
	}
}

// glossary is the dictionary's enabled terms for the prompt, when the
// settings send them.
func (r *run) glossary() []compose.Term {
	if !r.settings.Dictionary.Enabled || !r.settings.Dictionary.Glossary {
		return nil
	}
	var out []compose.Term
	for _, t := range r.terms {
		if t.Enabled {
			out = append(out, compose.Term{Term: t.Term, Aliases: t.Aliases})
		}
	}
	return out
}
