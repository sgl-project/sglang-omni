package session

import (
	"context"
	"errors"
	"strings"
	"time"
	"unicode"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
)

// listen records and recognises: the recognised text, or false when the
// session ended (cancelled, failed, too short, nothing heard).
func (r *run) listen() (string, bool) {
	if r.req.SpeechWait != "" {
		r.fail(&Problem{Code: ProblemSpeechWait, Err: errors.New(r.req.SpeechWait)})
		return "", false
	}
	cfg := r.m.cfg
	// The recording, at once: the user is speaking already.
	capCtx, capCancel := context.WithCancel(r.ctx)
	defer capCancel()
	capDone := make(chan capResult, 1)
	go func() {
		// MaxSeconds: the user's length limit only (0: a podcast runs until Stop; StopGrace covers a lost stop).
		params := proto.CaptureParams{Devices: r.settings.Audio.Devices, SampleRate: asr.SampleRate, MaxSeconds: float64(r.settings.Audio.MaxSeconds)}
		res, err := cfg.Audio.Record(capCtx, r.id, params, r.onAudio)
		r.pcm.close()
		capDone <- capResult{res, err}
	}()
	svcDone := r.svcDone
	if r.resolved {
		svcDone = nil
	}
	var (
		word         string
		heard        chan heardText
		said         string
		endPause     <-chan time.Time
		stopEndPause func() bool
		quiet        bool // the user is silent
		openDone     chan openResult
		stream       asr.Stream
		openStarted  bool
		openReturned bool
		connected    bool
		prefilled    bool
		stopCh       = r.stop
		sceneDone    <-chan struct{}
	)
	clearEndPause := func() {
		endPause = nil
		if stopEndPause != nil {
			stopEndPause()
			stopEndPause = nil
		}
	}
	defer clearEndPause()
	sceneDone = r.sceneReadySignal()
	refreshHead := func() {
		if !r.head.enabled.Load() {
			return
		}
		quiet = r.pauseWatch != nil && r.pauseWatch.quiet.Load()
		if stopCh == nil {
			r.head.ask(said)
		} else if quiet {
			r.head.pause(said)
		}
	}
	maybeWarm := func() {
		if !r.cleanupMayContinue() {
			return
		}
		cleanupReady := !r.init.split || r.init.cleanupReady
		if r.head.enabled.Load() && (cleanupReady || !r.init.split) {
			refreshHead()
		}
		if r.svc.LLM == nil {
			return
		}
		remote := r.svc.LLMInfo.Remote
		if r.init.split && (!r.init.cleanupRequested || !r.init.llmReady) {
			return
		}
		if !connected && remote {
			connected = true
			r.warmConnect()
		}
		if r.sceneReadySignal() != nil || prefilled || r.init.split && (!r.init.cleanupReady || !remote && (!openReturned || openDone != nil)) {
			return
		}
		prefilled = true
		r.warmPrefill()
	}
	open := func() chan openResult {
		st, ok := r.streamer()
		if !ok {
			return nil
		}
		word = r.endWord()
		latePartials := r.prepareHeadInputs(word != "")
		heard = r.heard
		ch := make(chan openResult)
		request := asr.StreamRequest{
			SampleRate:   asr.SampleRate,
			Lang:         r.speechLang(),
			Hints:        append([]string(nil), r.hints()...),
			Prompt:       r.speechContext(),
			LiveResult:   r.settings.Audio.LiveResult,
			EarlyPieces:  r.earlyPieces(),
			LatePartials: latePartials,
		}
		go func() {
			s, err := st.Open(r.ctx, request)
			if r.ctx.Err() != nil {
				if s != nil {
					s.Close()
				}
				return
			}
			select {
			case ch <- openResult{s, err}:
			case <-r.ctx.Done():
				if s != nil {
					s.Close()
				}
			}
		}()
		return ch
	}
	startOpen := func() {
		if openStarted || !r.resolved || r.init.split && r.needsSpeechHints() && !r.init.termsReady {
			return
		}
		maybeWarm()
		openStarted = true
		openDone = open()
		if openDone == nil {
			openReturned = true
			r.setOpening(false)
			maybeWarm()
		}
	}

	var (
		finals     chan streamEnd
		early      *streamEnd
		stopTimer  <-chan time.Time
		captured   capResult
		recordDone bool
	)
	// Until the speech model's stream is open, the session waits for it
	// (Snapshot.Waiting, after Config.WaitNote).
	r.opening.Store(true)
	waitNote := r.waitTimer(r.m.snapshot(r).StartedAt)
	if waitNote == nil {
		r.noteWait()
	}
	if r.resolved {
		if p := r.lacks(); p != nil {
			r.fail(p)
			return "", false
		}
		startOpen()
	}
	for !recordDone {
		select {
		case <-r.ctx.Done():
			r.endCancelled(stream)
			return "", false
		case s := <-svcDone:
			svcDone = nil
			if r.ctx.Err() != nil || r.cancelled.Load() {
				r.endCancelled(stream)
				return "", false
			}
			if s.err != nil {
				r.fail(ProblemOf(s.err))
				return "", false
			}
			r.useServices(s)
			if p := r.lacks(); p != nil {
				r.fail(p)
				return "", false
			}
			startOpen()
		case result := <-r.init.asrDone:
			r.init.asrDone = nil
			if r.ctx.Err() != nil || r.cancelled.Load() {
				r.endCancelled(stream)
				return "", false
			}
			if result.Err != nil {
				r.fail(ProblemOf(result.Err))
				return "", false
			}
			if !r.useASRReadiness(result) {
				r.endCancelled(stream)
				return "", false
			}
			if p := r.lacks(); p != nil {
				r.fail(p)
				return "", false
			}
			startOpen()
		case terms := <-r.init.termsDone:
			r.init.termsDone = nil
			if r.ctx.Err() != nil || r.cancelled.Load() {
				r.endCancelled(stream)
				return "", false
			}
			r.useTerms(terms)
			r.maybeCleanupReady()
			startOpen()
			maybeWarm()
		case snippets := <-r.init.snippetsDone:
			r.init.snippetsDone = nil
			if r.ctx.Err() != nil || r.cancelled.Load() {
				r.endCancelled(stream)
				return "", false
			}
			r.useSnippets(snippets)
			r.maybeCleanupReady()
			maybeWarm()
		case result := <-r.init.llmDone:
			r.init.llmDone = nil
			if !r.useLLMReadiness(result) {
				r.endCancelled(stream)
				return "", false
			}
			maybeWarm()
		case <-sceneDone:
			sceneDone = nil
			r.markSceneReady()
			maybeWarm()
		case o := <-openDone:
			openDone = nil
			openReturned = true
			if r.ctx.Err() != nil || r.cancelled.Load() {
				if o.st != nil {
					o.st.Close()
				}
				r.endCancelled(stream)
				return "", false
			}
			if o.err != nil {
				if o.st != nil {
					o.st.Close()
				}
				// As after Stop: what was recorded is kept, to Retry.
				r.recognitionFailed = true
				r.endFailed(r.asrProblem(o.err), nil)
				return "", false
			}
			r.setOpening(false)
			stream = o.st
			r.stream = stream
			r.enableEarlyPieces()
			finals = r.feed(stream)
			maybeWarm()
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
		case update := <-heard:
			if update.candidate {
				continue
			}
			text := update.text
			if text != said {
				said = text
				clearEndPause()
				if _, ends := withoutEndWord(text, word); ends {
					pause := make(chan time.Time, 1)
					endPause = pause
					stopEndPause = cfg.AfterFunc(cfg.EndWordPause, func() {
						pause <- r.now()
					})
				}
				switch {
				case stopCh == nil:
					endPause = nil
					r.head.ask(said)
				case quiet:
					r.head.pause(said)
				}
			}
		case <-r.silent:
			quiet = r.pauseWatch != nil && r.pauseWatch.quiet.Load()
			if quiet && stopCh != nil {
				r.head.pause(said)
			}
		case <-endPause:
			clearEndPause()
			r.logf("info", "the end word was said last: stopping the recording")
			r.stopRecording()
		case <-stopCh:
			select {
			case update := <-heard:
				if !update.candidate {
					said = update.text
				}
			default:
			}
			r.head.ask(said)
			stopCh = nil
			clearEndPause()
			at := r.stoppedAt()
			if at.IsZero() {
				at = r.recordStopAt(r.now())
			}
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
			r.recordStopAt(r.now())
			close(r.recorded)
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
		captured.res.Frames, captured.res.Restarts, first.Round(time.Millisecond), r.m.snapshot(r).Device)
	if first > slowMic {
		r.logf("warning", "the microphone gave its first audio %v after the start: what was said before is not in the recording", first.Round(time.Millisecond))
	}
	if audioLen < cfg.MinRecording {
		if stream != nil && !r.stopStreamReader(stream) {
			r.recognitionFailed = true
			r.endFailed(&Problem{Code: ProblemInternal, Err: errors.New("the speech stream reader did not stop")}, nil)
			return "", false
		}
		if r.typedOnly() {
			return "", true
		}
		r.m.end(r, Cancelled, func(s *Snapshot) { s.Problem = &Problem{Code: ProblemTooShort} }, nil)
		return "", false
	}
	r.head.ask(said)
	asrStart := r.now()

	if r.init.split || !r.resolved {
		if !r.awaitASRReadiness(stream, startOpen, maybeWarm, &sceneDone) {
			return "", false
		}
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
			case <-r.ctx.Done():
				r.endCancelled(stream)
				return "", false
			case result := <-r.init.llmDone:
				r.init.llmDone = nil
				if !r.useLLMReadiness(result) {
					r.endCancelled(stream)
					return "", false
				}
				maybeWarm()
			case terms := <-r.init.termsDone:
				r.init.termsDone = nil
				if r.cancelled.Load() || r.ctx.Err() != nil {
					r.endCancelled(stream)
					return "", false
				}
				r.useTerms(terms)
				maybeWarm()
			case snippets := <-r.init.snippetsDone:
				r.init.snippetsDone = nil
				if r.cancelled.Load() || r.ctx.Err() != nil {
					r.endCancelled(stream)
					return "", false
				}
				r.useSnippets(snippets)
				maybeWarm()
			case <-sceneDone:
				sceneDone = nil
				if r.cancelled.Load() || r.ctx.Err() != nil {
					r.endCancelled(stream)
					return "", false
				}
				r.markSceneReady()
				maybeWarm()
			case o := <-openDone:
				openDone = nil
				openReturned = true
				if r.ctx.Err() != nil || r.cancelled.Load() {
					if o.st != nil {
						o.st.Close()
					}
					r.endCancelled(stream)
					return "", false
				}
				if o.err != nil {
					if o.st != nil {
						o.st.Close()
					}
					r.recognitionFailed = true
					r.endFailed(r.asrProblem(o.err), nil)
					return "", false
				}
				r.setOpening(false)
				stream = o.st
				r.stream = stream
				r.enableEarlyPieces()
				finals = r.feed(stream)
				maybeWarm()
			case <-waitNote:
				waitNote = nil
				r.waitDue.Store(true)
				r.noteWait()
			case <-actx.Done():
				if r.ctx.Err() != nil || r.cancelled.Load() {
					r.endCancelled(stream)
					return "", false
				}
				r.recognitionFailed = true
				r.endFailed(r.timeoutProblem(actx), nil)
				return "", false
			}
		}
		var silent bool
		if early != nil {
			res, silent = early.res, early.silent
		} else {
			end, waitErr := r.waitFinal(actx, finals, heard, &sceneDone, maybeWarm)
			res, err, silent = end.res, end.err, end.silent
			if waitErr != nil {
				err = waitErr
			}
		}
		if err != nil {
			salvaged = r.salvage(stream)
		}
		if !r.stopStreamReader(stream) {
			r.recognitionFailed = true
			r.endFailed(&Problem{Code: ProblemInternal, Err: errors.New("the speech stream reader did not stop")}, nil)
			return "", false
		}
		if silent && r.typedOnly() {
			return "", true
		}
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
		if !speech && r.typedOnly() {
			return "", true
		}
		if !speech {
			r.logf("info", "no speech in %v: not sent to %s", audioLen.Round(time.Millisecond), r.svc.ASRInfo.ID)
			r.recognitionFailed = true
			r.m.update(r, func(s *Snapshot) { s.Timings.Sent = 0 })
			r.finish(Done, &Problem{Code: ProblemNoSpeech}, "", finishOpts{})
			return "", false
		}
		sent := asr.Duration(upload, asr.SampleRate)
		r.m.update(r, func(s *Snapshot) { s.Timings.Sent = sent })
		request := asr.FileRequest{Audio: upload, Format: "pcm", SampleRate: asr.SampleRate, Lang: r.speechLang(), Hints: r.hints(), Prompt: r.speechContext(), MaxBytes: r.settings.Audio.UploadMB << 20}
		var ok bool
		res, err, ok = r.transcribeFile(actx, ft, request, &sceneDone, maybeWarm)
		if !ok {
			return "", false
		}
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
	raw, _ := withoutEndWord(strings.TrimSpace(r.inScript(res.Text, res.Lang)), word)
	r.logf("info", "recognised %s in %v", r.logText(raw), asrTime.Round(time.Millisecond))
	r.m.update(r, func(s *Snapshot) { s.Raw, s.Timings.ASR = raw, asrTime })
	if raw == "" && r.typedOnly() {
		return "", true
	}
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
	return strings.TrimSpace(r.m.snapshot(r).Partial)
}

// isCancelled reports whether a capture error is the end of a job that
// Stop discarded (cancelled after Config.StopGrace).
func isCancelled(err error, stopped bool) bool {
	var pe *proto.Error
	return stopped && errors.As(err, &pe) && pe.Code == proto.CodeCancelled
}

// inScript writes recognised Chinese text in the script of dictation
// (SpeechScript). Text in another language stays as it is, or its kanji
// would be rewritten: said is the language the speech model reported (""
// when it did not, as for a stream's partials and pieces), and kana marks
// Japanese either way.
func (r *run) inScript(text, said string) string {
	if c := lang.Canonical(said); c != "" && c != "zh" && c != "yue" {
		return text
	}
	if strings.ContainsFunc(text, func(c rune) bool { return unicode.In(c, unicode.Hiragana, unicode.Katakana) }) {
		return text
	}
	var reads []string
	if r.m.cfg.Languages != nil {
		reads = r.m.cfg.Languages()
	}
	return lang.RenderChineseScript(text, SpeechScript(r.settings, r.m.cfg.Locale(), reads))
}
