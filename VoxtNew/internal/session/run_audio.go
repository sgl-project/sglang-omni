package session

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/dict"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/dsp"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
)

// streamer is the session's speech model as a streamer, when it streams.
func (r *run) streamer() (asr.Streamer, bool) {
	st, ok := r.svc.ASR.(asr.Streamer)
	return st, ok && r.svc.ASR.Caps().Stream
}

// hints are the dictionary's terms for the speech model (dict.Hints:
// the enabled ones, the most recently changed first).
func (r *run) hints() []string {
	if !r.settings.Dictionary.Enabled || !r.settings.Dictionary.ASRHints {
		return nil
	}
	return dict.Hints(r.terms)
}

// speechLang is the language hint of the speech model: the user's
// main language (General.MainLanguage, else the UI's), also when they
// speak other languages (Voxt's policy, measured in docs/eval.md: the
// hint "zh" cost nothing on English speech, while without a hint Qwen3-ASR
// took a mixed dictation that started in English for English and
// translated its Chinese). The model decides whether to use it
// (internal/omni: Qwen3-ASR takes it when it knows the language).
func (r *run) speechLang() string {
	return MainLanguage(r.settings, r.m.cfg.Locale())
}

// onAudio handles the recording's events (on the helper's reader
// goroutine: it must not block).
func (r *run) onAudio(ev AudioEvent) {
	switch {
	case ev.PCM != nil:
		r.pcm.append(ev.PCM.Data)
	case ev.Started != nil:
		st, at := ev.Started, r.now()
		r.m.update(r, func(s *Snapshot) {
			s.HeardAt, s.Device, s.FellBack = at, st.DeviceName, st.FellBack
			s.Waiting = r.waitOf(s)
		})
		if st.FellBack {
			r.logf("warning", "recording from %s: %s", st.DeviceName, st.FallbackReason)
		}
	case ev.Level != nil:
		v := Level(ev.Level.RMS)
		r.m.update(r, func(s *Snapshot) {
			s.Levels = append(s.Levels, v)
			if len(s.Levels) > Bars {
				s.Levels = s.Levels[len(s.Levels)-Bars:]
			}
		})
	case ev.Restarted != nil:
		r.logf("warning", "the input restarted (%s, attempt %d)", ev.Restarted.Reason, ev.Restarted.Attempt)
	}
}

// feed sends the recording to st as it grows, then Finish; its partials
// go to the snapshot. The returned channel gets the stream's end: the
// final result or the error.
func (r *run) feed(st asr.Stream) chan streamEnd {
	// Two may come (the sender's silent end, the reader's): the first
	// counts, the other must not block.
	end := make(chan streamEnd, 2)
	sendErr := make(chan error, 1)
	hp, lev := r.streamDSP()
	go func() {
		// The gate opens the model's detector, which may take a moment:
		// here, not on the session's loop (the recording waits meanwhile).
		gate, closeGate := r.gate()
		defer closeGate()
		off, sent := 0, 0
		for {
			chunk, closed, err := r.pcm.next(r.ctx, off)
			if err != nil {
				return
			}
			if len(chunk) > 0 {
				off += len(chunk)
				// The recording is kept as it came; the stages make copies.
				raw := chunk
				if hp != nil {
					chunk = hp.Process(chunk)
				}
				if gate != nil {
					if hp != nil {
						gate.Hear(raw) // the filter never holds back more
					}
					chunk = gate.Push(chunk)
				}
				if len(chunk) == 0 {
					continue
				}
				if lev != nil {
					chunk = lev.Process(chunk)
				}
				if err := st.Send(chunk); err != nil {
					sendErr <- err
					return
				}
				sent += len(chunk)
				continue
			}
			if closed {
				d := pcmDuration(sent)
				r.m.update(r, func(s *Snapshot) { s.Timings.Sent = d })
				if sent > 0 && (hp != nil || lev != nil) {
					rep := dsp.Report{SpeechDB: -120, FloorDB: -120}
					if lev != nil {
						rep = lev.Report()
					}
					if hp != nil {
						rep.HighPass, rep.Clipped = true, hp.Clipped
					}
					r.noteDSP(rep)
				}
				if gate != nil {
					r.noteGate(gate)
					if !gate.Open() {
						end <- streamEnd{silent: true}
						return
					}
					if gate.Dropped() > 0 {
						r.logf("info", "voice activity: held back %v of silence before the speech", pcmDuration(gate.Dropped()).Round(10*time.Millisecond))
					}
				}
				if err := st.Finish(); err != nil {
					sendErr <- err
				}
				return
			}
		}
	}()
	go func() {
		events := st.Events()
		for {
			select {
			case ev, ok := <-events:
				if !ok {
					end <- streamEnd{err: &asr.Error{Kind: asr.KindProtocol, Detail: "the stream ended without a final result"}}
					return
				}
				switch ev.Kind {
				case asr.EventPartial:
					text := ev.Text
					r.m.update(r, func(s *Snapshot) { s.Partial = text })
				case asr.EventFinal:
					res := ev.Result
					res.Text = ev.Text
					end <- streamEnd{res: res}
					return
				case asr.EventError:
					end <- streamEnd{err: ev.Err}
					return
				}
			case err := <-sendErr:
				// The reader usually has the real reason; give it a moment.
				select {
				case ev, ok := <-events:
					if ok && ev.Kind == asr.EventError {
						err = ev.Err
					}
				case <-time.After(500 * time.Millisecond):
				}
				end <- streamEnd{err: asr.NetError(r.ctx, "", err)}
				return
			}
		}
	}()
	return end
}

// vadParams are the local voice activity detection's parameters, and
// false when the settings turn it off (settings.VAD).
func (r *run) vadParams() (vad.Params, bool) {
	v := r.settings.Audio.VAD
	if v.Off {
		return vad.Params{}, false
	}
	return vad.Params{
		OnsetDB: v.OnsetDB, OffsetDB: v.OffsetDB, MinLevelDB: v.MinLevelDB, HangoverMS: v.HangoverMS,
		LeadMS: v.LeadMS, TailMS: v.TailMS, LongGapMS: v.LongGapMS, GapKeepMS: v.GapKeepMS,
		PreRollMS: v.PreRollMS, MaxHoldMS: v.MaxHoldMS,
	}.Resolved(), true
}

// dspParams are the sound processing's parameters (settings.Clean), and
// false when the settings turn it off.
func (r *run) dspParams() (dsp.Params, bool) {
	c := r.settings.Audio.Clean
	if c.Off || c.HighPassOff && c.LevelOff {
		return dsp.Params{}, false
	}
	return dsp.Params{
		HighPassOff: c.HighPassOff, LevelOff: c.LevelOff, HighPassHz: c.HighPassHz,
		TargetDB: c.TargetDB, MaxGainDB: c.MaxGainDB, LoudDB: c.LoudDB, MaxCutDB: c.MaxCutDB,
		NoiseCeilDB: c.NoiseCeilDB, CeilingDB: c.CeilingDB,
	}.Resolved(), true
}

// prepare is what a file transcriber gets of the recording pcm: processed
// (internal/dsp: the high-pass filter, one gain from the speech, the
// limiter) and without its leading and trailing silence, long pauses
// shortened (internal/vad), each when the settings have it on; false when
// the detection finds no speech at all (nothing is sent). The voice
// activity model finds the speech when there is one (vad_model.go); else
// the energy detection runs on the filtered recording and on the
// original, and keeps the speech either finds. pcm itself is not changed:
// the history keeps it.
func (r *run) prepare(pcm []byte) ([]byte, bool) {
	p, vadOn := r.vadParams()
	dp, dspOn := r.dspParams()
	upload := pcm
	var an vad.Analysis
	switch {
	case dspOn:
		f := dsp.ProcessFile(pcm, asr.SampleRate, dp, p)
		upload, an = f.PCM, f.Analysis
		r.noteDSP(f.Report)
	case vadOn:
		an = vad.Analyze(pcm, asr.SampleRate, p)
	}
	if !vadOn {
		return upload, true
	}
	if m, ok := r.modelAnalysis(pcm, an); ok {
		an = m
	}
	if !an.Speech {
		r.logf("info", "voice activity: no speech (noise floor %.0f dBFS, loudest %.0f dBFS)", an.FloorDB, an.PeakDB)
		return nil, false
	}
	c := vad.Compact(upload, asr.SampleRate, an, p)
	if c.Changed {
		r.logf("info", "voice activity: %d segments, sending %v of %v", len(an.Segments), c.Kept.Round(10*time.Millisecond), c.Original.Round(10*time.Millisecond))
	}
	return c.PCM, true
}

// noteDSP logs what the sound processing did and puts it in the snapshot
// (and so in the history entry).
func (r *run) noteDSP(rep dsp.Report) {
	var stages []string
	if rep.HighPass {
		stages = append(stages, "highpass")
	}
	if rep.Level {
		stages = append(stages, "level")
	}
	desc := strings.Join(stages, "+")
	speech := "no speech"
	if rep.SpeechDB > -120 {
		speech = fmt.Sprintf("speech %.1f dBFS", rep.SpeechDB)
	}
	extra := ""
	if rep.MinGainDB != rep.MaxGainDB {
		extra = fmt.Sprintf(" (%+.1f…%+.1f)", rep.MinGainDB, rep.MaxGainDB)
	}
	r.logf("info", "sound: %s, %s over a floor of %.0f dBFS, gain %+.1f dB%s, %d samples limited, %d clipped by the filter",
		desc, speech, rep.FloorDB, rep.GainDB, extra, rep.Limited, rep.Clipped)
	gain := math.Round(rep.GainDB*10) / 10
	r.dsp, r.gainDB = desc, gain
	r.m.update(r, func(s *Snapshot) { s.DSP, s.GainDB = desc, gain })
}

// streamDSP are the stages a streaming speech model's audio goes through:
// the high-pass filter before the gate, the level stage after it; nil
// when the settings turn them off.
func (r *run) streamDSP() (*dsp.HighPass, *dsp.Leveler) {
	dp, ok := r.dspParams()
	if !ok {
		return nil, nil
	}
	var hp *dsp.HighPass
	var lev *dsp.Leveler
	if !dp.HighPassOff {
		hp = dsp.NewHighPass(asr.SampleRate, dp.HighPassHz)
	}
	if !dp.LevelOff {
		vp, _ := r.vadParams()
		lev = dsp.NewLeveler(asr.SampleRate, dp, vp)
	}
	return hp, lev
}

// pcmDuration is how long n bytes of the recording's PCM play.
func pcmDuration(n int) time.Duration {
	return time.Duration(n/2) * time.Second / asr.SampleRate
}
