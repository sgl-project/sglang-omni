package omni

import (
	"context"
	"encoding/json"
	"slices"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

// pieces is the incremental Final of a long live session (behavior.md
// 3.7a): once the audio passes ASRTuning.PiecesAfter, stretches that
// end in a pause are recognised in the background, each by a Final
// request with the session's language hint (else the first piece's
// language) and context bias; at Finish only the rest is left. Its
// fields are guarded by the live session's mu.
type pieces struct {
	wake   chan struct{} // new audio (cap 1)
	exited chan struct{} // closed when the loop has stopped
	// note (chenye): Finish cancels planning while the session context keeps a claimed piece alive.
	cancelPlanning context.CancelFunc
	// note (chenye): Final joins only the current authoritative piece's transcription.
	activePieceDone chan struct{}
	tuning          ASRTuning // policy for this stream; guarded by live.mu
	policy          uint64    // bumped when a safe late policy change takes effect
	claimed         bool      // a piece has started; its boundaries can no longer change
	// done is the bytes of audio recognised, texts their text in order,
	// language the language they carry ("" until known when no hint).
	done     int
	texts    []string
	language string
	busy     time.Duration // the time the pieces' requests took
}

// samplesOf is d of audio in samples.
func samplesOf(d time.Duration) int { return int(d.Seconds() * sampleRate) }

// recognisePieces is the loop of s's pieces: on new audio, a piece
// when one is due, until Finish (or a piece failed: the rest waits for
// Finish).
func (s *live) recognisePieces(planningContext context.Context, language string, b bias) {
	pc := s.pc
	defer close(pc.exited)
	looked := 0 // the audio's length at the last look for a pause
	var lookedPolicy uint64
	for {
		select {
		case <-pc.wake:
		case <-planningContext.Done():
			return
		}
		s.mu.Lock()
		finished, ended, audio, start, t, policy := s.finished, s.ended, s.audio, pc.done, pc.tuning, pc.policy
		s.mu.Unlock()
		if policy != lookedPolicy {
			looked, lookedPolicy = 0, policy
		}
		total := len(audio) / 2
		from := start / 2
		if finished || ended || planningContext.Err() != nil {
			return
		}
		if total < samplesOf(t.PiecesAfter) || total-from < samplesOf(t.PieceSeconds) || total-looked < sampleRate {
			continue
		}
		looked = total
		cut, ok := s.findCut(planningContext, audio[start:], t)
		if !ok {
			continue
		}
		s.mu.Lock()
		var activePieceDone chan struct{}
		if s.finished || s.ended || planningContext.Err() != nil {
			s.mu.Unlock()
			return
		} else if pc.policy != policy || pc.done != start {
			s.mu.Unlock()
			continue
		} else {
			activePieceDone = make(chan struct{})
			pc.claimed = true
			pc.activePieceDone = activePieceDone
		}
		s.mu.Unlock()
		piece := audio[start : start+cut*2]
		s.startFinalAt((start + cut*2) / 2)
		begin := time.Now()
		var r finalResult
		var err error
		if sent, ok := s.a.speechOnly(s.ctx, piece); ok {
			req := finalRequest{pcm: padPiece(sent, &t), language: language, context: b.context,
				maxTokens: t.tokenBudget(asr.Duration(sent, sampleRate)), layout: t.AudioLayout}
			r, err = transcribe(s.ctx, s.l, req, s.a.rt.cfg.StallTimeout)
			if err == nil {
				r.text = s.a.trimEcho(s.ctx, s.l, req, s.a.plainText(r.text), r.length, b.terms)
			}
		}
		took := time.Since(begin)
		if err != nil {
			s.mu.Lock()
			close(activePieceDone)
			pc.activePieceDone = nil
			s.mu.Unlock()
			if s.ctx.Err() == nil {
				s.a.rt.logf("warning", "%s: a piece of the recording failed after %v, the rest waits for Stop: %v", s.a.spec.ID, took.Round(time.Millisecond), err)
			}
			return
		}
		if language == "" {
			language = r.language
		}
		s.mu.Lock()
		pc.done += len(piece)
		pc.texts = append(pc.texts, r.text)
		pc.language = language
		pc.busy += took
		n := len(pc.texts)
		close(activePieceDone)
		pc.activePieceDone = nil
		// The text so far is final: the session can start its clean-up of
		// it instead of waiting for Stop (the Final repeats it as its prefix).
		s.emit(asr.Event{Kind: asr.EventPiece, Text: joinParts(pc.texts)})
		s.mu.Unlock()
		looked = 0
		select { // the rest may hold another piece already
		case pc.wake <- struct{}{}:
		default:
		}
		s.a.rt.logf("info", "%s: piece %d of the recording, %v of audio from %v, in %v", s.a.spec.ID, n,
			asr.Duration(piece, sampleRate).Round(10*time.Millisecond), asr.Duration(audio[:start], sampleRate).Round(10*time.Millisecond), took.Round(time.Millisecond))
	}
}

// earlyPieces is Qwen3-ASR's policy when the session works on the pieces
// while it records (asr.StreamRequest.EarlyPieces): the first at 20 s.
var earlyPieces = Pieces{After: 20 * time.Second, Min: 10 * time.Second, Target: 15 * time.Second, Max: 25 * time.Second}

// startFinalAt tells the server the sample the Final starts from once the
// piece before it is recognised: it prefills the audio from there on while
// the recording goes on.
func (s *live) startFinalAt(sample int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.a.spec.Kind != kindQwen3ASR || s.session == "" || !s.preview || s.finished || s.ended {
		return
	}
	msg, _ := json.Marshal(map[string]any{"type": "input_audio_buffer.final_start", "start_sample": sample})
	select {
	case s.out <- msg:
	default:
	}
}

// padPiece pads a piece shorter than the policy's shortest chunk with
// silence.
func padPiece(pcm []byte, t *ASRTuning) []byte {
	if short := samplesOf(time.Duration(t.MinChunkSeconds*float64(time.Second))) - len(pcm)/2; short > 0 {
		return append(slices.Clip(pcm), make([]byte, short*2)...)
	}
	return pcm
}

// findCut is where the next piece of pending (the audio not yet
// recognised, PCM16) ends, in samples: the middle of the longest pause
// of at least PauseMin between PieceMin and PieceMax into it (Silero's;
// without it, or when it fails, none is found), else, once pending
// reaches PieceMax, the quietest CutWindow there. False: wait for more
// audio.
func (s *live) findCut(planningContext context.Context, pending []byte, t ASRTuning) (int, bool) {
	total := len(pending) / 2
	from, to := samplesOf(t.PieceMin), min(total, samplesOf(t.PieceMax))
	if to <= from {
		return 0, false
	}
	if s.a.vad != nil {
		vadContext, cancel := context.WithTimeout(planningContext, 15*time.Second)
		segs, err := s.a.vad.Speech(vadContext, pending[from*2:to*2], sampleRate)
		cancel()
		if err == nil {
			if at, ok := longestPause(segs, to-from, samplesOf(t.PauseMin), to == total); ok {
				return from + at, true
			}
		} else if planningContext.Err() != nil {
			return 0, false
		} else {
			s.a.rt.logf("warning", "%s: looking for a pause to cut the recording at: %v", s.a.spec.ID, err)
		}
	}
	if total < samplesOf(t.PieceMax) {
		return 0, false
	}
	return quietest(pending, from, to, max(samplesOf(t.CutWindow), 1)), true
}

// EnableEarlyPieces changes only this stream's policy. It can shorten the
// schedule only before a piece has been claimed; later readiness keeps the
// established segmentation, including when the user has stopped sending audio.
func (s *live) EnableEarlyPieces() bool {
	if s.a.spec.Kind != kindQwen3ASR || s.req.LiveResult && s.conn != nil || s.pc == nil {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.finished || s.ended {
		return false
	}
	if s.pc.tuning.PiecesAfter <= earlyPieces.After {
		return true
	}
	if s.pc.claimed || s.pc.done != 0 || len(s.pc.texts) != 0 {
		return false
	}
	t := s.pc.tuning
	t.PiecesAfter, t.PieceMin, t.PieceSeconds, t.PieceMax = earlyPieces.After, earlyPieces.Min, earlyPieces.Target, earlyPieces.Max
	s.pc.tuning = t
	s.pc.policy++
	select {
	case s.pc.wake <- struct{}{}:
	default:
	}
	return true
}

// longestPause is the middle of the longest pause of at least minPause
// samples around the speech segs in audio of n samples (before the
// first, between two, after the last; all of it when there is no
// speech). atEnd: the audio ends at the newest sample, so the pause after
// the last speech may be the end of the recording, and a cut there would
// leave a piece of silence that the model fills with words ("Okay.").
func longestPause(segs []vad.Segment, n, minPause int, atEnd bool) (int, bool) {
	best, at := 0, 0
	consider := func(from, to int) {
		if to-from >= minPause && to-from > best {
			best, at = to-from, from+(to-from)/2
		}
	}
	prev := 0
	for _, sg := range segs {
		consider(prev, min(sg.Start, n))
		prev = max(prev, sg.End)
	}
	if !atEnd {
		consider(prev, n)
	}
	return at, best > 0
}
