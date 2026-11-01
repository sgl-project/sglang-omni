package omni

import (
	"context"
	"slices"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
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
func (s *live) recognisePieces(language string, b bias) {
	pc := s.pc
	defer close(pc.exited)
	t := s.a.spec.ASR
	looked := 0 // the audio's length at the last look for a pause
	for {
		select {
		case <-pc.wake:
		case <-s.ctx.Done():
			return
		}
		s.mu.Lock()
		finished, audio, start := s.finished, s.audio, pc.done
		s.mu.Unlock()
		total := len(audio) / 2
		from := start / 2
		if finished {
			return
		}
		if total < samplesOf(t.PiecesAfter) || total-from < samplesOf(t.PieceSeconds) || total-looked < sampleRate {
			continue
		}
		looked = total
		cut, ok := s.findCut(audio[start:])
		if !ok {
			continue
		}
		piece := audio[start : start+cut*2]
		begin := time.Now()
		req := finalRequest{pcm: padPiece(piece, t), language: language, context: b.context,
			maxTokens: t.tokenBudget(asr.Duration(piece, sampleRate)), layout: t.AudioLayout}
		r, err := transcribe(s.ctx, s.l, req, s.a.rt.cfg.StallTimeout)
		if err == nil {
			r.text = s.a.trimEcho(s.ctx, s.l, req, r.text, r.length, b.terms)
		}
		took := time.Since(begin)
		if err != nil {
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
func (s *live) findCut(pending []byte) (int, bool) {
	t := s.a.spec.ASR
	total := len(pending) / 2
	from, to := samplesOf(t.PieceMin), min(total, samplesOf(t.PieceMax))
	if to <= from {
		return 0, false
	}
	if s.a.vad != nil {
		ctx, cancel := context.WithTimeout(s.ctx, 15*time.Second)
		segs, err := s.a.vad.Speech(ctx, pending[from*2:to*2], sampleRate)
		cancel()
		if err == nil {
			if at, ok := longestPause(segs, to-from, samplesOf(t.PauseMin)); ok {
				return from + at, true
			}
		} else if s.ctx.Err() == nil {
			s.a.rt.logf("warning", "%s: looking for a pause to cut the recording at: %v", s.a.spec.ID, err)
		}
	}
	if total < samplesOf(t.PieceMax) {
		return 0, false
	}
	return quietest(pending, from, to, max(samplesOf(t.CutWindow), 1)), true
}

// longestPause is the middle of the longest pause of at least minPause
// samples around the speech segs in audio of n samples (before the
// first, between two, after the last; all of it when there is no
// speech).
func longestPause(segs []vad.Segment, n, minPause int) (int, bool) {
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
	consider(prev, n)
	return at, best > 0
}
