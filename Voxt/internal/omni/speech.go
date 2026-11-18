package omni

import (
	"context"

	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

// speechOnly is what of pcm a SpeechOnly model gets: the speech Silero
// finds, each pause over LongGapMS shortened (vad.Compact); false when
// there is none, and nothing is sent. MOSS answers audio without speech
// with words ("The.", subtitle credits, a refusal), and a long pause with
// timestamps or another language. Without Silero, or when it fails, pcm
// itself.
func (a *ASR) speechOnly(ctx context.Context, pcm []byte) ([]byte, bool) {
	if !a.spec.ASR.SpeechOnly || a.vad == nil {
		return pcm, true
	}
	segs, err := a.vad.Speech(ctx, pcm, sampleRate)
	if err != nil {
		if ctx.Err() == nil {
			a.rt.logf("warning", "%s: looking for the speech to send: %v", a.spec.ID, err)
		}
		return pcm, true
	}
	if len(segs) == 0 {
		return nil, false
	}
	return vad.Compact(pcm, sampleRate, vad.Analysis{Segments: segs, Speech: true}, vad.Params{}).PCM, true
}

// pauseCut keeps a SpeechOnly model's live pauses off its realtime
// socket: the server decodes fixed windows, and one without speech comes
// back as words. It passes the speech (vad.Detector's), PreRollMS before
// it and GapKeepMS/2 after it. Not safe for concurrent use.
type pauseCut struct {
	d       *vad.Detector
	keep    int    // bytes passed after the speech
	preRoll int    // bytes
	after   int    // bytes passed since the speech
	held    []byte // the latest pre-roll while it drops
}

func newPauseCut() *pauseCut {
	p := vad.Defaults()
	bytesOf := func(ms int) int { return ms * sampleRate / 1000 * 2 }
	keep := bytesOf(p.GapKeepMS / 2)
	return &pauseCut{d: vad.NewDetector(sampleRate, p), keep: keep, preRoll: bytesOf(p.PreRollMS), after: keep}
}

// push takes the next chunk and returns what to send: nil in a pause.
func (c *pauseCut) push(pcm []byte) []byte {
	c.d.Write(pcm)
	if c.d.Speaking() {
		out := append(c.held, pcm...)
		c.held, c.after = nil, 0
		return out
	}
	if c.after < c.keep {
		c.after += len(pcm)
		return pcm
	}
	c.held = append(c.held, pcm...)
	if cut := (len(c.held) - c.preRoll) &^ 1; cut > 0 {
		c.held = c.held[:copy(c.held, c.held[cut:])]
	}
	return nil
}
