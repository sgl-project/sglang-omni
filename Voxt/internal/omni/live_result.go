package omni

import (
	"encoding/json"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// liveResultTimeout bounds finishLive's wait for the server's whole
// text: the audio still queued for the socket (at most maxQueued, 30 s)
// and the last segment to recognise, then the Final runs instead.
const liveResultTimeout = 20 * time.Second

// finishLive ends a LiveResult stream: transcription.done follows the
// audio still queued for the socket, and the server answers, once it has
// recognised the last segment, with the whole text, which is the result.
// When the socket fails first, or the answer takes liveResultTimeout,
// the Final recognises pcm, the whole recording, as without LiveResult.
func (s *live) finishLive(pcm []byte) {
	start := time.Now()
	done, _ := json.Marshal(map[string]string{"type": "transcription.done"})
	fallback := func(why string) {
		s.a.rt.logf("warning", "%s: live result: %s; recognising the whole recording", s.a.spec.ID, why)
		s.mu.Lock()
		s.preview = false
		s.mu.Unlock()
		s.finalPass(pcm)
	}
	select {
	case s.out <- done:
	default:
		fallback("the socket's queue is full")
		return
	}
	timer := time.NewTimer(liveResultTimeout)
	defer timer.Stop()
	var text string
	select {
	case text = <-s.completed:
	case <-s.readDone:
		select {
		case text = <-s.completed:
		default:
			fallback("the socket closed")
			return
		}
	case <-timer.C:
		fallback("no answer in " + liveResultTimeout.String())
		return
	case <-s.ctx.Done():
		s.end(asr.Event{Kind: asr.EventError, Err: asr.NetError(s.ctx, Provider, s.ctx.Err())})
		return
	}
	took := time.Since(start)
	s.a.rt.logf("info", "%s: live result of %v of audio in %v", s.a.spec.ID, asr.Duration(pcm, sampleRate).Round(100*time.Millisecond), took.Round(time.Millisecond))
	res := asr.Result{Text: text, Lang: s.language, Duration: asr.Duration(pcm, sampleRate), Model: s.a.spec.ID, Latency: took}
	s.end(asr.Event{Kind: asr.EventFinal, Text: text, Result: res})
}
