package omni

import (
	"context"
	"encoding/json"
	"time"

	"github.com/coder/websocket"
)

// previewDrain bounds the wait for the socket's writer to deliver the
// queued audio: at most maxQueued (30 s) on a loopback socket. A server
// that stops reading must not stall the stream's end.
const previewDrain = 5 * time.Second

// previewCompleteTimeout bounds the wait for the server to recognise
// what the socket took: the same job liveResultTimeout gives 20 s.
const previewCompleteTimeout = 20 * time.Second

// previewQuiet calls the server unresponsive: a server recognising the
// tail sends partials along the way, so this long without a word means
// nothing more is coming, and the wrap-up gives up instead of running
// the full previewCompleteTimeout behind a dead server.
const previewQuiet = previewDrain

// wrapUpPreview drains the preview's queue and awaits the server's
// recognition of it, so Salvage weighs the recording's tail too. It
// matters only when the stream's own text is the fallback, so it runs
// once, and only the fallback paths call it: a Final that failed, and a
// Salvage asked while the Final is still running (it took too long).
// A successful Final never waits for it. A LiveResult stream skips it:
// finishLive owns s.out until it ends the stream.
func (s *live) wrapUpPreview() {
	s.mu.Lock()
	ok := s.finalOnly
	s.mu.Unlock()
	if !ok {
		return
	}
	s.wrapUpOnce.Do(func() {
		if s.conn != nil && s.drainPreview() {
			s.completePreview()
		}
	})
}

// drainPreview hands the audio queued for the preview's socket to the
// server: Send accepted it, and closing the socket with the queue
// pending would lose it. A writer that does not drain in time is stuck
// on a server that stopped reading; the socket closes so the writer
// lets go, and false says the wrap-up is pointless: nothing more can
// reach the server, and its write would block on the stuck writer's
// lock.
func (s *live) drainPreview() (drained bool) {
	s.mu.Lock()
	close(s.out)
	s.mu.Unlock()
	select {
	case <-s.writeDone:
		return true
	case <-s.ctx.Done():
		return false
	case <-time.After(previewDrain):
		s.a.rt.logf("warning", "%s: the preview's writer did not drain in %v", s.a.spec.ID, previewDrain)
		s.conn.CloseNow()
		return false
	}
}

// completePreview wraps the live session up (transcription.done, as
// finishLive) and waits, bounded, for the server's answer: the
// recognition of everything the socket took, segments first. Salvage
// weighs the preview's text when the Final fails, and a fast-failing
// Final does not give the tail's recognition time to arrive on its own.
// The bound covers the write too: a server that stopped reading must
// not stall the stream's end, and a quiet one (previewQuiet) is given
// up on before the full timeout.
func (s *live) completePreview() {
	done, _ := json.Marshal(map[string]string{"type": "transcription.done"})
	ctx, cancel := context.WithTimeout(s.ctx, previewCompleteTimeout)
	defer cancel()
	if err := s.conn.Write(ctx, websocket.MessageText, done); err != nil {
		return // the socket is gone or the write timed out: the read side said why
	}
	for {
		s.mu.Lock()
		left := previewQuiet - time.Since(s.lastRecv)
		s.mu.Unlock()
		if left <= 0 {
			s.a.rt.logf("warning", "%s: the server said nothing for %v; the wrap-up gives up", s.a.spec.ID, previewQuiet)
			return
		}
		select {
		case <-s.completed:
			return
		case <-s.readDone:
			return
		case <-ctx.Done():
			if s.ctx.Err() == nil {
				s.a.rt.logf("warning", "%s: the server did not wrap the live session up in %v", s.a.spec.ID, previewCompleteTimeout)
			}
			return
		case <-time.After(left):
		}
	}
}
