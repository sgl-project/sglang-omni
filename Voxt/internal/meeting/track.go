package meeting

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

type track struct {
	src        Source
	job        string
	audio      session.Audio
	stopping   bool
	transcribe *trackTranscriber
	done       chan struct{} // closed once Record returned, its PCM is handled and its transcriber finished
}

func (m *Manager) runTrack(ctx context.Context, src Source, audio session.Audio) *StartError {
	if audio == nil {
		return &StartError{
			Source:   src,
			Err:      fmt.Errorf("meeting: no capture for %s", src),
			Guidance: GuidanceSystem,
		}
	}
	job := m.nextJob(src)
	m.mu.Lock()
	if m.timeline == nil { // the meeting ended while a reconnect waited for the lock
		m.mu.Unlock()
		return &StartError{Source: src, Err: fmt.Errorf("meeting: ended before %s reopened", src)}
	}
	m.timeline.Begin(src, job, m.timeline.Elapsed())
	done := make(chan struct{})
	m.tracks[src] = &track{src: src, job: job, audio: audio, done: done}
	m.mu.Unlock()

	tx, err := m.newTrackTranscriber(ctx, src, job)
	if err != nil { // the speech model, not the capture
		m.mu.Lock()
		delete(m.tracks, src)
		m.mu.Unlock()
		return &StartError{Source: src, Err: err, Guidance: GuidanceModel}
	}
	if tx != nil {
		m.mu.Lock()
		m.tracks[src].transcribe = tx
		m.mu.Unlock()
	}

	started := make(chan error, 1)
	// The capture callback runs on the helper's reader and must not
	// block (helper.Call). Apple Speech's Send is itself a helper call,
	// so the PCM goes through a queue to a goroutine of its own.
	pcm := newPCMQueue()
	pumped := make(chan struct{})
	go func() {
		defer close(pumped)
		for {
			data, ok := pcm.pop()
			if !ok {
				return
			}
			_ = m.archive.Write(src, data)
			m.mu.Lock()
			var tr *trackTranscriber
			if tx := m.tracks[src]; tx != nil {
				tr = tx.transcribe
			}
			runCtx := m.runCtx
			m.mu.Unlock()
			if tr != nil && runCtx != nil {
				_ = tr.onPCM(runCtx, data)
			}
		}
	}()
	go func() {
		_, err := audio.Record(ctx, job, proto.CaptureParams{}, func(ev session.AudioEvent) {
			if ev.Started != nil {
				select {
				case started <- nil:
				default:
				}
			}
			if ev.PCM != nil {
				if !m.timeline.acceptJob(src, ev.PCM.Job) {
					return
				}
				pcm.push(ev.PCM.Data)
			}
			if ev.Level != nil {
				if !m.timeline.acceptJob(src, ev.Level.Job) {
					return
				}
				m.mu.Lock()
				levels := map[Source][]float64{}
				for k, v := range m.snap.Levels {
					levels[k] = v
				}
				// A new map and slice: snapshots handed out keep theirs.
				h := append(append([]float64(nil), levels[src]...), session.Level(ev.Level.RMS))
				levels[src] = h[max(len(h)-session.Bars, 0):]
				m.snap.Levels = levels
				m.snap.Elapsed = m.timeline.Elapsed()
				m.mu.Unlock()
				m.noteLevel(src, ev.Level.RMS)
			}
		})
		pcm.close()
		<-pumped
		if tx != nil {
			_ = tx.finish(context.Background())
		}
		close(done)
		if err != nil {
			select {
			case started <- err:
			default:
			}
		}
		m.mu.Lock()
		tr := m.tracks[src]
		intentional := tr != nil && tr.stopping
		m.mu.Unlock()
		if !intentional {
			m.handleTrackLost(ctx, src)
		}
	}()
	select {
	case err := <-started:
		if err != nil {
			guidance := GuidanceMic
			if src == SourceSystem {
				guidance = GuidanceSystem
			}
			return &StartError{Source: src, Err: err, Guidance: guidance}
		}
		return nil
	case <-time.After(5 * time.Second):
		return &StartError{Source: src, Err: fmt.Errorf("meeting: capture start timeout")}
	case <-ctx.Done():
		return &StartError{Source: src, Err: ctx.Err()}
	}
}

// stop ends the capture first, so the transcriber sees its last PCM
// before it finishes.
func (t *track) stop() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = t.audio.Stop(ctx, t.job)
	select {
	case <-t.done:
	case <-ctx.Done(): // a helper that never answers Stop; its PCM pump may still be running
	}
}

// pcmQueue hands capture PCM from the helper's reader to the track's
// goroutine; it never blocks push, as a dropped chunk is lost audio.
type pcmQueue struct {
	mu     sync.Mutex
	chunks [][]byte
	closed bool
	ready  chan struct{}
}

func newPCMQueue() *pcmQueue { return &pcmQueue{ready: make(chan struct{}, 1)} }

func (q *pcmQueue) push(b []byte) {
	q.mu.Lock()
	q.chunks = append(q.chunks, b)
	q.mu.Unlock()
	q.wake()
}

func (q *pcmQueue) close() {
	q.mu.Lock()
	q.closed = true
	q.mu.Unlock()
	q.wake()
}

func (q *pcmQueue) wake() {
	select {
	case q.ready <- struct{}{}:
	default:
	}
}

// pop waits for the next chunk; false once closed and drained.
func (q *pcmQueue) pop() ([]byte, bool) {
	for {
		q.mu.Lock()
		if len(q.chunks) > 0 {
			b := q.chunks[0]
			q.chunks = q.chunks[1:]
			q.mu.Unlock()
			return b, true
		}
		closed := q.closed
		q.mu.Unlock()
		if closed {
			return nil, false
		}
		<-q.ready
	}
}
