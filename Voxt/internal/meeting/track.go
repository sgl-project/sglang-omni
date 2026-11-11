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
	done       chan struct{} // closed once Record returned and its PCM is handled
}

func (m *Manager) runTrack(ctx context.Context, src Source, audio session.Audio) *StartError {
	if audio == nil {
		return &StartError{
			Source:   src,
			Err:      fmt.Errorf("meeting: no capture for %s", src),
			Guidance: "meeting.system.unavailable",
		}
	}
	job := m.nextJob(src)
	m.mu.Lock()
	m.timeline.Begin(src, job, m.timeline.Elapsed())
	done := make(chan struct{})
	m.tracks[src] = &track{src: src, job: job, audio: audio, done: done}
	m.mu.Unlock()

	tx, err := m.newTrackTranscriber(ctx, src, job)
	if err != nil {
		guidance := "meeting.mic.unavailable"
		if src == SourceSystem {
			guidance = "meeting.system.unavailable"
		}
		m.mu.Lock()
		delete(m.tracks, src)
		m.mu.Unlock()
		return &StartError{Source: src, Err: err, Guidance: guidance}
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
				level := map[Source]float64{src: ev.Level.RMS}
				for k, v := range m.snap.Level {
					if k != src {
						level[k] = v
					}
				}
				m.snap.Level = level // a new map: snapshots handed out keep theirs
				m.snap.Elapsed = m.timeline.Elapsed()
				m.mu.Unlock()
				m.noteLevel(src, ev.Level.RMS)
			}
		})
		pcm.close()
		<-pumped
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
			guidance := "meeting.mic.unavailable"
			if src == SourceSystem {
				guidance = "meeting.system.unavailable"
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
	_ = t.audio.Stop(context.Background(), t.job)
	select {
	case <-t.done:
	case <-time.After(5 * time.Second): // a helper that never answers Stop
	}
	if t.transcribe != nil {
		_ = t.transcribe.finish(context.Background())
		t.transcribe = nil
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
