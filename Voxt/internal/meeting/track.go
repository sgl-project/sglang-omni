package meeting

import (
	"context"
	"fmt"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

type track struct {
	src      Source
	job      string
	audio    session.Audio
	stopping bool
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
	m.tracks[src] = &track{src: src, job: job, audio: audio}
	m.mu.Unlock()

	started := make(chan error, 1)
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
				_ = m.archive.Write(src, ev.PCM.Data)
			}
			if ev.Level != nil {
				if !m.timeline.acceptJob(src, ev.Level.Job) {
					return
				}
				m.mu.Lock()
				if m.snap.Level == nil {
					m.snap.Level = map[Source]float64{}
				}
				m.snap.Level[src] = ev.Level.RMS
				m.snap.Elapsed = m.timeline.Elapsed()
				s := m.snap
				m.mu.Unlock()
				m.notify(s)
			}
		})
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

func (t *track) stop() {
	t.stopping = true
	_ = t.audio.Stop(context.Background(), t.job)
}
