package meeting

// Filing a meeting that ended: at once, after the final pass, or once its
// question (history off, or started by mistake) is answered Keep.

import (
	"context"
	"fmt"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// fileLocked files a meeting that ended: done at once, or after the final
// pass over its recording. It is called with mu held and lets go of it.
func (m *Manager) fileLocked(ctx context.Context, mt store.Meeting, elapsed time.Duration) error {
	m.snap.Elapsed = elapsed
	if m.cfg.Settings().NoFinalPass || m.cfg.ASR == nil {
		m.state = StateDone
		m.snap.State = StateDone
		meetingID := m.id
		auto := !m.cfg.Settings().NoAutoSummary
		m.mu.Unlock()
		if err := m.cfg.Store.UpdateMeeting(ctx, mt); err != nil {
			return err
		}
		m.mu.Lock()
		m.resetAfterDone()
		s := m.snap
		m.mu.Unlock()
		m.notify(s)
		if auto {
			go func() {
				_ = m.GenerateMeetingSummary(context.Background(), meetingID)
			}()
		}
		return nil
	}
	m.state = StateFinishing
	m.snap.State = StateFinishing
	m.snap.Stage = StageTranscribing
	m.snap.Progress = 0
	mt.Status = store.MeetingFinishing
	m.mu.Unlock()
	if err := m.cfg.Store.UpdateMeeting(ctx, mt); err != nil {
		return err
	}
	m.notify(m.Current())
	go m.runFinalize(context.Background(), mt)
	return nil
}

// Save keeps an asking-state meeting.
func (m *Manager) Save(ctx context.Context, id string) error {
	var saved *Snapshot
	defer func() { // once mu is free
		if saved != nil {
			m.notify(*saved)
		}
	}()
	m.mu.Lock()
	if err := m.requireID(id); err != nil {
		m.mu.Unlock()
		return err
	}
	if m.state != StateAsking {
		m.mu.Unlock()
		return fmt.Errorf("meeting: not asking")
	}
	mt := store.Meeting{
		Seq: m.seq, ID: m.id, CreatedAt: m.snap.Started, EndedAt: m.cfg.Now(),
		Status: store.MeetingDone, Mode: toStoreMode(m.mode),
		DurationMS: m.timeline.Elapsed().Milliseconds(), Audio: m.id,
	}
	if m.snap.Mistake && m.cfg.Privacy().KeepHistory {
		// It asked only because little was heard: kept, it is filed as
		// any meeting that ended.
		m.snap.Mistake = false
		return m.fileLocked(ctx, mt, m.timeline.Elapsed())
	}
	if err := m.cfg.Store.UpdateMeeting(ctx, mt); err != nil {
		m.mu.Unlock()
		return err
	}
	meetingID := m.id
	auto := !m.cfg.Settings().NoAutoSummary
	m.state = StateDone
	m.snap.State = StateDone
	m.resetAfterDone()
	s := m.snap
	saved = &s // the meeting no longer asks: its overlay goes
	m.mu.Unlock()
	if auto {
		go func() {
			_ = m.GenerateMeetingSummary(context.Background(), meetingID)
		}()
	}
	return nil
}
