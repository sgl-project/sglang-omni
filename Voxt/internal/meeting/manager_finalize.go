package meeting

import (
	"context"
	"fmt"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Reprocess reruns the final transcription pass for a stored meeting.
func (m *Manager) Reprocess(ctx context.Context, id string) error {
	mt, _, err := m.cfg.Store.Meeting(ctx, id)
	if err != nil {
		return err
	}
	switch mt.Status {
	case store.MeetingInterrupted, store.MeetingFinishing, store.MeetingDone:
	default:
		return fmt.Errorf("meeting: cannot reprocess %s", mt.Status)
	}
	mt.Status = store.MeetingFinishing
	if err := m.cfg.Store.UpdateMeeting(ctx, mt); err != nil {
		return err
	}
	if m.id == id {
		m.mu.Lock()
		m.snap.State = StateFinishing
		m.snap.Stage = StageTranscribing
		m.snap.Progress = 0
		m.mu.Unlock()
		m.notify(m.snap)
	}
	err = FinalizePass(ctx, m.cfg, mt, m.finalizeProgress(id))
	if err != nil {
		return err
	}
	if m.id == id {
		m.mu.Lock()
		m.state = StateDone
		m.snap.State = StateDone
		m.resetAfterDone()
		m.mu.Unlock()
		m.notify(m.snap)
	}
	return nil
}

func (m *Manager) runFinalize(ctx context.Context, mt store.Meeting) {
	err := FinalizePass(ctx, m.cfg, mt, m.finalizeProgress(mt.ID))
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.id != mt.ID {
		return
	}
	if err != nil {
		m.state = StateInterrupted
		m.snap.State = StateInterrupted
		m.notify(m.snap)
		return
	}
	meetingID := mt.ID
	auto := !m.cfg.Settings().NoAutoSummary
	m.state = StateDone
	m.snap.State = StateDone
	m.resetAfterDone()
	m.notify(m.snap)
	if auto {
		go func() {
			_ = m.GenerateMeetingSummary(context.Background(), meetingID)
		}()
	}
}

func (m *Manager) finalizeProgress(id string) func(float64) {
	return func(p float64) {
		m.mu.Lock()
		if m.id != id {
			m.mu.Unlock()
			return
		}
		m.snap.Progress = p
		s := m.snap
		m.mu.Unlock()
		m.notify(s)
	}
}
