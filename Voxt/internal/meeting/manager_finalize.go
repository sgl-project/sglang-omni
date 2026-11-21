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
		m.notify(m.Current())
	}
	cfg := m.cfg
	if fromStoreMode(mt.Mode) == ModeImport {
		cfg = importPassConfig(cfg)
	}
	err = FinalizePass(ctx, cfg, mt, m.finalizeProgress(id))
	if err != nil {
		return err
	}
	if fromStoreMode(mt.Mode) == ModeImport {
		return m.finishImport(ctx, mt, !m.cfg.Settings().NoAutoSummary)
	}
	// The correction and the summary run as after a live finalize; a
	// failure surfaces to the caller (the meeting menu corrects again).
	var cerr error
	if !m.cfg.Settings().NoAutoSummary {
		cerr = m.correctAndSummarize(ctx, id)
	}
	if m.id == id {
		m.mu.Lock()
		m.state = StateDone
		m.snap.State = StateDone
		m.resetAfterDone()
		m.mu.Unlock()
		m.notify(m.Current())
	}
	return cerr
}

func (m *Manager) runFinalize(ctx context.Context, mt store.Meeting) {
	// The final pass holds what it uses; the recording hold would keep MTD's predecessor alive.
	m.mu.Lock()
	if m.releaseModels != nil {
		m.releaseModels()
		m.releaseModels = nil
	}
	m.mu.Unlock()
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
		go m.correctAndSummarize(context.Background(), meetingID)
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

// correctAndSummarize corrects the transcript, then summarizes it; a
// correction that fails leaves the summary to the recognised text, and
// the caller hears of it.
func (m *Manager) correctAndSummarize(ctx context.Context, id string) error {
	var first error
	if err := m.CorrectTranscript(ctx, id); err != nil {
		m.warn("correct: " + err.Error())
		first = err
	}
	if err := m.GenerateMeetingSummary(ctx, id); err != nil {
		m.warn("summary: " + err.Error())
		if first == nil {
			first = err
		}
	}
	if m.cfg.Updated != nil {
		m.cfg.Updated(id)
	}
	return first
}

func (m *Manager) warn(msg string) {
	if m.cfg.Warn != nil {
		m.cfg.Warn(msg)
	}
}
