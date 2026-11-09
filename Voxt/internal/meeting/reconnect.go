package meeting

import (
	"context"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

var reconnectDelays = []time.Duration{time.Second, 2 * time.Second, 4 * time.Second}

// handleTrackLost tries to reopen src after an unexpected capture end.
func (m *Manager) handleTrackLost(ctx context.Context, src Source) {
	if ctx.Err() != nil { // the run was cancelled: an end, not a loss
		return
	}
	m.mu.Lock()
	if !m.state.Active() || m.tracks[src] == nil {
		m.mu.Unlock()
		return
	}
	failures := m.trackFails[src] + 1
	m.trackFails[src] = failures
	runCtx := m.runCtx
	mode := m.mode
	m.mu.Unlock()

	if failures > len(reconnectDelays)+1 {
		m.degradeOrInterrupt(ctx, src)
		return
	}
	var wait time.Duration
	if failures > 1 {
		wait = reconnectDelays[failures-2]
	}
	if wait > 0 {
		select {
		case <-time.After(wait):
		case <-ctx.Done():
			return
		}
	}
	m.mu.Lock()
	if !m.state.Active() || !modeNeedsSource(mode, src) {
		m.mu.Unlock()
		return
	}
	m.mu.Unlock()

	if err := m.runTrack(runCtx, src, m.audioFor(src)); err != nil {
		m.handleTrackLost(ctx, src)
		return
	}
	m.mu.Lock()
	m.trackFails[src] = 0
	m.mu.Unlock()
}

func (m *Manager) degradeOrInterrupt(ctx context.Context, src Source) {
	m.mu.Lock()
	other := otherSource(src)
	otherUp := m.tracks[other] != nil
	mode := m.mode
	m.mu.Unlock()

	if mode == ModeMeeting && otherUp {
		m.mu.Lock()
		problem := &Problem{Key: "meeting.track.degraded." + string(src)}
		if src == SourceSystem {
			m.mode = ModeRecording
		} else {
			m.mode = ModeSubtitles
		}
		m.snap.Mode = m.mode
		delete(m.tracks, src)
		m.mu.Unlock()
		m.refreshProblem(problem)
		return
	}
	_ = m.interrupt(ctx, "")
}

func otherSource(s Source) Source {
	if s == SourceMic {
		return SourceSystem
	}
	return SourceMic
}

// interrupt ends a meeting after capture loss or disk pressure; WAVs stay.
func (m *Manager) interrupt(ctx context.Context, id string) error {
	m.mu.Lock()
	if id != "" {
		if err := m.requireID(id); err != nil {
			m.mu.Unlock()
			return err
		}
	}
	if !m.state.Active() {
		m.mu.Unlock()
		return nil
	}
	m.stopAllTracksLocked()
	if m.archive != nil {
		_ = m.archive.Close()
	}
	if m.cancel != nil {
		m.cancel()
	}
	ended := m.cfg.Now()
	elapsed := m.timeline.Elapsed()
	mt := store.Meeting{
		Seq: m.seq, ID: m.id, CreatedAt: m.snap.Started, EndedAt: ended,
		Status: store.MeetingInterrupted, Mode: toStoreMode(m.mode),
		DurationMS: elapsed.Milliseconds(), Audio: m.id,
	}
	m.state = StateInterrupted
	m.snap.State = StateInterrupted
	m.snap.Elapsed = elapsed
	s := m.snap
	m.mu.Unlock()
	if err := m.cfg.Store.UpdateMeeting(ctx, mt); err != nil {
		return err
	}
	m.mu.Lock()
	m.resetAfterDone()
	m.mu.Unlock()
	m.notify(s)
	return nil
}
