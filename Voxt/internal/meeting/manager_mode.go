package meeting

import (
	"context"
	"fmt"
)

// SetMode switches the meeting to mode without resetting the timeline:
// the tracks both modes record go on recording, those only the new mode
// has start, and those only the old one has stop. The new tracks start
// first, so a failure (system audio not allowed, say) leaves the meeting
// recording as it was.
func (m *Manager) SetMode(_ context.Context, id string, mode Mode) error {
	m.modeMu.Lock()
	defer m.modeMu.Unlock()
	m.mu.Lock()
	if err := m.requireID(id); err != nil {
		m.mu.Unlock()
		return err
	}
	if m.state != StateRecording && m.state != StatePaused {
		m.mu.Unlock()
		return fmt.Errorf("meeting: not active")
	}
	old, next := m.mode.Sources(), mode.Sources()
	if m.state == StatePaused { // nothing records: Resume starts the new mode's tracks
		m.mode = mode
		m.snap.Mode = mode
		s := m.snap
		m.mu.Unlock()
		m.notify(s)
		return nil
	}
	runCtx := m.runCtx
	m.mu.Unlock()
	added, dropped := sourcesToAdd(old, next), sourcesToAdd(next, old)
	for _, src := range added {
		if se := m.runTrack(runCtx, src, m.audioFor(src)); se != nil {
			m.mu.Lock()
			m.dropTracksLocked(added)
			m.mu.Unlock()
			return se
		}
	}
	if m.modeStarted != nil {
		m.modeStarted()
	}
	m.mu.Lock()
	if m.id == id && m.state == StatePaused {
		// Paused while the tracks started: Pause stopped them, and Resume
		// starts the new mode's, as when SetMode found it paused.
		m.dropTracksLocked(added) // lets go of mu while they stop
		if m.id == id && m.state == StatePaused {
			m.mode = mode
			m.snap.Mode = mode
			s := m.snap
			m.mu.Unlock()
			m.notify(s)
			return nil
		}
		m.mu.Unlock() // resumed meanwhile, in the old mode
		return fmt.Errorf("meeting: not active")
	}
	if m.id != id || m.state != StateRecording { // ended while the tracks started
		m.dropTracksLocked(added)
		m.mu.Unlock()
		return fmt.Errorf("meeting: not active")
	}
	m.mode = mode
	m.snap.Mode = mode
	m.dropTracksLocked(dropped)
	s := m.snap
	m.mu.Unlock()
	m.notify(s)
	return nil
}

// dropTracksLocked stops the tracks of sources and forgets them and
// their losses: the mode does not have them, so nothing would stop or
// reopen them later, and a track that could not start must not count
// towards giving the source up (handleTrackLost). It lets go of mu while
// they stop.
func (m *Manager) dropTracksLocked(sources []Source) {
	m.stopTracksLocked(sources)
	for _, src := range sources {
		if tr := m.tracks[src]; tr != nil && tr.stopping { // not one reopened meanwhile
			delete(m.tracks, src)
		}
		delete(m.trackFails, src)
	}
}

// sourcesToAdd are the sources of new that old lacks.
func sourcesToAdd(old, new []Source) []Source {
	have := map[Source]bool{}
	for _, s := range old {
		have[s] = true
	}
	var add []Source
	for _, s := range new {
		if !have[s] {
			add = append(add, s)
		}
	}
	return add
}
