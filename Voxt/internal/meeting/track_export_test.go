package meeting

import "time"

// SetLostFinalWaitForTest shortens how long a lost track waits for its
// Final and returns the restore.
func SetLostFinalWaitForTest(d time.Duration) func() {
	old := lostFinalWait
	lostFinalWait = d
	return func() { lostFinalWait = old }
}

func TrackDoneForTest(m *Manager, src Source) <-chan struct{} {
	m.mu.Lock()
	defer m.mu.Unlock()
	tr := m.tracks[src]
	if tr == nil {
		return nil
	}
	return tr.settled
}
