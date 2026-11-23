package meeting

func TrackDoneForTest(m *Manager, src Source) <-chan struct{} {
	m.mu.Lock()
	defer m.mu.Unlock()
	tr := m.tracks[src]
	if tr == nil {
		return nil
	}
	return tr.settled
}
