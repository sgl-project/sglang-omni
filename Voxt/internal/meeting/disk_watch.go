package meeting

import (
	"context"
	"time"
)

func (m *Manager) watchDisk(ctx context.Context) {
	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
			free, err := m.cfg.FreeSpace(m.cfg.Dir)
			if err == nil {
				m.noteFreeSpace(free)
			}
			if err != nil || free >= diskMinRecordBytes {
				continue
			}
			m.mu.Lock()
			id := m.id
			active := m.state.Active()
			m.mu.Unlock()
			if active && id != "" {
				_ = m.interrupt(context.Background(), id)
			}
			return
		}
	}
}
