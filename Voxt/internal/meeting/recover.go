package meeting

import (
	"context"
	"os"
	"path/filepath"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Recover marks unfinished meetings interrupted and repairs empty WAV headers.
func (m *Manager) Recover(ctx context.Context) ([]store.Meeting, error) {
	list, err := m.cfg.Store.Unfinished(ctx)
	if err != nil {
		return nil, err
	}
	var out []store.Meeting
	for _, mt := range list {
		dir := filepath.Join(m.cfg.Dir, mt.Audio)
		for _, name := range []string{"mic.wav", "system.wav"} {
			path := filepath.Join(dir, name)
			if _, err := os.Stat(path); err == nil {
				_ = FixWAVHeader(path)
			}
		}
		mt.Status = store.MeetingInterrupted
		if mt.EndedAt.IsZero() {
			mt.EndedAt = m.now()
		}
		if err := m.cfg.Store.UpdateMeeting(ctx, mt); err != nil {
			return nil, err
		}
		out = append(out, mt)
	}
	return out, nil
}

func (m *Manager) now() time.Time {
	if m.cfg.Now != nil {
		return m.cfg.Now()
	}
	return time.Now()
}
