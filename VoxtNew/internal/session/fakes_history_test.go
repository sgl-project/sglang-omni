package session

import (
	"context"
	"path/filepath"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// memHistory's clean-up and recordings folder (fakes_test.go): the
// session only writes recordings into AudioDir.

func (h *memHistory) Prune(context.Context, store.Retention) (int, error)      { return 0, nil }
func (h *memHistory) CountPrune(context.Context, store.Retention) (int, error) { return 0, nil }
func (h *memHistory) DeleteAudio(context.Context) (int, error)                 { return 0, nil }
func (h *memHistory) DefaultAudioDir() string                                  { return h.dir }
func (h *memHistory) SetAudioDir(string)                                       {}
func (h *memHistory) AudioPath(name string) string                             { return filepath.Join(h.dir, name) }
