package meeting

// Importing a recording file as a meeting: the file's decoded sound is
// the meeting's one track (ModeImport), the final pass gives the whole
// file to MTD so every voice is Speaker 1..N, then the same correction
// and summary as a recorded meeting's run.

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// ImportRequest is a recording file the user picked, already decoded.
type ImportRequest struct {
	// Name is the file's name, the meeting's title until the summary
	// names it.
	Name string
	// CreatedAt is the file's date, the meeting's date.
	CreatedAt time.Time
	// PCM is the whole file's sound (16-bit mono, Rate).
	PCM  []byte
	Rate int // 0 is asr.SampleRate
}

// ErrNoHistory is an import refused while Keep a history is off: the
// import saves the whole transcript, and unlike a recorded meeting it
// has no Save/Discard step to ask with.
var ErrNoHistory = errors.New("meeting: Keep a history is off")

// Import stores a recording file as a meeting and runs the pass a
// recorded meeting gets: MTD speaker labels, then the correction and
// summary. A failure before the pass drops the meeting; a failed or
// cancelled pass leaves it interrupted, for Continue processing.
func (m *Manager) Import(ctx context.Context, req ImportRequest, onProgress func(float64)) (string, error) {
	if !m.cfg.Privacy().KeepHistory {
		return "", ErrNoHistory
	}
	rate := req.Rate
	if rate <= 0 {
		rate = asr.SampleRate
	}
	id := core.NewID(m.cfg.Now())
	audioDir := filepath.Join(m.cfg.Dir, id)
	if err := os.MkdirAll(audioDir, 0o755); err != nil {
		return "", err
	}
	fail := func(err error) (string, error) {
		_ = m.cfg.Store.DeleteMeeting(ctx, id)
		_ = os.RemoveAll(audioDir)
		return "", err
	}
	if err := os.WriteFile(filepath.Join(audioDir, string(SourceSystem)+".wav"), asr.WAV(req.PCM, rate), 0o644); err != nil {
		return fail(err)
	}
	dur := asr.Duration(req.PCM, rate)
	mt, err := m.cfg.Store.CreateMeeting(ctx, store.Meeting{
		ID: id, CreatedAt: req.CreatedAt, EndedAt: req.CreatedAt.Add(dur),
		Status: store.MeetingFinishing, Mode: toStoreMode(ModeImport),
		Title: req.Name, DurationMS: dur.Milliseconds(), Audio: id,
	})
	if err != nil {
		return fail(err)
	}
	cfg := m.cfg
	if cfg.Settings().NoFinalPass {
		// The "skip the live final pass" setting is for recorded
		// meetings; an import's only pass is this one.
		inner := cfg.Settings
		cfg.Settings = func() settings.Meeting {
			out := inner()
			out.NoFinalPass = false
			return out
		}
	}
	if err := FinalizePass(ctx, cfg, mt, onProgress); err != nil {
		return id, err
	}
	if !m.cfg.Settings().NoAutoSummary {
		if err := m.correctAndSummarize(ctx, id); err != nil {
			return id, err
		}
	}
	return id, nil
}
