package meeting

// Importing a recording file as a meeting: the file's decoded sound is
// the meeting's one track (ModeImport), the final pass gives it to MTD in
// two-minute requests so every voice is Speaker 1..N, then the same
// correction and summary as a recorded meeting's run.

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
	// Decode writes the recording as wavPath and returns its PCM. A cancel
	// may leave partial decoded audio there for Continue processing. Nil uses PCM.
	Decode func(ctx context.Context, wavPath string) ([]byte, error)
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
	m.mu.Lock()
	if m.state.Active() || m.importing {
		m.mu.Unlock()
		return "", ErrBusy
	}
	m.importing = true
	m.mu.Unlock()
	defer func() {
		m.mu.Lock()
		m.importing = false
		m.mu.Unlock()
	}()
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
	wavPath := filepath.Join(audioDir, string(SourceSystem)+".wav")
	mt := store.Meeting{
		ID: id, CreatedAt: req.CreatedAt, Status: store.MeetingFinishing,
		Mode: toStoreMode(ModeImport), Title: req.Name, Audio: id,
	}
	if req.Decode == nil {
		wav, err := os.Create(wavPath)
		if err != nil {
			return fail(err)
		}
		_, writeErr := wav.Write(asr.WAVHeader(len(req.PCM), rate))
		if writeErr == nil {
			_, writeErr = wav.Write(req.PCM)
		}
		closeErr := wav.Close()
		if writeErr != nil {
			return fail(writeErr)
		}
		if closeErr != nil {
			return fail(closeErr)
		}
		dur := asr.Duration(req.PCM, rate)
		mt.EndedAt = req.CreatedAt.Add(dur)
		mt.DurationMS = dur.Milliseconds()
	}
	mt, err := m.cfg.Store.CreateMeeting(ctx, mt)
	if err != nil {
		return fail(err)
	}
	if req.Decode != nil {
		pcm, decodeErr := req.Decode(ctx, wavPath)
		if decodeErr != nil {
			if errors.Is(decodeErr, context.Canceled) || errors.Is(decodeErr, context.DeadlineExceeded) {
				fixErr := FixWAVHeader(wavPath)
				markErr := markFinalizeFailed(context.WithoutCancel(ctx), m.cfg, mt)
				return id, errors.Join(decodeErr, fixErr, markErr)
			}
			return fail(decodeErr)
		}
		req.PCM = pcm
		dur := asr.Duration(req.PCM, rate)
		mt.EndedAt = req.CreatedAt.Add(dur)
		mt.DurationMS = dur.Milliseconds()
		if err := m.cfg.Store.UpdateMeeting(ctx, mt); err != nil {
			markErr := markFinalizeFailed(context.WithoutCancel(ctx), m.cfg, mt)
			return id, errors.Join(err, markErr)
		}
	}
	if err := finalizePass(ctx, importPassConfig(m.cfg), mt, req.PCM, rate, onProgress); err != nil {
		return id, err
	}
	return id, m.finishImport(ctx, mt, !m.cfg.Settings().NoAutoSummary)
}

// finishImport keeps an import finishing through correction and summary.
// A cancel there leaves it interrupted with its WAV for Continue processing;
// any other result ends done and applies the ordinary audio retention.
func (m *Manager) finishImport(ctx context.Context, mt store.Meeting, summarize bool) error {
	mt.Status = store.MeetingFinishing
	if err := m.cfg.Store.UpdateMeeting(ctx, mt); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			current, _, readErr := m.cfg.Store.Meeting(context.WithoutCancel(ctx), mt.ID)
			if readErr != nil {
				return errors.Join(err, readErr)
			}
			if markErr := markFinalizeFailed(context.WithoutCancel(ctx), m.cfg, current); markErr != nil {
				return errors.Join(err, markErr)
			}
		}
		return err
	}
	var first error
	if summarize {
		first = m.correctAndSummarize(ctx, mt.ID)
	}
	current, _, err := m.cfg.Store.Meeting(context.WithoutCancel(ctx), mt.ID)
	if err != nil {
		return err
	}
	if errors.Is(first, context.Canceled) || errors.Is(first, context.DeadlineExceeded) {
		if err := markFinalizeFailed(context.WithoutCancel(ctx), m.cfg, current); err != nil {
			return errors.Join(first, err)
		}
		return first
	}
	if err := applyAudioRetention(context.WithoutCancel(ctx), m.cfg, current, ModeImport, true); err != nil {
		return err
	}
	current.Status = store.MeetingDone
	if err := m.cfg.Store.UpdateMeeting(context.WithoutCancel(ctx), current); err != nil {
		return err
	}
	return first
}

// importPassConfig forces the final pass on for an import: the "skip the
// live final pass" setting is for recorded meetings; an import's only
// pass is this one.
func importPassConfig(cfg Config) Config {
	if !cfg.Settings().NoFinalPass {
		return cfg
	}
	inner := cfg.Settings
	cfg.Settings = func() settings.Meeting {
		out := inner()
		out.NoFinalPass = false
		return out
	}
	return cfg
}
