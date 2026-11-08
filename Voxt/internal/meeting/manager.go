package meeting

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Manager owns at most one live meeting and writes its WAV archives.
type Manager struct {
	cfg Config

	mu         sync.Mutex
	id         string
	seq        int64
	mode       Mode
	state      State
	snap       Snapshot
	timeline   *Timeline
	archive    *Archive
	cancel     context.CancelFunc
	runCtx     context.Context
	tracks     map[Source]*track
	trackFails map[Source]int
	jobGen     int
	subs       []func(Snapshot)
}

// NewManager returns a meeting manager from cfg.
func NewManager(cfg Config) *Manager {
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.FreeSpace == nil {
		cfg.FreeSpace = func(string) (int64, error) {
			return diskMinStartBytes * 2, nil
		}
	}
	if cfg.Settings == nil {
		cfg.Settings = func() settings.Meeting { return settings.Meeting{} }
	}
	if cfg.Privacy == nil {
		cfg.Privacy = func() settings.Privacy {
			return settings.Privacy{KeepHistory: true}
		}
	}
	return &Manager{cfg: cfg, tracks: map[Source]*track{}, trackFails: map[Source]int{}}
}

// Start opens a meeting and begins capture on every track mode needs.
func (m *Manager) Start(ctx context.Context, req StartRequest) (string, error) {
	if req.Mode == "" {
		req.Mode = ModeMeeting
	}
	free, err := m.cfg.FreeSpace(m.cfg.Dir)
	if err != nil {
		return "", err
	}
	if free < diskMinStartBytes {
		return "", fmt.Errorf("meeting: not enough disk space")
	}
	m.mu.Lock()
	if m.state.Active() {
		m.mu.Unlock()
		return "", ErrBusy
	}
	started := m.cfg.Now()
	id := core.NewID(started)
	m.id = id
	m.mode = req.Mode
	m.state = StateStarting
	m.jobGen = 0
	m.timeline = newTimeline(m.cfg.Now, started)
	m.snap = Snapshot{
		ID: id, Mode: req.Mode, State: StateStarting, Started: started,
		Level: map[Source]float64{SourceMic: -120, SourceSystem: -120},
		Live:  map[Source]string{},
	}
	rec := m.snap
	m.mu.Unlock()
	m.notify(rec)

	folder := id
	audioDir := filepath.Join(m.cfg.Dir, folder)
	if err := os.MkdirAll(audioDir, 0o755); err != nil {
		return "", err
	}
	mt := store.Meeting{
		ID: id, CreatedAt: started, Status: store.MeetingRecording,
		Mode: toStoreMode(req.Mode), Audio: folder,
	}
	created, err := m.cfg.Store.CreateMeeting(ctx, mt)
	if err != nil {
		return "", err
	}
	m.mu.Lock()
	m.seq = created.Seq
	m.archive = newArchive(audioDir)
	runCtx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel
	m.runCtx = runCtx
	m.trackFails = map[Source]int{}
	m.mu.Unlock()
	go m.watchDisk(runCtx)

	sources := req.Mode.Sources()
	if err := m.startTracks(runCtx, sources); err != nil {
		cancel()
		_ = m.cfg.Store.DeleteMeeting(ctx, id)
		_ = os.RemoveAll(audioDir)
		m.mu.Lock()
		m.resetIdle()
		m.mu.Unlock()
		m.notify(m.snap)
		return "", err
	}
	m.mu.Lock()
	m.state = StateRecording
	m.snap.State = StateRecording
	m.snap.Elapsed = m.timeline.Elapsed()
	m.mu.Unlock()
	m.notify(m.snap)
	return id, nil
}

func (m *Manager) startTracks(ctx context.Context, sources []Source) error {
	for _, src := range sources {
		se := m.runTrack(ctx, src, m.audioFor(src))
		if se != nil {
			m.stopAllTracks()
			return se
		}
	}
	return nil
}

// Pause stops capture without ending the meeting.
func (m *Manager) Pause(_ context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.requireID(id); err != nil {
		return err
	}
	if m.state != StateRecording {
		return fmt.Errorf("meeting: not recording")
	}
	m.stopAllTracksLocked()
	m.timeline.NotePause()
	m.state = StatePaused
	m.snap.State = StatePaused
	m.snap.Elapsed = m.timeline.Elapsed()
	return nil
}

// Resume restarts capture with new job ids.
func (m *Manager) Resume(ctx context.Context, id string) error {
	m.mu.Lock()
	if err := m.requireID(id); err != nil {
		m.mu.Unlock()
		return err
	}
	if m.state != StatePaused {
		m.mu.Unlock()
		return fmt.Errorf("meeting: not paused")
	}
	m.timeline.NoteResume()
	m.state = StateRecording
	m.snap.State = StateRecording
	sources := m.mode.Sources()
	runCtx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel
	m.runCtx = runCtx
	m.mu.Unlock()
	if err := m.startTracks(runCtx, sources); err != nil {
		return err
	}
	m.mu.Lock()
	m.snap.Elapsed = m.timeline.Elapsed()
	m.mu.Unlock()
	m.notify(m.snap)
	return nil
}

// SetMode switches capture tracks without resetting the timeline.
func (m *Manager) SetMode(ctx context.Context, id string, mode Mode) error {
	m.mu.Lock()
	if err := m.requireID(id); err != nil {
		m.mu.Unlock()
		return err
	}
	if m.state != StateRecording && m.state != StatePaused {
		m.mu.Unlock()
		return fmt.Errorf("meeting: not active")
	}
	wasPaused := m.state == StatePaused
	old := m.mode.Sources()
	m.stopAllTracksLocked()
	m.mode = mode
	m.snap.Mode = mode
	if wasPaused {
		m.mu.Unlock()
		return nil
	}
	sources := mode.Sources()
	runCtx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel
	m.runCtx = runCtx
	m.mu.Unlock()
	// Start any newly required sources; already-running overlap keeps timeline.
	need := sourcesToAdd(old, sources)
	if len(need) == 0 {
		return nil
	}
	return m.startTracks(runCtx, need)
}

// Finish ends capture and persists the meeting row.
func (m *Manager) Finish(ctx context.Context, id string) error {
	m.mu.Lock()
	if err := m.requireID(id); err != nil {
		m.mu.Unlock()
		return err
	}
	if !m.state.Active() {
		m.mu.Unlock()
		return fmt.Errorf("meeting: not active")
	}
	keep := m.cfg.Privacy().KeepHistory
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
		Status: store.MeetingDone, Mode: toStoreMode(m.mode),
		DurationMS: elapsed.Milliseconds(), Audio: m.id,
	}
	if !keep {
		m.state = StateAsking
		m.snap.State = StateAsking
		mt.Status = store.MeetingAsking
		m.mu.Unlock()
		_ = m.cfg.Store.UpdateMeeting(ctx, mt)
		m.notify(m.snap)
		return nil
	}
	m.state = StateDone
	m.snap.State = StateDone
	m.snap.Elapsed = elapsed
	m.mu.Unlock()
	if err := m.cfg.Store.UpdateMeeting(ctx, mt); err != nil {
		return err
	}
	m.resetAfterDone()
	m.notify(m.snap)
	return nil
}

// Save keeps an asking-state meeting.
func (m *Manager) Save(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.requireID(id); err != nil {
		return err
	}
	if m.state != StateAsking {
		return fmt.Errorf("meeting: not asking")
	}
	mt := store.Meeting{
		Seq: m.seq, ID: m.id, CreatedAt: m.snap.Started, EndedAt: m.cfg.Now(),
		Status: store.MeetingDone, Mode: toStoreMode(m.mode),
		DurationMS: m.timeline.Elapsed().Milliseconds(), Audio: m.id,
	}
	if err := m.cfg.Store.UpdateMeeting(ctx, mt); err != nil {
		return err
	}
	m.state = StateDone
	m.snap.State = StateDone
	m.resetAfterDone()
	return nil
}

// Discard drops the meeting and deletes its files.
func (m *Manager) Discard(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.requireID(id); err != nil {
		return err
	}
	if !m.state.Active() && m.state != StateAsking {
		return fmt.Errorf("meeting: not active")
	}
	folder := filepath.Join(m.cfg.Dir, m.id)
	_ = m.cfg.Store.DeleteMeeting(ctx, m.id)
	_ = os.RemoveAll(folder)
	m.resetIdle()
	return nil
}

// Current returns the latest snapshot.
func (m *Manager) Current() Snapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	s := m.snap
	if m.timeline != nil {
		s.Elapsed = m.timeline.Elapsed()
	}
	return s
}

// Subscribe calls fn whenever the snapshot changes.
func (m *Manager) Subscribe(fn func(Snapshot)) (cancel func()) {
	m.mu.Lock()
	m.subs = append(m.subs, fn)
	m.mu.Unlock()
	fn(m.Current())
	return func() {
		m.mu.Lock()
		defer m.mu.Unlock()
		for i, f := range m.subs {
			if fmt.Sprintf("%p", f) == fmt.Sprintf("%p", fn) {
				m.subs = append(m.subs[:i], m.subs[i+1:]...)
				return
			}
		}
	}
}

// Reprocess is implemented in C1-3a.
func (m *Manager) Reprocess(string) error {
	return fmt.Errorf("meeting: reprocess not implemented")
}

func (m *Manager) audioFor(src Source) session.Audio {
	switch src {
	case SourceMic:
		return m.cfg.Mic
	case SourceSystem:
		return m.cfg.System
	default:
		return nil
	}
}

func (m *Manager) nextJob(src Source) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.jobGen++
	return fmt.Sprintf("%s/%s/%d", m.id, src, m.jobGen)
}

// SetTranslate toggles live translation target language (C1-4 wires the UI).
func (m *Manager) SetTranslate(id, target string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.requireID(id); err != nil {
		return err
	}
	m.snap.Translate = target
	return nil
}

func (m *Manager) requireID(id string) error {
	if m.id != id {
		return ErrNoMeeting
	}
	return nil
}

func (m *Manager) notify(s Snapshot) {
	for _, fn := range m.subs {
		fn(s)
	}
}

func (m *Manager) resetIdle() {
	m.id = ""
	m.seq = 0
	m.state = StateIdle
	m.snap = Snapshot{
		State: StateIdle,
		Level: map[Source]float64{}, Live: map[Source]string{},
	}
	m.timeline = nil
	m.archive = nil
	m.tracks = map[Source]*track{}
	m.trackFails = map[Source]int{}
	m.runCtx = nil
}

func (m *Manager) resetAfterDone() {
	m.resetIdle()
}

func (m *Manager) stopAllTracks() {
	m.mu.Lock()
	m.stopAllTracksLocked()
	m.mu.Unlock()
}

func (m *Manager) stopAllTracksLocked() {
	for _, tr := range m.tracks {
		tr.stop()
	}
}

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
