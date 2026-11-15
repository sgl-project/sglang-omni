package session

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Manager runs one session at a time. Goroutine-safe.
type Manager struct {
	cfg Config

	mu sync.Mutex
	// pub orders publication: taken before mu is released, so snapshots
	// reach OnChange in the order of their revisions.
	pub  sync.Mutex
	cur  *run
	snap Snapshot // the current (or last) session's
	rev  uint64
	last string // the last result delivered or shown, for "paste the last result"
}

// New returns a Manager; nothing runs until Start.
func New(cfg Config) *Manager {
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.Settings == nil {
		cfg.Settings = settings.Defaults
	}
	if cfg.Locale == nil {
		cfg.Locale = func() string { return "en" }
	}
	if cfg.Log == nil {
		cfg.Log = func(string, string) {}
	}
	if cfg.NewID == nil {
		var n int
		var mu sync.Mutex
		cfg.NewID = func() string { mu.Lock(); defer mu.Unlock(); n++; return "s" + itoa(n) }
	}
	def := func(d *time.Duration, v time.Duration) {
		if *d <= 0 {
			*d = v
		}
	}
	def(&cfg.MinRecording, DefaultMinRecording)
	def(&cfg.ASRTimeout, DefaultASRTimeout)
	def(&cfg.ASRPerSecond, DefaultASRPerSecond)
	def(&cfg.AudioRetention, DefaultAudioRetention)
	def(&cfg.LLMBase, DefaultLLMBase)
	def(&cfg.LLMStall, DefaultLLMStall)
	def(&cfg.LLMPerChar, DefaultLLMPerChar)
	def(&cfg.LLMTimeout, DefaultLLMTimeout)
	def(&cfg.WriteTimeout, DefaultWriteTimeout)
	def(&cfg.ReadTimeout, DefaultReadTimeout)
	def(&cfg.RewriteReadTimeout, DefaultRewriteReadTimeout)
	def(&cfg.DeliverTime, DefaultDeliverTimeout)
	def(&cfg.StopGrace, DefaultStopGrace)
	def(&cfg.EndWordPause, DefaultEndWordPause)
	def(&cfg.WaitNote, DefaultWaitNote)
	return &Manager{cfg: cfg, snap: Snapshot{State: Idle}}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for ; n > 0; n /= 10 {
		b = append([]byte{byte('0' + n%10)}, b...)
	}
	return string(b)
}

// Current is the current session's snapshot (or the last one's, ended;
// Idle before the first).
func (m *Manager) Current() Snapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.snap.clone()
}

// Last is the last result a session delivered or showed ("" for none):
// what "paste the last result" pastes.
func (m *Manager) Last() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.last
}

// SetLast remembers a result as the last one (the app reads the newest
// history entry at launch).
func (m *Manager) SetLast(text string) {
	m.mu.Lock()
	if m.last == "" {
		m.last = text
	}
	m.mu.Unlock()
}

// Start begins a session: it records at once, and resolves the services
// meanwhile. It fails with ErrBusy while a session is live.
func (m *Manager) Start(req StartRequest) (string, error) {
	m.mu.Lock()
	if m.cur != nil && m.snap.State.Live() {
		m.mu.Unlock()
		return "", ErrBusy
	}
	at := req.At
	if at.IsZero() {
		at = m.cfg.Now()
	}
	ctx, cancel := context.WithCancel(context.Background())
	r := &run{
		m: m, id: m.cfg.NewID(), req: req, settings: m.cfg.Settings(), ctx: ctx, cancel: cancel,
		stop: make(chan struct{}), recorded: make(chan struct{}), pcm: newPCMBuffer(),
	}
	r.llmCtx, r.llmStop = context.WithCancel(ctx)
	m.cur = r
	m.rev++
	m.snap = Snapshot{
		Rev: m.rev, ID: r.id, Kind: req.Kind, State: r.firstState(), Try: req.Try, Note: req.Note, HoldBinding: req.HoldBinding,
		Target: req.Target, StartedAt: at, MaxLen: time.Duration(r.settings.Audio.MaxSeconds) * time.Second,
	}
	if req.Kind == store.KindTranslate {
		r.target = TranslateTarget(r.settings, m.cfg.Locale(), req.TargetLanguage)
		m.snap.TargetLanguage = r.target
	}
	if t := req.Text; t != nil {
		m.snap.Raw, m.snap.Selection = strings.TrimSpace(t.Raw), t.Selection
	}
	s := m.snap.clone()
	m.pub.Lock()
	m.mu.Unlock()
	m.publish(s)
	m.pub.Unlock()
	r.startScene()
	go r.main()
	return r.id, nil
}

// Stop ends the recording of session id; it goes on to recognition. A
// session that is not recording, or not the current one, is left alone.
func (m *Manager) Stop(id string) {
	m.mu.Lock()
	r := m.cur
	ok := r != nil && r.id == id && m.snap.State == Recording
	m.mu.Unlock()
	if ok {
		r.stopOnce.Do(func() { close(r.stop) })
	}
}

// Cancel ends session id where it is: nothing more is delivered or kept.
func (m *Manager) Cancel(id string) {
	m.mu.Lock()
	r := m.cur
	ok := r != nil && r.id == id && m.snap.State.Live()
	m.mu.Unlock()
	if ok {
		r.cancelled.Store(true)
		r.cancel()
	}
}

// SkipLLM ends session id's language model step where it is, when its
// snapshot says it can (Snapshot.CanSkip): the recognised text goes on
// in place of the answer (#3821). Otherwise it does nothing.
func (m *Manager) SkipLLM(id string) {
	m.mu.Lock()
	r := m.cur
	if r != nil && r.id == id && m.snap.State == Composing && m.snap.CanSkip {
		r.skipped.Store(true)
		r.llmStop()
	}
	m.mu.Unlock()
}

// StopLLM ends session id's answer where it is, when its snapshot says it
// can (Snapshot.CanStop): the request to the model ends and the card keeps
// the answer so far, to copy, insert or follow up (#3886).
func (m *Manager) StopLLM(id string) {
	m.mu.Lock()
	r := m.cur
	if r != nil && r.id == id && m.snap.CanStop() {
		r.stopped.Store(true)
		r.llmStop()
	}
	m.mu.Unlock()
}

// Rethink asks the language model again at thinking level t (nil: the
// service's setting) while it cleans up session id's Transcribe text
// (#3940, the preview's thinking menu): the request ends where it is and
// the text is cleaned up again from the start. The service's setting
// stays. Otherwise it does nothing.
func (m *Manager) Rethink(id string, t *llm.Thinking) {
	m.mu.Lock()
	r := m.cur
	m.mu.Unlock()
	if r == nil || r.id != id {
		return
	}
	m.update(r, func(s *Snapshot) {
		if s.Kind != store.KindTranscribe || s.State != Composing || !s.CanSkip || r.skipped.Load() {
			return
		}
		s.Thinking, r.think = t, t
		r.rethink.Store(true)
		r.llmStop()
	})
}

// update changes the snapshot of run r, if r is still the current,
// live session, and publishes it. It reports whether it did.
func (m *Manager) update(r *run, fn func(*Snapshot)) bool {
	m.mu.Lock()
	if m.cur != r || r.ended {
		m.mu.Unlock()
		return false
	}
	fn(&m.snap)
	m.rev++
	m.snap.Rev = m.rev
	s := m.snap.clone()
	m.pub.Lock()
	m.mu.Unlock()
	m.publish(s)
	m.pub.Unlock()
	return true
}

// end makes r's session end in state st (once). Before the end is
// published, kept (when not nil) gets the final snapshot: the history
// entry exists by the time the UI sees the end.
func (m *Manager) end(r *run, st State, fn func(*Snapshot), kept func(Snapshot)) bool {
	m.mu.Lock()
	if m.cur != r || r.ended {
		m.mu.Unlock()
		return false
	}
	if fn != nil {
		fn(&m.snap)
	}
	m.snap.State = st
	r.ended = true
	if st == Done && m.snap.Result != "" && !m.snap.Try && !m.snap.Note {
		m.last = m.snap.Result
	}
	m.rev++
	m.snap.Rev = m.rev
	s := m.snap.clone()
	m.pub.Lock()
	m.mu.Unlock()
	if kept != nil {
		kept(s)
	}
	m.publish(s)
	m.pub.Unlock()
	return true
}

func (m *Manager) publish(s Snapshot) {
	if m.cfg.OnChange != nil {
		m.cfg.OnChange(s)
	}
}
