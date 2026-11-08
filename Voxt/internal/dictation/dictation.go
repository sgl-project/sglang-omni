// Package dictation is System dictation: macOS's own speech recognition
// (SFSpeechRecognizer), on this Mac only, as a speech model (an
// asr.Streamer, and a FileTranscriber for History's Retry) over the
// helper's speech.* methods; and Services, the
// local models with it listed after their speech models while the helper
// can recognise. It is built into macOS: nothing to download, and it is
// not in the models catalogue.
//
// The language is the session's speech hint (the user's main language);
// "" is the Mac's own dictation language. macOS without an on-device
// model for it fails the stream's Open with ErrNoLanguage, Speech
// Recognition not allowed with ErrNotAllowed.
//
// Imports: asr, helper/proto, local.
package dictation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
)

// ModelID is System dictation's id among the speech models (settings
// keep it).
const ModelID = "asr.system"

// ProviderName is asr.Error.Provider of its errors.
const ProviderName = "system"

var (
	// ErrNotAllowed: macOS's Speech Recognition is not allowed for Voxt.
	ErrNotAllowed = errors.New("dictation: Speech Recognition is not allowed")
	// ErrNoLanguage: macOS has no on-device model for the language.
	ErrNoLanguage = errors.New("dictation: no on-device speech recognition for the language")
)

// Caller is how the provider calls the helper (helper.Supervisor).
type Caller interface {
	Call(ctx context.Context, id, method string, params any, notify func(proto.Message)) (json.RawMessage, error)
}

// Provider recognises speech with macOS's recognizer, through the helper.
type Provider struct{ Helper Caller }

func (Provider) Caps() asr.Caps { return asr.Caps{Stream: true, File: true, Hints: true} }

// Transcribe recognises a whole recording (PCM) by streaming it: History's
// Retry. A live session streams (Caps.Stream comes first).
func (p Provider) Transcribe(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	return asr.StreamAll(ctx, p, asr.StreamRequest{SampleRate: req.SampleRate, Lang: req.Lang, Hints: req.Hints}, req.Audio, asr.Feed{})
}

// jobs numbers the speech.recognize jobs, whose ids speech.audio names.
var jobs atomic.Int64

// Open starts a speech.recognize job and returns once its recognizer
// runs (speech.started), or the job failed.
func (p Provider) Open(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	ctx, cancel := context.WithCancel(ctx)
	s := &stream{helper: p.Helper, ctx: ctx, cancel: cancel, job: fmt.Sprintf("speech-%d", jobs.Add(1)),
		started: make(chan struct{}), ended: make(chan error, 1), events: make(chan asr.Event, 64)}
	params := proto.SpeechParams{Lang: req.Lang, SampleRate: req.SampleRate, Hints: req.Hints}
	go func() {
		raw, err := p.Helper.Call(ctx, s.job, proto.MethodSpeechRecognize, params, s.notified)
		var res proto.SpeechResult
		if err == nil {
			err = json.Unmarshal(raw, &res)
		}
		s.final = res.Text
		s.ended <- err
	}()
	select {
	case <-s.started:
	case err := <-s.ended:
		if err != nil {
			cancel()
			return nil, asrError(ctx, err)
		}
		s.ended <- nil // answered already: it started
	}
	go s.forward()
	return s, nil
}

// stream is one speech.recognize job.
type stream struct {
	helper  Caller
	ctx     context.Context
	cancel  context.CancelFunc
	job     string
	started chan struct{}
	once    sync.Once
	ended   chan error // the job's answer; final is set first
	final   string
	events  chan asr.Event
	closed  atomic.Bool

	mu      sync.Mutex
	partial string // the last speech.partial, for Salvage
}

// notified takes the job's notifications (on the helper's reader: it
// must not block).
func (s *stream) notified(m proto.Message) {
	switch m.Method {
	case proto.MethodSpeechStarted:
		s.once.Do(func() { close(s.started) })
	case proto.MethodSpeechPartial:
		var p proto.SpeechPartialParams
		if m.DecodeParams(&p) != nil {
			return
		}
		s.mu.Lock()
		s.partial = p.Text
		s.mu.Unlock()
		select {
		case s.events <- asr.Event{Kind: asr.EventPartial, Text: p.Text}:
		default: // a later partial repeats it
		}
	}
}

// forward ends Events with the job's answer (none after Close).
func (s *stream) forward() {
	defer close(s.events)
	err := <-s.ended
	switch {
	case s.closed.Load():
	case err != nil:
		s.events <- asr.Event{Kind: asr.EventError, Err: asrError(s.ctx, err)}
	default:
		s.events <- asr.Event{Kind: asr.EventFinal, Text: s.final, Result: asr.Result{Text: s.final, Model: ModelID}}
	}
}

func (s *stream) Send(pcm []byte) error {
	return s.audio(proto.SpeechAudioParams{Job: s.job, Data: pcm})
}

func (s *stream) Finish() error {
	return s.audio(proto.SpeechAudioParams{Job: s.job, End: true})
}

func (s *stream) audio(p proto.SpeechAudioParams) error {
	if _, err := s.helper.Call(s.ctx, "", proto.MethodSpeechAudio, p, nil); err != nil {
		return asrError(s.ctx, err)
	}
	return nil
}

func (s *stream) Close() {
	s.closed.Store(true)
	s.cancel() // the supervisor cancels the job
}

func (s *stream) Events() <-chan asr.Event { return s.events }

// Salvage is the text heard so far (asr.Salvager).
func (s *stream) Salvage() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.partial
}

// asrError is the helper's error as the session reads a speech model's.
func asrError(ctx context.Context, err error) error {
	var pe *proto.Error
	if !errors.As(err, &pe) {
		return asr.NetError(ctx, ProviderName, err) // the helper went away, or the session cancelled
	}
	e := &asr.Error{Kind: asr.KindServer, Provider: ProviderName, Code: pe.Code, Detail: pe.Message, Err: err}
	switch pe.Code {
	case proto.CodePermissionDenied:
		e.Kind, e.Err = asr.KindConfig, ErrNotAllowed
	case proto.CodeUnsupported:
		e.Kind, e.Err = asr.KindConfig, ErrNoLanguage
	case proto.CodeCancelled:
		e.Kind = asr.KindCancelled
	}
	return e
}

// Services is base with System dictation listed after its speech models
// while a helper that can recognise runs (SetHelper). Goroutine-safe.
type Services struct {
	local.Services
	name   func(id string) string
	helper atomic.Pointer[Caller]
}

// NewServices is base and System dictation; name says a model's Name
// (uitext.ModelName).
func NewServices(base local.Services, name func(id string) string) *Services {
	return &Services{Services: base, name: name}
}

// SetHelper sets the helper that recognises: nil when none runs that
// can (not ready, or older than speech.recognize).
func (s *Services) SetHelper(h Caller) {
	if h == nil {
		s.helper.Store(nil)
	} else {
		s.helper.Store(&h)
	}
}

func (s *Services) Installed(kind local.Kind) []local.Model {
	out := s.Services.Installed(kind)
	if kind == local.ASR && s.helper.Load() != nil {
		out = append(slices.Clip(out), local.Model{Kind: local.ASR, ID: ModelID, Name: s.name(ModelID)})
	}
	return out
}

func (s *Services) ASR(ctx context.Context, id string) (asr.Provider, local.Model, error) {
	m, ok := local.Choose(s.Installed(local.ASR), id)
	h := s.helper.Load()
	if !ok || m.ID != ModelID || h == nil {
		return s.Services.ASR(ctx, id)
	}
	return Provider{Helper: *h}, m, nil
}
