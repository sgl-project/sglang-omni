package session

import (
	"cmp"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// frame is 100 ms of a 16 kHz recording of a voice (a 220 Hz tone at
// -20 dBFS, which the voice activity detection takes for speech);
// quietFrame 100 ms of a quiet room.
var (
	frame      = tone(asr.BytesPerSecond/10, 220, 0.14)
	quietFrame = tone(asr.BytesPerSecond/10, 220, 0.0005)
)

// tone is n bytes of PCM16 of a sine of freq Hz and amplitude amp (of
// full scale).
func tone(n int, freq, amp float64) []byte {
	b := make([]byte, n)
	for i := 0; i+1 < n; i += 2 {
		v := int16(amp * 32767 * math.Sin(2*math.Pi*freq*float64(i/2)/asr.SampleRate))
		binary.LittleEndian.PutUint16(b[i:], uint16(v))
	}
	return b
}

// fakeAudio records frames every tick until Stop or ctx.
type fakeAudio struct {
	tick     time.Duration
	startErr error // Record fails before any audio
	maxFrame int   // the recording ends by itself after this many frames (0: never)
	failAt   int   // after this many frames Record fails with failErr
	failErr  error
	deaf     bool // ignore Stop (a capture.stop that raced its start)
	quiet    bool // nobody speaks: quietFrame instead of frame
	// speaks says whether the user speaks in the 100 ms frame n (nil:
	// always, unless quiet).
	speaks func(n int) bool
	// voice is the frame of speech (nil: frame).
	voice []byte
	// startWait: the microphone gives its first audio this late.
	startWait time.Duration

	mu      sync.Mutex
	stops   map[string]chan struct{}
	late    map[string]func(AudioEvent) // the callback of an ended recording
	frames  int
	stopped int
	running int // Records that have not returned
}

func newFakeAudio() *fakeAudio {
	return &fakeAudio{tick: time.Millisecond, stops: map[string]chan struct{}{}, late: map[string]func(AudioEvent){}}
}

func (f *fakeAudio) Record(ctx context.Context, id string, p proto.CaptureParams, fn func(AudioEvent)) (proto.CaptureResult, error) {
	if f.startErr != nil {
		return proto.CaptureResult{}, f.startErr
	}
	stop := make(chan struct{})
	f.mu.Lock()
	f.stops[id] = stop
	f.running++
	f.mu.Unlock()
	defer func() {
		f.mu.Lock()
		f.late[id] = fn
		f.running--
		f.mu.Unlock()
	}()
	if f.startWait > 0 {
		select {
		case <-time.After(f.startWait):
		case <-ctx.Done():
			return proto.CaptureResult{}, &proto.Error{Code: proto.CodeCancelled, Message: "cancelled"}
		case <-stop:
			return proto.CaptureResult{Reason: proto.CaptureStopped}, nil
		}
	}
	fn(AudioEvent{Started: &proto.CaptureStartedParams{Job: id, DeviceName: "MacBook Air Microphone", SampleRate: 16000}})
	n := 0
	t := time.NewTicker(f.tick)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return proto.CaptureResult{}, &proto.Error{Code: proto.CodeCancelled, Message: "cancelled"}
		case <-stop:
			return proto.CaptureResult{Reason: proto.CaptureStopped, Frames: n}, nil
		case <-t.C:
			fn(AudioEvent{Level: &proto.LevelParams{Job: id, RMS: -20}})
			data := frame
			if f.voice != nil {
				data = f.voice
			}
			if f.quiet || f.speaks != nil && !f.speaks(n) {
				data = quietFrame
			}
			fn(AudioEvent{PCM: &proto.PCMParams{Job: id, Seq: n, Data: data}})
			n++
			f.mu.Lock()
			f.frames++
			f.mu.Unlock()
			if f.failAt > 0 && n == f.failAt {
				return proto.CaptureResult{}, f.failErr
			}
			if f.maxFrame > 0 && n == f.maxFrame {
				return proto.CaptureResult{Reason: proto.CaptureMaxSeconds, Frames: n}, nil
			}
		}
	}
}

func (f *fakeAudio) Stop(ctx context.Context, id string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stopped++
	if f.deaf {
		return nil
	}
	if ch, ok := f.stops[id]; ok {
		delete(f.stops, id)
		close(ch)
	}
	return nil
}

func (f *fakeAudio) recorded() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.frames
}

// waitStopped waits for every recording to return, as the helper's
// capture ends when a session cancels or stops it; false when one still
// runs after d.
func (f *fakeAudio) waitStopped(d time.Duration) bool {
	deadline := time.Now().Add(d)
	for {
		f.mu.Lock()
		n := f.running
		f.mu.Unlock()
		if n == 0 {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(time.Millisecond)
	}
}

// lateEvents plays events to an ended recording's callback.
func (f *fakeAudio) lateEvents(id string) {
	f.mu.Lock()
	fn := f.late[id]
	f.mu.Unlock()
	if fn == nil {
		return
	}
	fn(AudioEvent{Level: &proto.LevelParams{Job: id, RMS: 0}})
	fn(AudioEvent{PCM: &proto.PCMParams{Job: id, Data: frame}})
	fn(AudioEvent{Started: &proto.CaptureStartedParams{Job: id, DeviceName: "late"}})
}

// fakeStreamer streams: a partial per frame, the final on Finish.
type fakeStreamer struct {
	final    string
	openErr  error
	finalErr error         // the stream fails instead of answering
	errAfter int           // the stream fails after this many sends
	hang     bool          // no final after Finish
	openWait time.Duration // Open takes this long
	silent   bool          // no partials
	salvage  string        // the text its streams salvage (asr.Salvager), "" for none

	mu      sync.Mutex
	streams []*fakeStream
	hints   []string
}

func (f *fakeStreamer) Caps() asr.Caps { return asr.Caps{Stream: true, Hints: true} }

func (f *fakeStreamer) Open(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	if f.openWait > 0 {
		select {
		case <-time.After(f.openWait):
		case <-ctx.Done():
			return nil, &asr.Error{Kind: asr.KindCancelled}
		}
	}
	if f.openErr != nil {
		return nil, f.openErr
	}
	s := &fakeStream{f: f, ctx: ctx, events: make(chan asr.Event, 1024)}
	f.mu.Lock()
	f.streams = append(f.streams, s)
	f.hints = req.Hints
	f.mu.Unlock()
	return s, nil
}

type fakeStream struct {
	f      *fakeStreamer
	ctx    context.Context
	mu     sync.Mutex
	bytes  int
	data   []byte // what was sent
	sends  int
	closed bool
	done   bool
	events chan asr.Event
}

func (s *fakeStream) emit(ev asr.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.done {
		return
	}
	if ev.Kind != asr.EventPartial {
		s.done = true
	}
	s.events <- ev
	if s.done {
		close(s.events)
	}
}

func (s *fakeStream) Send(pcm []byte) error {
	s.mu.Lock()
	s.bytes += len(pcm)
	s.data = append(s.data, pcm...)
	s.sends++
	n := s.sends
	s.mu.Unlock()
	if s.f.errAfter > 0 && n >= s.f.errAfter {
		s.emit(asr.Event{Kind: asr.EventError, Err: &asr.Error{Kind: asr.KindNetwork, Detail: "connection reset"}})
		return nil
	}
	if !s.f.silent {
		s.emit(asr.Event{Kind: asr.EventPartial, Text: "partial"})
	}
	return nil
}

func (s *fakeStream) Salvage() string { return s.f.salvage }

func (s *fakeStream) Finish() error {
	switch {
	case s.f.hang:
	case s.f.finalErr != nil:
		s.emit(asr.Event{Kind: asr.EventError, Err: s.f.finalErr})
	default:
		s.emit(asr.Event{Kind: asr.EventFinal, Text: s.f.final, Result: asr.Result{Text: s.f.final, Model: "fake-stream"}})
	}
	return nil
}

func (s *fakeStream) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.closed {
		s.closed = true
		if !s.done {
			close(s.events)
		}
	}
}

func (s *fakeStream) Events() <-chan asr.Event { return s.events }

// fakeFile transcribes whole recordings.
type fakeFile struct {
	text string
	err  error
	hang bool

	mu    sync.Mutex
	calls int
	bytes int
	audio []byte // the last request's audio
	lang  string // the last request's language hint
}

func (f *fakeFile) Caps() asr.Caps { return asr.Caps{File: true, Prompt: true} }

func (f *fakeFile) Transcribe(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	f.mu.Lock()
	f.calls++
	f.bytes = len(req.Audio)
	f.audio = req.Audio
	f.lang = req.Lang
	f.mu.Unlock()
	if f.hang {
		<-ctx.Done()
		return asr.Result{}, &asr.Error{Kind: asr.KindCancelled, Detail: ctx.Err().Error()}
	}
	return asr.Result{Text: f.text, Model: "fake-file"}, f.err
}

// fakeLLM answers chats: text, or texts in turn (the last repeats),
// streamed in pieces of chunk runes to OnDelta (0: in one), each after
// pause; err fails every request, or only request failOn (from 1).
type fakeLLM struct {
	text   string
	texts  []string
	err    error
	failOn int
	hang   bool
	chunk  int
	pause  time.Duration

	mu   sync.Mutex
	reqs []llm.Request
}

func (f *fakeLLM) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	f.mu.Lock()
	f.reqs = append(f.reqs, req)
	text := f.text
	if len(f.texts) > 0 {
		text = f.texts[min(len(f.reqs), len(f.texts))-1]
	}
	failed := f.err != nil && (f.failOn == 0 || f.failOn == len(f.reqs))
	f.mu.Unlock()
	if f.hang {
		if req.OnDelta != nil {
			req.OnDelta("Half an ans")
		}
		<-ctx.Done()
		return llm.Response{}, ctx.Err()
	}
	if failed {
		return llm.Response{}, f.err
	}
	if req.OnDelta != nil {
		rs := []rune(text)
		n := f.chunk
		if n <= 0 {
			n = len(rs)
		}
		for i := 0; i < len(rs); i += n {
			if f.pause > 0 {
				time.Sleep(f.pause)
			}
			req.OnDelta(string(rs[i:min(i+n, len(rs))]))
		}
	}
	return llm.Response{Text: text}, nil
}

func (f *fakeLLM) requests() []llm.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]llm.Request(nil), f.reqs...)
}

// fakeText is the app in front.
type fakeText struct {
	focus proto.FocusResult
	err   error
	// check is the paste's check when one is asked ("": inserted);
	// secure the app with secure input on.
	check  string
	secure *proto.App

	mu      sync.Mutex
	pasted  []string
	presses [][]proto.KeyPress
	verify  []int
}

func (f *fakeText) Focus(context.Context) (proto.FocusResult, error) { return f.focus, f.err }

func (f *fakeText) Paste(_ context.Context, p proto.PasteParams) (proto.PasteResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.pasted = append(f.pasted, p.Text)
	f.presses = append(f.presses, p.Press)
	f.verify = append(f.verify, p.VerifyMS)
	res := proto.PasteResult{Posted: true, Restored: true, SecureInput: f.secure}
	if p.VerifyMS > 0 {
		res.Check = cmp.Or(f.check, proto.CheckInserted)
		if res.Check == proto.CheckUnchanged {
			res.Restored, res.Kept, res.PressSkipped = false, true, len(p.Press) > 0
		}
	}
	return res, nil
}

func (f *fakeText) pastes() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.pasted...)
}

// memHistory keeps entries in memory.
type memHistory struct {
	dir     string
	mu      sync.Mutex
	entries []store.Entry
}

func (h *memHistory) Put(_ context.Context, e store.Entry) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.entries = append(h.entries, e)
	return nil
}
func (h *memHistory) Get(context.Context, string) (store.Entry, error) {
	return store.Entry{}, errors.New("no")
}
func (h *memHistory) List(context.Context, store.HistoryQuery) ([]store.Entry, error) {
	return nil, nil
}
func (h *memHistory) Delete(context.Context, string) error                { return nil }
func (h *memHistory) DeleteAll(context.Context, store.Kind) (int, error)  { return 0, nil }
func (h *memHistory) AudioDir() string                                    { return h.dir }
func (h *memHistory) ExpireAudio(context.Context, time.Time) (int, error) { return 0, nil }

func (h *memHistory) all() []store.Entry {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]store.Entry(nil), h.entries...)
}

// recorder keeps every snapshot.
type recorder struct {
	mu    sync.Mutex
	snaps []Snapshot
	ended chan Snapshot
}

func newRecorder() *recorder { return &recorder{ended: make(chan Snapshot, 16)} }

func (r *recorder) add(s Snapshot) {
	r.mu.Lock()
	r.snaps = append(r.snaps, s)
	r.mu.Unlock()
	if s.State.Ended() {
		r.ended <- s
	}
}

func (r *recorder) all() []Snapshot {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]Snapshot(nil), r.snaps...)
}

func (r *recorder) wait(t *testing.T) Snapshot {
	t.Helper()
	select {
	case s := <-r.ended:
		return s
	case <-time.After(5 * time.Second):
		t.Fatalf("the session did not end; last snapshots %+v", tail(r.all()))
	}
	return Snapshot{}
}

// waitFor waits until a snapshot satisfies ok.
func (r *recorder) waitFor(t *testing.T, what string, ok func(Snapshot) bool) Snapshot {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		for _, s := range r.all() {
			if ok(s) {
				return s
			}
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("no snapshot %s; last %+v", what, tail(r.all()))
	return Snapshot{}
}

func tail(s []Snapshot) []Snapshot { return s[max(0, len(s)-3):] }

func writeFileForTest(t *testing.T, dir, name string) bool {
	_, err := os.Stat(filepath.Join(dir, name))
	return err == nil
}

var errHelperForTest = core.ErrHelper

// fakeSelector is the selection of the app in front: text read through
// Accessibility, or copyText only when ⌘C is allowed.
type fakeSelector struct {
	text     string
	copyText string
	err      error
	delay    time.Duration

	mu    sync.Mutex
	calls []proto.SelectionParams
}

func (f *fakeSelector) Selection(ctx context.Context, p proto.SelectionParams) (proto.SelectionResult, error) {
	f.mu.Lock()
	f.calls = append(f.calls, p)
	f.mu.Unlock()
	if f.delay > 0 {
		select {
		case <-time.After(f.delay):
		case <-ctx.Done():
			return proto.SelectionResult{}, ctx.Err()
		}
	}
	if f.err != nil {
		return proto.SelectionResult{}, f.err
	}
	switch {
	case f.text != "":
		return proto.SelectionResult{Text: f.text, Via: proto.ViaAX, MS: 2}, nil
	case f.copyText != "" && p.AllowCopy:
		return proto.SelectionResult{Text: f.copyText, Via: proto.ViaCopy, MS: 40}, nil
	}
	return proto.SelectionResult{Via: proto.ViaNone, MS: 1}, nil
}

func (f *fakeSelector) params() []proto.SelectionParams {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]proto.SelectionParams(nil), f.calls...)
}
