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

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
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
	maxDone  chan struct{}
	failAt   int // after this many frames Record fails with failErr
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
	// file plays a fixed recording instead of the tone: its bytes go out
	// in 100 ms frames (the last one may be short), drained closes once
	// the last frame was delivered, and the recording then waits for
	// Stop, as the helper's file input does.
	file     []byte
	fileTick time.Duration
	drained  chan struct{}
	// notified closes once notifyAt frames were sent (0: never).
	notifyAt     int
	notified     chan struct{}
	firstPCM     chan struct{}
	firstPCMOnce sync.Once

	mu      sync.Mutex
	stops   map[string]chan struct{}
	late    map[string]func(AudioEvent) // the callback of an ended recording
	frames  int
	stopped int
	params  []proto.CaptureParams // of each Record
	running int                   // Records that have not returned
}

func newFakeAudio() *fakeAudio {
	return &fakeAudio{tick: time.Millisecond, stops: map[string]chan struct{}{}, late: map[string]func(AudioEvent){}, firstPCM: make(chan struct{})}
}

func (f *fakeAudio) Record(ctx context.Context, id string, p proto.CaptureParams, fn func(AudioEvent)) (proto.CaptureResult, error) {
	if f.startErr != nil {
		return proto.CaptureResult{}, f.startErr
	}
	stop := make(chan struct{})
	f.mu.Lock()
	f.params = append(f.params, p)
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
	if len(f.file) > 0 {
		var ticker *time.Ticker
		var ticks <-chan time.Time
		if f.fileTick > 0 {
			ticker = time.NewTicker(f.fileTick)
			defer ticker.Stop()
			ticks = ticker.C
		}
		for off := 0; off < len(f.file); off += len(frame) {
			if ticks != nil {
				select {
				case <-ctx.Done():
					return proto.CaptureResult{}, &proto.Error{Code: proto.CodeCancelled, Message: "cancelled"}
				case <-stop:
					return proto.CaptureResult{Reason: proto.CaptureStopped, Frames: n}, nil
				case <-ticks:
				}
			}
			fn(AudioEvent{PCM: &proto.PCMParams{Job: id, Seq: n, Data: f.file[off:min(off+len(frame), len(f.file))]}})
			n++
		}
		f.mu.Lock()
		f.frames = n
		f.mu.Unlock()
		if f.drained != nil {
			close(f.drained)
		}
		select {
		case <-ctx.Done():
			return proto.CaptureResult{}, &proto.Error{Code: proto.CodeCancelled, Message: "cancelled"}
		case <-stop:
			return proto.CaptureResult{Reason: proto.CaptureStopped, Frames: n}, nil
		}
	}
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
			f.firstPCMOnce.Do(func() { close(f.firstPCM) })
			fn(AudioEvent{PCM: &proto.PCMParams{Job: id, Seq: n, Data: data}})
			n++
			f.mu.Lock()
			f.frames++
			f.mu.Unlock()
			if f.notifyAt > 0 && n == f.notifyAt {
				close(f.notified)
			}
			if f.failAt > 0 && n == f.failAt {
				return proto.CaptureResult{}, f.failErr
			}
			if (f.maxFrame > 0 && n == f.maxFrame) || (p.MaxSeconds > 0 && float64(n) >= p.MaxSeconds*10) { // 100 ms frames
				if f.maxDone != nil && f.maxFrame > 0 && n == f.maxFrame {
					close(f.maxDone)
				}
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

// fakeFile transcribes whole recordings.
type fakeFile struct {
	text              string
	said              string // the language it reports (asr.Result.Lang)
	err               error
	hang              bool
	transcribeGate    <-chan struct{}
	transcribeStarted chan<- struct{}

	mu     sync.Mutex
	calls  int
	bytes  int
	audio  []byte // the last request's audio
	lang   string // the last request's language hint
	prompt string // the last request's prompt
	most   int    // the last request's MaxBytes
}

func (f *fakeFile) Caps() asr.Caps { return asr.Caps{File: true, Prompt: true} }

func (f *fakeFile) Transcribe(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	f.mu.Lock()
	f.calls++
	f.bytes = len(req.Audio)
	f.audio = req.Audio
	f.lang, f.prompt, f.most = req.Lang, req.Prompt, req.MaxBytes
	f.mu.Unlock()
	if f.transcribeStarted != nil {
		select {
		case f.transcribeStarted <- struct{}{}:
		default:
		}
	}
	if f.transcribeGate != nil {
		select {
		case <-f.transcribeGate:
		case <-ctx.Done():
			return asr.Result{}, &asr.Error{Kind: asr.KindCancelled, Detail: ctx.Err().Error()}
		}
	}
	if f.hang {
		<-ctx.Done()
		return asr.Result{}, &asr.Error{Kind: asr.KindCancelled, Detail: ctx.Err().Error()}
	}
	return asr.Result{Text: f.text, Lang: f.said, Model: "fake-file"}, f.err
}

// fakeLLM answers chats: text, or texts in turn (the last repeats),
// streamed in pieces of chunk runes to OnDelta (0: in one), each after
// pause; err fails every request, or only request failOn (from 1).
type fakeLLM struct {
	text      string
	texts     []string
	err       error
	failOn    int
	failDelta string
	hang      bool
	hangOn    int    // with hang: only the hangOn-th request hangs (0: all)
	hangDelta string // with hang: the delta streamed before hanging ("Half an ans" when empty)
	chunk     int
	pause     time.Duration
	finish    string // the response's FinishReason
	drop      bool   // the response reports the images dropped (llm.Response.ImagesDropped)
	// copyExample answers with the earlier assistant message when there is
	// one, as a model that repeats the prompt's example does.
	copyExample bool

	mu               sync.Mutex
	reqs             []llm.Request
	answered         int           // the requests answered so far
	answerCh         chan struct{} // closed on the next answer; see answerSignal
	firstRequest     chan struct{}
	firstRequestOnce sync.Once
}

func (f *fakeLLM) ensureSignals() {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.firstRequest == nil {
		f.firstRequest = make(chan struct{})
	}
}

func (f *fakeLLM) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	f.mu.Lock()
	f.reqs = append(f.reqs, req)
	firstRequest := f.firstRequest
	text := f.text
	if len(f.texts) > 0 {
		text = f.texts[min(len(f.reqs), len(f.texts))-1]
	}
	if f.copyExample {
		for _, message := range req.Messages {
			if message.Role == llm.Assistant {
				text = message.Content
			}
		}
	}
	failed := f.err != nil && (f.failOn == 0 || f.failOn == len(f.reqs))
	failDelta := f.failDelta
	hang := f.hang && (f.hangOn == 0 || f.hangOn == len(f.reqs))
	f.mu.Unlock()
	if firstRequest != nil {
		f.firstRequestOnce.Do(func() { close(firstRequest) })
	}
	if hang {
		if req.OnDelta != nil {
			req.OnDelta(cmp.Or(f.hangDelta, "Half an ans"))
		}
		<-ctx.Done()
		return llm.Response{}, ctx.Err()
	}
	if failed {
		if req.OnDelta != nil && failDelta != "" {
			req.OnDelta(failDelta)
		}
		return llm.Response{}, f.err
	}
	if req.OnDelta == nil && f.pause > 0 {
		select {
		case <-time.After(f.pause):
		case <-ctx.Done():
			return llm.Response{}, ctx.Err()
		}
	}
	if req.OnDelta != nil {
		rs := []rune(text)
		n := f.chunk
		if n <= 0 {
			n = len(rs)
		}
		for i := 0; i < len(rs); i += n {
			if f.pause > 0 {
				select {
				case <-time.After(f.pause):
				case <-ctx.Done(): // a real service's stream ends with the request
					return llm.Response{}, ctx.Err()
				}
			}
			req.OnDelta(string(rs[i:min(i+n, len(rs))]))
		}
	}
	f.mu.Lock()
	f.answered++
	if f.answerCh != nil {
		close(f.answerCh)
		f.answerCh = nil
	}
	f.mu.Unlock()
	return llm.Response{Text: text, FinishReason: f.finish, ImagesDropped: f.drop}, nil
}

func (f *fakeLLM) requests() []llm.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]llm.Request(nil), f.reqs...)
}

// answerSignal returns a channel that closes when the next request is answered.
func (f *fakeLLM) answerSignal() <-chan struct{} {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.answerCh == nil {
		f.answerCh = make(chan struct{})
	}
	return f.answerCh
}

func (f *fakeLLM) answeredCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.answered
}

// fakeText is the app in front.
type fakeText struct {
	focus proto.FocusResult
	err   error
	// check is the paste's check when one is asked ("": inserted);
	// secure the app with secure input on.
	check  string
	secure *proto.App
	// onFocus, when set, runs in each focus read before it answers; an
	// error ends that read.
	onFocus func(ctx context.Context) error

	mu      sync.Mutex
	pasted  []string
	presses [][]proto.KeyPress
	verify  []int
}

func (f *fakeText) Focus(ctx context.Context) (proto.FocusResult, error) {
	if f.onFocus != nil {
		if err := f.onFocus(ctx); err != nil {
			return proto.FocusResult{}, err
		}
	}
	return f.focus, f.err
}

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
	dir        string
	mu         sync.Mutex
	entries    []store.Entry
	putStarted chan struct{}
	putRelease chan struct{}
	putBlockID string
	putOnce    sync.Once
}

func (h *memHistory) Put(_ context.Context, e store.Entry) error {
	if h.putStarted != nil && (h.putBlockID == "" || h.putBlockID == e.ID) {
		h.putOnce.Do(func() { close(h.putStarted) })
		<-h.putRelease
	}
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
func (r *recorder) waitFor(t testing.TB, what string, ok func(Snapshot) bool) Snapshot {
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
	text      string
	copyText  string
	copyImage []byte
	steps     []string
	err       error
	delay     time.Duration
	// held reads come first with the ⌘C skipped: the shortcut's keys
	// are still down.
	held int

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
	if f.mu.Lock(); len(f.calls) <= f.held {
		f.mu.Unlock()
		return proto.SelectionResult{Via: proto.ViaNone, MS: 800, Steps: []string{"focus=nil", "copy: skipped held=option"}}, nil
	}
	f.mu.Unlock()
	switch {
	case f.text != "":
		return proto.SelectionResult{Text: f.text, Via: proto.ViaAX, MS: 2}, nil
	case f.copyText != "" && p.AllowCopy:
		return proto.SelectionResult{Text: f.copyText, Image: f.copyImage, Via: proto.ViaCopy, MS: 40}, nil
	}
	return proto.SelectionResult{Via: proto.ViaNone, MS: 1, Steps: f.steps}, nil
}

func (f *fakeSelector) params() []proto.SelectionParams {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]proto.SelectionParams(nil), f.calls...)
}
