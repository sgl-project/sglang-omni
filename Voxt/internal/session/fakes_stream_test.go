package session

import (
	"context"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// fakeStreamer streams: a partial per frame, the final on Finish.
type fakeStreamer struct {
	final             string
	openErr           error
	finalErr          error           // the stream fails instead of answering
	errAfter          int             // the stream fails after this many sends
	hang              bool            // no final after Finish
	openWait          time.Duration   // Open takes this long
	openGate          chan struct{}   // Open waits for it to close (nil: no wait)
	finishGate        <-chan struct{} // Finish waits for it to close (nil: no wait)
	finishStarted     chan struct{}   // signalled once Finish begins
	finishEmitted     chan bool       // whether Finish emitted its Final event
	partialGate       <-chan struct{}
	partialReturnGate <-chan struct{}
	silent            bool // no partials
	// partials are what its partials say, one per send, the last one
	// repeated (nil: "partial").
	partials []string
	// pieces are the cumulative "final so far" texts it reports while
	// recording (asr.EventPiece): pieces[i-1] after the i-th send.
	pieces              []string
	candidates          []string
	salvage             string // the text its streams salvage (asr.Salvager), "" for none
	salvageRequiresOpen bool
	onSend              func(int) // called after the i-th send (nil: not)

	mu               sync.Mutex
	streams          []*fakeStream
	openStarted      chan struct{}
	openDone         chan struct{}
	firstSend        chan struct{}
	firstPartial     chan struct{}
	finalEvent       chan struct{}
	openStartedOnce  sync.Once
	openDoneOnce     sync.Once
	firstSendOnce    sync.Once
	firstPartialOnce sync.Once
	finalOnce        sync.Once
	hints            []string
	prompt           string // the last stream's prompt
	// liveResult is the last stream's StreamRequest.LiveResult.
	liveResult            bool
	earlyPieces           bool
	supportsEarlyPieces   bool
	closeLeavesEventsOpen bool
}

func (f *fakeStreamer) ensureSignals() {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.openStarted == nil {
		f.openStarted = make(chan struct{})
	}
	if f.openDone == nil {
		f.openDone = make(chan struct{})
	}
	if f.firstSend == nil {
		f.firstSend = make(chan struct{})
	}
	if f.firstPartial == nil {
		f.firstPartial = make(chan struct{})
	}
	if f.finalEvent == nil {
		f.finalEvent = make(chan struct{})
	}
}

func (f *fakeStreamer) Caps() asr.Caps { return asr.Caps{Stream: true, Hints: true} }

func (f *fakeStreamer) Open(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	f.ensureSignals()
	f.openStartedOnce.Do(func() { close(f.openStarted) })
	defer f.openDoneOnce.Do(func() { close(f.openDone) })
	if f.openWait > 0 {
		select {
		case <-time.After(f.openWait):
		case <-ctx.Done():
			return nil, &asr.Error{Kind: asr.KindCancelled}
		}
	}
	if f.openGate != nil {
		select {
		case <-f.openGate:
		case <-ctx.Done():
			return nil, &asr.Error{Kind: asr.KindCancelled}
		}
	}
	if f.openErr != nil {
		return nil, f.openErr
	}
	s := &fakeStream{
		f: f, ctx: ctx, events: make(chan asr.Event, 1024),
		latePartials: req.LatePartials, candidates: f.candidates,
		partialGate: f.partialGate, partialReturnGate: f.partialReturnGate,
		salvageRequiresOpen: f.salvageRequiresOpen,
	}
	f.mu.Lock()
	f.streams = append(f.streams, s)
	f.hints, f.prompt, f.liveResult, f.earlyPieces = req.Hints, req.Prompt, req.LiveResult, req.EarlyPieces
	f.mu.Unlock()
	return s, nil
}

func (f *fakeStreamer) latePartialsRequested() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.streams) > 0 && f.streams[len(f.streams)-1].latePartials
}

type fakeStream struct {
	f                   *fakeStreamer
	ctx                 context.Context
	mu                  sync.Mutex
	bytes               int
	data                []byte // what was sent
	sends               int
	closed              bool
	done                bool
	earlyPieces         bool
	latePartials        bool
	candidates          []string
	partialGate         <-chan struct{}
	partialReturnGate   <-chan struct{}
	salvageRequiresOpen bool
	lifecycle           []string
	events              chan asr.Event
}

func (s *fakeStream) emit(ev asr.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.done {
		return
	}
	if ev.Kind == asr.EventFinal || ev.Kind == asr.EventError {
		s.done = true
	}
	s.events <- ev
	if ev.Kind == asr.EventPartial {
		s.f.firstPartialOnce.Do(func() { close(s.f.firstPartial) })
	}
	if s.done {
		s.f.finalOnce.Do(func() { close(s.f.finalEvent) })
		close(s.events)
	}
}

func (s *fakeStream) Send(pcm []byte) error {
	s.mu.Lock()
	if s.done || s.closed {
		s.mu.Unlock()
		return &asr.Error{Kind: asr.KindProtocol, Detail: "the stream has ended"}
	}
	s.bytes += len(pcm)
	s.data = append(s.data, pcm...)
	s.sends++
	n := s.sends
	s.mu.Unlock()
	s.f.firstSendOnce.Do(func() { close(s.f.firstSend) })
	if s.f.errAfter > 0 && n >= s.f.errAfter {
		s.emit(asr.Event{Kind: asr.EventError, Err: &asr.Error{Kind: asr.KindNetwork, Detail: "connection reset"}})
		return nil
	}
	if !s.f.silent {
		if n == 1 && s.partialGate != nil {
			select {
			case <-s.partialGate:
			case <-s.ctx.Done():
				return &asr.Error{Kind: asr.KindCancelled}
			}
		}
		text := "partial"
		if len(s.f.partials) > 0 {
			text = s.f.partials[min(n, len(s.f.partials))-1]
		}
		s.emit(asr.Event{Kind: asr.EventPartial, Text: text})
		if n == 1 && s.partialReturnGate != nil {
			select {
			case <-s.partialReturnGate:
			case <-s.ctx.Done():
				return &asr.Error{Kind: asr.KindCancelled}
			}
		}
	}
	if n <= len(s.f.pieces) && s.f.pieces[n-1] != "" {
		s.emit(asr.Event{Kind: asr.EventPiece, Text: s.f.pieces[n-1]})
	}
	if s.f.onSend != nil {
		s.f.onSend(n)
	}
	return nil
}

func (s *fakeStream) Salvage() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lifecycle = append(s.lifecycle, "salvage")
	if s.salvageRequiresOpen && s.closed {
		return ""
	}
	return s.f.salvage
}

func (s *fakeStream) Finish() error {
	defer func() {
		if s.f.finishEmitted != nil {
			s.mu.Lock()
			emitted := s.done
			s.mu.Unlock()
			select {
			case s.f.finishEmitted <- emitted:
			default:
			}
		}
	}()
	if s.f.finishStarted != nil {
		select {
		case s.f.finishStarted <- struct{}{}:
		default:
		}
	}
	if s.latePartials {
		for _, text := range s.candidates {
			s.emit(asr.Event{Kind: asr.EventCandidate, Text: text})
		}
	}
	if s.f.finishGate != nil {
		select {
		case <-s.f.finishGate:
		case <-s.ctx.Done():
			return &asr.Error{Kind: asr.KindCancelled, Detail: s.ctx.Err().Error()}
		}
	}
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
		s.lifecycle = append(s.lifecycle, "close")
		if !s.done && !s.f.closeLeavesEventsOpen {
			close(s.events)
		}
	}
}

func (s *fakeStream) Events() <-chan asr.Event { return s.events }

func (s *fakeStream) EnableEarlyPieces() bool {
	s.f.mu.Lock()
	supported := s.f.supportsEarlyPieces
	s.f.mu.Unlock()
	if !supported {
		return false
	}
	s.mu.Lock()
	s.earlyPieces = true
	s.mu.Unlock()
	return true
}
