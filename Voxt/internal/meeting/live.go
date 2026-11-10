package meeting

import (
	"context"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

const (
	maxStreamLiveAge  = 10 * time.Minute
	stalePartialLive  = 30 * time.Second
	defaultLiveRateHz = 16000
)

// Live runs one meeting track's streaming ASR with Committer and stream rotation (design C1 §4.7).
type Live struct {
	Clock    func() time.Time
	Streamer asr.Streamer
	Req      asr.StreamRequest
	Speech   *Speech
	Source   Source
	OnCommit func(Segment)

	committer   Committer
	stream      asr.Stream
	streamOpen  time.Time
	timeline    time.Duration
	lastPartial time.Time
	gotPartial  bool
	rotateDue   bool
}

// Open starts the first recognition stream.
func (l *Live) Open(ctx context.Context) error {
	if l.Clock == nil {
		l.Clock = time.Now
	}
	if l.Speech == nil {
		l.Speech = NewSpeech(l.sampleRate(), nil)
	}
	st, err := l.Streamer.Open(ctx, l.Req)
	if err != nil {
		return err
	}
	l.stream = st
	l.streamOpen = l.Clock()
	l.committer.BeginPhrase(0)
	return nil
}

// Push sends PCM, drains ASR events, and may commit segments or rotate the stream.
func (l *Live) Push(ctx context.Context, pcm []byte) error {
	if l.stream == nil {
		return nil
	}
	if err := l.stream.Send(pcm); err != nil {
		return err
	}
	l.timeline += pcmDuration(pcm)
	l.drainEvents()
	speaking, silence := l.Speech.Push(pcm)
	if l.stalePartial(speaking) {
		return l.rotate(ctx, true)
	}
	if l.Clock().Sub(l.streamOpen) >= maxStreamLiveAge {
		l.rotateDue = true
	}
	seg, ok := l.committer.Tick(l.timeline, silence)
	if ok {
		l.emit(seg)
	}
	if l.rotateDue {
		return l.rotate(ctx, false)
	}
	return nil
}

// Finish ends the live track: Finish the stream and commit any tail.
func (l *Live) Finish(ctx context.Context) error {
	if l.stream == nil {
		return nil
	}
	err := l.finishStream(l.stream)
	l.stream = nil
	return err
}

func (l *Live) finishStream(st asr.Stream) error {
	_ = st.Finish()
	for ev := range st.Events() {
		switch ev.Kind {
		case asr.EventPartial:
			l.committer.Partial(ev.Text)
		case asr.EventFinal:
			if seg, ok := l.committer.Final(ev.Text, l.timeline); ok {
				l.emit(seg)
			}
		case asr.EventError:
			return ev.Err
		}
	}
	return nil
}

func (l *Live) rotate(ctx context.Context, immediate bool) error {
	old := l.stream
	if old == nil {
		return nil
	}
	next, err := l.Streamer.Open(ctx, l.Req)
	if err != nil {
		return err
	}
	l.stream = next
	l.streamOpen = l.Clock()
	l.gotPartial = false
	l.lastPartial = time.Time{}
	l.rotateDue = false
	if err := l.finishStream(old); err != nil && !immediate {
		return err
	}
	return nil
}

func (l *Live) drainEvents() {
	for {
		select {
		case ev, ok := <-l.stream.Events():
			if !ok {
				return
			}
			switch ev.Kind {
			case asr.EventPartial:
				l.committer.Partial(ev.Text)
				l.lastPartial = l.Clock()
				l.gotPartial = true
			case asr.EventFinal, asr.EventError:
				return
			}
		default:
			return
		}
	}
}

func (l *Live) stalePartial(speaking bool) bool {
	if !speaking {
		return false
	}
	if l.gotPartial && l.Clock().Sub(l.lastPartial) < stalePartialLive {
		return false
	}
	if !l.gotPartial && l.Clock().Sub(l.streamOpen) < stalePartialLive {
		return false
	}
	return true
}

func (l *Live) emit(seg Segment) {
	seg.Source = l.Source
	if l.OnCommit != nil {
		l.OnCommit(seg)
	}
}

func (l *Live) LiveText() string {
	return l.committer.Live()
}

func (l *Live) sampleRate() int {
	if l.Req.SampleRate > 0 {
		return l.Req.SampleRate
	}
	return defaultLiveRateHz
}
