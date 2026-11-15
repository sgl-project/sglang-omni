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
	// lang is the language the last Final heard, for the next stream when
	// Req has no hint: its live text then stays in the meeting's language (#3981).
	lang string
	// reopen: a rotation's Open failed; the next Push tries again.
	reopen bool
}

// Open starts the first recognition stream.
func (l *Live) Open(ctx context.Context) error {
	if l.Clock == nil {
		l.Clock = time.Now
	}
	if l.Speech == nil {
		l.Speech = NewSpeech(l.sampleRate(), nil)
	}
	if err := l.open(ctx); err != nil {
		return err
	}
	l.committer.BeginPhrase(0)
	return nil
}

// open opens a stream with Req, its language hint else the one the last
// Final heard.
func (l *Live) open(ctx context.Context) error {
	req := l.Req
	if req.Lang == "" {
		req.Lang = l.lang
	}
	st, err := l.Streamer.Open(ctx, req)
	l.reopen = err != nil
	if err != nil {
		return err
	}
	l.stream, l.streamOpen = st, l.Clock()
	l.gotPartial, l.lastPartial, l.rotateDue = false, time.Time{}, false
	return nil
}

// Push sends PCM, drains ASR events, and may commit segments or rotate the stream.
func (l *Live) Push(ctx context.Context, pcm []byte) error {
	if l.stream == nil {
		if !l.reopen {
			return nil
		}
		if err := l.open(ctx); err != nil {
			return err
		}
	}
	if err := l.stream.Send(pcm); err != nil {
		return err
	}
	l.timeline += pcmDuration(pcm)
	l.drainEvents()
	speaking, silence := l.Speech.Push(pcm)
	if tail := l.committer.Live(); collapseLoops(tail) != tail {
		// The model loops (#3981): commit the text once and restart it, so the
		// loop does not run on into the next segments.
		if seg, ok := l.committer.Tick(l.timeline, silenceFinalizeLive); ok {
			l.emit(seg)
		}
		return l.rotate(ctx, true, false)
	}
	if l.stalePartial(speaking) {
		return l.rotate(ctx, true, true)
	}
	if l.Clock().Sub(l.streamOpen) >= maxStreamLiveAge {
		l.rotateDue = true
	}
	seg, ok := l.committer.Tick(l.timeline, silence)
	if ok {
		l.emit(seg)
	}
	if l.rotateDue {
		return l.rotate(ctx, false, true)
	}
	return nil
}

// Finish ends the live track: Finish the stream and commit any tail.
func (l *Live) Finish(ctx context.Context) error {
	if l.stream == nil {
		return nil
	}
	err := l.finishStream(l.stream, true)
	l.stream = nil
	return err
}

// finishStream ends st; commit false keeps its Final's text out (a looping
// stream's Final repeats what was committed), but not its language.
func (l *Live) finishStream(st asr.Stream, commit bool) error {
	_ = st.Finish()
	for ev := range st.Events() {
		switch ev.Kind {
		case asr.EventPartial:
			l.committer.Partial(ev.Text)
		case asr.EventFinal:
			if ev.Result.Lang != "" {
				l.lang = ev.Result.Lang
			}
			if !commit {
				continue
			}
			if seg, ok := l.committer.Final(ev.Text, l.timeline); ok {
				l.emit(seg)
			}
		case asr.EventError:
			return ev.Err
		}
	}
	return nil
}

// rotate finishes the stream, then opens the next one with the language
// its Final heard (Push is blocked meanwhile: no audio is lost; a failed
// Open is retried at the next Push).
func (l *Live) rotate(ctx context.Context, immediate, commitFinal bool) error {
	old := l.stream
	if old == nil {
		return nil
	}
	ferr := l.finishStream(old, commitFinal)
	l.stream = nil
	// The new stream's text starts from scratch; none of it is committed yet.
	l.committer.committed, l.committer.cumulative = "", ""
	if err := l.open(ctx); err != nil {
		return err
	}
	if ferr != nil && !immediate {
		return ferr
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
