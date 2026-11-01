package asr

import (
	"context"
	"time"
)

// Feed is how StreamAll sends a recording.
type Feed struct {
	// Frame is the size of each Send (0: 100 ms, as the helper sends).
	Frame time.Duration
	// Pace sends at this many times real time (1: as fast as it was
	// spoken; 0: as fast as the stream takes it). Pace 1 makes a live
	// stream see the timing of a real recording.
	Pace float64
	// OnPartial, if set, gets each Partial's text.
	OnPartial func(text string)
}

// StreamAll recognises a whole recording (PCM16 mono at req.SampleRate)
// through a stream: Open, Send it frame by frame, Finish, and wait for
// the Final (at most FinalTimeout after Finish). Tests use it; sessions
// feed a stream as they record.
func StreamAll(ctx context.Context, s Streamer, req StreamRequest, pcm []byte, f Feed) (Result, error) {
	st, err := s.Open(ctx, req)
	if err != nil {
		return Result{}, err
	}
	defer st.Close()
	frame := f.Frame
	if frame <= 0 {
		frame = 100 * time.Millisecond
	}
	sendErr := make(chan error, 1)
	go func() {
		var next time.Time
		for _, fr := range Frames(pcm, frame, req.SampleRate) {
			if f.Pace > 0 {
				if next.IsZero() {
					next = time.Now()
				}
				next = next.Add(time.Duration(float64(Duration(fr, req.SampleRate)) / f.Pace))
				if d := time.Until(next); d > 0 {
					select {
					case <-time.After(d):
					case <-ctx.Done():
						sendErr <- ctx.Err()
						return
					}
				}
			}
			if err := st.Send(fr); err != nil {
				sendErr <- err
				return
			}
		}
		sendErr <- st.Finish()
	}()
	var finished <-chan time.Time
	for {
		select {
		case ev, ok := <-st.Events():
			if !ok {
				return Result{}, &Error{Kind: KindProtocol, Detail: "the stream ended without a final result"}
			}
			switch ev.Kind {
			case EventPartial:
				if f.OnPartial != nil {
					f.OnPartial(ev.Text)
				}
			case EventFinal:
				r := ev.Result
				r.Text = ev.Text
				return r, nil
			case EventError:
				return Result{}, ev.Err
			}
		case err := <-sendErr:
			if err != nil {
				// The reader usually has the real reason; give it a moment.
				select {
				case ev, ok := <-st.Events():
					if ok && ev.Kind == EventError {
						return Result{}, ev.Err
					}
				case <-time.After(500 * time.Millisecond):
				}
				return Result{}, NetError(ctx, "", err)
			}
			finished = time.After(FinalTimeout)
		case <-finished:
			return Result{}, &Error{Kind: KindNetwork, Detail: "no final result in time", Retryable: true}
		case <-ctx.Done():
			return Result{}, NetError(ctx, "", ctx.Err())
		}
	}
}

// Recognize transcribes a whole recording with p, in one request when p
// is a FileTranscriber and streaming it otherwise.
func Recognize(ctx context.Context, p Provider, pcm []byte, lang string, hints []string, prompt string, f Feed) (Result, error) {
	if ft, ok := p.(FileTranscriber); ok && p.Caps().File {
		return ft.Transcribe(ctx, FileRequest{Audio: pcm, Format: "pcm", SampleRate: SampleRate, Lang: lang, Hints: hints, Prompt: prompt})
	}
	if st, ok := p.(Streamer); ok {
		return StreamAll(ctx, st, StreamRequest{SampleRate: SampleRate, Lang: lang, Hints: hints, Prompt: prompt}, pcm, f)
	}
	return Result{}, &Error{Kind: KindConfig, Detail: "the speech model neither takes a whole recording nor streams"}
}
