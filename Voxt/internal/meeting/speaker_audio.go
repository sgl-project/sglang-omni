package meeting

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/speaker"
)

// Loading Core ML can take seconds on first use. Even then the speaker
// queue holds at most 30 seconds (960 KB), independently of transcription.
const speakerQueueBytes = 30 * asr.BytesPerSecond

type voicePacket struct {
	at      time.Duration
	pcm     []byte
	barrier chan struct{}
}

type voiceAnchor struct {
	input, end, at time.Duration
	padding        bool
}

type voiceRun struct {
	ctx     context.Context
	cancel  context.CancelFunc
	queue   chan voicePacket
	done    chan struct{}
	bytes   atomic.Int64
	stop    sync.Once
	mu      sync.Mutex
	closed  bool
	model   speaker.Model
	update  func([]speaker.Frame)
	warn    func(string)
	anchors []voiceAnchor // worker-owned; bounded by the model's short lookahead
	input   time.Duration
}

func newVoiceRun(model speaker.Model, update func([]speaker.Frame), warn func(string)) *voiceRun {
	ctx, cancel := context.WithCancel(context.Background())
	r := &voiceRun{ctx: ctx, cancel: cancel, queue: make(chan voicePacket, 600), done: make(chan struct{}), model: model, update: update, warn: warn}
	go r.work()
	return r
}

// offer never calls a model, waits on inference, or takes Manager.mu. A
// stalled classifier can lose labels, but cannot delay words or capture.
func (r *voiceRun) offer(at time.Duration, pcm []byte) {
	for len(pcm) > 0 {
		n := min(len(pcm), asr.BytesPerSecond/5)
		r.mu.Lock()
		if r.closed || r.ctx.Err() != nil {
			r.mu.Unlock()
			return
		}
		if r.bytes.Add(int64(n)) > speakerQueueBytes {
			r.bytes.Add(-int64(n))
			r.mu.Unlock()
			r.fail(errors.New("speaker queue limit reached; transcription continues"))
			return
		}
		packet := voicePacket{at: at, pcm: append([]byte(nil), pcm[:n]...)}
		select {
		case r.queue <- packet:
			r.mu.Unlock()
		default:
			r.bytes.Add(-int64(n))
			r.mu.Unlock()
			r.fail(errors.New("speaker packet limit reached; transcription continues"))
			return
		}
		at += pcmDuration(pcm[:n])
		pcm = pcm[n:]
	}
}

// drain advances the right context with silence without resetting voice
// identities. Pausing/resuming does not recreate or renumber the model.
func (r *voiceRun) drain(ctx context.Context) {
	barrier := make(chan struct{})
	select {
	case r.queue <- voicePacket{barrier: barrier}:
	case <-ctx.Done():
		return
	case <-r.done:
		return
	}
	select {
	case <-barrier:
	case <-ctx.Done():
	case <-r.done:
	}
}

func (r *voiceRun) close() { r.cancel(); <-r.done }

func (r *voiceRun) fail(err error) {
	r.stop.Do(func() {
		r.cancel()
		if r.warn != nil {
			if !errors.Is(err, context.Canceled) {
				r.warn("audio speaker identification stopped: " + err.Error())
			}
		}
	})
}

func (r *voiceRun) work() {
	defer close(r.done)
	defer func() {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.closed = true
		// Release queued audio after failure; the meeting keeps the run so
		// capture cannot accidentally restart a new, renumbered classifier.
		for {
			select {
			case <-r.queue:
			default:
				r.bytes.Store(0)
				return
			}
		}
	}()
	stream, err := r.model.Open(r.ctx)
	if err != nil {
		r.fail(err)
		return
	}
	defer stream.Close()
	for {
		select {
		case <-r.ctx.Done():
			return
		case p := <-r.queue:
			if p.barrier != nil {
				// 1.2 s exceeds the classifier's 1.04 s right context. Padding is
				// not mapped onto meeting time or offered to speech recognition.
				for i := 0; i < 6; i++ {
					if err := r.push(stream, voicePacket{pcm: make([]byte, asr.BytesPerSecond/5)}, true); err != nil {
						r.fail(err)
						return
					}
				}
				close(p.barrier)
				continue
			}
			r.bytes.Add(-int64(len(p.pcm)))
			if err := r.push(stream, p, false); err != nil {
				r.fail(err)
				return
			}
		}
	}
}

func (r *voiceRun) push(stream speaker.Stream, p voicePacket, padding bool) error {
	end := r.input + pcmDuration(p.pcm)
	r.anchors = append(r.anchors, voiceAnchor{input: r.input, end: end, at: p.at, padding: padding})
	r.input = end
	frames, err := stream.Push(r.ctx, p.pcm)
	if err != nil {
		return err
	}
	var mapped []speaker.Frame
	for _, f := range frames {
		for _, a := range r.anchors {
			start, end := max(f.Start, a.input), min(f.End, a.end)
			if a.padding || end <= start {
				continue
			}
			mapped = append(mapped, speaker.Frame{Start: a.at + start - a.input, End: a.at + end - a.input, Scores: f.Scores})
		}
	}
	if len(mapped) > 0 {
		r.update(mapped)
	}
	// Keep a generous context tail, not a meeting-long list of PCM anchors.
	for len(r.anchors) > 0 && r.anchors[0].end < r.input-30*time.Second {
		r.anchors = r.anchors[1:]
	}
	return nil
}
