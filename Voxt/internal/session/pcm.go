package session

import (
	"context"
	"sync"
)

// pcmBuffer is a session's recording: PCM appended as the helper sends
// it, read by the stream feeder as it grows, and whole at the end.
type pcmBuffer struct {
	mu     sync.Mutex
	data   []byte
	closed bool
	signal chan struct{} // closed and replaced at each change
}

func newPCMBuffer() *pcmBuffer { return &pcmBuffer{signal: make(chan struct{})} }

func (b *pcmBuffer) append(p []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed || len(p) == 0 {
		return
	}
	b.data = append(b.data, p...)
	close(b.signal)
	b.signal = make(chan struct{})
}

// close ends the recording: nothing more is appended.
func (b *pcmBuffer) close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.closed {
		b.closed = true
		close(b.signal)
	}
}

// bytes is the recording so far (shared: do not modify).
func (b *pcmBuffer) bytes() []byte {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.data[:len(b.data):len(b.data)]
}

// next waits until there is audio after off, or the end, and returns it
// with whether the recording ended (then the audio is all there is).
func (b *pcmBuffer) next(ctx context.Context, off int) ([]byte, bool, error) {
	for {
		b.mu.Lock()
		data, closed, sig := b.data, b.closed, b.signal
		b.mu.Unlock()
		if off < len(data) || closed {
			if off > len(data) {
				off = len(data)
			}
			return data[off:len(data):len(data)], closed, nil
		}
		select {
		case <-sig:
		case <-ctx.Done():
			return nil, false, ctx.Err()
		}
	}
}
