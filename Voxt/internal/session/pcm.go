package session

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

const (
	pcmSpillAt    = 4 << 20
	pcmSpillBatch = 256 << 10
)

type pcmSpool interface {
	io.ReaderAt
	io.WriterAt
	io.Closer
}

// pcmBuffer keeps a session's original PCM. Short recordings stay in
// memory. A writer moves longer recordings' prefixes to disk while the
// capture callback appends and the ASR feeder reads.
type pcmBuffer struct {
	mu        sync.Mutex
	data      []byte   // suffix not yet on disk
	file      pcmSpool // immutable prefix, once published
	stored    int      // bytes in the prefix
	closed    bool     // no more appends
	discarded bool     // the session and history writer are finished
	signal    chan struct{}

	spillDone chan struct{}
	spillErr  error
	openSpool func() (pcmSpool, error)
}

func newPCMBuffer() *pcmBuffer {
	return &pcmBuffer{signal: make(chan struct{}), openSpool: openPCMSpool}
}

func (b *pcmBuffer) append(p []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed || len(p) == 0 {
		return
	}
	b.data = append(b.data, p...)
	b.wake()
	if b.spillDone == nil && b.spillErr == nil && len(b.data) >= b.spillThreshold() {
		b.spillDone = make(chan struct{})
		go b.spill()
	}
}

// wake notifies a feeder waiting for more data. The caller holds mu.
func (b *pcmBuffer) wake() {
	close(b.signal)
	b.signal = make(chan struct{})
}

func (b *pcmBuffer) spillThreshold() int {
	if b.file == nil {
		return pcmSpillAt
	}
	return pcmSpillBatch
}

// spill never holds mu during I/O. Until a write succeeds, all of that
// batch is readable in memory. A failed write leaves the complete suffix
// in memory and disables future writes for this recording.
func (b *pcmBuffer) spill() {
	b.mu.Lock()
	for !b.discarded && len(b.data) >= b.spillThreshold() {
		file, stored, batch := b.file, b.stored, b.data
		b.mu.Unlock()

		newFile := file == nil
		var err error
		if newFile {
			file, err = b.openSpool()
		}
		if err == nil {
			var n int
			n, err = file.WriteAt(batch, int64(stored))
			if err == nil && n != len(batch) {
				err = io.ErrShortWrite
			}
		}

		b.mu.Lock()
		if err != nil {
			b.spillErr = err
		}
		if err != nil || b.discarded {
			if newFile && file != nil {
				b.mu.Unlock()
				closeErr := file.Close()
				b.mu.Lock()
				b.spillErr = errors.Join(b.spillErr, closeErr)
			}
			break
		}
		b.file, b.stored = file, stored+len(batch)
		// Never overwrite memory that a feeder or reader may still hold.
		b.data = append([]byte(nil), b.data[len(batch):]...)
	}
	close(b.spillDone)
	b.spillDone = nil
	b.mu.Unlock()
}

func openPCMSpool() (pcmSpool, error) {
	f, err := os.CreateTemp("", "voxt-pcm-*")
	if err != nil {
		return nil, err
	}
	// Unlink before storing audio. The open handle survives until Close,
	// including on macOS and Linux. A crash leaves no recording behind.
	if err := os.Remove(f.Name()); err != nil {
		return nil, errors.Join(err, f.Close())
	}
	return f, nil
}

// close ends capture. The buffered audio can still be read and saved.
func (b *pcmBuffer) close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.closed {
		b.closed = true
		b.wake()
	}
}

func (b *pcmBuffer) discard() error {
	b.mu.Lock()
	b.closed, b.discarded = true, true
	done := b.spillDone
	b.wake()
	b.mu.Unlock()
	if done != nil {
		<-done
	}
	b.mu.Lock()
	file, err := b.file, b.spillErr
	b.file, b.data, b.stored = nil, nil, 0
	b.mu.Unlock()
	if file != nil {
		return errors.Join(err, file.Close())
	}
	return err
}

func (b *pcmBuffer) size() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.stored + len(b.data)
}

func (b *pcmBuffer) snapshot() (pcmSpool, int, []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.file, b.stored, b.data[:len(b.data):len(b.data)]
}

// bytes materializes the recording for providers that require an audio
// slice. Streaming providers and history saving read it incrementally.
func (b *pcmBuffer) bytes() ([]byte, error) {
	file, stored, tail := b.snapshot()
	if file == nil {
		return tail, nil
	}
	data := make([]byte, stored+len(tail))
	if _, err := file.ReadAt(data[:stored], 0); err != nil {
		return nil, err
	}
	copy(data[stored:], tail)
	return data, nil
}

func (b *pcmBuffer) reader() (io.Reader, int) {
	file, stored, tail := b.snapshot()
	if file == nil {
		return bytes.NewReader(tail), len(tail)
	}
	return io.MultiReader(io.NewSectionReader(file, 0, int64(stored)), bytes.NewReader(tail)), stored + len(tail)
}

// next waits for audio after off or the end of capture. The returned
// bytes may be shared and must not be modified.
func (b *pcmBuffer) next(ctx context.Context, off int) ([]byte, bool, error) {
	for {
		if err := ctx.Err(); err != nil {
			return nil, false, err
		}
		b.mu.Lock()
		file, stored, data, closed, sig := b.file, b.stored, b.data, b.closed, b.signal
		discarded := b.discarded
		b.mu.Unlock()
		if discarded {
			return nil, false, io.ErrClosedPipe
		}
		if file != nil && off < stored {
			chunk := make([]byte, min(pcmSpillBatch, stored-off))
			_, err := file.ReadAt(chunk, int64(off))
			return chunk, closed, err
		}
		if at := off - stored; at >= 0 && at < len(data) {
			return data[at:len(data):len(data)], closed, nil
		}
		if closed {
			return nil, true, nil
		}
		select {
		case <-sig:
		case <-ctx.Done():
			return nil, false, ctx.Err()
		}
	}
}

func (b *pcmBuffer) saveWAV(dir, name string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".recording-*.wav")
	if err != nil {
		return err
	}
	pcm, size := b.reader()
	header := asr.WAVHeader(size, asr.SampleRate)
	_, writeErr := f.Write(header[:])
	if writeErr == nil {
		_, writeErr = io.CopyN(f, pcm, int64(size))
	}
	if err = errors.Join(writeErr, f.Close()); err == nil {
		err = os.Rename(f.Name(), filepath.Join(dir, name))
	}
	if err != nil {
		return errors.Join(err, os.Remove(f.Name()))
	}
	return nil
}

func (r *run) discardPCM() {
	if err := r.pcm.discard(); err != nil {
		r.logf("warning", "spooling the recording: %v", err)
	}
}
