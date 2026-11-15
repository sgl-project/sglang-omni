package meeting

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// Archive writes per-source WAV files under a meeting folder.
type Archive struct {
	dir string
	mu  sync.Mutex
	w   map[Source]*wavWriter
}

func newArchive(dir string) *Archive {
	return &Archive{dir: dir, w: map[Source]*wavWriter{}}
}

func (a *Archive) path(src Source) string {
	return filepath.Join(a.dir, string(src)+".wav")
}

// Write appends PCM16 mono frames to src's WAV.
func (a *Archive) Write(src Source, pcm []byte) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	w, ok := a.w[src]
	if !ok {
		if err := os.MkdirAll(a.dir, 0o755); err != nil {
			return err
		}
		f, err := os.OpenFile(a.path(src), os.O_CREATE|os.O_RDWR, 0o644)
		if err != nil {
			return err
		}
		w = &wavWriter{f: f}
		if err := w.writeHeader(); err != nil {
			return errors.Join(err, f.Close())
		}
		a.w[src] = w
	}
	if _, err := w.f.Seek(0, io.SeekEnd); err != nil {
		return err
	}
	_, err := w.f.Write(pcm)
	return err
}

// Close finalises headers and closes files.
func (a *Archive) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	var first error
	for _, w := range a.w {
		if err := w.finalize(); err != nil && first == nil {
			first = err
		}
		w.f.Close()
	}
	a.w = map[Source]*wavWriter{}
	return first
}

type wavWriter struct {
	f *os.File
}

func (w *wavWriter) writeHeader() error {
	// Placeholder RIFF header; data length is patched on finalize.
	hdr := make([]byte, 44)
	copy(hdr[0:], "RIFF")
	copy(hdr[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(hdr[16:], 16)
	binary.LittleEndian.PutUint16(hdr[20:], 1)
	binary.LittleEndian.PutUint16(hdr[22:], 1)
	binary.LittleEndian.PutUint32(hdr[24:], asr.SampleRate)
	binary.LittleEndian.PutUint32(hdr[28:], asr.BytesPerSecond)
	binary.LittleEndian.PutUint16(hdr[32:], 2)
	binary.LittleEndian.PutUint16(hdr[34:], 16)
	copy(hdr[36:], "data")
	_, err := w.f.WriteAt(hdr, 0)
	return err
}

func (w *wavWriter) finalize() error {
	info, err := w.f.Stat()
	if err != nil {
		return err
	}
	dataBytes := info.Size() - 44
	if dataBytes < 0 {
		dataBytes = 0
	}
	riffSize := uint32(36 + dataBytes)
	var word [4]byte
	binary.LittleEndian.PutUint32(word[:], riffSize)
	if _, err := w.f.WriteAt(word[:], 4); err != nil {
		return err
	}
	binary.LittleEndian.PutUint32(word[:], uint32(dataBytes))
	_, err = w.f.WriteAt(word[:], 40)
	return err
}

// FixWAVHeader patches a WAV data chunk length from the file size.
func FixWAVHeader(path string) error {
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		return err
	}
	info, err := f.Stat()
	if err != nil {
		return errors.Join(err, f.Close())
	}
	if info.Size() < 44 {
		return errors.Join(fmt.Errorf("meeting: wav %s too short", path), f.Close())
	}
	return errors.Join((&wavWriter{f: f}).finalize(), f.Close())
}
