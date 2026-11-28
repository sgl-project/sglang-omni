package meeting

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// Archive writes per-source WAV files under a meeting folder. Frames with
// a capture time (PCMParams.HostMS) are placed by it, so the same
// position in every file is the same moment: the first timed frame of a
// period (the meeting's start, each resume) sets the origin (what another
// track captured before it is cut), a gap is padded with silence and an
// overlap is cut, and Close pads the files to one length (a track the
// meeting stopped, by a change of mode, goes on in silence).
type Archive struct {
	dir string
	mu  sync.Mutex
	w   map[Source]*wavWriter
	// origin is the host time (ms) at sample base of every file; 0 until
	// the period's first timed frame.
	origin float64
	base   int64
	timed  bool // a frame had a capture time
}

// archiveSlack is the misplacement left alone (capture-time jitter,
// clock drift): 20 ms.
const archiveSlack = asr.SampleRate / 50

// archiveMaxGap is how far a frame may land past the longest file, or
// before where its own file ends: further is a clock that went wrong, and
// the frame is appended as it comes. Past the longest file, not past its
// own: a track that starts late (Recording switched to Meeting after an
// hour) still lines up.
var archiveMaxGap int64 = 30 * 60 * asr.SampleRate

func newArchive(dir string) *Archive {
	return &Archive{dir: dir, w: map[Source]*wavWriter{}}
}

func (a *Archive) path(src Source) string {
	return filepath.Join(a.dir, string(src)+".wav")
}

// Write appends PCM16 mono frames to src's WAV.
func (a *Archive) Write(src Source, pcm []byte) error { return a.WriteAt(src, pcm, 0) }

// WriteAt writes PCM16 mono frames to src's WAV where hostMS (the
// capture time of their first sample) places them; 0 appends them.
func (a *Archive) WriteAt(src Source, pcm []byte, hostMS float64) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	w, err := a.writer(src)
	if err != nil {
		return err
	}
	if hostMS > 0 {
		a.timed = true
		if a.origin == 0 {
			a.origin = hostMS
		}
		want := a.base + int64(math.Round((hostMS-a.origin)*asr.SampleRate/1000))
		// Else a clock gone wrong: appended as it comes.
		if want-w.samples >= -archiveMaxGap && want-a.longest() <= archiveMaxGap {
			if want < a.base {
				// Before the period's first frame (the other track's): that
				// part is cut, so a period never runs into the one before.
				skip := a.base - want
				if skip >= int64(len(pcm)/2) {
					return nil
				}
				pcm, want = pcm[2*skip:], a.base
			}
			switch d := want - w.samples; {
			case d > archiveSlack:
				if err := w.silence(d); err != nil {
					return err
				}
			case d < -archiveSlack:
				pcm = pcm[2*min(-d, int64(len(pcm)/2)):]
			}
		}
	}
	return w.append(pcm)
}

// NewPeriod starts a period after a pause: every file is padded to the
// longest one, where the period's first timed frame lands. A file that
// cannot be padded still gets the new period (its frames are cut or
// padded from where it ends).
func (a *Archive) NewPeriod() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	var first error
	if a.timed {
		first = a.padLocked()
	}
	a.base, a.origin = a.longest(), 0
	return first
}

// padLocked pads every file to the longest one.
func (a *Archive) padLocked() error {
	var first error
	longest := a.longest()
	for _, w := range a.w {
		if err := w.silence(longest - w.samples); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// longest is the end of the longest file, in samples (the period's base
// when none is longer).
func (a *Archive) longest() int64 {
	n := a.base
	for _, w := range a.w {
		n = max(n, w.samples)
	}
	return n
}

func (a *Archive) writer(src Source) (*wavWriter, error) {
	if w, ok := a.w[src]; ok {
		return w, nil
	}
	if err := os.MkdirAll(a.dir, 0o755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(a.path(src), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	w := &wavWriter{f: f}
	if err := w.writeHeader(); err != nil {
		return nil, errors.Join(err, f.Close())
	}
	info, err := f.Stat()
	if err != nil {
		return nil, errors.Join(err, f.Close())
	}
	w.samples = max(info.Size()-44, 0) / 2
	a.w[src] = w
	return w, nil
}

// Close finalises headers and closes files.
func (a *Archive) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	var first error
	if a.timed {
		first = a.padLocked()
	}
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
	f       *os.File
	samples int64 // in the data chunk
}

func (w *wavWriter) append(pcm []byte) error {
	if len(pcm) == 0 {
		return nil
	}
	if _, err := w.f.Seek(0, io.SeekEnd); err != nil {
		return err
	}
	n, err := w.f.Write(pcm)
	w.samples += int64(n / 2)
	return err
}

// silence appends n samples of silence (none for n ≤ 0), in blocks of 16 s:
// a track that joins an hour in pads an hour.
func (w *wavWriter) silence(n int64) error {
	if n <= 0 {
		return nil
	}
	if _, err := w.f.Seek(0, io.SeekEnd); err != nil {
		return err
	}
	zeros := make([]byte, 2*min(n, 16*asr.SampleRate))
	for n > 0 {
		k := min(n, int64(len(zeros)/2))
		written, err := w.f.Write(zeros[:2*k])
		w.samples += int64(written / 2)
		if err != nil {
			return err
		}
		n -= k
	}
	return nil
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
