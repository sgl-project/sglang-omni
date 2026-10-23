package asr

import (
	"bytes"
	"encoding/binary"
	"errors"
	"time"
)

// BytesPerSecond is the size of a second of VoxtNew's PCM.
const BytesPerSecond = SampleRate * 2

// Duration is how long pcm (PCM16 mono at rate Hz; 0: SampleRate) plays.
func Duration(pcm []byte, rate int) time.Duration {
	if rate <= 0 {
		rate = SampleRate
	}
	return time.Duration(len(pcm)/2) * time.Second / time.Duration(rate)
}

// Silence is d of silent PCM at SampleRate.
func Silence(d time.Duration) []byte {
	n := int(d.Seconds()*SampleRate) * 2
	return make([]byte, n)
}

// WAV wraps PCM16 mono at rate Hz (0: SampleRate) in a WAV header.
func WAV(pcm []byte, rate int) []byte {
	if rate <= 0 {
		rate = SampleRate
	}
	var b bytes.Buffer
	b.Grow(44 + len(pcm))
	w := func(v any) { binary.Write(&b, binary.LittleEndian, v) }
	b.WriteString("RIFF")
	w(uint32(36 + len(pcm)))
	b.WriteString("WAVEfmt ")
	w(uint32(16))
	w(uint16(1)) // PCM
	w(uint16(1)) // mono
	w(uint32(rate))
	w(uint32(rate * 2))
	w(uint16(2))
	w(uint16(16))
	b.WriteString("data")
	w(uint32(len(pcm)))
	b.Write(pcm)
	return b.Bytes()
}

// ReadWAV returns the PCM of a 16-bit PCM WAV file and its rate and
// channels. Callers that need VoxtNew's format check them (16 kHz mono).
func ReadWAV(data []byte) (pcm []byte, rate, channels int, err error) {
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, 0, errors.New("not a WAV file")
	}
	bits := 0
	for p := 12; p+8 <= len(data); {
		id, size := string(data[p:p+4]), int(binary.LittleEndian.Uint32(data[p+4:p+8]))
		body := data[p+8:]
		if size > len(body) {
			size = len(body)
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, 0, 0, errors.New("short fmt chunk")
			}
			if f := binary.LittleEndian.Uint16(body[0:2]); f != 1 && f != 0xFFFE {
				return nil, 0, 0, errors.New("not PCM")
			}
			channels = int(binary.LittleEndian.Uint16(body[2:4]))
			rate = int(binary.LittleEndian.Uint32(body[4:8]))
			bits = int(binary.LittleEndian.Uint16(body[14:16]))
		case "data":
			if bits != 16 {
				return nil, 0, 0, errors.New("not 16-bit PCM")
			}
			return body[:size], rate, channels, nil
		}
		p += 8 + size + size%2
	}
	return nil, 0, 0, errors.New("no data chunk")
}

// Frames splits pcm into frames of d (the last one shorter), at rate Hz
// (0: SampleRate).
func Frames(pcm []byte, d time.Duration, rate int) [][]byte {
	if rate <= 0 {
		rate = SampleRate
	}
	n := int(d.Seconds()*float64(rate)) * 2
	if n <= 0 {
		n = len(pcm)
	}
	var out [][]byte
	for len(pcm) > 0 {
		k := min(n, len(pcm))
		out = append(out, pcm[:k])
		pcm = pcm[k:]
	}
	return out
}

// Rechunk collects what Send gets into frames of exactly size bytes, for
// streams that want fixed frames: Add returns the full frames so far,
// Rest what is left at Finish.
type Rechunk struct {
	Size int
	buf  []byte
}

// Add appends pcm and returns the full frames it completes.
func (r *Rechunk) Add(pcm []byte) [][]byte {
	r.buf = append(r.buf, pcm...)
	var out [][]byte
	for r.Size > 0 && len(r.buf) >= r.Size {
		f := make([]byte, r.Size)
		copy(f, r.buf)
		out = append(out, f)
		r.buf = r.buf[r.Size:]
	}
	return out
}

// Rest returns what is left (shorter than Size) and empties the buffer.
func (r *Rechunk) Rest() []byte {
	f := r.buf
	r.buf = nil
	return f
}
