package audio

import (
	"encoding/binary"
	"errors"
	"io"
	"math"
)

func init() {
	RegisterFormat("wav", "RIFF????WAVE", func(r io.ReadSeeker) (Decoder, error) { return newWAV(r) })
}

// A wavDecoder decodes WAV files of integer PCM of 8 to 32 bits, or of
// 32 or 64 bit floats.
type wavDecoder struct {
	r        io.ReadSeeker
	br       io.Reader
	rate     int
	channels int
	bits     int
	float    bool
	start    int64 // the data's offset
	length   int64 // frames, or -1
	pos      int64 // frame
	buf      []byte
}

var errWAV = errors.New("audio: invalid WAV")

func newWAV(r io.ReadSeeker) (*wavDecoder, error) {
	start, err := r.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}
	var riff [12]byte
	if _, err := io.ReadFull(r, riff[:]); err != nil {
		return nil, errWAV
	}
	d := &wavDecoder{r: r}
	off := start + 12
	haveFmt := false
	for {
		var h [8]byte
		if _, err := io.ReadFull(r, h[:]); err != nil {
			return nil, errWAV
		}
		size := int64(binary.LittleEndian.Uint32(h[4:]))
		off += 8
		switch string(h[:4]) {
		case "fmt ":
			if size < 16 || size > 1<<16 {
				return nil, errWAV
			}
			b := make([]byte, size)
			if _, err := io.ReadFull(r, b); err != nil {
				return nil, errWAV
			}
			tag := binary.LittleEndian.Uint16(b)
			d.channels = int(binary.LittleEndian.Uint16(b[2:]))
			d.rate = int(binary.LittleEndian.Uint32(b[4:]))
			d.bits = int(binary.LittleEndian.Uint16(b[14:]))
			if tag == 0xfffe && size >= 40 {
				// WAVE_FORMAT_EXTENSIBLE: the tag starts the sub-format.
				tag = binary.LittleEndian.Uint16(b[24:])
			}
			switch {
			case tag == 1 && d.bits >= 8 && d.bits <= 32 && d.bits%8 == 0:
			case tag == 3 && (d.bits == 32 || d.bits == 64):
				d.float = true
			default:
				return nil, errors.New("audio: WAV of an encoding other than PCM or float")
			}
			if d.channels < 1 || d.rate < 1 {
				return nil, errWAV
			}
			haveFmt = true
		case "data":
			if !haveFmt {
				return nil, errWAV
			}
			d.start = off
			d.length = -1
			if size != 0 && size != 0xffffffff {
				// Not one written as it streamed.
				d.length = size / int64(d.frameBytes())
			}
			d.br = readerOf(r)
			return d, nil
		}
		// Chunks are padded to even sizes.
		size += size & 1
		if _, err := r.Seek(off+size, io.SeekStart); err != nil {
			return nil, errWAV
		}
		off += size
	}
}

func (d *wavDecoder) frameBytes() int { return d.channels * d.bits / 8 }
func (d *wavDecoder) SampleRate() int { return d.rate }
func (d *wavDecoder) Channels() int   { return d.channels }
func (d *wavDecoder) Len() int64      { return d.length }

func (d *wavDecoder) Read(dst []float32) (int, error) {
	fb := d.frameBytes()
	n := len(dst) / d.channels
	if d.length >= 0 {
		n = int(min(int64(n), d.length-d.pos))
	}
	if n <= 0 {
		return 0, io.EOF
	}
	if cap(d.buf) < n*fb {
		d.buf = make([]byte, n*fb)
	}
	b := d.buf[:n*fb]
	got, err := io.ReadFull(d.br, b)
	n = got / fb
	if n == 0 {
		if err == nil || err == io.ErrUnexpectedEOF {
			err = io.EOF
		}
		return 0, err
	}
	bs := d.bits / 8
	for i := range n * d.channels {
		s := b[i*bs:]
		var v float32
		switch {
		case d.float && bs == 4:
			v = math.Float32frombits(binary.LittleEndian.Uint32(s))
		case d.float:
			v = float32(math.Float64frombits(binary.LittleEndian.Uint64(s)))
		case bs == 1:
			v = (float32(s[0]) - 128) / 128
		case bs == 2:
			v = float32(int16(binary.LittleEndian.Uint16(s))) / (1 << 15)
		case bs == 3:
			v = float32(int32(uint32(s[0])<<8|uint32(s[1])<<16|uint32(s[2])<<24)>>8) / (1 << 23)
		default:
			v = float32(int32(binary.LittleEndian.Uint32(s))) / (1 << 31)
		}
		dst[i] = v
	}
	d.pos += int64(n)
	return n * d.channels, nil
}

func (d *wavDecoder) SetPosition(frame int64) error {
	if frame < 0 || d.length >= 0 && frame > d.length {
		return errors.New("audio: seek out of range")
	}
	if _, err := d.r.Seek(d.start+frame*int64(d.frameBytes()), io.SeekStart); err != nil {
		return err
	}
	d.br = readerOf(d.r)
	d.pos = frame
	return nil
}
