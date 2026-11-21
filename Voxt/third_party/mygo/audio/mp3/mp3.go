// Package mp3 decodes MP3 files for package audio, with
// github.com/hajimehoshi/go-mp3. Import it for its side effect:
//
//	import _ "github.com/egoist/mygo/audio/mp3"
package mp3

import (
	"encoding/binary"
	"io"

	"github.com/egoist/mygo/audio"
	gomp3 "github.com/hajimehoshi/go-mp3"
)

func init() {
	open := func(r io.ReadSeeker) (audio.Decoder, error) { return NewDecoder(r) }
	// An ID3 tag, or the sync word of a frame of MPEG-1 or 2 layer III.
	audio.RegisterFormat("mp3", "ID3", open)
	for _, b := range []string{"\xff\xfb", "\xff\xfa", "\xff\xf3", "\xff\xf2", "\xff\xe3", "\xff\xe2"} {
		audio.RegisterFormat("mp3", b, open)
	}
}

// NewDecoder returns a decoder of the MP3 file r, which decodes to stereo.
func NewDecoder(r io.ReadSeeker) (audio.Decoder, error) {
	d, err := gomp3.NewDecoder(r)
	if err != nil {
		return nil, err
	}
	return &decoder{d: d}, nil
}

// go-mp3 decodes to 16-bit stereo: 4 bytes a frame.
const frameBytes = 4

type decoder struct {
	d   *gomp3.Decoder
	buf []byte
}

func (d *decoder) SampleRate() int { return d.d.SampleRate() }
func (d *decoder) Channels() int   { return 2 }

func (d *decoder) Len() int64 {
	if n := d.d.Length(); n > 0 {
		return n / frameBytes
	}
	return -1
}

func (d *decoder) Read(dst []float32) (int, error) {
	n := len(dst) / 2 * frameBytes
	if cap(d.buf) < n {
		d.buf = make([]byte, n)
	}
	b := d.buf[:n]
	got, err := io.ReadFull(d.d, b)
	got -= got % frameBytes
	for i := 0; i < got/2; i++ {
		dst[i] = float32(int16(binary.LittleEndian.Uint16(b[2*i:]))) / (1 << 15)
	}
	if got > 0 {
		return got / 2, nil
	}
	if err == io.ErrUnexpectedEOF {
		err = io.EOF
	}
	return 0, err
}

func (d *decoder) SetPosition(frame int64) error {
	_, err := d.d.Seek(frame*frameBytes, io.SeekStart)
	return err
}
