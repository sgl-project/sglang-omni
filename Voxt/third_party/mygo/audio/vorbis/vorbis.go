// Package vorbis decodes Ogg Vorbis files for package audio, with
// github.com/jfreymuth/oggvorbis. Import it for its side effect:
//
//	import _ "github.com/egoist/mygo/audio/vorbis"
package vorbis

import (
	"io"

	"github.com/egoist/mygo/audio"
	"github.com/jfreymuth/oggvorbis"
)

func init() {
	audio.RegisterFormat("vorbis", "OggS", func(r io.ReadSeeker) (audio.Decoder, error) { return NewDecoder(r) })
}

// NewDecoder returns a decoder of the Ogg Vorbis file r.
func NewDecoder(r io.ReadSeeker) (audio.Decoder, error) {
	v, err := oggvorbis.NewReader(r)
	if err != nil {
		return nil, err
	}
	return decoder{v}, nil
}

type decoder struct{ *oggvorbis.Reader }

func (d decoder) Len() int64 {
	if n := d.Length(); n > 0 {
		return n
	}
	return -1
}
