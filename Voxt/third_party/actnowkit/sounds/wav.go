package sounds

import (
	"bytes"
	"encoding/binary"
	"math"
)

// ToneWAV is a 16-bit mono WAV of a sine at freq Hz for seconds at gain
// (0–1), with 5 ms fades: a test sound that needs no catalogue.
func ToneWAV(freq, seconds, gain float64) []byte {
	const rate = 44100
	n := int(seconds * rate)
	fade := rate * 5 / 1000
	var pcm bytes.Buffer
	for i := range n {
		v := gain * math.Sin(2*math.Pi*freq*float64(i)/rate)
		if i < fade {
			v *= float64(i) / float64(fade)
		}
		if r := n - i; r < fade {
			v *= float64(r) / float64(fade)
		}
		binary.Write(&pcm, binary.LittleEndian, int16(v*32767))
	}
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+pcm.Len()))
	b.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), uint16(1), uint16(1), uint32(rate), uint32(rate * 2), uint16(2), uint16(16)} {
		binary.Write(&b, binary.LittleEndian, v)
	}
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(pcm.Len()))
	b.Write(pcm.Bytes())
	return b.Bytes()
}

// PCM is decoded audio: two channels of float32 samples at Rate.
type PCM struct {
	Rate        int
	Left, Right []float32
}

// EngineRate is the rate DecodeWAV converts to (the family's sounds are
// 44.1 kHz already).
const EngineRate = 44100

// DecodeWAV decodes a RIFF WAVE file (PCM 8/16/24/32-bit or IEEE float
// 32-bit, mono or more channels: the first two are kept, mono is doubled)
// to stereo float32 at EngineRate (linear interpolation when the file's
// rate differs). ok is false for anything else (the caller plays the
// bytes some other way).
func DecodeWAV(b []byte) (p PCM, ok bool) {
	if len(b) < 12 || string(b[:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return p, false
	}
	var format, channels, bits uint16
	var rate uint32
	var data []byte
	for i := 12; i+8 <= len(b); {
		id, size := string(b[i:i+4]), int(binary.LittleEndian.Uint32(b[i+4:i+8]))
		body := b[i+8:]
		if size > len(body) {
			size = len(body)
		}
		body = body[:size]
		switch id {
		case "fmt ":
			if size < 16 {
				return p, false
			}
			format = binary.LittleEndian.Uint16(body[0:2])
			channels = binary.LittleEndian.Uint16(body[2:4])
			rate = binary.LittleEndian.Uint32(body[4:8])
			bits = binary.LittleEndian.Uint16(body[14:16])
			if format == 0xFFFE && size >= 26 { // WAVE_FORMAT_EXTENSIBLE: the subformat's first two bytes
				format = binary.LittleEndian.Uint16(body[24:26])
			}
		case "data":
			data = body
		}
		i += 8 + size + size%2
	}
	if channels == 0 || rate == 0 || data == nil || !(format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32) || format == 3 && bits == 32) {
		return p, false
	}
	width := int(bits) / 8
	frame := width * int(channels)
	n := len(data) / frame
	sample := func(off int) float32 {
		s := data[off : off+width]
		switch {
		case format == 3:
			return math.Float32frombits(binary.LittleEndian.Uint32(s))
		case bits == 8:
			return (float32(s[0]) - 128) / 128
		case bits == 16:
			return float32(int16(binary.LittleEndian.Uint16(s))) / 32768
		case bits == 24:
			v := int32(uint32(s[0])<<8|uint32(s[1])<<16|uint32(s[2])<<24) >> 8
			return float32(v) / 8388608
		default:
			return float32(int32(binary.LittleEndian.Uint32(s))) / 2147483648
		}
	}
	left, right := make([]float32, n), make([]float32, n)
	for i := range n {
		left[i] = sample(i * frame)
		if channels > 1 {
			right[i] = sample(i*frame + width)
		} else {
			right[i] = left[i]
		}
	}
	p = PCM{Rate: EngineRate, Left: left, Right: right}
	if int(rate) != EngineRate {
		p.Left, p.Right = resample(left, int(rate)), resample(right, int(rate))
	}
	return p, true
}

func resample(in []float32, from int) []float32 {
	if len(in) == 0 {
		return in
	}
	n := int(int64(len(in)) * EngineRate / int64(from))
	out := make([]float32, n)
	step := float64(from) / EngineRate
	for i := range out {
		x := float64(i) * step
		j := int(x)
		f := float32(x - float64(j))
		a := in[min(j, len(in)-1)]
		b := in[min(j+1, len(in)-1)]
		out[i] = a + (b-a)*f
	}
	return out
}
