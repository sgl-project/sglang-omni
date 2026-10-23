package main

import (
	"encoding/binary"
	"fmt"
	"math"
)

const (
	outRate   = 16000
	trimLevel = 0.001 // -60 dBFS
	padSecs   = 0.3
	peakLevel = 0.7079457843841379 // -3 dBFS
	sincHalf  = 32                 // taps on each side, at the lower rate
)

// readWAV reads a mono WAV of 16-bit PCM or 32-bit float samples.
func readWAV(b []byte) ([]float64, int, error) {
	if len(b) < 12 || string(b[:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a WAV file")
	}
	var format, channels, bits uint16
	var rate uint32
	for p := 12; p+8 <= len(b); {
		id, size := string(b[p:p+4]), int(binary.LittleEndian.Uint32(b[p+4:]))
		body := b[p+8 : min(p+8+size, len(b))]
		switch id {
		case "fmt ":
			if len(body) < 16 {
				return nil, 0, fmt.Errorf("short fmt chunk")
			}
			format, channels = binary.LittleEndian.Uint16(body), binary.LittleEndian.Uint16(body[2:])
			rate, bits = binary.LittleEndian.Uint32(body[4:]), binary.LittleEndian.Uint16(body[14:])
		case "data":
			if channels != 1 {
				return nil, 0, fmt.Errorf("%d channels, want mono", channels)
			}
			switch {
			case format == 1 && bits == 16:
				x := make([]float64, len(body)/2)
				for i := range x {
					x[i] = float64(int16(binary.LittleEndian.Uint16(body[2*i:]))) / 32768
				}
				return x, int(rate), nil
			case format == 3 && bits == 32:
				x := make([]float64, len(body)/4)
				for i := range x {
					x[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(body[4*i:])))
				}
				return x, int(rate), nil
			}
			return nil, 0, fmt.Errorf("format %d, %d bits: want 16-bit PCM or 32-bit float", format, bits)
		}
		p += 8 + size + size%2
	}
	return nil, 0, fmt.Errorf("no data chunk")
}

// process trims, resamples to outRate, pads and normalises x: its peak
// to -3 dBFS, then gain dB more; length (seconds) pads the end further.
func process(x []float64, rate int, gain, length float64) []float64 {
	from, to := 0, len(x)
	for from < to && math.Abs(x[from]) < trimLevel {
		from++
	}
	for to > from && math.Abs(x[to-1]) < trimLevel {
		to--
	}
	y := resample(x[from:to], rate, outRate)
	pad := int(padSecs * outRate)
	out := make([]float64, pad, len(y)+2*pad)
	out = append(out, y...)
	out = append(out, make([]float64, max(pad, int(length*outRate)-len(out)))...)
	peak := 0.0
	for _, v := range out {
		peak = max(peak, math.Abs(v))
	}
	if peak > 0 {
		for i := range out {
			out[i] *= peakLevel / peak * math.Pow(10, gain/20)
		}
	}
	return out
}

// resample converts x from rate in to rate out with a Blackman-windowed
// sinc, cut off a little under the lower rate's Nyquist frequency.
func resample(x []float64, in, out int) []float64 {
	if in == out {
		return append([]float64(nil), x...)
	}
	ratio := float64(in) / float64(out)
	cut := 0.95 * math.Min(1, 1/ratio) // of the input's Nyquist
	half := float64(sincHalf) / cut    // taps on each side, in input samples
	n := int(float64(len(x)) / ratio)
	y := make([]float64, n)
	for i := range y {
		t := float64(i) * ratio
		lo, hi := int(math.Ceil(t-half)), int(math.Floor(t+half))
		var sum float64
		for k := max(lo, 0); k <= min(hi, len(x)-1); k++ {
			d := t - float64(k)
			w := 0.42 + 0.5*math.Cos(math.Pi*d/half) + 0.08*math.Cos(2*math.Pi*d/half)
			sum += x[k] * cut * sinc(cut*d) * w
		}
		y[i] = sum
	}
	return y
}

func sinc(v float64) float64 {
	if v == 0 {
		return 1
	}
	return math.Sin(math.Pi*v) / (math.Pi * v)
}

// writeWAV is x as a 16-bit mono PCM WAV at outRate, with triangular
// dither from seed.
func writeWAV(x []float64, seed uint32) []byte {
	state := seed*2654435761 + 1
	rnd := func() float64 { // xorshift32, in [0, 1)
		state ^= state << 13
		state ^= state >> 17
		state ^= state << 5
		return float64(state) / (1 << 32)
	}
	data := make([]byte, 2*len(x))
	for i, v := range x {
		s := math.Round(v*32767 + rnd() - rnd())
		binary.LittleEndian.PutUint16(data[2*i:], uint16(int16(max(-32768, min(32767, s)))))
	}
	h := make([]byte, 44)
	copy(h, "RIFF")
	binary.LittleEndian.PutUint32(h[4:], uint32(36+len(data)))
	copy(h[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(h[16:], 16)
	binary.LittleEndian.PutUint16(h[20:], 1)
	binary.LittleEndian.PutUint16(h[22:], 1)
	binary.LittleEndian.PutUint32(h[24:], outRate)
	binary.LittleEndian.PutUint32(h[28:], outRate*2)
	binary.LittleEndian.PutUint16(h[32:], 2)
	binary.LittleEndian.PutUint16(h[34:], 16)
	copy(h[36:], "data")
	binary.LittleEndian.PutUint32(h[40:], uint32(len(data)))
	return append(h, data...)
}
