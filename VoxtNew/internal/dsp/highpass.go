package dsp

import (
	"encoding/binary"
	"math"
)

// butterworthQ are the quality factors of the two second-order sections of
// a fourth-order Butterworth filter: 1 / (2·cos(π/8)) and 1 / (2·cos(3π/8)).
var butterworthQ = [2]float64{0.5411961001461970, 1.3065629648763766}

// biquad is a second-order section in transposed direct form II,
// coefficients normalised by a0.
type biquad struct {
	b0, b1, b2, a1, a2 float64
	s1, s2             float64
}

// highPassBiquad is RBJ's (Audio EQ Cookbook) high-pass section at hz for
// a sample rate, with quality factor q (bilinear transform, prewarped).
func highPassBiquad(rate, hz, q float64) biquad {
	w := 2 * math.Pi * hz / rate
	cw, alpha := math.Cos(w), math.Sin(w)/(2*q)
	a0 := 1 + alpha
	return biquad{
		b0: (1 + cw) / 2 / a0,
		b1: -(1 + cw) / a0,
		b2: (1 + cw) / 2 / a0,
		a1: -2 * cw / a0,
		a2: (1 - alpha) / a0,
	}
}

func (f *biquad) step(x float64) float64 {
	y := f.b0*x + f.s1
	f.s1 = f.b1*x - f.a1*y + f.s2
	f.s2 = f.b2*x - f.a2*y
	return y
}

// settle puts the section in the state it reaches after a constant input
// x0 forever: its output is 0 (a high-pass passes no DC), so the first
// samples carry no step from 0 to the input's offset.
func (f *biquad) settle(x0 float64) {
	f.s2 = f.b2 * x0
	f.s1 = f.b1*x0 + f.s2
}

// response is the section's gain at hz.
func (f *biquad) response(rate, hz float64) float64 {
	w := 2 * math.Pi * hz / rate
	z1 := complex(math.Cos(w), -math.Sin(w)) // z⁻¹
	z2 := z1 * z1
	num := complex(f.b0, 0) + complex(f.b1, 0)*z1 + complex(f.b2, 0)*z2
	den := 1 + complex(f.a1, 0)*z1 + complex(f.a2, 0)*z2
	return abs(num / den)
}

func abs(c complex128) float64 { return math.Hypot(real(c), imag(c)) }

// primeSamples is how much of the start (20 ms at 16 kHz) gives the offset
// the filter settles on before its first sample.
const primeSamples = 320

// HighPass is a fourth-order Butterworth high-pass filter (two biquads,
// −24 dB per octave) over PCM16 mono. It keeps its state between calls,
// so a stream filtered chunk by chunk comes out as the whole recording
// filtered at once (when the first chunk has at least 20 ms). Not safe for
// concurrent use.
type HighPass struct {
	rate, hz float64
	s        [2]biquad
	primed   bool
	// Clipped counts the output samples that went past PCM16's range and
	// were clamped (only an input at full scale with strong low
	// frequencies does that).
	Clipped int
}

// NewHighPass is a filter at hz for PCM16 mono at rate Hz (rate <= 0:
// 16 kHz; hz <= 0: DefaultHighPassHz). hz is held under a fifth of the
// rate.
func NewHighPass(rate int, hz float64) *HighPass {
	if rate <= 0 {
		rate = DefaultRate
	}
	if !(hz > 0) {
		hz = DefaultHighPassHz
	}
	hz = min(hz, float64(rate)/5)
	h := &HighPass{rate: float64(rate), hz: hz}
	for i, q := range butterworthQ {
		h.s[i] = highPassBiquad(h.rate, hz, q)
	}
	return h
}

// Response is the filter's gain at hz (1 is 0 dB), from its coefficients.
func (h *HighPass) Response(hz float64) float64 {
	return h.s[0].response(h.rate, hz) * h.s[1].response(h.rate, hz)
}

// Process filters pcm (PCM16 mono, little-endian; an odd last byte is
// dropped) into a new slice. The first call settles the filter on the
// mean of its first 20 ms, so a DC offset does not ring at the start.
func (h *HighPass) Process(pcm []byte) []byte {
	n := len(pcm) / 2
	out := make([]byte, n*2)
	if n == 0 {
		return out
	}
	if !h.primed {
		h.primed = true
		k := min(n, primeSamples)
		var sum float64
		for i := range k {
			sum += float64(int16(binary.LittleEndian.Uint16(pcm[i*2:])))
		}
		h.s[0].settle(sum / float64(k))
		// The second section's input is the first's output: 0.
	}
	for i := range n {
		x := float64(int16(binary.LittleEndian.Uint16(pcm[i*2:])))
		y := h.s[1].step(h.s[0].step(x))
		v, clipped := toInt16(y)
		if clipped {
			h.Clipped++
		}
		binary.LittleEndian.PutUint16(out[i*2:], uint16(v))
	}
	return out
}

// toInt16 rounds v to a PCM16 sample, clamping (and saying so) past the
// range.
func toInt16(v float64) (int16, bool) {
	v = math.Round(v)
	switch {
	case v > math.MaxInt16:
		return math.MaxInt16, true
	case v < math.MinInt16:
		return math.MinInt16, true
	case v != v: // NaN
		return 0, true
	}
	return int16(v), false
}
