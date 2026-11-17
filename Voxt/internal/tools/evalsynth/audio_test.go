package main

import (
	"math"
	"testing"
)

func tone(freq float64, rate, n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = 0.5 * math.Sin(2*math.Pi*freq*float64(i)/float64(rate))
	}
	return x
}

// rms of the middle half (away from the edges' ramp).
func rms(x []float64) float64 {
	var s float64
	mid := x[len(x)/4 : 3*len(x)/4]
	for _, v := range mid {
		s += v * v
	}
	return math.Sqrt(s / float64(len(mid)))
}

func TestResample(t *testing.T) {
	pass := resample(tone(1000, 24000, 24000), 24000, 16000)
	if len(pass) != 16000 || math.Abs(rms(pass)-0.5/math.Sqrt2) > 0.005 {
		t.Errorf("1 kHz: %d samples, rms %.4f", len(pass), rms(pass))
	}
	// Above the new Nyquist frequency: gone.
	if stop := resample(tone(9000, 24000, 24000), 24000, 16000); rms(stop) > 0.005 {
		t.Errorf("9 kHz: rms %.4f", rms(stop))
	}
}

func TestProcessAndWAV(t *testing.T) {
	x := append(append(make([]float64, 2400), tone(440, 24000, 12000)...), make([]float64, 4800)...)
	y := process(x, 24000, 0, 0)
	if want := 8000 + 2*4800; math.Abs(float64(len(y)-want)) > 2 {
		t.Errorf("%d samples, want %d (0.5 s of tone and 0.3 s on each side)", len(y), want)
	}
	b := writeWAV(y, 7)
	if b2 := writeWAV(y, 7); string(b) != string(b2) {
		t.Error("not reproducible")
	}
	back, rate, err := readWAV(b)
	if err != nil || rate != 16000 || len(back) != len(y) {
		t.Fatalf("read back: %v %d %d", err, rate, len(back))
	}
	peak := peakOf(back)
	if math.Abs(20*math.Log10(peak)+3) > 0.1 {
		t.Errorf("peak %.2f dBFS, want -3", 20*math.Log10(peak))
	}
	if y := process(x, 24000, 1, 1.5); len(y) != 24000 || math.Abs(20*math.Log10(peakOf(y))+2) > 0.01 {
		t.Errorf("-length 1.5 -gain 1: %d samples, peak %.2f dBFS", len(y), 20*math.Log10(peakOf(y)))
	}
}

func peakOf(x []float64) float64 {
	p := 0.0
	for _, v := range x {
		p = max(p, math.Abs(v))
	}
	return p
}
