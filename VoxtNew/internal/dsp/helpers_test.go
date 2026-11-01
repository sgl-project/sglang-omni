package dsp

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
)

const rate = 16000

func sec(s float64) int { return int(s * rate) }

// loadSample is the 10.5 s speech sample as samples (float, PCM16 scale).
func loadSample(t testing.TB) []float64 {
	t.Helper()
	b, err := os.ReadFile("../../testdata/speech/sample-zh-en.wav")
	if err != nil {
		t.Skipf("no sample: %v", err)
	}
	pcm, r, ch, err := asr.ReadWAV(b)
	if err != nil || r != rate || ch != 1 {
		t.Fatalf("sample: rate %d channels %d err %v", r, ch, err)
	}
	return toFloat(pcm)
}

func toFloat(pcm []byte) []float64 {
	s := make([]float64, len(pcm)/2)
	for i := range s {
		s[i] = float64(int16(binary.LittleEndian.Uint16(pcm[i*2:])))
	}
	return s
}

// toPCM rounds and clamps s to PCM16.
func toPCM(s []float64) []byte {
	b := make([]byte, len(s)*2)
	for i, v := range s {
		v = math.Round(min(max(v, -32768), 32767))
		binary.LittleEndian.PutUint16(b[i*2:], uint16(int16(v)))
	}
	return b
}

func sine(n int, hz, db, phase float64) []float64 {
	a := fullScale * math.Sqrt2 * math.Pow(10, db/20) // db is the RMS
	s := make([]float64, n)
	for i := range s {
		s[i] = a * math.Sin(2*math.Pi*hz*float64(i)/rate+phase)
	}
	return s
}

// white is n samples of white noise at db RMS, from seed.
func white(n int, db float64, seed uint64) []float64 {
	r := rand.New(rand.NewPCG(seed, 0x5eed))
	a := fullScale * math.Pow(10, db/20)
	s := make([]float64, n)
	for i := range s {
		s[i] = r.NormFloat64() * a
	}
	return s
}

// fan is n samples of a fan's noise at db RMS: white noise through a
// one-pole low-pass at 400 Hz (most of its energy low), with a 120 Hz
// blade tone 10 dB under it.
func fan(n int, db float64, seed uint64) []float64 {
	w := white(n, 0, seed)
	a := math.Exp(-2 * math.Pi * 400 / rate)
	s := make([]float64, n)
	var y float64
	for i, v := range w {
		y = a*y + (1-a)*v
		s[i] = y
	}
	return scaleTo(add(s, sine(n, 120, rmsDB(s)-10, 0.7)), db)
}

// scaleTo scales s to db RMS.
func scaleTo(s []float64, db float64) []float64 {
	r := rmsDB(s)
	g := math.Pow(10, (db-r)/20)
	out := make([]float64, len(s))
	for i, v := range s {
		out[i] = v * g
	}
	return out
}

func rmsDB(s []float64) float64 {
	if len(s) == 0 {
		return minDB
	}
	var sum float64
	for _, v := range s {
		sum += v * v
	}
	return powerDB(sum / float64(len(s)) / (fullScale * fullScale))
}

func peakOf(s []float64) float64 {
	var m float64
	for _, v := range s {
		m = max(m, math.Abs(v))
	}
	if m == 0 {
		return minDB
	}
	return 20 * math.Log10(m/fullScale)
}

func add(dst []float64, src ...[]float64) []float64 {
	out := append([]float64(nil), dst...)
	for _, s := range src {
		for i := range min(len(out), len(s)) {
			out[i] += s[i]
		}
	}
	return out
}

func gainOf(s []float64, db float64) []float64 {
	g := math.Pow(10, db/20)
	out := make([]float64, len(s))
	for i, v := range s {
		out[i] = v * g
	}
	return out
}

// pad puts before and after seconds of zeros around s.
func pad(s []float64, before, after float64) []float64 {
	out := make([]float64, sec(before), sec(before)+len(s)+sec(after))
	out = append(out, s...)
	return append(out, make([]float64, sec(after))...)
}

// stream runs pcm through the stream path in chunks of n samples: the
// high-pass, a gate when gated, the leveler; it returns what went out.
func stream(pcm []byte, n int, p Params, vp vad.Params, gated bool) ([]byte, *Leveler) {
	p = p.Resolved()
	hp := NewHighPass(rate, p.HighPassHz)
	var g *vad.Gate
	if gated {
		g = vad.NewGate(rate, vp)
	}
	lev := NewLeveler(rate, p, vp)
	var out []byte
	for off := 0; off < len(pcm); off += n * 2 {
		raw := pcm[off:min(off+n*2, len(pcm))]
		c := hp.Process(raw)
		if g != nil {
			g.Hear(raw)
			c = g.Push(c)
		}
		if len(c) > 0 {
			out = append(out, lev.Process(c)...)
		}
	}
	return out, lev
}

// sampleLevel is the speech sample's speech level (dBFS), as a file
// measures it.
func sampleLevel(s []float64) float64 {
	return ProcessFile(toPCM(s), rate, Params{}, vad.Params{}).Report.SpeechDB
}

// raise is the gain speech at level dBFS gets: up to the target, at most
// the most gain.
func raise(level float64) float64 {
	d := Defaults()
	return min(d.MaxGainDB, max(0, d.TargetDB-level))
}
