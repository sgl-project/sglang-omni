package dsp

import (
	"bytes"
	"math"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
)

func db(g float64) float64 { return 20 * math.Log10(g) }

// TestHighPassResponse: the filter's gain at the frequencies that matter,
// from its coefficients and measured on sines (after the filter settles).
func TestHighPassResponse(t *testing.T) {
	cases := []struct {
		hz, lo, hi float64 // dB
	}{
		{0.5, -200, -90}, // DC, near enough
		{50, -17, -15.5}, // mains hum: 4th order, 80 Hz: -16.4 dB
		{60, -11, -9.5},
		{80, -3.2, -2.8}, // the corner
		{100, -0.9, -0.4},
		{200, -0.05, 0.01},
		{1000, -0.01, 0.01},
		{4000, -0.01, 0.01},
	}
	h := NewHighPass(rate, 0)
	for _, c := range cases {
		if g := db(h.Response(c.hz)); g < c.lo || g > c.hi {
			t.Errorf("%v Hz: %.2f dB, want %v…%v", c.hz, g, c.lo, c.hi)
		}
		if c.hz < 20 {
			continue
		}
		in := sine(sec(2), c.hz, -20, 0)
		out := toFloat(NewHighPass(rate, 0).Process(toPCM(in)))
		got := rmsDB(out[sec(1):]) - rmsDB(in[sec(1):])
		if math.Abs(got-db(h.Response(c.hz))) > 0.1 {
			t.Errorf("%v Hz measured %.2f dB, response %.2f dB", c.hz, got, db(h.Response(c.hz)))
		}
		t.Logf("%5v Hz: %6.2f dB", c.hz, got)
	}
}

// TestHighPassDC: an offset goes, without a click at the start (the filter
// settles on it) and with a short transient when it jumps later.
func TestHighPassDC(t *testing.T) {
	in := sine(sec(1), 1000, -30, 0)
	for i := range in {
		in[i] += 3000 // about -21 dBFS of DC
	}
	out := toFloat(NewHighPass(rate, 0).Process(toPCM(in)))
	tone := 32768 * math.Sqrt2 * math.Pow(10, -30.0/20)
	// The tone's own start rings a little; the offset adds nothing.
	if p := peakOf(out[:sec(0.05)]); p > db(tone/32768)+1.5 {
		t.Errorf("start: peak %.1f dBFS, the tone is %.1f", p, db(tone/32768))
	}
	if m := mean(out[sec(0.1):]); math.Abs(m) > 2 {
		t.Errorf("DC left: %.1f", m)
	}
	// A jump of the offset halfway: gone within 50 ms.
	for i := sec(0.5); i < len(in); i++ {
		in[i] -= 6000
	}
	out = toFloat(NewHighPass(rate, 0).Process(toPCM(in)))
	if p := peakOf(out[sec(0.55):]); p > db(tone/32768)+0.5 {
		t.Errorf("after the jump: peak %.1f dBFS", p)
	}
}

func mean(s []float64) float64 {
	var m float64
	for _, v := range s {
		m += v
	}
	return m / float64(len(s))
}

// TestHighPassStream: chunk by chunk is the whole at once, byte for byte.
func TestHighPassStream(t *testing.T) {
	s := loadSample(t)
	pcm := toPCM(add(s, hum(len(s), -35)))
	whole := NewHighPass(rate, 0).Process(pcm)
	for _, n := range []int{320, 1600, 1601, 4096} {
		h := NewHighPass(rate, 0)
		var got []byte
		for off := 0; off < len(pcm); off += n * 2 {
			got = append(got, h.Process(pcm[off:min(off+n*2, len(pcm))])...)
		}
		if !bytes.Equal(got, whole) {
			t.Errorf("chunks of %d differ", n)
		}
	}
}

func TestGainFor(t *testing.T) {
	p := Defaults()
	cases := []struct {
		name                   string
		speech, floor, peak, g float64
	}{
		{"no speech", minDB, -70, minDB, 0},
		{"normal", -20, -65, -3, 0},
		{"just under the target", -23.3, -65, -6, 0},
		{"quiet", -30, -65, -12, 7},
		{"very quiet: capped", -45, -75, -28, 12},
		{"noisy room: the floor holds it", -38, -55, -20, 5},
		{"floor already at the ceiling", -38, -48, -20, 0},
		{"peaky: the limiter would take too much", -35, -75, -4, 9},
		{"loud", -8, -60, 0, -4},
		{"very loud: cut at most 6", -2, -60, 0, -6},
	}
	for _, c := range cases {
		if g := p.GainFor(c.speech, c.floor, c.peak); math.Abs(g-c.g) > 1e-9 {
			t.Errorf("%s: %.2f, want %.2f", c.name, g, c.g)
		}
	}
}

func TestParamsResolved(t *testing.T) {
	if got := (Params{}).Resolved(); got != Defaults() {
		t.Errorf("zero: %+v", got)
	}
	p := Params{TargetDB: 3, MaxGainDB: -1, LoudDB: -30, CeilingDB: math.NaN(), HighPassHz: 5000, RiseMS: -1}.Resolved()
	d := Defaults()
	if p.TargetDB != d.TargetDB || p.MaxGainDB != d.MaxGainDB || p.LoudDB != d.TargetDB || p.CeilingDB != d.CeilingDB || p.HighPassHz != d.HighPassHz || p.RiseMS != d.RiseMS {
		t.Errorf("resolved %+v", p)
	}
	if h := NewHighPass(8000, 5000); h.hz != 1600 {
		t.Errorf("corner %v", h.hz)
	}
}

// TestQuietVoice: a quiet voice in a quiet room is raised toward the
// target, as a file and as a stream; a normal one is left as it is (the
// filter only: the same bytes as the high-pass alone).
func TestQuietVoice(t *testing.T) {
	s := loadSample(t)
	level := sampleLevel(s)
	for _, c := range []struct {
		name string
		gain float64
		want float64
	}{{"quiet -12 dB", -12, raise(level - 12)}, {"quiet -20 dB", -20, raise(level - 20)}, {"normal", 0, 0}} {
		x := add(pad(gainOf(s, c.gain), 1, 1), white(sec(12.5), -70, 1))
		pcm := toPCM(x)
		f := ProcessFile(pcm, rate, Params{}, vad.Params{})
		if math.Abs(f.Report.GainDB-c.want) > 0.5 {
			t.Errorf("%s: file gain %+.2f, want %+.1f (speech %.1f dBFS)", c.name, f.Report.GainDB, c.want, f.Report.SpeechDB)
		}
		filtered := rmsDB(toFloat(NewHighPass(rate, 0).Process(pcm)))
		if d := rmsDB(toFloat(f.PCM)) - filtered; math.Abs(d-f.Report.GainDB) > 0.1 {
			t.Errorf("%s: %+.2f dB louder with a gain of %+.2f dB", c.name, d, f.Report.GainDB)
		}
		// No gain: the filter alone, but for the limiter on its overshoot.
		if only := NewHighPass(rate, 0).Process(pcm); c.want == 0 && !bytes.Equal(f.PCM, only) &&
			(f.Report.Limited == 0 || math.Abs(rmsDB(toFloat(f.PCM))-rmsDB(toFloat(only))) > 0.01) {
			t.Errorf("%s: more than the filter (%+v)", c.name, f.Report)
		}
		out, lev := stream(pcm, 1600, Params{}, vad.Params{}, true)
		if r := lev.Report(); math.Abs(r.GainDB-f.Report.GainDB) > 1 {
			t.Errorf("%s: stream gain %+.2f, file %+.2f", c.name, r.GainDB, f.Report.GainDB)
		}
		if p := peakOf(toFloat(out)); p > -1+0.01 {
			t.Errorf("%s: stream peak %.2f", c.name, p)
		}
	}
}

// TestSilenceNotRaised: noise alone gets no gain, as a file and as a
// stream, and a stream's gain holds through silence: it does not creep up
// before the speech, in the pauses or after it.
func TestSilenceNotRaised(t *testing.T) {
	for _, db := range []float64{-80, -65, -50} {
		pcm := toPCM(white(sec(5), db, 2))
		if f := ProcessFile(pcm, rate, Params{}, vad.Params{}); f.Report.GainDB != 0 || f.Analysis.Speech {
			t.Errorf("noise %v: %+v", db, f.Report)
		}
		out, lev := stream(pcm, 1600, Params{}, vad.Params{}, false)
		if lev.GainDB() != 0 || math.Abs(rmsDB(toFloat(out))-rmsDB(white(sec(5), db, 2))) > 0.2 {
			t.Errorf("noise %v: stream gain %+.2f", db, lev.GainDB())
		}
	}
	// Quiet speech between 3 s of room before and after: the gain holds
	// before (0), during the pause after it (what speech set) and the
	// room around it never rises over NoiseCeilDB.
	s := loadSample(t)
	x := add(pad(gainOf(s, -20), 3, 3), white(sec(16.5), -64, 3))
	pcm := toPCM(x)
	hp := NewHighPass(rate, 0)
	lev := NewLeveler(rate, Params{}, vad.Params{})
	var gains []float64
	var out []byte
	for off := 0; off < len(pcm); off += 640 { // 20 ms
		out = append(out, lev.Process(hp.Process(pcm[off:min(off+640, len(pcm))]))...)
		gains = append(gains, lev.GainDB())
	}
	at := func(sec float64) float64 { return gains[int(sec*50)] }
	if at(2.5) != 0 {
		t.Errorf("before the speech: %+.2f dB", at(2.5))
	}
	end := at(13.4)
	if end < 8 || math.Abs(at(16.4)-end) > 0.01 {
		t.Errorf("gain at the end of speech %+.2f, 3 s later %+.2f", end, at(16.4))
	}
	o := toFloat(out)
	if r := rmsDB(o[sec(14):sec(16.4)]); r > Defaults().NoiseCeilDB+1 {
		t.Errorf("room after the speech at %.1f dBFS", r)
	}
}

// TestGainSmooth: once set, a stream's gain moves at most a little per
// 20 ms, never in a step, and a voice that gets louder halfway brings
// it down within a few seconds.
func TestGainSmooth(t *testing.T) {
	s := loadSample(t)
	x := append(gainOf(s, -25), gainOf(s, -10)...)
	x = add(pad(x, 0.5, 0.5), white(sec(22), -70, 4))
	pcm := toPCM(x)
	hp := NewHighPass(rate, 0)
	lev := NewLeveler(rate, Params{}, vad.Params{})
	var prev float64
	started := false
	var gains []float64
	for off := 0; off < len(pcm); off += 640 { // 20 ms
		lev.Process(hp.Process(pcm[off:min(off+640, len(pcm))]))
		g := lev.GainDB()
		if started && math.Abs(g-prev) > 1 {
			t.Fatalf("at %.2f s the gain went %+.2f → %+.2f in 20 ms", float64(off/2)/rate, prev, g)
		}
		started = started || lev.started
		prev = g
		gains = append(gains, g)
	}
	// The halves' speech is at about -45 and -30 dBFS: +12 (the most),
	// then about +7 (to -23), near it 5 s after the change (the level is
	// averaged over 3 s of speech, the pauses do not count).
	want := raise(sampleLevel(s) - 10)
	first, later, second := gains[10*50], gains[16*50], gains[21*50]
	if first < 11.5 || math.Abs(later-want) > 1.5 || math.Abs(second-want) > 0.5 {
		t.Errorf("gain %+.1f in the quiet half, %+.1f 5 s into the louder one, %+.1f at its end", first, later, second)
	}
	t.Logf("gain %+.1f, then %+.1f; report %+v", first, second, lev.Report())
}

// TestLimiter: whatever the input and gain, nothing goes out over the
// ceiling.
func TestLimiter(t *testing.T) {
	sq := make([]float64, sec(1))
	for i := range sq {
		sq[i] = 32767
		if i/40%2 == 1 { // 200 Hz
			sq[i] = -32768
		}
	}
	s := loadSample(t)
	for _, c := range []struct {
		name string
		x    []float64
		p    Params
	}{
		{"full-scale square", sq, Params{}},
		{"full-scale square, 40 Hz", func() []float64 {
			q := make([]float64, sec(1))
			for i := range q {
				q[i] = 32767 * math.Copysign(1, math.Sin(2*math.Pi*40*float64(i)/rate))
			}
			return q
		}(), Params{}},
		{"quiet speech raised a lot", gainOf(s, -20), Params{MaxGainDB: 30, TargetDB: -6, LoudDB: -3, LimitDB: 30}},
		{"loud, clipped speech", gainOf(s, 12), Params{}},
	} {
		pcm := toPCM(c.x)
		f := ProcessFile(pcm, rate, c.p, vad.Params{})
		out, _ := stream(pcm, 1600, c.p, vad.Params{}, false)
		ceil := c.p.Resolved().CeilingDB + 0.01
		if p := peakOf(toFloat(f.PCM)); p > ceil {
			t.Errorf("%s: file peak %.2f dBFS", c.name, p)
		}
		if p := peakOf(toFloat(out)); p > ceil {
			t.Errorf("%s: stream peak %.2f dBFS", c.name, p)
		}
		t.Logf("%s: gain %+.1f, %d samples limited, %d clipped by the filter", c.name, f.Report.GainDB, f.Report.Limited, f.Report.Clipped)
	}
}

// TestExtremes: digital silence, nothing, a byte, a few samples.
func TestExtremes(t *testing.T) {
	for _, pcm := range [][]byte{nil, {1}, {1, 2, 3}, toPCM(make([]float64, sec(2))), toPCM(sine(100, 1000, -10, 0))} {
		f := ProcessFile(pcm, rate, Params{}, vad.Params{})
		if len(f.PCM) != len(pcm)/2*2 || f.Report.GainDB != 0 || f.Analysis.Speech {
			t.Errorf("%d bytes: %d out, %+v", len(pcm), len(f.PCM), f.Report)
		}
		out, lev := stream(pcm, 1600, Params{}, vad.Params{}, false)
		if len(out) != len(pcm)/2*2 || lev.GainDB() != 0 {
			t.Errorf("%d bytes: stream %d out, gain %v", len(pcm), len(out), lev.GainDB())
		}
	}
	// Zeros stay zeros.
	zero := toPCM(make([]float64, sec(1)))
	if f := ProcessFile(zero, rate, Params{}, vad.Params{}); !bytes.Equal(f.PCM, zero) {
		t.Error("zeros changed")
	}
	// Both stages off: a copy.
	pcm := toPCM(sine(sec(1), 50, -20, 0))
	f := ProcessFile(pcm, rate, Params{HighPassOff: true, LevelOff: true}, vad.Params{})
	if !bytes.Equal(f.PCM, pcm) || &f.PCM[0] == &pcm[0] {
		t.Error("off: not a copy")
	}
}

// TestTenMinutesFast: ten minutes of audio take well under a second or
// two, as a file and as a stream (not timed under the race detector).
func TestTenMinutesFast(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	if raceOn {
		t.Skip("timings mean nothing under the race detector")
	}
	s := loadSample(t)
	var x []float64
	for len(x) < sec(600) {
		x = append(x, gainOf(s, -15)...)
	}
	pcm := toPCM(add(x, white(len(x), -66, 6)))
	start := time.Now()
	ProcessFile(pcm, rate, Params{}, vad.Params{})
	file := time.Since(start)
	start = time.Now()
	stream(pcm, 1600, Params{}, vad.Params{}, true)
	str := time.Since(start)
	t.Logf("10 min: file %v, stream %v", file, str)
	if file > 5*time.Second || str > 5*time.Second {
		t.Errorf("slow: file %v, stream %v", file, str)
	}
}
