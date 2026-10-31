package vad

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
)

const rate = 16000

func sec(s float64) int { return int(s * rate) }

func dur(samples int) time.Duration { return time.Duration(samples) * time.Second / rate }

// loadSample is the 10.5 s speech sample as samples.
func loadSample(t testing.TB) []int16 {
	t.Helper()
	b, err := os.ReadFile("../../testdata/speech/sample-zh-en.wav")
	if err != nil {
		t.Skipf("no sample: %v", err)
	}
	pcm, r, ch, err := asr.ReadWAV(b)
	if err != nil || r != rate || ch != 1 {
		t.Fatalf("sample: rate %d channels %d err %v", r, ch, err)
	}
	return toSamples(pcm)
}

// onsetOf is where the speech in s starts, in seconds (about 0.3: the
// sample has 0.3 s of silence first, and its first word starts softly).
func onsetOf(s []int16) float64 {
	return dur(Analyze(samplesPCM(s), rate, Params{}).Segments[0].Start).Seconds()
}

func toSamples(pcm []byte) []int16 {
	s := make([]int16, len(pcm)/2)
	for i := range s {
		s[i] = int16(binary.LittleEndian.Uint16(pcm[i*2:]))
	}
	return s
}

func toPCM(s []float64) []byte {
	b := make([]byte, len(s)*2)
	for i, v := range s {
		v = math.Round(min(max(v, -32768), 32767))
		binary.LittleEndian.PutUint16(b[i*2:], uint16(int16(v)))
	}
	return b
}

func samplesPCM(s []int16) []byte {
	b := make([]byte, len(s)*2)
	for i, v := range s {
		binary.LittleEndian.PutUint16(b[i*2:], uint16(v))
	}
	return b
}

// noiseSpec is the background added to a variant: white noise and a 50 Hz
// hum, as RMS in dBFS (0: none).
type noiseSpec struct {
	name           string
	whiteDB, humDB float64
}

// totalDB is the noise's RMS level in dBFS (-120 for none).
func (n noiseSpec) totalDB() float64 {
	p := 0.0
	if n.whiteDB != 0 {
		p += math.Pow(10, n.whiteDB/10)
	}
	if n.humDB != 0 {
		p += math.Pow(10, n.humDB/10)
	}
	if p == 0 {
		return minDB
	}
	return 10 * math.Log10(p)
}

// noise is n samples of the background, from a fixed seed.
func (ns noiseSpec) noise(n int, seed uint64) []float64 {
	r := rand.New(rand.NewPCG(seed, 0x5eed))
	out := make([]float64, n)
	white := 32768 * math.Pow(10, ns.whiteDB/20)
	hum := 32768 * math.Sqrt2 * math.Pow(10, ns.humDB/20)
	for i := range out {
		if ns.whiteDB != 0 {
			out[i] += r.NormFloat64() * white
		}
		if ns.humDB != 0 {
			out[i] += hum * math.Sin(2*math.Pi*50*float64(i)/rate+0.3)
		}
	}
	return out
}

var (
	quiet = noiseSpec{"silence", 0, 0}
	white = noiseSpec{"white -50", -50, 0}
	noisy = noiseSpec{"white -42 + hum -50", -42, -50}
)

const (
	padSec = 3.0 // silence before and after the sample
	gapSec = 2.0 // silence inserted inside it
)

// variant is the sample with padSec of silence before and after, gapSec
// inserted at the quietest frame between 4.5 s and 6.5 s, and ns under all
// of it. at maps a sample index of the original to the variant.
type variant struct {
	pcm    []byte
	insert int // where the gap went, in the sample
	at     func(i int) int
}

func makeVariant(s []int16, ns noiseSpec, seed uint64) variant {
	frame := rate / 50
	insert, low := 0, math.Inf(1)
	pcm := samplesPCM(s)
	for f := sec(4.5) / frame; (f+1)*frame <= sec(6.5); f++ {
		if e := frameDB(pcm[f*frame*2 : (f+1)*frame*2]); e < low {
			insert, low = f*frame, e
		}
	}
	pre, gap := sec(padSec), sec(gapSec)
	total := pre + len(s) + gap + pre
	mix := ns.noise(total, seed)
	at := func(i int) int {
		if i < insert {
			return pre + i
		}
		return pre + gap + i
	}
	for i, v := range s {
		mix[at(i)] += float64(v)
	}
	return variant{pcm: toPCM(mix), insert: insert, at: at}
}

// speechFrames are the 20 ms frames of the sample within 35 dB of its
// loudest one, as sample offsets, with their levels.
func speechFrames(s []int16) (starts []int, levels []float64) {
	pcm := samplesPCM(s)
	frame := rate / 50
	peak := minDB
	var all []float64
	for f := 0; (f+1)*frame <= len(s); f++ {
		e := frameDB(pcm[f*frame*2 : (f+1)*frame*2])
		all = append(all, e)
		peak = max(peak, e)
	}
	for f, e := range all {
		if e >= peak-35 {
			starts = append(starts, f*frame)
			levels = append(levels, e)
		}
	}
	return starts, levels
}

// outAt maps a sample of the variant into the compacted output: the span
// it is in, its index and its offset in the output (-1: cut).
func outAt(spans []Span, v int) (span, out int) {
	o := 0
	for k, s := range spans {
		if v >= s.Start && v < s.End {
			return k, o + v - s.Start
		}
		o += s.End - s.Start
	}
	return -1, -1
}

func TestCompactVariants(t *testing.T) {
	s := loadSample(t)
	starts, levels := speechFrames(s)
	frame := rate / 50
	fade := rate * fadeMS / 1000
	for k, ns := range []noiseSpec{quiet, white, noisy} {
		t.Run(ns.name, func(t *testing.T) {
			v := makeVariant(s, ns, uint64(k+1))
			a := Analyze(v.pcm, rate, Params{})
			c := Compact(v.pcm, rate, a, Params{})
			t.Logf("%s: orig %.1fs → %.1fs (sample %.1fs, gap inserted at %.2fs; %d segments, floor %.1f dBFS, peak %.1f dBFS)",
				ns.name, c.Original.Seconds(), c.Kept.Seconds(), dur(len(s)).Seconds(),
				dur(v.insert).Seconds(), len(a.Segments), a.FloorDB, a.PeakDB)
			if !a.Speech || !c.Changed {
				t.Fatalf("speech %v changed %v", a.Speech, c.Changed)
			}
			if limit := dur(len(s)) + 1500*time.Millisecond; c.Kept > limit {
				t.Errorf("kept %v, want at most %v", c.Kept, limit)
			}
			if c.Kept != dur(len(c.PCM)/2) {
				t.Errorf("Kept %v but PCM plays %v", c.Kept, dur(len(c.PCM)/2))
			}
			// The output is the variant's spans, byte for byte, but for
			// the fades at the cuts.
			o := 0
			for _, sp := range c.Spans {
				n := sp.End - sp.Start
				lo, hi := 0, n
				if sp.Start > 0 {
					lo = fade
				}
				if sp.End < len(v.pcm)/2 {
					hi = n - fade
				}
				if !bytes.Equal(c.PCM[(o+lo)*2:(o+hi)*2], v.pcm[(sp.Start+lo)*2:(sp.Start+hi)*2]) {
					t.Errorf("span %v differs from the variant", sp)
				}
				o += n
			}
			// No speech frame is cut, and its level in the output is the
			// original's (where it stands 10 dB above the noise).
			lost, checked := 0, 0
			for i, f := range starts {
				k0, o0 := outAt(c.Spans, v.at(f))
				k1, _ := outAt(c.Spans, v.at(f+frame-1))
				if k0 < 0 || k0 != k1 {
					lost++
					t.Errorf("speech frame at %.2fs (%.1f dBFS) is cut", dur(f).Seconds(), levels[i])
					continue
				}
				sp := c.Spans[k0]
				vs := v.at(f)
				if vs-sp.Start < fade || sp.End-(vs+frame) < fade || levels[i] < ns.totalDB()+10 {
					continue
				}
				checked++
				if d := frameDB(c.PCM[o0*2:(o0+frame)*2]) - levels[i]; math.Abs(d) >= 1 {
					t.Errorf("speech frame at %.2fs: %.1f dBFS, output %+.2f dB", dur(f).Seconds(), levels[i], d)
				}
			}
			t.Logf("%d speech frames, %d lost, %d level-checked", len(starts), lost, checked)
		})
	}
}

func TestNoSpeech(t *testing.T) {
	click := func(base []float64, at float64, seed uint64) {
		// A 10 ms key click: a decaying burst peaking near -6 dBFS.
		r := rand.New(rand.NewPCG(seed, 1))
		for i := range sec(0.010) {
			base[sec(at)+i] += r.NormFloat64() * 16000 * math.Exp(-float64(i)/40)
		}
	}
	cases := map[string][]byte{
		"digital silence": make([]byte, sec(3)*2),
		"white -50":       toPCM(white.noise(sec(3), 7)),
		"white -42 + hum": toPCM(noisy.noise(sec(3), 8)),
	}
	for _, bg := range []noiseSpec{quiet, white} {
		one := bg.noise(sec(3), 9)
		click(one, 1, 1)
		cases["click, "+bg.name] = toPCM(one)
		typing := bg.noise(sec(3), 10)
		for k := range 12 {
			click(typing, 0.5+0.15*float64(k), uint64(k+2))
		}
		cases["typing, "+bg.name] = toPCM(typing)
	}
	for name, pcm := range cases {
		a := Analyze(pcm, rate, Params{})
		c := Compact(pcm, rate, a, Params{})
		if a.Speech || c.Changed || !bytes.Equal(c.PCM, pcm) {
			t.Errorf("%s: speech %v %v, changed %v", name, a.Speech, a.Segments, c.Changed)
		}
		g := NewGate(rate, Params{})
		for _, ch := range asr.Frames(pcm, 100*time.Millisecond, rate) {
			if out := g.Push(ch); out != nil {
				t.Errorf("%s: the gate sent %d bytes", name, len(out))
			}
		}
		if g.Open() || g.Heard() {
			t.Errorf("%s: gate open %v heard %v", name, g.Open(), g.Heard())
		}
	}
}

func TestSampleItself(t *testing.T) {
	s := loadSample(t)
	pcm := samplesPCM(s)
	a := Analyze(pcm, rate, Params{})
	c := Compact(pcm, rate, a, Params{})
	t.Logf("sample: orig %.1fs, segments %v, floor %.1f, peak %.1f", c.Original.Seconds(), a.Segments, a.FloorDB, a.PeakDB)
	// 0.3 s of silence at each end is within LeadMS and TailMS and its
	// pauses are shorter than LongGapMS: nothing to cut.
	if !a.Speech || c.Changed || &c.PCM[0] != &pcm[0] || c.Kept != c.Original {
		t.Errorf("speech %v changed %v kept %v", a.Speech, c.Changed, c.Kept)
	}
	if got := a.Segments[0].Start; got < sec(0.25) || got > sec(0.35) {
		t.Errorf("speech starts at %v, want about 0.3s", dur(got))
	}
	// The last word ends about half a second before the end (the take is
	// padded to 10.5 s), so the speech ends there, with or without more
	// silence after it.
	last := a.Segments[len(a.Segments)-1].End
	if last < len(s)-sec(0.7) || last > len(s) {
		t.Errorf("speech ends at %v, want within 0.7 s of the end (%v)", dur(last), dur(len(s)))
	}
	if a := Analyze(samplesPCM(append(s, make([]int16, sec(1))...)), rate, Params{}); a.Segments[len(a.Segments)-1].End != last {
		t.Errorf("with silence after it, speech ends at %v, want %v", dur(a.Segments[len(a.Segments)-1].End), dur(last))
	}
}

func TestSpeechAtStart(t *testing.T) {
	s := loadSample(t)
	on := sec(onsetOf(s))
	trimmed := samplesPCM(s[on:])
	a := Analyze(trimmed, rate, Params{})
	if !a.Speech || a.Segments[0].Start > sec(0.04) {
		t.Fatalf("segments %v, want the first from about 0", a.Segments)
	}
	// Under noise too: the percentile floor is the noise's, not InitFloorDB.
	n := white.noise(len(s)-on, 3)
	for i, v := range s[on:] {
		n[i] += float64(v)
	}
	if a := Analyze(toPCM(n), rate, Params{}); !a.Speech || a.Segments[0].Start > sec(0.04) {
		t.Fatalf("noisy: segments %v, want the first from about 0", a.Segments)
	}
}

func TestLongPauseInsideSpeech(t *testing.T) {
	// Two bursts of "speech" 3 s apart: the pause becomes GapKeepMS with a
	// fade at the join, and the ends keep LeadMS and TailMS.
	n := white.noise(sec(8), 11)
	for _, at := range []float64{1, 5} {
		for i := range sec(1) {
			n[sec(at)+i] += 6000 * math.Sin(2*math.Pi*220*float64(i)/rate)
		}
	}
	pcm := toPCM(n)
	a := Analyze(pcm, rate, Params{})
	c := Compact(pcm, rate, a, Params{})
	want := []Span{{sec(0.7), sec(2.2)}, {sec(4.8), sec(6.3)}}
	if len(a.Segments) != 2 || fmt.Sprint(c.Spans) != fmt.Sprint(want) {
		t.Fatalf("segments %v spans %v, want %v", a.Segments, c.Spans, want)
	}
	if c.Kept != 3*time.Second {
		t.Errorf("kept %v", c.Kept)
	}
	// The join fades out and in.
	j := sec(1.5) * 2
	if v := int16(binary.LittleEndian.Uint16(c.PCM[j-2:])); math.Abs(float64(v)) > 100 {
		t.Errorf("sample before the join %d, want faded", v)
	}
}

func TestDetectorChunking(t *testing.T) {
	s := loadSample(t)
	v := makeVariant(s, white, 5)
	whole := NewDetector(rate, Params{})
	whole.Write(v.pcm)
	parts := NewDetector(rate, Params{})
	for p := v.pcm; len(p) > 0; {
		k := min(777, len(p)) // odd: frames and samples split across writes
		parts.Write(p[:k])
		p = p[k:]
	}
	if fmt.Sprint(whole.Segments()) != fmt.Sprint(parts.Segments()) || whole.FloorDB() != parts.FloorDB() {
		t.Fatalf("whole %v, in parts %v", whole.Segments(), parts.Segments())
	}
	if !whole.Heard() || whole.Speaking() {
		t.Errorf("heard %v speaking %v", whole.Heard(), whole.Speaking())
	}
	// The causal detector agrees with Analyze where speech starts after
	// silence (within a frame or two).
	a := Analyze(v.pcm, rate, Params{})
	if d := whole.Segments()[0].Start - a.Segments[0].Start; d < -2*rate/50 || d > 2*rate/50 {
		t.Errorf("detector starts at %v, Analyze at %v", dur(whole.Segments()[0].Start), dur(a.Segments[0].Start))
	}
}

// pushAll feeds pcm to a gate in chunks of size bytes and returns what it
// sent and the end (in seconds) of the first chunk that produced output.
func pushAll(g *Gate, pcm []byte, size int) (sent []byte, first float64) {
	first = -1
	for p, off := pcm, 0; len(p) > 0; {
		k := min(size, len(p))
		out := g.Push(p[:k])
		off += k
		p = p[k:]
		if out != nil && first < 0 {
			first = dur(off / 2).Seconds()
		}
		sent = append(sent, out...)
	}
	return sent, first
}

func TestGate(t *testing.T) {
	s := loadSample(t)
	p := Defaults()
	for k, ns := range []noiseSpec{quiet, white, noisy} {
		v := makeVariant(s, ns, uint64(20+k))
		for _, size := range []int{sec(0.1) * 2, 999} {
			g := NewGate(rate, Params{})
			sent, first := pushAll(g, v.pcm, size)
			// The speech in the sample starts at about 0.3 s.
			onset := padSec + onsetOf(s)
			if first < onset || first > onset+0.2 {
				t.Errorf("%s/%d: first output after %.2fs, want soon after %.2fs", ns.name, size, first, onset)
			}
			if !g.Open() || !g.Heard() {
				t.Errorf("%s/%d: open %v heard %v", ns.name, size, g.Open(), g.Heard())
			}
			if !bytes.Equal(sent, v.pcm[len(v.pcm)-len(sent):]) || g.Dropped() != len(v.pcm)-len(sent) {
				t.Errorf("%s/%d: sent %d bytes, dropped %d, of %d: not the stream's tail",
					ns.name, size, len(sent), g.Dropped(), len(v.pcm))
			}
			dropped := dur(g.Dropped() / 2).Seconds()
			want := onset - float64(p.PreRollMS)/1000
			if math.Abs(dropped-want) > 0.1 {
				t.Errorf("%s/%d: dropped %.2fs, want about %.2fs", ns.name, size, dropped, want)
			}
			t.Logf("%s, chunks of %d bytes: dropped %.2fs, first output after %.2fs", ns.name, size, dropped, first)
		}
	}
}

func TestGateMaxHold(t *testing.T) {
	for k, ns := range []noiseSpec{white, noisy} {
		pcm := toPCM(ns.noise(sec(6), uint64(30+k)))
		g := NewGate(rate, Params{})
		sent, first := pushAll(g, pcm, sec(0.1)*2)
		if !g.Open() || g.Heard() || math.Abs(first-5) > 1e-9 {
			t.Errorf("%s: open %v heard %v first %.2f", ns.name, g.Open(), g.Heard(), first)
		}
		from := sec(5-0.3) * 2
		if !bytes.Equal(sent, pcm[from:]) || g.Dropped() != from {
			t.Errorf("%s: sent %d bytes, dropped %d; want from %d", ns.name, len(sent), g.Dropped(), from)
		}
		g2 := NewGate(rate, Params{})
		if sent, _ := pushAll(g2, pcm[:sec(2)*2], sec(0.1)*2); len(sent) != 0 || g2.Open() {
			t.Errorf("%s: 2 s of noise opened the gate", ns.name)
		}
		if g2.Dropped() > sec(2)*2 {
			t.Errorf("%s: dropped %d", ns.name, g2.Dropped())
		}
	}
}

func TestEdgeInputs(t *testing.T) {
	for _, pcm := range [][]byte{nil, {}, {1}, {1, 2, 3}} {
		a := Analyze(pcm, 0, Params{})
		c := Compact(pcm, -5, a, Params{})
		if a.Speech || c.Changed || len(c.PCM) != len(pcm) {
			t.Errorf("%v: %+v %+v", pcm, a, c)
		}
		g := NewGate(0, Params{})
		if g.Push(pcm) != nil {
			t.Errorf("%v: gate sent", pcm)
		}
		d := NewDetector(-1, Params{})
		d.Write(pcm)
		if d.Heard() || len(d.Segments()) != 0 || d.FloorDB() != -50 {
			t.Errorf("%v: detector %v %v", pcm, d.Segments(), d.FloorDB())
		}
	}
	// Segments past the end of pcm are clipped.
	pcm := make([]byte, sec(4)*2)
	c := Compact(pcm, rate, Analysis{Segments: []Segment{{sec(3.5), sec(9)}}, Speech: true}, Params{})
	if !c.Changed || fmt.Sprint(c.Spans) != fmt.Sprint([]Span{{sec(3.2), sec(4)}}) {
		t.Errorf("clipped: %v", c.Spans)
	}
}

func TestResolved(t *testing.T) {
	if Defaults().Resolved() != Defaults() || (Params{}).Resolved() != Defaults() {
		t.Fatal("defaults do not resolve to themselves")
	}
	p := Params{FrameMS: 1000, OnsetDB: 5, OffsetDB: 9, MinLevelDB: -500, LongGapMS: 200, GapKeepMS: 800, LeadMS: -3, InitFloorDB: 7}.Resolved()
	if p.FrameMS != 100 || p.OffsetDB != 5 || p.MinLevelDB != -120 || p.GapKeepMS != 200 || p.LeadMS != 300 || p.InitFloorDB != -50 {
		t.Errorf("%+v", p)
	}
}

// tenMinutes is ten minutes of the noisy variant, repeated.
func tenMinutes(t testing.TB) []byte {
	v := makeVariant(loadSample(t), white, 40)
	var b []byte
	for len(b) < sec(600)*2 {
		b = append(b, v.pcm...)
	}
	return b[:sec(600)*2]
}

func TestTenMinutesFast(t *testing.T) {
	pcm := tenMinutes(t)
	start := time.Now()
	a := Analyze(pcm, rate, Params{})
	c := Compact(pcm, rate, a, Params{})
	took := time.Since(start)
	t.Logf("10 min (%d MB): Analyze+Compact %v, orig %.0fs → %.0fs, %d segments",
		len(pcm)>>20, took, c.Original.Seconds(), c.Kept.Seconds(), len(a.Segments))
	if took > 2*time.Second { // the target is 200 ms; this only catches a blow-up
		t.Errorf("took %v", took)
	}
}

func BenchmarkAnalyzeCompact10Min(b *testing.B) {
	pcm := tenMinutes(b)
	b.SetBytes(int64(len(pcm)))
	b.ReportAllocs()
	for b.Loop() {
		Compact(pcm, rate, Analyze(pcm, rate, Params{}), Params{})
	}
}

func TestMerge(t *testing.T) {
	a := Analysis{Segments: []Segment{{100, 200}, {500, 600}}, Speech: true, FloorDB: -60, PeakDB: -20, Samples: 1000}
	b := Analysis{Segments: []Segment{{150, 300}, {700, 800}, {900, 900}}, Speech: true, FloorDB: -40, PeakDB: -10, Samples: 1000}
	m := Merge(a, b)
	want := []Segment{{100, 300}, {500, 600}, {700, 800}}
	if !slices.Equal(m.Segments, want) || !m.Speech || m.FloorDB != -60 || m.PeakDB != -10 || m.Samples != 1000 {
		t.Errorf("merged %+v", m)
	}
	if m := Merge(Analysis{}, Analysis{}); m.Speech || len(m.Segments) != 0 {
		t.Errorf("nothing: %+v", m)
	}
	if m := Merge(Analysis{}, b); !m.Speech || len(m.Segments) != 2 {
		t.Errorf("one side: %+v", m)
	}
}

// TestGateHear: a gate that gets one version of a stream and hears
// another opens when either has speech (here the heard one: the pushed
// one is quiet).
func TestGateHear(t *testing.T) {
	s := loadSample(t)
	v := makeVariant(s, white, 40)
	silent := toPCM(white.noise(len(v.pcm)/2, 41))
	g := NewGate(rate, Params{})
	var sent []byte
	for off := 0; off < len(silent); off += 3200 {
		end := min(off+3200, len(silent))
		g.Hear(v.pcm[off:end])
		sent = append(sent, g.Push(silent[off:end])...)
	}
	onset := padSec + 0.3 - float64(Defaults().PreRollMS)/1000
	if !g.Open() || !g.Heard() || math.Abs(dur(g.Dropped()/2).Seconds()-onset) > 0.1 || !bytes.Equal(sent, silent[g.Dropped():]) {
		t.Errorf("open %v heard %v dropped %v", g.Open(), g.Heard(), dur(g.Dropped()/2))
	}
}
