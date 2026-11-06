package vad

import (
	"math"
	"slices"
)

// DefaultRate is the rate used when a caller passes rate <= 0.
const DefaultRate = 16000

const (
	minDB = -120.0 // the level of a silent frame
	// digitalSilenceDB is under one bit of dither: such frames are "no
	// signal", not noise, and do not move the noise floor.
	digitalSilenceDB = -96.0
	riseTauMS        = 750.0 // the noise floor's time constant going up
	fallRate         = 0.5   // and the share of the way down it goes per frame
	floorPercentile  = 0.10  // Analyze's initial noise floor
)

// Params tune the detector; a zero field takes its default (Defaults()).
type Params struct {
	FrameMS     int     // analysis frame: 20
	OnsetDB     float64 // speech starts this far above the noise floor: 10
	OffsetDB    float64 // and lasts while frames stay this far above it: 6 (hysteresis)
	MinLevelDB  float64 // frames quieter than this (dBFS) are never speech: -60
	OnsetMS     int     // consecutive time above the onset threshold to start speech: 60 (rejects clicks)
	HangoverMS  int     // time below the offset threshold before speech ends: 300
	LeadMS      int     // kept before the first speech: 300
	TailMS      int     // kept after the last speech: 300
	LongGapMS   int     // a pause longer than this is shortened…: 1000
	GapKeepMS   int     // …to this (half after the speech before it, half before the next): 400
	MinSavingMS int     // when compacting would save less than this, keep the original: 500
	PreRollMS   int     // streaming gate: audio sent from before the onset: 300
	MaxHoldMS   int     // streaming gate: opens anyway after this much leading audio: 5000
	InitFloorDB float64 // the noise floor before any frame: -50
}

// Defaults are the tuned parameters.
func Defaults() Params {
	return Params{
		FrameMS:     20,
		OnsetDB:     10,
		OffsetDB:    6,
		MinLevelDB:  -60,
		OnsetMS:     60,
		HangoverMS:  300,
		LeadMS:      300,
		TailMS:      300,
		LongGapMS:   1000,
		GapKeepMS:   400,
		MinSavingMS: 500,
		PreRollMS:   300,
		MaxHoldMS:   5000,
		InitFloorDB: -50,
	}
}

// Resolved fills zero (and negative) fields with their defaults and clamps
// the rest to sensible ranges: frames of 5–100 ms, OffsetDB at most
// OnsetDB, levels within [-120, 0] dBFS, GapKeepMS at most LongGapMS.
func (p Params) Resolved() Params {
	d := Defaults()
	pos := func(v *int, def int) {
		if *v <= 0 {
			*v = def
		}
	}
	posF := func(v *float64, def float64) {
		if !(*v > 0) { // also NaN
			*v = def
		}
	}
	negF := func(v *float64, def float64) {
		if !(*v < 0) {
			*v = def
		}
		*v = max(*v, minDB)
	}
	pos(&p.FrameMS, d.FrameMS)
	p.FrameMS = min(max(p.FrameMS, 5), 100)
	posF(&p.OnsetDB, d.OnsetDB)
	posF(&p.OffsetDB, d.OffsetDB)
	p.OnsetDB = min(p.OnsetDB, 60)
	p.OffsetDB = min(p.OffsetDB, p.OnsetDB)
	negF(&p.MinLevelDB, d.MinLevelDB)
	negF(&p.InitFloorDB, d.InitFloorDB)
	pos(&p.OnsetMS, d.OnsetMS)
	pos(&p.HangoverMS, d.HangoverMS)
	pos(&p.LeadMS, d.LeadMS)
	pos(&p.TailMS, d.TailMS)
	pos(&p.LongGapMS, d.LongGapMS)
	pos(&p.GapKeepMS, d.GapKeepMS)
	p.GapKeepMS = min(p.GapKeepMS, p.LongGapMS)
	pos(&p.MinSavingMS, d.MinSavingMS)
	pos(&p.PreRollMS, d.PreRollMS)
	pos(&p.MaxHoldMS, d.MaxHoldMS)
	return p
}

// Segment is a stretch of speech, in samples, End exclusive.
type Segment struct{ Start, End int }

// Analysis is what Analyze found in a recording.
type Analysis struct {
	Segments []Segment
	Speech   bool    // any segment
	FloorDB  float64 // the noise floor estimate at the end
	PeakDB   float64 // the loudest frame
	Samples  int     // the recording's length in samples
}

// Analyze runs the detector over a whole recording (PCM16 mono at rate Hz;
// rate <= 0: DefaultRate). Its noise floor starts at a low percentile of the
// frame levels rather than at InitFloorDB, so speech at the very start is
// found too. A segment still open at the end ends with the recording.
func Analyze(pcm []byte, rate int, p Params) Analysis {
	d := NewDetector(rate, p)
	n := len(pcm) / 2
	levels := make([]float64, 0, n/d.frame)
	for off := 0; off+d.frame*2 <= len(pcm); off += d.frame * 2 {
		levels = append(levels, frameDB(pcm[off:off+d.frame*2]))
	}
	d.floor = initialFloor(levels, d.p.InitFloorDB)
	for _, e := range levels {
		d.step(e)
	}
	a := Analysis{FloorDB: d.floor, PeakDB: d.peak, Samples: n}
	a.Segments = append(a.Segments, d.segs...)
	if d.speaking {
		a.Segments = append(a.Segments, Segment{d.segStart, n})
	}
	a.Speech = len(a.Segments) > 0
	return a
}

// initialFloor is the floorPercentile level of the frames that carry a
// signal, never above init (all digital silence: init).
func initialFloor(levels []float64, init float64) float64 {
	var s []float64
	for _, e := range levels {
		if e > digitalSilenceDB {
			s = append(s, e)
		}
	}
	if len(s) == 0 {
		return init
	}
	slices.Sort(s)
	return min(s[int(float64(len(s)-1)*floorPercentile)], init)
}

// frameDB is the level of a frame of PCM16 (an even number of bytes) in
// dBFS, at least minDB.
func frameDB(b []byte) float64 {
	n := len(b) / 2
	if n == 0 {
		return minDB
	}
	var sum int64
	for i := 0; i+1 < len(b); i += 2 {
		v := int64(int16(uint16(b[i]) | uint16(b[i+1])<<8))
		sum += v * v
	}
	if sum == 0 {
		return minDB
	}
	ms := float64(sum) / float64(n) / (32768.0 * 32768.0)
	return max(10*math.Log10(ms), minDB)
}

// Detector is the streaming detector (Analyze runs it over a whole
// recording). It is not safe for concurrent use.
type Detector struct {
	p         Params
	rate      int
	frame     int // samples per frame
	onsetN    int // frames above the onset threshold that start speech
	hangN     int // frames below the offset threshold that end it
	riseAlpha float64

	floor    float64
	peak     float64
	pos      int // samples consumed in whole frames
	rem      []byte
	speaking bool
	run      int // consecutive frames above onset while not speaking
	runStart int // the sample where that run began
	segStart int
	lastEnd  int // the end of the last frame at or above the offset threshold
	hang     int
	segs     []Segment
	first    int // the start of the first segment; -1 before any
}

// NewDetector is a detector for PCM16 mono at rate Hz (rate <= 0:
// DefaultRate), its noise floor starting at p.InitFloorDB.
func NewDetector(rate int, p Params) *Detector {
	if rate <= 0 {
		rate = DefaultRate
	}
	p = p.Resolved()
	frame := max(rate*p.FrameMS/1000, 1)
	frames := func(ms int) int { return max((ms+p.FrameMS/2)/p.FrameMS, 1) }
	return &Detector{
		p:         p,
		rate:      rate,
		frame:     frame,
		onsetN:    frames(p.OnsetMS),
		hangN:     frames(p.HangoverMS),
		riseAlpha: 1 - math.Exp(-float64(p.FrameMS)/riseTauMS),
		floor:     p.InitFloorDB,
		peak:      minDB,
		first:     -1,
	}
}

// Write feeds PCM; whole frames are analysed and an incomplete one
// (including an odd byte) waits for the next Write.
func (d *Detector) Write(pcm []byte) {
	fb := d.frame * 2
	if len(d.rem) > 0 {
		k := min(fb-len(d.rem), len(pcm))
		d.rem = append(d.rem, pcm[:k]...)
		pcm = pcm[k:]
		if len(d.rem) < fb {
			return
		}
		d.step(frameDB(d.rem))
		d.rem = d.rem[:0]
	}
	for len(pcm) >= fb {
		d.step(frameDB(pcm[:fb]))
		pcm = pcm[fb:]
	}
	d.rem = append(d.rem, pcm...)
}

// step advances the detector by one frame of level e.
func (d *Detector) step(e float64) {
	start, end := d.pos, d.pos+d.frame
	d.pos = end
	d.peak = max(d.peak, e)
	if d.speaking {
		if e >= d.floor+d.p.OffsetDB && e > d.p.MinLevelDB {
			d.lastEnd, d.hang = end, 0
		} else if d.hang++; d.hang >= d.hangN {
			d.segs = append(d.segs, Segment{d.segStart, d.lastEnd})
			d.speaking, d.run = false, 0
		}
		if e < d.floor && e > digitalSilenceDB {
			d.floor += (e - d.floor) * fallRate
		}
		return
	}
	if e > d.floor+d.p.OnsetDB && e > d.p.MinLevelDB {
		if d.run == 0 {
			d.runStart = start
		}
		if d.run++; d.run >= d.onsetN {
			d.speaking = true
			d.segStart, d.lastEnd, d.hang = d.runStart, end, 0
			if d.first < 0 {
				d.first = d.runStart
			}
		}
		// A candidate onset does not raise the floor.
		return
	}
	d.run = 0
	switch {
	case e <= digitalSilenceDB:
	case e < d.floor:
		d.floor += (e - d.floor) * fallRate
	default:
		d.floor += (e - d.floor) * d.riseAlpha
	}
}

// Speaking reports whether the detector is in speech now.
func (d *Detector) Speaking() bool { return d.speaking }

// Heard reports whether speech started at least once.
func (d *Detector) Heard() bool { return d.first >= 0 }

// Segments are the closed segments, plus the open one ended at the current
// sample.
func (d *Detector) Segments() []Segment {
	s := slices.Clone(d.segs)
	if d.speaking {
		s = append(s, Segment{d.segStart, d.pos})
	}
	return s
}

// FloorDB is the noise floor estimate now.
func (d *Detector) FloorDB() float64 { return d.floor }

// LevelDB is the level of PCM16 (an even number of bytes) in dBFS, as the
// detector measures a frame: 10·log10(mean(x²)) with x in [-1, 1], at
// least -120.
func LevelDB(pcm []byte) float64 { return frameDB(pcm) }

// Merge is the speech either analysis found (of the same recording, say
// before and after a filter): their segments' union, in order. Its noise
// floor is a's; its peak the louder.
func Merge(a, b Analysis) Analysis {
	m := Analysis{FloorDB: a.FloorDB, PeakDB: max(a.PeakDB, b.PeakDB), Samples: max(a.Samples, b.Samples)}
	all := append(slices.Clone(a.Segments), b.Segments...)
	slices.SortFunc(all, func(x, y Segment) int { return x.Start - y.Start })
	for _, s := range all {
		if s.End <= s.Start {
			continue
		}
		if n := len(m.Segments); n > 0 && s.Start <= m.Segments[n-1].End {
			m.Segments[n-1].End = max(m.Segments[n-1].End, s.End)
			continue
		}
		m.Segments = append(m.Segments, s)
	}
	m.Speech = len(m.Segments) > 0
	return m
}
