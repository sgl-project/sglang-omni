package dsp

import (
	"encoding/binary"
	"math"
	"slices"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
)

// fullScale is PCM16's full scale: a sample of x is x/fullScale in [-1, 1].
const fullScale = 32768.0

// limiter is a peak limiter without look-ahead: a sample that would go
// past the ceiling sets the gain so it lands on it, and the gain then
// recovers toward 1 with the release's time constant. No sample goes out
// louder than the ceiling.
type limiter struct {
	ceil, rel, g float64
	n            int // samples turned down
}

func newLimiter(rate int, ceilDB float64, releaseMS int) limiter {
	return limiter{
		ceil: fullScale * dbToGain(ceilDB),
		rel:  math.Exp(-1000 / (float64(releaseMS) * float64(rate))),
		g:    1,
	}
}

func (l *limiter) do(x float64) float64 {
	l.g = 1 - (1-l.g)*l.rel
	if a := math.Abs(x); a*l.g > l.ceil {
		l.g = l.ceil / a
		l.n++
	}
	return x * l.g
}

func sample(pcm []byte, i int) float64 {
	return float64(int16(binary.LittleEndian.Uint16(pcm[i*2:])))
}

func put(pcm []byte, i int, v float64) {
	s, _ := toInt16(v)
	binary.LittleEndian.PutUint16(pcm[i*2:], uint16(s))
}

// peakDB is the highest absolute sample of PCM16 in dBFS.
func peakDB(pcm []byte) float64 {
	var m float64
	for i := range len(pcm) / 2 {
		m = max(m, math.Abs(sample(pcm, i)))
	}
	if m == 0 {
		return minDB
	}
	return 20 * math.Log10(m/fullScale)
}

func powerDB(p float64) float64 {
	if p <= 0 {
		return minDB
	}
	return max(10*math.Log10(p), minDB)
}

// isSpeechFrame says whether a frame inside speech counts toward its level:
// at or over the detector's threshold to stay in speech (the quiet frames
// of the hang-over and of short pauses do not).
func isSpeechFrame(e, floorDB float64, vp vad.Params) bool {
	return e >= floorDB+vp.OffsetDB && e > vp.MinLevelDB
}

// speechPercentile is the percentile of the frames' peaks taken as the
// speech's peak: a lone click does not hold the gain down.
const speechPercentile = 0.99

// Measure is the speech's level in a recording vad.Analyze found speech
// in: the power mean of the frames inside its segments that count
// (isSpeechFrame) and the 99th percentile of their peaks; -120 for both
// when none counts.
func Measure(pcm []byte, rate int, an vad.Analysis, vp vad.Params) (speechDB, peak float64) {
	if rate <= 0 {
		rate = DefaultRate
	}
	vp = vp.Resolved()
	frame := max(rate*vp.FrameMS/1000, 1)
	n := len(pcm) / 2
	var sum float64
	var peaks []float64
	for _, s := range an.Segments {
		for at := max(s.Start, 0); at+frame <= min(s.End, n); at += frame {
			f := pcm[at*2 : (at+frame)*2]
			e := vad.LevelDB(f)
			if !isSpeechFrame(e, an.FloorDB, vp) {
				continue
			}
			sum += math.Pow(10, e/10)
			peaks = append(peaks, peakDB(f))
		}
	}
	if len(peaks) == 0 {
		return minDB, minDB
	}
	slices.Sort(peaks)
	return powerDB(sum / float64(len(peaks))), peaks[int(float64(len(peaks)-1)*speechPercentile)]
}

// File is a whole recording processed for a file transcription.
type File struct {
	PCM []byte // the processed audio, as long as the original
	// Analysis is the speech vad.Analyze found in the original or in the
	// filtered audio (vad.Merge): what to keep when trimming PCM. The
	// filter lowers a hum's floor, so it may find speech the original
	// hid; and where a hum made the original all speech, the original
	// keeps it all: the filter never makes the trim cut more.
	Analysis vad.Analysis
	// Filtered is vad.Analyze's of the filtered audio alone (before the
	// gain), whose speech the gain is measured on.
	Filtered vad.Analysis
	Report   Report
}

// ProcessFile processes a whole recording (PCM16 mono at rate Hz;
// rate <= 0: 16 kHz): the high-pass filter, the voice activity detection
// on the filtered audio (and on the original: File.Analysis), one gain
// for the whole recording from its speech (Params.GainFor; none without
// speech) and the limiter. pcm is not changed. A whole recording gets one
// gain rather than a following one: the speech model hears it at once,
// so the gain can be measured on all of it and cannot pump.
func ProcessFile(pcm []byte, rate int, p Params, vp vad.Params) File {
	if rate <= 0 {
		rate = DefaultRate
	}
	p = p.Resolved()
	var rep Report
	out := pcm
	if !p.HighPassOff {
		hp := NewHighPass(rate, p.HighPassHz)
		out = hp.Process(pcm)
		rep.HighPass, rep.Clipped = true, hp.Clipped
	}
	an := vad.Analyze(out, rate, vp)
	both := an
	if rep.HighPass {
		both = vad.Merge(an, vad.Analyze(pcm, rate, vp))
	}
	rep.FloorDB = an.FloorDB
	rep.SpeechDB, rep.PeakDB = minDB, minDB
	if an.Speech {
		rep.SpeechDB, rep.PeakDB = Measure(out, rate, an, vp)
	}
	if !p.LevelOff {
		rep.Level = true
		rep.GainDB = p.GainFor(rep.SpeechDB, an.FloorDB, rep.PeakDB)
		rep.MinGainDB, rep.MaxGainDB = rep.GainDB, rep.GainDB
		g := dbToGain(rep.GainDB)
		lim := newLimiter(rate, p.CeilingDB, p.ReleaseMS)
		dst := make([]byte, len(pcm)/2*2)
		for i := range len(dst) / 2 {
			put(dst, i, lim.do(sample(out, i)*g))
		}
		out, rep.Limited = dst, lim.n
	}
	if !rep.HighPass && !rep.Level {
		out = append([]byte(nil), pcm[:len(pcm)/2*2]...)
	}
	return File{PCM: out, Analysis: both, Filtered: an, Report: rep}
}

// Leveler is the level stage of a stream (the speech model hears the
// recording as it is spoken): it follows the speech's level with its own
// voice activity detector and moves the gain slowly toward
// Params.GainFor, then limits. It measures speech only: in silence and
// pauses the gain holds, so it never creeps up on the noise.
//
// Each chunk is measured before it is turned up or down, so a chunk sees
// its own speech: the first one with speech (the gate's opening: what came
// before the onset and the onset) sets the first gain at once (over
// 10 ms), and the speech after it is not heard first too quiet. From then
// on the gain follows with RiseMS and FallMS, and the level it follows is
// the mean of the first second of speech, then averaged over TrackMS.
// Not safe for concurrent use.
type Leveler struct {
	p                Params
	vp               vad.Params
	rate, frame      int
	det              *vad.Detector
	pend             []byte
	recent           []float64 // the last frames' levels while not in speech, newest last
	recentPk         []float64 // and their peaks
	onsetN           int
	speaking         bool
	n                int     // speech frames measured
	initN            int     // frames averaged before tracking
	pow              float64 // the speech's mean power
	track, peakFall  float64 // per frame
	peak             float64 // the speech's peak (dBFS), falling slowly
	want             float64 // the gain it goes toward
	g                float64 // the gain now
	started          bool    // a gain was set from speech
	rise, fall, fast float64 // per-sample smoothing
	lim              limiter
	rep              Report
	hasMin           bool
}

// minSpeechFrames are the frames of speech measured before a gain.
const minSpeechFrames = 3

// NewLeveler is a level stage for PCM16 mono at rate Hz (rate <= 0:
// 16 kHz), detecting speech with vp.
func NewLeveler(rate int, p Params, vp vad.Params) *Leveler {
	if rate <= 0 {
		rate = DefaultRate
	}
	p = p.Resolved()
	vp = vp.Resolved()
	frameMS := float64(vp.FrameMS)
	coef := func(ms float64) float64 { return 1 - math.Exp(-1000/(ms*float64(rate))) }
	return &Leveler{
		p:        p,
		vp:       vp,
		rate:     rate,
		frame:    max(rate*vp.FrameMS/1000, 1),
		det:      vad.NewDetector(rate, vp),
		onsetN:   max((vp.OnsetMS+vp.FrameMS/2)/vp.FrameMS, 1),
		initN:    max(int(1000/frameMS), 1),
		track:    1 - math.Exp(-frameMS/float64(p.TrackMS)),
		peakFall: frameMS / 1000, // 1 dB per second of speech
		peak:     minDB,
		rise:     coef(float64(p.RiseMS)),
		fall:     coef(float64(p.FallMS)),
		fast:     coef(10),
		lim:      newLimiter(rate, p.CeilingDB, p.ReleaseMS),
		rep:      Report{Level: true, SpeechDB: minDB, PeakDB: minDB, FloorDB: vp.InitFloorDB},
	}
}

// Process measures chunk (PCM16 mono; an odd last byte is dropped), then
// returns it turned up or down and limited, as a new slice.
func (l *Leveler) Process(chunk []byte) []byte {
	l.measure(chunk)
	fast := false
	if !l.started && l.n >= minSpeechFrames {
		l.started, fast = true, true
	}
	n := len(chunk) / 2
	out := make([]byte, n*2)
	for i := range n {
		c := l.rise
		switch {
		case fast:
			c = l.fast
		case l.want < l.g:
			c = l.fall
		}
		l.g += (l.want - l.g) * c
		put(out, i, l.lim.do(sample(chunk, i)*dbToGain(l.g)))
	}
	if l.started {
		if !l.hasMin {
			l.rep.MinGainDB, l.rep.MaxGainDB, l.hasMin = l.g, l.g, true
		}
		l.rep.MinGainDB, l.rep.MaxGainDB = min(l.rep.MinGainDB, l.g), max(l.rep.MaxGainDB, l.g)
	}
	return out
}

// measure runs the detector over chunk's whole frames and updates the
// speech's level, its peak and the gain to go toward.
func (l *Leveler) measure(chunk []byte) {
	l.pend = append(l.pend, chunk...)
	fb := l.frame * 2
	off := 0
	for ; off+fb <= len(l.pend); off += fb {
		f := l.pend[off : off+fb]
		e, pk := vad.LevelDB(f), peakDB(f)
		l.det.Write(f)
		sp := l.det.Speaking()
		floor := l.det.FloorDB()
		if sp && !l.speaking {
			// The frames of the onset, before the detector was sure.
			for i, re := range l.recent {
				if isSpeechFrame(re, floor, l.vp) {
					l.add(re, l.recentPk[i])
				}
			}
		}
		l.speaking = sp
		if sp {
			l.recent, l.recentPk = l.recent[:0], l.recentPk[:0]
			if isSpeechFrame(e, floor, l.vp) {
				l.add(e, pk)
			}
		} else {
			l.recent, l.recentPk = append(l.recent, e), append(l.recentPk, pk)
			if len(l.recent) > l.onsetN {
				l.recent, l.recentPk = l.recent[1:], l.recentPk[1:]
			}
		}
	}
	l.pend = l.pend[:copy(l.pend, l.pend[off:])]
	l.rep.FloorDB = l.det.FloorDB()
	if l.n >= minSpeechFrames {
		l.rep.SpeechDB, l.rep.PeakDB = powerDB(l.pow), l.peak
		l.want = l.p.GainFor(l.rep.SpeechDB, l.rep.FloorDB, l.peak)
	}
}

// add takes a speech frame of level e and peak pk.
func (l *Leveler) add(e, pk float64) {
	p := math.Pow(10, e/10)
	l.n++
	if l.n <= l.initN {
		l.pow += (p - l.pow) / float64(l.n)
	} else {
		l.pow += (p - l.pow) * l.track
	}
	l.peak = max(pk, l.peak-l.peakFall)
}

// Report says what the stage did so far.
func (l *Leveler) Report() Report {
	r := l.rep
	r.GainDB = l.g
	r.Limited = l.lim.n
	return r
}

// GainDB is the gain now.
func (l *Leveler) GainDB() float64 { return l.g }
