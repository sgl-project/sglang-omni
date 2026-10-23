package dsp

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
)

// A variant of the speech sample: what a microphone might give.
type variantSpec struct {
	name  string
	gain  float64 // on the sample, dB
	noise func(n int) []float64
}

const (
	padBefore = 1.5 // seconds of room around the sample
	padAfter  = 1.5
	roomDB    = -66 // the room's noise (white), dBFS
)

// hum is mains hum: 50 Hz at db and its second harmonic 15 dB under.
func hum(n int, db float64) []float64 {
	return add(sine(n, 50, db, 0.3), sine(n, 100, db-15, 1.1))
}

var variants = []variantSpec{
	{"original", 0, nil},
	{"quiet (-20 dB)", -20, nil},
	{"hum 50+100 Hz (-35 dBFS)", 0, func(n int) []float64 { return hum(n, -35) }},
	{"quiet + hum -45 dBFS", -20, func(n int) []float64 { return hum(n, -45) }},
	{"white noise -45 dBFS", 0, func(n int) []float64 { return white(n, -45, 3) }},
	{"fan noise -45 dBFS", 0, func(n int) []float64 { return fan(n, -45, 4) }},
	{"quiet + fan -55 dBFS", -20, func(n int) []float64 { return fan(n, -55, 5) }},
	{"loud, close (+8 dB, clipped)", 8, nil},
}

// build is the variant: the sample at its gain with padBefore and padAfter
// of room noise around it and its noise under all of it, and the offset
// of the sample in it.
func (v variantSpec) build(s []float64) ([]byte, int) {
	x := pad(gainOf(s, v.gain), padBefore, padAfter)
	x = add(x, white(len(x), roomDB, 99))
	if v.noise != nil {
		x = add(x, v.noise(len(x)))
	}
	return toPCM(x), sec(padBefore)
}

// levels are a recording's statistics: its RMS and peak, its speech's
// level (the frames vad.Analyze put in speech that count) and its noise
// floor (the RMS of the frames it left out).
type levels struct{ rms, peak, speech, floor float64 }

func measureLevels(pcm []byte) levels {
	an := vad.Analyze(pcm, rate, vad.Params{})
	sp, _ := Measure(pcm, rate, an, vad.Params{})
	s := toFloat(pcm)
	var quiet []float64
	prev := 0
	for _, g := range an.Segments {
		quiet = append(quiet, s[prev:max(g.Start, prev)]...)
		prev = g.End
	}
	quiet = append(quiet, s[min(prev, len(s)):]...)
	return levels{rmsDB(s), peakOf(s), sp, rmsDB(quiet)}
}

// cutWords are the frames of speech in the sample (within 30 dB of its
// loudest) that a compaction of the variant left out.
func cutWords(s []float64, at int, spans []vad.Span) (cut, total int) {
	pcm := toPCM(s)
	frame := rate / 50
	loud := minDB
	var lv []float64
	for f := 0; (f+1)*frame <= len(s); f++ {
		e := vad.LevelDB(pcm[f*frame*2 : (f+1)*frame*2])
		lv = append(lv, e)
		loud = max(loud, e)
	}
	for f, e := range lv {
		if e < loud-30 {
			continue
		}
		total++
		a, b := at+f*frame, at+(f+1)*frame
		in := false
		for _, sp := range spans {
			if a >= sp.Start && b <= sp.End {
				in = true
			}
		}
		if !in {
			cut++
		}
	}
	return cut, total
}

// TestCompareVariants processes each variant as a file and as a stream
// and reports the levels before and after, what the voice activity
// detection keeps of it before and after, and the gains. The table goes to
// the log (go test -v -run TestCompareVariants); VOXTNEW_DSP_OUT=dir also
// writes each variant before and after as WAV files.
func TestCompareVariants(t *testing.T) {
	s := loadSample(t)
	dir := os.Getenv("VOXTNEW_DSP_OUT")
	var tab strings.Builder
	fmt.Fprintf(&tab, "\n| variant | RMS | peak | speech | floor | VAD kept | speech frames cut | gain (file) | gain (stream, end; min–max) | limited |\n|---|---|---|---|---|---|---|---|---|---|\n")
	for _, v := range variants {
		pcm, at := v.build(s)
		f := ProcessFile(pcm, rate, Params{}, vad.Params{})
		before, after := measureLevels(pcm), measureLevels(f.PCM)
		raw := vad.Analyze(pcm, rate, vad.Params{})
		keptRaw := vad.Compact(pcm, rate, raw, vad.Params{})
		keptNew := vad.Compact(f.PCM, rate, f.Analysis, vad.Params{})
		_, lev := stream(pcm, 1600, Params{}, vad.Params{}, true)
		sr := lev.Report()
		r := f.Report
		cutRaw, total := cutWords(gainOf(s, v.gain), at, keptRaw.Spans)
		cut, _ := cutWords(gainOf(s, v.gain), at, keptNew.Spans)
		fmt.Fprintf(&tab, "| %s | %.1f → %.1f | %.1f → %.1f | %.1f → %.1f | %.1f → %.1f | %.2f s → %.2f s | %d → %d of %d | %+.1f dB | %+.1f dB (%+.1f…%+.1f) | %d |\n",
			v.name, before.rms, after.rms, before.peak, after.peak, before.speech, after.speech, before.floor, after.floor,
			keptRaw.Kept.Seconds(), keptNew.Kept.Seconds(), cutRaw, cut, total, r.GainDB, sr.GainDB, sr.MinGainDB, sr.MaxGainDB, r.Limited)

		// No word is cut, and the output never clips.
		if cut > 0 {
			t.Errorf("%s: %d of %d speech frames cut", v.name, cut, total)
		}
		if after.peak > -1+0.01 {
			t.Errorf("%s: peak %.2f dBFS after", v.name, after.peak)
		}
		// A stream ends near the file's gain.
		if d := math.Abs(sr.GainDB - r.GainDB); d > 2 {
			t.Errorf("%s: stream gain %+.1f, file %+.1f", v.name, sr.GainDB, r.GainDB)
		}
		if dir != "" {
			name := strings.NewReplacer(" ", "-", "(", "", ")", "", "+", "", ",", "").Replace(v.name)
			_ = os.WriteFile(dir+"/"+name+".wav", asr.WAV(pcm, rate), 0o644)
			_ = os.WriteFile(dir+"/"+name+".processed.wav", asr.WAV(f.PCM, rate), 0o644)
		}
	}
	t.Log(tab.String())
}
