package layout

import (
	"math"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// speech is a few words' levels: quiet, a word, a pause, a louder word.
func speech() []float64 {
	var out []float64
	add := func(n int, db float64) {
		for range n {
			out = append(out, session.Level(db))
		}
	}
	add(20, -58)
	add(6, -22)
	add(5, -50)
	add(3, -14)
	add(4, -30)
	add(22, -58)
	return out
}

// playWave feeds levels to a WavePlay as a microphone hands them over,
// batch at a time and each batch up to late seconds after it was due,
// and asks the bars at every frame of a display of hz; each calls with
// the bars.
func playWave(levels []float64, batch int, late float64, hz int, each func(now time.Time, bars [session.Bars]float64, w *WavePlay)) {
	var w WavePlay
	start := time.Unix(1000, 0)
	at := func(seconds float64) time.Time { return start.Add(time.Duration(seconds * float64(time.Second))) }
	w.Feed("a", nil, 0, start)
	fed := 0
	frames := (len(levels)/WaveRate + 1) * hz
	for frame := range frames {
		now := float64(frame) / float64(hz)
		for fed+batch <= len(levels) {
			// The batch that ends at the level fed+batch is recorded
			// (fed+batch)/WaveRate seconds in; every third one comes late.
			due := float64(fed+batch) / WaveRate
			if (fed/batch)%3 == 2 {
				due += late
			}
			if due > now {
				break
			}
			fed += batch
			w.Feed("a", levels[max(0, fed-session.Bars):fed], fed, at(due))
		}
		each(at(now), w.Bars(at(now)), &w)
	}
}

// TestWavePlayGlides: whatever the batches, the bars move a little every
// frame and never jump, where bars taken as the levels come stand still
// between two batches and then move the batch's places at once.
func TestWavePlayGlides(t *testing.T) {
	levels := speech()
	// The most a bar may change in a frame at 60 a second: a third of the
	// largest step between two levels (0.82), and the catching up.
	const most = 0.82/3 + 0.12
	for _, c := range []struct {
		name  string
		batch int
		late  float64
	}{{"one by one", 1, 0}, {"two at a time", 2, 0}, {"two, some late", 2, 0.02}, {"four at a time", 4, 0}} {
		var last [session.Bars]float64
		var worst, lastPos float64
		still, frames := 0, 0
		playWave(levels, c.batch, c.late, 60, func(_ time.Time, bars [session.Bars]float64, w *WavePlay) {
			moved := false
			for i, v := range bars {
				worst = max(worst, math.Abs(v-last[i]))
				moved = moved || v != last[i]
			}
			if w.pos < lastPos {
				t.Fatalf("%s: the bars went back: %v after %v", c.name, w.pos, lastPos)
			}
			if w.pos > float64(w.n) {
				t.Fatalf("%s: bar %v is past the newest level %d", c.name, w.pos, w.n)
			}
			// Between the first word and the last one's end.
			if w.n > 22 && w.n < 40 {
				frames++
				if !moved {
					still++
				}
			}
			last, lastPos = bars, w.pos
		})
		if worst > most {
			t.Errorf("%s: a bar jumped by %.2f in a frame, want at most %.2f", c.name, worst, most)
		}
		if still > frames/10 {
			t.Errorf("%s: the bars stood still in %d of %d frames while a word passed", c.name, still, frames)
		}
	}
}

// TestWavePlayIsTheLevels: once the levels are played out the bars are
// the levels' own, and on the way the newest bar is at most a batch (and
// a little) behind the newest level.
func TestWavePlayIsTheLevels(t *testing.T) {
	levels := speech()
	for _, batch := range []int{1, 2, 4} {
		var end [session.Bars]float64
		var behind float64
		playWave(levels, batch, 0, 60, func(_ time.Time, bars [session.Bars]float64, w *WavePlay) {
			end = bars
			if w.n > 20 {
				behind = max(behind, float64(w.n)-w.pos)
			}
			for i, v := range bars {
				if v < 0 || v > 1 {
					t.Fatalf("bar %d = %v", i, v)
				}
			}
		})
		// WaveBars' follower, over the whole recording rather than from
		// the oldest bar that shows.
		fed := len(levels) / batch * batch
		var want [session.Bars]float64
		s := 0.0
		for i, v := range levels[:fed] {
			s = waveFollow(s, v)
			if at := i - (fed - session.Bars); at >= 0 {
				want[at] = s
			}
		}
		for i := range end {
			if math.Abs(end[i]-want[i]) > 1e-3 {
				t.Errorf("batches of %d: played out, bar %d = %.4f, want the level's %.4f", batch, i, end[i], want[i])
			}
		}
		if behind > float64(batch)+0.5 {
			t.Errorf("batches of %d: the newest bar was %.2f levels behind", batch, behind)
		}
	}
}

// TestWavePlayStarts: another recording starts from its own levels, and
// one met half way (more new levels than a snapshot keeps) shows what the
// snapshot has at once.
func TestWavePlayStarts(t *testing.T) {
	var w WavePlay
	now := time.Unix(1000, 0)
	loud := make([]float64, session.Bars)
	for i := range loud {
		loud[i] = session.Level(-15)
	}
	w.Feed("a", loud, 400, now)
	if got := w.Bars(now); got != WaveBars(loud) {
		t.Errorf("met half way: %v, want %v", got, WaveBars(loud))
	}
	w.Feed("b", nil, 0, now.Add(time.Second))
	if got := w.Bars(now.Add(time.Second)); got != ([session.Bars]float64{}) {
		t.Errorf("another recording kept the last one's bars: %v", got)
	}
}
