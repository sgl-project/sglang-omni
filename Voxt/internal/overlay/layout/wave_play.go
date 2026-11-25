package layout

import (
	"math"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// WaveRate is how many levels a second of recording has (the helper's
// capture.level).
const WaveRate = 20

const (
	// waveKeep is how many bars a WavePlay keeps: the ones that show and
	// the ones still to come in under the newest.
	waveKeep = session.Bars + waveBehind + 1
	// waveBehind is the most levels the newest bar may be behind the
	// newest level.
	waveBehind = 7
	// waveBatchGap parts two batches of levels: the ones that come closer
	// together came in one.
	waveBatchGap = 20 * time.Millisecond
	// waveSlack is how much of a level the newest bar keeps in hand for a
	// batch that comes a little late.
	waveSlack = 0.3
	// waveEase is how quickly the bars take up a batch that came early or
	// late, in seconds.
	waveEase = 0.12
)

// WavePlay plays a recording's levels out between their arrivals, so that
// the waveform glides rather than steps: the levels come about WaveRate a
// second, but several at a time (a microphone hands its audio over a
// tenth of a second at once), and bars that took them as they came would
// stand still and then jump two places. It shows the same bars as
// WaveBars, each level's bar passing from one place to the next in the
// time the level took to record, and nothing that was not recorded: the
// bars between two levels are the line between them. The newest bar is at
// most one batch behind the newest level.
//
// The zero value is ready. Feed it each snapshot's levels as they come,
// and ask Bars at every frame.
type WavePlay struct {
	id  string
	n   int               // the levels fed so far
	env [waveKeep]float64 // their bars (waveFollow): the last is the n-th level's
	s   float64           // the newest of them
	// pos is the level under the newest bar, at most n, as of at.
	pos float64
	at  time.Time
	// came is when the last batch of levels started to come, batch how
	// many it brought so far, and most how many a batch brings.
	came        time.Time
	batch, most float64
}

// Feed takes a snapshot's levels (session.Snapshot.Levels, of count in
// all) of the recording id, at now. Another recording, or more new
// levels than the snapshot keeps, starts the bars again from what it has.
func (w *WavePlay) Feed(id string, levels []float64, count int, now time.Time) {
	fresh := count - w.n
	if id != w.id || fresh < 0 || fresh > len(levels) {
		*w = WavePlay{id: id, n: count, pos: float64(count), at: now, came: now}
		for _, v := range levels {
			w.push(v)
		}
		return
	}
	if fresh == 0 {
		return
	}
	for _, v := range levels[len(levels)-fresh:] {
		w.push(v)
	}
	w.n = count
	if now.Sub(w.came) > waveBatchGap {
		// The batch before is whole: a larger one counts at once, a
		// smaller one little by little (one late batch is not the rule).
		if w.batch >= w.most {
			w.most = w.batch
		} else {
			w.most += (w.batch - w.most) * 0.25
		}
		w.came, w.batch = now, 0
	}
	w.batch += float64(fresh)
}

func (w *WavePlay) push(level float64) {
	w.s = waveFollow(w.s, level)
	copy(w.env[:], w.env[1:])
	w.env[waveKeep-1] = w.s
}

// Bars are the bars at now, 0–1, the newest on the right.
func (w *WavePlay) Bars(now time.Time) Wave {
	w.advance(now)
	var out Wave
	// The bar i shows the level pos − (Bars−1−i), which is at env's
	// last − (n − that).
	first := float64(waveKeep-1) - float64(w.n) + w.pos - float64(session.Bars-1)
	for i := range out {
		at := first + float64(i)
		lo := math.Floor(at)
		a, b := w.bar(int(lo)), w.bar(int(lo)+1)
		out[i] = a + (b-a)*(at-lo)
	}
	return out
}

func (w *WavePlay) bar(i int) float64 {
	if i < 0 {
		return 0
	}
	return w.env[min(i, waveKeep-1)]
}

// advance moves pos to now: on by the time that passed, towards where
// the last batch puts it. A batch of k levels is k/WaveRate seconds of
// sound, and the next one comes when as much again is recorded: so the
// newest bar starts the batch's length (and waveSlack) behind and reaches
// the newest level as the next batch is due.
func (w *WavePlay) advance(now time.Time) {
	dt := now.Sub(w.at).Seconds()
	if dt <= 0 {
		return
	}
	w.at = now
	n := float64(w.n)
	lead := min(max(w.most, w.batch, 1)+waveSlack, waveBehind)
	want := min(n, n-lead+now.Sub(w.came).Seconds()*WaveRate)
	pos := w.pos + dt*WaveRate
	pos += (want - pos) * (1 - math.Exp(-dt/waveEase))
	// Never back, never past the newest level, never out of what is kept.
	w.pos = min(max(pos, w.pos, n-waveBehind), n)
}
