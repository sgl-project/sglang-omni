package meeting

import (
	"slices"
	"time"
)

// Quiet (Snapshot.Quiet) without Silero: a track hears someone when its
// level stays SpeechOverFloorDB above the room's floor for speechHoldMin,
// in all, through dips of up to speechGapMax; a level counts only next to
// another one over the floor, so a key click never does.
// The floor is a low percentile of the last quietFloorWindow's levels:
// the pauses between words keep it at the room's noise while someone
// talks, and unlike the quietest level it does not sink with the noise's
// jitter. A steady noise
// becomes the floor, however loud (11-25: a MacBook microphone averaged
// -32 to -42 dBFS with nobody speaking, and the energy detector, which
// never raises its floor while it thinks someone speaks, took -35 dBFS of
// noise for half a minute of speech); a key click is too short. With
// Silero the live recognition's detector decides instead (noteSpeech).
// SpeechOverFloorDB keeps soft speech heard (11-25, replayed on the
// helper's 50 ms levels: speech about 7 dB over −35 dBFS noise went up to
// 8.5 s unheard at 10 dB, 3 s at 8 dB and at 6 dB) while noise is not:
// −38 dBFS noise jittering 3 dB per level was heard 352 times in 1000
// minutes at 6 dB, 47 at 7 dB, 7 at 8 dB.
const (
	SpeechOverFloorDB = 8
	quietFloorWindow  = 30 * time.Second
	speechHoldMin     = 250 * time.Millisecond
	speechGapMax      = 300 * time.Millisecond // a dip between syllables
	quietFloorPct     = 0.2
	quietFloorMinDB   = -96 // digital silence: as quiet as a floor gets
)

type levelAt struct {
	at time.Duration
	db float64
}

// roomFloor follows one track's levels.
type roomFloor struct {
	recent   []levelAt     // the last quietFloorWindow, oldest first
	prev     time.Duration // the last level's time: each level lasts until the next
	over     time.Duration // the time over the floor, dips of up to speechGapMax included
	last     time.Duration // when the last level over the floor ended
	overOnce bool          // a level was ever counted over the floor
	prevOver bool          // the level before was over the floor
}

// heard takes the level db at the meeting's time at and reports whether
// someone is speaking.
func (f *roomFloor) heard(at time.Duration, db float64) bool {
	db = max(db, quietFloorMinDB)
	dt := max(at-f.prev, 0)
	f.prev = at
	cut := 0
	for cut < len(f.recent) && at-f.recent[cut].at > quietFloorWindow {
		cut++
	}
	f.recent = append(f.recent[cut:], levelAt{at, db})
	levels := make([]float64, len(f.recent))
	for i, l := range f.recent {
		levels[i] = l.db
	}
	slices.Sort(levels)
	floor := levels[int(quietFloorPct*float64(len(levels)-1))]
	if db < floor+SpeechOverFloorDB {
		f.prevOver = false
		return false
	}
	if !f.prevOver { // one level alone is a click: a run of two is a sound
		f.prevOver = true
		return false
	}
	if !f.overOnce || at-dt-f.last > speechGapMax {
		f.over = 0 // a pause: speech starts over
	}
	f.overOnce, f.over, f.last = true, f.over+dt, at
	return f.over >= speechHoldMin
}
