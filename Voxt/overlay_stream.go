package main

// The pace of an answer streaming into the card (#3836): a fast model
// sends many words a second, and the card jumped on every chunk. The card
// shows the draft at a steady pace instead, a little behind the model,
// and grows in steps of a few lines (cardBodyHeight). When the model
// ends, the rest of its answer streams on, catching up within about a
// second, instead of showing at once.

import (
	"math"
	"time"
)

const (
	// revealFrame is how often the shown text moves on.
	revealFrame = 33 * time.Millisecond
	// The pace, in runes a second: what is left to show over
	// revealCatchUp, between revealMin and revealMax.
	revealMin, revealMax = 30.0, 160.0
	revealCatchUp        = time.Second
)

// revealRunes is how many runes of a total-rune draft show dt after shown
// did: at least one more while some are left, all of a draft that got
// shorter. Once the model ended (not live) no cap holds the rest back.
func revealRunes(shown, total int, dt time.Duration, live bool) int {
	left := total - shown
	if left <= 0 {
		return total
	}
	rate := min(max(float64(left)/revealCatchUp.Seconds(), revealMin), revealMax)
	if !live {
		rate = max(2*float64(left)/revealCatchUp.Seconds(), revealMax) // the rest in about a second
	}
	return min(total, shown+max(1, int(math.Round(rate*dt.Seconds()))))
}
