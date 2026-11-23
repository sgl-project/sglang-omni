package platform

import (
	"math"
	"time"
)

// DoubleClick is how far apart in time and on each axis two presses make
// a double click.
type DoubleClick struct {
	Time   time.Duration
	DX, DY float64
}

// DefaultDoubleClick is the double click where the system tells none.
var DefaultDoubleClick = DoubleClick{Time: 500 * time.Millisecond, DX: 5, DY: 5}

// ClickCounter counts quick successive presses for a backend whose system
// reports no count with a press, by the system's double-click time and
// distance where it has them.
type ClickCounter struct {
	ms     uint32
	x, y   float64
	button int
	count  int
}

// Press counts a press of button at x, y at ms, the event's time in
// milliseconds, which may wrap around as the system's does: one more than
// the last press's count when that press was of the same button and
// within d of it; else 1. The event's time rather than now's keeps presses
// waiting for a busy main thread as far apart as they were made.
func (c *ClickCounter) Press(button int, x, y float64, ms uint32, d DoubleClick) int {
	if c.count > 0 && button == c.button && ms-c.ms <= uint32(d.Time.Milliseconds()) && math.Abs(x-c.x) <= d.DX && math.Abs(y-c.y) <= d.DY {
		c.count++
	} else {
		c.count = 1
	}
	c.ms, c.x, c.y, c.button = ms, x, y, button
	return c.count
}
