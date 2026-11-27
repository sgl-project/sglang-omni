package chart

import (
	"math"
	"sort"
	"time"
)

// Time maps times from From to To onto the plot, as d3's scaleTime and
// Swift Charts' scales of dates. Its ticks fall on round times of the
// times' location, a second, a quarter of an hour, a day, the first of a
// month, chosen for the span: midnights for a week, months for a year.
//
//	x := chart.NewTime(start, end)
//	ticks := x.Ticks(6)
type Time struct {
	From, To time.Time
	// Format labels the ticks; nil labels each with what changed since
	// a round time of the coarser unit, as d3's multi-scale format:
	// "2026", "Mar", "Mar 5", "14:00", "14:05:30".
	Format func(t time.Time) string
}

// NewTime returns a time scale from from to to.
func NewTime(from, to time.Time) Time { return Time{From: from, To: to} }

// At returns where t is along the plot.
func (s Time) At(t time.Time) float32 {
	span := s.To.Sub(s.From)
	if span == 0 {
		return 0.5
	}
	return float32(float64(t.Sub(s.From)) / float64(span))
}

// Span returns the stretch from a to b along the plot.
func (s Time) Span(a, b time.Time) Span { return Span{s.At(a), s.At(b)} }

// Invert returns the time at f along the plot.
func (s Time) Invert(f float32) time.Time {
	return s.From.Add(time.Duration(float64(f) * float64(s.To.Sub(s.From))))
}

// unit is a calendar unit of time ticks.
type unit int

const (
	unitMilli unit = iota
	unitSecond
	unitMinute
	unitHour
	unitDay
	unitWeek
	unitMonth
	unitYear
)

// interval is a step of ticks: every n units, about d long.
type interval struct {
	u unit
	n int
	d time.Duration
}

const (
	day   = 24 * time.Hour
	week  = 7 * day
	month = 30 * day
	year  = 365 * day
)

// intervals are d3-time's tick intervals, by their length.
var intervals = []interval{
	{unitSecond, 1, time.Second}, {unitSecond, 5, 5 * time.Second}, {unitSecond, 15, 15 * time.Second}, {unitSecond, 30, 30 * time.Second},
	{unitMinute, 1, time.Minute}, {unitMinute, 5, 5 * time.Minute}, {unitMinute, 15, 15 * time.Minute}, {unitMinute, 30, 30 * time.Minute},
	{unitHour, 1, time.Hour}, {unitHour, 3, 3 * time.Hour}, {unitHour, 6, 6 * time.Hour}, {unitHour, 12, 12 * time.Hour},
	{unitDay, 1, day}, {unitDay, 2, 2 * day},
	{unitWeek, 1, week},
	{unitMonth, 1, month}, {unitMonth, 3, 3 * month},
	{unitYear, 1, year},
}

// tickInterval returns the interval of about count ticks from lo to hi,
// as d3-time's tickInterval: the one closest to the span over count, in
// years beyond them and milliseconds below them.
func tickInterval(lo, hi time.Time, count int) interval {
	target := time.Duration(float64(hi.Sub(lo)) / float64(max(1, count)))
	i := sort.Search(len(intervals), func(i int) bool { return intervals[i].d > target })
	switch {
	case i == len(intervals):
		n := tickStep(float64(lo.Year()), float64(hi.Year())+float64(hi.YearDay())/366, float64(count))
		return interval{unitYear, max(1, int(math.Round(n))), year}
	case i == 0:
		n := tickStep(0, float64(hi.Sub(lo).Milliseconds()), float64(count))
		return interval{unitMilli, max(1, int(math.Round(n))), time.Millisecond}
	}
	if float64(target)/float64(intervals[i-1].d) < float64(intervals[i].d)/float64(target) {
		return intervals[i-1]
	}
	return intervals[i]
}

// floor returns the latest round time of the interval at or before t, in
// t's location: the field of the unit a multiple of n, as d3-time's every.
func (iv interval) floor(t time.Time) time.Time {
	y, mo, d := t.Date()
	h, mi, s := t.Clock()
	loc := t.Location()
	switch iv.u {
	case unitMilli:
		ms := t.Nanosecond() / 1e6
		return time.Date(y, mo, d, h, mi, s, (ms-ms%iv.n)*1e6, loc)
	case unitSecond:
		return time.Date(y, mo, d, h, mi, s-s%iv.n, 0, loc)
	case unitMinute:
		return time.Date(y, mo, d, h, mi-mi%iv.n, 0, 0, loc)
	case unitHour:
		return time.Date(y, mo, d, h-h%iv.n, 0, 0, 0, loc)
	case unitDay:
		return time.Date(y, mo, d-(d-1)%iv.n, 0, 0, 0, 0, loc)
	case unitWeek: // weeks start on Sunday, as d3's timeWeek
		return time.Date(y, mo, d-int(t.Weekday()), 0, 0, 0, 0, loc)
	case unitMonth:
		m := int(mo) - 1
		return time.Date(y, time.Month(m-m%iv.n+1), 1, 0, 0, 0, 0, loc)
	default:
		return time.Date(y-y%iv.n, 1, 1, 0, 0, 0, 0, loc)
	}
}

// next returns the round time of the interval after t, a round time.
func (iv interval) next(t time.Time) time.Time {
	switch iv.u {
	case unitMilli:
		return t.Add(time.Duration(iv.n) * time.Millisecond)
	case unitSecond, unitMinute:
		return t.Add(iv.d)
	case unitHour:
		y, mo, d := t.Date()
		return time.Date(y, mo, d, t.Hour()+iv.n, 0, 0, 0, t.Location())
	case unitDay:
		y, mo, d := t.Date()
		n := time.Date(y, mo, d+iv.n, 0, 0, 0, 0, t.Location())
		if n.Month() != mo { // days of a month count from its first
			n = time.Date(n.Year(), n.Month(), 1, 0, 0, 0, 0, t.Location())
		}
		return n
	case unitWeek:
		y, mo, d := t.Date()
		return time.Date(y, mo, d+7*iv.n, 0, 0, 0, 0, t.Location())
	case unitMonth:
		y, mo, _ := t.Date()
		return time.Date(y, mo+time.Month(iv.n), 1, 0, 0, 0, 0, t.Location())
	default:
		return time.Date(t.Year()+iv.n, 1, 1, 0, 0, 0, 0, t.Location())
	}
}

// Ticks returns about count round times within the domain.
func (s Time) Ticks(count int) []Tick {
	lo, hi := s.From, s.To
	if hi.Before(lo) {
		lo, hi = hi, lo
	}
	if count <= 0 || !lo.Before(hi) {
		return nil
	}
	iv := tickInterval(lo, hi, count)
	format := s.Format
	if format == nil {
		format = formatTime
	}
	var out []Tick
	t := iv.floor(lo)
	for !t.After(hi) && len(out) < 1000 {
		if !t.Before(lo) {
			out = append(out, Tick{At: s.At(t), Value: float64(t.UnixMilli()), Label: format(t)})
		}
		t = iv.next(t)
	}
	if s.To.Before(s.From) {
		for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
			out[i], out[j] = out[j], out[i]
		}
	}
	return out
}

// formatTime labels a tick with the finest unit it is not round in, as
// d3's multi-scale time format.
func formatTime(t time.Time) string {
	_, mo, d := t.Date()
	h, mi, s := t.Clock()
	switch {
	case t.Nanosecond() != 0:
		return t.Format(".000")
	case s != 0:
		return t.Format(":05")
	case mi != 0:
		return t.Format("15:04")
	case h != 0:
		return t.Format("15:00")
	case d != 1:
		return t.Format("Jan 2")
	case mo != 1:
		return t.Format("Jan")
	}
	return t.Format("2006")
}
