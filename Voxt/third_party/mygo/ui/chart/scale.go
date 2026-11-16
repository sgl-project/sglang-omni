package chart

import (
	"math"
	"strconv"
)

// Span is a stretch of a chart's plot along one axis, in fractions of it:
// from the left edge (0) to the right (1) along x, and from the bottom (0)
// to the top (1) along y, the way values grow. Scales return spans for the
// marks: a bar's band along x, its stretch from 0 to its value along y.
type Span struct{ Lo, Hi float32 }

// Mid returns the middle of the span.
func (s Span) Mid() float32 { return (s.Lo + s.Hi) / 2 }

// Tick is a value a scale marks along an axis: where it is, in a fraction
// of the plot, and its label.
type Tick struct {
	At    float32
	Value float64
	Label string
}

// Linear maps numbers from Lo to Hi onto the plot, Lo at its start, as
// d3's scaleLinear and Swift Charts' automatic scales of numbers. Values
// outside the domain go outside the plot unless Clamp says otherwise.
//
//	y := chart.NewLinear(0, 87).Nice(10) // 0 to 90, ticks every 10
//	bar := y.Span(0, 42)
type Linear struct {
	Lo, Hi float64
	// Clamp keeps values outside the domain at its ends.
	Clamp bool
	// Format labels the ticks; nil labels them with as many decimals as
	// their step needs.
	Format func(v float64) string
}

// NewLinear returns a linear scale from lo to hi.
func NewLinear(lo, hi float64) Linear { return Linear{Lo: lo, Hi: hi} }

// Extent returns the smallest and the largest of values, the domain of a
// scale showing them, and 0, 1 for none. Include 0 for bars and areas:
//
//	lo, hi := chart.Extent(sales...)
//	y := chart.NewLinear(min(lo, 0), hi).Nice(5)
func Extent(values ...float64) (lo, hi float64) {
	if len(values) == 0 {
		return 0, 1
	}
	lo, hi = math.Inf(1), math.Inf(-1)
	for _, v := range values {
		if math.IsNaN(v) {
			continue
		}
		lo, hi = min(lo, v), max(hi, v)
	}
	if lo > hi {
		return 0, 1
	}
	return lo, hi
}

// At returns where v is along the plot.
func (s Linear) At(v float64) float32 {
	if s.Hi == s.Lo {
		return 0.5
	}
	f := (v - s.Lo) / (s.Hi - s.Lo)
	if s.Clamp {
		f = max(0, min(1, f))
	}
	return float32(f)
}

// Span returns the stretch from a to b along the plot, as a bar's from 0
// to its value.
func (s Linear) Span(a, b float64) Span { return Span{s.At(a), s.At(b)} }

// Invert returns the value at f along the plot, as where the pointer is.
func (s Linear) Invert(f float32) float64 {
	v := s.Lo + float64(f)*(s.Hi-s.Lo)
	if s.Clamp {
		v = max(min(s.Lo, s.Hi), min(max(s.Lo, s.Hi), v))
	}
	return v
}

// Nice extends the domain to round values, those of about count ticks, as
// d3's nice: 0 to 87 becomes 0 to 90 for 10 ticks, and 0 to 100 for 5.
func (s Linear) Nice(count int) Linear {
	lo, hi := s.Lo, s.Hi
	reverse := hi < lo
	if reverse {
		lo, hi = hi, lo
	}
	var prev float64
	for range 10 {
		step := tickIncrement(lo, hi, float64(count))
		if step == prev {
			break
		}
		switch {
		case step > 0:
			lo, hi = math.Floor(lo/step)*step, math.Ceil(hi/step)*step
		case step < 0:
			lo, hi = math.Ceil(lo*step)/step, math.Floor(hi*step)/step
		default:
			return s
		}
		prev = step
	}
	if reverse {
		lo, hi = hi, lo
	}
	s.Lo, s.Hi = lo, hi
	return s
}

// Ticks returns about count round values within the domain, 1, 2 or 5
// times a power of ten apart, as d3's ticks and Swift Charts' automatic
// axis marks: 0, 20, 40, 60, 80 and 100 for 0 to 100 and 5.
func (s Linear) Ticks(count int) []Tick {
	values := ticks(s.Lo, s.Hi, float64(count))
	format := s.Format
	if format == nil {
		step := math.Abs(tickStep(s.Lo, s.Hi, float64(count)))
		decimals := 0
		if step > 0 && step < 1 {
			decimals = max(0, int(-math.Floor(math.Log10(step)+1e-9)))
			if step*math.Pow10(decimals) != math.Round(step*math.Pow10(decimals)) {
				decimals++
			}
		}
		format = func(v float64) string {
			if v == 0 {
				v = 0 // no -0
			}
			return strconv.FormatFloat(v, 'f', decimals, 64)
		}
	}
	out := make([]Tick, len(values))
	for i, v := range values {
		out[i] = Tick{At: s.At(v), Value: v, Label: format(v)}
	}
	return out
}

// The tick spacing that d3-array's ticks uses: steps of 1, 2 or 5 times a
// power of ten, the closest to (hi-lo)/count.
var (
	e10 = math.Sqrt(50)
	e5  = math.Sqrt(10)
	e2  = math.Sqrt(2)
)

// tickSpec returns the first and last tick's index and the step between
// ticks, negative for the inverse of the step (1/-inc) when it is below 1,
// so that ticks such as 0.3 come out exact.
func tickSpec(lo, hi, count float64) (i1, i2, inc float64) {
	step := (hi - lo) / max(0, count)
	power := math.Floor(math.Log10(step))
	e := step / math.Pow(10, power)
	factor := 1.0
	switch {
	case e >= e10:
		factor = 10
	case e >= e5:
		factor = 5
	case e >= e2:
		factor = 2
	}
	if power < 0 {
		inc = math.Pow(10, -power) / factor
		i1, i2 = math.Round(lo*inc), math.Round(hi*inc)
		if i1/inc < lo {
			i1++
		}
		if i2/inc > hi {
			i2--
		}
		inc = -inc
	} else {
		inc = math.Pow(10, power) * factor
		i1, i2 = math.Round(lo/inc), math.Round(hi/inc)
		if i1*inc < lo {
			i1++
		}
		if i2*inc > hi {
			i2--
		}
	}
	if i2 < i1 && 0.5 <= count && count < 2 {
		return tickSpec(lo, hi, count*2)
	}
	return i1, i2, inc
}

// ticks returns the round values between lo and hi, in their order.
func ticks(lo, hi, count float64) []float64 {
	if !(count > 0) || math.IsNaN(lo) || math.IsNaN(hi) || math.IsInf(lo, 0) || math.IsInf(hi, 0) {
		return nil
	}
	if lo == hi {
		return []float64{lo}
	}
	reverse := hi < lo
	if reverse {
		lo, hi = hi, lo
	}
	i1, i2, inc := tickSpec(lo, hi, count)
	if !(i2 >= i1) || math.IsInf(inc, 0) || math.IsNaN(inc) {
		return nil
	}
	n := int(i2-i1) + 1
	out := make([]float64, n)
	for i := range n {
		k := i
		if reverse {
			k = n - 1 - i
		}
		if inc < 0 {
			out[i] = (i1 + float64(k)) / -inc
		} else {
			out[i] = (i1 + float64(k)) * inc
		}
	}
	return out
}

// tickIncrement returns the step between ticks, or its negative inverse
// below 1, as d3-array's tickIncrement.
func tickIncrement(lo, hi, count float64) float64 {
	_, _, inc := tickSpec(lo, hi, count)
	return inc
}

// tickStep returns the step between ticks, signed as hi-lo.
func tickStep(lo, hi, count float64) float64 {
	reverse := hi < lo
	if reverse {
		lo, hi = hi, lo
	}
	inc := tickIncrement(lo, hi, count)
	if inc < 0 {
		inc = 1 / -inc
	}
	if reverse {
		return -inc
	}
	return inc
}

// Band divides the plot into a band for each of its labels, as d3's
// scaleBand and Swift Charts' scales of categories: the bars of a bar
// chart, one for each month.
//
//	x := chart.NewBand(months...).Padding(0.2, 0.1)
//	bar := x.Band(3) // April's
type Band struct {
	Labels []string
	// Inner is the room between bands, and Outer the room before the
	// first and after the last, in fractions of a band's step.
	Inner, Outer float32
	// Align places the bands in the room the outer padding leaves: 0 at
	// the start, 0.5 (as NewBand) in the middle, 1 at the end.
	Align float32
}

// NewBand returns a band scale of labels, without padding.
func NewBand(labels ...string) Band { return Band{Labels: labels, Align: 0.5} }

// Padding sets the room between bands, and before and after them, in
// fractions of a band's step: 0.2 leaves a fifth of each step between bars.
func (b Band) Padding(inner, outer float32) Band {
	b.Inner, b.Outer = max(0, min(1, inner)), max(0, outer)
	return b
}

// step returns the distance from a band to the next, and where the first
// starts.
func (b Band) step() (step, start float32) {
	n := float32(len(b.Labels))
	step = 1 / max(1, n-b.Inner+2*b.Outer)
	start = (1 - step*(n-b.Inner)) * b.Align
	return step, start
}

// Band returns the stretch of the i-th band.
func (b Band) Band(i int) Span {
	step, start := b.step()
	lo := start + step*float32(i)
	return Span{lo, lo + b.Width()}
}

// Width returns the width of a band, in a fraction of the plot.
func (b Band) Width() float32 {
	step, _ := b.step()
	return step * (1 - b.Inner)
}

// At returns the middle of the i-th band.
func (b Band) At(i int) float32 { return b.Band(i).Mid() }

// Index returns the band at f along the plot, the one whose step holds it,
// so that the room between bars belongs to a bar; ok is false outside of
// every band's step.
func (b Band) Index(f float32) (i int, ok bool) {
	step, start := b.step()
	if len(b.Labels) == 0 {
		return 0, false
	}
	// A band's step starts half the inner padding before it.
	k := (f - start + step*b.Inner/2) / step
	if k < 0 || k >= float32(len(b.Labels)) {
		return 0, false
	}
	return int(k), true
}

// Ticks returns a tick in the middle of each band, labeled with its label.
func (b Band) Ticks() []Tick {
	out := make([]Tick, len(b.Labels))
	for i, l := range b.Labels {
		out[i] = Tick{At: b.At(i), Value: float64(i), Label: l}
	}
	return out
}
