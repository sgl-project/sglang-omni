package chart

import "math"

// Segment is a value's stretch in a stack, from Lo up to Hi.
type Segment struct{ Lo, Hi float64 }

// Stacking is how Stack piles series up, as Swift Charts' stacking
// methods.
type Stacking int

const (
	// StackStandard piles values up from 0, and negative values down from it.
	StackStandard Stacking = iota
	// StackNormalized piles them up as StackStandard does, in shares of the sum of
	// their sizes: the stacks of positive values go from 0 to 1.
	StackNormalized
	// StackCenter piles them up around 0, as a streamgraph.
	StackCenter
)

// Stack piles series up, as stacked bars and areas show them: series[s][i]
// is the i-th value of the s-th series, and the result's [s][i] is where
// it goes in the stack of the i-th values, the s-th from the bottom. NaN
// values take no room.
//
//	stacked := chart.Stack([][]float64{apples, pears}, chart.StackStandard)
//	for s, series := range stacked {
//		for i, seg := range series {
//			chart.Rect(c, x.Band(i), y.Span(seg.Lo, seg.Hi)).Background(colors[s])
//		}
//	}
func Stack(series [][]float64, how Stacking) [][]Segment {
	n := 0
	for _, s := range series {
		n = max(n, len(s))
	}
	out := make([][]Segment, len(series))
	for s := range series {
		out[s] = make([]Segment, n)
	}
	value := func(s, i int) float64 {
		if i >= len(series[s]) || math.IsNaN(series[s][i]) {
			return 0
		}
		return series[s][i]
	}
	for i := range n {
		up, down, total := 0.0, 0.0, 0.0
		for s := range series {
			v := value(s, i)
			total += math.Abs(v)
			switch {
			case how == StackCenter:
				out[s][i] = Segment{up, up + v}
				up += v
			case v < 0:
				out[s][i] = Segment{down + v, down}
				down += v
			default:
				out[s][i] = Segment{up, up + v}
				up += v
			}
		}
		shift, scale := 0.0, 1.0
		switch how {
		case StackCenter:
			shift = -up / 2
		case StackNormalized:
			if total > 0 {
				scale = 1 / total
			}
		}
		for s := range series {
			seg := &out[s][i]
			seg.Lo, seg.Hi = (seg.Lo+shift)*scale, (seg.Hi+shift)*scale
		}
	}
	return out
}
