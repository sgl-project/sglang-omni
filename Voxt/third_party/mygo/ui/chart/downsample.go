package chart

import (
	"math"
	"sort"
)

// MinMax reduces a line of many points to the lowest and highest of each
// pixel column of the plot, in their order, and the ends of each run of
// defined points, so that a line through them looks the same and narrow
// spikes still show. xs and ys are fractions, as for Line; xs must not
// decrease, or the columns come out wrong, and a NaN or infinite value
// breaks the line. Points left or right of the plot go, but for the
// nearest one on each side. pixelWidth is the plot's width in pixels,
// ceil(r.W * p.Scale()).
//
// MinMax appends to dstX[:0] and dstY[:0] and returns them, so that a
// view keeping them reduces every frame without allocating; they must not
// share memory with xs or ys. Keep the points themselves for selections
// and readings.
func MinMax(dstX, dstY, xs, ys []float32, pixelWidth int) ([]float32, []float32) {
	dstX, dstY = dstX[:0], dstY[:0]
	if pixelWidth <= 0 {
		return dstX, dstY
	}
	n := min(len(xs), len(ys))
	finite := func(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }
	appendPoint := func(i int) { dstX = append(dstX, xs[i]); dstY = append(dstY, ys[i]) }
	runs(n, func(i int) bool { return finite(xs[i]) && finite(ys[i]) }, func(from, to int) {
		// Binary search leaves at most one point beyond either visible edge.
		lo := from + sort.Search(to-from, func(i int) bool { return xs[from+i] >= 0 })
		if lo == to {
			return
		}
		hi := from + sort.Search(to-from, func(i int) bool { return xs[from+i] > 1 })
		if lo == from && lo == hi {
			return
		}
		if lo > from {
			lo--
		}
		if hi < to {
			hi++
		}
		if len(dstX) > 0 {
			dstX = append(dstX, float32(math.NaN()))
			dstY = append(dstY, float32(math.NaN()))
		}
		last := -1
		emit := func(i int) {
			if i != last {
				appendPoint(i)
				last = i
			}
		}
		emit(lo)
		for i := lo; i < hi; {
			bucket := int(math.Floor(float64(xs[i]) * float64(pixelWidth)))
			low, high := i, i
			i++
			for i < hi && xs[i] >= 0 && xs[i] <= 1 && bucket == int(math.Floor(float64(xs[i])*float64(pixelWidth))) {
				if ys[i] < ys[low] {
					low = i
				}
				if ys[i] > ys[high] {
					high = i
				}
				i++
			}
			emit(min(low, high))
			emit(max(low, high))
		}
		emit(hi - 1)
	})
	return dstX, dstY
}
