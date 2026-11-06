package richtext

import "math/bits"

// heights sums the heights of the paragraphs of an editor, measured as
// they are laid out and estimated for the others, in two Fenwick trees:
// the heights measured, and how many are not, which the estimate
// multiplies. The top of a paragraph, and the paragraph at a height, take
// O(log n) whatever the estimate, as a text area's do.
type heights struct {
	measured []float64 // 1-based
	unknown  []int32
	est      float64
}

// reset sums the heights hs, 0 for those not measured.
func (h *heights) reset(hs []float32) {
	n := len(hs)
	if cap(h.measured) > 4*(n+1) {
		h.measured, h.unknown = nil, nil
	}
	h.measured = append(h.measured[:0], make([]float64, n+1)...)
	h.unknown = append(h.unknown[:0], make([]int32, n+1)...)
	for i, x := range hs {
		if x > 0 {
			h.measured[i+1] = float64(x)
		} else {
			h.unknown[i+1] = 1
		}
	}
	for i := 1; i <= n; i++ {
		if j := i + i&-i; j <= n {
			h.measured[j] += h.measured[i]
			h.unknown[j] += h.unknown[i]
		}
	}
}

// add changes the height measured of paragraph p by dm, and the count of
// those not measured by du.
func (h *heights) add(p int, dm float64, du int32) {
	for i := p + 1; i < len(h.measured); i += i & -i {
		h.measured[i] += dm
		h.unknown[i] += du
	}
}

// top returns the top of paragraph p: the height of those before it.
func (h *heights) top(p int) float64 {
	var m float64
	var u int32
	for i := min(p, len(h.measured)-1); i > 0; i -= i & -i {
		m += h.measured[i]
		u += h.unknown[i]
	}
	return m + float64(u)*h.est
}

// at returns the paragraph at y, the last for y past the end.
func (h *heights) at(y float64) int {
	n := len(h.measured) - 1
	if n <= 0 {
		return 0
	}
	pos := 0
	var sum float64
	for step := 1 << (bits.Len(uint(n)) - 1); step > 0; step >>= 1 {
		if next := pos + step; next <= n {
			if s := sum + h.measured[next] + float64(h.unknown[next])*h.est; s <= y {
				pos, sum = next, s
			}
		}
	}
	return min(pos, n-1)
}

// totals returns the sum of the heights measured and how many are not.
func (h *heights) totals() (float64, int) {
	var m float64
	var u int32
	for i := len(h.measured) - 1; i > 0; i -= i & -i {
		m += h.measured[i]
		u += h.unknown[i]
	}
	return m, int(u)
}
