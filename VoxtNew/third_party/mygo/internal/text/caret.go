package text

import "slices"

// Affinity chooses the preceding or following logical text at a boundary.
// The two edges can occupy different positions in bidi text or at a wrap.
type Affinity uint8

const (
	Downstream Affinity = iota
	Upstream
)

type CaretPosition struct {
	Index    int
	Affinity Affinity
}

type Range struct{ Start, End int }

type caretStop struct {
	position CaretPosition
	x        float32
	// arrival is the visual direction from which this edge is reached.
	arrival int
}

func (l *Layout) visualStops(li int) []caretStop {
	line := &l.Lines[li]
	if line.stops != nil {
		return line.stops
	}
	if l.boundaries == nil {
		var b Boundaries
		b.Reset(l.Runes)
		l.boundaries = b.GraphemeOffsets()
	}
	var stops []caretStop
	line.upstream = slices.Clone(line.lineCarets())
	for i := 0; i < len(line.Glyphs); {
		g := line.Glyphs[i]
		x0, x1 := g.X, g.X+g.Advance
		i++
		for i < len(line.Glyphs) && line.Glyphs[i].Cluster == g.Cluster && line.Glyphs[i].Runes == g.Runes {
			x0 = min(x0, line.Glyphs[i].X)
			x1 = max(x1, line.Glyphs[i].X+line.Glyphs[i].Advance)
			i++
		}
		if g.Runes <= 0 {
			continue
		}
		for k := 0; k <= g.Runes; k++ {
			at := g.Cluster + k
			if _, ok := slices.BinarySearch(l.boundaries, at); !ok {
				continue
			}
			x := x0 + (x1-x0)*float32(k)/float32(g.Runes)
			if g.RTL {
				x = x1 - (x1-x0)*float32(k)/float32(g.Runes)
			}
			affinity, arrival := Downstream, 0
			if k == 0 {
				arrival = -1
			}
			if k == g.Runes {
				affinity, arrival = Upstream, 1
			}
			if g.RTL {
				arrival = -arrival
			}
			if affinity == Upstream && at >= line.Start && at <= line.End {
				line.upstream[at-line.Start] = x
			}
			stops = append(stops, caretStop{CaretPosition{at, affinity}, x, arrival})
		}
	}
	// Keep trimmed spaces and empty lines navigable too.
	carets := line.lineCarets()
	first, _ := slices.BinarySearch(l.boundaries, line.Start)
	for k := first; k < len(l.boundaries) && l.boundaries[k] <= line.End; k++ {
		at := l.boundaries[k]
		if !slices.ContainsFunc(stops, func(s caretStop) bool { return s.position.Index == at }) {
			stops = append(stops, caretStop{CaretPosition{at, Downstream}, carets[at-line.Start], 0})
		}
	}
	if len(stops) == 0 {
		stops = append(stops, caretStop{CaretPosition{line.Start, Downstream}, line.X, 0})
	}
	slices.SortStableFunc(stops, func(a, b caretStop) int {
		if a.x < b.x {
			return -1
		}
		if a.x > b.x {
			return 1
		}
		return 0
	})
	line.stops = stops
	return stops
}

func (l *Layout) caretLine(p CaretPosition) int {
	li := l.LineAt(p.Index)
	if p.Affinity == Upstream && li > 0 && l.Lines[li].Start == p.Index && l.Lines[li-1].End == p.Index {
		li--
	}
	return li
}

// CaretAt preserves affinity at bidi boundaries and soft line breaks.
func (l *Layout) CaretAt(p CaretPosition) (x, y, h float32) {
	if len(l.Lines) == 0 {
		return 0, 0, 0
	}
	p.Index = max(0, min(p.Index, len(l.Runes)))
	li := l.caretLine(p)
	l.visualStops(li)
	line := &l.Lines[li]
	i := max(0, min(p.Index-line.Start, len(line.carets)-1))
	x = line.carets[i]
	if p.Affinity == Upstream {
		x = line.upstream[i]
	}
	return x, line.Y, line.Height
}

// PositionAt hit-tests whole graphemes, retaining the visual edge chosen.
func (l *Layout) PositionAt(x, y float32) CaretPosition {
	if len(l.Lines) == 0 {
		return CaretPosition{}
	}
	li := len(l.Lines) - 1
	for i := range l.Lines {
		if y < l.Lines[i].Y+l.Lines[i].Height {
			li = i
			break
		}
	}
	stops := l.visualStops(li)
	best, distance := stops[0], float32(-1)
	for _, s := range stops {
		d := abs(s.x - x)
		// Affinity breaks ties at one offset, not across offsets sharing
		// an edge, such as the positions before and after a trimmed space.
		if distance < 0 || d < distance || d == distance && s.position.Index == best.position.Index && s.position.Affinity == Downstream {
			best, distance = s, d
		}
	}
	return best.position
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func arriving(stops []caretStop, x float32, direction int) CaretPosition {
	var p CaretPosition
	for _, s := range stops {
		// Native runs can disagree by a few float32 rounding bits at their
		// shared edge, especially when fallback fonts precede a bidi run.
		if abs(s.x-x) > 0.01 {
			continue
		}
		p = s.position
		if s.arrival == direction {
			return p
		}
	}
	return p
}

// MoveCaret advances one visual grapheme, preserving the edge arrived at.
func (l *Layout) MoveCaret(p CaretPosition, direction int) CaretPosition {
	if len(l.Lines) == 0 || direction == 0 {
		return p
	}
	if direction < 0 {
		direction = -1
	} else {
		direction = 1
	}
	li := l.caretLine(p)
	x, _, _ := l.CaretAt(p)
	stops := l.visualStops(li)
	if direction > 0 {
		for _, s := range stops {
			if s.x > x+0.001 {
				return arriving(stops, s.x, direction)
			}
		}
	} else {
		for i := len(stops) - 1; i >= 0; i-- {
			if stops[i].x < x-0.001 {
				return arriving(stops, stops[i].x, direction)
			}
		}
	}
	next := li + direction
	if l.Lines[li].RTL {
		next = li - direction
	}
	if next < 0 || next >= len(l.Lines) {
		return p
	}
	stops = l.visualStops(next)
	x = stops[0].x
	if direction < 0 {
		x = stops[len(stops)-1].x
	}
	return arriving(stops, x, direction)
}

// SelectionRanges returns logical ranges covered by a visual gesture.
// A gesture through mixed-direction text can cover discontiguous ranges.
func (l *Layout) SelectionRanges(a, b CaretPosition) []Range {
	if len(l.Lines) == 0 {
		return nil
	}
	ai, bi := l.caretLine(a), l.caretLine(b)
	ax, _, _ := l.CaretAt(a)
	bx, _, _ := l.CaretAt(b)
	if ai > bi {
		ai, bi, ax, bx = bi, ai, bx, ax
	}
	var ranges []Range
	for li := ai; li <= bi; li++ {
		line := &l.Lines[li]
		lo, hi := line.X, line.X+line.Width
		if ai == bi {
			lo, hi = min(ax, bx), max(ax, bx)
		} else {
			if li == ai {
				if line.RTL {
					hi = ax
				} else {
					lo = ax
				}
			}
			if li == bi {
				if line.RTL {
					lo = bx
				} else {
					hi = bx
				}
			}
		}
		l.visualStops(li)
		first, _ := slices.BinarySearch(l.boundaries, line.Start)
		for k := first; k+1 < len(l.boundaries) && l.boundaries[k] < line.End; k++ {
			start, end := l.boundaries[k], l.boundaries[k+1]
			if start < line.Start || end > line.End {
				continue
			}
			x0, _, _ := l.CaretAt(CaretPosition{start, Downstream})
			x1, _, _ := l.CaretAt(CaretPosition{end, Upstream})
			left, right := min(x0, x1), max(x0, x1)
			if right > left && left >= lo-0.01 && right <= hi+0.01 {
				ranges = append(ranges, Range{start, end})
			}
		}
		if li < bi && li+1 < len(l.Lines) && l.Lines[li+1].Start > line.End {
			ranges = append(ranges, Range{line.End, l.Lines[li+1].Start})
		}
	}
	slices.SortFunc(ranges, func(a, b Range) int { return a.Start - b.Start })
	out := ranges[:0]
	for _, r := range ranges {
		if n := len(out); n > 0 && out[n-1].End >= r.Start {
			out[n-1].End = max(out[n-1].End, r.End)
		} else {
			out = append(out, r)
		}
	}
	return out
}
