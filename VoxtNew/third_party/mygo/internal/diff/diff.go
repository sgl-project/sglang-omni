// Package diff compares sequences: the lines of two texts, or the words
// of two lines. It finds the changes with the histogram algorithm of git
// (xhistogram.c), which anchors on lines that occur rarely and so keeps
// the matches people expect in source code, falling back to Myers's O(ND)
// algorithm, with xdiff's cost limit, where every line repeats too often.
// Groups of changed lines are then slid to start and end where blank
// lines and indentation make them read as whole blocks, with git's indent
// heuristic.
package diff

// Hunk is a change: the elements A0 to A1 of the old sequence replaced by
// B0 to B1 of the new one. A0 == A1 is an insertion before A0, B0 == B1 a
// deletion.
type Hunk struct{ A0, A1, B0, B1 int }

// Ints compares two sequences of interned elements, equal elements having
// equal values, and returns the hunks turning a into b, in order.
func Ints(a, b []int32) []Hunk {
	d := differ{a: a, b: b}
	d.histogram(0, len(a), 0, len(b), 0)
	return d.hunks()
}

// Lines compares the lines of two texts and returns the hunks of lines
// turning a into b, slid to read as blocks of code.
func Lines(a, b []string) []Hunk {
	ia, ib := Intern(a, b)
	d := differ{a: ia, b: ib}
	d.histogram(0, len(ia), 0, len(ib), 0)
	if d.da == nil {
		return nil
	}
	compact(d.da, d.db, ia, a)
	compact(d.db, d.da, ib, b)
	return d.hunks()
}

// Intern numbers the distinct strings of a and b, equal strings sharing a
// number.
func Intern(a, b []string) ([]int32, []int32) {
	ids := make(map[string]int32, len(a)+len(b))
	conv := func(s []string) []int32 {
		out := make([]int32, len(s))
		for i, v := range s {
			id, ok := ids[v]
			if !ok {
				id = int32(len(ids))
				ids[v] = id
			}
			out[i] = id
		}
		return out
	}
	return conv(a), conv(b)
}

// differ marks the elements of a and b that the hunks change.
type differ struct {
	a, b       []int32
	da, db     []bool // changed elements
	next, cnt  []int  // histogram scratch, by position in a
	recs       map[int32]record
	recsCap    int
	myersV     []int
	myersExtra int
}

type record struct{ first, count int }

func (d *differ) mark(a0, a1, b0, b1 int) {
	if a0 == a1 && b0 == b1 {
		return
	}
	if d.da == nil {
		d.da, d.db = make([]bool, len(d.a)), make([]bool, len(d.b))
	}
	for i := a0; i < a1; i++ {
		d.da[i] = true
	}
	for i := b0; i < b1; i++ {
		d.db[i] = true
	}
}

// hunks turns the marks into hunks.
func (d *differ) hunks() []Hunk {
	if d.da == nil {
		return nil
	}
	var out []Hunk
	i, j := 0, 0
	for i < len(d.a) || j < len(d.b) {
		if i < len(d.a) && j < len(d.b) && !d.da[i] && !d.db[j] {
			i++
			j++
			continue
		}
		h := Hunk{A0: i, B0: j}
		for i < len(d.a) && d.da[i] {
			i++
		}
		for j < len(d.b) && d.db[j] {
			j++
		}
		h.A1, h.B1 = i, j
		if h.A0 == h.A1 && h.B0 == h.B1 {
			// Unpaired unchanged elements cannot happen; never loop.
			if i < len(d.a) {
				i++
			}
			if j < len(d.b) {
				j++
			}
			continue
		}
		out = append(out, h)
	}
	return out
}
