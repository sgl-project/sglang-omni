package diff

// compact slides the groups of changed lines of one side (r, its lines
// interned as ids and as text) up and down where the same lines frame
// them, as git's xdl_change_compact: groups merge when they meet, line
// up with a group of the other side (ro) where they can, and otherwise
// take the place the indent heuristic scores best.
func compact(r, ro []bool, ids []int32, lines []string) {
	g, gOther := groupInit(r), groupInit(ro)
	for {
		if g.end != g.start {
			var size, earliestEnd int
			endMatchingOther := -1
			for {
				size = g.end - g.start
				endMatchingOther = -1
				for g.slideUp(r, ids) {
					gOther.previous(ro)
				}
				earliestEnd = g.end
				if gOther.end > gOther.start {
					endMatchingOther = g.end
				}
				for g.slideDown(r, ids) {
					gOther.next(ro)
					if gOther.end > gOther.start {
						endMatchingOther = g.end
					}
				}
				if size == g.end-g.start {
					break
				}
			}
			switch {
			case g.end == earliestEnd:
				// It could not move.
			case endMatchingOther != -1:
				// Back to the last place matching a group of the other side.
				for gOther.end == gOther.start {
					if !g.slideUp(r, ids) || !gOther.previous(ro) {
						break
					}
				}
			default:
				shift := max(earliestEnd, g.end-size-1, g.end-maxSliding)
				best := -1
				var bestScore score
				for ; shift <= g.end; shift++ {
					var s score
					s.add(measure(lines, shift))
					s.add(measure(lines, shift-size))
					if best == -1 || s.cmp(bestScore) <= 0 {
						bestScore, best = s, shift
					}
				}
				for g.end > best {
					if !g.slideUp(r, ids) || !gOther.previous(ro) {
						break
					}
				}
			}
		}
		if !g.next(r) || !gOther.next(ro) {
			return
		}
	}
}

// group is a run of changed lines, maybe empty, between unchanged ones.
type group struct{ start, end int }

func groupInit(r []bool) group {
	g := group{}
	for g.end < len(r) && r[g.end] {
		g.end++
	}
	return g
}

func (g *group) next(r []bool) bool {
	if g.end >= len(r) {
		return false
	}
	g.start = g.end + 1
	for g.end = g.start; g.end < len(r) && r[g.end]; g.end++ {
	}
	return true
}

func (g *group) previous(r []bool) bool {
	if g.start <= 0 {
		return false
	}
	g.end = g.start - 1
	for g.start = g.end; g.start > 0 && r[g.start-1]; g.start-- {
	}
	return true
}

func (g *group) slideDown(r []bool, ids []int32) bool {
	if g.start < g.end && g.end < len(r) && ids[g.start] == ids[g.end] {
		r[g.start] = false
		g.start++
		r[g.end] = true
		g.end++
		for g.end < len(r) && r[g.end] {
			g.end++
		}
		return true
	}
	return false
}

func (g *group) slideUp(r []bool, ids []int32) bool {
	if g.start < g.end && g.start > 0 && ids[g.start-1] == ids[g.end-1] {
		g.start--
		r[g.start] = true
		g.end--
		r[g.end] = false
		for g.start > 0 && r[g.start-1] {
			g.start--
		}
		return true
	}
	return false
}

// The weights of git's indent heuristic (xdiffi.c), tuned on a corpus
// of real changes.
const (
	maxIndent                      = 200
	maxBlanks                      = 20
	startOfFilePenalty             = 1
	endOfFilePenalty               = 21
	totalBlankWeight               = -30
	postBlankWeight                = 6
	relativeIndentPenalty          = -4
	relativeIndentWithBlankPenalty = 10
	relativeOutdentPenalty         = 24
	relativeOutdentWithBlank       = 17
	relativeDedentPenalty          = 23
	relativeDedentWithBlank        = 17
	indentWeight                   = 60
	maxSliding                     = 100
)

// indent is how far a line is indented, tabs to multiples of 8, or -1
// for a blank line.
func indent(s string) int {
	n := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case ' ':
			n++
		case '\t':
			n += 8 - n%8
		case '\r', '\n', '\f', '\v':
		default:
			return n
		}
		if n >= maxIndent {
			return maxIndent
		}
	}
	return -1
}

type split struct {
	endOfFile             bool
	indent                int
	preBlank, preIndent   int
	postBlank, postIndent int
}

func measure(lines []string, at int) split {
	var m split
	if at >= len(lines) {
		m.endOfFile, m.indent = true, -1
	} else {
		m.indent = indent(lines[at])
	}
	m.preIndent = -1
	for i := at - 1; i >= 0; i-- {
		if m.preIndent = indent(lines[i]); m.preIndent != -1 {
			break
		}
		if m.preBlank++; m.preBlank == maxBlanks {
			m.preIndent = 0
			break
		}
	}
	m.postIndent = -1
	for i := at + 1; i < len(lines); i++ {
		if m.postIndent = indent(lines[i]); m.postIndent != -1 {
			break
		}
		if m.postBlank++; m.postBlank == maxBlanks {
			m.postIndent = 0
			break
		}
	}
	return m
}

type score struct{ effectiveIndent, penalty int }

func (s *score) add(m split) {
	if m.preIndent == -1 && m.preBlank == 0 {
		s.penalty += startOfFilePenalty
	}
	if m.endOfFile {
		s.penalty += endOfFilePenalty
	}
	postBlank := 0
	if m.indent == -1 {
		postBlank = 1 + m.postBlank
	}
	totalBlank := m.preBlank + postBlank
	s.penalty += totalBlankWeight*totalBlank + postBlankWeight*postBlank
	in := m.indent
	if in == -1 {
		in = m.postIndent
	}
	blanks := totalBlank != 0
	s.effectiveIndent += in
	switch {
	case in == -1, m.preIndent == -1, in == m.preIndent:
	case in > m.preIndent:
		s.penalty += pick(blanks, relativeIndentWithBlankPenalty, relativeIndentPenalty)
	case m.postIndent != -1 && m.postIndent > in:
		s.penalty += pick(blanks, relativeOutdentWithBlank, relativeOutdentPenalty)
	default:
		s.penalty += pick(blanks, relativeDedentWithBlank, relativeDedentPenalty)
	}
}

func pick(c bool, a, b int) int {
	if c {
		return a
	}
	return b
}

func (s score) cmp(o score) int {
	c := 0
	if s.effectiveIndent > o.effectiveIndent {
		c = 1
	} else if s.effectiveIndent < o.effectiveIndent {
		c = -1
	}
	return indentWeight*c + s.penalty - o.penalty
}
