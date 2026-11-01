package ui

// TextLines is where a text area laid out the lines of its text, kept in
// the app's state with TrackLines, for an app that paints beside them: line
// numbers by the first row of each line where the text wraps, or marks on
// runs of a line. The lines are the text's, between newlines; a line the
// area wraps takes several rows.
//
// The area lays out only the lines in view, and those around them: the
// heights of the others are estimated until it shows them, as the scroll
// bar's are. Between a frame's build and its layout the lines are the last
// frame's, in the area's last width; in Draw and DrawOver, this frame's.
type TextLines struct {
	ed *editor
}

// TrackLines keeps in l where the text area lays out its lines.
func (e *node) TrackLines(l *TextLines) *node {
	if l != nil {
		l.ed = e.st.editor
	}
	return e
}

// area returns the text area, synced with its text, or nil before it was
// first laid out.
func (l *TextLines) area() *area {
	if l == nil || l.ed == nil || l.ed.area == nil || l.ed.area.line.Height <= 0 {
		return nil
	}
	a, b := l.ed.area, &l.ed.buf
	if a.version != b.version || len(a.hs.measured) != len(b.paras)+1 {
		a.sync(b, a.params)
	}
	return a
}

// Count returns how many lines the text has.
func (l *TextLines) Count() int {
	if l == nil || l.ed == nil {
		return 0
	}
	return len(l.ed.buf.paras)
}

// Top returns the top of line i in the area's content, from the top of its
// first line: below the padding, and before the area scrolls.
func (l *TextLines) Top(i int) float32 {
	a := l.area()
	if a == nil {
		return float32(i) * l.lineHeight()
	}
	return float32(a.hs.top(max(0, min(i, len(l.ed.buf.paras)))))
}

// Height returns the height of the text: the top of the line after the
// last.
func (l *TextLines) Height() float32 { return l.Top(l.Count()) }

// At returns the line at y in the content, as Top measures it: the first
// for y above it, the last for y past the end.
func (l *TextLines) At(y float32) int {
	a := l.area()
	if a == nil {
		return max(0, min(int(y/l.lineHeight()), l.Count()-1))
	}
	return a.hs.at(float64(max(y, 0)))
}

// Rows returns where the rows of line i start, in runes from the start of
// the line: [0] for a line on one row. It lays the line out if the area has
// not, which measures its height.
func (l *TextLines) Rows(i int) []int {
	return l.AppendRows(nil, i)
}

// AppendRows appends where the rows of line i start to dst, as Rows.
func (l *TextLines) AppendRows(dst []int, i int) []int {
	a := l.area()
	if a == nil || i < 0 || i >= len(l.ed.buf.paras) {
		return append(dst, 0)
	}
	lay := a.paraLayout(l.ed, i)
	if len(lay.Lines) == 0 {
		return append(dst, 0)
	}
	for _, line := range lay.Lines {
		dst = append(dst, a.global(l.ed, i, line.Start)-l.ed.buf.start(i))
	}
	return dst
}

func (l *TextLines) lineHeight() float32 {
	if l == nil || l.ed == nil {
		return 18
	}
	return l.ed.lineHeight()
}
