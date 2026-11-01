package ui

import (
	"math"
	"slices"

	"github.com/egoist/mygo/internal/text"
)

// The layout is CSS flexbox, or grid (grid.go), with box-sizing:
// border-box. Sizes are border boxes (padding and border included, margins
// not); inf marks a size not known yet, such as the height of a column
// that grows with its content.

var inf = float32(math.Inf(1))

func finite(v float32) bool { return v < inf }

func (e *node) padX() float32 { return e.pad[1] + e.pad[3] + e.border[1] + e.border[3] }
func (e *node) padY() float32 { return e.pad[0] + e.pad[2] + e.border[0] + e.border[2] }

// contentX and contentY place the content box in the border box.
func (e *node) contentX() float32 { return e.pad[3] + e.border[3] }
func (e *node) contentY() float32 { return e.pad[0] + e.border[0] }

// m returns margin i, with automatic margins as 0.
func (e *node) m(i int) float32 {
	if isAuto(e.margin[i]) {
		return 0
	}
	return e.margin[i]
}

func (e *node) marginX() float32 { return e.m(1) + e.m(3) }
func (e *node) marginY() float32 { return e.m(0) + e.m(2) }

func (e *node) scrolls() bool { return e.flags&(flagScrollX|flagScrollY) != 0 }

// clampW and clampH apply the min and max sizes.
func (e *node) clampW(w, cbW float32) float32 {
	if v, ok := e.maxW.resolve(base(cbW)); ok {
		w = min(w, v)
	}
	if v, ok := e.minW.resolve(base(cbW)); ok {
		w = max(w, v)
	}
	return max(w, 0)
}

func (e *node) clampH(h, cbH float32) float32 {
	if v, ok := e.maxH.resolve(base(cbH)); ok {
		h = min(h, v)
	}
	if v, ok := e.minH.resolve(base(cbH)); ok {
		h = max(h, v)
	}
	return max(h, 0)
}

// base turns an unknown size into the -1 length.resolve takes for it.
func base(v float32) float32 {
	if finite(v) {
		return v
	}
	return -1
}

// layoutTree lays the frame out in a window of w×h DIPs and gives every
// element its box relative to the window.
func layoutTree(root *node, w, h float32) {
	rt := root.c.rt
	rt.consumed = false
	layoutBox(root, w, h)
	// Lists build rows as they lay out, which may ask to come into view.
	for _, e := range root.c.reveal {
		if !slices.Contains(rt.revealIDs, e.id) {
			rt.revealIDs = append(rt.revealIDs, e.id)
		}
	}
	if len(rt.revealIDs) > 0 {
		revealAll(root, rt.revealIDs)
		rt.revealIDs = rt.revealIDs[:0]
	}
	if rt.late {
		// Rows built as their lists laid out handled their input as the
		// view's elements do, and the next frame shows what that changed;
		// what they put above the window is laid out with it.
		rt.lateInput()
		if ov := root.c.overlay; ov != nil {
			if ov.parent == nil {
				root.add(ov)
			}
			layoutAbsolute(root)
		}
	}
	rt.animateLayout(root, w, h)
	// Reattach overlays to the geometry shown by this frame, after transitions.
	if ov := root.c.overlay; ov != nil {
		layoutAbsolute(ov)
	}
	if rt.late {
		// Lists resized by their transitions built rows as they laid out
		// anew.
		rt.lateInput()
	}
	place(root, 0, 0)
	rt.animateMatches()
}

// lateInput forgets the input of the rows built as their lists laid out,
// which they handled as the view's elements do: the next frame shows what
// that changed.
func (rt *engine) lateInput() {
	rt.late = false
	rt.forgetInput()
	if rt.consumed {
		rt.animating = true
	}
}

// place turns the boxes, laid out relative to their parents, into window
// coordinates, moving the content of scroll containers by their offset.
func place(e *node, x, y float32) {
	e.x += x
	e.y += y
	cx, cy := e.x, e.y
	if f := e.followX; f != nil {
		cx -= float32(f.followShift())
	}
	if e.scrolls() {
		// The content may no longer reach as far as the offset, as when a
		// page gives way to a shorter one, or the app asked for the end:
		// keep the offset within it.
		e.adoptScroll()
		s := e.st
		mx, my := e.maxScroll()
		if sx, sy := max(0, min(s.scrollX, mx)), max(0, min(s.scrollY, my)); sx != s.scrollX || sy != s.scrollY {
			s.beginMove(e.c.rt.frame)
			s.scrollTo(sx, sy)
		}
		if t := e.track; t != nil {
			t.MaxX, t.MaxY = float32(mx), float32(my)
		}
		if s.movedIn(e.c.rt.frame) {
			// What was built read the old offset, as List's rows or the
			// app from its ScrollState: build the next frame with the new.
			e.c.rt.animating = true
		}
		cx -= float32(e.scrollLeft())
		cy -= float32(s.scrollY - e.scrollBase)
	}
	for ch := e.first; ch != nil; ch = ch.next {
		// A copy of an element leaving the flow stays where the flow was.
		if ch.flags&flagAbsolute != 0 && ch.leaving != 1 {
			place(ch, e.x, e.y)
		} else {
			place(ch, cx, cy)
		}
	}
}

// relative returns how far the insets of an element in flow move it from
// where the layout put it, against a content box cw×ch. Where both Left
// and Right are set, Left wins, or Right in an element laid out right to
// left, as in CSS.
func (e *node) relative(cw, ch float32) (dx, dy float32) {
	left, lok := e.inset[3].resolve(base(cw))
	right, rok := e.inset[1].resolve(base(cw))
	if lok && !(rok && e.rtl()) {
		dx = left
	} else if rok {
		dx = -right
	}
	if v, ok := e.inset[0].resolve(base(ch)); ok {
		dy = v
	} else if v, ok := e.inset[2].resolve(base(ch)); ok {
		dy = -v
	}
	return dx, dy
}

// textParams returns how to lay out the element's text at a content width
// (0 for one line per paragraph).
func (e *node) textParams(width float32) text.Params {
	ts := e.resolvedText()
	style := text.Style{
		Family: ts.family, Size: ts.size, Weight: ts.weight, Italic: ts.italic, LineHeight: ts.lineHeight,
		LetterSpacing: ts.spacing, Features: ts.features, Lang: ts.lang,
	}
	if ts.fixedLine && ts.lineHeight > 0 {
		style.LineHeight = ts.lineHeight / style.FontSize()
	}
	p := text.Params{Text: e.text, Width: width, MaxLines: e.maxLines, Style: style, Spans: e.textSpans(), NoWrap: e.noWrap, Ellipsis: e.ellipsis, RTL: e.rtl()}
	switch {
	case ts.align == Center:
		p.Align = text.Center
	case e.rtl():
		// Lines start at the right of an element laid out right to left,
		// whatever they read, as in CSS, AppKit's cells and GTK's labels.
		// Left to right, a paragraph reading right to left starts at the
		// right, as Apple's guidelines align a paragraph by its language.
		p.Align = text.Right
		if ts.align == End {
			p.Align = text.Left
		}
	case ts.align == End:
		p.Align = text.End
	}
	return p
}

// resolvedText merges the text styles of the element and its ancestors.
func (e *node) resolvedText() textStyle {
	var out textStyle
	for p := e; p != nil && out.set != setAll; p = p.parent {
		t := &p.ts
		take := t.set &^ out.set
		if take == 0 {
			continue
		}
		if take&setFamily != 0 {
			out.family = t.family
		}
		if take&setSize != 0 {
			out.size = t.size
		}
		if take&setWeight != 0 {
			out.weight = t.weight
		}
		if take&setItalic != 0 {
			out.italic = t.italic
		}
		if take&setColor != 0 {
			out.color = t.color
		}
		if take&setLineHeight != 0 {
			out.lineHeight, out.fixedLine = t.lineHeight, t.fixedLine
		}
		if take&setAlign != 0 {
			out.align = t.align
		}
		if take&setUnderline != 0 {
			out.underline, out.wavy = t.underline, t.wavy
		}
		if take&setSpacing != 0 {
			out.spacing = t.spacing
		}
		if take&setFeatures != 0 {
			out.features = t.features
		}
		if take&setLang != 0 {
			out.lang = t.lang
		}
		if take&setStrike != 0 {
			out.strike = t.strike
		}
		if take&setDecoColor != 0 {
			out.decoColor = t.decoColor
		}
		if take&setDecoThick != 0 {
			out.decoThick = t.decoThick
		}
		if take&setBackground != 0 {
			out.background = t.background
		}
		if take&setSelection != 0 {
			out.selection = t.selection
		}
		out.set |= take
	}
	return out
}

// leafWidths returns the max-content and min-content widths of the
// element's own content (text, image, input), without padding.
func (e *node) leafWidths() (maxW, minW float32) {
	switch e.kind {
	case kindText:
		if e.text == "" {
			return 0, 0
		}
		// Measured once a frame: the layout asks for both widths a few
		// times.
		if e.leafOK {
			return e.leaf[0], e.leaf[1]
		}
		sys := textSystem()
		p := e.textParams(0)
		p.MaxLines = 0
		full := sys.Layout(p).Width
		min := float32(0)
		switch {
		case e.noWrap:
			min = full
		case !e.single:
			p.Width, p.NoBreakWords = 1, true
			min = sys.Layout(p).Width
		}
		e.leaf, e.leafOK = [2]float32{full, min}, true
		return full, min
	case kindImage:
		w, _ := e.intrinsicSize()
		return w, 0
	case kindIcon:
		// Icons keep their size where room is short, as text does.
		w, _ := e.intrinsicSize()
		return w, w
	case kindInput:
		return 200, 0
	}
	return 0, 0
}

// intrinsic returns the element's max-content or min-content width, its
// border box.
func intrinsic(e *node, maxContent bool) float32 {
	if e.form != nil {
		e.form.alignLabels()
	}
	if e.colFit != nil {
		e.colFit.apply()
	}
	if v, ok := e.width.resolve(-1); ok {
		return e.clampW(v, inf)
	}
	var w float32
	switch {
	case e.kind != kindBox:
		mx, mn := e.leafWidths()
		w = mn
		if maxContent {
			w = mx
		}
	case e.grid:
		w = gridIntrinsic(e, maxContent)
		if !maxContent && e.scrolls() {
			w = 0
		}
	default:
		n := 0
		for ch := e.first; ch != nil; ch = ch.next {
			if ch.flags&flagAbsolute != 0 {
				continue
			}
			cw := intrinsic(ch, maxContent) + ch.marginX()
			if e.row && (maxContent || !e.wrap) {
				w += cw
				n++
			} else {
				w = max(w, cw)
			}
		}
		if e.row && n > 1 && (maxContent || !e.wrap) {
			w += e.gapX * float32(n-1)
		}
		if !maxContent && e.scrolls() {
			w = 0
		}
	}
	return e.clampW(w+e.padX(), inf)
}

// fitWidth returns the width of an element sized by its content within
// avail DIPs, against a containing block cbW wide.
func fitWidth(e *node, avail, cbW float32) float32 {
	if v, ok := e.width.resolve(base(cbW)); ok {
		return e.clampW(v, cbW)
	}
	if e.aspect > 0 {
		if h, ok := e.height.resolve(-1); ok {
			return e.clampW(h*e.aspect, cbW)
		}
	}
	mx := intrinsic(e, true)
	if !finite(avail) {
		return e.clampW(mx, cbW)
	}
	mn := intrinsic(e, false)
	return e.clampW(min(mx, max(mn, avail)), cbW)
}

// heightAt returns the height of the element when it is w wide, against
// a containing block cbH high.
func heightAt(e *node, w, cbH float32) float32 {
	if v, ok := e.height.resolve(base(cbH)); ok {
		return e.clampH(v, cbH)
	}
	if e.aspect > 0 {
		return e.clampH(w/e.aspect, cbH)
	}
	for i := 0; i < e.nmeasure; i++ {
		if m := e.measures[i]; m.availW == w {
			return m.h
		}
	}
	h := contentHeight(e, w-e.padX()) + e.padY()
	h = e.clampH(h, cbH)
	if e.nmeasure < len(e.measures) {
		e.measures[e.nmeasure] = measure{availW: w, h: h}
		e.nmeasure++
	}
	return h
}

// contentHeight returns the height the content takes in a content box cw
// wide.
func contentHeight(e *node, cw float32) float32 {
	switch e.kind {
	case kindText:
		return textSystem().Layout(e.textParams(max(cw, 1))).Height
	case kindImage, kindIcon:
		if w, h := e.intrinsicSize(); w > 0 {
			return cw * h / w
		}
		return 0
	case kindInput:
		return e.inputHeight(cw)
	}
	if f := e.list; f != nil && f.n > 0 {
		// A List is as high as all its rows, as far as the heights known
		// tell, whichever it built.
		s := f.s
		return float32(s.heights.top(f.n, float64(max(e.gapY, 0))) - float64(max(e.gapY, 0)))
	}
	if e.first == nil {
		return 0
	}
	_, h := boxLayout(e, cw, inf, false)
	return h
}

// boxLayout lays out the children of a box, with flexbox or as a grid.
func boxLayout(e *node, cw, ch float32, commit bool) (usedW, usedH float32) {
	if e.form != nil {
		e.form.alignLabels()
	}
	if e.colFit != nil {
		e.colFit.apply()
	}
	if e.grid {
		return gridLayout(e, cw, ch, commit)
	}
	return flexLayout(e, cw, ch, commit)
}

// layoutBox gives the element its size and lays out its content.
func layoutBox(e *node, w, h float32) {
	e.w, e.h = w, h
	cw, ch := max(w-e.padX(), 0), max(h-e.padY(), 0)
	switch e.kind {
	case kindText:
		// Lists may build paragraphs while laying out, after the view's pass.
		e.prepareSelectable()
		e.tl = textSystem().Layout(e.textParams(max(cw, 1)))
		if ed := e.st.editor; ed != nil && e.flags&flagSelectable != 0 {
			// Selectable text hit-tests and selects in what it shows.
			ed.layout = e.tl
			ed.originX, ed.originY = e.contentX(), e.contentY()
		}
		return
	case kindInput:
		e.layoutInput(cw, ch)
		return
	case kindImage, kindIcon:
		return
	}
	if e.list != nil {
		e.layoutList(w, h)
		return
	}
	if e.overflow != nil {
		e.layoutToolbar(cw)
	}
	lw, lh := cw, ch
	if e.flags&flagScrollX != 0 {
		lw = inf
	}
	if e.flags&flagScrollY != 0 {
		lh = inf
	}
	uw, uh := boxLayout(e, lw, lh, true)
	if e.flags&flagScrollX != 0 && uw < cw && e.rtl() {
		// Laid out from the right of what the content takes, which goes
		// at the right of the container.
		for c := e.first; c != nil; c = c.next {
			if c.flags&flagAbsolute == 0 {
				c.x += cw - uw
			}
		}
	}
	if e.baselines {
		alignBaselines(e)
	}
	if e.scrolls() {
		e.contentW = float64(max(uw, cw) + e.padX())
		e.contentH = float64(max(uh, ch) + e.padY())
	}
	layoutAbsolute(e)
}

type flexItem struct {
	e                    *node
	base, hyp            float32
	minMain, maxMain     float32
	main, cross          float32
	frozen               bool
	marginMain, marginCr float32
	line                 int
}

type flexLine struct {
	start, end int
	main       float32 // sum of outer main sizes and gaps
	cross      float32
	pos        float32 // where it starts across the container
}

// flexScratch holds the items and lines of the flex containers being laid
// out, a nested container's after its parent's, reusing their memory
// frame after frame.
type flexScratch struct {
	items []flexItem
	lines []flexLine
}

// The margins of an item on the sides of its container's main and cross
// axes, by index into Element.margin: start (left or top) and end.
var (
	mainMargins  = [2][2]int{{0, 2}, {3, 1}} // [row] → start, end
	crossMargins = [2][2]int{{3, 1}, {0, 2}}
)

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// flexLayout lays out the in-flow children in a content box cw×ch (inf
// when unknown) and returns the size they take. With commit it gives them
// their boxes (relative to e) and lays them out in turn.
func flexLayout(e *node, cw, ch float32, commit bool) (usedW, usedH float32) {
	row := e.row
	mainSize, crossSize := ch, cw
	gap, crossGap := e.gapY, e.gapX
	if row {
		mainSize, crossSize = cw, ch
		gap, crossGap = e.gapX, e.gapY
	}
	align := e.align
	if align == alignAuto {
		align = Stretch
	}
	scratch := &e.c.rt.flex
	firstItem, firstLine := len(scratch.items), len(scratch.lines)
	defer func() { scratch.items, scratch.lines = scratch.items[:firstItem], scratch.lines[:firstLine] }()
	for c := e.first; c != nil; c = c.next {
		if c.flags&flagAbsolute != 0 {
			continue
		}
		it := flexItem{e: c}
		if row {
			it.marginMain, it.marginCr = c.marginX(), c.marginY()
		} else {
			it.marginMain, it.marginCr = c.marginY(), c.marginX()
		}
		// The flex base size.
		if v, ok := c.basis.resolve(base(mainSize)); ok {
			it.base = v
		} else if row {
			if v, ok := c.width.resolve(base(cw)); ok {
				it.base = v
			} else {
				it.base = intrinsic(c, true)
			}
		} else {
			if v, ok := c.height.resolve(base(ch)); ok {
				it.base = v
			} else {
				it.base = heightAt(c, crossOf(c, cw, align), ch)
			}
		}
		// The automatic minimum is the content's, unless it scrolls or
		// clips that way. Measuring the content cannot see the main size
		// this container has, so a maximum that is a percentage of it does
		// not bind there: clamp the minimum by the maximum here, where the
		// main size the maximum resolved against is known. A minimum the
		// element asked for itself still wins over the maximum.
		it.minMain, it.maxMain = 0, inf
		if row {
			if v, ok := c.maxW.resolve(base(cw)); ok {
				it.maxMain = v
			}
			if v, ok := c.minW.resolve(base(cw)); ok {
				it.minMain = v
			} else if !c.scrolls() && c.flags&flagClipX == 0 {
				it.minMain = min(intrinsic(c, false), it.maxMain)
				if v, ok := c.width.resolve(base(cw)); ok {
					it.minMain = min(it.minMain, v)
				}
			}
		} else {
			if v, ok := c.maxH.resolve(base(ch)); ok {
				it.maxMain = v
			}
			if v, ok := c.minH.resolve(base(ch)); ok {
				it.minMain = v
			} else if !c.scrolls() && c.flags&flagClipY == 0 && c.grow == 0 {
				it.minMain = min(heightAt(c, crossOf(c, cw, align), inf), it.maxMain)
				if v, ok := c.height.resolve(base(ch)); ok {
					it.minMain = min(it.minMain, v)
				}
			}
		}
		it.hyp = max(it.minMain, min(it.base, it.maxMain))
		// After the child's measures, which lay out containers in it.
		scratch.items = append(scratch.items, it)
	}
	items := scratch.items[firstItem:]

	// Break into lines.
	if len(items) > 0 {
		cur := flexLine{}
		for i := range items {
			outer := items[i].hyp + items[i].marginMain
			if e.wrap && finite(mainSize) && i > cur.start && cur.main+gap+outer > mainSize {
				cur.end = i
				scratch.lines = append(scratch.lines, cur)
				cur = flexLine{start: i}
			}
			if i > cur.start {
				cur.main += gap
			}
			cur.main += outer
			items[i].line = len(scratch.lines) - firstLine
		}
		cur.end = len(items)
		scratch.lines = append(scratch.lines, cur)
	}
	lines := scratch.lines[firstLine:]

	// Resolve the flexible lengths of each line.
	for li := range lines {
		ln := &lines[li]
		its := items[ln.start:ln.end]
		for i := range its {
			its[i].main = its[i].hyp
		}
		if !finite(mainSize) {
			continue
		}
		gaps := gap * float32(len(its)-1)
		resolveFlexible(its, mainSize-gaps, ln.main-gaps < mainSize-gaps)
		ln.main = gaps
		for _, it := range its {
			ln.main += it.main + it.marginMain
		}
	}

	// Cross sizes: natural ones first, then stretched ones fill the line.
	for li := range lines {
		ln := &lines[li]
		for i := ln.start; i < ln.end; i++ {
			it := &items[i]
			c := it.e
			if row {
				it.cross = heightAt(c, it.main, ch)
			} else {
				if v, ok := c.width.resolve(base(cw)); ok {
					it.cross = c.clampW(v, cw)
				} else if stretches(c, align, false) && finite(cw) {
					it.cross = c.clampW(cw-it.marginCr, cw)
				} else {
					it.cross = fitWidth(c, cw-it.marginCr, cw)
				}
			}
			ln.cross = max(ln.cross, it.cross+it.marginCr)
		}
	}
	if len(lines) == 1 && finite(crossSize) && !e.wrap {
		lines[0].cross = crossSize
	}
	// The lines across the container, as AlignContent places them.
	var linesCross float32
	for li, ln := range lines {
		if li > 0 {
			linesCross += crossGap
		}
		linesCross += ln.cross
	}
	var crossStart, crossBetween float32
	if ac := e.alignContent; ac != alignAuto && finite(crossSize) && len(lines) > 0 && (e.wrap || len(lines) > 1) {
		free := crossSize - linesCross
		if ac == Stretch {
			if free > 0 {
				for li := range lines {
					lines[li].cross += free / float32(len(lines))
				}
				linesCross = crossSize
			}
		} else {
			crossStart, crossBetween = justifyOffsets(ac, max(free, 0), len(lines))
			if free < 0 && (ac == Center || ac == End) {
				crossStart, _ = justifyOffsets(ac, free, len(lines))
			}
		}
	}
	for li := range lines {
		ln := &lines[li]
		for i := ln.start; i < ln.end; i++ {
			it := &items[i]
			c := it.e
			if !stretches(c, align, row) {
				continue
			}
			if row {
				if _, ok := c.height.resolve(base(ch)); !ok && c.aspect == 0 {
					it.cross = c.clampH(ln.cross-it.marginCr, ch)
				}
			} else if _, ok := c.width.resolve(base(cw)); !ok {
				it.cross = c.clampW(ln.cross-it.marginCr, cw)
			}
		}
	}

	// The size the lines take.
	var usedMain float32
	for _, ln := range lines {
		usedMain = max(usedMain, ln.main)
	}
	usedCross := linesCross
	if row {
		usedW, usedH = usedMain, usedCross
	} else {
		usedW, usedH = usedCross, usedMain
	}
	if !commit {
		return usedW, usedH
	}

	// Place the items, along the main axis from its start (the right of a
	// reversed row, the bottom of a reversed column), and the lines across
	// it from theirs (the bottom of a row wrapping in reverse). Right to
	// left, a row starts at the right, as does a column across it.
	flipMain := e.reverse != (row && e.rtl())
	flipCross := e.wrapReverse != (!row && e.rtl())
	contentMain := mainSize
	if !finite(contentMain) {
		contentMain = usedMain
	}
	contentCross := crossSize
	if !finite(contentCross) {
		contentCross = usedCross
	}
	r := b2i(row)
	ms, me := mainMargins[r][0], mainMargins[r][1]
	cs, ce := crossMargins[r][0], crossMargins[r][1]
	if flipMain {
		ms, me = me, ms
	}
	if flipCross {
		cs, ce = ce, cs
	}
	crossPos := crossStart
	for _, ln := range lines {
		its := items[ln.start:ln.end]
		free := max(contentMain-ln.main, 0)
		// Automatic margins take the free space first.
		autos := 0
		for i := range its {
			autos += b2i(isAuto(its[i].e.margin[ms])) + b2i(isAuto(its[i].e.margin[me]))
		}
		pos, between := justifyOffsets(e.justify, free, len(its))
		auto := float32(0)
		if autos > 0 {
			pos, between, auto = 0, 0, free/float32(autos)
		}
		for i := range its {
			it := &its[i]
			c := it.e
			startM, endM := c.m(ms), c.m(me)
			if isAuto(c.margin[ms]) {
				startM = auto
			}
			if isAuto(c.margin[me]) {
				endM = auto
			}
			mainOff := pos + startM
			if flipMain {
				mainOff = contentMain - mainOff - it.main
			}
			var crossOff float32
			free := ln.cross - it.cross - it.marginCr
			switch autoS, autoE := isAuto(c.margin[cs]), isAuto(c.margin[ce]); {
			case autoS && autoE:
				crossOff = max(free, 0) / 2
			case autoS:
				crossOff = max(free, 0)
			case autoE:
			default:
				switch selfAlign(c, align) {
				case Center:
					crossOff = free / 2
				case End:
					crossOff = free
				}
			}
			crossOff += crossPos + c.m(cs)
			if flipCross {
				crossOff = contentCross - crossOff - it.cross
			}
			if row {
				c.x = e.contentX() + mainOff
				c.y = e.contentY() + crossOff
				layoutBox(c, it.main, it.cross)
			} else {
				c.x = e.contentX() + crossOff
				c.y = e.contentY() + mainOff
				layoutBox(c, it.cross, it.main)
			}
			dx, dy := c.relative(cw, ch)
			c.x += dx
			c.y += dy
			pos += startM + it.main + endM + gap + between
		}
		crossPos += ln.cross + crossGap + crossBetween
	}
	return usedW, usedH
}

// crossOf returns the width a column child gets: stretched to the column
// or fitting its content.
func crossOf(c *node, cw float32, align Align) float32 {
	if v, ok := c.width.resolve(base(cw)); ok {
		return c.clampW(v, cw)
	}
	if stretches(c, align, false) && finite(cw) {
		return c.clampW(cw-c.marginX(), cw)
	}
	return fitWidth(c, cw-c.marginX(), cw)
}

// stretches reports whether c stretches across the line of a row (or a
// column), as aligning by align asks, which icons do not, as they keep
// their size, nor elements with automatic margins across it.
func stretches(c *node, align Align, row bool) bool {
	m := crossMargins[b2i(row)]
	return selfAlign(c, align) == Stretch && c.kind != kindIcon && !isAuto(c.margin[m[0]]) && !isAuto(c.margin[m[1]])
}

func selfAlign(c *node, align Align) Align {
	if c.self != alignAuto {
		return c.self
	}
	return align
}

// resolveFlexible grows or shrinks the items of a line to fill space,
// honoring their minimum and maximum sizes (CSS flexbox §9.7).
func resolveFlexible(its []flexItem, space float32, growing bool) {
	for i := range its {
		it := &its[i]
		it.frozen = false
		factor := it.e.grow
		if !growing {
			factor = it.e.shrink * it.base
		}
		if factor == 0 || (growing && it.base > it.hyp) || (!growing && it.base < it.hyp) {
			it.main, it.frozen = it.hyp, true
		}
	}
	for range its {
		free := space
		var sum float32
		for _, it := range its {
			if it.frozen {
				free -= it.main + it.marginMain
			} else {
				free -= it.base + it.marginMain
				if growing {
					sum += it.e.grow
				} else {
					sum += it.e.shrink * it.base
				}
			}
		}
		if sum == 0 {
			break
		}
		var violation float32
		for i := range its {
			it := &its[i]
			if it.frozen {
				continue
			}
			if growing {
				it.main = it.base + free*it.e.grow/sum
			} else {
				it.main = it.base + free*it.e.shrink*it.base/sum
			}
			clamped := max(it.minMain, min(it.main, it.maxMain))
			violation += clamped - it.main
			it.main = clamped
		}
		if violation == 0 {
			break
		}
		for i := range its {
			it := &its[i]
			if it.frozen {
				continue
			}
			if (violation > 0 && it.main == it.minMain) || (violation < 0 && it.main == it.maxMain) {
				it.frozen = true
			}
		}
	}
}

// justifyOffsets returns where the first item goes and the extra space
// between items.
func justifyOffsets(j Align, free float32, n int) (start, between float32) {
	switch j {
	case Center:
		return free / 2, 0
	case End:
		return free, 0
	case SpaceBetween:
		if n > 1 {
			return 0, free / float32(n-1)
		}
	case SpaceAround:
		if n > 0 {
			return free / float32(n) / 2, free / float32(n)
		}
	case SpaceEvenly:
		return free / float32(n+1), free / float32(n+1)
	}
	return 0, 0
}

// layoutAbsolute places the absolute children in the padding box. Right
// to left, as in CSS, one with neither Left nor Right goes to the right,
// and Right wins over Left where its width leaves no room for both.
func layoutAbsolute(e *node) {
	pw, ph := e.w-e.border[1]-e.border[3], e.h-e.border[0]-e.border[2]
	for c := e.first; c != nil; c = c.next {
		if c.flags&flagAbsolute == 0 || c.leaving != 0 {
			// Copies of elements leaving are placed by their transitions.
			continue
		}
		if c.attach != 0 {
			if c.popover != nil {
				attachTo(c, e, pw, ph)
			} else {
				attach(c, e, pw, ph)
			}
			continue
		}
		pw, ph := pw, ph
		var ox, oy float32
		if s := c.placeOn; s != nil {
			pw, ph = s.w, s.h
			ox, oy = s.x-e.border[3], s.y-e.border[0]
		}
		top, tok := c.inset[0].resolve(ph)
		right, rok := c.inset[1].resolve(pw)
		bottom, bok := c.inset[2].resolve(ph)
		left, lok := c.inset[3].resolve(pw)
		var w float32
		if v, ok := c.width.resolve(pw); ok {
			w = c.clampW(v, pw)
		} else if lok && rok {
			w = c.clampW(pw-left-right-c.marginX(), pw)
		} else {
			w = fitWidth(c, pw-c.marginX(), pw)
		}
		var h float32
		if v, ok := c.height.resolve(ph); ok {
			h = c.clampH(v, ph)
		} else if tok && bok {
			h = c.clampH(ph-top-bottom-c.marginY(), ph)
		} else {
			h = heightAt(c, w, ph)
		}
		x := left + c.m(3)
		rtl := c.rtl()
		if !lok && (rok || rtl) {
			x = pw - right - w - c.m(1)
		} else if lok && rok {
			// Automatic margins share the room left between the insets.
			free := pw - left - right - w - c.marginX()
			switch autoL, autoR := isAuto(c.margin[3]), isAuto(c.margin[1]); {
			case autoL && autoR:
				x += max(free, 0) / 2
			case autoL:
				x += max(free, 0)
			case !autoR && rtl:
				x += free
			}
		}
		y := top + c.m(0)
		if !tok && bok {
			y = ph - bottom - h - c.m(2)
		} else if tok && bok {
			free := ph - top - bottom - h - c.marginY()
			switch autoT, autoB := isAuto(c.margin[0]), isAuto(c.margin[2]); {
			case autoT && autoB:
				y += max(free, 0) / 2
			case autoT:
				y += max(free, 0)
			}
		}
		c.x, c.y = e.border[3]+ox+x, e.border[0]+oy+y
		layoutBox(c, w, h)
	}
}

// attachTo places c, a child of e whose padding box is pw×ph, attached to
// a point of the box of the element it is a popover of (AttachTo).
func attachTo(c, e *node, pw, ph float32) {
	var w float32
	if v, ok := c.width.resolve(pw); ok {
		w = c.clampW(v, pw)
	} else {
		w = fitWidth(c, pw, pw)
	}
	var h float32
	if v, ok := c.height.resolve(ph); ok {
		h = c.clampH(v, ph)
	} else {
		h = heightAt(c, w, ph)
	}
	t := laidOutVisualBox(c.popover)
	at, self := c.attach.anchors()
	at, self = at.physical(c.rtl()), self.physical(c.rtl())
	ax, ay := at.fractions()
	sx, sy := self.fractions()
	x := alongTarget(t.X, t.W, ax, sx, w, c.m(3)-c.m(1), c.c.w)
	y := alongTarget(t.Y, t.H, ay, sy, h, c.m(0)-c.m(2), c.c.h)
	dx, dy := c.relative(pw, ph)
	ox, oy := laidOutOrigin(e)
	c.x, c.y = x+dx-ox, y+dy-oy
	layoutBox(c, w, h)
}

// windowMargin keeps what overlays place apart from the window's edges.
const windowMargin = 4

// alongTarget returns where an element size long goes along an axis of the
// window, limit long, with its point at fraction self on target's point at
// fraction at, moved by off: on the other side of target, or the other way
// along it, where that overflows the window less, then moved into the
// window along target.
func alongTarget(t0, tsize, at, self, size, off, limit float32) float32 {
	pos := func(at, self, off float32) float32 { return t0 + at*tsize - self*size + off }
	over := func(p float32) float32 {
		return max(0, windowMargin-p) + max(0, p+size-(limit-windowMargin))
	}
	p := pos(at, self, off)
	if over(p) > 0 {
		if q := pos(1-at, 1-self, -off); over(q) < over(p) {
			p = q
		}
	}
	beside := at == 1 && self == 0 || at == 0 && self == 1
	if !beside && over(p) > 0 {
		p = max(windowMargin, min(p, limit-windowMargin-size))
	}
	return p
}

// laidOutOrigin returns where place will put the box of an element laid
// out, in the window: its position and those of the elements around it,
// less the offsets of the scroll containers among them.
func laidOutOrigin(e *node) (x, y float32) {
	x, y = e.x, e.y
	for ch, p := e, e.parent; p != nil; ch, p = p, p.parent {
		if ch.flags&flagAbsolute == 0 || ch.leaving == 1 {
			if f := p.followX; f != nil {
				x -= float32(f.followShift())
			}
			if p.scrolls() {
				x -= float32(p.scrollLeft())
				y -= float32(p.st.scrollY - p.scrollBase)
			}
		}
		x += p.x
		y += p.y
	}
	return x, y
}

// laidOutBox returns where place, or commit for an inline element, will
// put the box of an element of the frame, in the window, as its overlays
// are laid out; that of the last frame for one this frame did not build.
func laidOutBox(e *node) Rect {
	if e.st.seen != e.c.rt.frame || e.parent == nil {
		return e.Bounds()
	}
	if !e.isInline() {
		x, y := laidOutOrigin(e)
		return Rect{x, y, e.w, e.h}
	}
	// The box around its text in the paragraph's layout.
	start, end := e.runes[0], e.runes[1]
	para := e.parent
	for para.isInline() {
		start, end = start+para.runes[0], end+para.runes[0]
		para = para.parent
	}
	x, y := laidOutOrigin(para)
	x, y = x+para.contentX(), y+para.contentY()
	if para.tl == nil {
		return Rect{x, y, 0, 0}
	}
	var box Rect
	for i, r := range para.tl.Selection(start, end) {
		if f := (Rect{x + r.X, y + r.Y, r.W, r.H}); i == 0 {
			box = f
		} else {
			box = union(box, f)
		}
	}
	return box
}

// attach places c, attached to a point of the padding box of its parent e,
// pw×ph (Attach).
func attach(c, e *node, pw, ph float32) {
	var w float32
	if v, ok := c.width.resolve(pw); ok {
		w = c.clampW(v, pw)
	} else {
		w = fitWidth(c, pw, pw)
	}
	var h float32
	if v, ok := c.height.resolve(ph); ok {
		h = c.clampH(v, ph)
	} else {
		h = heightAt(c, w, ph)
	}
	at, self := c.attach.anchors()
	at, self = at.physical(c.rtl()), self.physical(c.rtl())
	ax, ay := at.fractions()
	sx, sy := self.fractions()
	dx, dy := c.relative(pw, ph)
	c.x = e.border[3] + ax*pw - sx*w + dx
	c.y = e.border[0] + ay*ph - sy*h + dy
	layoutBox(c, w, h)
}
