package ui

import (
	"time"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/scene"
	"github.com/egoist/mygo/internal/text"
)

// ContentTransition is how an element's content changes (Element
// ContentTransition): its text, its icon, or its picture.
//
// By default, the old content fades out as the new fades in; an icon
// shrinks away as the new grows in, as SwiftUI's replace symbol effect.
// Numeric rolls the characters of a line of text that change, as a
// counter's digits, those that stay staying: the new come up from below,
// or, Down, from above, as SwiftUI's numericText.
type ContentTransition struct {
	Numeric, Down bool
	// Animation is how the content changes: a spring arriving in about
	// 350 ms without bouncing when nil.
	Animation *Animation
}

// defaultContentAnimation is a ContentTransition's without an Animation.
var defaultContentAnimation = Animation{Spring: SpringOptions{Duration: 350 * time.Millisecond}}

// ContentTransition animates the element's content as it changes, as t
// says: the text of a Text, the icon of an Icon, the picture of an Image.
// Changes animate from frame to frame while the element is painted, in
// frames that only paint. While the desktop asks for less motion, content
// fades without moving.
//
//	ui.Text(c, strconv.Itoa(app.count)).ContentTransition(ui.ContentTransition{Numeric: true, Down: app.decreased})
func (e *node) ContentTransition(t ContentTransition) *node {
	e.content, e.contentSet = t, true
	return e
}

// contentRec is what an element's ContentTransition keeps from frame to
// frame: what it showed (text and its layout, icon, picture) and what it
// showed before, which goes as k goes from 0 to 1; frame is the last
// painting (engine.paints) that painted it.
type contentRec struct {
	frame         uint64
	text, oldText string
	tl, oldTL     *text.Layout
	svg, oldSVG   *SVG
	image, oldImg *Bitmap
	oldX, oldY    float32
	oldColor      Color
	k             channel
	still         bool
}

// contentStep notes what e shows now, from (x, y), in color, and returns
// its record, with how far the content changed (k) and whether it is
// changing.
func (p *Painter) contentStep(e *node, x, y float32, color Color) (*contentRec, float32, bool) {
	rt := p.rt
	st := e.st
	r := st.crec
	if r == nil || r.frame+1 < rt.paints {
		// Not painted in the frame before: it shows at once.
		r = &contentRec{}
		st.crec = r
		r.frame, r.text, r.tl, r.svg, r.image = rt.paints, e.text, e.tl, e.svg, e.image
		r.k = channel{value: 1, target: 1, at: rt.c.now}
		return r, 1, false
	}
	now := rt.c.now
	if r.frame != rt.paints {
		r.frame = rt.paints
		changed := false
		switch e.kind {
		case kindText:
			changed = e.text != r.text
		case kindIcon, kindImage:
			changed = e.svg != r.svg || e.image != r.image
		}
		if changed {
			if rt.instant() {
				r.k = channel{value: 1, target: 1, at: now}
			} else {
				r.oldText, r.oldTL, r.oldSVG, r.oldImg = r.text, r.tl, r.svg, r.image
				r.oldX, r.oldY, r.oldColor = x, y, color
				r.still = rt.preferences().ReduceMotion
				a := defaultContentAnimation
				if e.content.Animation != nil {
					a = *e.content.Animation
				}
				r.k = channel{at: now}
				headFor(&r.k, 1, a, now)
			}
		}
		r.text, r.tl, r.svg, r.image = e.text, e.tl, e.svg, e.image
		r.k.advance(now)
	}
	if !r.k.moving && !r.k.pending {
		r.oldTL, r.oldSVG, r.oldImg = nil, nil, nil
		return r, 1, false
	}
	p.AnimationFrame()
	return r, max(0, min(float32(r.k.value), 1)), true
}

// contentText paints the text of e changing, from (ox, oy), and reports
// whether it does: no transition otherwise.
func (p *Painter) contentText(e *node, ox, oy float32, ts textStyle, sp *spanPaint) bool {
	if e.first != nil || sp != nil {
		return false // rich text: its spans are its own
	}
	r, k, changing := p.contentStep(e, ox, oy, ts.color)
	if !changing || r.oldTL == nil || e.tl == nil {
		return false
	}
	saved := p.opacity
	defer func() { p.opacity = saved }()
	old, cur := r.oldTL, e.tl
	if !e.content.Numeric || r.still || len(old.Lines) != 1 || len(cur.Lines) != 1 {
		p.opacity = saved * (1 - k)
		p.textLayout(old, r.oldX, r.oldY, r.oldColor, ts, nil)
		p.opacity = saved * k
		p.textLayout(cur, ox, oy, ts.color, ts, nil)
		return true
	}
	// The characters the old and the new text share at their start and at
	// their end stay; those between roll.
	a, b := r.oldText, r.text
	pre := 0
	for pre < len(a) && pre < len(b) && a[pre] == b[pre] {
		pre++
	}
	for pre > 0 && pre < len(b) && !utf8.RuneStart(b[pre]) {
		pre--
	}
	suf := 0
	for suf < len(a)-pre && suf < len(b)-pre && a[len(a)-1-suf] == b[len(b)-1-suf] {
		suf++
	}
	for suf > 0 && !utf8.RuneStart(b[len(b)-suf]) {
		suf--
	}
	caret := func(l *text.Layout, s string, i int) float32 {
		x, _, _ := l.Caret(utf8.RuneCountInString(s[:i]))
		return x
	}
	oldPre, oldSuf := caret(old, a, pre), caret(old, a, len(a)-suf)
	curPre, curSuf := caret(cur, b, pre), caret(cur, b, len(b)-suf)
	h := cur.Lines[0].Height
	dir := float32(1)
	if e.content.Down {
		dir = -1
	}
	const far = 1e5
	band := func(x0, x1 float32, fn func()) {
		if x1 <= x0 {
			return
		}
		savedClip := p.clip
		p.pushClip(Rect{x0, oy - far, x1 - x0, 2 * far}, [4]float32{})
		fn()
		p.popClip()
		p.clip = savedClip
	}
	// What stays: the start, and the end, sliding to where it goes.
	p.opacity = saved
	if pre > 0 {
		band(ox-far, ox+curPre, func() { p.textLayout(cur, ox, oy, ts.color, ts, nil) })
	}
	dx := lerp(oldSuf-curSuf, 0, k)
	if suf > 0 {
		band(ox+curSuf+dx, ox+far, func() { p.textLayout(cur, ox+dx, oy, ts.color, ts, nil) })
	}
	// What changes: the old going, the new coming.
	p.opacity = saved * (1 - k)
	band(ox+oldPre, ox+oldSuf, func() { p.textLayout(old, ox, oy-dir*h*0.6*k, r.oldColor, ts, nil) })
	p.opacity = saved * k
	band(ox+curPre, ox+curSuf+dx, func() { p.textLayout(cur, ox, oy+dir*h*0.6*(1-k), ts.color, ts, nil) })
	return true
}

// contentIcon paints the icon of e changing, in box, and reports whether
// it does: the old shrinking away, the new growing in.
func (p *Painter) contentIcon(e *node, box Rect, c Color) bool {
	r, k, changing := p.contentStep(e, box.X, box.Y, c)
	if !changing {
		return false
	}
	saved, savedT := p.opacity, p.transform
	defer func() { p.opacity, p.transform = saved, savedT }()
	scaled := func(s float32, fn func()) {
		p.transform = savedT
		if !r.still && s != 1 {
			cx, cy := box.X+box.W/2, box.Y+box.H/2
			p.transform = savedT.Mul(scene.Translation(cx, cy)).Mul(scene.Scaling(s, s)).Mul(scene.Translation(-cx, -cy))
		}
		fn()
	}
	if r.oldSVG != nil {
		p.opacity = saved * (1 - k)
		scaled(1-0.5*k, func() { p.drawIcon(r.oldSVG, box, r.oldColor, e.rotate) })
	}
	p.opacity = saved * k
	scaled(0.5+0.5*k, func() { p.drawIcon(e.svg, box, c, e.rotate) })
	return true
}

// contentImage paints the picture of e changing, and reports whether it
// does: the old fading out as the new fades in.
func (p *Painter) contentImage(e *node) bool {
	box := e.contentBox()
	r, k, changing := p.contentStep(e, box.X, box.Y, e.resolvedText().color)
	if !changing {
		return false
	}
	saved := p.opacity
	defer func() { p.opacity = saved }()
	p.opacity = saved * (1 - k)
	if r.oldSVG != nil {
		p.drawSVG(r.oldSVG, box, e.fit, e.radius, r.oldColor, e.gray)
	} else if r.oldImg != nil && r.oldImg.w > 0 && r.oldImg.h > 0 {
		p.drawBitmap(r.oldImg, box, e.fit, e.radius, e.gray)
	}
	p.opacity = saved * k
	p.image(e)
	return true
}
