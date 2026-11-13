package ui

import (
	"math"
	"slices"
	"time"
)

// sincos returns the sine and the cosine of deg degrees.
func sincos(deg float32) (float32, float32) {
	sin, cos := math.Sincos(float64(deg) * math.Pi / 180)
	return float32(sin), float32(cos)
}

// CollapsibleParts are the parts of a collapsible without a look: its
// Trigger shows and hides what Panel builds.
type collapsibleParts struct {
	// Trigger is a button opening and closing the collapsible when
	// clicked, or with Enter or Space while it has the focus, which
	// Changed reports, and that assistive technology sees as a disclosure,
	// expanded while open. Give it children.
	Trigger  *node
	c        *context
	open     *bool
	progress float32
}

// CollapsibleBase creates a collapsible without a look, open while *open:
// its Trigger, and below it the panel Panel builds. Collapsible is
// CollapsibleBase with the theme's look.
func coreCollapsibleBase(c *context, open *bool) collapsibleParts {
	tr := coreRow(c).Focusable().Shrink(0)
	tr.flags |= flagClickable | flagHover
	tr.widget, tr.role = "Collapsible", RoleDisclosure
	*valueBinding[*bool](tr) = open
	tr.onValueInput(toggleInput)
	tr.expanded = *open
	p := tr.Animate("open", b2f(*open), 200*time.Millisecond)
	return collapsibleParts{Trigger: tr, c: c, open: open, progress: p}
}

// Progress returns how far the collapsible is open, from 0 closed to 1
// open, moving between them as it opens and closes: draw its arrow from
// it. It moves at once where the desktop asks for less motion.
func (p collapsibleParts) Progress() float32 { return p.progress }

// Panel builds fn in the panel of the collapsible while it is open, and as
// it opens and closes, when the panel grows and shrinks with Progress,
// clipping what fn built. It returns the column holding what fn builds,
// to style, and nil while the collapsible is closed.
func (p collapsibleParts) Panel(fn func()) *node {
	if !*p.open && p.progress == 0 {
		return nil
	}
	c := p.c
	clip := coreColumn(c).Shrink(0)
	var inner *node
	clip.Children(func() {
		inner = coreColumn(c).Shrink(0)
		inner.Children(fn)
	})
	if p.progress < 1 {
		// The height the content had in the last frame, which is 0 as it
		// starts to open, and grows as it does.
		clip.ClipY().Height(p.progress * inner.Bounds().H)
	}
	return inner
}

// Collapsible creates a disclosure, as SwiftUI's DisclosureGroup: label
// beside an arrow, which a click opens and closes, as do Enter and Space,
// showing below it what fn builds while *open is true. The arrow turns
// and the content grows into view as it opens. Changed reports a click.
//
//	ui.Collapsible(c, "Advanced", &app.advanced, func() {
//		ui.Checkbox(c, &app.verbose, "Verbose logging")
//	})
func coreCollapsible(c *context, label string, open *bool, fn func()) *node {
	t := c.theme
	root := coreColumn(c).Shrink(0)
	root.widget = "Collapsible"
	root.Children(func() {
		p := coreCollapsibleBase(c, open)
		tr := p.Trigger.AlignSelf(Start).AlignItems(Center).Gap(t.Space(1)).
			Padding(t.Space(1), 0).PaddingStart(t.Space(0.5)).PaddingEnd(t.Space(2)).Radius(t.Radius)
		tr.styleFn = func(tr *node) {
			if tr.Hovered() {
				tr.bg = t.SurfaceHover
			}
		}
		tr.Children(func() {
			disclosureArrow(c, 90*p.Progress())
			coreText(c, label).SingleLine()
		})
		tr.afterInput(func() {
			if tr.Changed() {
				root.st.markChanged()
			}
		})
		if panel := p.Panel(fn); panel != nil {
			// Below the label.
			panel.Padding(t.Space(1), 0).PaddingStart(t.Space(5.5)).Gap(t.Space(2))
		}
	})
	return root
}

// disclosureArrow draws the arrow of a disclosure: pointing right, turned
// by deg degrees clockwise, or right to left pointing left, turned as far
// the other way, as Finder's do.
func disclosureArrow(c *context, deg float32) *node {
	t := c.theme
	e := coreBox(c).Size(t.Space(4), t.Space(4)).Shrink(0)
	rtl := e.rtl()
	return e.Draw(func(p *Painter, r Rect) {
		p.StrokePath(disclosurePath(r, deg, rtl), 1.5, t.TextMuted)
	})
}

// disclosurePath returns the arrow of a disclosure in r, turned by deg
// degrees from pointing right, or from pointing left the other way right
// to left (rtl).
func disclosurePath(r Rect, deg float32, rtl bool) *Path {
	cx, cy, d := r.X+r.W/2, r.Y+r.H/2, r.W/8
	s := float32(1)
	if rtl {
		s, deg = -1, -deg
	}
	at := rotate(cx, cy, deg)
	var path Path
	path.MoveTo(at(-s*d, -2*d)).LineTo(at(s*d, 0)).LineTo(at(-s*d, 2*d))
	return &path
}

// accordionBuild is an Accordion being built: the headers of its
// sections, in order, and those of the last frame, for the arrows.
type accordionBuild struct {
	headers, last []uint64
}

// Accordion creates sections, one above the other in a bordered box, that
// fn builds with AccordionItem. Each opens and closes on its own; Up and
// Down move the focus between their headers, as do Home and End.
//
//	ui.Accordion(c, func() {
//		ui.AccordionItem(c, "General", &app.general, func() { ... })
//		ui.AccordionItem(c, "Privacy", &app.privacy, func() { ... })
//	})
//
// For one section open at a time, close the others as one opens:
//
//	for i, s := range sections {
//		open := app.section == i
//		if ui.AccordionItem(c, s.Title, &open, s.Build).Changed() {
//			app.section = -1
//			if open {
//				app.section = i
//			}
//		}
//	}
func coreAccordion(c *context, fn func()) *node {
	t := c.theme
	a := coreColumn(c).Radius(t.Radius).Border(1, t.Border).Clip()
	a.widget = "Accordion"
	last := coreLocal(a, "headers", func() []uint64 { return nil })
	saved := c.accordion
	ab := &accordionBuild{last: *last}
	c.accordion = ab
	a.Children(fn)
	c.accordion = saved
	*last = ab.headers
	return a
}

// AccordionItem creates a section of an Accordion: a header showing title,
// which a click opens and closes, as do Enter and Space, and below it what
// fn builds while *open is true. Changed reports a click.
func coreAccordionItem(c *context, title string, open *bool, fn func()) *node {
	t := c.theme
	ab := c.accordion
	if ab == nil {
		ab = &accordionBuild{}
	}
	item := coreColumn(c).Shrink(0)
	item.widget = "AccordionItem"
	if len(ab.headers) > 0 {
		item.BorderWidth(1, 0, 0, 0).BorderColor(t.Border)
	}
	item.Children(func() {
		p := coreCollapsibleBase(c, open)
		tr := p.Trigger.AlignItems(Center).Gap(t.Space(2)).Padding(t.Space(2.5), t.Space(3))
		tr.flags |= flagOwnRing
		ab.headers = append(ab.headers, tr.id)
		accordionKeys(c, tr, ab)
		tr.styleFn = func(tr *node) {
			if tr.Hovered() {
				tr.bg = t.SurfaceHover
			}
		}
		tr.DrawOver(func(p *Painter, r Rect) {
			// Inside the header, as the accordion clips what is around.
			if tr.FocusVisible() {
				p.FocusRing(Rect{r.X + 4, r.Y + 4, r.W - 8, r.H - 8}, [4]float32{t.Radius, t.Radius, t.Radius, t.Radius})
			}
		})
		tr.Children(func() {
			coreText(c, title).Grow(1).FontWeight(500)
			disclosureArrow(c, 90+180*p.Progress())
		})
		tr.afterInput(func() {
			if tr.Changed() {
				item.st.markChanged()
			}
		})
		if panel := p.Panel(fn); panel != nil {
			panel.Padding(0, t.Space(3), t.Space(3)).Gap(t.Space(2))
		}
	})
	return item
}

// accordionKeys moves the focus from an accordion's header to the others
// with the arrows, Home and End.
func accordionKeys(c *context, tr *node, ab *accordionBuild) {
	at := slices.Index(ab.last, tr.id)
	if at < 0 {
		return
	}
	to := -1
	switch {
	case tr.Shortcut(0, KeyDown):
		to = min(at+1, len(ab.last)-1)
	case tr.Shortcut(0, KeyUp):
		to = max(at-1, 0)
	case tr.Shortcut(0, KeyHome):
		to = 0
	case tr.Shortcut(0, KeyEnd):
		to = len(ab.last) - 1
	}
	rt := c.rt
	if to >= 0 && rt.states[ab.last[to]] != nil {
		rt.focused, rt.focusVisible = ab.last[to], true
		rt.consumed = true
	}
}
