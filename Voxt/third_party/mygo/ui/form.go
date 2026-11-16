package ui

import "github.com/egoist/mygo/internal/text"

// formBuild is a Form being built: the boxes of its fields' labels, which
// its layout makes as wide as the widest.
type formBuild struct {
	labels  []*node
	aligned bool
}

// fieldParts are the parts of a Field: its label, the column holding its
// control and the texts below it, and the control, which the label names
// and the texts describe.
type fieldParts struct {
	label, content, control *node
}

// Form creates a column of the fields that fn builds (Field, Fieldset),
// whose labels sit in a column of their own, beside the controls, as in
// macOS's forms: aligned to the right, as wide as the widest, and level
// with the first line of their control.
//
//	ui.Form(c, func() {
//		ui.Field(c, "Name", func() { ui.TextInput(c, &app.name) })
//		ui.Field(c, "Size", func() { ui.Select(c, &app.size, sizes) })
//	})
func coreForm(c *context, fn func()) *node {
	f := coreColumn(c).Gap(c.theme.Space(3))
	f.widget = "Form"
	fb := &formBuild{}
	f.form = fb
	saved := c.form
	c.form = fb
	f.Children(fn)
	c.form = saved
	return f
}

// alignLabels makes the labels of a form as wide as the widest, once a
// frame, before the form is measured or laid out.
func (fb *formBuild) alignLabels() {
	if fb.aligned {
		return
	}
	fb.aligned = true
	var w float32
	for _, l := range fb.labels {
		w = max(w, intrinsic(l, true))
	}
	for _, l := range fb.labels {
		l.width = px(w)
	}
}

// Field creates a field: label, above the control that fn builds, or
// beside it in a Form. The label names the control for assistive
// technology, unless the control names itself, as a check box with its own
// text does, when it names the field around it; a click on the label
// focuses the control, or clicks a check box, a switch or a radio button.
// Description and Error add texts below the control, which describe it:
//
//	ui.Field(c, "Email", func() {
//		ui.TextInput(c, &app.email)
//	}).Description("For receipts.").Error(app.emailError)
//
// The control is the first element fn builds that takes the focus, or a
// group of them, such as a RadioGroup.
func coreField(c *context, label string, fn func()) *node {
	t := c.theme
	fp := &fieldParts{}
	var f *node
	if c.form != nil {
		f = coreRow(c).Gap(t.Space(3)).AlignItems(Start)
		f.baselines = true
	} else {
		f = coreColumn(c).Gap(t.Space(1.5))
	}
	f.widget, f.field = "Field", fp
	f.Children(func() {
		if c.form != nil {
			box := coreRow(c).Justify(End).Shrink(0)
			c.form.labels = append(c.form.labels, box)
			box.Children(func() { fp.label = coreText(c, label) })
		} else {
			fp.label = coreText(c, label).FontWeight(500)
		}
		fp.content = coreColumn(c).Gap(t.Space(1.5))
		if c.form != nil {
			fp.content.Grow(1).MinWidth(0) // the rest of the row
		}
		fp.content.Children(fn)
	})
	ctrl := fieldControl(fp.content)
	fp.control = ctrl
	l := fp.label
	if ctrl == nil {
		return f
	}
	if namesItself(ctrl) {
		f.Role(RoleGroup).Label(label)
	} else if ctrl.nameFrom == nil || ctrl.nameFrom.label == "" {
		ctrl.nameFrom = l
	}
	l.flags |= flagClickable
	if l.Clicked() && !ctrl.disabled() {
		if to := focusIn(ctrl); to != nil {
			to.Focus()
		}
		if ctrl.flags&flagToggle != 0 {
			c.rt.clickLater = append(c.rt.clickLater, ctrl.id)
		}
		c.rt.consumed = true
	}
	l.styleFn = func(l *node) {
		if ctrl.IsDisabled() && !l.IsDisabled() {
			l.Opacity(0.5)
		}
	}
	return f
}

// fieldControl returns the control of a field: the first element inside e
// that takes the focus, or a group of them.
func fieldControl(e *node) *node {
	for ch := e.first; ch != nil; ch = ch.next {
		switch ch.role {
		case RoleRadioGroup, RoleToolbar, RoleList, RoleTable, RoleTree, RoleTabList:
			return ch
		}
		if ch.flags&(flagFocusable|flagEditable) != 0 {
			return ch
		}
		if ctrl := fieldControl(ch); ctrl != nil {
			return ctrl
		}
	}
	return nil
}

// focusIn returns what a click on the label of a field focuses: its
// control, or in a group of them, the one chosen, else the first.
func focusIn(e *node) *node {
	if e.flags&(flagFocusable|flagEditable) != 0 {
		return e
	}
	var first, chosen *node
	var walk func(e *node)
	walk = func(e *node) {
		for ch := e.first; ch != nil && chosen == nil; ch = ch.next {
			if ch.flags&flagFocusable != 0 && !ch.disabled() {
				if first == nil {
					first = ch
				}
				if ch.checked == 2 {
					chosen = ch
				}
			}
			walk(ch)
		}
	}
	walk(e)
	if chosen != nil {
		return chosen
	}
	return first
}

// namesItself reports whether a control has a name of its own, as a
// check box with its own text has, rather than a value.
func namesItself(e *node) bool {
	if e.label != "" {
		return true
	}
	switch e.role {
	case RoleButton, RoleLink, RoleCheckBox, RoleRadio, RoleSwitch, RoleToggleButton, RoleMenuButton, RoleTab, RoleDisclosure:
		return e.innerText() != ""
	case RoleAuto:
		return e.flags&flagClickable != 0 && e.flags&flagEditable == 0 && e.innerText() != ""
	}
	return false
}

// nameOf returns the name an element gives another (nameFrom): its Label,
// or the text it is.
func (e *node) nameOf() string {
	if e.label == "" && e.kind == kindText {
		return e.text
	}
	return e.label
}

// Fieldset creates a group of the fields that fn builds, under legend,
// which names the group for assistive technology. Disabled disables them
// all. In a Form, their labels line up with those of the other fields.
//
//	ui.Fieldset(c, "Shipping", func() {
//		ui.Field(c, "Address", func() { ui.TextInput(c, &app.address) })
//	})
func coreFieldset(c *context, legend string, fn func()) *node {
	t := c.theme
	above := c.parent != nil && c.parent.nchild > 0
	g := coreColumn(c).Gap(t.Space(3)).Shrink(0)
	g.widget, g.role, g.label = "Fieldset", RoleGroup, legend
	if above {
		// Apart from what is above, as a section.
		g.Margin(t.Space(3), 0, 0, 0)
	}
	g.Children(func() {
		coreText(c, legend).FontWeight(600)
		fn()
	})
	return g
}

// Description tells assistive technology more about the element than its
// name, as help text it reads after it. On a Field, the text shows below
// the control, and describes it.
func (e *node) Description(s string) *node {
	if s == "" {
		return e
	}
	target := e.fieldText(s, e.c.theme.TextMuted)
	target.description = joinDescription(target.description, s)
	return e
}

// Error marks the element's value as not valid, for msg, which assistive
// technology reads with it; an empty msg leaves it valid. On a Field, the
// message shows below the control in the theme's Danger color, and marks
// the control, whose border text inputs draw in that color.
func (e *node) Error(msg string) *node {
	if msg == "" {
		return e
	}
	target := e.fieldText(msg, e.c.theme.Danger)
	target.invalid = true
	// The error first, as what to read first.
	target.description = joinDescription(msg, target.description)
	return e
}

// fieldText shows s below the control of a field, and returns the element
// it describes: the control, or e.
func (e *node) fieldText(s string, color Color) *node {
	f := e.field
	if f == nil {
		return e
	}
	c := e.c
	f.content.Children(func() {
		coreText(c, s).TextColor(color).FontSize(c.theme.FontSize - 1)
	})
	if f.control != nil {
		return f.control
	}
	return e
}

func joinDescription(a, b string) string {
	switch {
	case a == "":
		return b
	case b == "":
		return a
	}
	return a + "\n" + b
}

// inputBorder is the border of the box of a text input: the theme's
// Danger while the input is not valid, its Accent while it has the focus,
// darker while the pointer is over the box.
func inputBorder(t *Theme, box, in *node) {
	switch {
	case in.invalid:
		box.borderC = t.Danger
	case in.Focused():
		box.borderC = t.Accent
	case box.Hovered():
		box.borderC = t.Border.Mix(t.Text, 0.25)
	}
}

// alignBaselines lines up the first lines of text of a row's children,
// as a field in a form does its label's and its control's, moving down
// those whose first line is higher. A child without text, as a switch,
// takes the baseline a line of the others' text centered on its first
// control would have.
func alignBaselines(e *node) {
	var ascent, descent float32
	for ch := e.first; ch != nil && ascent == 0; ch = ch.next {
		if l := firstLine(ch); l != nil && ch.flags&flagAbsolute == 0 {
			ascent, descent = l.Ascent, l.Descent
		}
	}
	if ascent == 0 {
		return
	}
	at := func(ch *node) (float32, bool) {
		if b, ok := firstBaseline(ch); ok {
			return b, true
		}
		if y, h, ok := firstControl(ch); ok {
			return y + h/2 + (ascent-descent)/2, true
		}
		return 0, false
	}
	var top float32
	for ch := e.first; ch != nil; ch = ch.next {
		if ch.flags&flagAbsolute == 0 {
			if b, ok := at(ch); ok {
				top = max(top, ch.y+b)
			}
		}
	}
	for ch := e.first; ch != nil; ch = ch.next {
		if ch.flags&flagAbsolute == 0 {
			if b, ok := at(ch); ok {
				ch.y = top - b
			}
		}
	}
}

// firstLine returns the first line of text in the element, as last laid
// out, or nil.
func firstLine(e *node) *text.Line {
	if e.flags&flagInvisible != 0 {
		return nil
	}
	switch e.kind {
	case kindText:
		if e.tl != nil && len(e.tl.Lines) > 0 {
			return &e.tl.Lines[0]
		}
		return nil
	case kindInput:
		if ed := e.st.editor; ed != nil {
			return ed.firstLine()
		}
		return nil
	}
	for ch := e.first; ch != nil; ch = ch.next {
		if ch.flags&flagAbsolute == 0 {
			if l := firstLine(ch); l != nil {
				return l
			}
		}
	}
	return nil
}

// firstControl returns the top and the height of the first element inside
// e, or e itself, that takes the focus, is clicked or holds nothing,
// relative to e's top.
func firstControl(e *node) (y, h float32, ok bool) {
	if e.flags&(flagFocusable|flagEditable|flagClickable) != 0 || e.first == nil {
		return 0, e.h, e.h > 0
	}
	for ch := e.first; ch != nil; ch = ch.next {
		if ch.flags&(flagAbsolute|flagInvisible) == 0 {
			if y, h, ok := firstControl(ch); ok {
				return ch.y + y, h, true
			}
		}
	}
	return 0, 0, false
}

// firstBaseline returns how far below the element's top the baseline of
// its first line of text is, as last laid out, and false for an element
// without text.
func firstBaseline(e *node) (float32, bool) {
	if e.flags&flagInvisible != 0 {
		return 0, false
	}
	switch e.kind {
	case kindText:
		if e.tl != nil && len(e.tl.Lines) > 0 {
			return e.contentY() + e.tl.Lines[0].Baseline, true
		}
		return 0, false
	case kindInput:
		if ed := e.st.editor; ed != nil {
			if l := ed.firstLine(); l != nil {
				return ed.originY + l.Baseline, true
			}
		}
		return 0, false
	}
	for ch := e.first; ch != nil; ch = ch.next {
		if ch.flags&(flagAbsolute|flagInvisible) != 0 {
			continue
		}
		if b, ok := firstBaseline(ch); ok {
			return ch.y + b, true
		}
	}
	return 0, false
}
