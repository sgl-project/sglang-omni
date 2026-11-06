package ui

// TextInputBuffer creates a single-line control editing indexed storage.
// It shares TextInput's selection, composition, undo and styling behavior.
func coreTextInputBuffer(c *context, value *TextBuffer) *node {
	return styledBufferInput(c, value, false)
}

// TextAreaBuffer creates a multiline control editing indexed storage.
// Ordinary edits and native queries never materialize the whole document.
// Layout retains the paragraphs in view, as TextArea does.
func coreTextAreaBuffer(c *context, value *TextBuffer) *node {
	return styledBufferInput(c, value, true)
}

// TextInputBufferBase and TextAreaBufferBase are the unstyled buffer controls.
func coreTextInputBufferBase(c *context, value *TextBuffer) *node {
	return bufferInputBase(c, value, false)
}
func coreTextAreaBufferBase(c *context, value *TextBuffer) *node {
	return bufferInputBase(c, value, true)
}

func styledBufferInput(c *context, value *TextBuffer, multiline bool) *node {
	t := c.theme
	e := bufferInputBase(c, value, multiline)
	e.Padding(t.Space(1.5), t.Space(2.5)).Radius(t.Radius).Background(t.Surface).Border(1, t.Border)
	if multiline {
		e.MinHeight(t.Space(20))
	}
	e.styleFn = func(e *node) { inputBorder(t, e, e) }
	return e
}

func (ed *editor) loadBuffer(s TextSnapshot) {
	clear(ed.undo)
	clear(ed.redo)
	clear(ed.queue)
	ed.undo, ed.redo, ed.queue = nil, nil, nil
	ed.compose, ed.compositionActive = "", false
	ed.composeCaret, ed.composeSelected = 0, TextInputRange{}
	ed.buf.setSnapshot(s)
	ed.published = s
	ed.caret = min(ed.caret, ed.buf.n)
	ed.anchor = min(ed.anchor, ed.buf.n)
	ed.visualSelection, ed.selected = false, nil
	ed.coalesce = false
	if ed.area != nil {
		ed.area.reveal = true
	}
}

func (ed *editor) syncBuffer() {
	if ed.document == nil {
		return
	}
	snapshot := ed.document.Snapshot()
	if snapshot != ed.published {
		ed.loadBuffer(snapshot)
	}
}

func bufferInputBase(c *context, value *TextBuffer, multiline bool) *node {
	if value == nil {
		panic("ui: a buffer input requires a non-nil TextBuffer")
	}
	e := c.newElement(kindInput)
	e.flags |= flagEditable | flagFocusable | flagHover
	e.widget = "TextInputBuffer"
	if multiline {
		e.widget = "TextAreaBuffer"
		e.flags |= flagScrollY
	}
	st := e.st
	if st.editor == nil || st.editor.document != value {
		st.editor = newEditor(c.rt.now)
		st.editor.document = value
		st.editor.loadBuffer(value.Snapshot())
		st.editor.caret, st.editor.anchor = st.editor.buf.n, st.editor.buf.n
	}
	ed := st.editor
	ed.syncBuffer()
	ed.multiline = multiline
	if multiline && ed.area == nil {
		ed.area = &area{reveal: true}
	}
	if ed.client == nil {
		ed.client = &widgetTextInput{ed: ed, rt: c.rt, id: e.id}
	}
	e.textClient = ed.client
	ed.rebuild()
	ed.ranges = ed.ranges[:0]
	e.onValueInput(bufferInput)

	return e
}
