package richtext

import (
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/egoist/mygo/ui"
)

// ParseRTF returns the attributed text of a document in Rich Text Format,
// as macOS apps (TextEdit, Pages) and WordPad copy it: paragraphs, their
// alignment and lists, bold, italic, underline, strikethrough and links.
// Fonts, sizes and colors are left out, as ParseHTML leaves them.
func ParseRTF(data []byte) Text {
	r := rtfReader{s: strings.TrimRight(string(data), "\x00"), uc: 1}
	r.state = rtfState{uc: 1}
	r.read()
	r.endParagraph(false)
	for len(r.paras) > 1 && len(r.paras[len(r.paras)-1].Runs) == 0 {
		r.paras = r.paras[:len(r.paras)-1]
	}
	if len(r.paras) == 0 {
		return Text{}
	}
	return TextOf(r.paras...)
}

// rtfState is what a group of RTF sets: the formats of characters, the
// style of the paragraph, and what the group holds.
type rtfState struct {
	a     Attributes
	para  ParagraphStyle
	skip  bool   // a destination whose text is not the document's
	field string // "inst" in a field's instruction, "result" in its result
	link  string // the URL of the field the group is in
	list  bool   // a paragraph's list text, which says its marker
	hide  bool   // hidden text, \v
	uc    int    // how many characters follow a \u as its fallback
}

type rtfReader struct {
	s     string
	i     int
	uc    int
	state rtfState
	stack []rtfState
	paras []Paragraph
	cur   Paragraph
	text  strings.Builder // the run being read
	runA  Attributes
	// inst is a field's instruction; marker is a paragraph's list text.
	inst   string
	marker strings.Builder
	listed bool
	level  int
}

// flush ends the run being read.
func (r *rtfReader) flush() {
	if r.text.Len() > 0 {
		r.cur.Runs = append(r.cur.Runs, Run{Text: r.text.String(), Attributes: r.runA})
		r.text.Reset()
	}
}

// emit adds text in the current attributes.
func (r *rtfReader) emit(s string) {
	st := &r.state
	switch {
	case st.list:
		r.marker.WriteString(s)
		return
	case st.field == "inst":
		r.inst += s
		return
	case st.skip, st.hide:
		return
	}
	a := st.a
	if st.field == "result" && st.link != "" {
		a.Link = st.link
	}
	if a != r.runA {
		r.flush()
		r.runA = a
	}
	r.text.WriteString(s)
}

// endParagraph ends the paragraph being read, in the style of the group.
func (r *rtfReader) endParagraph(always bool) {
	r.flush()
	if !always && len(r.cur.Runs) == 0 && !r.listed {
		return
	}
	p := r.cur
	p.Style.Align = r.state.para.Align
	if r.listed {
		m := strings.TrimSpace(r.marker.String())
		p.Style.List = Bullets
		if m != "" && (m[0] >= '0' && m[0] <= '9' || len(m) > 1 && strings.HasSuffix(m, ".")) {
			p.Style.List = Numbers
		}
		p.Style.Indent = r.level
	}
	r.paras = append(r.paras, p)
	r.cur = Paragraph{}
	r.marker.Reset()
	r.listed, r.level = false, 0
}

// read reads the document's groups, control words and text.
func (r *rtfReader) read() {
	s := r.s
	pendingSkip := false // after \* : the group is a destination to skip
	for r.i < len(s) {
		c := s[r.i]
		switch c {
		case '{':
			r.stack = append(r.stack, r.state)
			r.i++
		case '}':
			wasInst := r.state.field == "inst"
			if n := len(r.stack); n > 0 {
				r.state = r.stack[n-1]
				r.stack = r.stack[:n-1]
			}
			if wasInst && r.state.field != "inst" {
				// The field's group: its result links to the URL of its
				// instruction.
				r.state.link = safeLink(hyperlink(r.inst))
				r.inst = ""
			}
			r.i++
		case '\\':
			r.i++
			if r.i >= len(s) {
				return
			}
			c = s[r.i]
			switch {
			case c == '\'' && r.i+2 < len(s):
				// A character of the code page, as Windows-1252.
				if v, err := strconv.ParseUint(s[r.i+1:r.i+3], 16, 8); err == nil {
					r.emit(string(cp1252(byte(v))))
				}
				r.i += 3
			case c == '*':
				pendingSkip = true
				r.i++
			case c == '\\' || c == '{' || c == '}':
				r.emit(string(c))
				r.i++
			case c == '~':
				r.emit(" ")
				r.i++
			case c == '_':
				r.emit("\u2011") // a hyphen lines don't break at
				r.i++
			case c == '\n' || c == '\r':
				// \ and a newline is a \par.
				r.endParagraph(true)
				r.i++
			case isLetter(c):
				word, num, hasNum := r.word()
				if pendingSkip {
					pendingSkip = false
					if word != "fldinst" {
						r.state.skip = true
					}
				}
				r.control(word, num, hasNum)
			default:
				r.i++
			}
		case '\r', '\n':
			r.i++
		default:
			// Text up to the next special character.
			j := r.i
			for j < len(s) && s[j] != '\\' && s[j] != '{' && s[j] != '}' && s[j] != '\r' && s[j] != '\n' {
				j++
			}
			r.emit(s[r.i:j])
			r.i = j
		}
	}
}

func isLetter(c byte) bool { return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' }

// word reads a control word and its number, and the space ending them.
func (r *rtfReader) word() (string, int, bool) {
	s := r.s
	start := r.i
	for r.i < len(s) && isLetter(s[r.i]) {
		r.i++
	}
	word := s[start:r.i]
	numStart := r.i
	if r.i < len(s) && s[r.i] == '-' {
		r.i++
	}
	for r.i < len(s) && s[r.i] >= '0' && s[r.i] <= '9' {
		r.i++
	}
	num, err := strconv.Atoi(s[numStart:r.i])
	if r.i < len(s) && s[r.i] == ' ' {
		r.i++
	}
	return word, num, err == nil
}

// control applies a control word.
func (r *rtfReader) control(word string, num int, hasNum bool) {
	st := &r.state
	on := !hasNum || num != 0
	switch word {
	case "fonttbl", "colortbl", "stylesheet", "info", "pict", "header", "footer", "footnote",
		"listtable", "listoverridetable", "expandedcolortbl", "themedata", "colorschememapping", "latentstyles",
		"datastore", "xmlnstbl", "rsidtbl", "generator", "object", "shppict", "nonshppict":
		st.skip = true
	case "field":
		st.field, st.link = "", ""
	case "fldinst":
		st.field = "inst"
		st.skip = false
	case "fldrslt":
		st.field = "result"
	case "listtext", "pntext":
		// The marker of a list's item, as Word and WordPad write it.
		st.list = true
		r.listed = true
	case "ls":
		r.listed = true
	case "ilvl":
		r.level = max(0, num)
	case "par", "line", "sect", "page":
		if !st.skip && !st.list && st.field != "inst" {
			r.endParagraph(true)
		}
	case "pard":
		st.para = ParagraphStyle{}
	case "ql", "qj":
		st.para.Align = ui.Start
	case "qc":
		st.para.Align = ui.Center
	case "qr":
		st.para.Align = ui.End
	case "plain":
		st.a, st.hide = Attributes{}, false
	case "b":
		if on {
			st.a.Weight = 700
		} else {
			st.a.Weight = 0
		}
	case "i":
		st.a.Italic = on
	case "ul", "uld", "uldb", "ulw", "ulth":
		st.a.Underline = on
	case "ulnone":
		st.a.Underline = false
	case "strike", "striked":
		st.a.Strikethrough = on
	case "tab", "cell":
		r.emit("\t")
	case "row":
		if !st.skip && !st.list && st.field != "inst" {
			r.endParagraph(true)
		}
	case "v":
		st.hide = on
	case "emdash":
		r.emit("—")
	case "endash":
		r.emit("–")
	case "bullet":
		r.emit("•")
	case "lquote":
		r.emit("‘")
	case "rquote":
		r.emit("’")
	case "ldblquote":
		r.emit("“")
	case "rdblquote":
		r.emit("”")
	case "uc":
		st.uc = num
		r.uc = num
	case "u":
		if num < 0 {
			num += 65536
		}
		rn := rune(num)
		// A surrogate pair comes as two \u.
		if rn >= 0xD800 && rn <= 0xDBFF {
			r.skipFallback()
			if strings.HasPrefix(r.s[r.i:], "\\u") {
				r.i++
				_, low, _ := r.word()
				if low < 0 {
					low += 65536
				}
				rn = 0x10000 + (rn-0xD800)<<10 + (rune(low) - 0xDC00)
			}
		}
		if utf8.ValidRune(rn) {
			r.emit(string(rn))
		}
		r.skipFallback()
	}
}

// skipFallback skips the characters after a \u that stand for it where
// Unicode is not read.
func (r *rtfReader) skipFallback() {
	for n := r.state.uc; n > 0 && r.i < len(r.s); n-- {
		switch {
		case r.s[r.i] == '\\' && r.i+3 < len(r.s) && r.s[r.i+1] == '\'':
			r.i += 4
		case r.s[r.i] == '\\' || r.s[r.i] == '{' || r.s[r.i] == '}':
			return
		default:
			r.i++
		}
	}
}

// hyperlink returns the URL of a field's instruction, HYPERLINK "url".
func hyperlink(inst string) string {
	inst = strings.TrimSpace(inst)
	rest, ok := strings.CutPrefix(inst, "HYPERLINK")
	if !ok {
		return ""
	}
	rest = strings.TrimSpace(rest)
	// \l "name" links to a bookmark of the document.
	if after, ok := strings.CutPrefix(rest, "\\l"); ok {
		if u := hyperlink("HYPERLINK " + strings.TrimSpace(after)); u != "" {
			return "#" + u
		}
		return ""
	}
	if strings.HasPrefix(rest, "\"") {
		if end := strings.Index(rest[1:], "\""); end >= 0 {
			return rest[1 : end+1]
		}
	}
	if f := strings.Fields(rest); len(f) > 0 {
		return f[0]
	}
	return ""
}

// cp1252 returns the rune of a byte of Windows-1252.
func cp1252(b byte) rune {
	if b >= 0x80 && b < 0xA0 {
		return []rune("€\u0081‚ƒ„…†‡ˆ‰Š‹Œ\u008dŽ\u008f\u0090‘’“”•–—˜™š›œ\u009džŸ")[b-0x80]
	}
	return rune(b)
}
