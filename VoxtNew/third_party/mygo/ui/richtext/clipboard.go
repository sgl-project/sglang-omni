package richtext

import (
	"crypto/rand"
	"encoding/json"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/egoist/mygo/transfer"
)

// Format is the clipboard's format of attributed text copied from an
// Editor: its paragraphs, styles and runs as JSON, which other MyGo apps
// paste whole, as Lexical and Slate copy a format of their own first.
const Format transfer.Format = "application/x-mygo-rich-text"

// RTF is the format of Rich Text Format, which macOS apps (TextEdit,
// Notes, Pages) copy.
const RTF transfer.Format = "text/rtf"

// clipText is the JSON of the format, for paragraphs of a Text.
type clipText struct {
	// Copy names the copy within the process that made it, whose text,
	// attachments included, pastes as it was.
	Copy       string          `json:"copy,omitempty"`
	Paragraphs []clipParagraph `json:"paragraphs"`
}

type clipParagraph struct {
	Style ParagraphStyle `json:"style"`
	Runs  []clipRun      `json:"runs,omitempty"`
}

type clipRun struct {
	Text       string          `json:"text"`
	Attributes clipAttributes  `json:"attributes"`
	Attachment *clipAttachment `json:"attachment,omitempty"`
}

// clipAttributes are the Attributes of a run but its attachment, which
// clipAttachment describes.
type clipAttributes struct {
	Font          string  `json:"font,omitempty"`
	Size          float32 `json:"size,omitempty"`
	Weight        int     `json:"weight,omitempty"`
	Italic        bool    `json:"italic,omitempty"`
	Underline     bool    `json:"underline,omitempty"`
	Strikethrough bool    `json:"strikethrough,omitempty"`
	Color         string  `json:"color,omitempty"`
	Background    string  `json:"background,omitempty"`
	Code          bool    `json:"code,omitempty"`
	Link          string  `json:"link,omitempty"`
	Tag           string  `json:"tag,omitempty"`
}

type clipAttachment struct {
	Name   string  `json:"name,omitempty"`
	URL    string  `json:"url,omitempty"`
	Width  float32 `json:"width,omitempty"`
	Height float32 `json:"height,omitempty"`
}

// copies keeps the last text copied in this process, by the name its
// format carries, so that pasting it keeps its attachments. Names start
// with one random to the process, so that no other app's copy has one.
var (
	copies   sync.Map // string → Text
	copyName atomic.Uint64
	lastCopy atomic.Value // string
	process  = rand.Text()
)

// encodeClip returns t in the clipboard's format.
func encodeClip(t Text) []byte {
	name := process + "-" + strconv.FormatUint(copyName.Add(1), 36)
	if old, ok := lastCopy.Load().(string); ok {
		copies.Delete(old)
	}
	copies.Store(name, t)
	lastCopy.Store(name)
	c := clipOf(t)
	c.Copy = name
	b, _ := json.Marshal(c)
	return b
}

// MarshalJSON writes t as JSON, as the clipboard's Format holds it, to
// save it: paragraphs of a style each, of runs of text and attributes.
// Attachments keep their names, URLs and sizes; images and drawings are
// the app's to save.
func (t Text) MarshalJSON() ([]byte, error) { return json.Marshal(clipOf(t)) }

// UnmarshalJSON reads what MarshalJSON wrote.
func (t *Text) UnmarshalJSON(b []byte) error {
	var c clipText
	if err := json.Unmarshal(b, &c); err != nil {
		return err
	}
	*t = c.text()
	return nil
}

// clipOf returns the JSON form of t.
func clipOf(t Text) clipText {
	var c clipText
	for _, p := range t.Paragraphs() {
		cp := clipParagraph{Style: p.Style}
		for _, r := range p.Runs {
			a := r.Attributes
			cr := clipRun{Text: r.Text, Attributes: clipAttributes{
				Font: a.Font, Size: a.Size, Weight: a.Weight, Italic: a.Italic, Underline: a.Underline,
				Strikethrough: a.Strikethrough, Code: a.Code, Link: a.Link, Tag: a.Tag,
			}}
			if a.Color.A > 0 {
				cr.Attributes.Color = cssColor(a.Color)
			}
			if a.Background.A > 0 {
				cr.Attributes.Background = cssColor(a.Background)
			}
			if at := a.Attachment; at != nil {
				w, h := at.size()
				cr.Attachment = &clipAttachment{Name: at.Name, URL: at.URL, Width: w, Height: h}
			}
			cp.Runs = append(cp.Runs, cr)
		}
		c.Paragraphs = append(c.Paragraphs, cp)
	}
	return c
}

// decodeClip returns the text of the clipboard's format.
func decodeClip(b []byte) (Text, bool) {
	var c clipText
	if json.Unmarshal(b, &c) != nil {
		return Text{}, false
	}
	if c.Copy != "" {
		if t, ok := copies.Load(c.Copy); ok {
			return t.(Text), true
		}
	}
	if len(c.Paragraphs) == 0 {
		return Text{}, false
	}
	return c.text(), true
}

// text returns the text of the JSON form.
func (c clipText) text() Text {
	var paras []Paragraph
	for _, cp := range c.Paragraphs {
		p := Paragraph{Style: cp.Style}
		for _, cr := range cp.Runs {
			a := Attributes{
				Font: cr.Attributes.Font, Size: cr.Attributes.Size, Weight: cr.Attributes.Weight, Italic: cr.Attributes.Italic,
				Underline: cr.Attributes.Underline, Strikethrough: cr.Attributes.Strikethrough, Code: cr.Attributes.Code,
				Link: safeLink(cr.Attributes.Link), Tag: cr.Attributes.Tag,
			}
			a.Color, _ = parseCSSColor(cr.Attributes.Color)
			a.Background, _ = parseCSSColor(cr.Attributes.Background)
			if at := cr.Attachment; at != nil {
				a.Attachment = &Attachment{Name: at.Name, URL: at.URL, Width: side(at.Width), Height: side(at.Height)}
			}
			p.Runs = append(p.Runs, Run{Text: cr.Text, Attributes: a})
		}
		paras = append(paras, p)
	}
	if len(paras) == 0 {
		return Text{}
	}
	return TextOf(paras...)
}

// PlainText returns t as plain text, its attachments left out.
func (t Text) PlainText() string {
	return strings.ReplaceAll(t.String(), string(ObjectReplacement), "")
}

// clipboardData returns t as the clipboard takes it: in the format of
// the editor, as HTML and as plain text.
func clipboardData(t Text) transfer.Data {
	return transfer.New(transfer.NewItem(
		transfer.Bytes(Format, encodeClip(t)),
		transfer.Bytes(transfer.HTML, []byte(t.HTML())),
		transfer.Bytes(transfer.Text, []byte(t.PlainText())),
	))
}

// copySelection puts the selection on the clipboard.
func (v *view) copySelection() {
	a, z := v.selection()
	if a == z {
		return
	}
	v.services.WriteClipboardData(clipboardData(v.shown.Slice(a, z)))
}

// paste replaces the selection with what the clipboard holds: the richest
// format the editor reads, or its plain text in the typing attributes when
// matching the style.
func (v *view) paste(matchStyle bool) {
	var t Text
	ok := false
	if !matchStyle {
		d := v.services.ReadClipboardData(Format, transfer.HTML, RTF, transfer.Markdown, transfer.Text)
		if b, err := d.Read(Format); err == nil {
			t, ok = decodeClip(b)
		}
		if !ok {
			if b, err := d.Read(transfer.HTML); err == nil && len(b) > 0 {
				t, ok = ParseHTML(string(b)), true
			}
		}
		if !ok {
			// TextEdit, Pages and WordPad copy RTF without HTML.
			if b, err := d.Read(RTF); err == nil && len(b) > 0 {
				t, ok = ParseRTF(b), true
			}
		}
		if !ok {
			// Markdown apps and Markdown views copy it.
			if b, err := d.Read(transfer.Markdown); err == nil && len(b) > 0 {
				t, ok = ParseMarkdown(string(b)), true
			}
		}
		if !ok {
			if b, err := d.Read(transfer.Text); err == nil && len(b) > 0 {
				v.insert(string(b))
				return
			}
		}
	}
	if !ok {
		s := v.services.ReadClipboard()
		if s != "" {
			v.insert(s)
		}
		return
	}
	v.record(false)
	d := v.doc
	d.lock()
	v.lockedSync()
	a, z := v.selection() // as the sync moved it
	v.lockedPublish()
	v.lockedEdit(func(e *Edit) { e.Replace(a, z, t) })
	caret := a + t.Len()
	v.selAnchor, v.caret = caret, caret
	d.anchor, d.caret = caret, caret
	d.mu.Unlock()
	v.mode, v.coalesce = newStep, false
	v.reveal = true
}
