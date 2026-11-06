package svg

import (
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
)

// Limits on what a document may hold.
const (
	maxElems = 1 << 18
	maxDepth = 256
)

// elem is an element of the document.
type elem struct {
	name   string
	attrs  map[string]string
	kids   []*elem
	parent *elem
	text   string // of a style element
	// props are the declarations of its properties, in the order they
	// apply: attributes, style sheets, then the style attribute.
	props []decl
}

type decl struct{ name, value string }

// parseXML parses the document's elements.
func parseXML(data []byte) (*elem, error) {
	d := xml.NewDecoder(bytes.NewReader(data))
	d.Strict = false
	d.CharsetReader = charsetReader
	var root *elem
	var stack []*elem
	count, skip := 0, 0
	for {
		tok, err := d.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("svg: %w", err)
		}
		switch t := tok.(type) {
		case xml.StartElement:
			if skip > 0 || len(stack) >= maxDepth {
				skip++
				continue
			}
			if count++; count > maxElems {
				return nil, errors.New("svg: too many elements")
			}
			e := &elem{name: t.Name.Local, attrs: make(map[string]string, len(t.Attr))}
			for _, a := range t.Attr {
				// Namespaced attributes belong to editors, but for
				// xlink:href, which href takes precedence over.
				if a.Name.Space != "" && a.Name.Local != "href" {
					continue
				}
				if _, ok := e.attrs[a.Name.Local]; ok && a.Name.Space != "" {
					continue
				}
				e.attrs[a.Name.Local] = a.Value
			}
			if n := len(stack); n > 0 {
				e.parent = stack[n-1]
				e.parent.kids = append(e.parent.kids, e)
			} else {
				root = e
			}
			stack = append(stack, e)
		case xml.EndElement:
			if skip > 0 {
				skip--
				continue
			}
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
			if len(stack) == 0 {
				return finishDOM(root)
			}
		case xml.CharData:
			if n := len(stack); skip == 0 && n > 0 && stack[n-1].name == "style" {
				stack[n-1].text += string(t)
			}
		}
	}
	return finishDOM(root)
}

func finishDOM(root *elem) (*elem, error) {
	if root == nil || root.name != "svg" {
		return nil, errors.New("svg: no svg element")
	}
	cascade(root)
	return root, nil
}

// charsetReader reads the single-byte encodings old editors declare; other
// encodings are read as UTF-8.
func charsetReader(label string, r io.Reader) (io.Reader, error) {
	switch strings.ToLower(label) {
	case "iso-8859-1", "latin1", "latin-1", "windows-1252", "cp1252":
		data, err := io.ReadAll(r)
		if err != nil {
			return nil, err
		}
		var b strings.Builder
		for _, c := range data {
			b.WriteRune(rune(c))
		}
		return strings.NewReader(b.String()), nil
	}
	return r, nil
}

func walk(e *elem, fn func(*elem)) {
	fn(e)
	for _, k := range e.kids {
		walk(k, fn)
	}
}

// maxRules bounds the rules of a document's style sheets.
const maxRules = 4096

// cascade gives every element the declarations of its properties.
func cascade(root *elem) {
	var rules []rule
	walk(root, func(e *elem) {
		if t := e.attrs["type"]; e.name == "style" && (t == "" || t == "text/css") {
			rules = append(rules, parseCSS(e.text)...)
		}
	})
	rules = rules[:min(len(rules), maxRules)]
	// Rules are filed under their id, first class or type, so that an
	// element is matched against the rules it may match only.
	byID, byClass, byTag := map[string][]int{}, map[string][]int{}, map[string][]int{}
	var anyElem []int
	for i, r := range rules {
		switch s := &r.sel; {
		case s.id != "":
			byID[s.id] = append(byID[s.id], i)
		case len(s.classes) > 0:
			byClass[s.classes[0]] = append(byClass[s.classes[0]], i)
		case s.tag != "":
			byTag[s.tag] = append(byTag[s.tag], i)
		default:
			anyElem = append(anyElem, i)
		}
	}
	var matched []int
	walk(root, func(e *elem) {
		for name, v := range e.attrs {
			if properties[name] {
				e.props = append(e.props, decl{name, strings.TrimSpace(v)})
			}
		}
		matched = matched[:0]
		match := func(candidates []int) {
			for _, i := range candidates {
				if rules[i].sel.match(e) {
					matched = append(matched, i)
				}
			}
		}
		match(anyElem)
		match(byTag[e.name])
		if id := e.attrs["id"]; id != "" {
			match(byID[id])
		}
		for _, c := range strings.Fields(e.attrs["class"]) {
			match(byClass[c])
		}
		// By specificity, then in order.
		slices.SortFunc(matched, func(a, b int) int {
			if d := rules[a].sel.spec - rules[b].sel.spec; d != 0 {
				return d
			}
			return a - b
		})
		for _, i := range slices.Compact(matched) {
			e.props = append(e.props, rules[i].decls...)
		}
		if s, ok := e.attrs["style"]; ok {
			e.props = append(e.props, parseDecls(s)...)
		}
	})
}

// rule is a rule of a style sheet with a single selector.
type rule struct {
	sel   selector
	decls []decl
}

// selector is a compound selector: a type, an id and classes, any of
// which may be missing. Style sheets with combinators, pseudo-classes or
// attribute selectors are not matched.
type selector struct {
	tag     string
	id      string
	classes []string
	spec    int // specificity
}

func (s *selector) match(e *elem) bool {
	if s.tag != "" && s.tag != e.name || s.id != "" && e.attrs["id"] != s.id {
		return false
	}
	if len(s.classes) > 0 {
		have := strings.Fields(e.attrs["class"])
		for _, c := range s.classes {
			if !slices.Contains(have, c) {
				return false
			}
		}
	}
	return true
}

// parseCSS parses the rules of a style sheet.
func parseCSS(s string) []rule {
	var rules []rule
	for {
		s = strings.TrimSpace(s)
		switch {
		case s == "":
			return rules
		case strings.HasPrefix(s, "/*"):
			end := strings.Index(s[2:], "*/")
			if end < 0 {
				return rules
			}
			s = s[end+4:]
			continue
		case strings.HasPrefix(s, "<!--"):
			s = s[4:]
			continue
		case strings.HasPrefix(s, "-->"):
			s = s[3:]
			continue
		case s[0] == '@':
			// An at-rule ends at a semicolon or with its block.
			i := strings.IndexAny(s, ";{")
			if i < 0 {
				return rules
			}
			if s[i] == ';' {
				s = s[i+1:]
			} else {
				s = s[blockEnd(s, i):]
			}
			continue
		}
		open := strings.IndexByte(s, '{')
		if open < 0 {
			return rules
		}
		end := blockEnd(s, open)
		body := s[open+1 : end]
		body = strings.TrimSuffix(body, "}")
		decls := parseDecls(stripComments(body))
		for _, part := range strings.Split(stripComments(s[:open]), ",") {
			if sel, ok := parseSelector(strings.TrimSpace(part)); ok {
				rules = append(rules, rule{sel, decls})
			}
		}
		s = s[end:]
	}
}

// blockEnd returns the index after the block opening at s[open].
func blockEnd(s string, open int) int {
	depth := 0
	for i := open; i < len(s); i++ {
		switch s[i] {
		case '{':
			depth++
		case '}':
			if depth--; depth == 0 {
				return i + 1
			}
		}
	}
	return len(s)
}

func stripComments(s string) string {
	for {
		i := strings.Index(s, "/*")
		if i < 0 {
			return s
		}
		j := strings.Index(s[i+2:], "*/")
		if j < 0 {
			return s[:i]
		}
		s = s[:i] + " " + s[i+2+j+2:]
	}
}

func isNameChar(c byte) bool {
	return c == '-' || c == '_' || c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= 0x80
}

func nameEnd(s string, i int) int {
	for i < len(s) && isNameChar(s[i]) {
		i++
	}
	return i
}

func parseSelector(s string) (selector, bool) {
	var sel selector
	if s == "" {
		return sel, false
	}
	i := 0
	if s[0] == '*' {
		i = 1
	} else if isNameChar(s[0]) {
		i = nameEnd(s, 0)
		sel.tag = s[:i]
		sel.spec++
	}
	for i < len(s) {
		c := s[i]
		j := nameEnd(s, i+1)
		if j == i+1 || c != '.' && c != '#' {
			return selector{}, false
		}
		if c == '.' {
			sel.classes = append(sel.classes, s[i+1:j])
			sel.spec += 100
		} else {
			sel.id = s[i+1 : j]
			sel.spec += 10000
		}
		i = j
	}
	return sel, true
}

// parseDecls parses declarations, such as those of a style attribute.
func parseDecls(s string) []decl {
	var out []decl
	for _, part := range strings.Split(s, ";") {
		name, value, ok := strings.Cut(part, ":")
		if !ok {
			continue
		}
		name = strings.ToLower(strings.TrimSpace(name))
		value = strings.TrimSpace(value)
		if i := strings.Index(strings.ToLower(value), "!important"); i >= 0 {
			value = strings.TrimSpace(value[:i])
		}
		if properties[name] && value != "" {
			out = append(out, decl{name, value})
		}
	}
	return out
}
