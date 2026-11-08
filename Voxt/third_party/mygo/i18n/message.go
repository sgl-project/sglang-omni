package i18n

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// A Message is a message of ICU MessageFormat, parsed: text with arguments
// that the values given to Format replace, as ARB files and Flutter's
// gen_l10n write them:
//
//	Hello, {name}!
//	{count, plural, =0 {No files} one {# file} other {# files}}
//	{count, plural, offset:1 =0 {Nobody} =1 {{host}} one {{host} and # other} other {{host} and # others}}
//	{place, selectordinal, one {#st} two {#nd} few {#rd} other {#th}}
//	{gender, select, female {She} male {He} other {They}} replied.
//	{price, number} on {day, date, short} at {day, time, short}
//
// An argument {name} writes the value of name, a number formatted for the
// locale (as {name, number} does) and a time.Time as a short date and
// time. A plural argument picks the case of the number's plural category in
// the language (one, few, other...), or of an exact value, =0; # in its
// cases writes the number less the offset. A select argument picks the case
// that is the value, as a string, or other; every plural and select
// argument has an other case. Cases nest.
//
// An apostrophe quotes, as ICU's does: '{' writes a brace, '#' a #, '{a}'
// the text {a}, and two apostrophes write one. An apostrophe before any
// other character is one: "don't" needs no quoting.
type Message struct {
	source string
	nodes  []node
}

type nodeKind uint8

const (
	nodeText          nodeKind = iota // text
	nodeArg                           // {name}
	nodePound                         // # in a plural case
	nodeNumber                        // {name, number[, style]}
	nodeDate                          // {name, date[, style]}
	nodeTime                          // {name, time[, style]}
	nodePlural                        // {name, plural, ...}
	nodeSelectOrdinal                 // {name, selectordinal, ...}
	nodeSelect                        // {name, select, ...}
)

// node is a part of a message: text, or an argument named name.
type node struct {
	kind   nodeKind
	text   string // the text, or the name of the argument
	style  string // of number, date and time
	offset int64  // of plural
	cases  []msgCase
}

// msgCase is a case of a plural or a select argument.
type msgCase struct {
	key   string // a keyword, or "=" and a number
	exact Operands
	nodes []node
}

// ParseMessage parses a message of ICU MessageFormat.
func ParseMessage(s string) (*Message, error) {
	linkTimes()
	return parseMessage(s)
}

// parseMessage is ParseMessage for the catalogs, which link the
// formatting of dates as they are made.
func parseMessage(s string) (*Message, error) {
	p := &msgParser{s: s}
	nodes, err := p.message(0, false)
	if err != nil {
		return nil, err
	}
	if p.i < len(s) {
		return nil, p.errorf("unmatched }")
	}
	return &Message{source: s, nodes: nodes}, nil
}

// MustParseMessage is ParseMessage, but panics when s is not a message.
func MustParseMessage(s string) *Message {
	m, err := ParseMessage(s)
	if err != nil {
		panic(err)
	}
	return m
}

// String returns the message as it was written.
func (m *Message) String() string { return m.source }

// Arg is an argument of a message: its name and its kind, "" for {name},
// or "number", "date", "time", "plural", "selectordinal" or "select".
type Arg struct {
	Name string
	Kind string
}

// Args returns the arguments of the message, nested ones too, in the
// order they come; an argument used twice comes once, with its first kind
// other than "".
func (m *Message) Args() []Arg {
	var out []Arg
	var walk func([]node)
	walk = func(nodes []node) {
		for _, n := range nodes {
			if n.kind == nodeText || n.kind == nodePound {
				continue
			}
			kind := argKinds[n.kind]
			found := false
			for i := range out {
				if out[i].Name == n.text {
					found = true
					if out[i].Kind == "" {
						out[i].Kind = kind
					}
				}
			}
			if !found {
				out = append(out, Arg{n.text, kind})
			}
			for _, c := range n.cases {
				walk(c.nodes)
			}
		}
	}
	walk(m.nodes)
	return out
}

var argKinds = [...]string{
	nodeArg: "", nodeNumber: "number", nodeDate: "date", nodeTime: "time",
	nodePlural: "plural", nodeSelectOrdinal: "selectordinal", nodeSelect: "select",
	nodeText: "", nodePound: "",
}

// Cases returns the keys of the cases of the plural, selectordinal or
// select arguments named name, as "=0", "one" and "other", in the order
// they come, each once.
func (m *Message) Cases(name string) []string {
	var out []string
	var walk func([]node)
	walk = func(nodes []node) {
		for _, n := range nodes {
			for _, c := range n.cases {
				if n.text == name && !contains(out, c.key) {
					out = append(out, c.key)
				}
				walk(c.nodes)
			}
		}
	}
	walk(m.nodes)
	return out
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

type msgParser struct {
	s string
	i int
}

// SyntaxError is the error of a message that does not parse.
type SyntaxError struct {
	Message string // the message
	Offset  int    // the byte offset in it of the error
	Err     string
}

func (e *SyntaxError) Error() string {
	return fmt.Sprintf("i18n: %s at offset %d of %q", e.Err, e.Offset, e.Message)
}

func (p *msgParser) errorf(format string, args ...any) error {
	return &SyntaxError{Message: p.s, Offset: p.i, Err: fmt.Sprintf(format, args...)}
}

// message parses text and arguments until a } that it leaves, or the end.
// In a case of a plural argument, # is the number.
func (p *msgParser) message(depth int, inPlural bool) ([]node, error) {
	if depth > 32 {
		return nil, p.errorf("arguments nest too deep")
	}
	var nodes []node
	var text strings.Builder
	flush := func() {
		if text.Len() > 0 {
			nodes = append(nodes, node{kind: nodeText, text: text.String()})
			text.Reset()
		}
	}
	for p.i < len(p.s) {
		c := p.s[p.i]
		switch {
		case c == '\'':
			p.quoted(&text, inPlural)
		case c == '{':
			flush()
			n, err := p.argument(depth, inPlural)
			if err != nil {
				return nil, err
			}
			nodes = append(nodes, n)
		case c == '}':
			flush()
			return nodes, nil
		case c == '#' && inPlural:
			flush()
			nodes = append(nodes, node{kind: nodePound})
			p.i++
		default:
			text.WriteByte(c)
			p.i++
		}
	}
	flush()
	return nodes, nil
}

// quoted reads an apostrophe at p.i: two write one, one before a brace (or
// a # in a plural case) quotes up to the next lone apostrophe or the end,
// and any other is itself.
func (p *msgParser) quoted(text *strings.Builder, inPlural bool) {
	p.i++
	if p.i < len(p.s) && p.s[p.i] == '\'' {
		text.WriteByte('\'')
		p.i++
		return
	}
	if p.i >= len(p.s) || !(p.s[p.i] == '{' || p.s[p.i] == '}' || p.s[p.i] == '#' && inPlural) {
		text.WriteByte('\'')
		return
	}
	for p.i < len(p.s) {
		if p.s[p.i] == '\'' {
			if p.i+1 < len(p.s) && p.s[p.i+1] == '\'' {
				text.WriteByte('\'')
				p.i += 2
				continue
			}
			p.i++
			return
		}
		text.WriteByte(p.s[p.i])
		p.i++
	}
}

func (p *msgParser) space() {
	for p.i < len(p.s) {
		r, size := utf8.DecodeRuneInString(p.s[p.i:])
		if !isPatternSpace(r) {
			return
		}
		p.i += size
	}
}

// isPatternSpace is Unicode's Pattern_White_Space.
func isPatternSpace(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\v', '\f', '\r', 0x85, 0x200E, 0x200F, 0x2028, 0x2029:
		return true
	}
	return false
}

// isPatternSyntax reports whether r ends a name: Unicode's
// Pattern_Syntax, as ICU's argument names are Pattern_Syntax-free.
func isPatternSyntax(r rune) bool {
	if r < 0x80 {
		return r >= '!' && r <= '/' || r >= ':' && r <= '@' || r >= '[' && r <= '^' || r == '`' || r >= '{' && r <= '~'
	}
	return r >= 0xA1 && r <= 0xA7 || r == 0xA9 || r == 0xAB || r == 0xAC || r == 0xAE || r == 0xB0 || r == 0xB1 ||
		r == 0xB6 || r == 0xBB || r == 0xBF || r == 0xD7 || r == 0xF7 ||
		r >= 0x2010 && r <= 0x2027 || r >= 0x2030 && r <= 0x203E || r >= 0x2041 && r <= 0x2053 ||
		r >= 0x2055 && r <= 0x205E || r >= 0x2190 && r <= 0x245F || r >= 0x2500 && r <= 0x2775 ||
		r >= 0x2794 && r <= 0x2BFF || r >= 0x2E00 && r <= 0x2E7F || r >= 0x3001 && r <= 0x3003 ||
		r >= 0x3008 && r <= 0x3020 || r == 0x3030 || r == 0xFD3E || r == 0xFD3F || r == 0xFE45 || r == 0xFE46
}

// word reads a name or a keyword: characters other than white space and
// syntax.
func (p *msgParser) word() string {
	start := p.i
	for p.i < len(p.s) {
		r, size := utf8.DecodeRuneInString(p.s[p.i:])
		if isPatternSpace(r) || isPatternSyntax(r) {
			break
		}
		p.i += size
	}
	return p.s[start:p.i]
}

// argument parses an argument from its { to its }.
func (p *msgParser) argument(depth int, inPlural bool) (node, error) {
	p.i++ // {
	p.space()
	name := p.word()
	if name == "" {
		return node{}, p.errorf("an argument has no name")
	}
	if !validArgName(name) {
		return node{}, p.errorf("%q is not an argument name", name)
	}
	p.space()
	if p.i >= len(p.s) {
		return node{}, p.errorf("unclosed argument {%s", name)
	}
	if p.s[p.i] == '}' {
		p.i++
		return node{kind: nodeArg, text: name}, nil
	}
	if p.s[p.i] != ',' {
		return node{}, p.errorf("expected , or } after the argument name %q", name)
	}
	p.i++
	p.space()
	typ := p.word()
	p.space()
	n := node{text: name}
	switch typ {
	case "number", "date", "time":
		n.kind = map[string]nodeKind{"number": nodeNumber, "date": nodeDate, "time": nodeTime}[typ]
		if p.i < len(p.s) && p.s[p.i] == ',' {
			p.i++
			style, err := p.style()
			if err != nil {
				return node{}, err
			}
			if style == "" {
				return node{}, p.errorf("the %s argument %q has an empty style", typ, name)
			}
			n.style = style
		}
		if p.i >= len(p.s) || p.s[p.i] != '}' {
			return node{}, p.errorf("unclosed argument {%s, %s", name, typ)
		}
		p.i++
		return n, nil
	case "plural", "selectordinal", "select":
	case "":
		return node{}, p.errorf("the argument %q has no type", name)
	default:
		return node{}, p.errorf("unknown argument type %q (want number, date, time, plural, selectordinal or select)", typ)
	}
	if p.i >= len(p.s) || p.s[p.i] != ',' {
		return node{}, p.errorf("expected , after %s", typ)
	}
	p.i++
	switch typ {
	case "plural":
		n.kind = nodePlural
	case "selectordinal":
		n.kind = nodeSelectOrdinal
	default:
		n.kind = nodeSelect
	}
	plural := n.kind != nodeSelect
	p.space()
	if plural && strings.HasPrefix(p.s[p.i:], "offset:") {
		p.i += len("offset:")
		p.space()
		start := p.i
		for p.i < len(p.s) && p.s[p.i] >= '0' && p.s[p.i] <= '9' {
			p.i++
		}
		if start == p.i || p.i-start > 15 {
			return node{}, p.errorf("offset: needs a number")
		}
		for _, d := range p.s[start:p.i] {
			n.offset = n.offset*10 + int64(d-'0')
		}
	}
	for {
		p.space()
		if p.i >= len(p.s) {
			return node{}, p.errorf("unclosed %s argument %q", typ, name)
		}
		if p.s[p.i] == '}' {
			p.i++
			break
		}
		var c msgCase
		if plural && p.s[p.i] == '=' {
			p.i++
			start := p.i
			for p.i < len(p.s) && (p.s[p.i] >= '0' && p.s[p.i] <= '9' || p.s[p.i] == '.') {
				p.i++
			}
			value := p.s[start:p.i]
			o, err := parseOperands(value)
			if err != nil || strings.HasPrefix(value, ".") || strings.HasSuffix(value, ".") {
				return node{}, p.errorf("=%s is not a number", value)
			}
			c.key, c.exact = "="+value, o
		} else {
			c.key = p.word()
			if c.key == "" {
				return node{}, p.errorf("expected a case of the %s argument %q", typ, name)
			}
			if _, ok := ParsePlural(c.key); plural && !ok {
				return node{}, p.errorf("%q is not a plural category (zero, one, two, few, many, other)", c.key)
			}
		}
		for _, prev := range n.cases {
			if prev.key == c.key {
				return node{}, p.errorf("the %s argument %q has the case %s twice", typ, name, c.key)
			}
		}
		p.space()
		if p.i >= len(p.s) || p.s[p.i] != '{' {
			return node{}, p.errorf("expected { after the case %s", c.key)
		}
		p.i++
		nodes, err := p.message(depth+1, plural)
		if err != nil {
			return node{}, err
		}
		if p.i >= len(p.s) {
			return node{}, p.errorf("unclosed case %s", c.key)
		}
		p.i++ // }
		c.nodes = nodes
		n.cases = append(n.cases, c)
	}
	if !hasOther(n.cases) {
		return node{}, p.errorf("the %s argument %q has no other case", typ, name)
	}
	return n, nil
}

func hasOther(cases []msgCase) bool {
	for _, c := range cases {
		if c.key == "other" {
			return true
		}
	}
	return false
}

// style reads the style of a number, date or time argument up to its }:
// a keyword, as "short", or a skeleton, as "::currency/EUR", in which
// quoted text and nested braces may be.
func (p *msgParser) style() (string, error) {
	start, depth := p.i, 0
	for p.i < len(p.s) {
		switch p.s[p.i] {
		case '\'':
			end := strings.IndexByte(p.s[p.i+1:], '\'')
			if end < 0 {
				return "", p.errorf("unclosed quote in a style")
			}
			p.i += end + 2
			continue
		case '{':
			depth++
		case '}':
			if depth == 0 {
				return strings.TrimFunc(p.s[start:p.i], isPatternSpace), nil
			}
			depth--
		}
		p.i++
	}
	return "", p.errorf("unclosed style")
}

// validArgName reports whether name names an argument: a number without
// leading zeros, or a name that starts with a letter or _.
func validArgName(name string) bool {
	r, _ := utf8.DecodeRuneInString(name)
	if r >= '0' && r <= '9' {
		if len(name) > 1 && name[0] == '0' {
			return false
		}
		for i := 0; i < len(name); i++ {
			if name[i] < '0' || name[i] > '9' {
				return false
			}
		}
		return true
	}
	return r == '_' || unicode.IsLetter(r)
}
