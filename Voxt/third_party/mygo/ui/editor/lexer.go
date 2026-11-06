package editor

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// lexer is a tokenizer of the languages most code is in: identifiers and
// keywords, numbers, strings with escapes and interpolations, line and
// block comments, operators and punctuation, configured per language.
// Its state, which fits in 40 bits so that Markdown can hold it for its
// code blocks, packs:
//
//	bits 0-7    the mode: 0 code, 1+k in block comment k, 32+j in string j
//	bits 8-15   the depth of nested block comments
//	bits 16-39  up to three strings an interpolation is inside, 8 bits
//	            each: the string's index plus one, and the braces open in
//	            the interpolation
type lexer struct {
	lineComments  []string
	blockComments [][2]string
	nestComments  bool
	strings       []lexString
	words         map[string]TokenKind
	// foldCase looks keywords up in lower case, as SQL's.
	foldCase bool
	// identChars are characters other than letters, digits and _ in
	// identifiers, as $ in JavaScript.
	identChars string
	// capitalTypes makes capitalized identifiers types.
	capitalTypes bool
	// calls makes identifiers before ( functions.
	calls bool
	// keys makes identifiers and strings before : properties, as JSON's.
	keys bool
	// decorator starts annotations, as @ in Java and Python.
	decorator byte
	// directives makes # at the start of a line a preprocessor line, as
	// C's.
	directives bool
	// regexps lexes /.../ where an expression may start, as JavaScript.
	regexps bool
	// variables makes $name a variable, as in shells and PHP.
	variables bool
	operators string
}

// lexString is a kind of string literal.
type lexString struct {
	open, close string
	// escape starts escape sequences, 0 for raw strings.
	escape byte
	// multiline strings go on past the end of a line.
	multiline bool
	// interpOpen and interpClose enclose code inside the string, as ${
	// and } in JavaScript.
	interpOpen, interpClose string
	// prefix is an identifier that must come right before open, as f in
	// Python's f"…".
	prefix string
	// dollar marks $name variables inside the string, as shells do.
	dollar bool
	kind   TokenKind
}

const (
	lexModeComment = 1
	lexModeString  = 32
)

type lexOut struct{ toks []Token }

func (o *lexOut) add(end int, k TokenKind) {
	if n := len(o.toks); n > 0 {
		last := &o.toks[n-1]
		if last.End >= end {
			return
		}
		if last.Kind == k {
			last.End = end
			return
		}
	}
	o.toks = append(o.toks, Token{End: end, Kind: k})
}

// lexState is what a line of code being lexed knows.
type lexState struct {
	line    string
	out     lexOut
	mode    int
	depth   int
	stack   uint64
	prevDot bool // the last token was a dot
	regexOK bool // an expression may start
}

func (lx *lexer) Tokenize(line string, state uint64, tokens []Token) ([]Token, uint64) {
	st := lexState{line: line, out: lexOut{toks: tokens}, mode: int(state & 0xff), depth: int(state >> 8 & 0xff),
		stack: state >> 16 & 0xffffff, regexOK: true}
	for i := 0; i < len(line); {
		switch {
		case st.mode >= lexModeString:
			i = lx.inString(&st, i)
		case st.mode >= lexModeComment:
			i = lx.inComment(&st, i)
		default:
			i = lx.code(&st, i)
		}
	}
	if st.mode >= lexModeString && !lx.strings[st.mode-lexModeString].multiline {
		st.mode = 0
		st.stack = 0
	}
	return st.out.toks, uint64(st.mode) | uint64(st.depth&0xff)<<8 | st.stack<<16
}

func (lx *lexer) inComment(st *lexState, i int) int {
	c := lx.blockComments[st.mode-lexModeComment]
	line := st.line
	j := i
	for j < len(line) {
		if lx.nestComments && strings.HasPrefix(line[j:], c[0]) {
			st.depth++
			j += len(c[0])
			continue
		}
		if strings.HasPrefix(line[j:], c[1]) {
			j += len(c[1])
			if st.depth > 0 {
				st.depth--
				continue
			}
			st.mode = 0
			break
		}
		j++
	}
	st.out.add(j, TokenComment)
	return j
}

// inString lexes a string's contents from i until its end, the end of
// the line, or an interpolation, which it enters.
func (lx *lexer) inString(st *lexState, i int) int {
	k := st.mode - lexModeString
	s := &lx.strings[k]
	line := st.line
	n := len(line)
	for j := i; j < n; {
		c := line[j]
		if s.escape != 0 && c == s.escape {
			st.out.add(j, s.kind)
			e := min(n, j+2)
			if j+1 < n && line[j+1] >= utf8.RuneSelf {
				_, size := utf8.DecodeRuneInString(line[j+1:])
				e = j + 1 + size
			}
			st.out.add(e, TokenEscape)
			j = e
			continue
		}
		if strings.HasPrefix(line[j:], s.close) {
			st.out.add(j+len(s.close), s.kind)
			st.mode = 0
			st.regexOK = false
			return j + len(s.close)
		}
		if s.interpOpen != "" && strings.HasPrefix(line[j:], s.interpOpen) {
			st.out.add(j, s.kind)
			e := j + len(s.interpOpen)
			st.out.add(e, TokenEscape)
			if st.stack>>16 == 0 {
				st.stack = st.stack<<8 | uint64(k+1)<<3
				st.mode = 0
				st.regexOK = true
				return e
			}
			j = e
			continue
		}
		if s.dollar && c == '$' && j+1 < n && (isIdentByte(line[j+1]) || line[j+1] == '{') {
			st.out.add(j, s.kind)
			e := dollarEnd(line, j)
			st.out.add(e, TokenVariable)
			j = e
			continue
		}
		j++
	}
	st.out.add(n, s.kind)
	return n
}

// dollarEnd returns where a $name, ${name} or $1 at i ends.
func dollarEnd(line string, i int) int {
	n := len(line)
	j := i + 1
	if j < n && line[j] == '{' {
		for j < n && line[j] != '}' {
			j++
		}
		return min(n, j+1)
	}
	if j < n && strings.IndexByte("@*#?$!-0123456789", line[j]) >= 0 {
		return j + 1
	}
	for j < n && isIdentByte(line[j]) {
		j++
	}
	return j
}

// code lexes one token of code from i, and returns where it ends.
func (lx *lexer) code(st *lexState, i int) int {
	line := st.line
	n := len(line)
	c := line[i]
	out := &st.out
	if c == ' ' || c == '\t' || c == '\r' || c == '\f' {
		j := i + 1
		for j < n && (line[j] == ' ' || line[j] == '\t' || line[j] == '\r' || line[j] == '\f') {
			j++
		}
		out.add(j, TokenText)
		return j
	}
	// An interpolation's braces, and its end going back into its string.
	if st.stack != 0 {
		top := st.stack & 0xff
		s := &lx.strings[top>>3-1]
		braces := int(top & 7)
		switch {
		case c == '{' && s.interpClose == "}":
			if braces < 7 {
				st.stack++
			}
			out.add(i+1, TokenPunctuation)
			return i + 1
		case (braces == 0 || s.interpClose != "}") && strings.HasPrefix(line[i:], s.interpClose):
			st.stack >>= 8
			st.mode = lexModeString + int(top>>3) - 1
			out.add(i+len(s.interpClose), TokenEscape)
			return i + len(s.interpClose)
		case c == '}' && braces > 0:
			st.stack--
			out.add(i+1, TokenPunctuation)
			return i + 1
		}
	}
	// Block comments first: Lua's --[[ starts with its line comment's --.
	for k, b := range lx.blockComments {
		if strings.HasPrefix(line[i:], b[0]) {
			st.mode, st.depth = lexModeComment+k, 0
			out.add(i+len(b[0]), TokenComment)
			return i + len(b[0])
		}
	}
	for _, p := range lx.lineComments {
		if strings.HasPrefix(line[i:], p) {
			out.add(n, TokenComment)
			return n
		}
	}
	if j, ok := lx.stringStart(st, i, ""); ok {
		return j
	}
	if lx.directives && c == '#' && strings.TrimLeft(line[:i], " \t") == "" {
		j := i + 1
		for j < n && (line[j] == ' ' || line[j] == '\t') {
			j++
		}
		for j < n && isIdentByte(line[j]) {
			j++
		}
		out.add(j, TokenMeta)
		return j
	}
	if c == lx.decorator && lx.decorator != 0 && i+1 < n && identStart(line[i+1:], lx.identChars) {
		j := i + 1
		for j < n && (identPart(line[j:], lx.identChars) || line[j] == '.') {
			_, k := utf8.DecodeRuneInString(line[j:])
			j += k
		}
		out.add(j, TokenMeta)
		return j
	}
	if lx.variables && c == '$' && i+1 < n {
		if j := dollarEnd(line, i); j > i+1 {
			out.add(j, TokenVariable)
			st.prevDot, st.regexOK = false, false
			return j
		}
	}
	// Numbers.
	if c >= '0' && c <= '9' || c == '.' && i+1 < n && line[i+1] >= '0' && line[i+1] <= '9' && !st.prevDot {
		hex := c == '0' && i+1 < n && (line[i+1] == 'x' || line[i+1] == 'X')
		j := i + 1
		for j < n {
			d := line[j]
			switch {
			case d >= '0' && d <= '9', d >= 'a' && d <= 'z', d >= 'A' && d <= 'Z', d == '_':
				j++
				continue
			case d == '.' && j+1 < n && line[j+1] >= '0' && line[j+1] <= '9':
				j++
				continue
			case d == '+' || d == '-':
				e := line[j-1]
				if !hex && (e == 'e' || e == 'E') || hex && (e == 'p' || e == 'P') {
					j++
					continue
				}
			}
			break
		}
		out.add(j, TokenNumber)
		st.prevDot, st.regexOK = false, false
		return j
	}
	if identStart(line[i:], lx.identChars) {
		j := i
		for j < n && identPart(line[j:], lx.identChars) {
			if line[j] < utf8.RuneSelf {
				j++
			} else {
				_, k := utf8.DecodeRuneInString(line[j:])
				j += k
			}
		}
		word := line[i:j]
		// A string's prefix, as Python's f"…" or Rust's b"…".
		if j < n && lx.prefixes() {
			if k, ok := lx.stringStart(st, j, word); ok {
				return k
			}
		}
		kind := TokenVariable
		key := word
		if lx.foldCase {
			key = strings.ToLower(word)
		}
		if k, ok := lx.words[key]; ok && !st.prevDot {
			kind = k
		} else {
			k := j
			for k < n && (line[k] == ' ' || line[k] == '\t') {
				k++
			}
			next := byte(0)
			if k < n {
				next = line[k]
			}
			switch {
			case lx.keys && next == ':' && (k+1 >= n || line[k+1] != ':'):
				kind = TokenProperty
			case lx.calls && (next == '(' || next == '!' && k+1 < n && line[k+1] == '('):
				kind = TokenFunction
			case st.prevDot:
				kind = TokenProperty
			case lx.capitalTypes && unicode.IsUpper(firstRune(word)):
				kind = TokenType
			}
		}
		out.add(j, kind)
		st.prevDot = false
		st.regexOK = kind == TokenKeyword
		return j
	}
	if lx.regexps && c == '/' && st.regexOK && i+1 < n && line[i+1] != '/' && line[i+1] != '*' {
		j := i + 1
		class := false
		for j < n {
			d := line[j]
			if d == '\\' {
				j += 2
				continue
			}
			if d == '[' {
				class = true
			} else if d == ']' {
				class = false
			} else if d == '/' && !class {
				break
			}
			j++
		}
		if j < n {
			j++
			for j < n && line[j] >= 'a' && line[j] <= 'z' {
				j++
			}
			out.add(j, TokenRegexp)
			st.regexOK = false
			return j
		}
	}
	if strings.IndexByte("()[]{},;", c) >= 0 {
		out.add(i+1, TokenPunctuation)
		st.prevDot = false
		st.regexOK = c != ')' && c != ']' && c != '}'
		return i + 1
	}
	if c == '.' {
		out.add(i+1, TokenPunctuation)
		st.prevDot, st.regexOK = true, false
		return i + 1
	}
	if strings.IndexByte(lx.operators, c) >= 0 {
		j := i + 1
		for j < n && strings.IndexByte(lx.operators, line[j]) >= 0 && !lx.startsComment(line[j:]) {
			j++
		}
		out.add(j, TokenOperator)
		st.prevDot, st.regexOK = false, true
		return j
	}
	_, k := utf8.DecodeRuneInString(line[i:])
	out.add(i+k, TokenText)
	st.prevDot = false
	return i + k
}

func (lx *lexer) prefixes() bool {
	for _, s := range lx.strings {
		if s.prefix != "" {
			return true
		}
	}
	return false
}

func (lx *lexer) startsComment(s string) bool {
	for _, p := range lx.lineComments {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	for _, b := range lx.blockComments {
		if strings.HasPrefix(s, b[0]) {
			return true
		}
	}
	return false
}

// stringStart starts a string at i, after prefix, which it colors as the
// string.
func (lx *lexer) stringStart(st *lexState, i int, prefix string) (int, bool) {
	for k := range lx.strings {
		s := &lx.strings[k]
		if s.prefix != prefix || !strings.HasPrefix(st.line[i:], s.open) {
			continue
		}
		e := i + len(s.open)
		st.out.add(e, s.kind)
		st.mode = lexModeString + k
		st.prevDot = false
		return e, true
	}
	return i, false
}

func isIdentByte(c byte) bool {
	return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9'
}

func identStart(s string, extra string) bool {
	c := s[0]
	if c < utf8.RuneSelf {
		// - and . go on identifiers, but start none.
		return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c != '-' && c != '.' && extra != "" && strings.IndexByte(extra, c) >= 0
	}
	r, _ := utf8.DecodeRuneInString(s)
	return unicode.IsLetter(r)
}

func identPart(s string, extra string) bool {
	c := s[0]
	if c < utf8.RuneSelf {
		return isIdentByte(c) || extra != "" && strings.IndexByte(extra, c) >= 0
	}
	r, _ := utf8.DecodeRuneInString(s)
	return unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.Is(unicode.Mn, r)
}

func firstRune(s string) rune {
	r, _ := utf8.DecodeRuneInString(s)
	return r
}

// words makes a table of keywords: each string lists words of the kind
// before it.
func words(kinds ...any) map[string]TokenKind {
	m := map[string]TokenKind{}
	var k TokenKind
	for _, v := range kinds {
		switch v := v.(type) {
		case TokenKind:
			k = v
		case string:
			for _, w := range strings.Fields(v) {
				m[w] = k
			}
		}
	}
	return m
}
