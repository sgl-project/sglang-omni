package editor

import (
	"regexp"
	"slices"
	"time"
)

// TokenKind is what a token of code is, which the editor's theme colors.
type TokenKind uint8

// Kinds of tokens.
const (
	TokenText TokenKind = iota
	TokenKeyword
	TokenType
	TokenFunction
	TokenVariable
	TokenProperty
	TokenConstant
	TokenNumber
	TokenString
	TokenEscape // escapes and interpolations in strings
	TokenRegexp
	TokenComment
	TokenOperator
	TokenPunctuation
	TokenTag       // HTML and XML elements
	TokenAttribute // their attributes
	TokenMeta      // preprocessor lines, decorators, annotations
	TokenHeading
	TokenEmphasis
	TokenStrong
	TokenLink
	TokenCode // code in prose, as Markdown's
	TokenInserted
	TokenDeleted
	TokenInvalid
	tokenKinds
)

// Token is a run of a line of code of one kind, from the end of the
// token before it, or the start of the line, to End, a byte offset in
// the line.
type Token struct {
	End  int
	Kind TokenKind
}

// Tokenizer splits lines of code into tokens. Tokenize appends the tokens
// of line, which has no newline, to tokens, starting in state, and
// returns them with the state the line ends in, which the next line
// starts in: 0 at the start of the text, and states that compare equal
// for lines that go on alike, as being inside a comment.
type Tokenizer interface {
	Tokenize(line string, state uint64, tokens []Token) ([]Token, uint64)
}

// TokenizerFunc is a function that is a Tokenizer.
type TokenizerFunc func(line string, state uint64, tokens []Token) ([]Token, uint64)

// Tokenize calls f.
func (f TokenizerFunc) Tokenize(line string, state uint64, tokens []Token) ([]Token, uint64) {
	return f(line, state, tokens)
}

// BracketPair is an opening and a closing text, as ( and ), that the
// editor matches, closes as the opening is typed, and wraps a selection
// in.
type BracketPair struct {
	Open, Close string
	// NotIn lists kinds of tokens in which typing Open does not close it,
	// as a quote typed in a string.
	NotIn []TokenKind
}

// Language is what a Buffer knows of the language of its text: how to
// highlight it, comment it, match its brackets and indent it.
type Language struct {
	Name string
	// Extensions and Filenames find the language of a file, as ".go" or
	// "Makefile" (LanguageFor).
	Extensions []string
	Filenames  []string
	// LineComment starts a comment to the end of the line, as "//";
	// BlockComment encloses one, as {"/*", "*/"}. Toggling comments uses
	// the first, else the second.
	LineComment  string
	BlockComment [2]string
	// Brackets are matched, highlighted at the caret, and indent the
	// lines between them; AutoClose are closed as they are typed, and
	// wrap the selection.
	Brackets  []BracketPair
	AutoClose []BracketPair
	// IncreaseIndent matches a line after which the next is indented one
	// more level, as one ending in { or :; DecreaseIndent a line that is
	// one level less than the one before, as one starting with end, which
	// typing outdents. Without them, the Brackets decide; and a closing
	// bracket typed first on its line takes its opener's indentation.
	IncreaseIndent *regexp.Regexp
	DecreaseIndent *regexp.Regexp
	// WordChars are characters other than letters, digits and _ that
	// words hold, as - in CSS.
	WordChars string
	// TabSize and HardTabs are the indentation the language's files
	// usually have, which editors take unless set: Go's tabs.
	TabSize  int
	HardTabs bool
	// Tokenizer highlights the text a line at a time, as TextMate
	// grammars do (NewTokenHighlighter): the default highlighter, which
	// Highlighter replaces.
	Tokenizer Tokenizer
	// Highlighter makes the highlighter of a buffer of the language, for
	// highlighting of its own, as a parser's, as tree-sitter's: text is
	// the buffer's text, and invalidate draws the windows showing it
	// again, from any goroutine, for work done in the background. When
	// nil, Tokenizer highlights.
	Highlighter func(text Snapshot, invalidate func()) Highlighter
}

// Highlighter highlights a buffer's text and follows its edits, as Zed's
// syntax map does: editors give it the edits of the text, have it work
// toward the lines they show, and paint the tokens it gives. Its methods
// run on the main thread; what it does in the background, on snapshots,
// it tells of with the invalidate function it was made with.
//
// A highlighter can also know the structure of the text, as a parser's
// does: it may implement SyntaxSelector, Folder, BracketMatcher and
// Indenter, which editors then use rather than tokens and indentation.
type Highlighter interface {
	// Edit follows an edit of the text: old and new are the text before
	// and after it, and changes what it replaced, in order.
	Edit(old, new Snapshot, changes []Change)
	// Highlight works toward the tokens of the lines up to last, for at
	// most budget, and reports whether they are ready. Editors call it
	// before painting lines, and in later frames while it reports false.
	Highlight(last int, budget time.Duration) bool
	// Tokens appends the tokens of line i, whose text is text, to dst,
	// and reports whether they are final: false while they are a guess
	// made before Highlight's work on them is done, which editors show
	// until it is.
	Tokens(i int, text string, dst []Token) ([]Token, bool)
}

// Range is a range of a text's bytes, Start to End.
type Range struct{ Start, End int }

// SyntaxSelector is a Highlighter knowing the nodes of a syntax tree:
// Select Larger (SelectLarger) selects the smallest node holding more
// than the bytes start to end, rather than the word, string or brackets
// around them.
type SyntaxSelector interface {
	Enclosing(start, end int) (Range, bool)
}

// Folder is a Highlighter knowing what folds: FoldEnd returns the last
// line folding line hides, and whether line starts a fold, rather than
// what indentation and brackets say.
type Folder interface {
	FoldEnd(line int) (end int, ok bool)
}

// BracketMatcher is a Highlighter matching brackets, as do and end:
// EnclosingBrackets returns the innermost pair around bytes start to
// end, which editors highlight around the caret and move between. As in
// Zed, a pair is around them when its brackets are, or touch them: the
// opening bracket starts at or before start, and the closing one ends
// at or after end, so that a caret just after a closing bracket is in
// its pair.
type BracketMatcher interface {
	EnclosingBrackets(start, end int) (open, close Range, ok bool)
}

// Indenter is a Highlighter knowing indentation: IndentAfter returns how
// many levels more than the line holding offset (fewer, negative) a line
// that Return opens there is indented.
type Indenter interface {
	IndentAfter(offset int) (levels int, ok bool)
}

// tokenHighlighter highlights with a Tokenizer, a line at a time: it
// keeps the state each line starts in, valid for the lines before
// frontier; past it they are what they were before the edits since,
// which a line starting as it did then, past dirty, confirms.
type tokenHighlighter struct {
	t      Tokenizer
	text   Snapshot
	states []uint64 // the state line i starts in
	// frontier is how many lines from the top start in states known
	// right. confirmed is how many would be, were the edits since the
	// last pass to leave the states after them as they were: a pass that
	// tokenizes past the edits into a line starting as it did confirms
	// the lines up to it, and no further, as those after it were never
	// tokenized since an earlier edit.
	frontier  int
	confirmed int
	dirty     int // the last line an edit changed, -1 for none
	scratch   []Token
}

// NewTokenHighlighter returns the highlighter of text that a Language
// uses by default: t tokenizes each line from the state the line before
// ends in, and an edit tokenizes the lines after it again until one
// starts in the state it started in before, as TextMate grammars and
// Monaco do, a frame's budget at a time.
func NewTokenHighlighter(t Tokenizer, text Snapshot) Highlighter {
	return &tokenHighlighter{t: t, text: text, states: []uint64{0}, frontier: 1, confirmed: 1, dirty: -1}
}

func (h *tokenHighlighter) Edit(old, new Snapshot, changes []Change) {
	h.text = new
	for _, c := range changes {
		oldLine, oldEndLine := old.LineOf(c.OldStart), old.LineOf(c.OldEnd)
		newLine, newEndLine := new.LineOf(c.NewStart), new.LineOf(c.NewEnd)
		removed, added := oldEndLine-oldLine, newEndLine-newLine
		// The states of the lines after the change move with them.
		if from := newLine + 1; from < len(h.states) {
			h.states = slices.Delete(h.states, from, min(len(h.states), from+removed))
			h.states = slices.Insert(h.states, min(from, len(h.states)), make([]uint64, added)...)
		}
		// Line newLine starts as it did; those after it may not, and those
		// after a change within the lines confirmed move with them.
		h.frontier = min(h.frontier, newLine+1)
		if oldEndLine < h.confirmed {
			h.confirmed += added - removed
		} else {
			h.confirmed = min(h.confirmed, newLine+1)
		}
		if h.dirty > oldEndLine {
			h.dirty += added - removed
		}
		h.dirty = max(h.dirty, newEndLine)
	}
}

func (h *tokenHighlighter) Highlight(last int, budget time.Duration) bool {
	n := h.text.LineCount()
	last = min(last, n-1)
	if len(h.states) > n {
		h.states = h.states[:n]
	}
	if len(h.states) == 0 {
		h.states = append(h.states, 0)
	}
	h.frontier = max(1, min(h.frontier, len(h.states)))
	h.confirmed = min(h.confirmed, len(h.states))
	// Whatever way the pass ends, the lines it leaves known are those
	// confirmed: those past where it stopped wait for the next.
	defer func() {
		h.confirmed = h.frontier
		if h.frontier > h.dirty {
			h.dirty = -1
		}
	}()
	start := time.Now()
	count := 0
	for h.frontier <= last {
		stopped := false
		h.text.t.EachLine(h.frontier-1, last, func(i, _ int, line string) bool {
			if count++; count&63 == 0 && time.Since(start) > budget {
				stopped = true
				return false
			}
			var end uint64
			h.scratch, end = h.t.Tokenize(line, h.states[i], h.scratch[:0])
			if i+1 < len(h.states) {
				if h.states[i+1] == end && i+1 > h.dirty && h.confirmed > i+2 {
					// The lines after start as they did, up to those
					// confirmed before the edits.
					h.frontier = h.confirmed
					return false
				}
				h.states[i+1] = end
			} else {
				h.states = append(h.states, end)
			}
			h.frontier = i + 2
			return true
		})
		if stopped {
			return false
		}
	}
	return true
}

func (h *tokenHighlighter) Tokens(i int, text string, dst []Token) ([]Token, bool) {
	state, ok := h.stateOf(i)
	if len(text) > maxTokenize {
		return dst, ok
	}
	dst, _ = h.t.Tokenize(text, state, dst)
	return dst, ok
}

// stateOf returns the state line starts in, which Highlight may not have
// confirmed yet: then the line is tokenized from the state it had before
// the edits since, which is usually right.
func (h *tokenHighlighter) stateOf(line int) (uint64, bool) {
	if line < len(h.states) {
		return h.states[line], line < h.frontier
	}
	return 0, false
}

// maxTokenize leaves longer lines plain, as Pierre's
// tokenizeMaxLineLength does.
const maxTokenize = 10000

// syntaxState is a buffer's highlighter and the text it follows.
type syntaxState struct {
	lang *Language
	h    Highlighter
	text Snapshot
}

// highlighter returns the buffer's highlighter, following the edits
// since it was last used, or nil for plain text. Main thread only.
func (b *Buffer) highlighter() Highlighter {
	b.mu.Lock()
	lang := b.language
	if lang == nil || lang.Tokenizer == nil && lang.Highlighter == nil {
		b.mu.Unlock()
		return nil
	}
	s := b.syntax
	if s == nil || s.lang != lang {
		s = &syntaxState{lang: lang, text: Snapshot{b.text, b.version, b.crlf}}
		b.syntax = s
	}
	b.mu.Unlock()
	if s.h == nil {
		s.h = newHighlighter(lang, s.text, b.invalidate)
	}
	entries, version, ok := b.entriesSince(s.text.version)
	if !ok {
		// Too far behind: a highlighter of the text as it is.
		s.text = b.Snapshot()
		s.h = newHighlighter(lang, s.text, b.invalidate)
		return s.h
	}
	for _, e := range entries {
		next := Snapshot{e.text, e.version, b.crlf}
		s.h.Edit(s.text, next, publicChanges(e.changes))
		s.text = next
	}
	_ = version
	return s.h
}

func newHighlighter(l *Language, text Snapshot, invalidate func()) Highlighter {
	if l.Highlighter != nil {
		return l.Highlighter(text, invalidate)
	}
	return NewTokenHighlighter(l.Tokenizer, text)
}
