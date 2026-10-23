// Package rope stores text as a persistent tree of chunks of UTF-8: an
// AVL tree whose leaves own up to 4 KB of the text each, and whose nodes
// sum the bytes, runes, UTF-16 units and newlines below them, as Zed's
// rope sums its chunks. An edit copies the chunks it touches and the path
// to them and shares the rest of the tree with the text before it, so a
// version of the text is a pointer, kept as long as wanted and read from
// any goroutine while others are made. ui.TextBuffer and ui/editor's
// buffers keep their text in it.
package rope

import (
	"encoding/binary"
	"fmt"
	"io"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// ChunkBytes is the most bytes a leaf holds.
const ChunkBytes = 4096

// Node is a node of the tree: a leaf holding text, or a branch of two.
// A nil node is the empty text.
type Node struct {
	left, right *Node
	text        string // only leaves own bytes
	sum         Summary
	height      int
}

// Summary counts a text: its bytes, runes, UTF-16 units and newlines.
type Summary struct{ Bytes, Runes, Units, Lines int }

func (a Summary) add(b Summary) Summary {
	return Summary{a.Bytes + b.Bytes, a.Runes + b.Runes, a.Units + b.Units, a.Lines + b.Lines}
}

// Sum returns the summary of the text of n.
func Sum(n *Node) Summary {
	if n == nil {
		return Summary{}
	}
	return n.sum
}

func height(n *Node) int {
	if n == nil {
		return 0
	}
	return n.height
}

// Summarize counts s.
func Summarize(s string) Summary {
	n := CountRunes(s)
	return Summary{len(s), n, unitsForRunes(s, n), strings.Count(s, "\n")}
}

// CountRunes counts the runes of s, ASCII eight bytes at a time, as most
// code and logs are.
func CountRunes(s string) int {
	all := s
	for len(s) >= 8 {
		if binary.LittleEndian.Uint64([]byte(s))&0x8080808080808080 != 0 {
			return utf8.RuneCountInString(all)
		}
		s = s[8:]
	}
	for i := range len(s) {
		if s[i] >= utf8.RuneSelf {
			return utf8.RuneCountInString(all)
		}
	}
	return len(all)
}

func unitsForRunes(s string, n int) int {
	if n == len(s) {
		return n
	}
	for _, r := range s {
		if r > 0xffff {
			n++
		}
	}
	return n
}

func leaf(s string) *Node {
	if s == "" {
		return nil
	}
	return &Node{text: strings.Clone(s), sum: Summarize(s), height: 1}
}

func branch(a, b *Node) *Node {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	return &Node{left: a, right: b, sum: a.sum.add(b.sum), height: 1 + max(a.height, b.height)}
}

func balance(n *Node) *Node {
	if height(n.left)-height(n.right) > 1 {
		a := n.left
		if height(a.left) < height(a.right) {
			b := a.right
			a = branch(branch(a.left, b.left), b.right)
		}
		return branch(a.left, branch(a.right, n.right))
	}
	if height(n.right)-height(n.left) > 1 {
		b := n.right
		if height(b.right) < height(b.left) {
			a := b.left
			b = branch(a.left, branch(a.right, b.right))
		}
		return branch(branch(n.left, b.left), b.right)
	}
	return n
}

// Join returns the text of a followed by that of b.
func Join(a, b *Node) *Node {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	if a.left == nil && b.left == nil && a.sum.Bytes+b.sum.Bytes <= ChunkBytes {
		return leaf(a.text + b.text)
	}
	if a.height > b.height+1 {
		return balance(branch(a.left, Join(a.right, b)))
	}
	if b.height > a.height+1 {
		return balance(branch(Join(a, b.left), b.right))
	}
	return branch(a, b)
}

// Build returns a tree of s, owning its bytes, so that a small surviving
// fragment cannot pin a large input string.
func Build(s string) *Node {
	var leaves []*Node
	for start := 0; start < len(s); {
		end := start
		for end < len(s) && end-start < ChunkBytes {
			_, n := utf8.DecodeRuneInString(s[end:])
			end += n
		}
		leaves = append(leaves, leaf(s[start:end]))
		start = end
	}
	var build func(int, int) *Node
	build = func(a, z int) *Node {
		if a == z {
			return nil
		}
		if z-a == 1 {
			return leaves[a]
		}
		m := (a + z) / 2
		return branch(build(a, m), build(m, z))
	}
	return build(0, len(leaves))
}

// Split returns the text before byte at and the text from it.
func Split(n *Node, at int) (*Node, *Node) {
	if n == nil {
		return nil, nil
	}
	if at <= 0 {
		return nil, n
	}
	if at >= n.sum.Bytes {
		return n, nil
	}
	if n.left == nil {
		return leaf(n.text[:at]), leaf(n.text[at:])
	}
	if at < n.left.sum.Bytes {
		a, b := Split(n.left, at)
		return a, Join(b, n.right)
	}
	a, b := Split(n.right, at-n.left.sum.Bytes)
	return Join(n.left, a), b
}

// ReplaceBytes returns the text with bytes a to b replaced by s.
func ReplaceBytes(n *Node, a, b int, s string) *Node {
	total := Sum(n).Bytes
	a, b = max(0, min(a, total)), max(0, min(b, total))
	left, rest := Split(n, a)
	_, right := Split(rest, b-a)
	return Join(Join(left, Build(s)), right)
}

// Replace returns the text with runes start to end replaced by s.
func Replace(n *Node, start, end int, s string) *Node {
	total := Sum(n).Runes
	a, z := max(0, min(start, end, total)), max(0, min(max(start, end), total))
	return ReplaceBytes(n, ByteOfRune(n, a), ByteOfRune(n, z), s)
}

// ByteOfRune returns the byte rune index starts at.
func ByteOfRune(n *Node, index int) int {
	index = max(0, min(index, Sum(n).Runes))
	offset := 0
	for n != nil && n.left != nil {
		if index < n.left.sum.Runes {
			n = n.left
		} else {
			index -= n.left.sum.Runes
			offset += n.left.sum.Bytes
			n = n.right
		}
	}
	if n == nil {
		return offset
	}
	if n.sum.Bytes == n.sum.Runes {
		return offset + index
	}
	for at := range n.text {
		if index == 0 {
			return offset + at
		}
		index--
	}
	return offset + len(n.text)
}

// RuneOfUnit returns the rune holding UTF-16 unit index, or after it.
func RuneOfUnit(n *Node, index int) int {
	offset := 0
	for n != nil && n.left != nil {
		if index < n.left.sum.Units {
			n = n.left
		} else {
			index -= n.left.sum.Units
			offset += n.left.sum.Runes
			n = n.right
		}
	}
	if n == nil {
		return offset
	}
	if n.sum.Runes == n.sum.Units {
		return offset + min(index, n.sum.Runes)
	}
	for _, r := range n.text {
		count := utf16.RuneLen(r)
		if index < count {
			break
		}
		index -= count
		offset++
	}
	return offset
}

// ByteOfUnit returns the byte of UTF-16 unit index, rounded down to a
// rune.
func ByteOfUnit(n *Node, index int) int {
	off := 0
	for n != nil && n.left != nil {
		if index < n.left.sum.Units {
			n = n.left
		} else {
			index -= n.left.sum.Units
			off += n.left.sum.Bytes
			n = n.right
		}
	}
	if n == nil {
		return off
	}
	for i, c := range n.text {
		k := utf16.RuneLen(c)
		if k < 0 {
			k = 1
		}
		if index < k {
			return off + i
		}
		index -= k
	}
	return off + len(n.text)
}

// Prefix returns the summary of the text before byte at.
func Prefix(n *Node, at int) Summary {
	if n == nil || at <= 0 {
		return Summary{}
	}
	if at >= n.sum.Bytes {
		return n.sum
	}
	if n.left == nil {
		return Summarize(n.text[:at])
	}
	if at <= n.left.sum.Bytes {
		return Prefix(n.left, at)
	}
	return n.left.sum.add(Prefix(n.right, at-n.left.sum.Bytes))
}

// LineStart returns the byte line starts at, the end of the text past
// the last line.
func LineStart(n *Node, line int) int {
	if line <= 0 {
		return 0
	}
	if line > Sum(n).Lines {
		return Sum(n).Bytes
	}
	offset := 0
	for n.left != nil {
		if line <= n.left.sum.Lines {
			n = n.left
		} else {
			line -= n.left.sum.Lines
			offset += n.left.sum.Bytes
			n = n.right
		}
	}
	from := 0
	for range line {
		from += strings.IndexByte(n.text[from:], '\n') + 1
	}
	return offset + from
}

// Read writes the bytes start to end to out.
func Read(n *Node, start, end int, out *strings.Builder) {
	if n == nil || start >= end {
		return
	}
	if n.left == nil {
		out.WriteString(n.text[start:end])
		return
	}
	split := n.left.sum.Bytes
	if start < split {
		Read(n.left, start, min(end, split), out)
	}
	if end > split {
		Read(n.right, max(0, start-split), end-split, out)
	}
}

// Chunks calls fn with the text from a to b, a piece at a time, until fn
// returns false.
func Chunks(n *Node, a, b int, fn func(s string) bool) {
	a, b = max(0, a), min(b, Sum(n).Bytes)
	var walk func(n *Node, off int) bool
	walk = func(n *Node, off int) bool {
		if n == nil || off >= b || off+n.sum.Bytes <= a {
			return true
		}
		if n.left == nil {
			lo, hi := max(a-off, 0), min(b-off, len(n.text))
			return fn(n.text[lo:hi])
		}
		return walk(n.left, off) && walk(n.right, off+n.left.sum.Bytes)
	}
	walk(n, 0)
}

// WriteTo writes the text to w, a chunk at a time.
func WriteTo(n *Node, w io.Writer) (int64, error) {
	var total int64
	var err error
	Chunks(n, 0, Sum(n).Bytes, func(s string) bool {
		var k int
		k, err = io.WriteString(w, s)
		total += int64(k)
		if err == nil && k != len(s) {
			err = io.ErrShortWrite
		}
		return err == nil
	})
	return total, err
}

// Check reports what is wrong with the tree, for tests: an oversized or
// miscounted leaf, or an unbalanced or miscounted branch.
func Check(n *Node) error {
	_, err := check(n)
	return err
}

func check(n *Node) (Summary, error) {
	if n == nil {
		return Summary{}, nil
	}
	if n.left == nil {
		if len(n.text) > ChunkBytes+3 || n.sum != Summarize(n.text) || n.height != 1 {
			return Summary{}, fmt.Errorf("rope: invalid leaf %q", n.text)
		}
		return n.sum, nil
	}
	a, err := check(n.left)
	if err != nil {
		return a, err
	}
	b, err := check(n.right)
	if err != nil {
		return b, err
	}
	d := n.left.height - n.right.height
	if d < -1 || d > 1 || n.height != 1+max(n.left.height, n.right.height) || n.sum != a.add(b) {
		return Summary{}, fmt.Errorf("rope: invalid branch of height %d", n.height)
	}
	return n.sum, nil
}
