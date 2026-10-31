package richtext

// seq is a persistent sequence: an AVL tree whose leaves hold up to
// seqChunk items each, as internal/rope's leaves hold text, and whose
// nodes count the items below them. Replacing items copies the leaves it
// touches and the path to them and shares the rest with the sequence
// before it, so a version is a pointer, kept as long as wanted. An
// Text keeps its paragraphs in one, aligned with the lines of
// its text.
type seq[T any] struct {
	left, right *seq[T]
	items       []T // only leaves hold items
	n, height   int
}

const seqChunk = 32

func seqLen[T any](s *seq[T]) int {
	if s == nil {
		return 0
	}
	return s.n
}

func seqHeight[T any](s *seq[T]) int {
	if s == nil {
		return 0
	}
	return s.height
}

func seqLeaf[T any](items []T) *seq[T] {
	if len(items) == 0 {
		return nil
	}
	return &seq[T]{items: items, n: len(items), height: 1}
}

func seqBranch[T any](a, b *seq[T]) *seq[T] {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	return &seq[T]{left: a, right: b, n: a.n + b.n, height: 1 + max(a.height, b.height)}
}

func seqBalance[T any](s *seq[T]) *seq[T] {
	if seqHeight(s.left)-seqHeight(s.right) > 1 {
		a := s.left
		if seqHeight(a.left) < seqHeight(a.right) {
			b := a.right
			a = seqBranch(seqBranch(a.left, b.left), b.right)
		}
		return seqBranch(a.left, seqBranch(a.right, s.right))
	}
	if seqHeight(s.right)-seqHeight(s.left) > 1 {
		b := s.right
		if seqHeight(b.right) < seqHeight(b.left) {
			a := b.left
			b = seqBranch(a.left, seqBranch(a.right, b.right))
		}
		return seqBranch(seqBranch(s.left, b.left), b.right)
	}
	return s
}

// seqJoin returns the items of a followed by those of b.
func seqJoin[T any](a, b *seq[T]) *seq[T] {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	if a.left == nil && b.left == nil && a.n+b.n <= seqChunk {
		items := make([]T, 0, a.n+b.n)
		return seqLeaf(append(append(items, a.items...), b.items...))
	}
	if a.height > b.height+1 {
		return seqBalance(seqBranch(a.left, seqJoin(a.right, b)))
	}
	if b.height > a.height+1 {
		return seqBalance(seqBranch(seqJoin(a, b.left), b.right))
	}
	return seqBranch(a, b)
}

// seqBuild returns a sequence of items, which it keeps.
func seqBuild[T any](items []T) *seq[T] {
	if len(items) <= seqChunk {
		return seqLeaf(items)
	}
	m := len(items) / 2
	return seqBranch(seqBuild(items[:m:m]), seqBuild(items[m:]))
}

// seqSplit returns the items before index at and those from it.
func seqSplit[T any](s *seq[T], at int) (*seq[T], *seq[T]) {
	if s == nil {
		return nil, nil
	}
	if at <= 0 {
		return nil, s
	}
	if at >= s.n {
		return s, nil
	}
	if s.left == nil {
		return seqLeaf(s.items[:at:at]), seqLeaf(s.items[at:])
	}
	if at < s.left.n {
		a, b := seqSplit(s.left, at)
		return a, seqJoin(b, s.right)
	}
	a, b := seqSplit(s.right, at-s.left.n)
	return seqJoin(s.left, a), b
}

// seqReplace returns s with items i up to j replaced by items, which it
// keeps.
func seqReplace[T any](s *seq[T], i, j int, items []T) *seq[T] {
	left, rest := seqSplit(s, i)
	_, right := seqSplit(rest, j-i)
	return seqJoin(seqJoin(left, seqBuild(items)), right)
}

// seqAt returns item i.
func seqAt[T any](s *seq[T], i int) *T {
	for s.left != nil {
		if i < s.left.n {
			s = s.left
		} else {
			i -= s.left.n
			s = s.right
		}
	}
	return &s.items[i]
}

// seqSet returns s with item i replaced by v.
func seqSet[T any](s *seq[T], i int, v T) *seq[T] {
	if s.left == nil {
		items := make([]T, len(s.items))
		copy(items, s.items)
		items[i] = v
		return &seq[T]{items: items, n: s.n, height: 1}
	}
	if i < s.left.n {
		return &seq[T]{left: seqSet(s.left, i, v), right: s.right, n: s.n, height: s.height}
	}
	return &seq[T]{left: s.left, right: seqSet(s.right, i-s.left.n, v), n: s.n, height: s.height}
}

// seqEach calls fn with the items from index from on, and their indices,
// until it returns false.
func seqEach[T any](s *seq[T], from int, fn func(i int, v *T) bool) bool {
	return seqWalk(s, from, 0, fn)
}

func seqWalk[T any](s *seq[T], from, base int, fn func(i int, v *T) bool) bool {
	if s == nil || from >= s.n {
		return true
	}
	if s.left == nil {
		for k := max(from, 0); k < len(s.items); k++ {
			if !fn(base+k, &s.items[k]) {
				return false
			}
		}
		return true
	}
	if from < s.left.n && !seqWalk(s.left, from, base, fn) {
		return false
	}
	return seqWalk(s.right, from-s.left.n, base+s.left.n, fn)
}
