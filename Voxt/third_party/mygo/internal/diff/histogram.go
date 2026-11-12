package diff

// maxChain is how often an element may occur in a region of a for the
// histogram algorithm to anchor on it, as git's.
const maxChain = 64

// maxDepth bounds the recursion of histogram, past which a region is
// compared with Myers's algorithm.
const maxDepth = 256

// histogram compares a[a0:a1] with b[b0:b1]: it finds the longest common
// run of elements that occur least often in a, and compares what is
// before and after it the same way.
func (d *differ) histogram(a0, a1, b0, b1, depth int) {
	for {
		// Common prefix and suffix.
		for a0 < a1 && b0 < b1 && d.a[a0] == d.b[b0] {
			a0++
			b0++
		}
		for a0 < a1 && b0 < b1 && d.a[a1-1] == d.b[b1-1] {
			a1--
			b1--
		}
		if a0 == a1 || b0 == b1 {
			d.mark(a0, a1, b0, b1)
			return
		}
		if depth > maxDepth {
			d.myers(a0, a1, b0, b1)
			return
		}
		la0, la1, lb0, lb1, ok := d.lcs(a0, a1, b0, b1)
		if !ok {
			d.myers(a0, a1, b0, b1)
			return
		}
		if la0 == la1 {
			// Nothing in common.
			d.mark(a0, a1, b0, b1)
			return
		}
		depth++
		// Recurse into the smaller side, loop on the other.
		if la0-a0+lb0-b0 < a1-la1+b1-lb1 {
			d.histogram(a0, la0, b0, lb0, depth)
			a0, b0 = la1, lb1
		} else {
			d.histogram(la1, a1, lb1, b1, depth)
			a1, b1 = la0, lb0
		}
	}
}

// lcs finds the longest run common to a[a0:a1] and b[b0:b1] whose
// rarest element occurs least often in a. ok is false when every
// element both have occurs too often in a, for Myers's algorithm.
func (d *differ) lcs(a0, a1, b0, b1 int) (la0, la1, lb0, lb1 int, ok bool) {
	n := a1 - a0
	if cap(d.next) < n {
		d.next, d.cnt = make([]int, n), make([]int, n)
	}
	next, cnt := d.next[:n], d.cnt[:n]
	// Clearing costs the map's capacity: a small region takes a new one.
	if d.recs == nil || d.recsCap > 8*n+64 {
		d.recs, d.recsCap = make(map[int32]record, n), n
	} else {
		clear(d.recs)
	}
	recs := d.recs
	// From the end, so each record's chain goes forward from its first.
	for i := a1 - 1; i >= a0; i-- {
		r, seen := recs[d.a[i]]
		if seen {
			next[i-a0] = r.first
			r.count++
		} else {
			next[i-a0] = -1
			r.count = 1
		}
		r.first = i
		recs[d.a[i]] = r
	}
	// How rare each line is: its record's count in the region, as git's
	// CNT, not the count from it to the region's end, which makes the last
	// of many blank lines look unique.
	for i := a0; i < a1; i++ {
		cnt[i-a0] = recs[d.a[i]].count
	}
	best := maxChain + 1
	common := false
	la0, la1 = a0, a0
	for bi := b0; bi < b1; {
		bnext := bi + 1
		r, found := recs[d.b[bi]]
		if !found {
			bi = bnext
			continue
		}
		common = true
		if r.count > best {
			bi = bnext
			continue
		}
		for as := r.first; as >= 0; {
			s, bs, e, be := as, bi, as+1, bi+1
			rc := cnt[as-a0]
			for s > a0 && bs > b0 && d.a[s-1] == d.b[bs-1] {
				s--
				bs--
				if rc > 1 {
					rc = min(rc, cnt[s-a0])
				}
			}
			for e < a1 && be < b1 && d.a[e] == d.b[be] {
				if rc > 1 {
					rc = min(rc, cnt[e-a0])
				}
				e++
				be++
			}
			if bnext < be {
				bnext = be
			}
			if la1-la0 < e-s || rc < best {
				la0, la1, lb0, lb1, best = s, e, bs, be, rc
			}
			// The next occurrence past this run.
			np := next[as-a0]
			for np >= 0 && np < e {
				np = next[np-a0]
			}
			as = np
		}
		bi = bnext
	}
	if common && best > maxChain {
		return 0, 0, 0, 0, false
	}
	return la0, la1, lb0, lb1, true
}
