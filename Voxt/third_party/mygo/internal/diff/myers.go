package diff

// Myers's algorithm in linear space, as git's xdiff runs it (xdiffi.c):
// the middle snake of each region splits it in two, until the cost of a
// region passes a limit, where the furthest diagonals reached split it,
// which bounds the time on large unrelated texts at the price of a
// minimal diff.

const (
	maxCostMin = 256 // XDL_MAX_COST_MIN
	lineMax    = int(^uint(0) >> 2)
)

func (d *differ) myers(a0, a1, b0, b1 int) {
	n := len(d.a) + len(d.b) + 3
	if len(d.myersV) < 2*n {
		d.myersV = make([]int, 2*n)
	}
	cost := 1
	for cost*cost < n {
		cost <<= 1
	}
	d.myersExtra = max(cost, maxCostMin)
	d.recsCmp(a0, a1, b0, b1, false)
}

func (d *differ) recsCmp(off1, lim1, off2, lim2 int, needMin bool) {
	for off1 < lim1 && off2 < lim2 && d.a[off1] == d.b[off2] {
		off1++
		off2++
	}
	for off1 < lim1 && off2 < lim2 && d.a[lim1-1] == d.b[lim2-1] {
		lim1--
		lim2--
	}
	if off1 == lim1 || off2 == lim2 {
		d.mark(off1, lim1, off2, lim2)
		return
	}
	i1, i2, minLo, minHi := d.split(off1, lim1, off2, lim2, needMin)
	d.recsCmp(off1, i1, off2, i2, minLo)
	d.recsCmp(i1, lim1, i2, lim2, minHi)
}

// split finds where to split a[off1:lim1] and b[off2:lim2]: the middle
// snake, or past the cost limit the best diagonal reached.
func (d *differ) split(off1, lim1, off2, lim2 int, needMin bool) (int, int, bool, bool) {
	ha1, ha2 := d.a, d.b
	nd := len(d.a) + len(d.b) + 3
	base := len(d.b) + 1
	// Diagonal k is at base+k of each vector.
	vf, vb := d.myersV[:nd], d.myersV[nd:2*nd]
	dmin, dmax := off1-lim2, lim1-off2
	fmid, bmid := off1-off2, lim1-lim2
	odd := (fmid-bmid)&1 != 0
	fmin, fmax := fmid, fmid
	bmin, bmax := bmid, bmid
	vf[base+fmid] = off1
	vb[base+bmid] = lim1
	for ec := 1; ; ec++ {
		// Forward.
		if fmin > dmin {
			fmin--
			vf[base+fmin-1] = -1
		} else {
			fmin++
		}
		if fmax < dmax {
			fmax++
			vf[base+fmax+1] = -1
		} else {
			fmax--
		}
		for k := fmax; k >= fmin; k -= 2 {
			var i1 int
			if vf[base+k-1] >= vf[base+k+1] {
				i1 = vf[base+k-1] + 1
			} else {
				i1 = vf[base+k+1]
			}
			i2 := i1 - k
			for i1 < lim1 && i2 < lim2 && ha1[i1] == ha2[i2] {
				i1++
				i2++
			}
			vf[base+k] = i1
			if odd && bmin <= k && k <= bmax && vb[base+k] <= i1 {
				return i1, i2, true, true
			}
		}
		// Backward.
		if bmin > dmin {
			bmin--
			vb[base+bmin-1] = lineMax
		} else {
			bmin++
		}
		if bmax < dmax {
			bmax++
			vb[base+bmax+1] = lineMax
		} else {
			bmax--
		}
		for k := bmax; k >= bmin; k -= 2 {
			var i1 int
			if vb[base+k-1] < vb[base+k+1] {
				i1 = vb[base+k-1]
			} else {
				i1 = vb[base+k+1] - 1
			}
			i2 := i1 - k
			for i1 > off1 && i2 > off2 && ha1[i1-1] == ha2[i2-1] {
				i1--
				i2--
			}
			vb[base+k] = i1
			if !odd && fmin <= k && k <= fmax && i1 <= vf[base+k] {
				return i1, i2, true, true
			}
		}
		if needMin || ec < d.myersExtra {
			continue
		}
		// Too costly: split where the furthest diagonals got.
		fbest, fbest1 := -1, -1
		for k := fmax; k >= fmin; k -= 2 {
			i1 := min(vf[base+k], lim1)
			i2 := i1 - k
			if lim2 < i2 {
				i1, i2 = lim2+k, lim2
			}
			if fbest < i1+i2 {
				fbest, fbest1 = i1+i2, i1
			}
		}
		bbest, bbest1 := lineMax, lineMax
		for k := bmax; k >= bmin; k -= 2 {
			i1 := max(off1, vb[base+k])
			i2 := i1 - k
			if i2 < off2 {
				i1, i2 = off2+k, off2
			}
			if i1+i2 < bbest {
				bbest, bbest1 = i1+i2, i1
			}
		}
		if (lim1+lim2)-bbest < fbest-(off1+off2) {
			return fbest1, fbest - fbest1, true, false
		}
		return bbest1, bbest - bbest1, false, true
	}
}
