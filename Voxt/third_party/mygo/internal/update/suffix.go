package update

// suffixArray returns the suffix array of b, with the empty suffix first,
// by Larsson and Sadakane's qsufsort, as bsdiff builds it.
func suffixArray(b []byte) []int32 {
	n := int32(len(b))
	I := make([]int32, n+1)
	V := make([]int32, n+1)
	var buckets [256]int32
	for _, c := range b {
		buckets[c]++
	}
	for i := 1; i < 256; i++ {
		buckets[i] += buckets[i-1]
	}
	for i := 255; i > 0; i-- {
		buckets[i] = buckets[i-1]
	}
	buckets[0] = 0
	for i, c := range b {
		buckets[c]++
		I[buckets[c]] = int32(i)
	}
	I[0] = n
	for i, c := range b {
		V[i] = buckets[c]
	}
	V[n] = 0
	for i := 1; i < 256; i++ {
		if buckets[i] == buckets[i-1]+1 {
			I[buckets[i]] = -1
		}
	}
	I[0] = -1

	// Negative entries of I are lengths of sorted runs; V holds the group of
	// each suffix, by its first h bytes.
	for h := int32(1); I[0] != -(n + 1); h += h {
		var run int32
		i := int32(0)
		for i < n+1 {
			if I[i] < 0 {
				run -= I[i]
				i -= I[i]
				continue
			}
			if run != 0 {
				I[i-run] = -run
			}
			size := V[I[i]] + 1 - i
			split(I, V, i, size, h)
			i += size
			run = 0
		}
		if run != 0 {
			I[i-run] = -run
		}
	}
	for i := range n + 1 {
		I[V[i]] = i
	}
	return I
}

// split sorts the group of I[start:start+size] by the group of the suffix h
// bytes further, which doubles the bytes the suffixes are sorted by.
func split(I, V []int32, start, size, h int32) {
	for size >= 16 {
		// Three-way partition around the middle key.
		x := V[I[start+size/2]+h]
		var jj, kk int32
		for i := start; i < start+size; i++ {
			switch v := V[I[i]+h]; {
			case v < x:
				jj++
			case v == x:
				kk++
			}
		}
		jj += start
		kk += jj
		i, j, k := start, int32(0), int32(0)
		for i < jj {
			switch v := V[I[i]+h]; {
			case v < x:
				i++
			case v == x:
				I[i], I[jj+j] = I[jj+j], I[i]
				j++
			default:
				I[i], I[kk+k] = I[kk+k], I[i]
				k++
			}
		}
		for jj+j < kk {
			if V[I[jj+j]+h] == x {
				j++
			} else {
				I[jj+j], I[kk+k] = I[kk+k], I[jj+j]
				k++
			}
		}
		if jj > start {
			split(I, V, start, jj-start, h)
		}
		for i := range kk - jj {
			V[I[jj+i]] = kk - 1
		}
		if jj == kk-1 {
			I[jj] = -1
		}
		// The greater keys, iteratively.
		size = start + size - kk
		start = kk
	}
	if size <= 0 {
		return
	}
	// Selection sort for small groups.
	for k, j := start, int32(0); k < start+size; k += j {
		j = 1
		x := V[I[k]+h]
		for i := int32(1); k+i < start+size; i++ {
			v := V[I[k+i]+h]
			if v < x {
				x = v
				j = 0
			}
			if v == x {
				I[k+j], I[k+i] = I[k+i], I[k+j]
				j++
			}
		}
		for i := range j {
			V[I[k+i]] = k + j - 1
		}
		if j == 1 {
			I[k] = -1
		}
	}
}
