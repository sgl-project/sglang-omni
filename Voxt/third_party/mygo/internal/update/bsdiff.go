package update

import (
	"bufio"
	"bytes"
	"compress/flate"
	"encoding/binary"
	"errors"
	"io"
)

// Binary patches follow Colin Percival's bsdiff: the new file is made of
// blocks, each adding bytes of the old file to difference bytes, which are
// mostly zeros where code moved and its addresses changed, followed by
// extra bytes that are new. A patch is
//
//	the number of blocks, then the lengths of the first two streams (uvarints),
//	the control stream: per block, the lengths of its difference and extra
//	bytes (uvarints) and how far the old position moves after it (a varint),
//	the difference bytes of all blocks, and their extra bytes,
//
// each stream compressed with DEFLATE on its own, as it compresses best
// alone, and read in order: the updater patches from the delta file.

// Diff returns the patch that turns old into new.
func Diff(old, new []byte) []byte {
	sa := suffixArray(old)
	type block struct {
		diff, extra int
		seek        int
	}
	var blocks []block
	var diff, extra []byte
	var scan, pos, length int
	var lastScan, lastPos, lastOffset int
	for scan < len(new) {
		oldScore := 0
		scan += length
		for scsc := scan; scan < len(new); scan++ {
			length, pos = search(sa, old, new[scan:])
			for ; scsc < scan+length; scsc++ {
				if scsc+lastOffset < len(old) && old[scsc+lastOffset] == new[scsc] {
					oldScore++
				}
			}
			if length == oldScore && length != 0 || length > oldScore+8 {
				break
			}
			if scan+lastOffset < len(old) && old[scan+lastOffset] == new[scan] {
				oldScore--
			}
		}
		if length == oldScore && scan != len(new) {
			continue
		}

		// Extend the previous match forwards and this one backwards, as
		// long as at least half of the bytes match.
		lenF := 0
		for i, s, best := 0, 0, 0; lastScan+i < scan && lastPos+i < len(old); {
			if old[lastPos+i] == new[lastScan+i] {
				s++
			}
			i++
			if s*2-i > best*2-lenF {
				best, lenF = s, i
			}
		}
		lenB := 0
		if scan < len(new) {
			for i, s, best := 1, 0, 0; scan >= lastScan+i && pos >= i; i++ {
				if old[pos-i] == new[scan-i] {
					s++
				}
				if s*2-i > best*2-lenB {
					best, lenB = s, i
				}
			}
		}
		if overlap := lastScan + lenF - (scan - lenB); overlap > 0 {
			s, best, lenS := 0, 0, 0
			for i := range overlap {
				if new[lastScan+lenF-overlap+i] == old[lastPos+lenF-overlap+i] {
					s++
				}
				if new[scan-lenB+i] == old[pos-lenB+i] {
					s--
				}
				if s > best {
					best, lenS = s, i+1
				}
			}
			lenF += lenS - overlap
			lenB -= lenS
		}

		for i := range lenF {
			diff = append(diff, new[lastScan+i]-old[lastPos+i])
		}
		extra = append(extra, new[lastScan+lenF:scan-lenB]...)
		b := block{lenF, scan - lenB - (lastScan + lenF), pos - lenB - (lastPos + lenF)}
		if b.diff == 0 && b.extra == 0 && len(blocks) > 0 {
			// Only a move: the previous block makes it, so that every block
			// but the first makes bytes (Patch relies on it).
			blocks[len(blocks)-1].seek += b.seek
		} else {
			blocks = append(blocks, b)
		}
		lastScan, lastPos, lastOffset = scan-lenB, pos-lenB, pos-scan
	}
	var ctrl []byte
	for _, b := range blocks {
		ctrl = binary.AppendUvarint(ctrl, uint64(b.diff))
		ctrl = binary.AppendUvarint(ctrl, uint64(b.extra))
		ctrl = binary.AppendVarint(ctrl, int64(b.seek))
	}
	ctrl, diff, extra = deflate(ctrl), deflate(diff), deflate(extra)
	patch := binary.AppendUvarint(nil, uint64(len(blocks)))
	patch = binary.AppendUvarint(patch, uint64(len(ctrl)))
	patch = binary.AppendUvarint(patch, uint64(len(diff)))
	return bytes.Join([][]byte{patch, ctrl, diff, extra}, nil)
}

// search returns the longest prefix of b found in old, and where, by a
// binary search of the suffix array of old.
func search(sa []int32, old, b []byte) (length, pos int) {
	lo, hi := 0, len(sa)-1
	for hi-lo >= 2 {
		mid := lo + (hi-lo)/2
		if bytes.Compare(old[sa[mid]:], b) < 0 {
			lo = mid
		} else {
			hi = mid
		}
	}
	x, y := matchLen(old[sa[lo]:], b), matchLen(old[sa[hi]:], b)
	if x > y {
		return x, int(sa[lo])
	}
	return y, int(sa[hi])
}

func matchLen(a, b []byte) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// deflate compresses b.
func deflate(b []byte) []byte {
	var buf bytes.Buffer
	w, _ := flate.NewWriter(&buf, flate.BestCompression)
	w.Write(b)
	w.Close()
	return buf.Bytes()
}

var errDamagedPatch = errors.New("damaged patch")

// Patch writes to w the file of size bytes that patch makes of old, a file
// of oldSize bytes. Patches come from signed updates, but a damaged one
// fails cleanly: it cannot make more than size bytes, bytes read out of
// old count as zeros, as in bspatch, and the caller checks the file made.
func Patch(w io.Writer, size int64, old io.ReaderAt, oldSize int64, patch io.ReaderAt, patchSize int64) error {
	var head [3 * binary.MaxVarintLen64]byte
	n, _ := patch.ReadAt(head[:min(patchSize, int64(len(head)))], 0)
	var blocks, ctrlLen, diffLen uint64
	start, ok := uvarints(head[:n], &blocks, &ctrlLen, &diffLen)
	if !ok || size < 0 || blocks > uint64(size)+1 ||
		ctrlLen > uint64(patchSize-start) || diffLen > uint64(patchSize-start)-ctrlLen {
		return errDamagedPatch
	}
	extraStart := start + int64(ctrlLen) + int64(diffLen)
	ctrl := bufio.NewReader(flate.NewReader(io.NewSectionReader(patch, start, int64(ctrlLen))))
	diff := flate.NewReader(io.NewSectionReader(patch, start+int64(ctrlLen), int64(diffLen)))
	extra := flate.NewReader(io.NewSectionReader(patch, extraStart, patchSize-extraStart))

	buf := make([]byte, 32<<10)
	oldBuf := make([]byte, len(buf))
	var oldPos, left int64 = 0, size
	for range blocks {
		d, err1 := binary.ReadUvarint(ctrl)
		e, err2 := binary.ReadUvarint(ctrl)
		seek, err3 := binary.ReadVarint(ctrl)
		if err1 != nil || err2 != nil || err3 != nil || d > uint64(left) || e > uint64(left)-d {
			return errDamagedPatch
		}
		left -= int64(d + e)
		for d > 0 {
			n := int(min(d, uint64(len(buf))))
			if _, err := io.ReadFull(diff, buf[:n]); err != nil {
				return errDamagedPatch
			}
			clear(oldBuf[:n])
			// Only the part of the block inside old is read.
			if from, to := max(oldPos, 0), min(oldPos+int64(n), oldSize); from < to {
				if _, err := old.ReadAt(oldBuf[from-oldPos:to-oldPos], from); err != nil {
					return err
				}
			}
			for i := range n {
				buf[i] += oldBuf[i]
			}
			if _, err := w.Write(buf[:n]); err != nil {
				return err
			}
			oldPos += int64(n)
			d -= uint64(n)
		}
		if n, err := io.CopyN(w, extra, int64(e)); n != int64(e) {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return errDamagedPatch
			}
			return err
		}
		oldPos += seek
		if oldPos < -(1<<40) || oldPos > 1<<40 {
			return errDamagedPatch
		}
	}
	if left != 0 {
		return errDamagedPatch
	}
	return nil
}

// uvarints decodes uvarints from the start of b into vs and returns how
// many bytes they take.
func uvarints(b []byte, vs ...*uint64) (int64, bool) {
	n := 0
	for _, v := range vs {
		x, k := binary.Uvarint(b[n:])
		if k <= 0 {
			return 0, false
		}
		*v, n = x, n+k
	}
	return int64(n), true
}
