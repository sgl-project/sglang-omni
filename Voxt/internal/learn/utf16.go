package learn

import "unicode/utf16"

// utf16Len is s's length in UTF-16 units, the helper's ranges.
func utf16Len(s string) int {
	n := 0
	for _, r := range s {
		n += utf16.RuneLen(r)
	}
	return n
}

// runeAtUTF16 converts a UTF-16 offset into s to a rune index: -1 when it
// is outside s or in the middle of a surrogate pair.
func runeAtUTF16(s string, unit int) int {
	if unit < 0 {
		return -1
	}
	at, i := 0, 0
	for _, r := range s {
		if at >= unit {
			break
		}
		at += utf16.RuneLen(r)
		i++
	}
	if at != unit {
		return -1
	}
	return i
}
