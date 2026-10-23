//go:build js

package mygo

import "math/rand/v2"

// randomText returns 26 random base32 characters (128 bits), as
// crypto/rand's Text does, from the runtime's generator, which the
// browser's crypto.getRandomValues seeds: crypto/rand would add some
// 300 KB to the build.
func randomText() string {
	const base32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
	b := make([]byte, 26)
	for i := range b {
		b[i] = base32[rand.IntN(len(base32))]
	}
	return string(b)
}
