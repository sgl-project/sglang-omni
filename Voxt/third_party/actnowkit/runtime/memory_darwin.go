//go:build darwin

package runtime

import (
	"encoding/binary"
	"syscall"
)

func physicalMemory() uint64 {
	s, err := syscall.Sysctl("hw.memsize")
	if err != nil {
		return 0
	}
	// Sysctl returns the raw 8 bytes, minus one trailing NUL: pad back.
	b := make([]byte, 8)
	copy(b, s)
	return binary.LittleEndian.Uint64(b)
}
