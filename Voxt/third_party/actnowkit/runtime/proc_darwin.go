//go:build darwin

package runtime

import (
	"encoding/binary"
	"syscall"
	"time"

	"github.com/hehehai/actnowkit/internal/darwin"
)

// procExe is the executable of a running process; "" when unknown.
func procExe(pid int) string {
	p, err := darwin.ProcPath(pid)
	if err != nil {
		return ""
	}
	return p
}

// bootTime is when the system started (kern.boottime); zero when unknown.
func bootTime() time.Time {
	s, err := syscall.Sysctl("kern.boottime")
	if err != nil || len(s) < 8 {
		return time.Time{}
	}
	b := make([]byte, 16)
	copy(b, s)
	return time.Unix(int64(binary.LittleEndian.Uint64(b[:8])), 0)
}
