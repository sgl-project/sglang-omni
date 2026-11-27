//go:build darwin

package macx

import (
	"fmt"
	"net"
	"syscall"
)

// From <sys/un.h>.
const (
	solLocal     = 0
	localPeerPID = 2
	pidPathMax   = 4096
)

func peerPID(c *net.UnixConn) (int, error) {
	raw, err := c.SyscallConn()
	if err != nil {
		return 0, err
	}
	var pid int
	var sockErr error
	if err := raw.Control(func(fd uintptr) {
		pid, sockErr = syscall.GetsockoptInt(int(fd), solLocal, localPeerPID)
	}); err != nil {
		return 0, err
	}
	if sockErr != nil {
		return 0, fmt.Errorf("macx: LOCAL_PEERPID: %w", sockErr)
	}
	return pid, nil
}

func procParent(pid int) (int, bool) {
	if loadProcLib() != nil {
		return 0, false
	}
	return parentFromInfo(func(flavor int32, buf []byte) int32 {
		return procLib.info(int32(pid), flavor, 0, &buf[0], int32(len(buf)))
	})
}

func procPath(pid int) string {
	if loadProcLib() != nil {
		return ""
	}
	var buf [pidPathMax]byte
	n := procLib.path(int32(pid), &buf[0], pidPathMax)
	if n <= 0 {
		return ""
	}
	return string(buf[:n])
}
