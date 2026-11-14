package macx

import (
	"encoding/binary"
	"net"
	"os"
	"path/filepath"
)

// peerDepth bounds the walk up a process tree; deeper than this is not
// a real launch chain.
const peerDepth = 16

// peerSkipped are programs that only start the one that matters: a shell,
// env, login. The first process past them names who asked.
var peerSkipped = map[string]bool{"sh": true, "bash": true, "zsh": true, "dash": true, "fish": true, "env": true, "login": true}

// From <sys/proc_info.h>. PROC_PIDT_SHORTBSDINFO also answers for a
// root-owned process such as login, unlike PROC_PIDTBSDINFO. struct
// proc_bsdshortinfo is 64 bytes with pbsi_ppid after one uint32.
const (
	procPIDTShortBSDInfo = 13
	shortBSDInfoSize     = 64
	offShortParentPID    = 4
)

// PeerProgram is the best-effort file name of the program behind a
// connected Unix socket: the peer, or its nearest ancestor past shells and
// this executable itself (the peer is usually `Voxt history`). "" when it
// cannot tell; not on macOS, always. A program can forge it.
func PeerProgram(c net.Conn) string {
	uc, ok := c.(*net.UnixConn)
	if !ok {
		return ""
	}
	pid, err := peerPID(uc)
	if err != nil {
		return ""
	}
	self, _ := os.Executable()
	if resolved, err := filepath.EvalSymlinks(self); err == nil {
		self = resolved
	}
	return programOf(pid, procParent, procPath, self)
}

// programOf walks up from pid; parent and path are the platform's.
func programOf(pid int, parent func(int) (int, bool), path func(int) string, self string) string {
	for range peerDepth {
		p := path(pid)
		if p == "" {
			return ""
		}
		if name := filepath.Base(p); p != self && !peerSkipped[name] {
			return name
		}
		next, ok := parent(pid)
		if !ok || next <= 0 {
			return ""
		}
		pid = next
	}
	return ""
}

// parentFromInfo reads a parent pid through query, the platform's
// proc_pidinfo for one process.
func parentFromInfo(query func(flavor int32, buf []byte) int32) (int, bool) {
	var buf [shortBSDInfoSize]byte
	if query(procPIDTShortBSDInfo, buf[:]) != shortBSDInfoSize {
		return 0, false
	}
	return int(binary.LittleEndian.Uint32(buf[offShortParentPID:])), true
}
