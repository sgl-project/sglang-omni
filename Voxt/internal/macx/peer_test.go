package macx

import (
	"encoding/binary"
	"testing"
)

// TestProgramOf: past shells and this executable to the first other
// program; launchd's own job stops at launchd; no name when it runs out.
func TestProgramOf(t *testing.T) {
	const self = "/Applications/Voxt.app/Contents/MacOS/Voxt"
	paths := map[int]string{
		50: self, 40: "/bin/zsh", 30: "/usr/bin/login", 20: "/System/Applications/Terminal.app/Contents/MacOS/Terminal",
		11: self, 10: "/bin/bash", 1: "/sbin/launchd",
		5: "/usr/bin/env", 6: self,
	}
	parents := map[int]int{50: 40, 40: 30, 30: 20, 11: 10, 10: 1, 6: 5, 5: 0}
	parent := func(p int) (int, bool) { n, ok := parents[p]; return n, ok }
	path := func(p int) string { return paths[p] }
	for _, c := range []struct {
		pid  int
		want string
	}{{50, "Terminal"}, {11, "launchd"}, {6, ""}, {99, ""}} {
		if got := programOf(c.pid, parent, path, self); got != c.want {
			t.Errorf("pid %d: got %q want %q", c.pid, got, c.want)
		}
	}
}

// TestProgramOfThroughRootOwnedParent: proc_pidinfo refuses the long BSD
// flavor for a root-owned login, so the walk must get past it with the
// short one. Terminal -> login (root) -> bash -> Voxt.
func TestProgramOfThroughRootOwnedParent(t *testing.T) {
	const self = "/Applications/Voxt.app/Contents/MacOS/Voxt"
	const longFlavor, longSize, longParent = 3, 136, 16
	paths := map[int]string{
		50: self, 40: "/bin/bash", 30: "/usr/bin/login",
		20: "/System/Applications/Terminal.app/Contents/MacOS/Terminal",
	}
	parents := map[int]uint32{50: 40, 40: 30, 30: 20}
	rootOwned := map[int]bool{30: true}
	proc := func(pid int, flavor int32, buf []byte) int32 {
		ppid, ok := parents[pid]
		if !ok {
			return 0
		}
		switch {
		case flavor == longFlavor && len(buf) == longSize && !rootOwned[pid]:
			binary.LittleEndian.PutUint32(buf[longParent:], ppid)
			return longSize
		case flavor == 13 && len(buf) == 64:
			binary.LittleEndian.PutUint32(buf[0:], uint32(pid))
			binary.LittleEndian.PutUint32(buf[4:], ppid)
			return 64
		}
		return 0
	}
	parent := func(pid int) (int, bool) {
		return parentFromInfo(func(flavor int32, buf []byte) int32 { return proc(pid, flavor, buf) })
	}
	path := func(p int) string { return paths[p] }
	if got := programOf(50, parent, path, self); got != "Terminal" {
		t.Errorf("got %q want Terminal", got)
	}
	if _, ok := parent(99); ok {
		t.Error("a process proc_pidinfo refuses has no parent")
	}
}
