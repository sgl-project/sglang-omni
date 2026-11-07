package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"strings"
	"syscall"
	"unsafe"

	"github.com/ebitengine/purego"
)

var (
	procPidRusage   func(pid, flavor int32, buf unsafe.Pointer) int32
	procListAllPids func(buf unsafe.Pointer, size int32) int32
	procPidPath     func(pid int32, buf unsafe.Pointer, size uint32) int32
	// responsibleFor is private API: nil where libSystem lacks it.
	responsibleFor func(pid int32) int32
)

func init() {
	lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		panic(err)
	}
	purego.RegisterLibFunc(&procPidRusage, lib, "proc_pid_rusage")
	purego.RegisterLibFunc(&procListAllPids, lib, "proc_listallpids")
	purego.RegisterLibFunc(&procPidPath, lib, "proc_pidpath")
	if _, err := purego.Dlsym(lib, "responsibility_get_pid_responsible_for_pid"); err == nil {
		purego.RegisterLibFunc(&responsibleFor, lib, "responsibility_get_pid_responsible_for_pid")
	}
}

// state is the WebKit processes that ran before the app started.
type state map[int32]bool

func before() state {
	s := state{}
	for _, pid := range webKit() {
		s[pid] = true
	}
	return s
}

// usage counts the WebKit processes that started with the app and share
// its responsible process: WKWebView's, for its page, GPU and network.
func usage(pid int, s state) (app, total uint64, procs []int, err error) {
	app, err = footprint(int32(pid))
	if err != nil {
		return 0, 0, nil, err
	}
	total, procs = app, []int{pid}
	for _, p := range webKit() {
		if s[p] || responsibleFor != nil && responsibleFor(p) != responsibleFor(int32(pid)) {
			continue
		}
		if f, err := footprint(p); err == nil {
			total += f
			procs = append(procs, int(p))
		}
	}
	return app, total, procs, nil
}

// footprint returns the physical footprint of pid: ri_phys_footprint of
// its rusage_info_v2, after a 16-byte UUID and seven counters.
func footprint(pid int32) (uint64, error) {
	var info [512]byte
	if procPidRusage(pid, 2, unsafe.Pointer(&info[0])) != 0 {
		return 0, fmt.Errorf("proc_pid_rusage(%d) failed", pid)
	}
	return binary.LittleEndian.Uint64(info[72:]), nil
}

// webKit returns the processes of WebKit's XPC services.
func webKit() []int32 {
	pids := make([]int32, max(procListAllPids(nil, 0), 4096)+256)
	n := procListAllPids(unsafe.Pointer(&pids[0]), int32(4*len(pids)))
	var out []int32
	var path [4096]byte
	for _, pid := range pids[:max(n, 0)] {
		if l := procPidPath(pid, unsafe.Pointer(&path[0]), uint32(len(path))); l > 0 && strings.Contains(string(path[:l]), "/com.apple.WebKit.") {
			out = append(out, pid)
		}
	}
	return out
}

// stop asks the app to quit, which its WebKit processes follow.
func stop(p *os.Process, _ []int) { p.Signal(syscall.SIGTERM) }
