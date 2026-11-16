package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
)

type state struct{}

func before() state { return state{} }

// usage counts the app's descendants: WebKitGTK's web and network
// processes, and what they run.
func usage(pid int, _ state) (app, total uint64, procs []int, err error) {
	procs = append([]int{pid}, descendants(pid)...)
	for i, p := range procs {
		pss, err := pssOf(p)
		if err != nil {
			if i == 0 {
				return 0, 0, nil, err
			}
			continue
		}
		if i == 0 {
			app = pss
		}
		total += pss
	}
	return app, total, procs, nil
}

// pssOf returns the proportional set size of pid: its own pages, and its
// share of those it shares.
func pssOf(pid int) (uint64, error) {
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/smaps_rollup", pid))
	if err != nil {
		return 0, err
	}
	for line := range strings.SplitSeq(string(b), "\n") {
		if v, ok := strings.CutPrefix(line, "Pss:"); ok {
			kb, err := strconv.ParseUint(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(v), "kB")), 10, 64)
			return kb << 10, err
		}
	}
	return 0, fmt.Errorf("no Pss in /proc/%d/smaps_rollup", pid)
}

// descendants returns the processes pid started, and theirs.
func descendants(pid int) []int {
	entries, _ := os.ReadDir("/proc")
	children := map[int][]int{}
	for _, e := range entries {
		p, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		b, err := os.ReadFile("/proc/" + e.Name() + "/stat")
		if err != nil {
			continue
		}
		// pid (comm) state ppid …, where comm may hold anything.
		f := strings.Fields(string(b[strings.LastIndexByte(string(b), ')')+1:]))
		if len(f) < 2 {
			continue
		}
		if ppid, err := strconv.Atoi(f[1]); err == nil {
			children[ppid] = append(children[ppid], p)
		}
	}
	var out []int
	for queue := children[pid]; len(queue) > 0; queue = queue[1:] {
		out = append(out, queue[0])
		queue = append(queue, children[queue[0]]...)
	}
	return out
}

// stop asks the app to quit, which its WebKit processes follow.
func stop(p *os.Process, _ []int) { p.Signal(syscall.SIGTERM) }
