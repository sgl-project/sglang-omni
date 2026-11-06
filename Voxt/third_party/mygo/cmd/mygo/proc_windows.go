//go:build windows

package main

import (
	"os/exec"
	"strings"
	"syscall"
	"unsafe"
)

// shell runs a command line with cmd, which reads its own command line:
// quoted as an argument, the line would reach it with its quotes escaped
// as \", which cmd does not understand. /s makes it run what is between
// the outer quotes as it is.
func shell(line string) *exec.Cmd {
	cmd := exec.Command("cmd")
	cmd.SysProcAttr = &syscall.SysProcAttr{CmdLine: `cmd /s /c "` + line + `"`}
	return cmd
}

func setProcessGroup(cmd *exec.Cmd) {}

func terminate(cmd *exec.Cmd) { kill(cmd) }

func interrupt(cmd *exec.Cmd) { kill(cmd) }

// webviewProcess names WebView2's processes, which kill spares.
var webviewProcess = "msedgewebview2.exe"

// kill ends a process and those it started, except WebView2's. The browser
// process of a WebView2 user data folder runs under the first app that used
// the folder, and a build that mygo dev starts right after the previous one
// may find it still running and use it too: killing it would leave the new
// build's pages blank. WebView2's processes exit on their own once no app
// uses them.
func kill(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	tree := processTree(uint32(cmd.Process.Pid))
	// Killing the process fails once it has exited, when its id, and so the
	// tree, may belong to other processes.
	if cmd.Process.Kill() != nil {
		return
	}
	for _, pid := range tree[1:] {
		h, err := syscall.OpenProcess(syscall.PROCESS_TERMINATE, false, pid)
		if err != nil {
			continue
		}
		_ = syscall.TerminateProcess(h, 1)
		_ = syscall.CloseHandle(h)
	}
}

// processTree returns a process and its descendants, root first, leaving
// out WebView2's processes and theirs. A process counts as a child only if
// it started after its parent, whose id Windows may have given to another
// process since.
func processTree(root uint32) []uint32 {
	snap, err := syscall.CreateToolhelp32Snapshot(syscall.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return []uint32{root}
	}
	defer syscall.CloseHandle(snap)
	type proc struct {
		pid  uint32
		name string
	}
	children := map[uint32][]proc{}
	var e syscall.ProcessEntry32
	e.Size = uint32(unsafe.Sizeof(e))
	for err := syscall.Process32First(snap, &e); err == nil; err = syscall.Process32Next(snap, &e) {
		if e.ProcessID != e.ParentProcessID {
			children[e.ParentProcessID] = append(children[e.ParentProcessID], proc{e.ProcessID, syscall.UTF16ToString(e.ExeFile[:])})
		}
	}
	tree := []uint32{root}
	for i := 0; i < len(tree); i++ {
		parent, ok := started(tree[i])
		if !ok {
			continue
		}
		for _, c := range children[tree[i]] {
			if strings.EqualFold(c.name, webviewProcess) {
				continue
			}
			if at, ok := started(c.pid); ok && at >= parent {
				tree = append(tree, c.pid)
			}
		}
	}
	return tree
}

// started returns when a process started, in 100-nanosecond intervals.
func started(pid uint32) (int64, bool) {
	const processQueryLimitedInformation = 0x1000
	h, err := syscall.OpenProcess(processQueryLimitedInformation, false, pid)
	if err != nil {
		return 0, false
	}
	defer syscall.CloseHandle(h)
	var created, exited, kernel, user syscall.Filetime
	if syscall.GetProcessTimes(h, &created, &exited, &kernel, &user) != nil {
		return 0, false
	}
	return int64(created.HighDateTime)<<32 | int64(created.LowDateTime), true
}

func ignoreTTIN() {}
