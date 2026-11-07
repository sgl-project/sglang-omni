package main

// What this Mac is (from AudioSmith): macOS version, chip, memory, cores,
// for Settings → Diagnostics and bug reports. Read once, in the background
// (sysctl and sw_vers on macOS; /proc on Linux, for development).

import (
	"bufio"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// sysInfo describes the machine.
type sysInfo struct {
	OS     string // "macOS 15.4"
	Chip   string // "Apple M4"
	Memory int64  // bytes
	Cores  int
	Arch   string // "arm64"
}

// readSysInfo asks the system (it runs commands: not on the UI thread).
var readSysInfo = func() sysInfo {
	s := sysInfo{Cores: runtime.NumCPU(), Arch: runtime.GOARCH}
	switch runtime.GOOS {
	case "darwin":
		run := func(name string, args ...string) string {
			out, err := exec.Command(name, args...).Output()
			if err != nil {
				return ""
			}
			return strings.TrimSpace(string(out))
		}
		if v := run("/usr/bin/sw_vers", "-productVersion"); v != "" {
			s.OS = "macOS " + v
		}
		s.Chip = run("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string")
		s.Memory, _ = strconv.ParseInt(run("/usr/sbin/sysctl", "-n", "hw.memsize"), 10, 64)
	case "linux":
		s.OS = "Linux"
		if f, err := os.Open("/etc/os-release"); err == nil {
			sc := bufio.NewScanner(f)
			for sc.Scan() {
				if v, ok := strings.CutPrefix(sc.Text(), "PRETTY_NAME="); ok {
					s.OS = strings.Trim(v, `"`)
				}
			}
			f.Close()
		}
		if f, err := os.Open("/proc/cpuinfo"); err == nil {
			sc := bufio.NewScanner(f)
			for sc.Scan() {
				if k, v, ok := strings.Cut(sc.Text(), ":"); ok && strings.TrimSpace(k) == "model name" {
					s.Chip = strings.TrimSpace(v)
					break
				}
			}
			f.Close()
		}
		if f, err := os.Open("/proc/meminfo"); err == nil {
			sc := bufio.NewScanner(f)
			for sc.Scan() {
				if v, ok := strings.CutPrefix(sc.Text(), "MemTotal:"); ok {
					kb, _ := strconv.ParseInt(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(v), "kB")), 10, 64)
					s.Memory = kb << 10
				}
			}
			f.Close()
		}
	default:
		s.OS = runtime.GOOS
	}
	return s
}

// MemoryGB is the memory in whole GB (as Apple sells it).
func (s sysInfo) MemoryGB() int { return int((s.Memory + (1 << 29)) >> 30) }

// loadSysInfo reads the machine once (in the background).
func (a *app) loadSysInfo() {
	if a.sys != nil || a.sysLoading {
		return
	}
	a.sysLoading = true
	a.async(func() {
		s := readSysInfo()
		a.post(func() { a.sys, a.sysLoading = &s, false })
	})
}
