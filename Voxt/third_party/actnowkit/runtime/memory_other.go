//go:build !darwin

package runtime

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

func physicalMemory() uint64 { return meminfo() }

// meminfo reads MemTotal from /proc/meminfo (Linux).
func meminfo() uint64 {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok || k != "MemTotal" {
			continue
		}
		fields := strings.Fields(v)
		if len(fields) == 0 {
			return 0
		}
		n, err := strconv.ParseUint(fields[0], 10, 64)
		if err != nil {
			return 0
		}
		return n * 1024
	}
	return 0
}
