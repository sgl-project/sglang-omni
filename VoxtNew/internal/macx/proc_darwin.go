//go:build darwin

package macx

import (
	"encoding/binary"
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
)

var procLib struct {
	once   sync.Once
	err    error
	rusage func(pid int32, flavor int32, buf *byte) int32
}

// rusageInfoV0 is RUSAGE_INFO_V0; its struct rusage_info_v0 is 96 bytes:
// a 16-byte UUID, then uint64s (user and system time, idle and interrupt
// wake-ups, page-ins, wired, resident, phys_footprint, start, exit).
const (
	rusageInfoV0 = 0
	rusageSize   = 96
	offResident  = 16 + 6*8
	offFootprint = 16 + 7*8
)

func procMemory(pid int) (ProcMemory, error) {
	procLib.once.Do(func() {
		lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		if err != nil {
			procLib.err = err
			return
		}
		purego.RegisterLibFunc(&procLib.rusage, lib, "proc_pid_rusage")
	})
	if procLib.err != nil {
		return ProcMemory{}, procLib.err
	}
	if pid <= 0 {
		return ProcMemory{}, fmt.Errorf("macx: no process %d", pid)
	}
	var buf [rusageSize]byte
	if rc := procLib.rusage(int32(pid), rusageInfoV0, &buf[0]); rc != 0 {
		return ProcMemory{}, fmt.Errorf("macx: proc_pid_rusage(%d) failed", pid)
	}
	le := binary.LittleEndian // arm64 and amd64
	return ProcMemory{
		Footprint: int64(le.Uint64(buf[offFootprint:])),
		Resident:  int64(le.Uint64(buf[offResident:])),
	}, nil
}
