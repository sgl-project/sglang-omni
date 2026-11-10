//go:build darwin

package macx

import (
	"encoding/binary"
	"fmt"
	"syscall"
)

// pressureLevels are kern.memorystatus_vm_pressure_level's values
// (DISPATCH_MEMORYPRESSURE_NORMAL, _WARN, _CRITICAL).
var pressureLevels = map[uint32]string{1: "normal", 2: "warning", 4: "critical"}

func readSystemMemory() (SystemMemory, error) {
	var m SystemMemory
	level, err := syscall.SysctlUint32("kern.memorystatus_vm_pressure_level")
	if err != nil {
		return m, fmt.Errorf("macx: memory pressure: %w", err)
	}
	m.Pressure = pressureLevels[level]
	if m.Pressure == "" {
		m.Pressure = fmt.Sprintf("level %d", level)
	}
	if available, err := syscall.SysctlUint32("kern.memorystatus_level"); err == nil {
		m.AvailablePercent = int(available)
	}
	// struct xsw_usage: xsu_total, xsu_avail, xsu_used (uint64s), then
	// xsu_pagesize and xsu_encrypted. Sysctl drops a trailing NUL byte.
	if raw, err := syscall.Sysctl("vm.swapusage"); err == nil && len(raw) >= 24 {
		le := binary.LittleEndian // arm64 and amd64
		m.SwapTotalBytes = int64(le.Uint64([]byte(raw[0:8])))
		m.SwapUsedBytes = int64(le.Uint64([]byte(raw[16:24])))
	}
	m.Swapins = sysctlInt64("vm.compressor.swapper.swapins_total")
	m.Swapouts = sysctlInt64("vm.compressor.swapper.swapouts_total")
	return m, nil
}

// sysctlInt64 is a 64-bit counter, 0 when there is none.
func sysctlInt64(name string) int64 {
	raw, err := syscall.Sysctl(name)
	if err != nil {
		return 0
	}
	var buf [8]byte
	copy(buf[:], raw)
	return int64(binary.LittleEndian.Uint64(buf[:]))
}
