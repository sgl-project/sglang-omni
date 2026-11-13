//go:build windows

package runtime

import (
	"unsafe"

	"golang.org/x/sys/windows"
)

var globalMemoryStatusEx = windows.NewLazySystemDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")

type memoryStatusEx struct {
	sizeBytes                     uint32
	memoryLoadPercent             uint32
	totalPhysicalBytes            uint64
	availablePhysicalBytes        uint64
	totalPageFileBytes            uint64
	availablePageFileBytes        uint64
	totalVirtualBytes             uint64
	availableVirtualBytes         uint64
	availableExtendedVirtualBytes uint64
}

func physicalMemory() uint64 {
	status := memoryStatusEx{}
	status.sizeBytes = uint32(unsafe.Sizeof(status))
	result, _, _ := globalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&status)))
	if result == 0 {
		return 0
	}
	return status.totalPhysicalBytes
}
