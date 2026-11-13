//go:build windows

package runtime

import (
	"time"

	"golang.org/x/sys/windows"
)

var getTickCount64 = windows.NewLazySystemDLL("kernel32.dll").NewProc("GetTickCount64")

const maximumProcessPathLength = 32768

// procExe is the executable of a running process; "" when unknown.
func procExe(pid int) string {
	handle, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return ""
	}
	defer windows.CloseHandle(handle)
	buffer := make([]uint16, maximumProcessPathLength)
	size := uint32(len(buffer))
	if err := windows.QueryFullProcessImageName(handle, 0, &buffer[0], &size); err != nil {
		return ""
	}
	return windows.UTF16ToString(buffer[:size])
}

// bootTime is when the system started; zero when unknown.
func bootTime() time.Time {
	uptimeMilliseconds, _, _ := getTickCount64.Call()
	return time.Now().Add(-time.Duration(uptimeMilliseconds) * time.Millisecond)
}
