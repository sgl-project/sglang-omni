//go:build windows

package main

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"golang.org/x/sys/windows"
)

const processQueryLimitedInformation = 0x1000

var (
	user32                  = windows.NewLazySystemDLL("user32.dll")
	procGetForegroundWindow = user32.NewProc("GetForegroundWindow")
)

func windowsVersion() string {
	version := windows.RtlGetVersion()
	if version == nil {
		return "Windows"
	}
	return fmt.Sprintf("Windows %d.%d.%d", version.MajorVersion, version.MinorVersion, version.BuildNumber)
}

func frontmostApp() (proto.App, error) {
	windowHandle, _, callErr := procGetForegroundWindow.Call()
	if windowHandle == 0 {
		return proto.App{}, fmt.Errorf("GetForegroundWindow: %w", callErr)
	}
	return appForWindow(windowHandle)
}

func appForWindow(windowHandle uintptr) (proto.App, error) {
	var processID uint32
	if _, err := windows.GetWindowThreadProcessId(windows.HWND(windowHandle), &processID); err != nil {
		return proto.App{}, err
	}
	if processID == 0 {
		return proto.App{}, fmt.Errorf("foreground window has no process")
	}
	return appForProcess(processID), nil
}

func appForProcess(processID uint32) proto.App {
	app := proto.App{PID: int(processID)}
	process, err := windows.OpenProcess(processQueryLimitedInformation, false, processID)
	if err != nil {
		return app
	}
	defer windows.CloseHandle(process)
	path := make([]uint16, windows.MAX_LONG_PATH)
	pathLength := uint32(len(path))
	if err := windows.QueryFullProcessImageName(process, 0, &path[0], &pathLength); err != nil {
		return app
	}
	app.Name = strings.TrimSuffix(filepath.Base(windows.UTF16ToString(path[:pathLength])), ".exe")
	return app
}
