//go:build windows

// Package device creates Direct3D 11 devices: the device of package d3d11's
// renderer, and that of the Windows backend, which shows the controls of
// windows with a material through DirectComposition without the renderer.
// It depends on nothing of MyGo, so apps that draw no native UI link none
// of the renderer.
package device

import (
	"fmt"
	"syscall"
	"unsafe"
)

var procD3D11CreateDevice = syscall.NewLazyDLL(SystemDir() + `\d3d11.dll`).NewProc("D3D11CreateDevice")

// SystemDir returns the system directory, which system DLLs load from,
// never from the application directory or the search path.
func SystemDir() string {
	buf := make([]uint16, 260)
	n, _, _ := syscall.NewLazyDLL("kernel32.dll").NewProc("GetSystemDirectoryW").Call(uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	if n == 0 || int(n) > len(buf) {
		return `C:\Windows\System32`
	}
	return syscall.UTF16ToString(buf[:n])
}

// New creates a Direct3D 11 device and its immediate context, on the GPU
// or, when there is none, with Windows' software rasterizer (WARP), which
// software reports. The caller releases the device and the context.
func New() (device, context uintptr, software bool, err error) {
	if err := procD3D11CreateDevice.Find(); err != nil {
		return 0, 0, false, err
	}
	levels := []uint32{0xb000, 0xa100, 0xa000} // 11_0, 10_1, 10_0
	const bgraSupport = 0x20
	const hardware, warp = 1, 5
	var hr uintptr
	for _, driver := range []uintptr{hardware, warp} {
		var level uint32
		hr, _, _ = procD3D11CreateDevice.Call(0, driver, 0, bgraSupport,
			uintptr(unsafe.Pointer(&levels[0])), uintptr(len(levels)), 7,
			uintptr(unsafe.Pointer(&device)), uintptr(unsafe.Pointer(&level)), uintptr(unsafe.Pointer(&context)))
		if int32(uint32(hr)) >= 0 {
			return device, context, driver == warp, nil
		}
	}
	return 0, 0, false, fmt.Errorf("d3d11: cannot create a device: %#x", uint32(hr))
}
