//go:build windows

package d3d11

import (
	"fmt"
	"syscall"
	"unsafe"

	"github.com/egoist/mygo/internal/scene"
)

// RenderOffscreen draws s with a renderer of its own into a window that
// never shows, and returns the premultiplied BGRA rows of its back buffer
// and their stride: for tests, as packages drawing effects have.
func RenderOffscreen(s *scene.Scene) ([]byte, int, error) {
	hwnd, err := newHiddenWindow(s.Width, s.Height)
	if err != nil {
		return nil, 0, err
	}
	defer destroyWindow(hwnd)
	r, err := New(hwnd)
	if err != nil {
		return nil, 0, err
	}
	defer r.Release()
	if err := r.draw(s); err != nil {
		return nil, 0, err
	}
	return r.read()
}

// newHiddenWindow creates a window that never shows, for a swap chain.
func newHiddenWindow(w, h int) (uintptr, error) {
	class, _ := syscall.UTF16PtrFromString("STATIC")
	const wsPopup = 0x80000000
	hwnd, _, err := user32.NewProc("CreateWindowExW").Call(0, uintptr(unsafe.Pointer(class)), 0, wsPopup, 0, 0, uintptr(w), uintptr(h), 0, 0, 0, 0)
	if hwnd == 0 {
		return 0, fmt.Errorf("d3d11: cannot create a window: %w", err)
	}
	return hwnd, nil
}

func destroyWindow(hwnd uintptr) { user32.NewProc("DestroyWindow").Call(hwnd) }

// read copies the back buffer to memory: BGRA rows and their stride.
func (r *Renderer) read() ([]byte, int, error) {
	var back, staging uintptr
	if hr := call(r.swapChain, scGetBuffer, 0, uintptr(unsafe.Pointer(&iidID3D11Texture2D)), uintptr(unsafe.Pointer(&back))); failed(hr) {
		return nil, 0, fmt.Errorf("d3d11: no back buffer: %#x", uint32(hr))
	}
	defer free(&back)
	const usageStaging, cpuAccessRead, mapRead, ctxCopyResource = 3, 0x20000, 1, 47
	desc := texture2DDesc{Width: uint32(r.bw), Height: uint32(r.bh), MipLevels: 1, ArraySize: 1, Format: formatB8G8R8A8Unorm,
		SampleCount: 1, Usage: usageStaging, CPUAccessFlags: cpuAccessRead}
	if hr := call(r.device, devCreateTexture2D, uintptr(unsafe.Pointer(&desc)), 0, uintptr(unsafe.Pointer(&staging))); failed(hr) {
		return nil, 0, fmt.Errorf("d3d11: no staging texture: %#x", uint32(hr))
	}
	defer free(&staging)
	call(r.ctx, ctxCopyResource, staging, back)
	var m mapped
	if hr := call(r.ctx, ctxMap, staging, 0, mapRead, 0, uintptr(unsafe.Pointer(&m))); failed(hr) {
		return nil, 0, fmt.Errorf("d3d11: cannot map: %#x", uint32(hr))
	}
	defer call(r.ctx, ctxUnmap, staging, 0)
	pix := make([]byte, r.w*r.h*4)
	for y := range r.h {
		copy(pix[y*r.w*4:], unsafe.Slice((*byte)(ptr(m.Data+uintptr(y)*uintptr(m.RowPitch))), r.w*4))
	}
	return pix, r.w * 4, nil
}
