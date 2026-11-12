//go:build windows

package windows

import "strconv"

// On ARM64, structs of up to 16 bytes are passed in registers.
func putBounds(controller uintptr, r rect) {
	comCall(controller, ctlPutBounds,
		uintptr(uint32(r.Left))|uintptr(uint32(r.Top))<<32,
		uintptr(uint32(r.Right))|uintptr(uint32(r.Bottom))<<32)
}

// Go's Windows ARM64 calls do not set the floating point registers, which
// ICoreWebView2Controller::put_ZoomFactor reads: zoom with CSS instead.
func putZoom(w *window, f float64) {
	w.Eval("document.documentElement.style.zoom=" + strconv.FormatFloat(f, 'f', -1, 64))
}

const zoomByScript = true
