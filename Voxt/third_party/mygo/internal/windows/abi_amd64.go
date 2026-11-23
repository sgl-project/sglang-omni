//go:build windows

package windows

import (
	"math"
	"unsafe"
)

// On x64, structs larger than 8 bytes are passed by reference.
func putBounds(controller uintptr, r rect) {
	comCall(controller, ctlPutBounds, uintptr(unsafe.Pointer(&r)))
}

// Go's Windows x64 calls copy the first integer arguments to the XMM
// registers as well, so a double can be passed as its bits.
func putZoom(w *window, f float64) {
	comCall(w.controller, ctlPutZoomFactor, uintptr(math.Float64bits(f)))
}

const zoomByScript = false
