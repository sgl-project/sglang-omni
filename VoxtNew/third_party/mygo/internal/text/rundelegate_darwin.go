//go:build darwin

package text

import (
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// The leaves of rundelegate_darwin_*.s.
func runDelegateDealloc()
func runDelegateAscent()
func runDelegateDescent()
func runDelegateWidth()
func runDelegateFuncs() (dealloc, ascent, descent, width uintptr)

// runDelegateCallbacks is a CTRunDelegateCallbacks, version 1.
type runDelegateCallbacks struct {
	version                         int
	dealloc, ascent, descent, width uintptr
}

// delegates makes Core Text's run delegates, which give inline objects
// their size, from metrics the engine keeps, pinned, until it lays out the
// next text.
type delegates struct {
	create    uintptr // CTRunDelegateCreate
	attribute uintptr // kCTRunDelegateAttributeName
	callbacks runDelegateCallbacks
	metrics   [][3]float64
	pinner    runtime.Pinner
}

var runDelegates delegates

// initDelegates finds Core Text's run delegates in the framework text.
func initDelegates(text uintptr, optional func(lib uintptr, name string) uintptr) {
	d := &runDelegates
	if sym, err := purego.Dlsym(text, "CTRunDelegateCreate"); err == nil {
		d.create = sym
	}
	d.attribute = optional(text, "kCTRunDelegateAttributeName")
	dealloc, ascent, descent, width := runDelegateFuncs()
	d.callbacks = runDelegateCallbacks{version: 1, dealloc: dealloc, ascent: ascent, descent: descent, width: width}
}

// reset lets go of the metrics of the last text laid out, and makes room
// for n objects.
func (d *delegates) reset(n int) {
	d.pinner.Unpin()
	d.metrics = make([][3]float64, n)
	if n > 0 {
		d.pinner.Pin(&d.metrics[0])
		d.pinner.Pin(&d.callbacks)
	}
}

// delegate returns a run delegate, owned, giving object k its height and
// width, standing on the baseline.
func (d *delegates) delegate(k int, width, height float32) uintptr {
	if d.create == 0 || d.attribute == 0 || k >= len(d.metrics) {
		return 0
	}
	d.metrics[k] = [3]float64{float64(height), 0, float64(width)}
	r, _, _ := purego.SyscallN(d.create, uintptr(unsafe.Pointer(&d.callbacks)), uintptr(unsafe.Pointer(&d.metrics[k])))
	return r
}
