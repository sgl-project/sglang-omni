//go:build windows && (amd64 || arm64)

package windows

import (
	"fmt"
	"sync"
	"syscall"
	"unsafe"

	"github.com/ebitengine/purego"
)

// comCall calls method i of the COM interface obj. Pointers converted to
// uintptr in the call stay valid until it returns (go:uintptrescapes).
//
//go:uintptrescapes
func comCall(obj uintptr, i int, args ...uintptr) uintptr {
	vtbl := *(*uintptr)(native(obj))
	fn := *(*uintptr)(unsafe.Add(native(vtbl), i*int(unsafe.Sizeof(uintptr(0)))))
	var a [16]uintptr
	a[0] = obj
	n := copy(a[1:], args)
	r, _, _ := syscall.SyscallN(fn, a[:n+1]...)
	return r
}

var floatMethods sync.Map // method entry → func(uintptr, float64) uintptr

// comCallFloat calls method i of obj, which takes a double. syscall does
// not pass floating point arguments on ARM64, purego does, as for
// DirectWrite in internal/text.
func comCallFloat(obj uintptr, i int, f float64) uintptr {
	fn := *(*uintptr)(unsafe.Add(native(*(*uintptr)(native(obj))), i*int(unsafe.Sizeof(uintptr(0)))))
	m, ok := floatMethods.Load(fn)
	if !ok {
		var call func(uintptr, float64) uintptr
		purego.RegisterFunc(&call, fn)
		m, _ = floatMethods.LoadOrStore(fn, call)
	}
	return m.(func(uintptr, float64) uintptr)(obj, f)
}

func failed(hr uintptr) bool { return int32(uint32(hr)) < 0 }

func hresultError(what string, hr uintptr) error {
	return fmt.Errorf("mygo: %s failed (HRESULT %#08x)", what, uint32(hr))
}

func addRef(obj uintptr) {
	if obj != 0 {
		comCall(obj, 1)
	}
}

func release(obj uintptr) {
	if obj != 0 {
		comCall(obj, 2)
	}
}

// queryInterface returns obj's implementation of iid, or 0.
func queryInterface(obj uintptr, iid *GUID) uintptr {
	if obj == 0 {
		return 0
	}
	var out uintptr
	if failed(comCall(obj, 0, uintptr(unsafe.Pointer(iid)), uintptr(unsafe.Pointer(&out)))) {
		return 0
	}
	return out
}

// Interfaces a COM object must not claim to implement blindly: callers
// would then marshal through it or treat it as free threaded.
var (
	iidIMarshal     = guid("00000003-0000-0000-c000-000000000046")
	iidIAgileObject = guid("94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90")
	iidINoMarshal   = guid("ecc8691b-c1db-4dc0-855e-65f6c551af49")
)

// A handler is a COM object with a single method besides IUnknown, as the
// WebView2 event and completion handlers are. All handlers share one
// vtable, whose callbacks are created once; each object routes Invoke to
// its Go function. Handlers stay reachable through the handlers map until
// COM releases its last reference.
type handler struct {
	vtbl *[4]uintptr
	refs int32
	fn   func(a, b uintptr)
}

var (
	handlerVtbl [4]uintptr
	handlersMu  sync.Mutex
	handlers    = map[uintptr]*handler{}
)

func init() {
	handlerVtbl = [4]uintptr{
		syscall.NewCallback(func(this, riid, out uintptr) uintptr {
			iid := *(*GUID)(native(riid))
			if iid == iidIMarshal || iid == iidIAgileObject || iid == iidINoMarshal {
				*(*uintptr)(native(out)) = 0
				return eNoInterface
			}
			*(*uintptr)(native(out)) = this
			handlerRef(this, 1)
			return sOK
		}),
		syscall.NewCallback(func(this uintptr) uintptr { return uintptr(handlerRef(this, 1)) }),
		syscall.NewCallback(func(this uintptr) uintptr { return uintptr(handlerRef(this, -1)) }),
		syscall.NewCallback(func(this, a, b uintptr) uintptr {
			handlersMu.Lock()
			h := handlers[this]
			handlersMu.Unlock()
			if h != nil {
				h.fn(a, b)
			}
			return sOK
		}),
	}
}

func handlerRef(this uintptr, delta int32) int32 {
	handlersMu.Lock()
	defer handlersMu.Unlock()
	h := handlers[this]
	if h == nil {
		return 0
	}
	h.refs += delta
	if h.refs <= 0 {
		delete(handlers, this)
		return 0
	}
	return h.refs
}

// newHandler creates a handler object holding one reference, which the
// caller releases once it has handed the object to COM.
func newHandler(fn func(a, b uintptr)) uintptr {
	h := &handler{vtbl: &handlerVtbl, refs: 1, fn: fn}
	p := uintptr(unsafe.Pointer(h))
	handlersMu.Lock()
	handlers[p] = h
	handlersMu.Unlock()
	return p
}

// withHandler passes a new handler to call, which takes its own
// reference if it keeps the handler, and drops ours.
func withHandler(fn func(a, b uintptr), call func(h uintptr) uintptr) uintptr {
	h := newHandler(fn)
	defer release(h)
	return call(h)
}

// IStream helpers.

// readStream reads an IStream to its end.
func readStream(stream uintptr) []byte {
	var out []byte
	buf := make([]byte, 64<<10)
	for {
		var n uint32
		hr := comCall(stream, 3, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), uintptr(unsafe.Pointer(&n))) // ISequentialStream::Read
		out = append(out, buf[:n]...)
		if failed(hr) || n == 0 || hr == 1 { // S_FALSE: end of stream
			return out
		}
	}
}

// memStream returns a new IStream holding a copy of data.
func memStream(data []byte) uintptr {
	var p *byte
	if len(data) > 0 {
		p = &data[0]
	}
	s, _, _ := procSHCreateMemStream.Call(uintptr(unsafe.Pointer(p)), uintptr(len(data)))
	return s
}
