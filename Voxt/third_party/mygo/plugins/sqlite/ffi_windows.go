//go:build windows && (amd64 || arm64)

package sqlite

import (
	"github.com/ebitengine/purego"
	"syscall"
)

const supported = true

func openLibrary(path string) (uintptr, error) {
	h, err := syscall.LoadLibrary(path)
	return uintptr(h), err
}
func librarySymbol(h uintptr, name string) (uintptr, error) {
	return syscall.GetProcAddress(syscall.Handle(h), name)
}
func closeLibrary(h uintptr) { _ = syscall.FreeLibrary(syscall.Handle(h)) }

//go:uintptrescapes
func call(fn uintptr, args ...uintptr) uintptr {
	r, _, _ := purego.SyscallN(fn, args...)
	return r
}
