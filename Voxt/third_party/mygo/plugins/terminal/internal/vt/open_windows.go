//go:build windows && (amd64 || arm64)

package vt

import "syscall"

func open(path string) (uintptr, error) {
	h, err := syscall.LoadLibrary(path)
	return uintptr(h), err
}

func sym(h uintptr, name string) (uintptr, error) {
	return syscall.GetProcAddress(syscall.Handle(h), name)
}
