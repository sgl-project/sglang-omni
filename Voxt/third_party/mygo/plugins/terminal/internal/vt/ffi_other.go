//go:build !darwin && !(linux && (amd64 || arm64)) && !(windows && (amd64 || arm64))

package vt

import "errors"

// Supported reports whether the platform can load the library.
const Supported = false

func open(string) (uintptr, error) { return 0, errors.ErrUnsupported }

func sym(uintptr, string) (uintptr, error) { return 0, errors.ErrUnsupported }

func call(uintptr, ...uintptr) uintptr { panic("vt: libghostty-vt is not supported on this platform") }

func registerFunc(any, uintptr) {}

func newCallback(any) uintptr { return 0 }
