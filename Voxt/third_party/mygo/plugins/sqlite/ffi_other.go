//go:build !(darwin || linux || windows) || !(amd64 || arm64)

package sqlite

import "errors"

const supported = false

func openLibrary(string) (uintptr, error)            { return 0, errors.ErrUnsupported }
func librarySymbol(uintptr, string) (uintptr, error) { return 0, errors.ErrUnsupported }
func closeLibrary(uintptr)                           {}
func call(uintptr, ...uintptr) uintptr               { panic("sqlite: unsupported platform") }
