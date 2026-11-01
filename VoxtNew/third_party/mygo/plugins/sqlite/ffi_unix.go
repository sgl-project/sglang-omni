//go:build (darwin || linux) && (amd64 || arm64)

package sqlite

import (
	"runtime"

	"github.com/ebitengine/purego"
)

const supported = true

func openLibrary(path string) (uintptr, error) {
	flags := purego.RTLD_NOW | purego.RTLD_LOCAL
	if runtime.GOOS == "linux" {
		// WebKit loads the system SQLite globally. RTLD_LOCAL keeps our
		// symbols private, but its internal calls can still bind to that
		// other SQLite and mix incompatible global state. glibc's
		// RTLD_DEEPBIND makes this library resolve its own symbols first.
		const rtldDeepBind = 0x00008
		flags |= rtldDeepBind
	}
	return purego.Dlopen(path, flags)
}
func librarySymbol(h uintptr, name string) (uintptr, error) { return purego.Dlsym(h, name) }
func closeLibrary(h uintptr)                                { _ = purego.Dlclose(h) }

//go:uintptrescapes
func call(fn uintptr, args ...uintptr) uintptr {
	r, _, _ := purego.SyscallN(fn, args...)
	return r
}
