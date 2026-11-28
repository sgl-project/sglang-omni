//go:build darwin

package profile

import (
	"fmt"
	"sync"
	"syscall"

	"github.com/ebitengine/purego"
)

type dynamicLoaderInfo struct {
	Filename      *byte
	Base          uintptr
	SymbolName    *byte
	SymbolAddress uintptr
}

type signpostLibrary struct {
	dso      uintptr
	log      uintptr
	create   func(*byte, *byte) uintptr
	generate func(uintptr) uint64
	emit     func(uintptr, uintptr, uint8, uint64, *byte, *byte, *byte, uint32)
}

var nativeSignposts struct {
	once    sync.Once
	library signpostLibrary
	err     error
}

func loadSignposts() error {
	nativeSignposts.once.Do(func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				nativeSignposts.err = fmt.Errorf("os_signpost symbols: %v", recovered)
			}
		}()
		lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		if err != nil {
			nativeSignposts.err = err
			return
		}
		library := &nativeSignposts.library
		emitAddress, err := purego.Dlsym(lib, "_os_signpost_emit_with_name_impl")
		if err != nil {
			nativeSignposts.err = err
			return
		}
		var dladdr func(uintptr, *dynamicLoaderInfo) int32
		purego.RegisterLibFunc(&dladdr, lib, "dladdr")
		var image dynamicLoaderInfo
		if dladdr(emitAddress, &image) == 0 || image.Base == 0 {
			nativeSignposts.err = fmt.Errorf("dladdr could not find the signpost image")
			return
		}
		library.dso = image.Base
		purego.RegisterLibFunc(&library.create, lib, "os_log_create")
		purego.RegisterLibFunc(&library.generate, lib, "os_signpost_id_generate")
		purego.RegisterLibFunc(&library.emit, lib, "_os_signpost_emit_with_name_impl")
		subsystem, _ := syscall.BytePtrFromString("dev.hehehai.voxtnew.profiling")
		category, _ := syscall.BytePtrFromString("PointsOfInterest")
		library.log = library.create(subsystem, category)
		if library.log == 0 {
			nativeSignposts.err = fmt.Errorf("os_log_create returned nil")
		}
	})
	return nativeSignposts.err
}

// Signposts emits Points of Interest intervals without cgo.
type Signposts struct{}

func NewSignposts() *Signposts { return &Signposts{} }

func (s *Signposts) UnavailableReason() string {
	if err := loadSignposts(); err != nil {
		return err.Error()
	}
	return ""
}

func (s *Signposts) Begin(name string) func() {
	if loadSignposts() != nil {
		return func() {}
	}
	namePointer, err := syscall.BytePtrFromString(name)
	if err != nil {
		return func() {}
	}
	library := &nativeSignposts.library
	identifier := library.generate(library.log)
	if identifier == 0 {
		return func() {}
	}
	// The public begin/end calls are macros; their emit ABI takes an empty argument buffer.
	format, _ := syscall.BytePtrFromString("")
	buffer := [2]byte{}
	library.emit(library.dso, library.log, 1, identifier, namePointer, format, &buffer[0], uint32(len(buffer)))
	var once sync.Once
	return func() {
		once.Do(func() {
			library.emit(library.dso, library.log, 2, identifier, namePointer, format, &buffer[0], uint32(len(buffer)))
		})
	}
}
