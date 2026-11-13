//go:build darwin

// Package darwin calls the macOS libraries actnowkit needs (libSystem's
// notify(3), Foundation/AppKit through the Objective-C runtime, the
// Security framework) with purego: no cgo.
package darwin

import (
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
)

var (
	libSystemOnce sync.Once
	libSystemErr  error

	notifyPost                   func(name string) uint32
	notifyRegisterFileDescriptor func(name string, fd *int32, flags int32, token *int32) uint32
	notifyCancel                 func(token int32) uint32

	setxattr    func(path, name string, value *byte, size uintptr, position uint32, options int32) int32
	getxattr    func(path, name string, value *byte, size uintptr, position uint32, options int32) int
	removexattr func(path, name string, options int32) int32
	procPidpath func(pid int32, buf *byte, size uint32) int32
	errnoLoc    func() *int32
)

func loadLibSystem() error {
	libSystemOnce.Do(func() {
		lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_GLOBAL|purego.RTLD_NOW)
		if err != nil {
			libSystemErr = err
			return
		}
		purego.RegisterLibFunc(&notifyPost, lib, "notify_post")
		purego.RegisterLibFunc(&notifyRegisterFileDescriptor, lib, "notify_register_file_descriptor")
		purego.RegisterLibFunc(&notifyCancel, lib, "notify_cancel")
		purego.RegisterLibFunc(&setxattr, lib, "setxattr")
		purego.RegisterLibFunc(&getxattr, lib, "getxattr")
		purego.RegisterLibFunc(&removexattr, lib, "removexattr")
		purego.RegisterLibFunc(&procPidpath, lib, "proc_pidpath")
		purego.RegisterLibFunc(&errnoLoc, lib, "__error")
	})
	return libSystemErr
}

// notifyReuse is NOTIFY_REUSE: register on an existing descriptor.
const notifyReuse = 1

// NotifyPost posts a Darwin notification.
func NotifyPost(name string) error {
	if err := loadLibSystem(); err != nil {
		return err
	}
	if st := notifyPost(name); st != 0 {
		return fmt.Errorf("notify_post %s: status %d", name, st)
	}
	return nil
}

// NotifyRegister registers names on one file descriptor (created by the
// first registration) and returns it and the token of each name. Every
// delivery writes the token, a big-endian int32, to the descriptor.
func NotifyRegister(names []string) (fd int, tokens map[int32]string, err error) {
	if err := loadLibSystem(); err != nil {
		return -1, nil, err
	}
	tokens = map[int32]string{}
	var cfd int32 = -1
	for i, n := range names {
		var tok int32
		flags := int32(0)
		if i > 0 {
			flags = notifyReuse
		}
		if st := notifyRegisterFileDescriptor(n, &cfd, flags, &tok); st != 0 {
			for t := range tokens {
				notifyCancel(t)
			}
			return -1, nil, fmt.Errorf("notify_register_file_descriptor %s: status %d", n, st)
		}
		tokens[tok] = n
	}
	return int(cfd), tokens, nil
}

// NotifyCancel cancels registrations; the descriptor closes with the last.
func NotifyCancel(tokens map[int32]string) {
	if loadLibSystem() != nil {
		return
	}
	for t := range tokens {
		notifyCancel(t)
	}
}
