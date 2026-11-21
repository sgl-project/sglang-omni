//go:build windows && (amd64 || arm64)

package windows

import (
	"fmt"
	"syscall"
	"unsafe"
)

// Registry access for the settings of the current user.

const (
	keySetValue = 0x0002
	regSz       = 1
)

// regGet reads a string value, reporting whether it exists.
func regGet(root uintptr, path, name string) (string, bool) {
	data, ok := regGetBytes(root, path, name)
	if !ok {
		return "", false
	}
	buf := make([]uint16, len(data)/2)
	for i := range buf {
		buf[i] = uint16(data[2*i]) | uint16(data[2*i+1])<<8
	}
	return syscall.UTF16ToString(buf), true
}

// regGetBytes reads a value as it is stored, reporting whether it exists.
func regGetBytes(root uintptr, path, name string) ([]byte, bool) {
	var key uintptr
	if r, _, _ := procRegOpenKeyExW.Call(root, uintptr(unsafe.Pointer(u16(path))), 0, keyQueryValue, uintptr(unsafe.Pointer(&key))); r != 0 {
		return nil, false
	}
	defer procRegCloseKey.Call(key)
	var size uint32
	if r, _, _ := procRegQueryValueExW.Call(key, uintptr(unsafe.Pointer(u16(name))), 0, 0, 0, uintptr(unsafe.Pointer(&size))); r != 0 {
		return nil, false
	}
	data := make([]byte, size)
	if size > 0 {
		if r, _, _ := procRegQueryValueExW.Call(key, uintptr(unsafe.Pointer(u16(name))), 0, 0, uintptr(unsafe.Pointer(&data[0])), uintptr(unsafe.Pointer(&size))); r != 0 {
			return nil, false
		}
	}
	return data[:size], true
}

// regSet writes a string value, creating the key.
func regSet(root uintptr, path, name, value string) error {
	var key uintptr
	if r, _, _ := procRegCreateKeyExW.Call(root, uintptr(unsafe.Pointer(u16(path))), 0, 0, 0, keySetValue, 0, uintptr(unsafe.Pointer(&key)), 0); r != 0 {
		return fmt.Errorf("mygo: cannot create registry key %s: %w", path, syscall.Errno(r))
	}
	defer procRegCloseKey.Call(key)
	data := u16(value)
	size := (len(syscall.StringToUTF16(value))) * 2 // with the terminating NUL
	if r, _, _ := procRegSetValueExW.Call(key, uintptr(unsafe.Pointer(u16(name))), 0, regSz, uintptr(unsafe.Pointer(data)), uintptr(size)); r != 0 {
		return fmt.Errorf("mygo: cannot write registry value %s\\%s: %w", path, name, syscall.Errno(r))
	}
	return nil
}

// regDeleteValue removes a value; a missing one is not an error.
func regDeleteValue(root uintptr, path, name string) error {
	var key uintptr
	if r, _, _ := procRegOpenKeyExW.Call(root, uintptr(unsafe.Pointer(u16(path))), 0, keySetValue, uintptr(unsafe.Pointer(&key))); r != 0 {
		return nil
	}
	defer procRegCloseKey.Call(key)
	if r, _, _ := procRegDeleteValueW.Call(key, uintptr(unsafe.Pointer(u16(name)))); r != 0 && syscall.Errno(r) != syscall.ERROR_FILE_NOT_FOUND {
		return fmt.Errorf("mygo: cannot delete registry value %s\\%s: %w", path, name, syscall.Errno(r))
	}
	return nil
}

// regDeleteTree removes a key and everything in it.
func regDeleteTree(root uintptr, path string) error {
	if r, _, _ := procRegDeleteTreeW.Call(root, uintptr(unsafe.Pointer(u16(path)))); r != 0 && syscall.Errno(r) != syscall.ERROR_FILE_NOT_FOUND {
		return fmt.Errorf("mygo: cannot delete registry key %s: %w", path, syscall.Errno(r))
	}
	return nil
}
