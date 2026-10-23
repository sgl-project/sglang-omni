package webview2

import (
	"syscall"
	"unsafe"
)

var (
	advapi32             = syscall.NewLazyDLL("advapi32.dll") // a KnownDLL
	procRegOpenKeyExW    = advapi32.NewProc("RegOpenKeyExW")
	procRegQueryValueExW = advapi32.NewProc("RegQueryValueExW")
	procRegCloseKey      = advapi32.NewProc("RegCloseKey")
)

const (
	hkeyCurrentUser  = 0x80000001
	hkeyLocalMachine = 0x80000002
	keyQueryValue    = 0x0001
	keyWow6432Key    = 0x0200
)

// regString reads a string value from the 32-bit view of the registry,
// where EdgeUpdate records its clients.
func regString(machine bool, path, name string) string {
	root := uintptr(hkeyCurrentUser)
	if machine {
		root = hkeyLocalMachine
	}
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return ""
	}
	n, err := syscall.UTF16PtrFromString(name)
	if err != nil {
		return ""
	}
	var key uintptr
	if r, _, _ := procRegOpenKeyExW.Call(root, uintptr(unsafe.Pointer(p)), 0, keyQueryValue|keyWow6432Key, uintptr(unsafe.Pointer(&key))); r != 0 {
		return ""
	}
	defer procRegCloseKey.Call(key)
	var typ, size uint32
	if r, _, _ := procRegQueryValueExW.Call(key, uintptr(unsafe.Pointer(n)), 0, uintptr(unsafe.Pointer(&typ)), 0, uintptr(unsafe.Pointer(&size))); r != 0 || size == 0 {
		return ""
	}
	buf := make([]uint16, size/2+1)
	if r, _, _ := procRegQueryValueExW.Call(key, uintptr(unsafe.Pointer(n)), 0, uintptr(unsafe.Pointer(&typ)), uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size))); r != 0 {
		return ""
	}
	return syscall.UTF16ToString(buf)
}
