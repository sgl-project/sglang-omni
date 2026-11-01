package main

/*
#include <stdint.h>
#include <stdlib.h>
*/
import "C"
import "unsafe"

//export VoxtTailcatIdentity
func VoxtTailcatIdentity() *C.char {
	return C.CString(Identity())
}

//export VoxtTailcatStart
func VoxtTailcatStart(address *C.char, clientKey *C.char, remotePort C.int) *C.char {
	return C.CString(Start(cString(address), cString(clientKey), int(remotePort)))
}

//export VoxtTailcatStop
func VoxtTailcatStop(handle C.int64_t) C.int {
	return C.int(Stop(int64(handle)))
}

//export VoxtTailcatFree
func VoxtTailcatFree(p *C.char) {
	C.free(unsafe.Pointer(p))
}

func cString(p *C.char) string {
	if p == nil {
		return ""
	}
	return C.GoString(p)
}
