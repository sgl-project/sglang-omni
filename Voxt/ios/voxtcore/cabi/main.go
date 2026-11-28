// Command cabi is voxtcore's C ABI, built as a static library
// (-buildmode=c-archive, scripts/build-core.sh) that the iOS app links.
// Its functions are declared in VoxtKit/Sources/CVoxtCore/include/voxtcore.h:
//
//	char *VoxtCoreCall(const char *id, const char *method, const char *params);
//	int   VoxtCoreCancel(const char *id);
//	void  VoxtCoreFree(char *reply);
//
// A call blocks until it is answered (the app calls it off the main
// thread) and returns a JSON reply the caller frees with VoxtCoreFree.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"unsafe"

	"github.com/sgl-project/sglang-omni/Voxt/ios/voxtcore"
)

var calls voxtcore.Calls

//export VoxtCoreCall
func VoxtCoreCall(id, method, params *C.char) *C.char {
	reply := calls.Call(C.GoString(id), C.GoString(method), []byte(C.GoString(params)))
	return C.CString(string(reply))
}

//export VoxtCoreCancel
func VoxtCoreCancel(id *C.char) C.int {
	if calls.Cancel(C.GoString(id)) {
		return 1
	}
	return 0
}

//export VoxtCoreFree
func VoxtCoreFree(reply *C.char) { C.free(unsafe.Pointer(reply)) }

func main() {}
