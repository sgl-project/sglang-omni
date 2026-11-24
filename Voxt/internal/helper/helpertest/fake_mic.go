package helpertest

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

// serveMic serves mic.watch, and refuses any other method (serve's
// default). Options:
//
//	recording=IDS    mic.watch answers these bundle ids, joined by "/", as
//	                 the apps recording from an input (default none);
//	                 mic.changed comes from an events= script
//
// With no-output (an older helper) mic.watch is method_not_found too.
func (f *fake) serveMic(m proto.Message) {
	if m.Method != proto.MethodMicWatch || f.has("no-output") {
		f.fail(m.ID, proto.CodeMethodNotFound, "Unknown method %s", m.Method)
		return
	}
	res := proto.MicUsers{Apps: []proto.App{}}
	for i, id := range strings.FieldsFunc(f.opts["recording"], func(r rune) bool { return r == '/' }) {
		res.Apps = append(res.Apps, proto.App{PID: 7000 + i, BundleID: id})
	}
	f.result(m.ID, res)
}
