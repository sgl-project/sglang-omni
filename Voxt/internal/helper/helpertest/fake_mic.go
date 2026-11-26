package helpertest

import (
	"encoding/json"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

// serveMic serves mic.watch, and refuses any other method (serve's
// default). Options:
//
//	recording=IDS    mic.watch answers these bundle ids, joined by "/", as
//	                 the apps recording from an input (default none);
//	                 mic.changed comes from an events= script
//	hosts=NAMES      the processes of these names, joined by "/", run for
//	                 the first app recording (answered when the watch asks
//	                 for hosts, as the helper does)
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
	var p proto.MicWatchParams
	if json.Unmarshal(m.Params, &p) == nil && len(p.Hosts) > 0 {
		res.Hosts = []proto.MicHost{}
		for i, name := range strings.FieldsFunc(f.opts["hosts"], func(r rune) bool { return r == '/' }) {
			if len(res.Apps) > 0 {
				res.Hosts = append(res.Hosts, proto.MicHost{PID: 8000 + i, Name: name, App: res.Apps[0].PID})
			}
		}
	}
	f.result(m.ID, res)
}
