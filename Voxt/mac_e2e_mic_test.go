package main

import (
	"os"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meetdetect"
)

// TestMacHelperMicWatch asks the real helper which processes record from
// an input (Core Audio's process list): it needs no permission, and the
// helper and this process are never among them.
func TestMacHelperMicWatch(t *testing.T) {
	h := startMacHelper(t, macE2E(t))
	hello := h.Status().Hello
	if !h.Status().Has(proto.MethodMicWatch) {
		t.Fatalf("the helper does not list mic.watch: %v", hello.Methods)
	}
	mic := macCall[proto.MicUsers](t, h, proto.MethodMicWatch, proto.MicWatchParams{On: true, Hosts: meetdetect.HostNames()})
	t.Logf("mic.watch: %d process(es) recording, meeting processes %+v", len(mic.Apps), mic.Hosts)
	if mic.Apps == nil || mic.Hosts == nil {
		t.Errorf("mic.watch answered apps %v, hosts %v: want both lists", mic.Apps, mic.Hosts)
	}
	for _, u := range mic.Apps {
		if u.PID <= 0 || u.PID == hello.PID || u.PID == os.Getpid() {
			t.Errorf("mic.watch names pid %d (the helper and the app are left out): %+v", u.PID, u)
		}
	}
	macCall[proto.MicUsers](t, h, proto.MethodMicWatch, proto.MicWatchParams{On: false})
}
