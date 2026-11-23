package helpertest

import (
	"bufio"
	"io"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

// TestHotkeyConfigureKeepsTheNewest: hotkey.configure requests can
// complete out of order; an older one landing last changes nothing, so
// the helper's final bindings are the newest configuration's.
func TestHotkeyConfigureKeepsTheNewest(t *testing.T) {
	f := &fake{opts: map[string]string{}, out: bufio.NewWriter(io.Discard), trusted: true}
	bind := func(id string) proto.HotkeyBinding {
		return proto.HotkeyBinding{ID: id, Modifiers: map[string]string{proto.ModFn: proto.SideAny}, Behavior: proto.BehaviorTap}
	}
	configure := func(p proto.HotkeyConfigureParams) {
		m, err := proto.NewRequest("1", proto.MethodHotkeyConfigure, p)
		if err != nil {
			t.Fatal(err)
		}
		f.hotkeys(m)
	}
	recording := proto.HotkeyConfigureParams{Bindings: []proto.HotkeyBinding{bind("transcribe/0"), bind("finish")}, Generation: 1}
	stopped := proto.HotkeyConfigureParams{Bindings: []proto.HotkeyBinding{bind("transcribe/0")}, Generation: 2}
	configure(stopped)
	configure(recording) // decided first, lands last
	if f.hkGen != 2 || len(f.hkBindings) != 1 || f.hkBindings[0].ID != "transcribe/0" || f.bindings != 1 {
		t.Errorf("an older configuration landed last and won: gen %d, bindings %+v", f.hkGen, f.hkBindings)
	}
	// In order, each applies.
	configure(recording)
	configure(stopped)
	if f.hkGen != 2 || len(f.hkBindings) != 1 {
		t.Errorf("in-order configures: gen %d, bindings %+v", f.hkGen, f.hkBindings)
	}
	// Generation-less configurations (an old app) apply over one another.
	old := &fake{opts: map[string]string{}, out: bufio.NewWriter(io.Discard), trusted: true}
	m, err := proto.NewRequest("1", proto.MethodHotkeyConfigure, proto.HotkeyConfigureParams{Bindings: []proto.HotkeyBinding{bind("finish")}})
	if err != nil {
		t.Fatal(err)
	}
	old.hotkeys(m)
	m, err = proto.NewRequest("2", proto.MethodHotkeyConfigure, proto.HotkeyConfigureParams{})
	if err != nil {
		t.Fatal(err)
	}
	old.hotkeys(m)
	if len(old.hkBindings) != 0 {
		t.Errorf("generation-less configure: bindings %+v", old.hkBindings)
	}
}
