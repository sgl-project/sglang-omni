package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appicons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// snapshotIcons makes the fake supervisor answer app.icons with the fake
// helper's icons, none for com.example.* (a removed app), and counts the
// apps asked for. With a dir it answers the PNGs there (<bundle id>.png,
// as TestMacHelperIcons saves real ones), none for the others: the
// snapshots pass VOXTNEW_SNAPSHOT_ICONS, the tests of the icons' states
// "" (with it set to a dir without their apps they saw none).
func snapshotIcons(fs *fakeSupervisor, dir string) *[][]string {
	var asked [][]string
	fs.mu.Lock()
	defer fs.mu.Unlock()
	fs.handlers[proto.MethodAppIcons] = func(raw json.RawMessage) (any, error) {
		var p proto.AppIconsParams
		json.Unmarshal(raw, &p)
		var ids []string
		res := proto.AppIconsResult{Icons: []proto.AppIcon{}}
		for _, r := range p.Apps {
			ids = append(ids, r.BundleID)
			ic := proto.AppIcon{BundleID: r.BundleID}
			switch {
			case dir != "":
				ic.PNG, _ = os.ReadFile(filepath.Join(dir, r.BundleID+".png"))
			case !strings.HasPrefix(r.BundleID, "com.example."):
				ic.PNG = helpertest.Icon(r.BundleID, p.Size)
			}
			res.Icons = append(res.Icons, ic)
		}
		asked = append(asked, ids)
		return res, nil
	}
	return &asked
}

// manyApps makes app.list answer n installed apps besides the fake's.
func manyApps(fs *fakeSupervisor, n int) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	fs.handlers[proto.MethodAppList] = func(json.RawMessage) (any, error) {
		var apps []proto.AppInfo
		for i := range n {
			id := "org.many.app" + string(rune('a'+i%26)) + strings.Repeat("x", i/26)
			apps = append(apps, proto.AppInfo{BundleID: id, Name: "App " + id[12:], Path: "/Applications/" + id + ".app"})
		}
		return proto.AppListResult{Apps: apps}, nil
	}
}

// TestAppIcons: the rows that show ask for their icons, a batch at a
// time, and show them; a removed app shows the generic icon.
func TestAppIcons(t *testing.T) {
	a, fs, tt := aeTester(t)
	asked := snapshotIcons(fs, "")
	tt.SetSize(windowWidth, windowHeight)
	r := sampleRules()
	r.Groups[0].Apps = append(r.Groups[0].Apps, appctx.App{BundleID: "com.example.removed", Name: "Removed"})
	a.setRules(r)
	a.aePage.sel = "g-chat"
	frames(tt)
	if b, st := a.appIconBitmap(proto.AppRef{BundleID: "com.tinyspeck.slackmacgap"}, false); b == nil || st != appicons.Ready {
		t.Fatalf("slack: %v %v; asked %v", b, st, *asked)
	}
	if b, st := a.appIconBitmap(proto.AppRef{BundleID: "com.example.removed"}, false); b != nil || st != appicons.Missing {
		t.Errorf("removed: %v %v", b, st)
	}
	// A long list: the first rows at once, the rest only as they show,
	// never more than a batch a request.
	manyApps(fs, 200)
	*asked = nil
	click(t, tt, "Add apps…")
	frames(tt)
	count := func() (n int) {
		for _, ids := range *asked {
			if len(ids) > iconBatch {
				t.Errorf("a batch of %d", len(ids))
			}
			n += len(ids)
		}
		return n
	}
	before := count()
	if before == 0 || before > 2*iconBatch {
		t.Errorf("asked for %d icons of 200 rows: %v", before, *asked)
	}
	for range 10 {
		tt.Scroll(400, 300, 0, 300)
		frames(tt)
	}
	n := count()
	t.Logf("icons asked for: %d at first, %d after scrolling, of 200 rows", before, n)
	if n <= before || n >= 200 {
		t.Errorf("after scrolling: %d icons (before %d)", n, before)
	}
}

// TestAppIconsNoHelper: without a helper the rows show the generic icon,
// and ask again when one is ready; the icons on disk show without one.
func TestAppIconsNoHelper(t *testing.T) {
	a, fs, tt := aeTester(t)
	a.iconCache.Dir = t.TempDir()
	a.iconCache.Write("com.tinyspeck.slackmacgap", proto.IconSize, helpertest.Icon("com.tinyspeck.slackmacgap", 64))
	fs.Stop()
	frames(tt)
	asked := snapshotIcons(fs, "")
	a.setRules(sampleRules())
	a.aePage.sel = "g-chat"
	frames(tt)
	if b, st := a.appIconBitmap(proto.AppRef{BundleID: "com.tinyspeck.slackmacgap"}, false); b == nil || st != appicons.Ready {
		t.Errorf("slack from disk: %v %v", b, st)
	}
	if _, st := a.appIconBitmap(proto.AppRef{BundleID: "com.tencent.xinWeChat"}, false); st != appicons.Failed {
		t.Errorf("wechat without a helper: %v", st)
	}
	if len(*asked) != 0 {
		t.Errorf("asked a stopped helper: %v", *asked)
	}
	fs.Start(context.Background())
	frames(tt)
	if b, st := a.appIconBitmap(proto.AppRef{BundleID: "com.tencent.xinWeChat"}, false); b == nil || st != appicons.Ready {
		t.Errorf("wechat with a helper again: %v %v (asked %v)", b, st, *asked)
	}
	if fs.Status().State != helper.Ready {
		t.Error("not ready")
	}
}
