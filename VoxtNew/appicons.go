package main

// Apps' icons in the lists that name apps (App enhancement's members,
// its Add apps and Browsers dialogs, its preview). A row asks for its
// app's icon as it shows (appIcon); at the end of the page's build
// flushIcons sends the wanted ones to a goroutine a batch at a time: the
// disk cache (~/Library/Caches/VoxtNew/icons, a week fresh), then the
// helper's app.icons for the rest. Until an icon is there the row keeps
// its room empty; without one (no helper, a removed app, Linux) it shows
// the generic app icon.

import (
	"context"
	"encoding/json"
	"path/filepath"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appicons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
)

// iconBatch is how many icons one request asks for: a screenful of rows.
const iconBatch = 24

// appIconsState is the icons as the main thread has them.
type appIconsState struct {
	iconCache appicons.Cache // Dir "" in tests: nothing on disk
	iconBook  appicons.Book
	iconBmps  map[string]*ui.Bitmap // by appicons.Key
	iconBusy  bool                  // a batch on its way
}

// openIcons keeps the icons on disk in cacheDir/icons ("" keeps none).
func (a *app) openIcons(cacheDir string) {
	if cacheDir != "" {
		a.iconCache.Dir = filepath.Join(cacheDir, "icons")
	}
}

// appIconBitmap is ref's icon, nil while it is not there; want asks for
// it when it was not asked for yet (a row that shows).
func (a *app) appIconBitmap(ref proto.AppRef, want bool) (*ui.Bitmap, appicons.State) {
	k := appicons.Key(ref.BundleID)
	if b := a.iconBmps[k]; b != nil {
		return b, appicons.Ready
	}
	if want {
		a.iconBook.Want(ref)
	}
	return nil, a.iconBook.State(k)
}

// appIcon shows ref's icon in a square of size DIPs: the app's own, the
// generic app icon when it has none, nothing while it loads.
func (a *app) appIcon(c *ui.Context, ref proto.AppRef, size float32, want bool) ui.Element {
	t := c.Theme()
	bmp, st := a.appIconBitmap(ref, want)
	if bmp != nil {
		return ui.Image(c, bmp).Size(px(t, size), px(t, size)).Fit(ui.Contain).Shrink(0)
	}
	box := ui.Box(c).Size(px(t, size), px(t, size)).Center().Shrink(0)
	if st == appicons.Missing || st == appicons.Failed || ref.BundleID == "" {
		box.Children(func() {
			ui.Icon(c, icons.Widget2.Linear).Size(px(t, size*0.8), px(t, size*0.8)).TextColor(tok(t).Text2)
		})
	}
	return box
}

// flushIcons sends the next batch of wanted icons, unless one is on its
// way: its answer redraws, and the next build sends the next batch.
func (a *app) flushIcons() {
	if a.iconBusy || a.iconBook.Pending() == 0 {
		return
	}
	refs := a.iconBook.Take(iconBatch)
	a.iconBusy = true
	cache, sup := a.iconCache, a.host.sup
	var ask appicons.Ask
	if a.hstatus.Has(proto.MethodAppIcons) {
		ask = func(refs []proto.AppRef) ([]proto.AppIcon, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			raw, err := sup.Call(ctx, "", proto.MethodAppIcons, proto.AppIconsParams{Apps: refs, Size: proto.IconSize}, nil)
			if err != nil {
				return nil, err
			}
			var res proto.AppIconsResult
			if err := json.Unmarshal(raw, &res); err != nil {
				return nil, err
			}
			return res.Icons, nil
		}
	}
	a.async(func() {
		got := appicons.Fetch(cache, refs, proto.IconSize, ask)
		bmps := make([]*ui.Bitmap, len(got))
		for i, r := range got {
			if r.State == appicons.Ready {
				if b, err := ui.DecodeBitmap(r.PNG); err == nil {
					bmps[i] = b
				} else {
					got[i].State = appicons.Missing // not a picture
				}
			}
		}
		a.post(func() {
			a.iconBusy = false
			if a.iconBmps == nil {
				a.iconBmps = map[string]*ui.Bitmap{}
			}
			for i, r := range got {
				a.iconBook.Set(r.BundleID, r.State)
				if bmps[i] != nil {
					a.iconBmps[appicons.Key(r.BundleID)] = bmps[i]
				}
			}
		})
	})
}

// iconsHelperReady asks again for the icons that failed: a helper is
// ready now.
func (a *app) iconsHelperReady() { a.iconBook.Retry() }
