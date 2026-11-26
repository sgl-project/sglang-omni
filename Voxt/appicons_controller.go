package main

// App-icon requests: visible rows mark references wanted, then flushIcons
// fetches one batch through the disk cache and helper. Bitmap decoding stays
// on the worker before results return through a.post.

import (
	"context"
	"encoding/json"
	"path/filepath"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appicons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

const iconBatch = 24

type appIconsState struct {
	iconCache appicons.Cache // Dir "" in tests: nothing on disk
	iconBook  appicons.Book
	iconBmps  map[string]*appIconBitmap // by appicons.Key
	iconBusy  bool                      // a batch on its way
}

func (a *app) openIcons(cacheDir string) {
	if cacheDir != "" {
		a.iconCache.Dir = filepath.Join(cacheDir, "icons")
	}
}

// appIconBitmap is ref's icon, nil while it is not there; want asks for
// it when it was not asked for yet (a row that shows).
func (a *app) appIconBitmap(ref proto.AppRef, want bool) (*appIconBitmap, appicons.State) {
	key := appicons.Key(ref.BundleID)
	if bitmap := a.iconBmps[key]; bitmap != nil {
		return bitmap, appicons.Ready
	}
	if want {
		a.iconBook.Want(ref)
	}
	return nil, a.iconBook.State(key)
}

func (a *app) appEnhancementIcon(ref appEnhancementIconRef, want bool) (*appIconBitmap, bool) {
	bitmap, state := a.appIconBitmap(proto.AppRef{BundleID: ref.BundleID, Path: ref.Path}, want)
	return bitmap, state == appicons.Missing || state == appicons.Failed || ref.BundleID == ""
}

// flushIcons sends the next batch of wanted icons, unless one is on its
// way: its answer redraws, and the next build sends the next batch.
func (a *app) flushIcons() {
	if a.iconBusy || a.iconBook.Pending() == 0 {
		return
	}
	refs := a.iconBook.Take(iconBatch)
	a.iconBusy = true
	cache, supervisor := a.iconCache, a.host.sup
	var ask appicons.Ask
	if a.hstatus.Has(proto.MethodAppIcons) {
		ask = func(refs []proto.AppRef) ([]proto.AppIcon, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			raw, err := supervisor.Call(ctx, "", proto.MethodAppIcons, proto.AppIconsParams{Apps: refs, Size: proto.IconSize}, nil)
			if err != nil {
				return nil, err
			}
			var result proto.AppIconsResult
			if err := json.Unmarshal(raw, &result); err != nil {
				return nil, err
			}
			return result.Icons, nil
		}
	}
	a.async(func() {
		got := appicons.Fetch(cache, refs, proto.IconSize, ask)
		bitmaps := make([]*appIconBitmap, len(got))
		for index, result := range got {
			if result.State == appicons.Ready {
				if bitmap, err := decodeAppIconBitmap(result.PNG); err == nil {
					bitmaps[index] = bitmap
				} else {
					got[index].State = appicons.Missing
				}
			}
		}
		a.post(func() {
			a.iconBusy = false
			if a.iconBmps == nil {
				a.iconBmps = map[string]*appIconBitmap{}
			}
			for index, result := range got {
				a.iconBook.Set(result.BundleID, result.State)
				if bitmaps[index] != nil {
					a.iconBmps[appicons.Key(result.BundleID)] = bitmaps[index]
				}
			}
		})
	})
}

func (a *app) iconsHelperReady() { a.iconBook.Retry() }
