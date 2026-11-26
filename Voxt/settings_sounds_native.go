package main

import (
	"io/fs"
	"os"

	"github.com/egoist/mygo"
	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/cues"
)

// loadCueSounds is Voxt's own cues (internal/cues) with the family's
// sounds of resources/sounds (Contents/Resources/sounds) when the build
// has them.
func loadCueSounds() *sounds.Catalog {
	var family fs.FS
	if res, err := mygo.App.Path(mygo.PathResources); err == nil {
		family = os.DirFS(res)
	}
	cat, err := cues.Load(family, "sounds")
	if err != nil {
		return nil
	}
	return cat
}
