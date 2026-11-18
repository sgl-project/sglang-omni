package main

import (
	"os"

	"github.com/egoist/mygo"
	"github.com/hehehai/actnowkit/sounds"
)

// loadCueSounds reads resources/sounds (Contents/Resources/sounds); nil
// when the build has none.
func loadCueSounds() *sounds.Catalog {
	res, err := mygo.App.Path(mygo.PathResources)
	if err != nil {
		return nil
	}
	cat, err := sounds.Load(os.DirFS(res), "sounds")
	if err != nil {
		return nil
	}
	return cat
}
