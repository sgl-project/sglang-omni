// Package icons holds the Solar icons VoxtNew shows, each in its
// Linear variant (everywhere) and its Bold one (the sidebar's chosen
// page, active toggles). solar_gen.go is generated from the
// @iconify-json/solar package at a pinned version:
//
//	go generate ./internal/icons
//
// To use another icon, add its base name ("history", without -linear
// or -bold) to names.txt and generate again. Solar is by 480 Design,
// under CC BY 4.0; the attribution is in NOTICE and Settings → About.
package icons

//go:generate go run ./gen

import "github.com/egoist/mygo/ui"

// Icon is a Solar icon in its two variants.
type Icon struct {
	Name         string // Solar's base name, "soundwave"
	Linear, Bold *ui.SVG
}

// Get is the icon's Bold variant when bold is true, its Linear one
// otherwise.
func (i Icon) Get(bold bool) *ui.SVG {
	if bold {
		return i.Bold
	}
	return i.Linear
}

// solar parses an icon's two bodies (24×24 SVG content).
func solar(name, linear, bold string) Icon {
	return Icon{Name: name, Linear: parse(linear), Bold: parse(bold)}
}

func parse(body string) *ui.SVG {
	return ui.MustParseSVG([]byte(`<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">` + body + `</svg>`))
}
