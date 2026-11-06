package terminal

import (
	"strings"

	"github.com/egoist/mygo/ui"
)

// Font is the font of a terminal's text, with what Ghostty's font options
// set.
type Font struct {
	// Family is the font family, a comma-separated list as
	// ui.Element.Font takes it, to which ui.RegisterFont adds the app's
	// own fonts: "monospace", the system's, when empty.
	Family string
	// Size is the size in DIPs: 13 when zero.
	Size float32
	// Weight is how heavy text is, from 100 (thin) to 900 (black), of the
	// weights the family has: 400 when zero. Bold text is 300 heavier, and
	// at least 700.
	Weight int
	// Features turn OpenType features of the font on and off by tag, as
	// Ghostty's font-feature: "ss01" or "+ss01" turns one on, "calt=0" or
	// "-calt" off, and "cv05=2" picks an alternate. Coding fonts make
	// most of their ligatures with calt, and the others with liga.
	Features []string
	// LineHeight is the height of rows, as a multiple of the font's own
	// line height: 1 when zero, and 1.2 rows 20% taller, as Ghostty's
	// adjust-cell-height = 20%. Text is centered in its row.
	LineHeight float32
	// Thicken draws text with a thicker stroke, as Ghostty's font-thicken:
	// on macOS, with Core Text's font smoothing at its strongest, whatever
	// the text's color and the system's setting. It changes nothing on
	// Linux and Windows.
	Thicken bool
}

// fontKey is a Font at a scale, with the defaults, comparable.
type fontKey struct {
	family     string
	size       float32
	weight     int
	features   string
	lineHeight float32
	thicken    bool
	scale      float32
}

func (f *Font) key(scale float32) fontKey {
	k := fontKey{family: f.Family, size: f.Size, weight: f.Weight, features: featureList(f.Features), lineHeight: f.LineHeight, thicken: f.Thicken, scale: scale}
	if strings.TrimSpace(k.family) == "" {
		k.family = "monospace"
	}
	if k.size <= 0 {
		k.size = 13
	}
	if k.weight <= 0 {
		k.weight = 400
	}
	k.weight = min(max(k.weight, 100), 900)
	if k.lineHeight <= 0 {
		k.lineHeight = 1
	}
	return k
}

// variant returns the font of a variant of text: bold (1) and italic (2).
func (k fontKey) variant(i int) ui.Font {
	w := k.weight
	if i&1 != 0 {
		w = min(max(w+300, 700), 900)
	}
	return ui.Font{Family: k.family, Size: k.size, Weight: w, Italic: i&2 != 0, Features: k.features, Thicken: k.thicken}
}

// featureList converts features as Font.Features has them into the list
// ui.Font.Features takes.
func featureList(features []string) string {
	var out []string
	for _, f := range features {
		f = strings.TrimSpace(f)
		if tag, ok := strings.CutPrefix(f, "-"); ok {
			f = strings.TrimSpace(tag) + "=0"
		} else if tag, ok := strings.CutPrefix(f, "+"); ok {
			f = strings.TrimSpace(tag)
		}
		if f != "" {
			out = append(out, f)
		}
	}
	return strings.Join(out, ",")
}
