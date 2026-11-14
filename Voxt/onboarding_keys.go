package main

// The setup guide's keyboard (onboarding_steps.go's try steps): the two
// bottom rows of a Mac keyboard with the shortcut's keys lit, so that
// "Right ⌥" names a key one can find (#3644: a user could not tell which
// physical key the guide meant).

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// keyboardKey is a key of the drawing: its id, its legend and its width
// in key units.
type keyboardKey struct {
	id, legend string
	units      float32
}

// keyboardRows are the drawing's rows, the shift row over the bottom one,
// as wide as each other (14.6 units); the keys between the two shifts and
// the arrows' place have no legend.
var keyboardRows = [][]keyboardKey{
	{{"shift-left", "⇧", 2.3}, {"", "", 1}, {"", "", 1}, {"", "", 1}, {"", "", 1}, {"", "", 1}, {"", "", 1}, {"", "", 1}, {"", "", 1}, {"", "", 1}, {"", "", 1}, {"shift-right", "⇧", 2.3}},
	{{"fn", "fn", 1}, {"control-left", "⌃", 1}, {"option-left", "⌥", 1}, {"command-left", "⌘", 1.25}, {"space", "", 5.1}, {"command-right", "⌘", 1.25}, {"option-right", "⌥", 1}, {"", "", 3}},
}

// keyboardLit is which keys of the drawing chord presses: a modifier on
// its side, on both for either side; fn; the space bar.
func keyboardLit(chord hotkey.Chord) map[string]bool {
	lit := map[string]bool{}
	for family, side := range chord.Mods {
		if family == proto.ModFn {
			lit["fn"] = true
			continue
		}
		if side != proto.SideRight {
			lit[family+"-left"] = true
		}
		if side != proto.SideLeft {
			lit[family+"-right"] = true
		}
	}
	if chord.Key == "Space" {
		lit["space"] = true
	}
	return lit
}

// obKeyboard draws where keys' keys are on a Mac keyboard; nothing for a
// mouse button or keys it cannot read. A key off the two rows (a letter)
// follows the drawing as a keycap of its own.
func (a *app) obKeyboard(c *ui.Context, keys string) {
	chord, err := hotkey.Parse(keys)
	if err != nil || chord.Mouse() {
		return
	}
	t := c.Theme()
	k := tok(t)
	lit := keyboardLit(chord)
	unit, gap := px(t, 22), px(t, 3)
	cap := func(legend string, w float32, on bool) {
		b := ui.Box(c).Size(w, unit).Shrink(0).Center().Radius(px(t, 4)).Border(1, k.Line)
		col := k.Text2
		if on {
			b.Background(k.Accent).Border(1, k.Accent)
			col = ui.RGB(255, 255, 255)
		}
		b.Children(func() {
			if legend != "" {
				ui.Text(c, legend).FontSize(px(t, fsCaption)).TextColor(col).SingleLine()
			}
		})
	}
	ui.Row(c.Key("ob-keyboard")).Gap(px(t, sp3)).AlignItems(ui.End).Wrap().Label(i18n.T("The shortcut's keys, lit on a Mac keyboard")).Children(func() {
		ui.Column(c).Gap(gap).Shrink(0).Children(func() {
			for _, row := range keyboardRows {
				ui.Row(c).Gap(gap).Children(func() {
					for _, key := range row {
						cap(key.legend, unit*key.units+gap*(key.units-1), key.id != "" && lit[key.id])
					}
				})
			}
		})
		if chord.Key != "" && chord.Key != "Space" {
			cap(chord.Key, max(unit, px(t, float32(8*len(chord.Key)+12))), true)
		}
	})
}
