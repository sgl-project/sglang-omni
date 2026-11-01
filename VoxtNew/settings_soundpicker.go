package main

// The ActNow family's sound picker (v2, 10-20): a pop-up button with the
// chosen sound that opens a compact panel: on top a search field (names,
// descriptions, tags), the type menu (All, Suggested, each group of the
// family's sounds, macOS's system sounds) and the volume button (a menu of
// VolumeSteps, the current one checked; its icon muted, low or high);
// under them one line per sound: ▶ (Solar's turntable-music-note
// line-duotone, in the accent while it plays, a click again stops it),
// the name and its type in a lighter tone, ✓ on the chosen one. Clicking
// a name chooses and plays it; the panel stays open while the user tries
// sounds. The panel is soundPanelW wide; the type and volume buttons share
// one face. The same picker in every family app, in its own tokens.

import (
	"fmt"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
)

// soundPickerSpec is what a sound picker shows and does.
type soundPickerSpec struct {
	Name    string  // the button's accessible name ("Pick Sound")
	Width   float32 // the button's width in DIPs (0: soundPickerW)
	Event   sounds.Event
	Choice  string // the event's choice now
	Off     bool   // sounds are off: the button dims (the panel still opens)
	Volume  int    // percent
	Catalog *sounds.Catalog
	System  []string                   // macOS's system sounds
	Label   func(choice string) string // a choice's name
	Choose  func(choice string)
	Preview func(choice string)
	// Stop stops the preview (nil: none); Playing is the choice a preview
	// plays and how long it has left ("" none; nil: ▶ shows no state).
	Stop    func()
	Playing func() (choice string, left time.Duration)
	// SetVolume sets the volume (nil: the panel has no volume button).
	SetVolume func(percent int)
}

// soundPickerState is a picker's panel: open, the type (pickSuggested,
// pickAll, pickSystem or a group's id) and the search text.
type soundPickerState struct {
	open  bool
	view  string
	query string
}

const (
	pickSuggested = ""
	pickAll       = "*all"
	pickSystem    = "*macos"
)

// soundPicker is the button and its panel.
func soundPicker(c *ui.Context, sp soundPickerSpec) {
	t := c.Theme()
	k := tok(t)
	label := sp.Label(sp.Choice)
	w := sp.Width
	if w == 0 {
		w = soundPickerW
	}
	b := ui.ButtonBase(c.Key("btn")).Width(px(t, w)).Height(px(t, 26)).Padding(0, px(t, 8)).Gap(px(t, 6)).AlignItems(ui.Center).
		Radius(px(t, radiusControl)).Background(k.Hover).Label(sp.Name).Value(label).Tooltip(label).Shrink(0)
	st := ui.Local(b, "picker", func() soundPickerState { return soundPickerState{} })
	if b.Hovered() {
		b.Background(k.Pressed)
	}
	if b.Clicked() {
		st.open = !st.open
	}
	b.Children(func() {
		v := ui.Text(c, label).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
		mark := k.Text2
		if sp.Off {
			v.TextColor(k.Text3)
			mark = k.Text3
		}
		ui.Icon(c, icons.AltArrowDown.Linear).Size(px(t, 13), px(t, 13)).TextColor(mark)
	})
	ui.PopoverBase(c.Key("panel"), b, &st.open, func(panel ui.Element) {
		panel.Width(px(t, soundPanelW)).Padding(px(t, 8)).Radius(px(t, radiusCard)).Background(k.Bg).Border(1, k.Line).
			Shadow(0, 8, 24, 0, ui.RGBA(0, 0, 0, 0.18)).Label(i18n.T("Sounds")).Children(func() {
			ui.Column(c).Gap(px(t, 6)).Children(func() { soundPanel(c, sp, st) })
		})
	})
}

// soundViewName names a type of the menu.
func soundViewName(sp soundPickerSpec, view, lang string) string {
	switch view {
	case pickAll:
		return i18n.T("All")
	case pickSuggested:
		return i18n.T("Suggested")
	case pickSystem:
		return i18n.T("macOS")
	}
	return sp.Catalog.GroupName(view, lang)
}

// soundPanel is the panel's content: the top line, then the list.
func soundPanel(c *ui.Context, sp soundPickerSpec, st *soundPickerState) {
	t := c.Theme()
	k := tok(t)
	lang := i18n.Locale()
	cat := sp.Catalog
	if cat == nil && st.view != pickSystem {
		st.view = pickSystem // a build without the family's sounds
	}
	ui.Row(c.Key("top")).Gap(px(t, 6)).AlignItems(ui.Center).Children(func() {
		// The field keeps its accessible name; its placeholder is short, so
		// that the top line fits soundPanelW.
		searchField(c, "sound-q", &st.query, i18n.T("Search sounds"), 0).Placeholder(i18n.T("Search")).AutoFocus() // type to search at once
		soundTypeMenu(c, sp, st, lang)
		if sp.SetVolume != nil {
			soundVolumeMenu(c, sp)
		}
	})
	ui.Scroll(c.Key("list")).MaxHeight(px(t, 300)).Children(func() {
		ui.Column(c).Gap(px(t, 1)).Children(func() {
			rows := 0
			row := func(choice, name, kind string) {
				rows++
				soundRow(c, sp, choice, name, kind)
			}
			q := strings.TrimSpace(st.query)
			switch {
			case st.view == pickSystem:
				lq := strings.ToLower(q)
				for _, n := range sp.System {
					if lq == "" || strings.Contains(strings.ToLower(n), lq) {
						row(sounds.System(n), n, i18n.T("macOS"))
					}
				}
			default:
				var list []sounds.Sound
				switch {
				case st.view == pickSuggested && q == "":
					row(sounds.None, i18n.T("None"), "")
					list = cat.Suggested(sp.Event)
				case st.view == pickAll || st.view == pickSuggested:
					if q == "" {
						row(sounds.None, i18n.T("None"), "")
					}
					list = cat.Filter(sounds.Filter{Query: q})
				default:
					list = cat.Filter(sounds.Filter{Group: st.view, Query: q})
				}
				for _, s := range list {
					row(s.ID, s.Name.In(lang), cat.GroupName(s.Group, lang))
				}
			}
			if rows == 0 {
				ui.Text(c, i18n.T("No sounds match")).FontSize(px(t, fsCaption)).TextColor(k.Text3).Padding(px(t, 10), px(t, 8))
			}
		})
	})
}

// soundMenuButton is the face of the panel's two menu buttons, the type
// and the volume: the trigger's look (its fill, corners, height and
// padding), darker under the pointer and while pressed.
func soundMenuButton(c *ui.Context, key, label, value string) ui.Element {
	t := c.Theme()
	k := tok(t)
	b := ui.ButtonBase(c.Key(key)).Height(px(t, 28)).Padding(0, px(t, 8)).Gap(px(t, 4)).AlignItems(ui.Center).Justify(ui.Center).
		Radius(px(t, radiusControl)).Background(k.Hover).Label(label).Value(value).Tooltip(label + " " + value).Shrink(0)
	if b.Hovered() || b.Pressed() {
		b.Background(k.Pressed)
	}
	return b
}

// soundTypeMenu is the type menu: All, Suggested, each group, macOS; as
// wide as its value.
func soundTypeMenu(c *ui.Context, sp soundPickerSpec, st *soundPickerState, lang string) {
	t := c.Theme()
	k := tok(t)
	cur := soundViewName(sp, st.view, lang)
	b := soundMenuButton(c, "type", i18n.T("Type"), cur).MaxWidth(px(t, 112))
	b.Children(func() {
		ui.Text(c, cur).FontSize(px(t, fsCaption)).SingleLine().Ellipsis("…").MinWidth(0)
		ui.Icon(c, icons.AltArrowDown.Linear).Size(px(t, 12), px(t, 12)).TextColor(k.Text2)
	})
	b.Menu(func(m *ui.Menu) {
		var views []string
		if sp.Catalog != nil {
			views = append(views, pickAll, pickSuggested)
			for _, g := range sp.Catalog.Groups {
				views = append(views, g.ID)
			}
		}
		if len(sp.System) > 0 {
			views = append(views, pickSystem)
		}
		for i, v := range views {
			if i == 2 || v == pickSystem {
				m.Separator()
			}
			if m.Item(soundViewName(sp, v, lang)).Checked(v == st.view).Chosen() {
				st.view = v
			}
		}
	})
}

// soundVolumeMenu is the volume button, in the type menu's face: its icon
// says the level, its menu lists VolumeSteps loudest first, the current
// one checked.
func soundVolumeMenu(c *ui.Context, sp soundPickerSpec) {
	t := c.Theme()
	k := tok(t)
	cur := sounds.NearestVolumeStep(sp.Volume)
	ic := icons.VolumeLoud
	switch sounds.VolumeLevel(cur) {
	case 0:
		ic = icons.VolumeCross
	case 1:
		ic = icons.VolumeSmall
	}
	b := soundMenuButton(c, "volume", i18n.T("Volume"), fmt.Sprintf("%d%%", cur))
	b.Children(func() {
		ui.Icon(c, ic.Linear).Size(px(t, 16), px(t, 16)).TextColor(k.Text2)
	})
	b.Menu(func(m *ui.Menu) {
		for i := len(sounds.VolumeSteps) - 1; i >= 0; i-- {
			v := sounds.VolumeSteps[i]
			label := fmt.Sprintf("%d%%", v)
			if v == 0 {
				label = i18n.T("Mute")
			}
			if m.Item(label).Checked(v == cur).Chosen() {
				sp.SetVolume(v)
			}
		}
	})
}

// soundRow is one choice on one line: ▶ plays it (stops it while it
// plays); its name chooses and plays it; its type follows in a lighter
// tone; ✓ marks the chosen one.
func soundRow(c *ui.Context, sp soundPickerSpec, choice, name, kind string) {
	t := c.Theme()
	k := tok(t)
	chosen := choice == sp.Choice
	playing := false
	if sp.Playing != nil {
		if ch, left := sp.Playing(); ch != "" {
			playing = ch == choice
			c.After(left) // redraw when it ends
		}
	}
	ui.Row(c.Key("row-" + choice)).Gap(px(t, 2)).AlignItems(ui.Center).Children(func() {
		if choice != sounds.None {
			tip := i18n.T("Play %s", name)
			if playing {
				tip = i18n.T("Stop %s", name)
			}
			p := ghostBase(c.Key("play")).Size(px(t, 26), px(t, 26)).Center().Shrink(0).Label(tip).Tooltip(tip).Selected(playing)
			col := k.Text2
			switch {
			case playing:
				p.Background(k.AccentSoft)
				col = k.AccentInk
			case p.Hovered():
				col = k.Text
			}
			p.Children(func() {
				ui.Icon(c, icons.TurntableMusicNoteLineDuotone.Linear).Size(px(t, 15), px(t, 15)).TextColor(col)
			})
			if p.Clicked() {
				if playing && sp.Stop != nil {
					sp.Stop()
				} else {
					sp.Preview(choice)
				}
			}
		} else {
			ui.Box(c).Size(px(t, 26), px(t, 26)).Shrink(0)
		}
		r := ui.ButtonBase(c.Key("pick")).Grow(1).MinWidth(0).Height(px(t, 26)).Padding(0, px(t, 8)).Gap(px(t, 6)).AlignItems(ui.Center).
			Radius(px(t, radiusControl)).Label(name).Selected(chosen)
		switch {
		case chosen:
			r.Background(k.Pressed)
		case r.Hovered():
			r.Background(k.Hover)
		}
		if r.Clicked() {
			sp.Choose(choice)
			if choice != sounds.None {
				sp.Preview(choice)
			}
		}
		r.Children(func() {
			ui.Text(c, name).TextColor(k.Text).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
			if kind != "" {
				ui.Text(c, kind).FontSize(px(t, fsMicro)).TextColor(k.Text3).SingleLine().Shrink(0)
			}
			ui.Box(c).Grow(1)
			if chosen {
				ui.Icon(c, icons.Check.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.AccentInk)
			}
		})
	})
}

// soundPanelW is the panel's width in DIPs, the same in every family app,
// measured (10-20): the widest top line is the search field at its least
// (120, with its short placeholder), the type menu showing "Suggested"
// (97) and the volume button (32), with the gaps, the padding and the
// border: 279. The longest sound with its type ("Downlink Progress",
// "大功告成 进度") needs 194.
const soundPanelW = 280

// soundPickerW is the picker button's width.
const soundPickerW = 176
