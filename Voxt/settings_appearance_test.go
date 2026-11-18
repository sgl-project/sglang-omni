package main

import (
	"bytes"
	"context"
	"fmt"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestAppearancePalette(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/appearance")
	frames(tt)
	if !hasExact(tt, "Color") || hasExact(tt, "Hex") {
		t.Fatal("the color editor should start folded")
	}
	if !hasExact(tt, "Default") {
		t.Fatal("the original palette should say Default instead of showing a custom color")
	}
	colorRow, colorOK := tt.Find("Color")
	themeRow, themeOK := tt.Find("Theme")
	if !colorOK || !themeOK || math.Abs(float64(colorRow.X-themeRow.X)) > 1 {
		t.Fatalf("the Color and Theme rows should share a leading edge: %v and %v", colorRow, themeRow)
	}
	x, y := colorRow.X+160, colorRow.Y+colorRow.H/2
	before := tt.Image().RGBAAt(int(x), int(y))
	tt.Move(x, y)
	frames(tt)
	if after := tt.Image().RGBAAt(int(x), int(y)); after != before {
		t.Fatalf("hover changed the Color row background from %v to %v", before, after)
	}
	if hasExact(tt, "Hex") {
		t.Fatal("hover opened the color editor")
	}
	tt.ClickAt(colorRow.X+colorRow.W-12, y) // the swatch
	frames(tt)
	if !hasExact(tt, "Hex") {
		t.Fatal("clicking the color swatch did not open the editor")
	}
	tt.ClickAt(colorRow.X+5, y) // the title
	frames(tt)
	if hasExact(tt, "Hex") {
		t.Fatal("clicking the Color title did not close the editor")
	}
	tt.ClickAt(colorRow.X+5, y)
	frames(tt)
	click(t, tt, "Saturation and brightness")
	if got := a.settings.Get().Appearance.Palette; got == settings.DefaultPalette {
		t.Fatal("the color field did not change the palette")
	}
	click(t, tt, "Hex")
	tt.Key(ui.Cmd, ui.KeyA)
	tt.Type("#b25280")
	frames(tt)
	if ap := a.settings.Get().Appearance; ap.Palette != "#b25280" || ap.Theme != settings.ThemeSystem {
		t.Fatalf("saved custom color = %+v", ap)
	}
	if hasExact(tt, "Default") || !hasExact(tt, "Custom") {
		t.Fatal("the chosen custom color should say Custom instead of Default")
	}
	click(t, tt, "Dark")
	if ap := a.settings.Get().Appearance; ap.Palette != "#b25280" || ap.Theme != settings.ThemeDark {
		t.Fatalf("saved dark appearance = %+v", ap)
	}
	if got := tok(a.overlayTheme()).OverlayFace; got != paletteTokens("#b25280", true).OverlayFace {
		t.Fatalf("overlay face = %v", got)
	}
	tt.SetSize(windowMinWidth, windowMinHeight)
	a.updateSettings(func(s *settings.Settings) { s.Appearance.Scale = 1.25 })
	scrollToText(tt, "Reset color")
	click(t, tt, "Reset color")
	if ap := a.settings.Get().Appearance; ap.Palette != settings.DefaultPalette || ap.Scale != 1.25 {
		t.Fatalf("reset appearance at minimum window size = %+v", ap)
	}
	if !hasExact(tt, "Default") {
		t.Fatal("resetting the color did not restore the Default indicator")
	}
	a.navigate("/settings/general")
	a.navigate("/settings/appearance")
	frames(tt)
	if hasExact(tt, "Hex") {
		t.Fatal("the color editor should fold when the section reopens")
	}
}

func TestSystemAppearanceIsPreserved(t *testing.T) {
	a, _, tt := newTestApp(t)
	checkBackground := func(dark bool) {
		t.Helper()
		frames(tt)
		want := ui.Hex("#ffffff")
		if dark {
			want = ui.Hex("#151515")
		}
		got := tt.Image().RGBAAt(windowWidth-4, windowHeight-4)
		if got.R != want.R || got.G != want.G || got.B != want.B {
			t.Fatalf("background = %v, want %v (dark = %v)", got, want, dark)
		}
	}
	tt.SetDark(true)
	checkBackground(true)
	a.navigate("/settings/appearance")
	click(t, tt, "Dark")
	if got := a.settings.Get().Appearance.Theme; got != settings.ThemeDark {
		t.Fatalf("clicking Dark over a dark system theme saved %q", got)
	}
	tt.SetDark(false)
	a.settingsOpen = false
	checkBackground(true)
	a.navigate("/settings/appearance")
	frames(tt)
	if err := tt.ClickControl("System"); err != nil {
		t.Fatal(err)
	}
	if got := a.settings.Get().Appearance.Theme; got != settings.ThemeSystem {
		t.Fatalf("returning to System saved %q", got)
	}
	a.settingsOpen = false
	checkBackground(false)
	tt.SetDark(true)
	checkBackground(true)
}

// Adding a color picker must not recolor existing installations or a fresh setup.
func TestDefaultAppearanceRetainsOriginalPalette(t *testing.T) {
	upgraded, err := settings.Migrate([]byte(`{"version":4,"appearance":{"theme":"system","scale":1}}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, seed := range []string{"", settings.Defaults().Appearance.Palette, upgraded.Appearance.Palette} {
		for _, dark := range []bool{false, true} {
			background, surface, accent := "#ffffff", "#f7f7f7", "#0d9384"
			selection, recording := "#ebebeb", "#eb5230"
			if dark {
				background, surface, accent = "#151515", "#1c1c1c", "#2bb3a3"
				selection, recording = "#2b2b2b", "#ff6b47"
			}
			th := ui.Theme{Dark: dark}
			styleTheme(&th, seed)
			k := tok(&th)
			if th.Background != ui.Hex(background) || th.Accent != ui.Hex(accent) || k.Sidebar != ui.Hex(surface) || k.TranscriptFace != ui.Hex(surface) {
				t.Errorf("seed %q dark=%t: original background, accent or card color changed", seed, dark)
			}
			if k.NavSelected != ui.Hex(selection) || k.NavInk != k.Text || k.Waveform != ui.Hex(recording) {
				t.Errorf("seed %q dark=%t: original selection or recording color changed", seed, dark)
			}
		}
	}
}

func TestDefaultRecordingButtonColor(t *testing.T) {
	for _, dark := range []bool{false, true} {
		a, _, _ := newTestApp(t)
		draw := func(original bool) *ui.Tester {
			tt := ui.NewTester(func(c *ui.Context) {
				a.applyTheme(c)
				c.DisableAnimations()
				ui.Column(c).Children(func() {
					if original {
						recordButton(c, "Start")
					} else {
						a.tryRecordButton(c, "Start")
					}
				})
			}, 200, 100)
			tt.SetDark(dark)
			frames(tt)
			return tt
		}
		tt, original := draw(false), draw(true)
		r, ok := tt.Find("Start")
		if !ok {
			t.Fatal("recording button is missing")
		}
		for state := range 3 {
			for _, button := range []*ui.Tester{tt, original} {
				if state == 1 {
					button.Move(r.X+r.W/2, r.Y+r.H/2)
				} else if state == 2 {
					button.Press(r.X+r.W/2, r.Y+r.H/2)
				}
				frames(button)
			}
			if !bytes.Equal(tt.Image().Pix, original.Image().Pix) {
				t.Errorf("dark=%t state=%d: default recording button differs from its original colors", dark, state)
			}
		}
	}
}

func TestLateAppearancePreviewSavePreservesOtherSettings(t *testing.T) {
	a, _, _ := newTestApp(t)
	before := a.settings.Get()
	if err := a.settings.Update(context.Background(), func(s *settings.Settings) {
		s.Appearance.Palette = "#b25280"
	}); err != nil {
		t.Fatal(err)
	}
	paletteSaved := a.settings.Get()
	if err := a.settings.Update(context.Background(), func(s *settings.Settings) {
		s.Appearance.WindowTransparency = settings.MaxWindowTransparency
	}); err != nil {
		t.Fatal(err)
	}
	transparencySaved := a.settings.Get()
	a.updateSettings(func(s *settings.Settings) {
		s.Appearance.Scale = 1.25
		s.Privacy.KeepHistory = false
	})
	// Deliver the preview notifications after the other settings have applied.
	a.settingsChanged(paletteSaved, transparencySaved)
	a.settingsChanged(before, paletteSaved)
	if appearance := a.prefs.Appearance; appearance.Scale != 1.25 || appearance.Palette != "#b25280" ||
		appearance.WindowTransparency != settings.MaxWindowTransparency {
		t.Fatalf("late preview notifications changed appearance to %+v", appearance)
	}
	if a.scale != 3 || a.prefs.Privacy.KeepHistory || a.keepHistory {
		t.Fatalf("late preview notifications reverted scale or privacy: scale index %d, keep history %v", a.scale, a.keepHistory)
	}
}

func TestWindowTransparencySettings(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/appearance")
	initialOverlay := a.settings.Get().Overlay
	frames(tt)
	if hasExact(tt, "Window transparency") {
		t.Fatal("slider shown in opaque mode")
	}
	click(t, tt, "Transparent")
	click(t, tt, "Color")
	click(t, tt, "Hex")
	tt.Key(0, ui.KeyTab) // the slider follows the last picker control
	tt.Key(0, ui.KeyEnd)
	frames(tt)
	saved := a.settings.Get()
	if saved.Appearance.Theme != settings.ThemeTransparent || saved.Appearance.WindowTransparency != settings.MaxWindowTransparency || saved.Overlay != initialOverlay {
		t.Fatalf("transparency at maximum: %+v, overlay: %+v", saved.Appearance, saved.Overlay)
	}
	img := tt.Image()
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	if alpha := img.RGBAAt(w/2, h/2).A; alpha != 255 {
		t.Fatalf("the settings panel at maximum transparency has alpha %d", alpha)
	}
	if alpha := img.RGBAAt(w-4, h-4).A; alpha >= 255 {
		t.Fatalf("the page outside the settings panel has alpha %d", alpha)
	}
	tt.Key(0, ui.KeyHome)
	frames(tt)
	if got := a.settings.Get().Appearance.WindowTransparency; got != settings.MinWindowTransparency {
		t.Fatalf("transparency at minimum: %d%%", got)
	}
	click(t, tt, "Light")
	if a.settings.Get().Appearance.WindowTransparency != settings.MinWindowTransparency || hasExact(tt, "Window transparency") {
		t.Fatal("switching to light mode lost the transparency preference or kept the slider")
	}
}

func TestWindowTransparencyDragPreviewsBeforeSaving(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/appearance")
	click(t, tt, "Transparent")
	saved := a.settings.Get().Appearance.WindowTransparency
	label, labelOK := tt.Find("Window transparency")
	colorRow, rowOK := tt.Find("Color")
	if !labelOK || !rowOK {
		t.Fatal("appearance controls are missing")
	}
	y := label.Y + label.H/2
	right := colorRow.X + colorRow.W
	alpha := func() uint8 {
		img := tt.Image()
		return img.RGBAAt(img.Bounds().Dx()-4, img.Bounds().Dy()-4).A
	}
	initialAlpha := alpha()
	tt.Press(right-150, y)
	for _, x := range []float32{right - 130, right - 72, right - 25} {
		tt.Move(x, y)
		frames(tt)
		if got := a.settings.Get().Appearance.WindowTransparency; got != saved {
			t.Fatalf("saved %d%% while dragging, want the previous %d%%", got, saved)
		}
	}
	preview := int(a.windowTransparency)
	if preview <= saved || alpha() >= initialAlpha {
		t.Fatalf("drag did not preview the new value: %d%% to %d%%, alpha %d to %d", saved, preview, initialAlpha, alpha())
	}
	tt.Release(right-25, y)
	frames(tt)
	if got := a.settings.Get().Appearance.WindowTransparency; got != preview {
		t.Fatalf("saved %d%% after release, want %d%%", got, preview)
	}
	tt.Key(0, ui.KeyLeft)
	frames(tt)
	if got := a.settings.Get().Appearance.WindowTransparency; got != preview-1 {
		t.Fatalf("saved %d%% after keyboard step, want %d%%", got, preview-1)
	}
}

func TestPaletteTokens(t *testing.T) {
	for _, seed := range []string{settings.DefaultPalette, "#000000", "#ffffff", "#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff", "#ff00ff", "#b25280", "#002fa7"} {
		for _, dark := range []bool{false, true} {
			k := paletteTokens(seed, dark)
			th := ui.Theme{Dark: dark}
			styleTheme(&th, seed)
			if *tok(&th) != *k || *tok(inkTheme(&th)) != *k {
				t.Errorf("%s dark=%t: theme lost its custom color", seed, dark)
			}
			for _, bg := range []ui.Color{k.Bg, k.Sidebar, k.Subtle, k.Hover} {
				if contrastRatio(k.Text, bg) < 4.5 || contrastRatio(k.Text2, bg) < 4.5 {
					t.Errorf("%s dark=%t: unreadable text against %v", seed, dark, bg)
				}
			}
			if contrastRatio(k.AccentInk, k.Bg) < 4.5 || contrastRatio(k.OnAccent, k.Accent) < 3 {
				t.Errorf("%s dark=%t: unreadable accent", seed, dark)
			}
			if contrastRatio(k.Text, k.NavSelected) < 4.5 || contrastRatio(k.AccentInk, k.NavSelected) < 3 {
				t.Errorf("%s dark=%t: unreadable selected sidebar item", seed, dark)
			}
		}
	}
}

func TestPaletteContrast(t *testing.T) {
	for r := range 17 {
		for g := range 17 {
			for b := range 17 {
				seed := fmt.Sprintf("#%02x%02x%02x", min(r*16, 255), min(g*16, 255), min(b*16, 255))
				for _, dark := range []bool{false, true} {
					k := paletteTokens(seed, dark)
					for _, ink := range []ui.Color{k.Text, k.Text2, k.AccentInk} {
						if ratio := contrastRatio(ink, k.TranscriptFace); ratio < 4.5 {
							t.Fatalf("%s dark=%t: transcript contrast %.2f for %v on %v", seed, dark, ratio, ink, k.TranscriptFace)
						}
					}
					backdrop := ui.RGB(0, 0, 0)
					if dark {
						backdrop = ui.RGB(255, 255, 255)
					}
					for _, bg := range []ui.Color{k.Bg, k.Sidebar} {
						composited := bg.Alpha(1 - float32(settings.MaxWindowTransparency)/100).Over(backdrop)
						for _, ink := range []ui.Color{k.Text, k.Text2, k.Text3, k.AccentInk} {
							if ratio := contrastRatio(ink, composited); ratio < 4.5 {
								t.Fatalf("%s dark=%t: transparent contrast %.2f for %v on %v", seed, dark, ratio, ink, composited)
							}
						}
					}
				}
			}
		}
	}
}

func TestPalettePreviews(t *testing.T) {
	runSnapshots(t, "VOXTNEW_PALETTE_PREVIEWS", func(t *testing.T, out string) {
		a, _, tt := newTestApp(t)
		a.updateSettings(func(s *settings.Settings) { s.General.Language = "zh-Hans" })
		a.navigate("/settings/appearance")
		tt.SetSize(820, 585)
		tt.SetScale(2)
		save := func(name string) {
			t.Helper()
			frames(tt)
			file, err := os.Create(filepath.Join(out, "appearance-"+name+".png"))
			if err != nil {
				t.Fatal(err)
			}
			err = png.Encode(file, tt.Image())
			closeErr := file.Close()
			if err != nil || closeErr != nil {
				t.Fatalf("save %s: encode %v, close %v", file.Name(), err, closeErr)
			}
		}
		a.colorEditorOpen = true
		for _, preview := range []struct{ mode, color string }{
			{settings.ThemeLight, "#b25280"},
			{settings.ThemeDark, "#3979c5"},
			{settings.ThemeTransparent, "#9664c4"},
		} {
			a.updateSettings(func(s *settings.Settings) { s.Appearance.Theme, s.Appearance.Palette = preview.mode, preview.color })
			tt.SetDark(preview.mode != settings.ThemeLight)
			save(preview.mode)
		}
		a.colorEditorOpen = false
		save("collapsed")
	})
}
