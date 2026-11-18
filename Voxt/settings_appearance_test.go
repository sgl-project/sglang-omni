package main

import (
	"context"
	"fmt"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestAppearancePalette(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/appearance")
	frames(tt)
	if !hasExact(tt, "Color") || hasExact(tt, "Hex") {
		t.Fatal("the color editor should start folded")
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
	tt.ClickAt(colorRow.X+colorRow.W-12, y)
	frames(tt)
	if !hasExact(tt, "Hex") {
		t.Fatal("clicking the color swatch did not open the editor")
	}
	tt.ClickAt(colorRow.X+5, y)
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
	if appearance := a.settings.Get().Appearance; appearance.Palette != "#b25280" || appearance.Theme != settings.ThemeSystem {
		t.Fatalf("saved custom color = %+v", appearance)
	}
	click(t, tt, "Dark")
	if appearance := a.settings.Get().Appearance; appearance.Palette != "#b25280" || appearance.Theme != settings.ThemeDark {
		t.Fatalf("saved dark appearance = %+v", appearance)
	}
	if got := tok(overlayTheme(a.prefs.Appearance.Palette)).OverlayFace; got != paletteTokens("#b25280", true).OverlayFace {
		t.Fatalf("overlay face = %v", got)
	}
	tt.SetSize(windowMinWidth, windowMinHeight)
	a.updateSettings(func(current *settings.Settings) { current.Appearance.Scale = 1.25 })
	scrollToText(tt, "Reset color")
	click(t, tt, "Reset color")
	if appearance := a.settings.Get().Appearance; appearance.Palette != settings.DefaultPalette || appearance.Scale != 1.25 {
		t.Fatalf("reset appearance at minimum window size = %+v", appearance)
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
			theme := ui.Theme{Dark: dark}
			styleTheme(&theme, seed)
			k := tok(&theme)
			if theme.Background != ui.Hex(background) || theme.Accent != ui.Hex(accent) || k.Sidebar != ui.Hex(surface) || k.TranscriptFace != ui.Hex(surface) {
				t.Errorf("seed %q dark=%t: original background, accent or card color changed", seed, dark)
			}
			if k.NavSelected != ui.Hex(selection) || k.NavInk != k.Text || k.Waveform != ui.Hex(recording) {
				t.Errorf("seed %q dark=%t: original selection or recording color changed", seed, dark)
			}
		}
	}
}

// Late preview notifications must not roll back unrelated edits.
func TestLateAppearancePreviewSavePreservesOtherSettings(t *testing.T) {
	a, _, _ := newTestApp(t)
	before := a.settings.Get()
	if err := a.settings.Update(context.Background(), func(current *settings.Settings) {
		current.Appearance.Palette = "#b25280"
	}); err != nil {
		t.Fatal(err)
	}
	paletteSaved := a.settings.Get()
	if err := a.settings.Update(context.Background(), func(current *settings.Settings) {
		current.Appearance.WindowTransparency = settings.MaxWindowTransparency
	}); err != nil {
		t.Fatal(err)
	}
	transparencySaved := a.settings.Get()
	a.updateSettings(func(current *settings.Settings) {
		current.Appearance.Scale = 1.25
		current.Privacy.KeepHistory = false
	})
	// Deliver the preview notifications after the other settings have applied.
	a.settingsChanged(paletteSaved, transparencySaved)
	a.settingsChanged(before, paletteSaved)
	if appearance := a.prefs.Appearance; appearance.Scale != 1.25 || appearance.Palette != "#b25280" ||
		appearance.WindowTransparency != settings.MaxWindowTransparency {
		t.Fatalf("late preview notifications changed appearance to %+v", appearance)
	}
	if a.settingsForm.scale != 3 || a.prefs.Privacy.KeepHistory || a.settingsForm.keepHistory {
		t.Fatalf("late preview notifications reverted scale or privacy: scale index %d, keep history %v", a.settingsForm.scale, a.settingsForm.keepHistory)
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
	tt.Key(0, ui.KeyTab)
	tt.Key(0, ui.KeyEnd)
	frames(tt)
	saved := a.settings.Get()
	if saved.Appearance.Theme != settings.ThemeTransparent || saved.Appearance.WindowTransparency != settings.MaxWindowTransparency || saved.Overlay != initialOverlay {
		t.Fatalf("transparency at maximum: %+v, overlay: %+v", saved.Appearance, saved.Overlay)
	}
	image := tt.Image()
	width, height := image.Bounds().Dx(), image.Bounds().Dy()
	if alpha := image.RGBAAt(width/2, height/2).A; alpha != 255 {
		t.Fatalf("the settings panel at maximum transparency has alpha %d", alpha)
	}
	if alpha := image.RGBAAt(width-4, height-4).A; alpha >= 255 {
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
		image := tt.Image()
		return image.RGBAAt(image.Bounds().Dx()-4, image.Bounds().Dy()-4).A
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
	preview := int(a.settingsForm.windowTransparency)
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
			theme := ui.Theme{Dark: dark}
			styleTheme(&theme, seed)
			if *tok(&theme) != *k || *tok(inkTheme(&theme)) != *k {
				t.Errorf("%s dark=%t: theme lost its custom color", seed, dark)
			}
			for _, background := range []ui.Color{k.Bg, k.Sidebar, k.Subtle, k.Hover} {
				if contrastRatio(k.Text, background) < 4.5 || contrastRatio(k.Text2, background) < 4.5 {
					t.Errorf("%s dark=%t: unreadable text against %v", seed, dark, background)
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
	for red := range 17 {
		for green := range 17 {
			for blue := range 17 {
				seed := fmt.Sprintf("#%02x%02x%02x", min(red*16, 255), min(green*16, 255), min(blue*16, 255))
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
					for _, background := range []ui.Color{k.Bg, k.Sidebar} {
						composited := background.Alpha(1 - float32(settings.MaxWindowTransparency)/100).Over(backdrop)
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

func TestAppearancePaletteSnapshots(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.setPaletteColor("#b25280")
	if got := a.appearanceSettingsModel().Palette; got != "#b25280" {
		t.Fatalf("appearance palette = %q", got)
	}
	overlay := a.overlaySettingsModel()
	if overlay.Palette != "#b25280" || a.settingsForm.ovStyle.previewState(overlay.Palette).Palette != "#b25280" {
		t.Fatalf("overlay preview palette = %q", overlay.Palette)
	}
	if got := a.currentSessionOverlayLayout().Palette; got != "#b25280" {
		t.Fatalf("session layout palette = %q", got)
	}
	input, _ := a.sessionMenuProjectionInput()
	if got := buildSessionMenuViewState(i18n.CurrentTranslator(), input).Palette; got != "#b25280" {
		t.Fatalf("session menu palette = %q", got)
	}
	a.resetPaletteColor()
	if a.appearanceSettingsModel().Palette != "" || a.overlaySettingsModel().Palette != "" || a.currentSessionOverlayLayout().Palette != "" {
		t.Fatal("the default palette did not return to the original empty snapshot")
	}
}

type deferredAppearanceSettings struct {
	current settings.Settings
	updates []func(*settings.Settings)
}

func (service *deferredAppearanceSettings) Get() settings.Settings { return service.current }

func (service *deferredAppearanceSettings) Update(_ context.Context, update func(*settings.Settings)) error {
	service.updates = append(service.updates, update)
	return nil
}

func TestAppearancePreviewRevisionsAndReset(t *testing.T) {
	a, _, _ := newTestApp(t)
	deferred := &deferredAppearanceSettings{current: a.settings.Get()}
	a.settings = deferred
	a.postQueue = func(func()) {}
	settingsRevision, featureRevision := a.settingsRevision, a.featureRevision
	a.setPaletteColor("#b25280")
	if a.settingsRevision == settingsRevision || a.featureRevision == featureRevision {
		t.Fatal("a direct palette preview did not advance the view revisions")
	}
	settingsRevision, featureRevision = a.settingsRevision, a.featureRevision
	a.settingsForm.windowTransparency = 12
	a.saveWindowTransparency()
	if a.settingsRevision == settingsRevision || a.featureRevision == featureRevision {
		t.Fatal("a direct transparency preview did not advance the view revisions")
	}
	if len(deferred.updates) != 2 {
		t.Fatalf("deferred %d preview saves, want 2", len(deferred.updates))
	}
	a.resetPaletteColor()
	if len(deferred.updates) != 3 {
		t.Fatalf("reset deferred %d saves, want 3", len(deferred.updates))
	}
	next := deferred.current
	deferred.updates[0](&next)
	if next.Appearance.Palette != deferred.current.Appearance.Palette {
		t.Fatalf("an invalidated palette save changed the reset value to %q", next.Appearance.Palette)
	}
	deferred.updates[2](&next)
	if next.Appearance.Palette != settings.DefaultPalette {
		t.Fatalf("reset saved palette %q", next.Appearance.Palette)
	}
	a.resetPaletteColor()
	if len(deferred.updates) != 4 {
		t.Fatalf("repeated reset deferred %d saves, want 4", len(deferred.updates))
	}
	deferred.updates[0](&next)
	if next.Appearance.Palette != settings.DefaultPalette {
		t.Fatalf("an old save won after a repeated reset: %q", next.Appearance.Palette)
	}
}

func TestPalettePreviews(t *testing.T) {
	runSnapshots(t, "VOXTNEW_PALETTE_PREVIEWS", func(t *testing.T, out string) {
		a, _, tt := newTestApp(t)
		a.updateSettings(func(current *settings.Settings) { current.General.Language = "zh-Hans" })
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
		a.settingsForm.colorEditorOpen = true
		for _, preview := range []struct{ mode, color string }{
			{settings.ThemeLight, "#b25280"},
			{settings.ThemeDark, "#3979c5"},
			{settings.ThemeTransparent, "#9664c4"},
		} {
			a.updateSettings(func(current *settings.Settings) {
				current.Appearance.Theme, current.Appearance.Palette = preview.mode, preview.color
			})
			tt.SetDark(preview.mode != settings.ThemeLight)
			save(preview.mode)
		}
		a.settingsForm.colorEditorOpen = false
		save("collapsed")
	})
}
