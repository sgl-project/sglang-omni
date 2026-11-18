package main

import (
	"embed"
	"math"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

const openingDuration = 3200 * time.Millisecond

//go:embed assets/opening/*.png
var openingFiles embed.FS

var (
	openingMicrophone  = openingImage("microphone")
	openingBlink       = openingImage("microphone-blink")
	openingOrange      = openingImage("star-orange")
	openingPeach       = openingImage("star-peach")
	openingWordmark    = openingImage("voxt-wordmark")
	openingAttribution = openingImage("attribution")
)

func openingImage(name string) *ui.Bitmap {
	encoded, err := openingFiles.ReadFile("assets/opening/" + name + ".png")
	if err != nil {
		panic(err)
	}
	bitmap, err := ui.DecodeBitmap(encoded)
	if err != nil {
		panic(err)
	}
	return bitmap
}

func (a *app) startOpening() {
	version := a.version
	if a.prefs.Setup.OpeningShown && a.prefs.Setup.OpeningVersion == version {
		return
	}
	a.openingStarted = time.Now()
	a.updateSettings(func(s *settings.Settings) {
		s.Setup.OpeningShown, s.Setup.OpeningVersion = true, version
	})
}

func openingProgress(elapsed, start, duration time.Duration) float32 {
	progress := min(max(float64(elapsed-start)/float64(duration), 0), 1)
	return float32(1 - math.Pow(1-progress, 3))
}

func (a *app) openingView(c *ui.Context) bool {
	if a.openingStarted.IsZero() {
		return false
	}
	elapsed := time.Since(a.openingStarted)
	if elapsed >= openingDuration || a.prefs.Appearance.ReduceMotion || c.Preferences().ReduceMotion {
		a.openingStarted = time.Time{}
		return false
	}
	c.After(time.Second / 60)
	entrance := openingProgress(elapsed, 0, 450*time.Millisecond)
	wordmark := openingProgress(elapsed, time.Second, 500*time.Millisecond)
	attribution := openingProgress(elapsed, 1600*time.Millisecond, 600*time.Millisecond)
	orange := openingProgress(elapsed, 750*time.Millisecond, 440*time.Millisecond)
	peach := openingProgress(elapsed, 970*time.Millisecond, 420*time.Millisecond)
	tilt := -4 * (openingProgress(elapsed, 450*time.Millisecond, 280*time.Millisecond) - openingProgress(elapsed, 1000*time.Millisecond, 300*time.Millisecond))
	microphone := openingMicrophone
	if elapsed >= 730*time.Millisecond && elapsed < 830*time.Millisecond {
		microphone = openingBlink
	}
	ui.Column(c.Key("opening")).Fill().Background(ui.Hex("#141414")).Center().Label(i18n.T("Welcome to Voxt")).Children(func() {
		ui.Row(c).Gap(32).AlignItems(ui.Center).Translate(8, 0).Children(func() {
			ui.Box(c).Size(190, 190).Radius(45.6).Background(ui.Hex("#151515")).
				Opacity(entrance).Scale(.96+.04*entrance, .96+.04*entrance).Translate(0, 8*(1-entrance)).Children(func() {
				ui.Image(c, microphone).Absolute().Left(45).Top(24).Size(100, 142).Rotate(tilt)
				ui.Image(c, openingOrange).Absolute().Left(131).Top(22).Size(38, 38).Opacity(orange).Scale(.35+.65*orange, .35+.65*orange)
				ui.Image(c, openingPeach).Absolute().Left(152).Top(63).Size(24, 24).Opacity(peach).Scale(.35+.65*peach, .35+.65*peach)
			})
			ui.Column(c).Width(218).Gap(18).Children(func() {
				ui.Image(c, openingWordmark).Size(218, 60).Opacity(wordmark).Translate(0, 4*(1-wordmark))
				ui.Image(c, openingAttribution).Size(218, 29.08).Opacity(attribution).Translate(0, 4*(1-attribution))
			})
		})
	})
	return true
}
