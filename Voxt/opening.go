package main

import (
	"embed"
	"math"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
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

type openingViewState struct {
	Entrance, Wordmark, Attribution float32
	Orange, Peach, Tilt             float32
	Blink                           bool
}

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

func openingProgress(elapsed, start, duration time.Duration) float32 {
	progress := min(max(float64(elapsed-start)/float64(duration), 0), 1)
	return float32(1 - math.Pow(1-progress, 3))
}

func buildOpeningViewState(elapsed time.Duration) openingViewState {
	return openingViewState{
		Entrance:    openingProgress(elapsed, 0, 450*time.Millisecond),
		Wordmark:    openingProgress(elapsed, time.Second, 500*time.Millisecond),
		Attribution: openingProgress(elapsed, 1600*time.Millisecond, 600*time.Millisecond),
		Orange:      openingProgress(elapsed, 750*time.Millisecond, 440*time.Millisecond),
		Peach:       openingProgress(elapsed, 970*time.Millisecond, 420*time.Millisecond),
		Tilt: -4 * (openingProgress(elapsed, 450*time.Millisecond, 280*time.Millisecond) -
			openingProgress(elapsed, 1000*time.Millisecond, 300*time.Millisecond)),
		Blink: elapsed >= 730*time.Millisecond && elapsed < 830*time.Millisecond,
	}
}

func openingView(c *ui.Context, state openingViewState) {
	c.After(time.Second / 60)
	microphone := openingMicrophone
	if state.Blink {
		microphone = openingBlink
	}
	ui.Column(c.Key("opening")).Fill().Background(ui.Hex("#141414")).Center().Label(i18n.T("Welcome to Voxt")).Children(func() {
		ui.Row(c).Gap(32).AlignItems(ui.Center).Translate(8, 0).Children(func() {
			ui.Box(c).Size(190, 190).Radius(45.6).Background(ui.Hex("#151515")).
				Opacity(state.Entrance).Scale(.96+.04*state.Entrance, .96+.04*state.Entrance).Translate(0, 8*(1-state.Entrance)).Children(func() {
				ui.Image(c, microphone).Absolute().Left(45).Top(24).Size(100, 142).Rotate(state.Tilt)
				ui.Image(c, openingOrange).Absolute().Left(131).Top(22).Size(38, 38).Opacity(state.Orange).Scale(.35+.65*state.Orange, .35+.65*state.Orange)
				ui.Image(c, openingPeach).Absolute().Left(152).Top(63).Size(24, 24).Opacity(state.Peach).Scale(.35+.65*state.Peach, .35+.65*state.Peach)
			})
			ui.Column(c).Width(218).Gap(18).Children(func() {
				ui.Image(c, openingWordmark).Size(218, 60).Opacity(state.Wordmark).Translate(0, 4*(1-state.Wordmark))
				ui.Image(c, openingAttribution).Size(218, 29.08).Opacity(state.Attribution).Translate(0, 4*(1-state.Attribution))
			})
		})
	})
}
