package icons

import (
	_ "embed"

	"github.com/egoist/mygo/ui"
)

// Home's mascot: the brand microphone, drawn in the text's color with
// its eyes cut out, and the clothes it wears, in their own colors. Every
// file is in the microphone's units: its viewBox is -100 -16 600 600,
// and a piece of clothing's viewBox says where it sits on it.

var (
	//go:embed mascot-microphone.svg
	mascotMicrophone []byte
	//go:embed mascot-microphone-blink.svg
	mascotMicrophoneBlink []byte
	//go:embed mascot-hat-christmas.svg
	mascotHatChristmas []byte
	//go:embed mascot-hat-new-year.svg
	mascotHatNewYear []byte
	//go:embed mascot-hat-spring-festival.svg
	mascotHatSpring []byte
)

// MascotBox is the microphone's viewBox: left, top, width and height.
var MascotBox = [4]float32{-100, -16, 600, 600}

// MascotPiece is a piece of the mascot's clothing and where it sits, as
// its viewBox in the microphone's units.
type MascotPiece struct {
	SVG *ui.SVG
	Box [4]float32
}

var (
	MascotMicrophone      = ui.MustParseSVG(mascotMicrophone)
	MascotMicrophoneBlink = ui.MustParseSVG(mascotMicrophoneBlink)
	// MascotChristmas is the Christmas outfit: a Santa hat.
	MascotChristmas = []MascotPiece{{ui.MustParseSVG(mascotHatChristmas), [4]float32{10, -180, 320, 230}}}
	// MascotNewYear is the New Year outfit: a party hat.
	MascotNewYear = []MascotPiece{{ui.MustParseSVG(mascotHatNewYear), [4]float32{80, -225, 240, 275}}}
	// MascotSpring is the Spring Festival outfit: a red cap.
	MascotSpring = []MascotPiece{{ui.MustParseSVG(mascotHatSpring), [4]float32{76, -76, 256, 140}}}
)
