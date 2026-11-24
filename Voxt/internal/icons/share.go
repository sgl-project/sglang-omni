package icons

import "github.com/egoist/mygo/ui"

// The share card's and the hat studio's marks (share_card.go,
// share_studio.go) as the mockup draws them: 24-unit strokes in the
// text's color, the same in both variants. Shuffle, back, eraser, undo,
// clear and the mic are Tabler Icons' outline icons (MIT, see NOTICE),
// some redrawn slightly; the others were drawn for VoxtNew.

func stroke(name, width, body string) Icon {
	s := parse(`<g fill="none" stroke="currentColor" stroke-width="` + width + `" stroke-linecap="round" stroke-linejoin="round">` + body + `</g>`)
	return Icon{Name: "share-" + name, Linear: s, Bold: s}
}

var (
	// ShareShuffle: another background and hat.
	ShareShuffle = stroke("shuffle", "1.9", `<path d="M18 4l3 3l-3 3"/><path d="M18 20l3 -3l-3 -3"/><path d="M3 7h3a5 5 0 0 1 5 5a5 5 0 0 0 5 5h5"/><path d="M21 7h-5a4.98 4.98 0 0 0 -3 1"/><path d="M9 16a4.98 4.98 0 0 1 -3 1h-3"/>`)
	// ShareCopy and ShareCopied: copy the card, and the tick after it.
	ShareCopy   = stroke("copy", "1.9", `<rect x="8.5" y="8.5" width="11.5" height="11.5" rx="2.5"/><path d="M15.5 8.5V6.5A2.5 2.5 0 0 0 13 4H6.5A2.5 2.5 0 0 0 4 6.5V13a2.5 2.5 0 0 0 2.5 2.5h2"/>`)
	ShareCopied = stroke("copied", "2.2", `<path d="M5 12.5l4.5 4.5L19 7.5"/>`)
	// ShareMic: record (the card's round button).
	ShareMic = stroke("mic", "2", `<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>`)
	// The studio's tools.
	ShareEraser   = stroke("eraser", "1.9", `<path d="M19 20H8.5l-4.2-4.3a1 1 0 0 1 0-1.4l9.6-9.6a1 1 0 0 1 1.4 0l5 5a1 1 0 0 1 0 1.4L12 19.5"/><path d="M18 13.3L11.7 7"/>`)
	ShareUndo     = stroke("undo", "1.9", `<path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>`)
	ShareClear    = stroke("clear", "1.9", `<path d="M4 7h16"/><path d="M10 11v6M14 11v6"/><path d="M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2l1-12"/><path d="M9 7V4h6v3"/>`)
	ShareWardrobe = stroke("wardrobe", "1.9", `<rect x="5" y="3" width="14" height="16.5" rx="2"/><path d="M12 3v16.5"/><path d="M10 10.5v2M14 10.5v2"/><path d="M7.5 19.5V21M16.5 19.5V21"/>`)
	ShareClose    = stroke("close", "2.2", `<path d="M7 7l10 10M17 7L7 17"/>`)
	ShareBack     = stroke("back", "2", `<path d="M15 6l-6 6l6 6"/>`)
	// ShareHat is the studio's mark and Put on: a top hat. ShareHatBold
	// is Put on's, a little heavier.
	ShareHat     = stroke("hat", "1.9", `<path d="M7 16V7.2A1.7 1.7 0 0 1 8.7 5.5h6.6A1.7 1.7 0 0 1 17 7.2V16"/><path d="M7 12.6h10"/><path d="M3.5 16.5c2.4 1.3 5.2 2 8.5 2s6.1-.7 8.5-2"/>`)
	ShareHatBold = stroke("hat-bold", "2", `<path d="M7 16V7.2A1.7 1.7 0 0 1 8.7 5.5h6.6A1.7 1.7 0 0 1 17 7.2V16"/><path d="M7 12.6h10"/><path d="M3.5 16.5c2.4 1.3 5.2 2 8.5 2s6.1-.7 8.5-2"/>`)
)

// ShareStar is the logo's star, a four-pointed one of the mockup's
// curves, in the text's color: the card's orange and peach ones.
var ShareStar = ui.MustParseSVG([]byte(`<svg xmlns="http://www.w3.org/2000/svg" width="130" height="130" viewBox="0 0 130 130"><path fill="currentColor" d="M65 0Q65 65 130 65Q65 65 65 130Q65 65 0 65Q65 65 65 0Z"/></svg>`))
