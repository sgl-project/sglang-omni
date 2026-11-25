package main

import "github.com/egoist/mygo/ui"

// hudSmoothing is whether the floating windows' text is drawn with Core
// Text's font smoothing. Off: smoothing emboldens light text the most,
// and their white text over the dark HUD looked heavy to the user (10-16).
// Voxt's SwiftUI text, measured on the Mac, has the smoothed weight (the
// same pixels as ours with it on); unsmoothed glyphs carry about a fifth
// less ink (docs/behavior.md 5.30). The main window keeps it.
const hudSmoothing = false

// hudContent is the content of a floating window showing view: its text
// as hudSmoothing says (MyGo patch 0005, ui.Content.FontSmoothing).
func hudContent(view func(*ui.Context)) *ui.Content {
	return ui.View(view).FontSmoothing(hudSmoothing)
}
