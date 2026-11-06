package platform

import "sync/atomic"

// Texts are the words backends say of their own, in the language the app
// shows, as package mygo sets them (SetTexts): the buttons of the panels
// of pages' alert, confirm and prompt, the titles and buttons of file
// dialogs, and the title of the about panel. Backends read them with
// CurrentTexts from any thread.
type Texts struct {
	OK, Cancel, Open, Save, SelectFolder string
	// About is the title of the about panel with "\x00" for the app's
	// name: "About \x00".
	About string
}

// DefaultTexts are the texts in English, before package mygo sets others.
var DefaultTexts = Texts{OK: "OK", Cancel: "Cancel", Open: "Open", Save: "Save", SelectFolder: "Select Folder", About: "About \x00"}

var texts atomic.Pointer[Texts]

// SetTexts sets the texts backends say.
func SetTexts(t Texts) { texts.Store(&t) }

// CurrentTexts returns the texts backends say.
func CurrentTexts() Texts {
	if t := texts.Load(); t != nil {
		return *t
	}
	return DefaultTexts
}
