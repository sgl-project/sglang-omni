package main

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func (a *app) saveNotes() {
	notes := &a.dictUI.notes
	notes.dirty = false
	notes.seen = notes.text
	if value := notes.text; value != a.prefs.Dictionary.Notes {
		a.updateSettings(func(s *settings.Settings) { s.Dictionary.Notes = value })
	}
}

func (a *app) autosaveDictionaryNotes(changed, focused bool) {
	notes := &a.dictUI.notes
	if changed {
		notes.dirty = true
		notes.seq++
		if seq := notes.seq; a.win != nil {
			time.AfterFunc(promptSaveDelay, func() {
				a.post(func() {
					if notes.dirty && notes.seq == seq {
						a.saveNotes()
					}
				})
			})
		}
	} else if notes.dirty && !focused {
		a.saveNotes()
	}
}
