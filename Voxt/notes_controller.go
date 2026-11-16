package main

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type notesState struct {
	nv   notesView
	nvUI notesLocalUI
}

type notesView struct {
	list    []store.Note
	read    bool
	seq     int
	loading bool
	err     string
	view    notesViewState
	viewKey notesViewCacheKey
}

type notesViewCacheKey struct {
	seq           int
	locale        string
	loading       bool
	problems      string
	minute        int64
	location      string
	zoneOffset    int
	keys          string
	helperRecords bool
	editing       notesShortcutStateInput
	usedBy        string
}

func (a *app) loadNotes() {
	notes := &a.nv
	notes.seq++
	notes.loading = true
	seq := notes.seq
	repo := a.lib.store.Notes()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		list, err := repo.List(ctx)
		a.post(func() {
			notes := &a.nv
			if seq != notes.seq {
				return
			}
			notes.loading, notes.read = false, true
			if err != nil {
				notes.err = err.Error()
				return
			}
			notes.list, notes.err = list, ""
		})
	})
}

func (a *app) prepareNotesPage() {
	if !a.nv.read && !a.nv.loading {
		a.loadNotes()
	}
	current := a.prefs.Features.Note.Keys
	if seen := current + "\x00" + i18n.Locale(); seen != a.nvUI.keysFor {
		a.nvUI.keysFor = seen
		a.nvUI.keys = keyRow{sel: chordLabel(current), text: current}
		if current == "" {
			a.nvUI.keys.sel = i18n.T("Off")
		}
	}
}

func (a *app) notesPageModel(now time.Time) notesViewState {
	local := &a.nvUI
	current := a.prefs.Features.Note.Keys
	usedBy := a.usedByLines(recordedKeys(local.keys, current))
	_, zoneOffset := now.Zone()
	editing := notesShortcutStateInput{
		Selection: local.keys.sel, Custom: local.keys.custom, Typing: local.keys.typing,
		Edited: local.keys.edited, Text: local.keys.text, RecordDenied: local.keys.recordDenied, Unknown: local.keys.unknown,
	}
	problems := uitext.HotkeyProblemsFor(i18n.CurrentTranslator(), a.prefs.Features, hotkey.Note)[0]
	key := notesViewCacheKey{
		seq: a.nv.seq, locale: i18n.Locale(), loading: a.nv.loading, problems: strings.Join(problems, "\x00"),
		minute: now.Unix() / 60, location: now.Location().String(), zoneOffset: zoneOffset,
		keys: current, helperRecords: a.helperRecordsKeys(), editing: editing, usedBy: strings.Join(usedBy, "\x00"),
	}
	if key != a.nv.viewKey {
		a.nv.view = buildNotesViewState(i18n.CurrentTranslator(), notesViewStateInput{
			Notes: a.nv.list, Read: a.nv.read, Loading: a.nv.loading, Error: a.nv.err,
			Features: a.prefs.Features, Now: now, HelperRecords: key.helperRecords, UsedBy: usedBy, Editing: editing,
		})
		a.nv.viewKey = key
	}
	return a.nv.view
}

func (a *app) deleteDictionaryNote(id, text string) {
	a.deleteNote(store.Note{ID: id, Text: text})
}

func (a *app) deleteNote(note store.Note) {
	repo := a.lib.store.Notes()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		err := repo.Delete(ctx, note.ID)
		a.post(func() {
			if err != nil && !errors.Is(err, core.ErrNotFound) {
				a.toast(i18n.T("Could not delete the note: %v", err))
				return
			}
			a.toast(i18n.T("Deleted “%s”", noteExcerpt(note.Text)))
			a.loadNotes()
		})
	})
}

func (a *app) setNoteKeys(keys string) {
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Note = settings.Hotkey{Keys: keys, Trigger: settings.TriggerTap}
	})
	a.nvUI.keysFor = keys + "\x00" + i18n.Locale()
}

func (a *app) recordNoteKeys(keys string) notesShortcutViewState {
	chord, err := hotkey.Parse(keys)
	a.nvUI.keys.unknown = err != nil
	if err == nil {
		a.nvUI.keys.text, a.nvUI.keys.edited = chord.String(), false
		a.setNoteKeys(chord.String())
	}
	return a.notesPageModel(time.Now()).Shortcut
}

func (a *app) updateNoteKeysDraft(text string) notesShortcutViewState {
	a.nvUI.keys.text, a.nvUI.keys.edited = text, true
	return a.notesPageModel(time.Now()).Shortcut
}

func (a *app) setNoteHotkeyFocus(focused bool) {
	if focused {
		a.keysHeard = true
		if !a.nvUI.keys.recFocused {
			a.beginHotkeyRecord(&a.nvUI.keys, a.setNoteKeys)
		}
	} else if a.nvUI.keys.recFocused {
		a.cancelHotkeyRecord()
	}
}

func (a *app) openNotePermissions() { a.navigate("/settings/permissions") }
