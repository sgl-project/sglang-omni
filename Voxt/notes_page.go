package main

// The Notes page (/notes): what the user said after the note shortcut
// (Features.Note, right ⌘ by default), newest first, with its time and
// Delete; the shortcut changes here. A note session is a Transcribe
// session whose text is saved here instead of typed (session
// StartRequest.Note). Voxt 1.x's notes without its floating panel,
// titles, statuses and exports.
//
// The page keeps the notes in memory, read in the background when it
// shows, after a note is saved and after a delete.

import (
	"context"
	"errors"
	"slices"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// notesState is the Notes page's view state.
type notesState struct {
	nv notesView
}

type notesView struct {
	// list holds the notes of the last read; seq numbers the reads: only
	// the last one lands.
	list    []store.Note
	read    bool
	seq     int
	loading bool
	err     string
	listUI  ui.ListState
	// keys is the shortcut's controls, set from keysFor (the keys and
	// the language).
	keys    keyRow
	keysFor string
}

// loadNotes reads the notes in the background.
func (a *app) loadNotes() {
	n := &a.nv
	n.seq++
	n.loading = true
	seq := n.seq
	repo := a.lib.store.Notes()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		list, err := repo.List(ctx)
		a.post(func() {
			n := &a.nv
			if seq != n.seq {
				return
			}
			n.loading, n.read = false, true
			if err != nil {
				n.err = err.Error()
				return
			}
			n.list, n.err = list, ""
		})
	})
}

func (a *app) notesPage(c *ui.Context) {
	t := c.Theme()
	n := &a.nv
	if !n.read && !n.loading {
		a.loadNotes()
	}
	pageHeader(c, i18n.T("Notes"), func() {
		if len(n.list) > 0 {
			caption(c, i18n.N(len(n.list), "%d note", "%d notes", len(n.list))).SingleLine()
		}
	})
	ui.Column(c).Padding(px(t, sp1), px(t, sp6), px(t, sp2)).Gap(px(t, sp2)).Shrink(0).Children(func() {
		a.noteKeysRow(c)
		if n.err != "" {
			notice(c, "notes-err", noticeError, i18n.T("Could not read the notes."), n.err, func() {
				if ghostButton(c.Key("notes-retry"), i18n.T("Try again")).Clicked() {
					a.loadNotes()
				}
			}, nil)
		}
	})
	list := n.list
	ui.Column(c).Grow(1).MinHeight(0).Children(func() {
		switch {
		case !n.read:
		case len(list) == 0:
			hint := i18n.T("Choose a shortcut above, then press it in any app and speak. What you say is saved here instead of typed.")
			if k := a.prefs.Features.Note.Keys; k != "" && len(uitext.HotkeyProblems(a.prefs.Features, hotkey.Note)[0]) == 0 {
				hint = i18n.T("Press %s in any app, speak, and press it again. What you say is saved here instead of typed.", uitext.HotkeyText(k))
			}
			emptyState(c, notesIcon, i18n.T("No notes yet"), hint, nil)
		default:
			ui.List(c.Key("notes-list"), &n.listUI, len(list)).Grow(1).MinHeight(0).Gap(px(t, 2)).Padding(0, 0, px(t, sp4)).
				ItemKey(func(i int) any { return list[i].ID }).Rows(func(r ui.ListRow) {
				a.noteRow(c, list[r.Index])
			})
		}
	})
}

// noteRow is a note in the list: its text (selectable), its time and
// Delete.
func (a *app) noteRow(c *ui.Context, note store.Note) {
	t := c.Theme()
	k := tok(t)
	row := ui.Row(c.Key("note-"+note.ID)).Padding(px(t, 6), px(t, sp2)).Margin(0, px(t, sp4)).Radius(px(t, radiusControl)).
		Gap(px(t, sp2)).AlignItems(ui.Start)
	if row.Hovered() {
		row.Background(k.Hover)
	}
	row.Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
			ui.Text(c, note.Text).Selectable()
			caption(c, uitext.TimeAgo(note.CreatedAt)).SingleLine()
		})
		if iconButton(c.Key("delete"), icons.TrashBinMinimalistic, i18n.T("Delete %s", noteExcerpt(note.Text))).Clicked() {
			a.deleteNote(note)
		}
	})
}

// noteExcerpt is the start of a note, for the labels that name it.
func noteExcerpt(text string) string {
	if r := []rune(text); len(r) > 40 {
		return string(r[:40]) + "…"
	}
	return text
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

// noteKeysRow is the note shortcut: its keys (always a tap), Off, or
// typed ones; what is wrong with them and the other ActNow apps on them.
func (a *app) noteKeysRow(c *ui.Context) {
	t := c.Theme()
	n := &a.nv
	cur := a.prefs.Features.Note.Keys
	if seen := cur + "\x00" + i18n.Locale(); seen != n.keysFor { // changed elsewhere (Settings → Shortcuts' preset)
		n.keysFor = seen
		n.keys = keyRow{sel: chordLabel(cur), text: cur}
		if cur == "" {
			n.keys.sel = i18n.T("Off")
		}
	}
	r := &n.keys
	problems := uitext.HotkeyProblems(a.prefs.Features, hotkey.Note)[0]
	ui.Column(c.Key("notes-keys")).Gap(px(t, 6)).Children(func() {
		settingRow(c, "notes-keys-row", i18n.T("Shortcut"), i18n.T("Press it in any app to start a note, and again to save it."), func() {
			opts, byLabel := keyOptions(cur, true)
			if !slices.Contains(opts, r.sel) {
				r.sel = chordLabel(cur)
			}
			if r.custom {
				r.sel = i18n.T("Custom…")
			}
			if ui.Select(c.Key("notes-keys-sel"), &r.sel, opts).Label(i18n.T("Keys to record a note")).Width(px(t, 180)).Changed() {
				if r.sel == i18n.T("Custom…") {
					r.custom, r.text = true, cur
					return
				}
				keys, ok := byLabel[r.sel]
				if !ok {
					return
				}
				r.custom, r.text = false, keys
				a.setNoteKeys(keys)
			}
		})
		if r.custom {
			a.customKeys(c, r, i18n.T("Custom keys to record a note"), a.setNoteKeys)
		}
		if cur == "" {
			caption(c, i18n.T("Set a shortcut to record notes."))
		}
		problemLines(c, problems)
		a.usedByView(c, recordedKeys(*r, cur))
	})
}

func (a *app) setNoteKeys(keys string) {
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Note = settings.Hotkey{Keys: keys, Trigger: settings.TriggerTap}
	})
	a.nv.keysFor = keys + "\x00" + i18n.Locale() // keeps the controls (a typed shortcut stays in its field)
}
