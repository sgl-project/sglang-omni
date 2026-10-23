package main

// The History page's History settings dialog (Voxt's
// HistoryAudioSettingsSheet): the recordings (keeping them, as Settings →
// Privacy; their folder, moved with them; their size; exporting them;
// deleting them all) and the automatic clean-up (on, keeping 90 days and
// any number of entries by default, as Voxt). A clean-up setting that
// would delete entries now asks first, with their number. The work is in
// history_settings_ops.go.

import (
	"context"
	"slices"
	"strconv"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// histSettingsState is the dialog's state.
type histSettingsState struct {
	open bool
	// The clean-up's controls, and the setting and language they show
	// (formFor; "" = read them again, "asking" = a choice waits for its
	// count or its question).
	cleanOn           bool
	daysSel, countSel string
	formFor           string
	exportSel         string
	// The question before a clean-up setting deletes entries: open, how
	// many, the setting.
	pruneAsk bool
	pruneN   int
	pruneTo  settings.History
	// The question before deleting every recording.
	delAudioAsk bool
	// The recordings' number (-1: unknown) and size; usageFor is ""
	// (measure them), "loading" or "done"; usageSeq numbers the reads.
	usageN     int
	usageBytes int64
	usageFor   string
	usageSeq   int
	// audioMissing is the folder chosen for the recordings when it was
	// not there (the default one is in use meanwhile).
	audioMissing string
	// busy is the job running ("move", "export"), with its progress.
	busy        string
	done, total int
	cancel      context.CancelFunc
	// The clean-up runs; it runs again when it ends.
	pruning, pruneAgain bool
}

// keepDaysLabels and keepCountLabels are the choices' labels, in
// settings.KeepDaysChoices' and KeepCountChoices' order.
func keepDaysLabels() []string {
	return []string{i18n.T("1 year"), i18n.T("6 months"), i18n.T("3 months"), i18n.T("30 days"), i18n.T("1 week")}
}

func keepCountLabels() []string {
	out := []string{i18n.T("No limit")}
	for _, n := range settings.KeepCountChoices[1:] {
		out = append(out, strconv.Itoa(n))
	}
	return out
}

// histSearch is the History page's search.
func (a *app) histSearch() string { return strings.TrimSpace(a.hist.query) }

// historySettingsButton is the History page's button that opens the
// dialog, right of its search field.
func (a *app) historySettingsButton(c *ui.Context) {
	if iconButton(c.Key("hist-settings"), icons.Tuning2, i18n.T("History settings")).Shrink(0).Clicked() {
		a.openHistorySettingsDialog()
	}
}

// openHistorySettingsDialog opens the dialog, measuring the recordings
// and looking for a missing folder again.
func (a *app) openHistorySettingsDialog() {
	hs := &a.histSet
	hs.open, hs.formFor, hs.usageFor = true, "", ""
	if hs.busy == "" {
		a.applyAudioDir()
	}
}

// syncHistForm sets the clean-up's controls from the setting, unless a
// choice is on its way to it.
func (a *app) syncHistForm() {
	hs := &a.histSet
	h := a.prefs.History
	key := strconv.FormatBool(h.CleanupOff) + "/" + strconv.Itoa(h.Days()) + "/" + strconv.Itoa(h.Count()) + "/" + i18n.Locale()
	if hs.formFor == key || hs.formFor == "asking" {
		return
	}
	hs.formFor, hs.cleanOn = key, !h.CleanupOff
	hs.daysSel = keepDaysLabels()[max(slices.Index(settings.KeepDaysChoices, h.Days()), 0)]
	hs.countSel = keepCountLabels()[max(slices.Index(settings.KeepCountChoices, h.Count()), 0)]
}

// historySettingsDialog is the dialog, and its questions over it.
func (a *app) historySettingsDialog(c *ui.Context) {
	hs := &a.histSet
	if !hs.open {
		return
	}
	if hs.usageFor == "" {
		a.loadRecordingUsage()
	}
	a.syncHistForm()
	dialogFrame(c, "hist-set", &hs.open, i18n.T("History settings"), "", 560, func() {
		a.histRecordingsSection(c)
		a.histCleanupSection(c)
	}, func() {
		ui.Spacer(c)
		if primaryButton(c.Key("hist-set-done"), i18n.T("Done")).Clicked() {
			hs.open = false
		}
	})
	a.histSetQuestions(c)
}

// histRecordingsSection is the dialog's Recordings section.
func (a *app) histRecordingsSection(c *ui.Context) {
	t := c.Theme()
	hs := &a.histSet
	h := a.lib.store.History()
	busy := hs.busy != ""
	sectionTitleText(c, i18n.T("Recordings"))
	settingRow(c, "hist-set-audio", i18n.T("Keep the recordings"), i18n.T("Saves the audio with each history entry, to listen again or recognise it once more. When this is off, a recording whose recognition failed is still kept for 7 days, so that you can retry it."), func() {
		if ui.Switch(c.Key("hist-set-audio"), &a.settingsForm.keepAudio).Label(i18n.T("Save each recording")).Disabled(!a.prefs.Privacy.KeepHistory).Changed() {
			on := a.settingsForm.keepAudio
			a.updateSettings(func(s *settings.Settings) { s.Privacy.KeepAudio = on })
		}
	})
	chosen := a.prefs.History.AudioDir
	shown := chosen
	if shown == "" {
		shown = h.DefaultAudioDir()
	}
	move := i18n.T("Choosing another folder moves the recordings there: each one is copied and checked before the original is deleted.")
	settingRowNoteTip(c, "hist-set-dir", i18n.T("Storage location"), move, uitext.ShortPath(shown), func() {
		if ui.Button(c.Key("hist-set-choose"), i18n.T("Choose…")).Disabled(busy).Clicked() {
			chooseFolder(a, i18n.T("Move the recordings to…"), i18n.T("Move here"), a.moveRecordings)
		}
		if iconButton(c.Key("hist-set-reveal"), icons.FolderOpen, i18n.T("Show in Finder")).Clicked() {
			revealInFinder(h.AudioDir())
		}
		if chosen != "" && ui.Button(c.Key("hist-set-default"), i18n.T("Use the default")).Disabled(busy).Clicked() {
			a.moveRecordings("")
		}
	})
	if m := hs.audioMissing; m != "" {
		notice(c, "hist-set-missing", noticeWarn, i18n.T("%s is not available: new recordings go to %s until it is back.", uitext.ShortPath(m), uitext.ShortPath(h.DefaultAudioDir())), "", nil, nil)
	}
	usage := i18n.T("Measuring…")
	switch {
	case hs.usageFor != "done":
	case hs.usageN < 0:
		usage = i18n.T("Could not measure them")
	default:
		usage = i18n.N(hs.usageN, "%d recording · %s", "%d recordings · %s", hs.usageN, uitext.HumanBytes(hs.usageBytes))
	}
	settingRowNote(c, "hist-set-usage", i18n.T("Saved recordings"), usage, func() {
		if ui.Button(c.Key("hist-set-delete"), i18n.T("Delete all…")).Disabled(busy || hs.usageN <= 0).Clicked() {
			hs.delAudioAsk = true
		}
	})
	settingRow(c, "hist-set-export", i18n.T("Export recordings"), i18n.T("Copies the recordings into a folder you choose: a folder for each feature, each file named by its date and first words."), func() {
		opts := []string{i18n.T("All recordings"), a.shownLabel()}
		if !slices.Contains(opts, hs.exportSel) {
			hs.exportSel = opts[0]
		}
		ui.Select(c.Key("hist-set-scope"), &hs.exportSel, opts).Label(i18n.T("Which recordings")).Width(px(t, 200)).Shrink(1).MinWidth(px(t, 120))
		if ui.Button(c.Key("hist-set-export"), i18n.T("Export…")).Disabled(busy || hs.usageN == 0).Clicked() {
			only := hs.exportSel == opts[1]
			chooseFolder(a, i18n.T("Export the recordings to…"), i18n.T("Export here"), func(dir string) { a.exportRecordings(dir, only) })
		}
	})
	if busy {
		a.histWorkRow(c)
	}
}

// shownLabel names the recordings the History page shows: its tab, and
// its search.
func (a *app) shownLabel() string {
	tab := []string{i18n.T("Transcribe"), i18n.T("Translate"), i18n.T("Rewrite")}[min(max(a.hist.tab, 0), 2)]
	q := a.histSearch()
	if q == "" {
		return i18n.T("%s only", tab)
	}
	if r := []rune(q); len(r) > 16 {
		q = string(r[:16]) + "…"
	}
	return i18n.T("%s matching “%s”", tab, q)
}

// histWorkRow is the job's progress and its Cancel.
func (a *app) histWorkRow(c *ui.Context) {
	t := c.Theme()
	hs := &a.histSet
	ui.Row(c.Key("hist-set-busy")).Gap(px(t, sp3)).AlignItems(ui.Center).Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 6)).Children(func() {
			label := i18n.T("Exporting the recordings… %d of %d", hs.done, hs.total)
			if hs.busy == "move" {
				label = i18n.T("Moving the recordings… %d of %d", hs.done, hs.total)
			}
			caption(c, label)
			frac := 0.0
			if hs.total > 0 {
				frac = float64(hs.done) / float64(hs.total)
			}
			ui.Progress(c, frac)
		})
		if ui.Button(c.Key("hist-set-cancel"), i18n.T("Cancel")).Disabled(hs.cancel == nil).Clicked() && hs.cancel != nil {
			hs.cancel()
		}
	})
}

// histCleanupSection is the dialog's Auto-cleanup section.
func (a *app) histCleanupSection(c *ui.Context) {
	t := c.Theme()
	hs := &a.histSet
	sectionTitleText(c, i18n.T("Auto-cleanup")).Margin(px(t, sp4), 0, 0, 0)
	ask := func(next settings.History) {
		hs.formFor = "asking"
		a.askCleanup(next)
	}
	settingRow(c, "hist-set-clean", i18n.T("Clean up automatically"), i18n.T("Deletes old entries with their recordings: when VoxtNew starts, after each new entry, and when you change these settings."), func() {
		if ui.Switch(c.Key("hist-set-clean"), &hs.cleanOn).Label(i18n.T("Clean up the history")).Changed() {
			next := a.prefs.History
			next.CleanupOff = !hs.cleanOn
			if hs.cleanOn {
				ask(next)
			} else {
				a.saveCleanup(next)
			}
		}
	})
	if !hs.cleanOn {
		caption(c, i18n.T("Entries stay until you delete them."))
		return
	}
	settingRow(c, "hist-set-days", i18n.T("Keep entries for"), "", func() {
		opts := keepDaysLabels()
		if ui.Select(c.Key("hist-set-days"), &hs.daysSel, opts).Label(i18n.T("Keep entries for")).Width(px(t, 160)).Changed() {
			next := a.prefs.History
			next.KeepDays = settings.KeepDaysChoices[max(slices.Index(opts, hs.daysSel), 0)]
			ask(next)
		}
	})
	settingRow(c, "hist-set-count", i18n.T("Keep at most"), i18n.T("The newest entries of each feature: Transcribe, Translate and Rewrite each keep this many."), func() {
		opts := keepCountLabels()
		if ui.Select(c.Key("hist-set-count"), &hs.countSel, opts).Label(i18n.T("Keep at most")).Width(px(t, 160)).Changed() {
			next := a.prefs.History
			next.KeepCount = settings.KeepCountChoices[max(slices.Index(opts, hs.countSel), 0)]
			ask(next)
		}
	})
}

// histSetQuestions are the dialog's confirmations: a clean-up that
// deletes entries now, deleting every recording.
func (a *app) histSetQuestions(c *ui.Context) {
	t := c.Theme()
	hs := &a.histSet
	if was := hs.pruneAsk; was {
		n := hs.pruneN
		ui.Modal(c.Key("hist-prune-ask"), &hs.pruneAsk, func() {
			ui.Text(c, i18n.N(n, "Delete %d entry now?", "Delete %d entries now?", n)).Bold()
			ui.Text(c, i18n.N(n, "This will delete %d entry and its recording now, and others as they pass the limit. This cannot be undone.",
				"This will delete %d entries and their recordings now, and others as they pass the limit. This cannot be undone.", n)).Width(px(t, 380))
			ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
				if ui.Button(c.Key("hist-prune-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
					hs.pruneAsk = false
				}
				if recordButton(c.Key("hist-prune-ok"), i18n.T("Delete")).Clicked() {
					hs.pruneAsk = false
					a.saveCleanup(hs.pruneTo)
				}
			})
		})
		if !hs.pruneAsk && hs.formFor == "asking" {
			hs.formFor = "" // not saved: the controls show the setting again
		}
	}
	if hs.delAudioAsk {
		ui.Modal(c.Key("hist-delaudio-ask"), &hs.delAudioAsk, func() {
			ui.Text(c, i18n.T("Delete every recording?")).Bold()
			ui.Text(c, i18n.T("The recordings of every entry are deleted; the entries and their texts stay. This cannot be undone.")).Width(px(t, 380))
			ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
				if ui.Button(c.Key("hist-delaudio-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
					hs.delAudioAsk = false
				}
				if recordButton(c.Key("hist-delaudio-ok"), i18n.T("Delete the recordings")).Clicked() {
					hs.delAudioAsk = false
					a.deleteRecordings()
				}
			})
		})
	}
}

// historyFolderNotice is the History page's notice of a recordings
// folder that is missing.
func (a *app) historyFolderNotice(c *ui.Context) {
	m := a.histSet.audioMissing
	if m == "" {
		return
	}
	t := c.Theme()
	ui.Column(c).Padding(0, px(t, sp6), px(t, sp3)).Shrink(0).Children(func() {
		notice(c, "hist-missing", noticeWarn, i18n.T("%s is not available: new recordings go to %s until it is back.", uitext.ShortPath(m), uitext.ShortPath(a.lib.store.History().DefaultAudioDir())), "", func() {
			if ghostButton(c.Key("hist-missing-open"), i18n.T("History settings")).Clicked() {
				a.openHistorySettingsDialog()
			}
		}, nil)
	})
}

// showHistorySettings goes to the History page and opens the dialog
// (Settings → Privacy and Storage link to it).
func (a *app) showHistorySettings() {
	a.navigate("/history")
	a.openHistorySettingsDialog()
}
