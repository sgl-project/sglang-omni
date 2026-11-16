package main

// Import a recording file as a meeting (the Meetings page's Import
// recording…): the file's sound, decoded as `Voxt transcribe` decodes it
// (afconvert, which reads the sound of videos too), becomes the meeting's
// one track. MTD labels every voice Speaker 1..N, then the correction and
// summary run as after a recorded meeting. Without MTD there is no plain
// fallback: a dialog says so and opens Speech models.

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

// meetingImport is the import running from the Meetings page.
type meetingImport struct {
	name     string
	progress float64
	cancel   context.CancelFunc
}

// Test seams: the file dialog and the decoder (afconvert exists on a Mac).
var (
	chooseRecordingFile = func(a *app, then func(string)) {
		go func() {
			files, err := mygo.Dialog.Open(mygo.OpenDialogOptions{Parent: a.win, Title: i18n.T("Import recording"),
				Filters: []mygo.FileFilter{{Name: i18n.T("Audio or video"), Extensions: []string{"m4a", "mp3", "wav", "aiff", "caf", "mp4", "mov"}}}})
			if err != nil || len(files) == 0 {
				return
			}
			a.post(func() { then(files[0]) })
		}()
	}
	decodeRecording = decodeAudio
)

// meetingImport picks a recording to import. It needs MTD: the file is
// one mixed track, and only MTD tells its speakers apart.
func (a *app) meetingImport() {
	if a.meetMgr == nil || a.mv.importing != nil || a.meetingActive() {
		return
	}
	if !a.meetingMTDInstalled() {
		a.mv.importNoMTD = true
		return
	}
	chooseRecordingFile(a, a.importRecording)
}

func (a *app) importRecording(path string) {
	n := &a.mv
	if n.importing != nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Hour)
	imp := &meetingImport{name: filepath.Base(path), cancel: cancel}
	n.importing = imp
	a.async(func() {
		id, err := a.runImport(ctx, imp, path)
		a.post(func() {
			cancel()
			n.importing = nil
			switch {
			case err == nil:
				a.openMeetingDetail(id)
			case errors.Is(err, context.Canceled):
				// interrupted: the meeting's Continue processing finishes it
			case errors.Is(err, meeting.ErrNoDiarizer):
				n.importNoMTD = true // MTD went away between the check and the pass
			default:
				a.meetingFailed("import", i18n.T("Could not import the recording: %v", err), err)
			}
			a.loadMeetings()
		})
	})
}

func (a *app) runImport(ctx context.Context, imp *meetingImport, path string) (string, error) {
	pcm, err := decodeRecording(path)
	if err != nil {
		return "", err
	}
	req := meeting.ImportRequest{
		Name:      strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
		CreatedAt: time.Now(),
		PCM:       pcm,
	}
	if st, statErr := os.Stat(path); statErr == nil {
		req.CreatedAt = st.ModTime()
	}
	return a.meetMgr.Import(ctx, req, func(p float64) {
		a.post(func() { imp.progress = p })
	})
}

// meetingImportRow is the Meetings page's progress while a recording imports.
func (a *app) meetingImportRow(c *ui.Context, imp *meetingImport) {
	t := c.Theme()
	fillCard(c).Padding(px(t, sp3), px(t, sp4)).Children(func() {
		ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Center).Children(func() {
			ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp1)).Children(func() {
				ui.Text(c, i18n.T("Importing %s…", imp.name)).SingleLine().Ellipsis("…")
				ui.Progress(c, imp.progress).Label(i18n.T("Importing %s…", imp.name))
			})
			if ui.Button(c.Key("mt-import-cancel"), i18n.T("Cancel")).Shrink(0).Clicked() {
				imp.cancel()
			}
		})
	})
}

// meetingImportDialog explains that importing needs MTD and opens its page.
func (a *app) meetingImportDialog(c *ui.Context) {
	n := &a.mv
	open := n.importNoMTD
	if !open {
		return
	}
	dialogFrame(c, "mt-import-mtd", &open, i18n.T("MOSS-Transcribe-Diarize is not downloaded"), "", 420, func() {
		ui.Text(c, i18n.T("Importing a recording needs MOSS-Transcribe-Diarize to label its speakers. Download it on the Speech models page, then import again."))
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("mt-import-mtd-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
			open = false
		}
		if primaryButton(c.Key("mt-import-mtd-open"), i18n.T("Open Speech models")).Clicked() {
			open = false
			a.navigate(modelsPath)
		}
	})
	if !open {
		n.importNoMTD = false
	}
}
