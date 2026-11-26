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
	if a.meetMgr == nil || a.mvUI.importing != nil || a.meetingActive() {
		return
	}
	if !a.meetingMTDInstalled() {
		a.mvUI.importNoMTD = true
		return
	}
	if !a.prefs.Privacy.KeepHistory {
		a.mvUI.importNoHistory = true
		return
	}
	if !a.meetingDiarizerLanguage() {
		a.mvUI.importNoLanguage = true
		return
	}
	chooseRecordingFile(a, a.importRecording)
}

func (a *app) openMeetingPrivacySettings() { a.navigate("/settings/privacy") }

func (a *app) importRecording(path string) {
	n := &a.mvUI
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
			case errors.Is(err, context.Canceled) || (errors.Is(err, context.DeadlineExceeded) && id != ""):
				// interrupted: the meeting's Continue processing finishes it
			case errors.Is(err, meeting.ErrNoDiarizer):
				if a.meetingMTDInstalled() && !a.meetingDiarizerLanguage() {
					n.importNoLanguage = true
				} else {
					n.importNoMTD = true // MTD went away between the check and the pass
				}
			case errors.Is(err, meeting.ErrNoHistory):
				n.importNoHistory = true // turned off between the check and the pass
			default:
				a.meetingFailed("import", i18n.T("Could not import the recording: %v", err), err)
			}
			a.loadMeetings()
		})
	})
}

func (a *app) runImport(ctx context.Context, imp *meetingImport, path string) (string, error) {
	pcm, err := decodeRecording(ctx, path)
	if err != nil {
		return "", err
	}
	if !a.meetingDiarizerLanguage() {
		return "", meeting.ErrNoDiarizer
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
