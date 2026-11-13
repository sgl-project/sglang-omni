package main

// The socket of `Voxt transcribe` (transcribe_cli.go, internal/fileasr):
// open while the app runs, in the default data folder, 0600. It takes
// audio and gives back its text; it reads nothing of the user's.

import (
	"context"
	"errors"
	"fmt"
	"log"
	"path/filepath"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/fileasr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/historyapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

type transcribeSocketState struct {
	transcribeStop context.CancelFunc
}

// openTranscribeSocket runs once, when the app is ready.
func (a *app) openTranscribeSocket() {
	ln, err := historyapi.Listen(filepath.Join(a.home(), fileasr.SocketName))
	if err != nil {
		log.Printf("transcribe socket: %v", err)
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	srv := &fileasr.Server{Open: a.fileRecognizer, Done: a.logFileTranscription}
	go func() {
		if err := srv.Serve(ctx, ln); err != nil {
			log.Printf("transcribe socket: %v", err)
		}
	}()
	a.transcribeStop = cancel
}

// closeTranscribeSocket ends the transcriptions running and removes the
// socket, for quitting.
func (a *app) closeTranscribeSocket() {
	if a.transcribeStop != nil {
		a.transcribeStop()
	}
}

// fileRecognizer is Transcribe's speech model with the language hint of
// dictation, or lang when the command gives one. It reads the settings
// on the main thread.
func (a *app) fileRecognizer(ctx context.Context, lang string) (fileasr.Recognize, error) {
	type pick struct {
		id, lang string
		ok       bool
	}
	ch := make(chan pick, 1)
	a.post(func() {
		m, ok := a.featureModel(local.ASR, hotkey.Transcribe)
		ch <- pick{m.ID, session.SpeechLang(a.prefs, i18n.Locale()), ok}
	})
	var p pick
	select {
	case p = <-ch:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	if !p.ok {
		return nil, errors.New("no speech model is installed for Transcribe: set one up on Voxt's Speech models page")
	}
	if lang == "" {
		lang = p.lang
	}
	prov, _, err := a.local.ASR(ctx, p.id)
	if err != nil {
		return nil, err
	}
	return func(ctx context.Context, pcm []byte) (string, error) {
		res, err := asr.Recognize(ctx, prov, pcm, lang, nil, "", asr.Feed{})
		return res.Text, err
	}, nil
}

// logFileTranscription notes a `Voxt transcribe` in the logs, without
// its text.
func (a *app) logFileTranscription(segments int, audio time.Duration, err error) {
	e := core.LogEntry{Time: time.Now(), Level: core.SeverityInfo, Source: sessionSource,
		Message: fmt.Sprintf("transcribed a file for the command line: %s of audio, %d segments", audio.Round(time.Second), segments)}
	if err != nil {
		e.Level, e.Message = core.SeverityWarning, e.Message+": "+err.Error()
	}
	a.logs.Append(e)
}
