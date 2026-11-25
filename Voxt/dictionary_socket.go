package main

// The socket of `Voxt dictionary import` (dictionary_cli.go,
// internal/dictapi): open while the app runs, in the default data folder,
// 0600. It takes a dictionary file and merges its terms as Dictionary →
// Import… does (dictionary_import.go); it reads nothing of the user's and
// answers only the counts.

import (
	"context"
	"fmt"
	"log"
	"path/filepath"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/historyapi"
)

// dictionarySource is the logs' source of the command's imports.
const dictionarySource = "dictionary"

type dictionarySocketState struct {
	dictionaryStop context.CancelFunc
}

// openDictionarySocket runs once, when the app is ready.
func (a *app) openDictionarySocket() {
	ln, err := historyapi.Listen(filepath.Join(a.home(), dictapi.SocketName))
	if err != nil {
		log.Printf("dictionary socket: %v", err)
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	srv := &dictapi.Server{Import: a.importForCommand}
	go func() {
		if err := srv.Serve(ctx, ln); err != nil {
			log.Printf("dictionary socket: %v", err)
		}
	}()
	a.dictionaryStop = cancel
}

// closeDictionarySocket ends an import running and removes the socket,
// for quitting.
func (a *app) closeDictionarySocket() {
	if a.dictionaryStop != nil {
		a.dictionaryStop()
	}
}

// importForCommand merges a file the command sent, on the socket's
// goroutine; the page and the logs hear of it on the main thread.
func (a *app) importForCommand(ctx context.Context, name string, data []byte, dryRun bool) (dictapi.Result, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	res, err := importData(ctx, a.lib.store.Dictionary(), name, data, dryRun)
	res.DryRun = dryRun
	a.post(func() { a.commandImported(res, err) })
	return res, err
}

// commandImported notes a command's import in the logs (the counts, not
// the terms or errors), refreshes the page after a real run, and toasts a save.
func (a *app) commandImported(res dictapi.Result, err error) {
	what := "imported terms for the command line"
	if res.DryRun {
		what = "counted terms for the command line (dry run)"
	}
	e := core.LogEntry{Time: time.Now(), Level: core.SeverityInfo, Source: dictionarySource,
		Message: fmt.Sprintf("%s: %d new, %d updated, %d rows skipped", what, res.Added, res.Updated, res.Skipped)}
	if err != nil {
		e.Level, e.Message = core.SeverityWarning, what+": failed"
	}
	a.logs.Append(e)
	if !res.DryRun && (err != nil || res.Added+res.Updated > 0) {
		a.loadDict()
	}
	if err != nil || res.DryRun || res.Added+res.Updated == 0 {
		return
	}
	a.toast(importMessage(res))
}
