package main

import (
	"context"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/learn"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

type scanState struct {
	model    local.Model
	counted  bool
	count    int
	running  bool
	stopping bool
	cancel   context.CancelFunc
	read     int
	total    int
	found    int
	ended    bool
	stopped  bool
	failed   error
}

func dictionaryScanInput(state scanState) dictionaryScanStateInput {
	return dictionaryScanStateInput{
		Model: state.model, Counted: state.counted, Count: state.count, Running: state.running,
		Stopping: state.stopping, Read: state.read, Total: state.total, Found: state.found,
		Ended: state.ended, Stopped: state.stopped, Err: state.failed,
	}
}

func (a *app) scanModel() (local.Model, bool) {
	return dictionaryScanModel(a.prefs, a.installed(local.LLM))
}

func (a *app) scanOpen(id string) func(context.Context) (local.Chat, local.Model, error) {
	services := a.local
	return func(ctx context.Context) (local.Chat, local.Model, error) { return services.LLM(ctx, id) }
}

func (a *app) openDictionaryScan() {
	model, ok := a.scanModel()
	if ok {
		a.openScan(model)
	}
}

func (a *app) openScan(model local.Model) {
	state := &a.dv.scan
	if state.running {
		a.dictUI.scanOpen = true
		return
	}
	*state = scanState{model: model}
	a.dictUI.scanOpen = true
	scanner := a.scanner
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		count, err := scanner.Pending(ctx)
		a.post(func() {
			state := &a.dv.scan
			state.counted, state.count = true, count
			if err != nil {
				state.failed = err
			}
		})
	})
}

func scanFoundText(found int) string {
	return dictionaryScanFoundTextFor(i18n.CurrentTranslator(), found)
}

func (a *app) startScan() {
	state := &a.dv.scan
	ctx, cancel := context.WithCancel(context.Background())
	state.running, state.cancel, state.total = true, cancel, state.count
	scanner, modelID, open := a.scanner, state.model.ID, a.scanOpen(state.model.ID)
	languages := learn.Languages{Main: session.MainLanguage(a.prefs, i18n.Locale()), Others: a.prefs.General.OtherLanguages}
	go func() {
		defer cancel()
		found, err := scanner.Scan(ctx, modelID, open, languages, func(read, total, found int) {
			a.post(func() { a.dv.scan.read, a.dv.scan.total, a.dv.scan.found = read, total, found })
		})
		stopped := ctx.Err() != nil
		a.post(func() {
			state := &a.dv.scan
			state.running, state.stopping, state.ended, state.stopped, state.found = false, false, true, stopped, found
			if err != nil && !stopped {
				state.failed = err
			}
			if !a.dictUI.scanOpen {
				a.toast(scanFoundText(found))
			}
			a.loadSuggestions()
		})
	}()
}

func (a *app) stopDictionaryScan() {
	state := &a.dv.scan
	if !state.running || state.cancel == nil {
		return
	}
	state.stopping = true
	state.cancel()
}
