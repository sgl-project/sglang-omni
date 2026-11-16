package main

// Retry and Transcribe again run a history entry's recording through session.Retry.

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type entryRetry struct {
	id    string
	stage session.State
}

func (a *app) retryEntry(entry store.Entry) {
	if a.mgr == nil || a.hist.retry.id != "" || entry.Audio == "" {
		return
	}
	if _, ok := a.featureModel(local.ASR, string(entry.Kind)); !ok {
		a.toast(i18n.T("No speech model is installed."))
		return
	}
	if _, ok := a.featureModel(local.LLM, string(entry.Kind)); !ok && entry.Kind != store.KindTranscribe {
		a.toast(i18n.T("No language model is set up."))
		return
	}
	a.hist.retry = entryRetry{id: entry.ID, stage: session.Recognizing}
	manager := a.mgr
	a.async(func() {
		result, err := manager.Retry(context.Background(), session.RetryRequest{Entry: entry, OnStage: func(stage session.State) {
			a.post(func() {
				if a.hist.retry.id == entry.ID {
					a.hist.retry.stage = stage
				}
			})
		}})
		a.post(func() {
			a.hist.retry = entryRetry{}
			switch {
			case err == nil:
				a.toast(i18n.T("Recognised again in %s", uitext.StageTime(result.TotalMS)))
			case errors.Is(err, session.ErrNoRecording):
				a.toast(i18n.T("The recording is gone."))
			default:
				message, _ := uitext.ModelErrorText(err)
				a.toast(i18n.T("Retry failed: %s", message))
			}
			a.historyChanged()
		})
	})
}

func (a *app) retryFromNotice(id string) {
	history := a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		entry, err := history.Get(ctx, id)
		a.post(func() {
			if err != nil {
				a.toast(i18n.T("The entry is gone."))
				return
			}
			a.showMainWindow()
			a.openHistoryEntry(entry)
			a.retryEntry(entry)
		})
	})
}

const recordingSweep = time.Hour

func (a *app) expireRecordings() {
	history := a.lib.store.History()
	go func() {
		for {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			count, err := history.ExpireAudio(ctx, time.Now())
			cancel()
			if count > 0 || err != nil {
				message := fmt.Sprintf("deleted %d recordings kept for a retry", count)
				if err != nil {
					message += ": " + err.Error()
				}
				a.logs.Append(core.LogEntry{Time: time.Now(), Level: core.SeverityInfo, Source: sessionSource, Message: message})
				a.post(a.historyChanged)
			}
			time.Sleep(recordingSweep)
		}
	}()
}
