package main

import (
	"context"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func (a *app) privacySettingsModel(now time.Time) settingsPrivacyViewState {
	return buildSettingsPrivacyViewState(i18n.CurrentTranslator(), settingsPrivacyStateInput{
		Privacy: a.prefs.Privacy, HasUpdateFeed: a.upd != nil, ClearHistoryAsk: a.clearHistoryAsk,
		HistoryAccess: a.historyAccessModel(now),
	})
}

func (a *app) setSendWindow(on bool) {
	a.updateSettings(func(current *settings.Settings) { current.Privacy.SendWindow = on })
	if on {
		a.askScreenRecording()
	}
}

func (a *app) setSendClipboardImage(on bool) {
	a.updateSettings(func(current *settings.Settings) { current.Privacy.SendClipboardImage = on })
}

func (a *app) setSendClipboardImageWithSelection(on bool) {
	a.updateSettings(func(current *settings.Settings) { current.Privacy.SendClipboardImageWithSelection = on })
}

func (a *app) setKeepHistory(on bool) {
	a.updateSettings(func(current *settings.Settings) {
		current.Privacy.KeepHistory = on
		if !on {
			current.Privacy.KeepAudio = false
		}
	})
}

func (a *app) setKeepAudio(on bool) {
	a.updateSettings(func(current *settings.Settings) { current.Privacy.KeepAudio = on })
}

func (a *app) setLogText(on bool) {
	a.updateSettings(func(current *settings.Settings) { current.Privacy.LogText = on })
}

// askScreenRecording shows macOS's Screen Recording prompt (once) and,
// while Voxt is not allowed, says where to allow it.
func (a *app) askScreenRecording() {
	if !a.hstatus.Has(proto.MethodScreenWindow) {
		return
	}
	helperCall(a, proto.MethodPermissionsRequest, proto.PermissionsRequestParams{Permission: proto.PermissionScreenRecording}, 10*time.Second, func(r proto.PermissionsResult, err error) {
		if err == nil && !r.ScreenRecording {
			a.toast(i18n.T("To send screenshots, allow Voxt in System Settings → Privacy & Security → Screen & System Audio Recording."))
		}
	})
}

// clearHistory deletes every history entry, in the background.
func (a *app) clearHistory() {
	h := a.lib.store.History()
	a.async(func() {
		n, err := h.DeleteAll(context.Background(), "")
		a.post(func() {
			if n > 0 {
				a.historyChanged() // history_page.go
			}
			if err != nil {
				a.toast(i18n.T("Could not clear the history: %v", err))
				return
			}
			a.toast(i18n.N(n, "Deleted %d entry", "Deleted %d entries", n))
		})
	})
}
