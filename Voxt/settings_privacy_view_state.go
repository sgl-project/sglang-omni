package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

type settingsPrivacyViewState struct {
	Online                     string                 `json:"online"`
	SendWindow                 bool                   `json:"send_window"`
	SendClipboard              bool                   `json:"send_clipboard"`
	SendClipboardWithSelection bool                   `json:"send_clipboard_with_selection"`
	KeepHistory                bool                   `json:"keep_history"`
	KeepAudio                  bool                   `json:"keep_audio"`
	KeepAudioEnabled           bool                   `json:"keep_audio_enabled"`
	LogText                    bool                   `json:"log_text"`
	ClearHistoryAsk            bool                   `json:"clear_history_ask"`
	HistoryAccess              historyAccessViewState `json:"history_access"`
}

type settingsPrivacyStateInput struct {
	Privacy         settings.Privacy
	HasUpdateFeed   bool
	ClearHistoryAsk bool
	HistoryAccess   historyAccessViewState
}

func buildSettingsPrivacyViewState(translator *i18n.Translator, input settingsPrivacyStateInput) settingsPrivacyViewState {
	online := translator.T("Otherwise, Voxt goes online only to download models and read its announcements. Reading them sends nothing about you.")
	if input.HasUpdateFeed {
		online = translator.T("Otherwise, Voxt goes online only to check for updates, download models and read its announcements. Reading them sends nothing about you.")
	}
	return settingsPrivacyViewState{
		Online: online, SendWindow: input.Privacy.SendWindow, SendClipboard: input.Privacy.SendClipboardImage,
		SendClipboardWithSelection: input.Privacy.SendClipboardImageWithSelection,
		KeepHistory:                input.Privacy.KeepHistory, KeepAudio: input.Privacy.KeepAudio, KeepAudioEnabled: input.Privacy.KeepHistory,
		LogText: input.Privacy.LogText, ClearHistoryAsk: input.ClearHistoryAsk, HistoryAccess: input.HistoryAccess,
	}
}
