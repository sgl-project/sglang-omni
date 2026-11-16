package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type keyRecorderViewState struct {
	HelperRecordsKeys bool                `json:"helper_records_keys"`
	DisplayedKeys     string              `json:"displayed_keys"`
	Typed             customKeysViewState `json:"typed"`
	RawKeys           string              `json:"raw_keys"`
	Custom            bool                `json:"custom"`
	Typing            bool                `json:"typing"`
	Edited            bool                `json:"edited"`
	Unknown           bool                `json:"unknown"`
	RecordDenied      bool                `json:"record_denied"`
	Focused           bool                `json:"focused"`
}

type customKeysViewState struct {
	Valid         bool   `json:"valid"`
	CanonicalKeys string `json:"canonical_keys"`
	FormattedKeys string `json:"formatted_keys"`
	ErrorText     string `json:"error_text"`
}

func buildKeyRecorderViewState(translator *i18n.Translator, helperRecordsKeys bool, keys string) keyRecorderViewState {
	return keyRecorderViewState{
		HelperRecordsKeys: helperRecordsKeys,
		DisplayedKeys:     uitext.HotkeyTextFor(translator, keys),
		Typed:             buildCustomKeysViewState(translator, keys),
	}
}

func buildCustomKeysViewState(translator *i18n.Translator, keys string) customKeysViewState {
	chord, err := hotkey.Parse(keys)
	if err != nil {
		return customKeysViewState{ErrorText: uitext.ParseErrorTextFor(translator, err)}
	}
	return customKeysViewState{Valid: true, CanonicalKeys: chord.String(), FormattedKeys: uitext.HotkeyTextFor(translator, keys)}
}
