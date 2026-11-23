package main

// The key recorder's demo actions (key_recorder.go): record-only
// implementations for the fixture demo.

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func (host *meetingDemoHost) beginHotkeyRecord(row *keyRow, _ func(string)) string {
	host.feature.keysHeard = false
	if state := host.current().state.KeyRecorder; state != nil && state.HelperRecordsKeys {
		host.cancelHotkeyRecord()
		host.recorderActive = true
	}
	host.record("beginHotkeyRecord")
	return uitext.HotkeyTextFor(i18n.CurrentTranslator(), row.text)
}

func (host *meetingDemoHost) cancelHotkeyRecord() {
	host.feature.keysHeard = false
	if host.current().state.KeyRecorder == nil {
		host.record("cancelHotkeyRecord")
		return
	}
	if !host.recorderActive {
		return
	}
	host.recorderActive = false
	host.record("cancelHotkeyRecord")
}

func (host *meetingDemoHost) saveShortcutRecorderKeys(keys string, row *keyRow, save func(keys string)) string {
	row.text, row.edited = keys, false
	host.record("saveShortcutRecorderKeys %s", keys)
	save(keys)
	return uitext.HotkeyTextFor(i18n.CurrentTranslator(), keys)
}

func (host *meetingDemoHost) typeShortcutKeys(row *keyRow) customKeysViewState {
	row.edited = true
	host.record("typeShortcutKeys")
	return buildCustomKeysViewState(i18n.CurrentTranslator(), row.text)
}

func (host *meetingDemoHost) saveTypedShortcutKeys(row *keyRow, save func(keys string)) customKeysViewState {
	state := buildCustomKeysViewState(i18n.CurrentTranslator(), row.text)
	host.record("saveTypedShortcutKeys")
	if state.Valid {
		row.edited = false
		row.text = state.CanonicalKeys
		save(row.text)
	}
	return state
}

func (host *meetingDemoHost) markShortcutRecorderFocused() {}

func (host *meetingDemoHost) saveKeys(keys string) { host.record("saveKeys %s", keys) }
