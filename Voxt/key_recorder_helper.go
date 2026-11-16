package main

import (
	"errors"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// helperRecordsKeys reports whether the helper should capture chords.
func (a *app) helperRecordsKeys() bool {
	return a.hstatus.Has(proto.MethodHotkeyRecord) && a.hotkeys.Accessibility
}

func (a *app) markShortcutRecorderFocused() {
	a.keysHeard = true
}

func (a *app) beginHotkeyRecord(r *keyRow, save func(keys string)) string {
	if !a.helperRecordsKeys() {
		return uitext.HotkeyText(r.text)
	}
	a.cancelHotkeyRecord()
	a.recorderActive = true
	params := proto.HotkeyRecordParams{TimeoutSeconds: 15}
	helperCall(a, proto.MethodHotkeyRecord, params, 20*time.Second, func(res proto.HotkeyRecordResult, err error) {
		a.recorderActive = false
		if err != nil {
			var perr *proto.Error
			if errors.As(err, &perr) {
				switch perr.Code {
				case proto.CodePermissionDenied:
					r.recordDenied = true
				case proto.CodeTimeout:
					// back to the idle prompt
				case proto.CodeCancelled:
				}
			}
			return
		}
		r.recordDenied = false
		r.unknown = false
		ch, perr := hotkey.Parse(res.Keys)
		if perr != nil {
			r.unknown = true
			return
		}
		keys := ch.String()
		r.text, r.edited = keys, false
		save(keys)
	})
	return uitext.HotkeyText(r.text)
}

func (a *app) saveShortcutRecorderKeys(keys string, r *keyRow, save func(keys string)) string {
	r.text, r.edited = keys, false
	save(keys)
	return uitext.HotkeyText(keys)
}

func (a *app) customShortcutKeysState(keys string) customKeysViewState {
	chord, err := hotkey.Parse(keys)
	if err != nil {
		return customKeysViewState{ErrorText: uitext.ParseErrorText(err)}
	}
	return customKeysViewState{Valid: true, CanonicalKeys: chord.String(), FormattedKeys: uitext.HotkeyText(keys)}
}

func (a *app) typeShortcutKeys(r *keyRow) customKeysViewState {
	r.edited = true
	return a.customShortcutKeysState(r.text)
}

func (a *app) saveTypedShortcutKeys(r *keyRow, save func(keys string)) customKeysViewState {
	state := a.customShortcutKeysState(r.text)
	if state.Valid {
		r.edited = false
		r.text = state.CanonicalKeys
		save(r.text)
	}
	return state
}

func (a *app) cancelHotkeyRecord() {
	if !a.recorderActive {
		return
	}
	helperCall(a, proto.MethodHotkeyCancelRecord, proto.Empty{}, 5*time.Second, func(proto.Empty, error) {
		a.recorderActive = false
	})
}
