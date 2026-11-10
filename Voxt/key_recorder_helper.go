package main

import (
	"errors"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
)

// helperRecordsKeys reports whether the helper should capture chords.
func (a *app) helperRecordsKeys() bool {
	return a.hstatus.Has(proto.MethodHotkeyRecord) && a.hotkeys.Accessibility
}

func (a *app) beginHotkeyRecord(r *keyRow, save func(keys string)) {
	if !a.helperRecordsKeys() {
		return
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
}

func (a *app) cancelHotkeyRecord() {
	if !a.recorderActive {
		return
	}
	helperCall(a, proto.MethodHotkeyCancelRecord, proto.Empty{}, 5*time.Second, func(proto.Empty, error) {
		a.recorderActive = false
	})
}
