package proto

// ---- hotkey.record / hotkey.cancelRecord ----

// HotkeyRecordParams starts a one-shot chord capture in the helper event tap.
type HotkeyRecordParams struct {
	TimeoutSeconds float64 `json:"timeoutSeconds,omitempty"` // default 15
}

// HotkeyRecordResult is the grammar string the app parses with hotkey.Parse.
type HotkeyRecordResult struct {
	Keys string `json:"keys"`
}
