package proto

// ---- mic.* ----
//
// Which apps record from an audio input now (Settings → Meetings → Detect
// meetings). The helper reads Core Audio's list of processes and whether
// each runs an input; nothing is recorded and no permission is needed. Go
// decides which of them is a meeting (internal/meetdetect) and what to do
// about it.

// MicWatchParams turns the watch on or off: while on, the helper sends
// mic.changed when the apps recording change. The result is the MicUsers
// now.
type MicWatchParams struct {
	On bool `json:"on"`
}

// MicUsers are the processes recording from an input, by pid, without the
// helper and the app themselves: only PID and BundleID are set (a process
// without a bundle id has none).
type MicUsers struct {
	Apps []App `json:"apps"`
}
