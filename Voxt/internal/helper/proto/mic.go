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
// now. Hosts are the names of processes that mark a meeting (Zoom's
// aomhost): MicUsers.Hosts lists those that run.
type MicWatchParams struct {
	On    bool     `json:"on"`
	Hosts []string `json:"hosts,omitempty"`
}

// MicUsers are the processes recording from an input, by pid, without the
// helper and the app themselves: only PID and BundleID are set (a process
// without a bundle id has none). Hosts are the processes named in the
// watch's Hosts that run while an app records; nil from a helper that
// does not look for them (or a watch that named none).
type MicUsers struct {
	Apps  []App     `json:"apps"`
	Hosts []MicHost `json:"hosts,omitzero"`
}

// MicHost is a process mic.watch was asked about, by name, and the pid of
// the app it serves: its "-pid" argument's, else its parent's.
type MicHost struct {
	PID  int    `json:"pid"`
	Name string `json:"name"`
	App  int    `json:"app"`
}
