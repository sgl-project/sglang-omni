package settings

import "time"

// Models: how the local models' runtimes keep what they loaded (the Local
// models page's Advanced section).
type Models struct {
	// KeepLoaded is how long a loaded model stays in memory unused: one
	// of KeepChoices. "" (Automatic), or a value VoxtNew does not know,
	// keeps each runtime's own idle time (speech 10 min, language 5 min),
	// so older documents need no migration (10-17). The ActNow family's
	// memory rules stay above it: another app's memory request and the
	// memory budget still unload idle models (internal/actnow).
	KeepLoaded string `json:"keepLoaded,omitempty"`
}

// The choices of Models.KeepLoaded.
const (
	KeepAuto      = ""
	Keep5Min      = "5m"
	Keep15Min     = "15m"
	Keep30Min     = "30m"
	Keep1Hour     = "1h"
	KeepUntilQuit = "quit"
)

// KeepChoices are Models.KeepLoaded's choices in the order the UI lists
// them.
var KeepChoices = []string{KeepAuto, Keep5Min, Keep15Min, Keep30Min, Keep1Hour, KeepUntilQuit}

// KeepFor is the idle time KeepLoaded sets for every runtime: auto when
// it leaves each runtime its own; a negative d keeps models until VoxtNew
// quits.
func (m Models) KeepFor() (d time.Duration, auto bool) {
	switch m.KeepLoaded {
	case Keep5Min:
		return 5 * time.Minute, false
	case Keep15Min:
		return 15 * time.Minute, false
	case Keep30Min:
		return 30 * time.Minute, false
	case Keep1Hour:
		return time.Hour, false
	case KeepUntilQuit:
		return -1, false
	}
	return 0, true
}
