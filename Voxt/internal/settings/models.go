package settings

import (
	"strconv"
	"strings"
	"time"
)

// Models: how the local models' runtimes keep what they loaded (the Local
// models page's Advanced section), and the models hidden from the
// features' menus.
type Models struct {
	// KeepLoaded is how long a loaded model stays in memory unused: one
	// of KeepChoices. "" (Automatic), or a value VoxtNew does not know,
	// keeps each runtime's own idle time (speech 10 min, language 5 min),
	// so older documents need no migration (10-17). The ActNow family's
	// memory rules stay above it: another app's memory request and the
	// memory budget still unload idle models (internal/actnow).
	KeepLoaded string `json:"keepLoaded,omitempty"`
	// Hidden are the speech models (their ids) the user hid on the Speech
	// models page: the features' menus leave them out, unless one is in
	// use (Voxt 1.x's hiddenModelVisibilityIDs).
	Hidden []string `json:"hidden,omitempty"`
	// LLMWait is how long Transcribe, Translate and Rewrite wait at least
	// for a language model's answer: one of WaitChoices, or another
	// WaitMinutes the user typed (#3941). "" (Automatic),
	// or a value VoxtNew does not know, keeps the session's own limits
	// (internal/session's LLMTimeout and WriteTimeout), so older documents
	// need no migration (#3821). Either way a long text gets longer.
	LLMWait string `json:"llmWait,omitempty"`
	// Faster is each speech model's Faster transcription switch (its id:
	// on): whether its server verifies the model's speculative draft. A
	// model the user never switched is on while its draft is installed,
	// as 2.0.2 and 2.0.3 did (the download turned it on), so who
	// downloaded the draft keeps it (#3873).
	Faster map[string]bool `json:"faster,omitempty"`
}

// FasterOn says whether speech model id uses its draft, given whether
// the draft is installed: the switch, else the installed draft.
func (m Models) FasterOn(id string, draftInstalled bool) bool {
	if on, set := m.Faster[id]; set {
		return on && draftInstalled
	}
	return draftInstalled
}

// The choices of Models.LLMWait.
const (
	WaitAuto  = ""
	Wait1Min  = "1m"
	Wait2Min  = "2m"
	Wait5Min  = "5m"
	Wait10Min = "10m"
)

// WaitChoices are Models.LLMWait's choices in the order the UI lists them.
var WaitChoices = []string{WaitAuto, Wait1Min, Wait2Min, Wait5Min, Wait10Min}

// WaitMaxMinutes is the most minutes a custom LLMWait may be (#3941).
const WaitMaxMinutes = 60

// WaitMinutes is LLMWait for n minutes, the presets' form ("5m").
func WaitMinutes(n int) string { return strconv.Itoa(n) + "m" }

// LLMWaitFor is the least time LLMWait gives a language model's answer:
// a whole number of minutes in 1..WaitMaxMinutes; 0 for Automatic.
func (m Models) LLMWaitFor() time.Duration {
	s, ok := strings.CutSuffix(m.LLMWait, "m")
	n, err := strconv.Atoi(s)
	if !ok || err != nil || n < 1 || n > WaitMaxMinutes || WaitMinutes(n) != m.LLMWait { // WaitMinutes: no "+5m" or "05m"
		return 0
	}
	return time.Duration(n) * time.Minute
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
