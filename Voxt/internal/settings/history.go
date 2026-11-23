package settings

import "time"

// History: how long the history keeps its entries and where their
// recordings are (the History page's History settings dialog). Whether
// it keeps entries and recordings at all is Privacy's (KeepHistory,
// KeepAudio). Zero values are Voxt's defaults, so older documents need no
// migration (10-17).
type History struct {
	// CleanupOff keeps every entry until the user deletes it. Without it
	// the automatic clean-up (on, as Voxt's historyCleanupEnabled)
	// deletes, with their recordings, the entries older than KeepDays
	// and those past the newest KeepCount of their feature.
	CleanupOff bool `json:"cleanupOff,omitempty"`
	// KeepDays is one of KeepDaysChoices; 0, or a number not among them,
	// is DefaultKeepDays (Voxt's 90 days).
	KeepDays int `json:"keepDays,omitempty"`
	// KeepCount is one of KeepCountChoices: 0 (Voxt's default) keeps any
	// number.
	KeepCount int `json:"keepCount,omitempty"`
	// AudioDir is the folder of the recordings; "" is the data folder's
	// audio/.
	AudioDir string `json:"audioDir,omitempty"`
}

// DefaultKeepDays is how long the clean-up keeps an entry by default.
const DefaultKeepDays = 90

// KeepDaysChoices and KeepCountChoices are Voxt's choices in its order
// (HistoryRetentionPeriod, HistoryRetentionCount): 1 year, 6 months, 3
// months, 30 days, 1 week; no limit, then 50 to 1000.
var (
	KeepDaysChoices  = []int{365, 180, 90, 30, 7}
	KeepCountChoices = []int{0, 50, 100, 300, 500, 1000}
)

// Days is how many days the clean-up keeps an entry.
func (h History) Days() int {
	for _, d := range KeepDaysChoices {
		if d == h.KeepDays {
			return d
		}
	}
	return DefaultKeepDays
}

// Count is how many entries of each feature the clean-up keeps; 0 = any
// number.
func (h History) Count() int {
	for _, n := range KeepCountChoices {
		if n == h.KeepCount {
			return n
		}
	}
	return 0
}

// Cutoff is the time before which the clean-up deletes an entry at now;
// the zero time when the clean-up is off.
func (h History) Cutoff(now time.Time) time.Time {
	if h.CleanupOff {
		return time.Time{}
	}
	return now.AddDate(0, 0, -h.Days())
}

// PerKind is how many entries of each feature the clean-up keeps; 0
// when it is off or keeps any number.
func (h History) PerKind() int {
	if h.CleanupOff {
		return 0
	}
	return h.Count()
}

// SameCleanup reports whether h and o clean up alike.
func (h History) SameCleanup(o History) bool {
	return h.CleanupOff == o.CleanupOff && (h.CleanupOff || h.Days() == o.Days() && h.Count() == o.Count())
}
