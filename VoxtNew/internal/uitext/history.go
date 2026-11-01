package uitext

// The words of a history entry (the History page, Home's recent results,
// the menu bar): its title, its line of details, the delivery, the
// target language, the group that matched, the sound.

import (
	"fmt"
	"strings"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// EntryTitle is an entry's line in the list: the first line of its
// result, else of what was heard, else why it failed (failed true).
func EntryTitle(e store.Entry) (title string, failed bool) {
	if s := FirstLine(e.Result); s != "" {
		return s, false
	}
	if e.Status == store.StatusFailed && e.Error != "" {
		return FirstLine(e.Error), true
	}
	if s := FirstLine(e.RawText); s != "" {
		return s, false
	}
	if s := FirstLine(e.SourceText); s != "" {
		return s, false
	}
	return i18n.T("No text"), true
}

// FirstLine is the first line of s that is not blank, trimmed.
func FirstLine(s string) string {
	for line := range strings.SplitSeq(s, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			return line
		}
	}
	return ""
}

// EntryMeta is an entry's second line: when, the app, how long it was
// spoken.
func EntryMeta(e store.Entry) string {
	parts := []string{TimeAgo(e.CreatedAt)} // shell_status.go
	if e.AppName != "" {
		parts = append(parts, e.AppName)
	}
	if e.TargetLang != "" {
		parts = append(parts, "→ "+TargetLanguageLabel(e.TargetLang))
	}
	if e.Kind == store.KindRewrite && e.SourceText != "" {
		parts = append(parts, i18n.T("Selection"))
	}
	if e.AudioMS > 0 {
		parts = append(parts, ClockMS(e.AudioMS))
	}
	return strings.Join(parts, " · ")
}

// TargetLang is a translation's target language.
func TargetLang(e store.Entry) string {
	if e.TargetLang == "" {
		return ""
	}
	return TargetLanguageLabel(e.TargetLang)
}

// DeliveryLabel says where a result went.
func DeliveryLabel(e store.Entry) string {
	switch e.Delivery {
	case "insert":
		if e.InsertedName != "" {
			// The user moved to another app before it ended.
			return i18n.T("Inserted into %s", e.InsertedName)
		}
		return i18n.T("Inserted")
	case "card":
		return i18n.T("Shown in the answer card")
	case "copy":
		return i18n.T("Copied")
	}
	return ""
}

// SoundText is what the sound processing did to the speech model's
// copy of the recording (store.Entry.DSP): "Hum filtered, volume
// +7.5 dB"; "" for none.
func SoundText(e store.Entry) string {
	hp, level := false, false
	for _, st := range strings.Split(e.DSP, "+") {
		hp, level = hp || st == "highpass", level || st == "level"
	}
	gain := fmt.Sprintf("%+.1f dB", e.GainDB)
	switch {
	case hp && level && e.GainDB != 0:
		return i18n.T("Hum filtered, volume %s", gain)
	case hp && level:
		return i18n.T("Hum filtered, volume unchanged")
	case hp:
		return i18n.T("Hum filtered")
	case level && e.GainDB != 0:
		return i18n.T("Volume %s", gain)
	case level:
		return i18n.T("Volume unchanged")
	}
	return ""
}

// RuleText is an entry's app enhancement group and how it matched:
// "Team chat (app)"; "" for none.
func RuleText(e store.Entry) string {
	if e.RuleName == "" {
		return ""
	}
	if how := ruleSourceLabel(e.RuleSource); how != "" { // appenhance_preview.go
		return i18n.T("%s (%s)", e.RuleName, how)
	}
	return e.RuleName
}

// ---- The preview ----

// SourceText says how a group matched.
func SourceText(s appctx.Source, pattern string) string {
	switch s {
	case appctx.SourceURL:
		return i18n.T("by the website rule %s", pattern)
	case appctx.SourceBrowserApp:
		return i18n.T("as its browser")
	case appctx.SourceApp:
		return i18n.T("as its app")
	}
	return ""
}

// ruleSourceLabel says how a history entry's group matched.
func ruleSourceLabel(s string) string {
	switch appctx.Source(s) {
	case appctx.SourceURL:
		return i18n.T("website")
	case appctx.SourceBrowserApp:
		return i18n.T("browser")
	case appctx.SourceApp:
		return i18n.T("app")
	}
	return ""
}

// RecentResults are the results of the newest n entries that delivered
// one (done, with a text).
func RecentResults(es []store.Entry, n int) []string {
	var out []string
	for _, e := range es {
		if e.Status == store.StatusDone && strings.TrimSpace(e.Result) != "" {
			out = append(out, e.Result)
			if len(out) == n {
				break
			}
		}
	}
	return out
}
