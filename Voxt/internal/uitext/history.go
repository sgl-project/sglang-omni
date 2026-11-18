package uitext

// The words of a history entry (the History page, Home's recent results,
// the menu bar): its title, its line of details, the delivery, the
// target language, the group that matched, the sound.

import (
	"fmt"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// EntryTitle is an entry's line in the list: the first line of its
// result, else of what was heard, else why it failed (failed true).
func EntryTitle(e store.Entry) (title string, failed bool) {
	return EntryTitleFor(i18n.CurrentTranslator(), e)
}

func EntryTitleFor(translator *i18n.Translator, e store.Entry) (title string, failed bool) {
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
	return translator.T("No text"), true
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
	return EntryMetaFor(i18n.CurrentTranslator(), e, time.Now())
}

func EntryMetaFor(translator *i18n.Translator, e store.Entry, now time.Time) string {
	parts := []string{TimeAgoFor(translator, e.CreatedAt, now)}
	if e.AppName != "" {
		parts = append(parts, e.AppName)
	}
	if e.TargetLang != "" {
		parts = append(parts, "→ "+compose.LanguageName(e.TargetLang, translator.Locale()))
	}
	if e.Kind == store.KindRewrite && e.SourceText != "" {
		parts = append(parts, translator.T("Selection"))
	}
	if e.AudioMS > 0 {
		parts = append(parts, ClockMS(e.AudioMS))
	}
	return strings.Join(parts, " · ")
}

// TargetLang is a translation's target language.
func TargetLang(e store.Entry) string {
	return TargetLangFor(i18n.CurrentTranslator(), e)
}

func TargetLangFor(translator *i18n.Translator, e store.Entry) string {
	if e.TargetLang == "" {
		return ""
	}
	return compose.LanguageName(e.TargetLang, translator.Locale())
}

// DeliveryLabel says where a result went.
func DeliveryLabel(e store.Entry) string {
	return DeliveryLabelFor(i18n.CurrentTranslator(), e)
}

func DeliveryLabelFor(translator *i18n.Translator, e store.Entry) string {
	switch e.Delivery {
	case "insert":
		if e.InsertedName != "" {
			// The user moved to another app before it ended.
			return translator.T("Inserted into %s", e.InsertedName)
		}
		return translator.T("Inserted")
	case "card":
		return translator.T("Shown in the answer card")
	case "copy":
		return translator.T("Copied")
	}
	return ""
}

// SoundText is what the sound processing did to the speech model's
// copy of the recording (store.Entry.DSP): "Hum filtered, volume
// +7.5 dB"; "" for none.
func SoundText(e store.Entry) string {
	return SoundTextFor(i18n.CurrentTranslator(), e)
}

func SoundTextFor(translator *i18n.Translator, e store.Entry) string {
	hp, level := false, false
	for _, st := range strings.Split(e.DSP, "+") {
		hp, level = hp || st == "highpass", level || st == "level"
	}
	gain := fmt.Sprintf("%+.1f dB", e.GainDB)
	switch {
	case hp && level && e.GainDB != 0:
		return translator.T("Hum filtered, volume %s", gain)
	case hp && level:
		return translator.T("Hum filtered, volume unchanged")
	case hp:
		return translator.T("Hum filtered")
	case level && e.GainDB != 0:
		return translator.T("Volume %s", gain)
	case level:
		return translator.T("Volume unchanged")
	}
	return ""
}

// RuleText is an entry's app enhancement group and how it matched:
// "Team chat (app)"; "" for none.
func RuleText(e store.Entry) string {
	return RuleTextFor(i18n.CurrentTranslator(), e)
}

func RuleTextFor(translator *i18n.Translator, e store.Entry) string {
	if e.RuleName == "" {
		return ""
	}
	if how := ruleSourceLabelFor(translator, e.RuleSource); how != "" { // appenhance_preview.go
		return translator.T("%s (%s)", e.RuleName, how)
	}
	return e.RuleName
}

// ---- The preview ----

// SourceText says how a group matched.
func SourceText(s appctx.Source, pattern string) string {
	return SourceTextFor(i18n.CurrentTranslator(), s, pattern)
}

func SourceTextFor(translator *i18n.Translator, s appctx.Source, pattern string) string {
	switch s {
	case appctx.SourceURL:
		return translator.T("by the website rule %s", pattern)
	case appctx.SourceBrowserApp:
		return translator.T("by the browser")
	case appctx.SourceApp:
		return translator.T("by the app")
	}
	return ""
}

// ruleSourceLabel says how a history entry's group matched.
func ruleSourceLabel(s string) string {
	return ruleSourceLabelFor(i18n.CurrentTranslator(), s)
}

func ruleSourceLabelFor(translator *i18n.Translator, s string) string {
	switch appctx.Source(s) {
	case appctx.SourceURL:
		return translator.T("website")
	case appctx.SourceBrowserApp:
		return translator.T("browser")
	case appctx.SourceApp:
		return translator.T("app")
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
