package uitext

// The words of the other pages: an address pattern's error, the key after
// a paste, a dictionary term's error, a permission's state, the helper's
// failure, a log line's source, a language's name.

import (
	"errors"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/dict"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/lang"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/permission"
)

// PatternErrorText says why a website rule is not valid.
func PatternErrorText(err error) string {
	switch err {
	case appctx.ErrPatternEmpty:
		return i18n.T("Type a website, as in github.com.")
	case appctx.ErrPatternSpace:
		return i18n.T("A website rule has no spaces.")
	case appctx.ErrPatternScheme:
		return i18n.T("Only web addresses (http and https).")
	}
	return i18n.T("Start with a site, as in github.com or *.notion.so; * stands for anything.")
}

// AutoKeyLabel is the select's label of keys.
func AutoKeyLabel(keys string) string {
	switch c := appctx.CanonicalAutoKey(keys); c {
	case "":
		return i18n.T("None")
	case "Return":
		return i18n.T("Return ↩")
	default:
		return HotkeyText(c)
	}
}

// TermErrorText says why a term cannot be saved.
func TermErrorText(err error) string {
	switch {
	case errors.Is(err, dict.ErrEmpty):
		return i18n.T("Type the term as it must be written.")
	case errors.Is(err, dict.ErrLong):
		return i18n.T("A term has at most %d characters.", dict.MaxTermChars)
	case errors.Is(err, core.ErrExists):
		return i18n.T("This term is in the dictionary already.")
	}
	return i18n.T("Could not save the term: %v", err)
}

func PermissionText(p permission.State) (string, bool) {
	switch p {
	case permission.Granted:
		return i18n.T("Allowed"), true
	case permission.Denied:
		return i18n.T("Not allowed"), false
	case permission.Restricted:
		return i18n.T("Restricted by your organisation"), false
	case permission.NotDetermined:
		return i18n.T("Not asked yet"), false
	}
	return i18n.T("Unknown"), false
}

// HelperFailText says why the helper does not run.
func HelperFailText(st helper.Status) string {
	switch st.Reason {
	case helper.FailMissing:
		return i18n.T("Helper missing")
	case helper.FailProtocol:
		return i18n.T("Helper out of date")
	case helper.FailUpdated:
		return i18n.T("VoxtNew was updated: restart it to finish.")
	}
	if st.State == helper.Stopped {
		return i18n.T("Helper stopped")
	}
	return i18n.T("Helper stopped unexpectedly")
}

func LogSourceLabel(s string) string {
	switch s {
	case "":
		return i18n.T("All sources")
	case "app":
		return i18n.T("App")
	case helper.LogSource:
		return i18n.T("Helper")
	}
	return s
}

// LangLabel names a language in the UI's language, with its own name
// when that differs: "Japanese · 日本語", "日语 · 日本語".
func LangLabel(code string) string {
	l, ok := lang.Lookup(code)
	if !ok {
		return code
	}
	name := l.NameIn(i18n.Locale())
	if l.Native != "" && l.Native != name {
		return name + " · " + l.Native
	}
	return name
}
