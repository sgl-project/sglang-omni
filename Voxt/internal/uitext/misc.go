package uitext

// The words of the other pages: an address pattern's error, the key after
// a paste, a dictionary term's error, the helper's failure, a log line's
// source, a language's name.

import (
	"errors"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
)

// PatternErrorTextFor says why a website rule is not valid.
func PatternErrorTextFor(translator *i18n.Translator, err error) string {
	switch err {
	case appctx.ErrPatternEmpty:
		return translator.T("Type a website, as in github.com.")
	case appctx.ErrPatternSpace:
		return translator.T("A website rule cannot contain spaces.")
	case appctx.ErrPatternScheme:
		return translator.T("Only web addresses (http and https).")
	}
	return translator.T("Start with a site, as in github.com or *.notion.so. * matches anything.")
}

// AutoKeyLabel is the select's label of keys.
func AutoKeyLabel(keys string) string {
	return AutoKeyLabelFor(i18n.CurrentTranslator(), keys)
}

func AutoKeyLabelFor(translator *i18n.Translator, keys string) string {
	switch c := appctx.CanonicalAutoKey(keys); c {
	case "":
		return translator.T("None")
	case "Return":
		return translator.T("Return ↩")
	default:
		return HotkeyTextFor(translator, c)
	}
}

// TermErrorText says why a term cannot be saved.
func TermErrorText(err error) string {
	return TermErrorTextFor(i18n.CurrentTranslator(), err)
}

func TermErrorTextFor(translator *i18n.Translator, err error) string {
	switch {
	case errors.Is(err, dict.ErrEmpty):
		return translator.T("Type the term as it should be written.")
	case errors.Is(err, dict.ErrLong):
		return translator.T("A term can have at most %d characters.", dict.MaxTermChars)
	case errors.Is(err, core.ErrExists):
		return translator.T("This term is in the dictionary already.")
	}
	return translator.T("Could not save the term: %v", err)
}

// HelperFailText says why the helper does not run.
func HelperFailText(st helper.Status) string {
	return HelperFailTextFor(i18n.CurrentTranslator(), st)
}

func HelperFailTextFor(translator *i18n.Translator, st helper.Status) string {
	switch st.Reason {
	case helper.FailMissing:
		return translator.T("Helper missing")
	case helper.FailProtocol:
		return translator.T("Helper out of date")
	case helper.FailUpdated:
		return translator.T("Voxt was updated. Restart it to finish.")
	}
	if st.State == helper.Stopped {
		return translator.T("Helper stopped")
	}
	return translator.T("Helper stopped unexpectedly")
}

// LogSourceLabelFor names a log line's source.
func LogSourceLabelFor(translator *i18n.Translator, s string) string {
	switch s {
	case "":
		return translator.T("All sources")
	case "app":
		return translator.T("App")
	case helper.LogSource:
		return translator.T("Helper")
	case "dictionary":
		return translator.T("Dictionary")
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
