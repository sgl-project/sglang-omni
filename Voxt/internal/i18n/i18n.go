// Package i18n translates the UI. Messages are keyed by their English
// source text, as Apple's NSLocalizedString and Qt's tr() do:
//
//	ui.Button(c.Key("save"), i18n.T("Save"))
//	ui.Text(c, i18n.T("%d lines", len(lines)))
//	ui.Text(c, i18n.N(n, "%d line", "%d lines", n))
//	i18n.X("verb", "Record")  // same English, different translations
//
// so wrapping a string never changes the English UI, and missing
// translations fall back to English. Catalogs are JSON objects
// {source: translation} embedded from locales/<tag>.json; en.json is empty
// by design.
//
// TestKeyParity (parity_test.go) scans every non-test .go file for T/N/X
// calls with literal messages, and fails
// when zh-Hans lacks one, keeps one nothing uses, or changes its format
// verbs (a translation may reorder them with %[2]s). Every function is
// safe from any goroutine; switching the locale takes effect at the next
// build pass (the shell rebuilds the native menu itself).
//
// Imports: nothing internal.
package i18n

import (
	"embed"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
)

//go:embed locales/*.json
var localeFS embed.FS

// catalog is one locale's messages.
type catalog struct {
	tag      string
	messages map[string]string
}

var current atomic.Pointer[catalog]

func init() { SetLocale("en") }

// Supported lists the UI languages, as BCP-47 tags, English first.
func Supported() []string { return []string{"en", "zh-Hans"} }

// Name is the name of a supported language in that language, as a
// language picker shows it ("English", "简体中文"); the tag itself for
// others.
func Name(tag string) string {
	switch tag {
	case "en":
		return "English"
	case "zh-Hans":
		return "简体中文"
	}
	return tag
}

// Locale returns the active tag.
func Locale() string { return current.Load().tag }

// SetLocale activates the catalog for tag (Match(tag) when tag is not
// supported exactly) and returns the tag in use.
func SetLocale(tag string) string {
	tag = Match(tag)
	c := &catalog{tag: tag, messages: map[string]string{}}
	if data, err := localeFS.ReadFile("locales/" + tag + ".json"); err == nil {
		if err := json.Unmarshal(data, &c.messages); err != nil {
			panic("i18n: locales/" + tag + ".json: " + err.Error()) // caught by tests
		}
	}
	current.Store(c)
	return tag
}

// Match maps a system or user locale ("zh-CN", "zh_Hans_CN", "en-GB") to
// the closest supported tag, English when none is close.
func Match(tag string) string {
	t := strings.ToLower(strings.ReplaceAll(tag, "_", "-"))
	switch {
	case t == "zh" || strings.HasPrefix(t, "zh-hans") || t == "zh-cn" || t == "zh-sg" || strings.HasPrefix(t, "zh-cn-"):
		return "zh-Hans"
	}
	// Exact matches of other supported tags (none yet besides en).
	for _, s := range Supported() {
		if strings.EqualFold(s, tag) {
			return s
		}
	}
	return "en"
}

// T translates msg and, when args are given, formats it with fmt.Sprintf.
func T(msg string, args ...any) string {
	return format(lookup(msg), args)
}

// X translates msg in a context, for one English text that needs two
// translations; the catalog key is "context\x04msg" (gettext's convention).
func X(context, msg string, args ...any) string {
	if s, ok := current.Load().messages[context+"\x04"+msg]; ok {
		return format(s, args)
	}
	return format(msg, args)
}

// N picks the singular (one) or plural (other) English source for n by the
// active locale's plural rule, translates it and formats it with args.
// Locales without a singular (zh-Hans) always use other.
func N(n int, one, other string, args ...any) string {
	msg := other
	if n == 1 && Locale() == "en" {
		msg = one
	}
	return format(lookup(msg), args)
}

func lookup(msg string) string {
	if s, ok := current.Load().messages[msg]; ok && s != "" {
		return s
	}
	return msg
}

func format(s string, args []any) string {
	if len(args) == 0 {
		return s
	}
	return fmt.Sprintf(s, args...)
}
