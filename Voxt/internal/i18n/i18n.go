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
// {source: translation} embedded from locales/<tag>.json, or from every
// locales/<tag>/*.json merged when the locale is split by area (zh-Hans
// is, to keep unrelated pull requests from conflicting on one file);
// en.json is empty by design.
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
	"io/fs"
	"strings"
	"sync/atomic"
)

//go:embed locales
var localeFS embed.FS

// catalog is one locale's messages.
type catalog struct {
	tag      string
	messages map[string]string
	sources  map[string]string // translation → English source, for Source
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
	current.Store(newCatalog(tag))
	return tag
}

// Translator reads messages for one explicit locale without changing the
// process-wide catalog used by views.
type Translator struct {
	catalog *catalog
}

func NewTranslator(tag string) *Translator {
	return &Translator{catalog: newCatalog(Match(tag))}
}

func CurrentTranslator() *Translator {
	return &Translator{catalog: current.Load()}
}

func (t *Translator) Locale() string {
	return t.catalog.tag
}

func (t *Translator) T(msg string, args ...any) string {
	if translated, ok := t.catalog.messages[msg]; ok && translated != "" {
		msg = translated
	}
	return format(msg, args)
}

// X translates msg in a context using the translator's locale.
func (t *Translator) X(context, msg string, args ...any) string {
	if translated, ok := t.catalog.messages[context+"\x04"+msg]; ok {
		return format(translated, args)
	}
	return format(msg, args)
}

func (t *Translator) N(n int, one, other string, args ...any) string {
	msg := other
	if n == 1 && t.Locale() == "en" {
		msg = one
	}
	if translated, ok := t.catalog.messages[msg]; ok && translated != "" {
		msg = translated
	}
	return format(msg, args)
}

func newCatalog(tag string) *catalog {
	messages, err := readMessages(tag)
	if err != nil {
		panic("i18n: " + err.Error())
	}
	c := &catalog{tag: tag, messages: messages}
	c.sources = make(map[string]string, len(c.messages))
	for key, translated := range c.messages {
		if _, msg, ok := strings.Cut(key, "\x04"); ok {
			key = msg
		}
		c.sources[translated] = key
	}
	return c
}

// readMessages merges a locale's catalog: locales/<tag>.json when it
// exists, then every locales/<tag>/*.json in name order. A key in two
// files is an error.
func readMessages(tag string) (map[string]string, error) {
	var names []string
	if _, err := fs.Stat(localeFS, "locales/"+tag+".json"); err == nil {
		names = append(names, "locales/"+tag+".json")
	}
	if entries, err := localeFS.ReadDir("locales/" + tag); err == nil {
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".json") {
				names = append(names, "locales/"+tag+"/"+e.Name())
			}
		}
	}
	messages := map[string]string{}
	for _, name := range names {
		data, err := localeFS.ReadFile(name)
		if err != nil {
			return nil, err
		}
		m := map[string]string{}
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		for key, translated := range m {
			if _, dup := messages[key]; dup {
				return nil, fmt.Errorf("%s: %q is in another catalog file too", name, key)
			}
			messages[key] = translated
		}
	}
	return messages, nil
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

// Source is the English source of s, a translation T or X returned in the
// active locale without arguments, or "" (English, or not a translation):
// so a search finds a Chinese label by its English name too.
func Source(s string) string { return current.Load().sources[s] }

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
