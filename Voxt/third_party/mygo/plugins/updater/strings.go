package updater

import (
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/egoist/mygo/plugins/updater/internal/frontend"
)

// Strings are the texts of the update window in one language. The plugin
// has them in English, Chinese (zh-Hans, zh-Hant), Dutch, French, German,
// Italian, Japanese, Korean, Polish, Portuguese (pt-BR), Russian, Spanish,
// Turkish and Ukrainian, and Options.Strings changes them or adds
// languages.
//
// Some are formats, whose arguments are listed: indexed verbs such as
// %[2]s let a language order them as it needs.
type Strings struct {
	// Title of the window: "Software Update".
	Title string
	// MenuItem is the label of MenuItem: "Check for Updates…".
	MenuItem string

	Checking string // "Checking for updates…"
	Cancel   string
	OK       string

	UpToDate string // "You’re up to date!"
	// UpToDateMessage: the app's name and version.
	UpToDateMessage string

	Unavailable string // "Updates Unavailable"
	// UnavailableMessage explains that the app cannot write where it is
	// installed: the app's name.
	UnavailableMessage string
	// DevelopmentBuild explains that development builds do not update.
	DevelopmentBuild string

	Error        string // "Update Error!"
	CheckError   string // "An error occurred while checking for updates…"
	InstallError string // "An error occurred while installing the update…"

	// Available: the app's name.
	Available string
	// AvailableMessage: the app's name, the new version and the running
	// one.
	AvailableMessage   string
	ReleaseNotes       string // "Release Notes:"
	AutomaticDownloads string // the checkbox
	Skip               string // "Skip This Version"
	RemindLater        string // "Remind Me Later"
	Install            string // "Install Update"

	Downloading string // "Downloading update…"
	// Progress: the size downloaded and the whole size, as Megabytes.
	Progress string
	// Megabytes: a number, such as 12.3, with the decimal mark of the
	// language.
	Megabytes  string
	Installing string // "Installing update…"

	Ready string // "Ready to Relaunch"
	// ReadyMessage: the app's name and the new version.
	ReadyMessage string
	Later        string
	Relaunch     string // "Relaunch Now"
}

// formats lists the Strings that are formats, with the number of their
// arguments.
var formats = map[string]int{
	"UpToDateMessage":    2,
	"UnavailableMessage": 1,
	"Available":          1,
	"AvailableMessage":   3,
	"Progress":           2,
	"Megabytes":          1,
	"ReadyMessage":       2,
}

// check reports the first format of s, among those that are set, that fmt
// cannot fill with its arguments.
func (s *Strings) check() error {
	v := reflect.ValueOf(s).Elem()
	for i := range v.NumField() {
		name, text := v.Type().Field(i).Name, v.Field(i).String()
		n, ok := formats[name]
		if text == "" || !ok {
			continue
		}
		args := make([]any, n)
		for j := range args {
			args[j] = "\x00" + strconv.Itoa(j)
		}
		if out := fmt.Sprintf(text, args...); strings.Contains(out, "%!") {
			return fmt.Errorf("%s %q is not a format of %d strings", name, text, n)
		}
	}
	return nil
}

// merge fills the empty fields of s from fallback.
func (s *Strings) merge(fallback *Strings) {
	v, f := reflect.ValueOf(s).Elem(), reflect.ValueOf(fallback).Elem()
	for i := range v.NumField() {
		if v.Field(i).String() == "" {
			v.Field(i).Set(f.Field(i))
		}
	}
}

// text is the language the window shows, resolved once.
type text struct {
	Strings
	// lang is the language, such as "zh-Hans".
	lang string
	// comma is set for languages whose decimal mark is a comma, rtl for
	// those written from right to left.
	comma, rtl bool
}

// texts returns the texts of the window around its views.
func (t *text) texts() frontend.Texts {
	return frontend.Texts{
		Title:              t.Title,
		ReleaseNotes:       t.ReleaseNotes,
		AutomaticDownloads: t.AutomaticDownloads,
		Lang:               t.lang,
		RTL:                t.rtl,
	}
}

// newText returns the strings of the language the plugin has, or the app
// gives, that best matches locale, completed with the plugin's own and
// English.
func newText(locale string, extra map[string]Strings) *text {
	var available []string
	for l := range translations {
		available = append(available, l)
	}
	for l := range extra {
		available = append(available, l)
	}
	slices.Sort(available)
	lang := matchLanguage(locale, available)
	t := &text{lang: lang, comma: decimalComma[lang], rtl: rightToLeft[baseLanguage(lang)]}
	for l, s := range extra {
		if strings.EqualFold(l, lang) {
			t.Strings = s
		}
	}
	if s, ok := translations[lang]; ok {
		t.Strings.merge(&s)
	}
	t.Strings.merge(&english)
	return t
}

// checkStrings reports translations of the app that are not valid.
func checkStrings(extra map[string]Strings) error {
	var errs []error
	for l, s := range extra {
		if err := s.check(); err != nil {
			errs = append(errs, fmt.Errorf("strings of %q: %w", l, err))
		}
	}
	return errors.Join(errs...)
}

// matchLanguage returns the language among available, which are language
// tags such as "de" or "zh-Hant", that best matches locale, a language tag
// or a POSIX locale such as "zh_TW.UTF-8": the same language, script and
// region, else the same language and script, else the same language. It
// returns "en" without a match.
func matchLanguage(locale string, available []string) string {
	locale, _, _ = strings.Cut(locale, ".") // zh_TW.UTF-8
	locale, _, _ = strings.Cut(locale, "@") // sr_RS@latin
	parts := strings.FieldsFunc(locale, func(r rune) bool { return r == '-' || r == '_' })
	if len(parts) == 0 {
		return "en"
	}
	lang := strings.ToLower(parts[0])
	var script, region string
	for _, p := range parts[1:] {
		switch {
		case len(p) == 4:
			script = strings.ToUpper(p[:1]) + strings.ToLower(p[1:])
		case len(p) == 2 || len(p) == 3 && p[0] >= '0' && p[0] <= '9':
			region = strings.ToUpper(p)
		}
	}
	if lang == "zh" && script == "" {
		script = "Hans"
		switch region {
		case "TW", "HK", "MO":
			script = "Hant"
		}
	}
	var candidates []string
	if script != "" {
		candidates = append(candidates, lang+"-"+script+"-"+region, lang+"-"+script)
	}
	candidates = append(candidates, lang+"-"+region, lang)
	for _, c := range candidates {
		for _, a := range available {
			if strings.EqualFold(a, c) {
				return a
			}
		}
	}
	// Another variant of the language: pt-PT gets pt-BR. Chinese scripts
	// are not variants of each other.
	if lang != "zh" {
		for _, a := range available {
			if baseLanguage(a) == lang {
				return a
			}
		}
	}
	return "en"
}

func baseLanguage(tag string) string {
	l, _, _ := strings.Cut(tag, "-")
	return strings.ToLower(l)
}

// rightToLeft lists the languages written from right to left, for which
// the window is laid out from the right. The plugin has none of them, but
// apps may add them.
var rightToLeft = map[string]bool{"ar": true, "fa": true, "he": true, "ps": true, "ur": true, "yi": true}

// megabytes formats a size in decimal megabytes, as Finder does.
func (t *text) megabytes(n int64) string {
	s := strconv.FormatFloat(float64(n)/1e6, 'f', 1, 64)
	if t.comma {
		s = strings.Replace(s, ".", ",", 1)
	}
	return fmt.Sprintf(t.Megabytes, s)
}
