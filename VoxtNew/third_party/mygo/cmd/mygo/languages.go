package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/egoist/mygo/i18n"
)

// The languages a build declares to the systems, which they are in where
// the system's own texts are concerned: the localizations of an app
// bundle, which AppKit says its menus and panels in and System Settings
// lists for the app, the names of the desktop entry, the languages of the
// Windows installer and of the file properties of Windows. They are the
// catalogs of the app's strings (i18n.dir), the language the app is
// written in first; an app without them declares none, as before.

// appLanguage is a language the app ships.
type appLanguage struct {
	// Tag is the language's BCP 47 tag, as its catalog's file names it.
	Tag string
	// Name and Description are the app's name and description in the
	// language, the catalog's mygo.appName and mygo.appDescription, or ""
	// where it has none.
	Name, Description string
}

// Keys of an app's catalog that name the app and describe it, in the
// namespace of MyGo's own strings.
const (
	appNameKey        = "mygo.appName"
	appDescriptionKey = "mygo.appDescription"
)

// catalogDir returns the directory of the catalogs of the app's strings
// and the language the app is written in, or "" without translations.
func (c *Config) catalogDir() (dir, source string) {
	if c.I18n == nil {
		return "", ""
	}
	dir, source = c.I18n.Dir, c.I18n.Source
	if dir == "" {
		dir = "locales"
	}
	if source == "" {
		source = "en"
	}
	return filepath.Join(c.root, dir), source
}

// languages returns the languages the app ships, its own first, then those
// of the other catalogs by tag, or none for an app without translations.
// It reads the catalogs once.
func (c *Config) languages() ([]appLanguage, error) {
	if c.langsRead {
		return c.langs, c.langsErr
	}
	c.langsRead = true
	c.langs, c.langsErr = readLanguages(c.catalogDir())
	return c.langs, c.langsErr
}

// readLanguages reads the catalogs <tag>.json of dir: the tags of their
// names, source first, and the app's name and description in each.
func readLanguages(dir, source string) ([]appLanguage, error) {
	if dir == "" {
		return nil, nil
	}
	if i18n.Normalize(source) == "" {
		return nil, fmt.Errorf("i18n.source: %q is not a language tag, such as en, zh-Hans or pt-BR", source)
	}
	files, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, err
	}
	langs := []appLanguage{{Tag: source}}
	for _, f := range files {
		tag := strings.TrimSuffix(filepath.Base(f), ".json")
		if i18n.Normalize(tag) == "" {
			return nil, fmt.Errorf("%s: %q is not a language tag, such as en, zh-Hans or pt-BR", f, tag)
		}
		var catalog map[string]any
		data, err := os.ReadFile(f)
		if err == nil {
			err = json.Unmarshal(data, &catalog)
		}
		if err != nil {
			return nil, fmt.Errorf("%s: %w", f, err)
		}
		l := appLanguage{Tag: tag}
		l.Name, _ = catalog[appNameKey].(string)
		l.Description, _ = catalog[appDescriptionKey].(string)
		if strings.EqualFold(i18n.Normalize(tag), i18n.Normalize(source)) {
			// The app's own name and description are the configuration's.
			l.Name, l.Description = "", ""
		}
		for _, v := range []string{l.Name, l.Description} {
			if strings.ContainsAny(v, "\r\n") {
				return nil, fmt.Errorf("%s: the app's name and description cannot break the line", f)
			}
		}
		if i := slices.IndexFunc(langs, func(a appLanguage) bool { return strings.EqualFold(i18n.Normalize(a.Tag), i18n.Normalize(tag)) }); i >= 0 {
			langs[i] = l
			continue
		}
		langs = append(langs, l)
	}
	slices.SortStableFunc(langs[1:], func(a, b appLanguage) int { return strings.Compare(a.Tag, b.Tag) })
	return langs, nil
}

// lprojName returns the name of the localization folder of a tag in an
// app bundle, as Xcode names them: the language, with its script or its
// region, as zh-Hans, sr-Latn and pt-BR. Chinese is named by its script,
// zh-Hans or zh-Hant, but Hong Kong's, zh-HK; a tag of a script and a
// region keeps the script.
func lprojName(tag string) string {
	lang, script, region := tagParts(tag)
	if lang == "zh" {
		if region == "HK" {
			return "zh-HK"
		}
		if script == "Hant" {
			return "zh-Hant"
		}
		return "zh-Hans"
	}
	switch {
	case script != "":
		return lang + "-" + script
	case region != "":
		return lang + "-" + region
	}
	return lang
}

// desktopTag returns the locale of the localized keys of a desktop entry,
// which launchers match against the POSIX locale of the system,
// lang_COUNTRY@modifier: pt-BR is pt_BR, sr-Latn is sr@latin, and Chinese
// is named by the country of its script, zh-Hans zh_CN and zh-Hant zh_TW. A
// region of digits, as es-419's, has no POSIX name and is left out.
func desktopTag(tag string) string {
	lang, script, region := tagParts(tag)
	modifier := ""
	switch {
	case lang == "zh":
		if region == "" {
			region = "CN"
			if script == "Hant" {
				region = "TW"
			}
		}
	case script != "" && script != likelyScript(lang):
		modifier = map[string]string{"Latn": "latin", "Cyrl": "cyrillic", "Arab": "arabic", "Deva": "devanagari"}[script]
	}
	out := lang
	if len(region) == 2 {
		out += "_" + region
	}
	if modifier != "" {
		out += "@" + modifier
	}
	return out
}

// tagParts returns the language of a tag, its script, the most likely one
// for a Chinese region (zh-TW is Hant), and its region.
func tagParts(tag string) (lang, script, region string) {
	parts := strings.Split(i18n.Normalize(tag), "-")
	lang = parts[0]
	for _, p := range parts[1:] {
		switch {
		case len(p) == 4 && script == "" && region == "":
			script = p
		case region == "" && (len(p) == 2 || len(p) == 3 && p[0] >= '0' && p[0] <= '9'):
			region = p
		}
	}
	if lang == "zh" && script == "" {
		script = "Hans"
		if region == "TW" || region == "HK" || region == "MO" {
			script = "Hant"
		}
	}
	return lang, script, region
}

// likelyScript returns the script a language is written in where its
// POSIX locale names no modifier: sr_RS is Cyrillic, sr_RS@latin Latin.
func likelyScript(lang string) string {
	switch lang {
	case "az", "bs", "uz":
		return "Latn"
	case "be", "kk", "mn", "sr", "tg":
		return "Cyrl"
	}
	return ""
}

// windowsLanguage is a language of the Windows resources of the app: the
// ID Windows names it by and the app's name in it.
type windowsLanguage struct {
	id   uint16
	name string
}

// langEnUS is the language of the resources of an app that declares none.
const langEnUS = 0x0409

// windowsLanguages returns the languages of the Windows resources of the
// app: those it ships that Windows has an ID of, its own first, or en-US
// alone with the app's own name without any.
func windowsLanguages(c *Config, langs []appLanguage) []windowsLanguage {
	var out []windowsLanguage
	for _, l := range langs {
		id, ok := windowsLanguageID(l.Tag)
		if !ok {
			warnf("The file properties of the app are not in %s: Windows has no ID of the language", l.Tag)
			continue
		}
		if slices.ContainsFunc(out, func(w windowsLanguage) bool { return w.id == id }) {
			continue
		}
		name := l.Name
		if name == "" {
			name = c.Name
		}
		out = append(out, windowsLanguage{id: id, name: name})
	}
	if len(out) == 0 {
		out = append(out, windowsLanguage{id: langEnUS, name: c.Name})
	}
	return out
}

// windowsLanguageID returns the ID of the language of a tag, as 0x0409 for
// en-US, and whether Windows has one: of the language and region, or of
// the language and script, or of the language alone.
func windowsLanguageID(tag string) (uint16, bool) {
	for _, key := range tagKeys(tag) {
		if id, ok := windowsLanguageIDs[key]; ok {
			return id, true
		}
	}
	return 0, false
}

// tagKeys returns the keys of a tag in the tables of languages, most
// specific first: language-region, language-script, language, lowercased.
func tagKeys(tag string) []string {
	lang, script, region := tagParts(tag)
	lang, script, region = strings.ToLower(lang), strings.ToLower(script), strings.ToLower(region)
	var keys []string
	if region != "" {
		keys = append(keys, lang+"-"+region)
	}
	if script != "" {
		keys = append(keys, lang+"-"+script)
	}
	return append(keys, lang)
}

// nsisLanguage returns the name of the language file of NSIS of a tag,
// which MUI_LANGUAGE takes without its extension, as "SimpChinese", and
// whether NSIS has one.
func nsisLanguage(tag string) (string, bool) {
	for _, key := range tagKeys(tag) {
		if name, ok := nsisLanguages[key]; ok {
			return name, true
		}
	}
	return "", false
}

// nsisLanguageMacros returns the lines that give the Windows installer the
// languages of the app, a MUI_LANGUAGE each, the app's own first, which
// the installer falls back to; it picks the one of the system's language.
// It is English alone for an app that ships no languages. A language NSIS
// has no language file of fails the build, as it would fail makensis.
func nsisLanguageMacros(langs []appLanguage) (string, error) {
	seen := map[string]bool{}
	var b strings.Builder
	for _, l := range langs {
		name, ok := nsisLanguage(l.Tag)
		if !ok {
			return "", fmt.Errorf("the Windows installer cannot be in %s, which NSIS has no language file of; NSIS 3 has those of %s", l.Tag, strings.Join(nsisLanguageList(), ", "))
		}
		if seen[name] {
			continue
		}
		seen[name] = true
		fmt.Fprintf(&b, "!insertmacro MUI_LANGUAGE %s\n", nsisString(name))
	}
	if b.Len() == 0 {
		return "!insertmacro MUI_LANGUAGE \"English\"\n", nil
	}
	return b.String(), nil
}

// nsisLanguageList returns the tags of the languages NSIS has files of.
func nsisLanguageList() []string {
	var tags []string
	for tag := range nsisLanguages {
		tags = append(tags, tag)
	}
	slices.Sort(tags)
	return tags
}

// windowsLanguageIDs are the IDs (LANGIDs) of languages Windows has, by
// language and region, language and script, or language alone, which
// stands for the language's main region.
var windowsLanguageIDs = map[string]uint16{
	"af": 0x0436, "am": 0x045E, "ar": 0x0401, "ar-eg": 0x0C01, "ar-sa": 0x0401,
	"az": 0x042C, "az-latn": 0x042C, "ba": 0x046D, "be": 0x0423, "bg": 0x0402,
	"bn": 0x0445, "bs": 0x141A, "ca": 0x0403, "co": 0x0483, "cs": 0x0405,
	"cy": 0x0452, "da": 0x0406, "de": 0x0407, "de-at": 0x0C07, "de-ch": 0x0807,
	"de-de": 0x0407, "el": 0x0408, "en": 0x0409, "en-au": 0x0C09, "en-ca": 0x1009,
	"en-gb": 0x0809, "en-ie": 0x1809, "en-in": 0x4009, "en-nz": 0x1409, "en-sg": 0x4809,
	"en-us": 0x0409, "en-za": 0x1C09, "es": 0x0C0A, "es-ar": 0x2C0A, "es-cl": 0x340A,
	"es-co": 0x240A, "es-es": 0x0C0A, "es-mx": 0x080A, "es-us": 0x540A, "es-419": 0x580A,
	"et": 0x0425, "eu": 0x042D, "fa": 0x0429, "fi": 0x040B, "fil": 0x0464, "fo": 0x0438,
	"fr": 0x040C, "fr-be": 0x080C, "fr-ca": 0x0C0C, "fr-ch": 0x100C, "fr-fr": 0x040C,
	"ga": 0x083C, "gd": 0x0491, "gl": 0x0456, "gsw": 0x0484, "gu": 0x0447,
	"ha": 0x0468, "he": 0x040D, "hi": 0x0439, "hr": 0x041A, "hsb": 0x042E,
	"hu": 0x040E, "hy": 0x042B, "id": 0x0421, "ig": 0x0470, "ii": 0x0478,
	"is": 0x040F, "it": 0x0410, "it-ch": 0x0810, "it-it": 0x0410, "iu": 0x045D,
	"ja": 0x0411, "ka": 0x0437, "kk": 0x043F, "kl": 0x046F, "km": 0x0453,
	"kn": 0x044B, "ko": 0x0412, "kok": 0x0457, "ky": 0x0440, "lb": 0x046E,
	"lo": 0x0454, "lt": 0x0427, "lv": 0x0426, "mi": 0x0481, "mk": 0x042F,
	"ml": 0x044C, "mn": 0x0450, "moh": 0x047C, "mr": 0x044E, "ms": 0x043E,
	"mt": 0x043A, "nb": 0x0414, "ne": 0x0461, "nl": 0x0413, "nl-be": 0x0813,
	"nl-nl": 0x0413, "nn": 0x0814, "no": 0x0414, "nso": 0x046C, "oc": 0x0482,
	"or": 0x0448, "pa": 0x0446, "pl": 0x0415, "prs": 0x048C, "ps": 0x0463,
	"pt": 0x0416, "pt-br": 0x0416, "pt-pt": 0x0816, "qut": 0x0486, "quz": 0x046B,
	"rm": 0x0417, "ro": 0x0418, "ru": 0x0419, "rw": 0x0487, "sa": 0x044F,
	"sah": 0x0485, "se": 0x043B, "si": 0x045B, "sk": 0x041B, "sl": 0x0424,
	"sq": 0x041C, "sr": 0x281A, "sr-cyrl": 0x281A, "sr-latn": 0x241A, "sv": 0x041D,
	"sw": 0x0441, "ta": 0x0449, "te": 0x044A, "tg": 0x0428, "th": 0x041E,
	"tn": 0x0432, "tr": 0x041F, "tt": 0x0444, "ug": 0x0480, "uk": 0x0422,
	"ur": 0x0420, "uz": 0x0443, "vi": 0x042A, "wo": 0x0488, "xh": 0x0434,
	"yo": 0x046A, "zh": 0x0804, "zh-cn": 0x0804, "zh-hans": 0x0804, "zh-hant": 0x0404,
	"zh-hk": 0x0C04, "zh-mo": 0x1404, "zh-sg": 0x1004, "zh-tw": 0x0404, "zu": 0x0435,
}

// nsisLanguages are the language files every NSIS 3 has, which
// MUI_LANGUAGE takes without the .nlf extension, by language and region,
// language and script, or language alone.
var nsisLanguages = map[string]string{
	"af": "Afrikaans", "ar": "Arabic", "be": "Belarusian", "bg": "Bulgarian",
	"bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh",
	"da": "Danish", "de": "German", "el": "Greek", "en": "English",
	"es": "Spanish", "es-419": "SpanishInternational", "et": "Estonian", "eu": "Basque",
	"fa": "Farsi", "fi": "Finnish", "fr": "French", "ga": "Irish",
	"gl": "Galician", "he": "Hebrew", "hr": "Croatian", "hu": "Hungarian",
	"hy": "Armenian", "id": "Indonesian", "is": "Icelandic", "it": "Italian",
	"ja": "Japanese", "ka": "Georgian", "ko": "Korean", "lb": "Luxembourgish",
	"lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian", "mn": "Mongolian",
	"ms": "Malay", "nb": "Norwegian", "nl": "Dutch", "nn": "NorwegianNynorsk",
	"no": "Norwegian", "pl": "Polish", "pt": "PortugueseBR", "pt-br": "PortugueseBR",
	"pt-pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
	"sl": "Slovenian", "sq": "Albanian", "sr": "Serbian", "sr-latn": "SerbianLatin",
	"sv": "Swedish", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian",
	"uz": "Uzbek", "vi": "Vietnamese", "zh-hans": "SimpChinese", "zh-hant": "TradChinese",
}
