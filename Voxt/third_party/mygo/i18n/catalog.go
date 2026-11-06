package i18n

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"maps"
	"path"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/egoist/mygo/internal/i18nlink"
)

// A Catalog holds the messages of an app, or of a library, in the
// languages it is translated to: a file of ARB-style JSON per language,
// named after its BCP 47 tag, "en.json", "zh-Hant.json", whose keys name
// messages of ICU MessageFormat (see Message):
//
//	{
//	  "@@locale": "en",
//	  "filesSelected": "{count, plural, one {# file selected} other {# files selected}}",
//	  "@filesSelected": {
//	    "description": "The status bar, while files are selected.",
//	    "placeholders": {"count": {"type": "int"}}
//	  }
//	}
//
// A key starting with @ holds metadata, which only the file of the source
// language (the one the app is written in) needs. `mygo generate` writes
// typed functions for the keys of an app's catalog, which register it
// (Register), so that an app rarely uses a Catalog itself.
//
// A Catalog reads the file of a language when it first formats a message
// in it, and keeps it. It is safe for concurrent use.
type Catalog struct {
	source    string
	namespace string // of a library's catalog
	// set holds the languages, replaced whole as Add adds some, so that
	// formatting reads it without a lock.
	set   atomic.Pointer[langSet]
	addMu sync.Mutex

	candMu sync.Mutex
	cands  map[string]candidates // tags to look a key up in, by locale

	// texts caches the messages formatted without arguments, by locale
	// and key, as of a version of the registry: the controls of native
	// UI say theirs every frame.
	textMu       sync.RWMutex
	texts        map[textKey]string
	textsVersion uint64
}

type textKey struct{ locale, key string }

// langSet is the languages of a catalog and their messages.
type langSet struct {
	langs []string             // source first, then the others sorted
	data  map[string]*langMsgs // by tag
}

// langMsgs are the messages of a catalog in one language: the file they
// are in, read once.
type langMsgs struct {
	fsys fs.FS
	file string
	once sync.Once
	msgs map[string]*Message
}

// langs returns the languages of the catalog, source first.
func (c *Catalog) langs() []string { return c.set.Load().langs }

// candidates are the tags a locale looks messages up in, as of a version
// of the registry.
type candidates struct {
	version uint64
	tags    []string
}

// NewCatalog returns the catalog of the JSON files at the root of fsys,
// "<tag>.json", whose source language, the one its messages are written
// in first, is source: the language whose messages the others fall back
// to. Their names must be BCP 47 tags in their canonical form; their
// contents are read as they are needed. A message that does not parse, or
// a file that does not, is left out, and its keys fall back as missing ones
// do: `mygo generate` and `mygo i18n check` report them.
func NewCatalog(fsys fs.FS, source string) (*Catalog, error) {
	linkTimes()
	return newCatalog(fsys, source)
}

// newCatalog is NewCatalog without linking the formatting of dates.
func newCatalog(fsys fs.FS, source string) (*Catalog, error) {
	src := Normalize(source)
	if src == "" || IsPseudo(src) {
		return nil, fmt.Errorf("i18n: the source language %q is not a language", source)
	}
	files, err := catalogFiles(fsys)
	if err != nil {
		return nil, err
	}
	if files[src] == nil {
		return nil, fmt.Errorf("i18n: the catalog has no %s.json, the messages of its source language", src)
	}
	c := &Catalog{source: src}
	set := &langSet{data: files}
	for tag := range files {
		if tag != src {
			set.langs = append(set.langs, tag)
		}
	}
	slices.Sort(set.langs)
	set.langs = append([]string{src}, set.langs...)
	c.set.Store(set)
	return c, nil
}

// catalogFiles returns the languages of the JSON files at the root of
// fsys, named as NewCatalog says.
func catalogFiles(fsys fs.FS) (map[string]*langMsgs, error) {
	names, err := fs.Glob(fsys, "*.json")
	if err != nil {
		return nil, err
	}
	files := map[string]*langMsgs{}
	for _, name := range names {
		tag := strings.TrimSuffix(name, ".json")
		norm := Normalize(tag)
		if norm == "" || norm != tag {
			return nil, fmt.Errorf("i18n: %s is not named after a BCP 47 tag, as en.json or zh-Hant.json", name)
		}
		if IsPseudo(norm) {
			return nil, fmt.Errorf("i18n: %s: the pseudo locales come from the source language and have no files", name)
		}
		files[norm] = &langMsgs{fsys: fsys, file: name}
	}
	return files, nil
}

// Add adds the languages of the JSON files at the root of fsys, named as
// NewCatalog's, to the catalog: translations kept apart from it, as those
// of MyGo's own strings are, each in a package an app imports for a
// language it ships. It fails on a language the catalog has already, and
// on files not named after a language; an app's registered catalog ships
// the languages it adds. Add is safe for concurrent use with formatting,
// which says the new languages from then on.
func (c *Catalog) Add(fsys fs.FS) error {
	files, err := catalogFiles(fsys)
	if err != nil {
		return err
	}
	c.addMu.Lock()
	old := c.set.Load()
	set := &langSet{data: maps.Clone(old.data)}
	var added []string
	for tag, f := range files {
		if old.data[tag] != nil {
			c.addMu.Unlock()
			return fmt.Errorf("i18n: the catalog has %s already", tag)
		}
		set.data[tag] = f
		added = append(added, tag)
	}
	set.langs = append(slices.Clone(old.langs[1:]), added...)
	slices.Sort(set.langs)
	set.langs = append([]string{c.source}, set.langs...)
	c.set.Store(set)
	c.addMu.Unlock()

	registry.mu.Lock()
	isRegistered := slices.Contains(registry.catalogs, c)
	registry.version.Add(1) // the languages to look messages up in changed
	registry.mu.Unlock()
	if isRegistered {
		slices.Sort(added)
		declare(c.source, added)
	}
	return nil
}

// NewLibraryCatalog returns the catalog of a library, as NewCatalog does,
// whose messages an app changes, or translates to other languages, with
// keys of its own catalogs named "<namespace>.<key>": MyGo's own strings,
// those of its controls and menus, are a library's catalog whose
// namespace is "mygo", and an app's de.json that says
//
//	"mygo.cancel": "Abbrechen"
//
// translates the controls' "cancel" to German.
//
// A library's messages are said in the language the app shows (Locale),
// and fall back along its parents, then to the source language of the
// app, then to the library's. A library's catalog is not registered: the
// languages an app ships are its own.
func NewLibraryCatalog(fsys fs.FS, source, namespace string) (*Catalog, error) {
	linkTimes()
	return newLibraryCatalog(fsys, source, namespace)
}

// newLibraryCatalog is NewLibraryCatalog without linking the formatting
// of dates, for MyGo's own strings (internal/i18nlink), which have none.
func newLibraryCatalog(fsys fs.FS, source, namespace string) (*Catalog, error) {
	if namespace == "" || strings.ContainsAny(namespace, ".@ ") {
		return nil, fmt.Errorf("i18n: %q is not a namespace", namespace)
	}
	c, err := newCatalog(fsys, source)
	if err != nil {
		return nil, err
	}
	c.namespace = namespace
	return c, nil
}

// Source returns the source language of the catalog.
func (c *Catalog) Source() string { return c.source }

// Languages returns the languages of the catalog: its source language,
// then the others, sorted.
func (c *Catalog) Languages() []string { return slices.Clone(c.langs()) }

// Namespace returns the namespace under which apps change the messages of
// a library's catalog, or "" for an app's.
func (c *Catalog) Namespace() string { return c.namespace }

// Format formats the message of key in locale with args (see
// Message.Format), or returns key when the catalog has no such message.
//
// The message is the one of the catalog's language that best suits
// locale (Match), as "zh-Hant" for "zh-TW", or, when that language lacks
// it, of the language's parents (Parent), as "en-001" then "en" for
// "en-AU", else of the source language. Its cases follow the plural rules
// of the language it is in; numbers and dates, the conventions of locale.
// "" is the source language; the pseudo locales en-XA and ar-XB transform
// the source language's messages (IsPseudo).
//
// A message formatted without arguments is kept, so that saying it again
// in the same locale costs a map lookup and allocates nothing.
func (c *Catalog) Format(locale, key string, args map[string]any) string {
	if len(args) == 0 {
		version := registry.version.Load()
		c.textMu.RLock()
		s, ok := c.texts[textKey{locale, key}]
		ok = ok && c.textsVersion == version
		c.textMu.RUnlock()
		if ok {
			return s
		}
		s = c.format(locale, key, nil)
		c.textMu.Lock()
		if c.texts == nil || c.textsVersion != version || len(c.texts) >= 4096 {
			c.texts, c.textsVersion = map[textKey]string{}, version
		}
		c.texts[textKey{locale, key}] = s
		c.textMu.Unlock()
		return s
	}
	return c.format(locale, key, args)
}

func (c *Catalog) format(locale, key string, args map[string]any) string {
	m, lang, ok := c.lookup(locale, key)
	if !ok {
		return key
	}
	if locale == "" {
		locale = lang
	}
	return m.format(locale, lang, args)
}

// Message returns the message of key that Format formats in locale, and
// the language it is in.
func (c *Catalog) Message(locale, key string) (m *Message, lang string, ok bool) {
	return c.lookup(locale, key)
}

// lookup finds the message of key for locale among the catalog's and,
// for a library's, the overrides of the registered catalogs.
func (c *Catalog) lookup(locale, key string) (*Message, string, bool) {
	sources := c.sources()
	for _, tag := range c.candidates(locale, sources) {
		for _, s := range sources {
			if m := s.catalog.message(tag, s.prefix+key); m != nil {
				return m, tag, true
			}
		}
	}
	return nil, "", false
}

// lookupSource is a catalog to look a key up in, with the prefix of its
// keys.
type lookupSource struct {
	catalog *Catalog
	prefix  string
}

// sources returns the catalogs a key of c is looked up in: c, after the
// registered catalogs for a library's.
func (c *Catalog) sources() []lookupSource {
	if c.namespace == "" {
		return []lookupSource{{c, ""}}
	}
	var out []lookupSource
	for _, r := range registered() {
		out = append(out, lookupSource{r, c.namespace + "."})
	}
	return append(out, lookupSource{c, ""})
}

// candidates returns the tags to look a key up in for locale, in order.
func (c *Catalog) candidates(locale string, sources []lookupSource) []string {
	version := registry.version.Load()
	c.candMu.Lock()
	if cached, ok := c.cands[locale]; ok && cached.version == version {
		c.candMu.Unlock()
		return cached.tags
	}
	c.candMu.Unlock()

	var tags []string
	add := func(list []string) {
		for _, t := range list {
			if !slices.Contains(tags, t) {
				tags = append(tags, t)
			}
		}
	}
	tag := stripExtensions(Normalize(locale))
	if tag != "" && !IsPseudo(tag) {
		add(chain(tag))
		// The language each catalog has that suits locale best, as
		// zh-Hant for zh-TW, whose chain (zh-TW, zh) has none of it.
		for _, s := range sources {
			langs := s.catalog.langs()
			if slices.Contains(langs, tag) {
				continue
			}
			if m := Match([]string{tag}, langs); m != "" {
				add(chain(m))
			}
		}
	}
	// The source language of the app, then the library's.
	if len(sources) > 1 {
		add(chain(sources[0].catalog.source))
	}
	add(chain(c.source))

	c.candMu.Lock()
	if c.cands == nil {
		c.cands = map[string]candidates{}
	}
	if len(c.cands) > 64 {
		clear(c.cands) // locales are few; keep the cache from growing
	}
	c.cands[locale] = candidates{version, tags}
	c.candMu.Unlock()
	return tags
}

// message returns the message of key in the language tag, which the
// catalog has, or nil.
func (c *Catalog) message(tag, key string) *Message {
	l := c.set.Load().data[tag]
	if l == nil {
		return nil
	}
	l.once.Do(func() { l.msgs = l.load() })
	return l.msgs[key]
}

// load reads the messages of a language. A message that does not parse is
// left out, as mygo generate reports it: its key falls back.
func (l *langMsgs) load() map[string]*Message {
	data, err := fs.ReadFile(l.fsys, path.Clean(l.file))
	if err != nil {
		return nil
	}
	var raw map[string]json.RawMessage
	if json.Unmarshal(data, &raw) != nil {
		return nil
	}
	msgs := make(map[string]*Message, len(raw))
	for key, v := range raw {
		if strings.HasPrefix(key, "@") {
			continue
		}
		var s string
		if json.Unmarshal(v, &s) != nil {
			continue
		}
		if m, err := parseMessage(s); err == nil {
			msgs[key] = m
		}
	}
	return msgs
}

// registry holds the catalogs apps registered.
var registry struct {
	mu       sync.Mutex
	catalogs []*Catalog
	version  atomic.Uint64 // counts registrations, for caches
}

// Register adds an app's catalog to those the app ships, which the
// TypeScript and Go code `mygo generate` writes for it does: its
// languages become the app's (Languages), and its keys "<namespace>.<key>"
// change the messages of libraries, as MyGo's own. A library's catalog is
// not registered.
func Register(c *Catalog) {
	if c.namespace != "" {
		panic("i18n: Register of the catalog of a library, namespace " + c.namespace)
	}
	registry.mu.Lock()
	if slices.Contains(registry.catalogs, c) {
		registry.mu.Unlock()
		return
	}
	registry.catalogs = append(registry.catalogs, c)
	registry.version.Add(1)
	registry.mu.Unlock()
	declare(c.source, c.langs()[1:])
}

// registered returns the registered catalogs, the first first.
func registered() []*Catalog {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	return slices.Clone(registry.catalogs)
}

// localeFunc returns the language the app shows.
var localeFunc atomic.Pointer[func() string]

// SetLocaleFunc sets the function that returns the language the app shows,
// in which Locale, and so the functions `mygo generate` writes, say their
// messages. Package mygo sets it to its App.Locale; apps change the
// language with mygo.App.SetLocale, not here. f must be safe to call from
// any goroutine.
func SetLocaleFunc(f func() string) {
	if f == nil {
		localeFunc.Store(nil)
		return
	}
	localeFunc.Store(&f)
}

// Locale returns the language the app shows, as the function SetLocaleFunc
// set returns it, or "" before it is set, which catalogs read as their
// source language.
func Locale() string {
	if f := localeFunc.Load(); f != nil {
		return (*f)()
	}
	return ""
}

func init() {
	i18nlink.NewLibraryCatalog = func(fsys fs.FS, source, namespace string) (any, error) {
		return newLibraryCatalog(fsys, source, namespace)
	}
}
