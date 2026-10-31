// Package i18ngen reads the catalogs of an app's messages, checks them,
// and writes their typed accessors: Go functions in the catalogs'
// directory, which embed and register them, and a TypeScript module that
// formats the same messages in pages with mygo-runtime. `mygo generate`,
// `mygo dev`, `mygo build` and `mygo i18n check` use it, and so does
// internal/l10n for MyGo's own strings.
package i18ngen

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"math"
	"path"
	"regexp"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/egoist/mygo/i18n"
)

// Options say what catalogs to read.
type Options struct {
	// Dir names the directory of the catalogs in diagnostics, as
	// "locales".
	Dir string
	// Source is the language the messages are written in first, whose
	// file has every key and the metadata (default "en").
	Source string
	// Namespace is that of a library's catalog, as "mygo" for MyGo's own
	// strings, "" for an app's.
	Namespace string
	// Libraries are the catalogs of libraries whose messages an app's
	// keys "<namespace>.<key>" change, by namespace.
	Libraries map[string]*Catalog
	// Translations are the languages MyGo's own strings are translated
	// to, each in a package of github.com/egoist/mygo/locales, which the
	// accessors of an app's catalogs import for the app's languages.
	Translations []string
	// Own is set for MyGo's own strings (internal/l10n), whose accessors
	// embed messages/, the catalog of the source language without its
	// metadata (MessagesJSON), and make the catalog through
	// internal/i18nlink, which links no formatting of dates: every app
	// carries them.
	Own bool
}

// Catalog is the catalogs of the languages of an app or a library.
type Catalog struct {
	Source    string
	Namespace string
	// Languages are the languages of the files: the source first, then
	// the others sorted.
	Languages []string
	// Keys are the messages of the source language, in the order of its
	// file.
	Keys []*Key
	// Files are the files by language.
	Files map[string]*File
	// Translations are the translations of MyGo's own strings to the
	// app's languages, which its accessors import: those of
	// Options.Translations that suit a language of the catalogs best.
	Translations []string
	// Own is Options.Own.
	Own bool
}

// File is the catalog of one language.
type File struct {
	Tag  string
	Name string // as diagnostics name it, "locales/de.json"
	// Keys are the keys of its messages in the order they come, those of
	// libraries ("mygo.cancel") too.
	Keys     []string
	Messages map[string]string
	// Parsed holds the messages that parse.
	Parsed map[string]*i18n.Message
	Lines  map[string]int // of each key, metadata too
}

// Key is a message of the source language, which the accessors say.
type Key struct {
	Name        string
	Message     string
	Description string
	Params      []Param
	// GoName is the name of its Go function: Name, exported.
	GoName string
}

// Param is an argument of a message: a parameter of its accessors.
type Param struct {
	Name string // in the message
	// Type is one of ARB's: String, int, double, num, DateTime or Object.
	Type string
	// Go and TS are its names in the accessors.
	Go, TS string
}

// Types maps the types of placeholders to Go and TypeScript.
var types = map[string][2]string{
	"String":   {"string", "string"},
	"int":      {"int", "number"},
	"double":   {"float64", "number"},
	"num":      {"float64", "number"},
	"DateTime": {"time.Time", "Date"},
	"Object":   {"any", "unknown"},
}

// packagingKeys are keys of the namespace mygo that no library message
// has: the name and the description of the app in a language, which
// packaging writes into the files that list the app.
var packagingKeys = []string{"appName", "appDescription"}

// Diagnostic is a problem of a catalog.
type Diagnostic struct {
	File    string
	Line    int // 0 when it is the whole file's
	Warning bool
	Msg     string
}

func (d Diagnostic) String() string {
	pos := d.File
	if d.Line > 0 {
		pos += fmt.Sprintf(":%d", d.Line)
	}
	if d.Warning {
		return pos + ": warning: " + d.Msg
	}
	return pos + ": " + d.Msg
}

// Diagnostics are the problems of catalogs, in the order of their files.
type Diagnostics []Diagnostic

// Errors counts the problems that are not warnings.
func (ds Diagnostics) Errors() int {
	n := 0
	for _, d := range ds {
		if !d.Warning {
			n++
		}
	}
	return n
}

// Err returns the errors as one error, or nil without any.
func (ds Diagnostics) Err() error {
	var lines []string
	for _, d := range ds {
		if !d.Warning {
			lines = append(lines, d.String())
		}
	}
	if len(lines) == 0 {
		return nil
	}
	return fmt.Errorf("%s", strings.Join(lines, "\n"))
}

// Warnings returns the warnings.
func (ds Diagnostics) Warnings() Diagnostics {
	var out Diagnostics
	for _, d := range ds {
		if d.Warning {
			out = append(out, d)
		}
	}
	return out
}

// keyRe is what a key is: a name of letters, digits and _ that starts
// with a letter, as Flutter's gen_l10n has them, which accessors are
// named after.
var keyRe = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_]*$`)

// Load reads the catalogs at the root of fsys, "<tag>.json", and checks
// them. It fails only when it cannot read them; problems of their contents
// are its diagnostics, and a catalog with errors is still returned, with
// what could be read.
func Load(fsys fs.FS, opts Options) (*Catalog, Diagnostics, error) {
	if opts.Source == "" {
		opts.Source = "en"
	}
	l := &loader{opts: opts, c: &Catalog{Namespace: opts.Namespace, Own: opts.Own, Files: map[string]*File{}}}
	src := i18n.Normalize(opts.Source)
	if src == "" || src != opts.Source || i18n.IsPseudo(src) {
		l.errorf(opts.Dir, 0, "the source language %q is not a BCP 47 tag in its canonical form (as en, de-CH or zh-Hans)", opts.Source)
		return l.c, l.diags, nil
	}
	l.c.Source = src
	names, err := fs.Glob(fsys, "*.json")
	if err != nil {
		return nil, nil, err
	}
	for _, name := range names {
		tag := strings.TrimSuffix(name, ".json")
		display := path.Join(opts.Dir, name)
		if norm := i18n.Normalize(tag); norm != tag {
			hint := ""
			if norm != "" {
				hint = fmt.Sprintf(": rename it %s.json", norm)
			}
			l.errorf(display, 0, "the name of a catalog is a BCP 47 tag in its canonical form, as en.json or zh-Hant.json%s", hint)
			continue
		}
		if i18n.IsPseudo(tag) {
			l.errorf(display, 0, "%s is a pseudo locale, made from the source language: remove the file", tag)
			continue
		}
		data, err := fs.ReadFile(fsys, name)
		if err != nil {
			return nil, nil, err
		}
		if f := l.readFile(display, tag, data); f != nil {
			l.c.Files[tag] = f
		}
	}
	srcFile := l.c.Files[src]
	if srcFile == nil {
		l.errorf(path.Join(opts.Dir, src+".json"), 0, "missing: it holds the messages of the source language, %s", src)
		return l.c, l.sorted(), nil
	}
	for tag := range l.c.Files {
		if tag != src {
			l.c.Languages = append(l.c.Languages, tag)
		}
	}
	slices.Sort(l.c.Languages)
	l.c.Languages = append([]string{src}, l.c.Languages...)
	l.sourceKeys(srcFile)
	for _, tag := range l.c.Languages {
		l.checkFile(l.c.Files[tag])
		if m := i18n.Match([]string{tag}, opts.Translations); m != "" && !slices.Contains(l.c.Translations, m) {
			l.c.Translations = append(l.c.Translations, m)
		}
	}
	slices.Sort(l.c.Translations)
	return l.c, l.sorted(), nil
}

// sorted returns the diagnostics by file, then by line, those of a whole
// file last.
func (l *loader) sorted() Diagnostics {
	slices.SortStableFunc(l.diags, func(a, b Diagnostic) int {
		if c := strings.Compare(a.File, b.File); c != 0 {
			return c
		}
		la, lb := a.Line, b.Line
		if la == 0 {
			la = math.MaxInt
		}
		if lb == 0 {
			lb = math.MaxInt
		}
		return la - lb
	})
	return l.diags
}

type loader struct {
	opts  Options
	c     *Catalog
	diags Diagnostics
	meta  map[string]json.RawMessage // of the source's keys
}

func (l *loader) errorf(file string, line int, format string, args ...any) {
	l.diags = append(l.diags, Diagnostic{File: file, Line: line, Msg: fmt.Sprintf(format, args...)})
}

func (l *loader) warnf(file string, line int, format string, args ...any) {
	l.diags = append(l.diags, Diagnostic{File: file, Line: line, Warning: true, Msg: fmt.Sprintf(format, args...)})
}

// readFile reads a catalog: an object of messages, metadata and
// @@-attributes, in order.
func (l *loader) readFile(name, tag string, data []byte) *File {
	f := &File{Tag: tag, Name: name, Messages: map[string]string{}, Parsed: map[string]*i18n.Message{}, Lines: map[string]int{}}
	dec := json.NewDecoder(bytes.NewReader(data))
	line := func() int { return 1 + bytes.Count(data[:dec.InputOffset()], []byte("\n")) }
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		l.errorf(name, 0, "not a JSON object of messages")
		return nil
	}
	meta := map[string]json.RawMessage{}
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			l.errorf(name, line(), "%v", err)
			return nil
		}
		key := tok.(string)
		at := line()
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			l.errorf(name, at, "%v", err)
			return nil
		}
		if _, dup := f.Lines[key]; dup {
			l.errorf(name, at, "%q comes twice", key)
			continue
		}
		f.Lines[key] = at
		switch {
		case key == "@@locale":
			var locale string
			if json.Unmarshal(raw, &locale) != nil || i18n.Normalize(locale) != tag {
				l.errorf(name, at, "@@locale %s is not %s, the language the file is named after", raw, tag)
			}
		case strings.HasPrefix(key, "@@"):
			// Other attributes of the file, as @@last_modified.
		case strings.HasPrefix(key, "@"):
			meta[key[1:]] = raw
		default:
			var msg string
			if json.Unmarshal(raw, &msg) != nil {
				l.errorf(name, at, "%q is not a string", key)
				continue
			}
			f.Keys = append(f.Keys, key)
			f.Messages[key] = msg
			if m, err := i18n.ParseMessage(msg); err != nil {
				l.errorf(name, at, "%q does not parse: %s", key, strings.TrimPrefix(err.(*i18n.SyntaxError).Err, "i18n: "))
			} else {
				f.Parsed[key] = m
			}
		}
	}
	if _, err := dec.Token(); err != nil {
		l.errorf(name, line(), "%v", err)
		return nil
	}
	if _, err := dec.Token(); err != io.EOF {
		l.errorf(name, line(), "text after the object")
	}
	// Translations may carry metadata, which tools copy; the source's is
	// the one that counts.
	if tag == l.opts.Source {
		l.meta = meta
	}
	return f
}

// sourceKeys reads the keys of the source language and their parameters.
func (l *loader) sourceKeys(f *File) {
	taken := map[string]string{} // Go names
	for _, name := range []string{"In", "Messages", "Catalog"} {
		taken[name] = ""
	}
	for _, key := range f.Keys {
		at := f.Lines[key]
		if l.override(key) {
			continue
		}
		if !keyRe.MatchString(key) {
			l.errorf(f.Name, at, "%q is not a key: a letter, then letters, digits and _, as filesSelected", key)
			continue
		}
		k := &Key{Name: key, Message: f.Messages[key], GoName: goName(key)}
		if other, ok := taken[k.GoName]; ok {
			if other == "" {
				l.errorf(f.Name, at, "%q would name its Go function %s, which the generated code declares", key, k.GoName)
			} else {
				l.errorf(f.Name, at, "%q and %q would both name their Go function %s", other, key, k.GoName)
			}
			continue
		}
		taken[k.GoName] = key
		l.readMeta(f, k)
		l.c.Keys = append(l.c.Keys, k)
	}
	for key := range l.meta {
		if _, ok := f.Messages[key]; !ok {
			l.warnf(f.Name, f.Lines["@"+key], "@%s describes no message", key)
		}
	}
}

// override reports whether key changes a library's message,
// "<namespace>.<key>".
func (l *loader) override(key string) bool {
	ns, _, ok := strings.Cut(key, ".")
	return ok && l.opts.Namespace == "" && ns != "" && keyRe.MatchString(ns)
}

// readMeta reads the description and the placeholders of a key, and its
// parameters: the placeholders in their order, then the other arguments
// of the message in theirs, typed by what they declare or how the
// message uses them.
func (l *loader) readMeta(f *File, k *Key) {
	at := f.Lines["@"+k.Name]
	if at == 0 {
		at = f.Lines[k.Name]
	}
	var declared []string
	declaredType := map[string]string{}
	if raw, ok := l.meta[k.Name]; ok {
		var meta struct {
			Description  string          `json:"description"`
			Placeholders json.RawMessage `json:"placeholders"`
		}
		if err := json.Unmarshal(raw, &meta); err != nil {
			l.errorf(f.Name, at, "@%s: %v", k.Name, err)
		}
		k.Description = meta.Description
		if len(meta.Placeholders) > 0 && string(meta.Placeholders) != "null" {
			names, values, err := orderedObject(meta.Placeholders)
			if err != nil {
				l.errorf(f.Name, at, "@%s.placeholders: %v", k.Name, err)
			}
			for _, name := range names {
				var p struct {
					Type string `json:"type"`
				}
				if err := json.Unmarshal(values[name], &p); err != nil {
					l.errorf(f.Name, at, "@%s.placeholders.%s: %v", k.Name, name, err)
					continue
				}
				if p.Type != "" {
					if _, ok := types[p.Type]; !ok {
						l.errorf(f.Name, at, "@%s.placeholders.%s: type %q is not String, int, double, num, DateTime or Object", k.Name, name, p.Type)
						p.Type = ""
					}
				}
				declared = append(declared, name)
				declaredType[name] = p.Type
			}
		}
	}
	m := f.Parsed[k.Name]
	var args []i18n.Arg
	if m != nil {
		args = m.Args()
	}
	kinds := map[string]string{}
	for _, a := range args {
		kinds[a.Name] = a.Kind
	}
	names := slices.Clone(declared)
	for _, a := range args {
		if !slices.Contains(names, a.Name) {
			names = append(names, a.Name)
		}
	}
	for _, name := range names {
		kind, used := kinds[name]
		if !used && m != nil {
			l.warnf(f.Name, at, "@%s declares the placeholder %s, which the message does not use", k.Name, name)
		}
		typ := declaredType[name]
		if typ == "" {
			typ = inferType(kind)
		} else if !compatible(typ, kind) {
			l.errorf(f.Name, f.Lines[k.Name], "%q uses %s as a %s argument, which its type %s cannot be", k.Name, name, kind, typ)
		}
		k.Params = append(k.Params, Param{Name: name, Type: typ, Go: goParam(name), TS: tsParam(name)})
	}
	if m != nil {
		l.checkPlurals(f, k.Name, m)
	}
}

// inferType is the type of an argument without a declared one: a number
// for plural, selectordinal and number arguments, a time for date and
// time ones, a string otherwise.
func inferType(kind string) string {
	switch kind {
	case "plural", "selectordinal", "number":
		return "int"
	case "date", "time":
		return "DateTime"
	}
	return "String"
}

// compatible reports whether an argument of type typ can be used as an
// argument of kind.
func compatible(typ, kind string) bool {
	switch kind {
	case "plural", "selectordinal", "number":
		return typ == "int" || typ == "double" || typ == "num" || typ == "Object"
	case "date", "time":
		return typ == "DateTime" || typ == "Object"
	case "select":
		return typ == "String" || typ == "Object"
	}
	return true
}

// checkFile checks a file against the source language: its keys, their
// arguments and their plural forms.
func (l *loader) checkFile(f *File) {
	src := l.c.Files[l.c.Source]
	params := map[string][]Param{}
	for _, k := range l.c.Keys {
		params[k.Name] = k.Params
	}
	for _, key := range f.Keys {
		at := f.Lines[key]
		if l.override(key) {
			l.checkOverride(f, key)
			continue
		}
		if f == src {
			continue
		}
		ps, ok := params[key]
		if !ok {
			if _, inSource := src.Messages[key]; inSource {
				continue // a bad key, reported in the source
			}
			l.errorf(f.Name, at, "%q is not a key of %s", key, src.Name)
			continue
		}
		if m := f.Parsed[key]; m != nil {
			l.checkArgs(f, key, m, ps)
			l.checkPlurals(f, key, m)
		}
	}
	if f == src {
		return
	}
	var missing []string
	for _, k := range l.c.Keys {
		if _, ok := f.Messages[k.Name]; !ok {
			missing = append(missing, k.Name)
		}
	}
	if len(missing) > 0 {
		list := strings.Join(missing, ", ")
		if len(missing) > 8 {
			list = strings.Join(missing[:8], ", ") + fmt.Sprintf(" and %d more", len(missing)-8)
		}
		l.warnf(f.Name, 0, "%d of %d messages missing, which say the language they fall back to: %s", len(missing), len(l.c.Keys), list)
	}
}

// checkArgs checks the arguments of a translation against the parameters
// of its key.
func (l *loader) checkArgs(f *File, key string, m *i18n.Message, params []Param) {
	at := f.Lines[key]
	args := m.Args()
	for _, a := range args {
		i := slices.IndexFunc(params, func(p Param) bool { return p.Name == a.Name })
		if i < 0 {
			var names []string
			for _, p := range params {
				names = append(names, p.Name)
			}
			have := "no arguments"
			if len(names) > 0 {
				have = "the arguments " + strings.Join(names, ", ")
			}
			l.errorf(f.Name, at, "%q uses the argument %s, but the source message has %s", key, a.Name, have)
			continue
		}
		if !compatible(params[i].Type, a.Kind) {
			l.errorf(f.Name, at, "%q uses %s as a %s argument, which its type %s cannot be", key, a.Name, a.Kind, params[i].Type)
		}
	}
	for _, p := range params {
		if !slices.ContainsFunc(args, func(a i18n.Arg) bool { return a.Name == p.Name }) {
			l.warnf(f.Name, at, "%q does not use the argument %s", key, p.Name)
		}
	}
}

// checkPlurals warns of the cases of plural and selectordinal arguments
// that the language's rules do not have, and of those it lacks.
func (l *loader) checkPlurals(f *File, key string, m *i18n.Message) {
	at := f.Lines[key]
	for _, a := range m.Args() {
		if a.Kind != "plural" && a.Kind != "selectordinal" {
			continue
		}
		var have []i18n.Plural
		kind := "plural"
		if a.Kind == "plural" {
			have = i18n.PluralCategories(f.Tag)
		} else {
			kind = "ordinal"
			have = i18n.OrdinalCategories(f.Tag)
		}
		cases := m.Cases(a.Name)
		var extra, lacking []string
		for _, c := range cases {
			if p, ok := i18n.ParsePlural(c); ok && !slices.Contains(have, p) {
				extra = append(extra, c)
			}
		}
		for _, p := range have {
			if !slices.Contains(cases, p.String()) {
				lacking = append(lacking, p.String())
			}
		}
		if len(extra) > 0 {
			l.warnf(f.Name, at, "%q: %s has no %s %s of %s (its %s categories are %s)", key, f.Tag, kind, plural(len(extra), "category", "categories"), strings.Join(extra, ", "), kind, names(have))
		}
		if len(lacking) > 0 {
			l.warnf(f.Name, at, "%q: the %s argument %s lacks %s, a %s category of %s, which says other in its place", key, a.Kind, a.Name, strings.Join(lacking, ", "), kind, f.Tag)
		}
	}
}

func plural(n int, one, other string) string {
	if n == 1 {
		return one
	}
	return other
}

func names(ps []i18n.Plural) string {
	var out []string
	for _, p := range ps {
		out = append(out, p.String())
	}
	return strings.Join(out, ", ")
}

// checkOverride checks a key that changes a library's message.
func (l *loader) checkOverride(f *File, key string) {
	at := f.Lines[key]
	ns, name, _ := strings.Cut(key, ".")
	lib, ok := l.opts.Libraries[ns]
	if !ok {
		l.errorf(f.Name, at, "%q: no library has the namespace %s", key, ns)
		return
	}
	var params []Param
	if i := slices.IndexFunc(lib.Keys, func(k *Key) bool { return k.Name == name }); i >= 0 {
		params = lib.Keys[i].Params
	} else if !(ns == "mygo" && slices.Contains(packagingKeys, name)) {
		l.errorf(f.Name, at, "%q: %s has no message %s", key, ns, name)
		return
	}
	if m := f.Parsed[key]; m != nil {
		l.checkArgs(f, key, m, params)
		l.checkPlurals(f, key, m)
	}
}

// orderedObject reads a JSON object, keeping the order of its keys.
func orderedObject(data []byte) ([]string, map[string]json.RawMessage, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return nil, nil, fmt.Errorf("not an object")
	}
	var keys []string
	values := map[string]json.RawMessage{}
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, nil, err
		}
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return nil, nil, err
		}
		key := tok.(string)
		if _, dup := values[key]; !dup {
			keys = append(keys, key)
		}
		values[key] = raw
	}
	return keys, values, nil
}

// goName is the name of the Go function of a key: its words, split at _,
// capitalized.
func goName(key string) string {
	var b strings.Builder
	for _, w := range strings.Split(key, "_") {
		if w == "" {
			continue
		}
		r, size := utf8.DecodeRuneInString(w)
		b.WriteRune(unicode.ToUpper(r))
		b.WriteString(w[size:])
	}
	return b.String()
}

var goKeywords = strings.Fields(`break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var`)

// goReserved are the names the generated functions use, and the
// predeclared identifiers they name, which a parameter would shadow.
var goReserved = []string{"msgs", "i18n", "In", "Catalog", "time", "string", "any", "int", "float64", "map", "nil", "true", "false"}

func goParam(name string) string {
	if r, _ := utf8.DecodeRuneInString(name); unicode.IsDigit(r) {
		return "arg" + name
	}
	if slices.Contains(goKeywords, name) || slices.Contains(goReserved, name) {
		return name + "_"
	}
	return name
}

var tsReserved = strings.Fields(`arguments await break case catch class const continue debugger default delete do else enum eval export extends false finally for function if implements import in instanceof interface let new null package private protected public return static super switch this throw true try typeof var void while with yield`)

func tsParam(name string) string {
	if r, _ := utf8.DecodeRuneInString(name); unicode.IsDigit(r) {
		return "arg" + name
	}
	if slices.Contains(tsReserved, name) {
		return name + "_"
	}
	return name
}
