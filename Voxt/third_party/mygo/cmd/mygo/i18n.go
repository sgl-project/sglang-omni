package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/egoist/mygo/i18n"
	"github.com/egoist/mygo/internal/i18ngen"
	"github.com/egoist/mygo/internal/l10n"
)

func runI18n(args []string) error {
	if len(args) == 0 || args[0] != "check" {
		fmt.Fprint(os.Stderr, `Usage: mygo i18n check [dir]

Checks the catalogs of the app's messages (i18n in mygo.json): that their
messages parse, that translations have the arguments of the source
language's messages and the plural forms of their languages, and which
messages they lack. Errors exit with status 1; warnings do not.
`)
		if len(args) > 0 && (args[0] == "-h" || args[0] == "--help" || args[0] == "help") {
			return nil
		}
		return fmt.Errorf("mygo i18n: unknown subcommand; want check")
	}
	flags := newFlags("i18n check", "[dir]", "Checks the catalogs of the app's messages (i18n in mygo.json).")
	if err := flags.Parse(args[1:]); err != nil {
		return err
	}
	c, err := loadConfig(dirArg(flags.Args()))
	if err != nil {
		return err
	}
	if c.I18n == nil {
		return fmt.Errorf("the project has no catalogs: set i18n in %s, as \"i18n\": {\"dir\": \"locales\", \"source\": \"en\"}", c.configName())
	}
	cat, diags, err := loadCatalogs(c)
	if err != nil {
		return err
	}
	for _, d := range diags {
		if d.Warning {
			warnf("%s", d)
		} else {
			con.println("  " + red(con.sym.fail) + " " + d.String())
		}
	}
	if n := diags.Errors(); n > 0 {
		return fmt.Errorf("%d %s in the catalogs", n, pluralWord(n, "error", "errors"))
	}
	logf("%d messages in %s", len(cat.Keys), strings.Join(cat.Languages, ", "))
	return nil
}

func pluralWord(n int, one, other string) string {
	if n == 1 {
		return one
	}
	return other
}

// framework is MyGo's own catalog, whose messages the keys "mygo.<key>"
// of apps change.
var framework = sync.OnceValues(func() (*i18ngen.Catalog, error) {
	c, diags, err := i18ngen.Load(l10n.FS(), i18ngen.Options{Dir: "mygo", Source: "en", Namespace: l10n.Namespace})
	if err == nil && diags.Errors() > 0 {
		err = diags.Err()
	}
	return c, err
})

// loadCatalogs reads and checks the catalogs of the app.
func loadCatalogs(c *Config) (*i18ngen.Catalog, i18ngen.Diagnostics, error) {
	dir := c.path(c.I18n.Dir)
	if !isDir(dir) {
		return nil, nil, fmt.Errorf("i18n.dir %s is not a directory: create it with %s.json, the messages of the source language", c.I18n.Dir, c.I18n.Source)
	}
	fw, err := framework()
	if err != nil {
		return nil, nil, fmt.Errorf("MyGo's own catalog: %v", err)
	}
	return i18ngen.Load(os.DirFS(dir), i18ngen.Options{
		Dir:       filepath.ToSlash(c.I18n.Dir),
		Source:    c.I18n.Source,
		Libraries: map[string]*i18ngen.Catalog{l10n.Namespace: fw},
		// The accessors import MyGo's translations to the app's languages.
		Translations: l10n.Languages,
	})
}

// i18nModule returns where the TypeScript module of the catalogs goes:
// next to the client, named after the catalogs' directory, as
// src/locales.ts, or "" for an app without a frontend.
func i18nModule(c *Config) string {
	if c.I18n == nil || !c.frontend {
		return ""
	}
	return filepath.Join(filepath.Dir(c.path(c.Bindings)), filepath.Base(c.path(c.I18n.Dir))+".ts")
}

// i18nOutput is what writeI18n did.
type i18nOutput struct {
	written  []string            // the files it wrote, relative to the root
	warnings i18ngen.Diagnostics // of the catalogs
	catalog  *i18ngen.Catalog    // the catalogs read
}

// writeI18n checks the catalogs of the app and writes their accessors,
// the Go functions in the catalogs' directory and the TypeScript module,
// where they changed. Errors of the catalogs fail it.
func writeI18n(c *Config) (*i18nOutput, error) {
	out := &i18nOutput{}
	if c.I18n == nil {
		return out, nil
	}
	cat, diags, err := loadCatalogs(c)
	if err != nil {
		return nil, err
	}
	if err := diags.Err(); err != nil {
		return nil, fmt.Errorf("the catalogs have errors (mygo i18n check lists them all):\n%v", err)
	}
	out.warnings, out.catalog = diags.Warnings(), cat
	dir := c.path(c.I18n.Dir)
	pkg, err := i18ngen.PackageName(dir)
	if err != nil {
		return nil, err
	}
	src, err := i18ngen.Go(cat, pkg)
	if err != nil {
		return nil, err
	}
	files := map[string][]byte{filepath.Join(dir, i18ngen.GoFile): src}
	if mod := i18nModule(c); mod != "" {
		ts, err := i18ngen.TS(cat)
		if err != nil {
			return nil, err
		}
		files[mod] = ts
	}
	for _, path := range slices.Sorted(func(yield func(string) bool) {
		for p := range files {
			if !yield(p) {
				return
			}
		}
	}) {
		if old, err := os.ReadFile(path); err == nil && bytes.Equal(old, files[path]) {
			continue
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return nil, err
		}
		if err := os.WriteFile(path, files[path], 0o644); err != nil {
			return nil, err
		}
		out.written = append(out.written, relPathTo(c.root, path))
	}
	return out, nil
}

// generateI18n writes the accessors of the catalogs as a step of a
// command, and shows the warnings of the catalogs.
func generateI18n(c *Config) error {
	if c.I18n == nil {
		return nil
	}
	t := con.start("Generating the accessors of the catalogs")
	out, err := writeI18n(c)
	if err != nil {
		t.fail()
		return err
	}
	if len(out.written) > 0 {
		t.done("Wrote " + cyan(strings.Join(out.written, ", ")))
	} else {
		t.done("The accessors of " + c.I18n.Dir + " are up to date")
	}
	for _, d := range out.warnings {
		warnf("%s", d)
	}
	return nil
}

// checkCatalogImported warns when the app's Go code does not import the
// package of its catalogs, which registers their languages as the app's:
// without it, an app whose pages alone say the messages would show them
// in its source language only. compiled are the directories of the
// packages the app is built from.
func checkCatalogImported(c *Config, compiled []string) {
	if imported, path := catalogImported(c, compiled); !imported {
		warnf("The app does not import %s, which makes the languages of its catalogs the app's: import it, as import _ %q where its Go code says none of the messages", path, path)
	}
}

// catalogImported reports whether the package of the catalogs is among
// those the app is built from, and its import path.
func catalogImported(c *Config, compiled []string) (bool, string) {
	if c.I18n == nil {
		return true, ""
	}
	dir := c.path(c.I18n.Dir)
	path := "./" + filepath.ToSlash(c.I18n.Dir)
	var out bytes.Buffer
	cmd := goCommand(dir, nil, "list", "-f", "{{.ImportPath}}", ".")
	cmd.Stdout = &out
	if cmd.Run() == nil {
		path = strings.TrimSpace(out.String())
	}
	for _, d := range compiled {
		if sameDir(d, dir) {
			return true, path
		}
	}
	return false, path
}

func sameDir(a, b string) bool {
	if a == b {
		return true
	}
	ia, err1 := os.Stat(a)
	ib, err2 := os.Stat(b)
	return err1 == nil && err2 == nil && os.SameFile(ia, ib)
}

// compiledDirs returns the directories of the packages the app is built
// from.
func compiledDirs(c *Config) []string {
	var out bytes.Buffer
	cmd := goCommand(c.root, nil, "list", "-e", "-deps", "-f", "{{.Dir}}", c.Main)
	cmd.Stdout = &out
	if cmd.Run() != nil {
		return nil
	}
	return strings.Fields(out.String())
}

// devLocale returns the locale of mygo dev -lang, as a canonical tag:
// a language, or a pseudo locale.
func devLocale(lang string) (string, error) {
	if lang == "" {
		return "", nil
	}
	tag := i18n.Normalize(lang)
	if tag == "" {
		return "", fmt.Errorf("-lang %q is not a language: a BCP 47 tag, as de, zh-Hant or pt-BR, or a pseudo locale, en-XA or ar-XB", lang)
	}
	return tag, nil
}
