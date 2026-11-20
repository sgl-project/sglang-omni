package main

import (
	"context"
	"fmt"
)

func runGenerate(args []string) error {
	fs := newFlags("generate", "[-o file] [dir]", `Writes the typed TypeScript client for the services bound with mygo.Bind
and the events declared with mygo.NewEvent. With i18n in mygo.json, it first
checks the catalogs of the app's messages and writes their accessors: Go
functions in the catalogs' directory, and a TypeScript module next to the
client, named after the directory (src/locales.ts).`)
	out := fs.String("o", "", "output file (default: bindings from mygo.json)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	c, err := loadConfig(dirArg(fs.Args()))
	if err != nil {
		return err
	}
	if *out != "" {
		c.Bindings = *out
		c.frontend = true
	}
	if c.Bindings == "" && c.I18n == nil {
		return fmt.Errorf("the project has no frontend to write a client for: set bindings in %s, or pass -o", c.configName())
	}
	// The accessors come first: the app's Go code calls them.
	if err := generateI18n(c); err != nil {
		return err
	}
	if c.I18n != nil {
		if dirs := compiledDirs(c); dirs != nil {
			checkCatalogImported(c, dirs)
		}
	}
	if c.Bindings == "" {
		return nil
	}
	return writeClient(context.Background(), c, true)
}

func dirArg(args []string) string {
	if len(args) > 0 {
		return args[0]
	}
	return "."
}
