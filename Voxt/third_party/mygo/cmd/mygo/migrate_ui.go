package main

import (
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/egoist/mygo/internal/uimigrate"
)

func runMigrateUI(args []string) error {
	f := flag.NewFlagSet("migrate-ui", flag.ContinueOnError)
	write := f.Bool("write", false, "write changes (by default only list affected files)")
	if err := f.Parse(args); err != nil {
		return err
	}
	dir := "."
	if f.NArg() > 1 {
		return fmt.Errorf("migrate-ui accepts one directory")
	}
	if f.NArg() == 1 {
		dir = f.Arg(0)
	}
	sources := map[string][]byte{}
	modes := map[string]fs.FileMode{}
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "node_modules", "vendor", "build", ".mygo":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		source, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		sources[path] = source
		info, err := d.Info()
		if err != nil {
			return err
		}
		modes[path] = info.Mode().Perm()
		return nil
	})
	if err != nil {
		return err
	}
	results, err := uimigrate.Files(sources)
	if err != nil {
		return err
	}
	var paths []string
	for path := range results {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	for _, path := range paths {
		result := results[path]
		for _, note := range result.Notes {
			fmt.Printf("%s: %s\n", path, note)
		}
		if !result.Changed {
			continue
		}
		if *write {
			if err := os.WriteFile(path, result.Source, modes[path]); err != nil {
				return err
			}
		}
		fmt.Println(path)
	}
	return nil
}
