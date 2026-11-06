package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/egoist/mygo/internal/uivet"
)

type vetPackage struct {
	ImportPath, Dir, Export            string
	GoFiles, CgoFiles, CompiledGoFiles []string
	DepOnly                            bool
	Error                              *struct{ Err string }
}

func runVetUI(args []string) error {
	flags := flag.NewFlagSet("vet", flag.ContinueOnError)
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() > 1 {
		return fmt.Errorf("vet accepts one app directory")
	}
	dir := "."
	if flags.NArg() == 1 {
		dir = flags.Arg(0)
	}
	cmd := exec.Command("go", "vet", "./...")
	cmd.Dir = dir
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("go vet: %w", err)
	}
	cmd = exec.Command("go", "list", "-export", "-deps", "-json", "./...")
	cmd.Dir = dir
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err = cmd.Start(); err != nil {
		return err
	}
	var packages []vetPackage
	exports := map[string]string{}
	decoder := json.NewDecoder(stdout)
	for {
		var pkg vetPackage
		err = decoder.Decode(&pkg)
		if err == io.EOF {
			break
		}
		if err != nil {
			_ = cmd.Wait()
			return err
		}
		if pkg.Error != nil {
			_ = cmd.Wait()
			return fmt.Errorf("%s", pkg.Error.Err)
		}
		exports[pkg.ImportPath] = pkg.Export
		if !pkg.DepOnly {
			packages = append(packages, pkg)
		}
	}
	if err = cmd.Wait(); err != nil {
		return fmt.Errorf("load Go packages: %w", err)
	}
	fs := token.NewFileSet()
	imported := importer.ForCompiler(fs, "gc", func(path string) (io.ReadCloser, error) {
		file := exports[path]
		if file == "" {
			return nil, fmt.Errorf("missing compiled export for %s", path)
		}
		return os.Open(file)
	})
	failures := 0
	for _, pkg := range packages {
		if pkg.ImportPath == "github.com/egoist/mygo/ui" {
			continue
		}
		var files []*ast.File
		// Files that import "C" are in CgoFiles: check them with the rest,
		// against an empty package C whose members are not reported undefined.
		for _, name := range append(pkg.GoFiles, pkg.CgoFiles...) {
			f, err := parser.ParseFile(fs, filepath.Join(pkg.Dir, name), nil, parser.ParseComments)
			if err != nil {
				return err
			}
			files = append(files, f)
		}
		info := &types.Info{Defs: map[*ast.Ident]types.Object{}, Uses: map[*ast.Ident]types.Object{}, Types: map[ast.Expr]types.TypeAndValue{}}
		checked, err := (&types.Config{Importer: imported, FakeImportC: true}).Check(pkg.ImportPath, fs, files, info)
		if err != nil {
			return err
		}
		for _, d := range uivet.Analyze(fs, files, checked, info) {
			fmt.Fprintln(os.Stderr, d.String())
			failures++
		}
	}
	if failures > 0 {
		return fmt.Errorf("%d native UI lifetime issue(s)", failures)
	}
	return nil
}
