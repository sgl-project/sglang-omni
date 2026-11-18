// Command declhash prints a line per top-level declaration of the Go
// packages in the given directories: directory, package, kind, name, build
// constraints and a hash of its source as go/printer formats it, doc
// comment and inner comments included. A pure move of code between files
// (refactor batches "safe": split a file, rename a file) leaves the sorted
// output unchanged.
//
//	go run ./internal/tools/declhash DIR... > before.txt   (on the base commit)
//	go run ./internal/tools/declhash DIR... > after.txt    (on the branch)
//	diff before.txt after.txt                              (empty for a pure move)
//
// DIR/... takes DIR and the directories under it, but for testdata,
// third_party, the Swift helper and the build output (helper/ and build/
// next to go.mod) and hidden ones. Imports are left out (a split adds import
// blocks); a declaration moved to a file with other build constraints
// (//go:build, or a _darwin.go name) shows as changed. Test files are
// included, marked (test).
//
// -nocomments leaves the comments out of the hashes: a move that also
// updated the file names comments mention ("// overlay.go") shows as
// changed without it and unchanged with it, which tells such edits from
// changes of code.
//
//	go run ./internal/tools/declhash [-nocomments] DIR...
package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/printer"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// noComments leaves comments out of the hashes (-nocomments).
var noComments bool

func main() {
	args := os.Args[1:]
	if len(args) > 0 && args[0] == "-nocomments" {
		noComments, args = true, args[1:]
	}
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: declhash [-nocomments] DIR[/...]...")
		os.Exit(2)
	}
	var lines []string
	for _, arg := range args {
		dirs := []string{arg}
		if root, ok := strings.CutSuffix(arg, "/..."); ok {
			var err error
			if dirs, err = walk(root); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(2)
			}
		}
		for _, dir := range dirs {
			l, err := hashDir(dir)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(2)
			}
			lines = append(lines, l...)
		}
	}
	slices.Sort(lines)
	for _, l := range lines {
		fmt.Println(l)
	}
}

// walk returns root and the directories under it, but for those skipped.
func walk(root string) ([]string, error) {
	var dirs []string
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			return nil
		}
		name := d.Name()
		if p != root && (strings.HasPrefix(name, ".") || name == "testdata" || name == "third_party" ||
			(name == "helper" || name == "build") && fileExists(filepath.Join(filepath.Dir(p), "go.mod"))) {
			return filepath.SkipDir
		}
		dirs = append(dirs, p)
		return nil
	})
	return dirs, err
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// hashDir returns the lines of the .go files in dir.
func hashDir(dir string) ([]string, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		return nil, err
	}
	var lines []string
	for _, file := range files {
		l, err := hashFile(file)
		if err != nil {
			return nil, err
		}
		lines = append(lines, l...)
	}
	return lines, nil
}

func hashFile(file string) ([]string, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, file, nil, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		return nil, err
	}
	pkg := filepath.ToSlash(filepath.Dir(file)) + " " + f.Name.Name
	if strings.HasSuffix(file, "_test.go") {
		pkg += " (test)"
	}
	if c := constraints(f, file); c != "" {
		pkg += " [" + c + "]"
	}
	var lines []string
	for _, d := range f.Decls {
		kind, name := "", ""
		var doc *ast.CommentGroup
		switch d := d.(type) {
		case *ast.FuncDecl:
			kind, name, doc = "func", d.Name.Name, d.Doc
			if d.Recv != nil {
				var b bytes.Buffer
				printer.Fprint(&b, fset, d.Recv.List[0].Type)
				name = "(" + b.String() + ")." + name
			}
		case *ast.GenDecl:
			if d.Tok == token.IMPORT {
				continue
			}
			kind, doc = d.Tok.String(), d.Doc
			var names []string
			for _, s := range d.Specs {
				switch s := s.(type) {
				case *ast.TypeSpec:
					names = append(names, s.Name.Name)
				case *ast.ValueSpec:
					for _, n := range s.Names {
						names = append(names, n.Name)
					}
				}
			}
			name = strings.Join(names, ",")
		}
		var b bytes.Buffer
		if doc != nil && !noComments {
			for _, c := range doc.List {
				b.WriteString(c.Text + "\n")
			}
		}
		if noComments {
			dropComments(d)
		}
		printer.Fprint(&b, fset, d)
		// A bare node prints without its inner comments: hash them too.
		for _, cg := range f.Comments {
			if !noComments && cg.Pos() > d.Pos() && cg.End() < d.End() {
				b.WriteString(cg.Text())
			}
		}
		sum := sha256.Sum256(b.Bytes())
		lines = append(lines, fmt.Sprintf("%s %s %s %x", pkg, kind, name, sum[:8]))
	}
	return lines, nil
}

// constraints returns the file's build constraints: its //go:build line
// and the GOOS/GOARCH of its name.
func constraints(f *ast.File, file string) string {
	var cs []string
	for _, cg := range f.Comments {
		if cg.Pos() >= f.Package {
			break
		}
		for _, c := range cg.List {
			if constraint.IsGoBuild(c.Text) {
				if expr, err := constraint.Parse(c.Text); err == nil {
					cs = append(cs, expr.String())
				}
			}
		}
	}
	base := strings.TrimSuffix(strings.TrimSuffix(filepath.Base(file), ".go"), "_test")
	parts := strings.Split(base, "_")
	for _, p := range parts[1:] {
		if knownOS[p] || knownArch[p] {
			cs = append(cs, p)
		}
	}
	return strings.Join(cs, " ")
}

var (
	knownOS = map[string]bool{"aix": true, "android": true, "darwin": true, "dragonfly": true, "freebsd": true, "hurd": true,
		"illumos": true, "ios": true, "js": true, "linux": true, "nacl": true, "netbsd": true, "openbsd": true, "plan9": true,
		"solaris": true, "wasip1": true, "windows": true, "zos": true}
	knownArch = map[string]bool{"386": true, "amd64": true, "arm": true, "arm64": true, "loong64": true, "mips": true,
		"mips64": true, "mips64le": true, "mipsle": true, "ppc64": true, "ppc64le": true, "riscv64": true, "s390x": true, "wasm": true}
)

// dropComments removes the comments the printer prints with a bare node:
// the doc comments and line comments of the declaration and its parts.
func dropComments(d ast.Decl) {
	ast.Inspect(d, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.FuncDecl:
			n.Doc = nil
		case *ast.GenDecl:
			n.Doc = nil
		case *ast.ValueSpec:
			n.Doc, n.Comment = nil, nil
		case *ast.TypeSpec:
			n.Doc, n.Comment = nil, nil
		case *ast.Field:
			n.Doc, n.Comment = nil, nil
		}
		return true
	})
}
