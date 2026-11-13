package archtest

import (
	"bufio"
	"bytes"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// moduleRoot returns the directory of go.mod above the test's directory
// and the module path.
func moduleRoot(t *testing.T) (dir, module string) {
	t.Helper()
	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatal(err)
	}
	for {
		data, err := os.ReadFile(filepath.Join(dir, "go.mod"))
		if err == nil {
			for line := range strings.Lines(string(data)) {
				if m, ok := strings.CutPrefix(strings.TrimSpace(line), "module "); ok {
					return dir, strings.TrimSpace(m)
				}
			}
			t.Fatalf("%s/go.mod names no module", dir)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("no go.mod above the test")
		}
		dir = parent
	}
}

// goFile is a Go file of the module: its path from the module root (with
// slashes) and its contents.
type goFile struct {
	path string
	data []byte
}

func (f goFile) test() bool { return strings.HasSuffix(f.path, "_test.go") }

// dir is the file's directory from the module root, "." for the root.
func (f goFile) dir() string {
	if i := strings.LastIndexByte(f.path, '/'); i >= 0 {
		return f.path[:i]
	}
	return "."
}

// generated reports whether the file says it is generated, as Go's
// convention has it: a line "// Code generated … DO NOT EDIT." before the
// package clause.
func (f goFile) generated() bool {
	sc := bufio.NewScanner(bytes.NewReader(f.data))
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "package ") {
			return false
		}
		if strings.HasPrefix(line, "// Code generated ") && strings.HasSuffix(line, " DO NOT EDIT.") {
			return true
		}
	}
	return false
}

// lines counts lines as wc -l does, and a last line without a newline.
func (f goFile) lines() int {
	n := bytes.Count(f.data, []byte("\n"))
	if len(f.data) > 0 && f.data[len(f.data)-1] != '\n' {
		n++
	}
	return n
}

// goFiles returns the module's Go files but those of third_party, testdata
// and hidden directories, of the Swift helper (helper/ at the root) and of
// the build output (build/ at the root).
func goFiles(t *testing.T) []goFile {
	t.Helper()
	root, _ := moduleRoot(t)
	var files []goFile
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name := d.Name()
		if d.IsDir() {
			if p != root && (strings.HasPrefix(name, ".") || name == "third_party" || name == "testdata" ||
				(name == "helper" || name == "build") && filepath.Dir(p) == root) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(name, ".go") {
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, p)
		if err != nil {
			return err
		}
		files = append(files, goFile{filepath.ToSlash(rel), data})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(files) < 50 {
		t.Fatalf("found %d Go files under %s: is it the module?", len(files), root)
	}
	return files
}
