package tsgen

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
)

// sources reads Go source files to enrich the generated client with
// parameter names, documentation and enum values. Everything it provides is
// optional: the client is still correct when sources are unavailable, e.g.
// in binaries built with -trimpath.
type sources struct {
	fset   *token.FileSet
	files  map[string]*ast.File
	dirs   map[string][]*ast.File
	pkgDir map[string]string
}

func newSources() *sources {
	return &sources{
		fset:   token.NewFileSet(),
		files:  map[string]*ast.File{},
		dirs:   map[string][]*ast.File{},
		pkgDir: map[string]string{},
	}
}

func (s *sources) file(path string) *ast.File {
	if f, ok := s.files[path]; ok {
		return f
	}
	f, err := parser.ParseFile(s.fset, path, nil, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		f = nil
	}
	s.files[path] = f
	return f
}

// dirFiles parses the non-test Go files of a directory.
func (s *sources) dirFiles(dir string) []*ast.File {
	if files, ok := s.dirs[dir]; ok {
		return files
	}
	var files []*ast.File
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		if f := s.file(filepath.Join(dir, name)); f != nil {
			files = append(files, f)
		}
	}
	s.dirs[dir] = files
	return files
}

// funcPackage extracts the package path from a runtime function name such
// as "github.com/x/app/svc.(*Greeter).Greet".
func funcPackage(name string) string {
	slash := strings.LastIndex(name, "/")
	dot := strings.Index(name[slash+1:], ".")
	if dot < 0 {
		return ""
	}
	return name[:slash+1+dot]
}

// learnPC records the directory of the package that contains pc.
func (s *sources) learnPC(pc uintptr) (file string, line int) {
	fn := runtime.FuncForPC(pc)
	if fn == nil {
		return "", 0
	}
	file, line = fn.FileLine(fn.Entry())
	if pkg := funcPackage(fn.Name()); pkg != "" && filepath.IsAbs(file) {
		if _, ok := s.pkgDir[pkg]; !ok {
			s.pkgDir[pkg] = filepath.Dir(file)
		}
	}
	return file, line
}

// resolvePackages asks the go command for the directories of packages that
// were not discovered through program counters.
func (s *sources) resolvePackages(pkgs []string) {
	var missing []string
	for _, p := range pkgs {
		if _, ok := s.pkgDir[p]; !ok && p != "main" && strings.Contains(strings.Split(p, "/")[0], ".") {
			missing = append(missing, p)
		}
	}
	if len(missing) == 0 {
		return
	}
	goBin, err := exec.LookPath("go")
	if err != nil {
		return
	}
	out, err := exec.Command(goBin, append([]string{"list", "-e", "-f", "{{.ImportPath}} {{.Dir}}"}, missing...)...).Output()
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(out), "\n") {
		pkg, dir, ok := strings.Cut(line, " ")
		if ok && dir != "" {
			s.pkgDir[pkg] = dir
		}
	}
}

// funcDecl finds the declaration of the method containing the given
// source position.
func (s *sources) funcDecl(file string, line int, name string) *ast.FuncDecl {
	if file == "" {
		return nil
	}
	f := s.file(file)
	if f == nil {
		return nil
	}
	for _, d := range f.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != name {
			continue
		}
		start, end := s.fset.Position(fd.Pos()).Line, s.fset.Position(fd.End()).Line
		if start <= line && line <= end {
			return fd
		}
	}
	return nil
}

// paramNames returns the names of all parameters of fd; unnamed parameters
// yield "".
func paramNames(fd *ast.FuncDecl) []string {
	var names []string
	for _, p := range fd.Type.Params.List {
		if len(p.Names) == 0 {
			names = append(names, "")
			continue
		}
		for _, n := range p.Names {
			if n.Name == "_" {
				names = append(names, "")
			} else {
				names = append(names, n.Name)
			}
		}
	}
	return names
}

// typeDecl finds the declaration of a named type in its package directory.
func (s *sources) typeDecl(t reflect.Type) (*ast.TypeSpec, *ast.GenDecl) {
	dir, ok := s.pkgDir[t.PkgPath()]
	if !ok {
		return nil, nil
	}
	name := t.Name()
	if i := strings.IndexByte(name, '['); i >= 0 {
		name = name[:i]
	}
	for _, f := range s.dirFiles(dir) {
		for _, d := range f.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok || gd.Tok != token.TYPE {
				continue
			}
			for _, spec := range gd.Specs {
				if ts := spec.(*ast.TypeSpec); ts.Name.Name == name {
					return ts, gd
				}
			}
		}
	}
	return nil, nil
}

// typeDoc returns the documentation of a named type.
func (s *sources) typeDoc(t reflect.Type) string {
	ts, gd := s.typeDecl(t)
	if ts == nil {
		return ""
	}
	if ts.Doc != nil {
		return ts.Doc.Text()
	}
	if gd.Doc != nil && len(gd.Specs) == 1 {
		return gd.Doc.Text()
	}
	return ""
}

// fieldDoc returns the documentation of a struct field.
func (s *sources) fieldDoc(owner reflect.Type, goName string) string {
	if owner.Name() == "" {
		return ""
	}
	ts, _ := s.typeDecl(owner)
	if ts == nil {
		return ""
	}
	st, ok := ts.Type.(*ast.StructType)
	if !ok {
		return ""
	}
	for _, f := range st.Fields.List {
		for _, n := range f.Names {
			if n.Name != goName {
				continue
			}
			if f.Doc != nil {
				return f.Doc.Text()
			}
			if f.Comment != nil {
				return f.Comment.Text()
			}
			return ""
		}
	}
	return ""
}

// valueDoc returns the documentation of the variable declared at the
// given position, used for events.
func (s *sources) valueDoc(file string, line int) string {
	if file == "" {
		return ""
	}
	f := s.file(file)
	if f == nil {
		return ""
	}
	for _, d := range f.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.VAR {
			continue
		}
		for _, spec := range gd.Specs {
			vs := spec.(*ast.ValueSpec)
			start, end := s.fset.Position(vs.Pos()).Line, s.fset.Position(vs.End()).Line
			if line < start || line > end {
				continue
			}
			if vs.Doc != nil {
				return vs.Doc.Text()
			}
			if gd.Doc != nil && len(gd.Specs) == 1 {
				return gd.Doc.Text()
			}
			return ""
		}
	}
	return ""
}

// enumValues returns TypeScript literals of the constants declared with
// type t, or nil if there are none or one cannot be evaluated.
func (s *sources) enumValues(t reflect.Type) []string {
	dir, ok := s.pkgDir[t.PkgPath()]
	if !ok || t.Name() == "" {
		return nil
	}
	var values []string
	seen := map[string]bool{}
	for _, f := range s.dirFiles(dir) {
		for _, d := range f.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok || gd.Tok != token.CONST {
				continue
			}
			var lastType ast.Expr
			var lastValues []ast.Expr
			for iota, spec := range gd.Specs {
				vs := spec.(*ast.ValueSpec)
				if len(vs.Values) > 0 {
					lastType, lastValues = vs.Type, vs.Values
				}
				ident, ok := lastType.(*ast.Ident)
				if !ok || ident.Name != t.Name() {
					continue
				}
				for i := range vs.Names {
					if vs.Names[i].Name == "_" {
						continue
					}
					if i >= len(lastValues) {
						return nil
					}
					lit, ok := evalConst(lastValues[i], int64(iota), t.Kind())
					if !ok {
						return nil
					}
					if !seen[lit] {
						seen[lit] = true
						values = append(values, lit)
					}
				}
			}
		}
	}
	return values
}

// evalConst evaluates simple constant expressions (literals and integer
// arithmetic with iota) to a TypeScript literal.
func evalConst(e ast.Expr, iota int64, kind reflect.Kind) (string, bool) {
	if kind == reflect.String {
		if lit, ok := e.(*ast.BasicLit); ok && lit.Kind == token.STRING {
			s, err := strconv.Unquote(lit.Value)
			if err != nil {
				return "", false
			}
			return quote(s), true
		}
		return "", false
	}
	if !isNumber(kind) {
		return "", false
	}
	v, ok := evalInt(e, iota)
	if !ok {
		return "", false
	}
	return strconv.FormatInt(v, 10), true
}

func evalInt(e ast.Expr, iota int64) (int64, bool) {
	switch e := e.(type) {
	case *ast.BasicLit:
		if e.Kind != token.INT {
			return 0, false
		}
		v, err := strconv.ParseInt(strings.ReplaceAll(e.Value, "_", ""), 0, 64)
		return v, err == nil
	case *ast.Ident:
		return iota, e.Name == "iota"
	case *ast.ParenExpr:
		return evalInt(e.X, iota)
	case *ast.UnaryExpr:
		v, ok := evalInt(e.X, iota)
		switch e.Op {
		case token.SUB:
			return -v, ok
		case token.ADD:
			return v, ok
		}
		return 0, false
	case *ast.BinaryExpr:
		x, ok1 := evalInt(e.X, iota)
		y, ok2 := evalInt(e.Y, iota)
		if !ok1 || !ok2 {
			return 0, false
		}
		switch e.Op {
		case token.ADD:
			return x + y, true
		case token.SUB:
			return x - y, true
		case token.MUL:
			return x * y, true
		case token.SHL:
			return x << y, y >= 0 && y < 63
		case token.OR:
			return x | y, true
		}
	}
	return 0, false
}
