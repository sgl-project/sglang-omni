// Package uimigrate migrates source syntax to the UI's checked value API.
// It resolves the actual MyGo import alias and local declaration types;
// it never performs global text replacements.
package uimigrate

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
)

// Result is a source migration and any changes requiring human review.
type Result struct {
	Source  []byte
	Changed bool
	Notes   []string
}

type migration struct {
	alias   string
	kinds   map[*ast.Object]string
	structs map[string]map[string]string
	objects map[*ast.Object]string
	returns map[string][]typeFact
	notes   []string
}

type typeFact struct{ kind, object string }
type packageFacts struct {
	structs map[string]map[string]string
	returns map[string][]typeFact
}

// File transforms one file. Files adds declarations from sibling package files.
func File(filename string, source []byte) (Result, error) {
	results, err := Files(map[string][]byte{filename: source})
	return results[filename], err
}

// Files migrates a source set, sharing syntax facts within each directory and
// package. It needs no successfully compiled old API or external Go tooling.
func Files(sources map[string][]byte) (map[string]Result, error) {
	fs := token.NewFileSet()
	files := map[string]*ast.File{}
	groups := map[string]*packageFacts{}
	var names []string
	for name := range sources {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		f, err := parser.ParseFile(fs, name, sources[name], parser.ParseComments)
		if err != nil {
			return nil, err
		}
		files[name] = f
		group := filepath.Dir(name) + "\x00" + f.Name.Name
		facts := groups[group]
		if facts == nil {
			facts = &packageFacts{structs: map[string]map[string]string{}, returns: map[string][]typeFact{}}
			groups[group] = facts
		}
		m := newMigration(importAlias(f), facts)
		ast.Inspect(f, func(node ast.Node) bool {
			switch n := node.(type) {
			case *ast.TypeSpec:
				if structure, ok := n.Type.(*ast.StructType); ok {
					fields := map[string]string{}
					for _, field := range structure.Fields.List {
						if kind := m.uiType(field.Type); kind != "" {
							for _, id := range field.Names {
								fields[id.Name] = kind
							}
						}
					}
					facts.structs[n.Name.Name] = fields
				}
			case *ast.FuncDecl:
				if n.Recv != nil || n.Type.Results == nil {
					return true
				}
				var results []typeFact
				for _, field := range n.Type.Results.List {
					fact := m.fact(field.Type)
					for range max(1, len(field.Names)) {
						results = append(results, fact)
					}
				}
				facts.returns[n.Name.Name] = results
			}
			return true
		})
	}
	results := map[string]Result{}
	for _, name := range names {
		f := files[name]
		alias := importAlias(f)
		if alias == "" || alias == "_" {
			results[name] = Result{Source: sources[name]}
			continue
		}
		if alias == "." {
			results[name] = Result{Source: sources[name], Notes: []string{"replace the dot import with an explicit MyGo UI alias before migrating"}}
			continue
		}
		m := newMigration(alias, groups[filepath.Dir(name)+"\x00"+f.Name.Name])
		m.collect(f)
		rewriteTypes(reflect.ValueOf(f), m)
		ast.Walk(visitor{m: m}, f)
		var out bytes.Buffer
		if err := format.Node(&out, fs, f); err != nil {
			return nil, err
		}
		formatted, err := format.Source(out.Bytes())
		if err != nil {
			return nil, err
		}
		results[name] = Result{Source: formatted, Changed: !bytes.Equal(sources[name], formatted), Notes: m.notes}
	}
	return results, nil
}

func importAlias(f *ast.File) string {
	for _, im := range f.Imports {
		if strings.Trim(im.Path.Value, "\"") == "github.com/egoist/mygo/ui" {
			if im.Name != nil {
				return im.Name.Name
			}
			return "ui"
		}
	}
	return ""
}
func newMigration(alias string, facts *packageFacts) *migration {
	return &migration{alias: alias, kinds: map[*ast.Object]string{}, objects: map[*ast.Object]string{}, structs: facts.structs, returns: facts.returns}
}
func (m *migration) fact(e ast.Expr) typeFact {
	if kind := m.uiType(e); kind != "" {
		return typeFact{kind: kind}
	}
	if star, ok := e.(*ast.StarExpr); ok {
		e = star.X
	}
	if name, ok := e.(*ast.Ident); ok {
		return typeFact{object: name.Name}
	}
	return typeFact{}
}
func (m *migration) expressionFact(e ast.Expr) typeFact {
	if kind := m.kind(e); kind != "" {
		return typeFact{kind: kind}
	}
	switch e := e.(type) {
	case *ast.Ident:
		return typeFact{object: m.objects[e.Obj]}
	case *ast.UnaryExpr:
		return m.expressionFact(e.X)
	case *ast.CompositeLit:
		return m.fact(e.Type)
	case *ast.CallExpr:
		if name, ok := e.Fun.(*ast.Ident); ok && len(m.returns[name.Name]) == 1 {
			return m.returns[name.Name][0]
		}
	}
	return typeFact{}
}
func (m *migration) remember(object *ast.Object, fact typeFact) {
	if object == nil {
		return
	}
	if fact.kind != "" {
		m.kinds[object] = fact.kind
	}
	if fact.object != "" {
		m.objects[object] = fact.object
	}
}

func (m *migration) uiType(e ast.Expr) string {
	if star, ok := e.(*ast.StarExpr); ok {
		e = star.X
	}
	if ix, ok := e.(*ast.IndexExpr); ok {
		e = ix.X
	}
	if ix, ok := e.(*ast.IndexListExpr); ok {
		e = ix.X
	}
	if s, ok := e.(*ast.SelectorExpr); ok {
		if p, ok := s.X.(*ast.Ident); ok && p.Name == m.alias {
			switch s.Sel.Name {
			case "Context":
				return "Context"
			case "Element":
				return "Element"
			case "SelectParts", "ComboboxParts", "TabsParts", "CollapsibleParts", "SegmentedParts", "ToastParts":
				return s.Sel.Name
			}
		}
	}
	return ""
}

func (m *migration) collect(f *ast.File) {
	ast.Inspect(f, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.TypeSpec:
			if s, ok := n.Type.(*ast.StructType); ok {
				fields := map[string]string{}
				for _, field := range s.Fields.List {
					if k := m.uiType(field.Type); k != "" {
						for _, id := range field.Names {
							fields[id.Name] = k
						}
						advice := "Handle for persistent control identity"
						if k == "Context" {
							advice = "Services for retained window services"
						}
						m.notes = append(m.notes, fmt.Sprintf("review stored %s fields on %s; use %s", k, n.Name.Name, advice))
					}
				}
				m.structs[n.Name.Name] = fields
			}
		case *ast.Field:
			k := m.uiType(n.Type)
			for _, id := range n.Names {
				if k != "" && id.Obj != nil {
					m.kinds[id.Obj] = k
				}
				typ := n.Type
				if s, ok := typ.(*ast.StarExpr); ok {
					typ = s.X
				}
				if name, ok := typ.(*ast.Ident); ok && id.Obj != nil {
					m.objects[id.Obj] = name.Name
				}
			}
		case *ast.ValueSpec:
			k := m.uiType(n.Type)
			for _, id := range n.Names {
				m.remember(id.Obj, m.fact(n.Type))
			}
			for _, id := range n.Names {
				if k != "" && id.Obj != nil {
					m.kinds[id.Obj] = k
				}
			}
		}
		return true
	})
	for range 3 {
		ast.Inspect(f, func(node ast.Node) bool {
			if a, ok := node.(*ast.AssignStmt); ok {
				if len(a.Rhs) == 1 {
					if call, ok := a.Rhs[0].(*ast.CallExpr); ok {
						if name, ok := call.Fun.(*ast.Ident); ok {
							facts := m.returns[name.Name]
							if len(facts) == len(a.Lhs) {
								for i, lhs := range a.Lhs {
									if id, ok := lhs.(*ast.Ident); ok {
										m.remember(id.Obj, facts[i])
									}
								}
								return true
							}
						}
					}
				}
				for i, lhs := range a.Lhs {
					if i >= len(a.Rhs) {
						break
					}
					if id, ok := lhs.(*ast.Ident); ok {
						m.remember(id.Obj, m.expressionFact(a.Rhs[i]))
					}
				}
			}
			if v, ok := node.(*ast.ValueSpec); ok {
				for i, name := range v.Names {
					if i < len(v.Values) {
						m.remember(name.Obj, m.expressionFact(v.Values[i]))
					}
				}
			}
			return true
		})
	}
}

var queries = map[string]bool{"Clicked": true, "Clicks": true, "Changed": true, "Submitted": true, "Hovered": true, "Pressed": true, "Focused": true, "FocusVisible": true, "FocusWithin": true, "Valid": true, "Shortcut": true, "ID": true, "Bounds": true, "IsDisabled": true, "PointerPosition": true, "Dragged": true, "TextSelection": true, "Composing": true, "DroppedFiles": true, "FileDragOver": true, "Dragging": true, "ClickModifiers": true, "DoubleClicked": true, "RightClicked": true, "PressedOutside": true, "Highlighted": true, "Loop": true, "Animate": true, "AnimateWith": true}
var nonElements = map[string]bool{"View": true, "NewTester": true, "Render": true, "Local": true, "RGB": true, "RGBA": true, "Hex": true, "LightTheme": true, "DarkTheme": true, "ParseSVG": true, "MustParseSVG": true, "NewBitmap": true, "DecodeBitmap": true, "Shape": true, "ShapeText": true, "ShapeRichText": true, "NewRouter": true, "NewTextBuffer": true, "Fixed": true, "Fr": true, "FitContent": true, "Overlay": true, "Drop": true, "DragOver": true, "DropData": true, "DataDragOver": true, "RegisterFont": true, "AlertDialog": true, "FocusedValue": true}

func (m *migration) kind(e ast.Expr) string {
	switch e := e.(type) {
	case *ast.Ident:
		return m.kinds[e.Obj]
	case *ast.SelectorExpr:
		if kind := m.kind(e.X); kind == "Parts" || strings.HasSuffix(kind, "Parts") {
			switch e.Sel.Name {
			case "Trigger", "Input", "Track", "List", "Root":
				return "Element"
			}
		}
		if id, ok := e.X.(*ast.Ident); ok {
			return m.structs[m.objects[id.Obj]][e.Sel.Name]
		}
	case *ast.CallExpr:
		if name, ok := e.Fun.(*ast.Ident); ok && len(m.returns[name.Name]) == 1 {
			return m.returns[name.Name][0].kind
		}
		fn := e.Fun
		if ix, ok := fn.(*ast.IndexExpr); ok {
			fn = ix.X
		}
		if ix, ok := fn.(*ast.IndexListExpr); ok {
			fn = ix.X
		}
		if s, ok := fn.(*ast.SelectorExpr); ok {
			if p, ok := s.X.(*ast.Ident); ok && p.Name == m.alias && !nonElements[s.Sel.Name] {
				if strings.HasSuffix(s.Sel.Name, "Base") && (s.Sel.Name == "SelectBase" || s.Sel.Name == "TabsBase" || s.Sel.Name == "ComboboxBase" || s.Sel.Name == "CollapsibleBase" || s.Sel.Name == "SegmentedBase" || s.Sel.Name == "ToastBase") {
					return "Parts"
				}
				return "Element"
			}
			if m.kind(s.X) == "Element" && !queries[s.Sel.Name] {
				return "Element"
			}
			if m.kind(s.X) == "Context" && s.Sel.Name == "Root" {
				return "Element"
			}
		}
	}
	return ""
}

var exprType = reflect.TypeOf((*ast.Expr)(nil)).Elem()

func rewriteTypes(v reflect.Value, m *migration) {
	if !v.IsValid() {
		return
	}
	if v.Kind() == reflect.Interface {
		if v.IsNil() {
			return
		}
		if v.Type() == exprType && v.CanSet() {
			if star, ok := v.Interface().(*ast.StarExpr); ok {
				if k := m.uiType(star); k != "" && k != "Context" {
					if s, ok := star.X.(*ast.SelectorExpr); ok {
						s.Sel.Name = k
					}
					v.Set(reflect.ValueOf(ast.Expr(star.X)))
					return
				}
			}
			if s, ok := v.Interface().(*ast.SelectorExpr); ok {
				if k := m.uiType(s); k != "" {
					s.Sel.Name = k
				}
			}
		}
		rewriteTypes(v.Elem(), m)
		return
	}
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return
		}
		// Object/scope links form cycles and are not syntax children.
		if v.Type() == reflect.TypeOf((*ast.Object)(nil)) || v.Type() == reflect.TypeOf((*ast.Scope)(nil)) {
			return
		}
		rewriteTypes(v.Elem(), m)
		return
	}
	if v.Kind() == reflect.Struct {
		for i := 0; i < v.NumField(); i++ {
			rewriteTypes(v.Field(i), m)
		}
	}
	if v.Kind() == reflect.Slice {
		for i := 0; i < v.Len(); i++ {
			rewriteTypes(v.Index(i), m)
		}
	}
}

type visitor struct {
	m       *migration
	frame   string
	returns []string
}

func (v visitor) Visit(n ast.Node) ast.Visitor {
	if n == nil {
		return nil
	}
	switch n := n.(type) {
	case *ast.FuncDecl:
		return v.function(n.Type)
	case *ast.FuncLit:
		return v.function(n.Type)
	case *ast.BinaryExpr:
		if n.Op == token.EQL || n.Op == token.NEQ {
			var e ast.Expr
			if isNil(n.X) {
				e = n.Y
			} else if isNil(n.Y) {
				e = n.X
			}
			if e != nil && v.m.kind(e) != "" {
				// Parent rewriting handles replacement of the whole expression.
			}
		}
	case *ast.CallExpr:
		if sel, ok := n.Fun.(*ast.SelectorExpr); ok {
			if sel.Sel.Name == "Key" && v.m.kind(sel.X) == "Element" && len(n.Args) == 1 {
				if root := v.m.constructor(sel.X); root != nil && len(root.Args) > 0 && v.m.kind(root.Args[0]) == "Context" {
					root.Args[0] = &ast.CallExpr{Fun: &ast.SelectorExpr{X: root.Args[0], Sel: ast.NewIdent("Key")}, Args: n.Args}
					if before, ok := sel.X.(*ast.CallExpr); ok {
						*n = *before
					}
				} else {
					v.m.notes = append(v.m.notes, "move a stored element's Key to Context.Key before its constructor")
				}
			}
			if sel.Sel.Name == "Rows" && v.m.kind(sel.X) == "Element" && len(n.Args) == 1 {
				if _, ok := n.Args[0].(*ast.FuncLit); !ok {
					if id, ok := n.Args[0].(*ast.Ident); !ok || id.Obj == nil || id.Obj.Kind != ast.Fun {
						sel.Sel.Name = "GridRows"
					}
				}
			}
		}
	case *ast.AssignStmt:
		for i, rhs := range n.Rhs {
			if i < len(n.Lhs) && isNil(rhs) {
				if k := v.m.kind(n.Lhs[i]); k == "Element" {
					n.Rhs[i] = v.zero(k)
				}
			}
		}
	case *ast.ReturnStmt:
		for i, rhs := range n.Results {
			if i < len(v.returns) && isNil(rhs) && v.returns[i] != "" && v.returns[i] != "Context" {
				n.Results[i] = v.zero(v.returns[i])
			}
		}
	}
	// Replace nil comparisons in expression fields without altering other
	// pointer comparisons. This pass uses collected declaration identities.
	replaceComparisons(reflect.ValueOf(n), v.m)
	return v
}

func (v visitor) function(t *ast.FuncType) visitor {
	if t.Params != nil {
		for _, p := range t.Params.List {
			if v.m.uiType(p.Type) == "Context" && len(p.Names) > 0 {
				v.frame = p.Names[0].Name
			}
		}
	}
	v.returns = nil
	if t.Results != nil {
		for _, r := range t.Results.List {
			for range max(len(r.Names), 1) {
				v.returns = append(v.returns, v.m.uiType(r.Type))
			}
		}
	}
	return v
}

func (v visitor) builderCall(call *ast.CallExpr) bool {
	fn := call.Fun
	if ix, ok := fn.(*ast.IndexExpr); ok {
		fn = ix.X
	}
	if ix, ok := fn.(*ast.IndexListExpr); ok {
		fn = ix.X
	}
	s, ok := fn.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	if id, ok := s.X.(*ast.Ident); ok && id.Name == v.m.alias {
		return !nonElements[s.Sel.Name] || s.Sel.Name == "Overlay" || s.Sel.Name == "View" || s.Sel.Name == "NewTester" || s.Sel.Name == "Render"
	}
	switch s.Sel.Name {
	case "Children", "Rows":
		return v.m.kind(s.X) == "Element"
	case "Popup", "Panel":
		kind := v.m.kind(s.X)
		return kind == "Parts" || strings.HasSuffix(kind, "Parts")
	case "View":
		return len(call.Args) > 0 && v.m.kind(call.Args[0]) == "Context"
	}
	return false
}

func hasFrame(t *ast.FuncType, m *migration) bool {
	if t.Params != nil {
		for _, p := range t.Params.List {
			if m.uiType(p.Type) == "Context" {
				return true
			}
		}
	}
	return false
}
func buildCallback(t *ast.FuncType) bool {
	if t.Params != nil {
		for _, p := range t.Params.List {
			e := p.Type
			if s, ok := e.(*ast.StarExpr); ok {
				e = s.X
			}
			switch e := e.(type) {
			case *ast.Ident:
				if e.Name == "error" {
					return false
				}
			case *ast.SelectorExpr:
				if e.Sel.Name == "Menu" || e.Sel.Name == "Painter" || e.Sel.Name == "InputEvent" {
					return false
				}
			}
		}
	}
	return true
}
func (v visitor) zero(k string) ast.Expr {
	return &ast.CompositeLit{Type: &ast.SelectorExpr{X: ast.NewIdent(v.m.alias), Sel: ast.NewIdent(k)}}
}
func isNil(e ast.Expr) bool { id, ok := e.(*ast.Ident); return ok && id.Name == "nil" }

func replaceComparisons(v reflect.Value, m *migration) {
	if !v.IsValid() {
		return
	}
	if v.Kind() == reflect.Interface && v.Type() == exprType && v.CanSet() && !v.IsNil() {
		if b, ok := v.Interface().(*ast.BinaryExpr); ok && (b.Op == token.EQL || b.Op == token.NEQ) {
			var e ast.Expr
			if isNil(b.X) {
				e = b.Y
			} else if isNil(b.Y) {
				e = b.X
			}
			if e != nil && m.kind(e) == "Element" {
				var replacement ast.Expr = &ast.CallExpr{Fun: &ast.SelectorExpr{X: e, Sel: ast.NewIdent("Valid")}}
				if b.Op == token.EQL {
					replacement = &ast.UnaryExpr{Op: token.NOT, X: replacement}
				}
				v.Set(reflect.ValueOf(replacement))
				return
			}
		}
	}
	// Only direct expression children are needed here; ast.Walk visits the rest.
	if v.Kind() == reflect.Pointer && !v.IsNil() {
		v = v.Elem()
	}
	if v.Kind() == reflect.Struct {
		for i := 0; i < v.NumField(); i++ {
			f := v.Field(i)
			if f.Kind() == reflect.Interface && f.Type() == exprType {
				replaceComparisons(f, m)
			} else if f.Kind() == reflect.Slice {
				for j := 0; j < f.Len(); j++ {
					if f.Index(j).Kind() == reflect.Interface && f.Index(j).Type() == exprType {
						replaceComparisons(f.Index(j), m)
					}
				}
			}
		}
	}
}

func callbackHasRow(t *ast.FuncType) bool {
	if t.Params != nil {
		for _, p := range t.Params.List {
			if s, ok := p.Type.(*ast.SelectorExpr); ok && s.Sel.Name == "ListRow" {
				return true
			}
		}
	}
	return false
}

// constructor finds the imported UI factory at the start of a fluent chain.
func (m *migration) constructor(e ast.Expr) *ast.CallExpr {
	call, ok := e.(*ast.CallExpr)
	if !ok {
		return nil
	}
	fn := call.Fun
	if ix, ok := fn.(*ast.IndexExpr); ok {
		fn = ix.X
	}
	if ix, ok := fn.(*ast.IndexListExpr); ok {
		fn = ix.X
	}
	sel, ok := fn.(*ast.SelectorExpr)
	if !ok {
		return nil
	}
	if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == m.alias && !nonElements[sel.Sel.Name] {
		return call
	}
	return m.constructor(sel.X)
}
