// Package uivet checks native UI build lifetimes using Go type information.
package uivet

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sort"
)

const uiPath = "github.com/egoist/mygo/ui"

type Diagnostic struct {
	Position token.Position
	Message  string
}

func (d Diagnostic) String() string { return fmt.Sprintf("%s: %s", d.Position, d.Message) }

// Analyze flags per-pass values in storage and values passed to or captured by
// goroutines. Callback signatures and persistent Handle/Services are allowed.
func Analyze(fs *token.FileSet, files []*ast.File, pkg *types.Package, info *types.Info) []Diagnostic {
	if pkg.Path() == uiPath {
		return nil
	}
	var out []Diagnostic
	report := func(pos token.Pos, message string) { out = append(out, Diagnostic{fs.Position(pos), message}) }
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.StructType:
				for _, field := range n.Fields.List {
					if transient(info.TypeOf(field.Type)) {
						report(field.Pos(), "native UI build value stored in a struct; keep ui.Handle or ui.Services instead")
					}
				}
			case *ast.ValueSpec:
				for _, name := range n.Names {
					if object, ok := info.Defs[name].(*types.Var); ok && object.Parent() == pkg.Scope() && transient(object.Type()) {
						report(name.Pos(), "native UI build value stored in a package variable; keep ui.Handle or ui.Services instead")
					}
				}
			case *ast.GoStmt:
				seen := map[types.Object]bool{}
				for _, arg := range n.Call.Args {
					if transient(info.TypeOf(arg)) {
						report(arg.Pos(), "native UI build value passed to a goroutine; publish results with Window.Update")
					}
				}
				ast.Inspect(n.Call.Fun, func(child ast.Node) bool {
					id, ok := child.(*ast.Ident)
					if !ok {
						return true
					}
					object := info.Uses[id]
					if _, ok := object.(*types.Var); !ok {
						return true
					}
					if object == nil || seen[object] || object.Pos() >= n.Pos() || !transient(object.Type()) {
						return true
					}
					seen[object] = true
					report(id.Pos(), "native UI build value captured by a goroutine; publish results with Window.Update")
					return true
				})
			}
			return true
		})
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i].Position, out[j].Position
		if a.Filename != b.Filename {
			return a.Filename < b.Filename
		}
		return a.Offset < b.Offset
	})
	return out
}

func transient(t types.Type) bool {
	if t == nil {
		return false
	}
	t = types.Unalias(t)
	switch t := t.(type) {
	case *types.Pointer:
		return transient(t.Elem())
	case *types.Slice:
		return transient(t.Elem())
	case *types.Array:
		return transient(t.Elem())
	case *types.Map:
		return transient(t.Key()) || transient(t.Elem())
	case *types.Chan:
		return transient(t.Elem())
	case *types.Named:
		object := t.Obj()
		if object.Pkg() == nil || object.Pkg().Path() != uiPath {
			return false
		}
		switch object.Name() {
		case "Element", "Context", "TabsParts", "SelectParts", "ComboboxParts", "CollapsibleParts", "SegmentedParts", "ToastParts", "ListRow":
			return true
		}
	}
	return false
}
