package archtest

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
)

const (
	mygoPath          = "github.com/egoist/mygo"
	viewAllowlistSize = 2
)

type rootFile struct {
	goFile
	fset   *token.FileSet
	syntax *ast.File
}

// viewFile reports whether path is a root view file. MyGo UI imports catch
// dialogs, cards and shell views; the name patterns keep view files covered
// when they contain only layout or view state and import no MyGo UI package.
func viewFile(path string, syntax *ast.File) bool {
	if strings.Contains(path, "/") || strings.HasSuffix(path, "_test.go") {
		return false
	}
	if strings.HasPrefix(path, "ui_") || strings.HasPrefix(path, "overlay_") ||
		strings.HasSuffix(path, "_page.go") || strings.HasSuffix(path, "_view.go") ||
		strings.HasSuffix(path, "_dialog.go") || strings.HasSuffix(path, "_dialogs.go") {
		return true
	}
	for _, spec := range syntax.Imports {
		importPath, _ := strconv.Unquote(spec.Path.Value)
		if importPath == mygoUI || strings.HasPrefix(importPath, mygoUI+"/") {
			return true
		}
	}
	return false
}

func parseRootFiles(t *testing.T) []rootFile {
	t.Helper()
	var files []rootFile
	for _, f := range goFiles(t) {
		if f.dir() != "." || f.test() {
			continue
		}
		files = append(files, parseRootFile(t, f.path, string(f.data)))
	}
	return files
}

func parseRootFile(t *testing.T, path, source string) rootFile {
	t.Helper()
	fset := token.NewFileSet()
	syntax, err := parser.ParseFile(fset, path, source, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	return rootFile{goFile{path: path, data: []byte(source)}, fset, syntax}
}

func appType(expr ast.Expr, aliases map[string]bool) bool {
	if star, ok := expr.(*ast.StarExpr); ok {
		expr = star.X
	}
	ident, ok := expr.(*ast.Ident)
	return ok && aliases[ident.Name]
}

func valueTypeName(expr ast.Expr) string {
	if star, ok := expr.(*ast.StarExpr); ok {
		expr = star.X
	}
	ident, ok := expr.(*ast.Ident)
	if !ok {
		return ""
	}
	return ident.Name
}

func parameterTypeAt(syntax *ast.File, position token.Pos, name string) string {
	for _, declaration := range syntax.Decls {
		fn, ok := declaration.(*ast.FuncDecl)
		if !ok || position < fn.Pos() || position > fn.End() {
			continue
		}
		for _, field := range functionValueFields(fn) {
			for _, parameter := range field.Names {
				if parameter.Name == name {
					return valueTypeName(field.Type)
				}
			}
		}
	}
	return ""
}

func domainPackage(path, module string) bool {
	internal := module + "/internal/"
	allowedInternal := path == internal+"icons" || path == internal+"i18n" || path == internal+"overlay/layout"
	switch {
	case path == "" || path == mygoPath || strings.HasPrefix(path, mygoPath+"/") || allowedInternal:
		return false
	case strings.HasPrefix(path, internal) || viewIOImport(path):
		return true
	default:
		return strings.Contains(strings.Split(path, "/")[0], ".")
	}
}

type typeAlias struct {
	expression ast.Expr
	imports    map[string]string
}

type packageTypes struct {
	aliases   map[string]typeAlias
	actions   map[string]bool
	viewTypes map[string]bool
}

func packageTypesFor(files []rootFile, views map[string]bool) packageTypes {
	types := packageTypes{
		aliases:   map[string]typeAlias{},
		actions:   map[string]bool{},
		viewTypes: map[string]bool{},
	}
	viewNames := map[string]bool{}
	for _, file := range files {
		imports := importNames(file.syntax)
		for _, declaration := range file.syntax.Decls {
			gen, ok := declaration.(*ast.GenDecl)
			if !ok || gen.Tok != token.TYPE {
				continue
			}
			for _, raw := range gen.Specs {
				spec := raw.(*ast.TypeSpec)
				name := spec.Name.Name
				if views[file.path] {
					viewNames[name] = true
				}
				if strings.HasSuffix(name, "Actions") {
					switch spec.Type.(type) {
					case *ast.StructType, *ast.InterfaceType:
						types.actions[name] = true
					}
				}
				if spec.Assign.IsValid() {
					types.aliases[name] = typeAlias{spec.Type, imports}
				}
			}
		}
	}
	for name := range viewNames {
		types.viewTypes[types.name(ast.NewIdent(name))] = true
	}
	return types
}

func (types packageTypes) name(expression ast.Expr) string {
	if star, ok := expression.(*ast.StarExpr); ok {
		return types.name(star.X)
	}
	identifier, ok := expression.(*ast.Ident)
	if !ok {
		return ""
	}
	if alias, ok := types.aliases[identifier.Name]; ok {
		if name := types.name(alias.expression); name != "" {
			return name
		}
	}
	return identifier.Name
}

func (types packageTypes) domain(expression ast.Expr, imports map[string]string, module string) bool {
	switch expression := expression.(type) {
	case *ast.StarExpr:
		return types.domain(expression.X, imports, module)
	case *ast.ArrayType:
		return types.domain(expression.Elt, imports, module)
	case *ast.Ident:
		alias, ok := types.aliases[expression.Name]
		return ok && types.domain(alias.expression, alias.imports, module)
	case *ast.SelectorExpr:
		identifier, ok := expression.X.(*ast.Ident)
		return ok && domainPackage(imports[identifier.Name], module)
	}
	return false
}

func projectionResults(declaration *ast.FuncDecl, imports map[string]string, module string, types packageTypes) (viewState, domain bool) {
	if declaration.Type.Results == nil {
		return false, false
	}
	for _, result := range declaration.Type.Results.List {
		name := types.name(result.Type)
		viewState = viewState || strings.HasSuffix(name, "ViewState") || strings.HasSuffix(name, "PreviewState")
		domain = domain || types.domain(result.Type, imports, module)
	}
	return viewState, domain
}

func functionValueFields(declaration *ast.FuncDecl) []*ast.Field {
	fields := append([]*ast.Field{}, declaration.Type.Params.List...)
	if declaration.Recv != nil {
		fields = append(fields, declaration.Recv.List...)
	}
	return fields
}

func localStateTypes(files []rootFile, views map[string]bool, aliases map[string]bool) map[string]bool {
	declarations := packageTypesFor(files, views)
	types := map[string]bool{}
	for _, file := range files {
		if !views[file.path] {
			continue
		}
		for _, declaration := range file.syntax.Decls {
			fn, ok := declaration.(*ast.FuncDecl)
			if !ok {
				continue
			}
			for _, field := range functionValueFields(fn) {
				name := declarations.name(field.Type)
				if declarations.viewTypes[name] && name != "" && !declarations.actions[name] && !aliases[name] {
					types[name] = true
				}
			}
		}
	}
	return types
}

func appAliases(files []rootFile) map[string]bool {
	aliases := map[string]bool{"app": true}
	for changed := true; changed; {
		changed = false
		for _, f := range files {
			for _, decl := range f.syntax.Decls {
				gen, ok := decl.(*ast.GenDecl)
				if !ok || gen.Tok != token.TYPE {
					continue
				}
				for _, spec := range gen.Specs {
					typeSpec, ok := spec.(*ast.TypeSpec)
					if ok && typeSpec.Assign.IsValid() && appType(typeSpec.Type, aliases) && !aliases[typeSpec.Name.Name] {
						aliases[typeSpec.Name.Name] = true
						changed = true
					}
				}
			}
		}
	}
	return aliases
}

func importNames(syntax *ast.File) map[string]string {
	imports := map[string]string{}
	for _, spec := range syntax.Imports {
		path, _ := strconv.Unquote(spec.Path.Value)
		name := path[strings.LastIndex(path, "/")+1:]
		if spec.Name != nil {
			name = spec.Name.Name
		}
		imports[name] = path
	}
	return imports
}

func expressionPath(expr ast.Expr) string {
	switch expr := expr.(type) {
	case *ast.Ident:
		return expr.Name
	case *ast.SelectorExpr:
		if prefix := expressionPath(expr.X); prefix != "" {
			return prefix + "." + expr.Sel.Name
		}
	}
	return ""
}

func resolvedExpressionPath(expr ast.Expr, imports map[string]string) string {
	path := expressionPath(expr)
	name, rest, _ := strings.Cut(path, ".")
	if importPath := imports[name]; importPath != "" {
		if rest == "" {
			return importPath
		}
		return importPath + "." + rest
	}
	return path
}

func calledIdent(expr ast.Expr) *ast.Ident {
	switch expr := expr.(type) {
	case *ast.Ident:
		return expr
	case *ast.IndexExpr:
		return calledIdent(expr.X)
	case *ast.IndexListExpr:
		return calledIdent(expr.X)
	}
	return nil
}

func uiContextType(expr ast.Expr, imports map[string]string) bool {
	if star, ok := expr.(*ast.StarExpr); ok {
		expr = star.X
	}
	return resolvedExpressionPath(expr, imports) == mygoUI+".Context"
}

func contextNames(syntax ast.Node, imports map[string]string) map[string]bool {
	names := map[string]bool{}
	ast.Inspect(syntax, func(node ast.Node) bool {
		fn, ok := node.(*ast.FuncType)
		if !ok {
			return true
		}
		for _, field := range fn.Params.List {
			if uiContextType(field.Type, imports) {
				for _, name := range field.Names {
					names[name.Name] = true
				}
			}
		}
		return true
	})
	return names
}

func serviceNames(syntax ast.Node, imports map[string]string, contexts map[string]bool) map[string]bool {
	names := map[string]bool{}
	addParams := func(fn *ast.FuncType) {
		for _, field := range fn.Params.List {
			if resolvedExpressionPath(field.Type, imports) == mygoUI+".Services" {
				for _, name := range field.Names {
					names[name.Name] = true
				}
			}
		}
	}
	ast.Inspect(syntax, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.FuncType:
			addParams(node)
		case *ast.AssignStmt:
			for i, value := range node.Rhs {
				if i < len(node.Lhs) && contextServices(value, contexts) {
					if name, ok := node.Lhs[i].(*ast.Ident); ok {
						names[name.Name] = true
					}
				}
			}
		case *ast.ValueSpec:
			for i, value := range node.Values {
				if i < len(node.Names) && contextServices(value, contexts) {
					names[node.Names[i].Name] = true
				}
			}
			if resolvedExpressionPath(node.Type, imports) == mygoUI+".Services" {
				for _, name := range node.Names {
					names[name.Name] = true
				}
			}
		}
		return true
	})
	return names
}

func contextServices(expr ast.Expr, contexts map[string]bool) bool {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return false
	}
	name, method, _ := strings.Cut(expressionPath(call.Fun), ".")
	return contexts[name] && method == "Services"
}

func viewIOImport(path string) bool {
	if path == "net/url" { // URL parsing does no network I/O.
		return false
	}
	for _, blocked := range []string{"os", "io", "net", "log", "syscall", "database", "plugin"} {
		if path == blocked || strings.HasPrefix(path, blocked+"/") {
			return true
		}
	}
	return false
}

func functionUsesDomain(declaration ast.Decl, imports map[string]string, module string) bool {
	found := false
	ast.Inspect(declaration, func(node ast.Node) bool {
		selector, ok := node.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		ident, ok := selector.X.(*ast.Ident)
		if !ok {
			return true
		}
		path := imports[ident.Name]
		found = found || domainPackage(path, module)
		return !found
	})
	return found
}

type functionCandidate struct {
	file        rootFile
	declaration ast.Decl
}

func functionHasLogic(candidate functionCandidate, module string, aliases map[string]bool, logicFunctions map[string]string) bool {
	if functionUsesDomain(candidate.declaration, importNames(candidate.file.syntax), module) {
		return true
	}
	f := candidate.file
	f.syntax = &ast.File{Imports: f.syntax.Imports, Decls: []ast.Decl{candidate.declaration}}
	return len(viewViolationsWithComposition(f, module, logicFunctions, aliases, false, false)) > 0
}

// logicFunctionFiles finds package-level logic and local-state projection
// methods, including helpers and function values in legacy view files. Pure
// UI helpers stay callable; calls through actions are selectors. Repeating
// the check catches wrappers of logic calls.
func logicFunctionFiles(files []rootFile, views map[string]bool, module string, aliases map[string]bool) map[string]string {
	declarations := packageTypesFor(files, views)
	localTypes := localStateTypes(files, views, aliases)
	cleanViews := map[string]bool{}
	for _, f := range files {
		if views[f.path] && len(viewViolations(f, module, nil, aliases, true)) == 0 {
			cleanViews[f.path] = true
		}
	}
	logicFunctions := map[string]string{}
	candidates := map[string]functionCandidate{}
	for _, f := range files {
		if cleanViews[f.path] {
			continue
		}
		for _, decl := range f.syntax.Decls {
			switch decl := decl.(type) {
			case *ast.FuncDecl:
				if decl.Recv != nil {
					if !views[f.path] {
						viewState, domain := projectionResults(decl, importNames(f.syntax), module, declarations)
						for _, receiver := range decl.Recv.List {
							name := declarations.name(receiver.Type)
							if name != "" && !declarations.actions[name] && (domain || viewState && localTypes[name]) {
								method := "." + decl.Name.Name
								logicFunctions[name+method] = f.path
								for alias := range declarations.aliases {
									if declarations.name(ast.NewIdent(alias)) == name {
										logicFunctions[alias+method] = f.path
									}
								}
							}
						}
					}
					continue
				}
				if views[f.path] {
					candidates[decl.Name.Name] = functionCandidate{f, decl}
				} else {
					logicFunctions[decl.Name.Name] = f.path
				}
			case *ast.GenDecl:
				if decl.Tok != token.VAR {
					continue
				}
				for _, spec := range decl.Specs {
					value, ok := spec.(*ast.ValueSpec)
					if !ok {
						continue
					}
					_, functionValue := value.Type.(*ast.FuncType)
					for _, val := range value.Values {
						if _, ok := val.(*ast.FuncLit); ok {
							functionValue = true
						}
					}
					if !functionValue {
						continue
					}
					for _, name := range value.Names {
						if views[f.path] {
							candidates[name.Name] = functionCandidate{f, &ast.GenDecl{Tok: token.VAR, Specs: []ast.Spec{value}}}
						} else {
							logicFunctions[name.Name] = f.path
						}
					}
				}
			}
		}
	}
	for changed := true; changed; {
		changed = false
		for name, candidate := range candidates {
			if logicFunctions[name] == "" && functionHasLogic(candidate, module, aliases, logicFunctions) {
				logicFunctions[name] = candidate.file.path
				changed = true
			}
		}
	}
	return logicFunctions
}

// viewViolations checks boundary properties 1 and 2. logicFunctions maps a
// package-level function or receiver-qualified projection method to its source
// file; calls through an actions interface are not rejected as logic functions.
func viewViolations(f rootFile, module string, logicFunctions map[string]string, aliases map[string]bool, checkImports bool) []string {
	return viewViolationsWithComposition(f, module, logicFunctions, aliases, checkImports, true)
}

func viewViolationsWithComposition(f rootFile, module string, logicFunctions map[string]string, aliases map[string]bool, checkImports, allowSameFileComposition bool) []string {
	var violations []string
	at := func(node ast.Node) string {
		return fmt.Sprintf("%s:%d", f.path, f.fset.Position(node.Pos()).Line)
	}
	add := func(node ast.Node, format string, args ...any) {
		violations = append(violations, at(node)+" "+fmt.Sprintf(format, args...))
	}
	internal := module + "/internal/"
	if checkImports {
		for _, spec := range f.syntax.Imports {
			importPath, _ := strconv.Unquote(spec.Path.Value)
			allowedInternal := importPath == internal+"icons" || importPath == internal+"i18n" || importPath == internal+"overlay/layout"
			switch {
			case importPath == mygoPath || strings.HasPrefix(importPath, mygoPath+"/") || allowedInternal:
				continue
			case strings.HasPrefix(importPath, internal):
				add(spec, "imports domain package %s: move domain access into view state construction or an actions implementation in a non-view file", importPath)
			case viewIOImport(importPath):
				add(spec, "imports I/O package %s: move I/O into an actions implementation in a non-view file", importPath)
			case strings.Contains(strings.Split(importPath, "/")[0], "."):
				add(spec, "imports third-party package %s: views depend on MyGo, the UI kit, view state, actions, internal/icons, internal/i18n and internal/overlay/layout only", importPath)
			}
		}
	}
	for _, decl := range f.syntax.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if fn.Recv != nil {
			for _, field := range fn.Recv.List {
				if appType(field.Type, aliases) {
					add(fn, "declares app method %s: move app access and other areas' fields into view state construction or actions; the view must receive only its state, actions and local UI state", fn.Name.Name)
				}
			}
		}
		for _, field := range fn.Type.Params.List {
			if appType(field.Type, aliases) {
				add(fn, "takes app as a parameter in %s: pass only the view's state, actions and local UI state", fn.Name.Name)
			}
		}
	}
	imports := importNames(f.syntax)
	contexts := contextNames(f.syntax, imports)
	services := serviceNames(f.syntax, imports, contexts)
	ast.Inspect(f.syntax, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.GoStmt:
			add(node, "starts a goroutine: move background work into an actions implementation and return results through a.post")
		case *ast.CallExpr:
			path := resolvedExpressionPath(node.Fun, imports)
			ident := calledIdent(node.Fun)
			if ident != nil {
				if ident.Name == "print" || ident.Name == "println" || imports["."] == "fmt" && strings.HasPrefix(ident.Name, "Print") {
					add(node, "writes output in a view: move I/O into an actions implementation")
				}
				if ident.Name == "helperCall" {
					add(node, "starts helper work in a view: move it into an actions implementation and return through a.post")
				}
				if logicFile := logicFunctions[ident.Name]; logicFile != "" && (!allowSameFileComposition || logicFile != f.path) {
					add(node, "calls logic function %s (%s): pass its result in view state or call a user intent through actions", ident.Name, logicFile)
				}
			}
			if path == "time.AfterFunc" {
				add(node, "schedules background work in a view: move it into an actions implementation and return through a.post")
			}
			if path == "fmt.Print" || path == "fmt.Printf" || path == "fmt.Println" ||
				path == mygoPath+".Dialog.Open" || path == mygoPath+".Dialog.Save" ||
				strings.HasPrefix(path, mygoPath+".Shell.") || strings.HasPrefix(path, mygoPath+".Clipboard.") ||
				path == mygoPath+".App.OpenURL" {
				add(node, "performs I/O in a view: move it into an actions implementation")
			}
			if selector, ok := node.Fun.(*ast.SelectorExpr); ok {
				platform := contextServices(selector.X, contexts)
				if receiver, ok := selector.X.(*ast.Ident); ok {
					platform = platform || contexts[receiver.Name] || services[receiver.Name]
					receiverType := parameterTypeAt(f.syntax, node.Pos(), receiver.Name)
					method := receiverType + "." + selector.Sel.Name
					if logicFile := logicFunctions[method]; logicFile != "" {
						add(node, "calls logic method %s (%s): pass its result in view state or call a user intent through actions", method, logicFile)
					}
				}
				if platform {
					switch selector.Sel.Name {
					case "ReadClipboard", "ReadClipboardData", "ClipboardFormats", "WriteClipboard", "WriteClipboardData", "OpenURL", "OpenURLThen":
						add(node, "performs platform I/O in a view: move it into an actions implementation")
					}
				}
			}
		}
		return true
	})
	return violations
}

func readViewAllowlist(t *testing.T) map[string]int {
	t.Helper()
	file, err := os.Open("view-allowlist.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	list := map[string]int{}
	scanner := bufio.NewScanner(file)
	for line := 1; scanner.Scan(); line++ {
		text, _, _ := strings.Cut(scanner.Text(), "#")
		fields := strings.Fields(text)
		if len(fields) == 0 {
			continue
		}
		if len(fields) != 1 {
			t.Fatalf("view-allowlist.txt:%d: want \"path  # area\", got %q", line, scanner.Text())
		}
		if previous, ok := list[fields[0]]; ok {
			t.Fatalf("view-allowlist.txt:%d: %s is on line %d already", line, fields[0], previous)
		}
		list[fields[0]] = line
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return list
}

// TestViewFiles keeps view files on the UI side of the boundary. Existing
// violations occupy one line per file in view-allowlist.txt; migration removes
// that line, and the list never grows.
func TestViewFiles(t *testing.T) {
	_, module := moduleRoot(t)
	files := parseRootFiles(t)
	views := map[string]bool{}
	for _, f := range files {
		views[f.path] = viewFile(f.path, f.syntax)
	}
	aliases := appAliases(files)
	logicFunctions := logicFunctionFiles(files, views, module, aliases)
	allow := readViewAllowlist(t)
	if len(allow) != viewAllowlistSize {
		t.Errorf("view-allowlist.txt has %d entries, want %d: remove entries and lower viewAllowlistSize as areas migrate; the list and its size may only shrink", len(allow), viewAllowlistSize)
	}
	seen := map[string]bool{}
	var failures []string
	for _, f := range files {
		if !views[f.path] {
			continue
		}
		violations := viewViolations(f, module, logicFunctions, aliases, true)
		line, listed := allow[f.path]
		if listed {
			seen[f.path] = true
			if len(violations) == 0 {
				failures = append(failures, fmt.Sprintf("%s is clean: remove its line from internal/archtest/view-allowlist.txt (line %d)", f.path, line))
			}
			continue
		}
		failures = append(failures, violations...)
	}
	for path, line := range allow {
		if !seen[path] {
			failures = append(failures, fmt.Sprintf("%s is not a current view file: remove its line from internal/archtest/view-allowlist.txt (line %d)", path, line))
		}
	}
	slices.Sort(failures)
	for _, failure := range failures {
		t.Errorf("%s; do not add new entries to internal/archtest/view-allowlist.txt: it only shrinks as areas migrate", failure)
	}
}

func TestViewFileRules(t *testing.T) {
	const module = "example.test/Voxt"
	source := `package main
import (f "fmt"; _ "example.test/Voxt/internal/store")
import ("github.com/egoist/mygo"; "github.com/egoist/mygo/ui")
type appAlias = app
func (b *appAlias) paint(c *ui.Context) {
	go func() {}()
	settingsName(); logic[string](); f.Println("probe")
	c.Services().OpenURL("https://example.com")
	mygo.Clipboard.Write("probe"); mygo.Clipboard.Read()
	mygo.Shell.OpenExternal("https://example.com"); mygo.Shell.OpenPath("probe")
	mygo.Clipboard.ReadText(); mygo.Clipboard.WriteText("probe")
	mygo.Clipboard.Formats(); mygo.Clipboard.ReadFormat(nil)
	mygo.Clipboard.ReadFiles(); mygo.Clipboard.WriteFiles("probe"); mygo.Clipboard.Flush()
	c.WriteClipboard("probe"); c.ReadClipboard(); c.OpenURLThen("https://example.com", nil)
	var s ui.Services
	s.WriteClipboard("probe"); s.OpenURL("https://example.com")
	actions.ShowItemInFolder("probe"); state.Dialog.Open(); c.Services().Invalidate()
}
`
	f := parseRootFile(t, "probe_page.go", source)
	aliases := appAliases([]rootFile{f})
	logic := map[string]string{"settingsName": "settings_filter.go", "logic": "logic.go"}
	got := viewViolations(f, module, logic, aliases, true)
	violations := strings.Join(got, "\n")
	for _, want := range []string{"internal/store", "declares app method paint", "starts a goroutine", "calls logic function settingsName", "calls logic function logic", "performs I/O", "performs platform I/O"} {
		if !strings.Contains(violations, want) {
			t.Errorf("no violation containing %q:\n%s", want, violations)
		}
	}
	if len(got) != 23 {
		t.Errorf("got %d violations, want 23 (actions, local dialog state and Invalidate are allowed):\n%s", len(got), violations)
	}
	if !viewFile("probe_page.go", f.syntax) {
		t.Error("a MyGo import must mark a root file as a view")
	}
	if viewFile("probe_page_test.go", f.syntax) {
		t.Error("a test file is not a view")
	}
}

func TestViewFileRulesRejectLogicMethodsOnLocalState(t *testing.T) {
	const module = "example.test/Voxt"
	view := parseRootFile(t, "probe_page.go", `package main
import "github.com/egoist/mygo/ui"
import _ "example.test/Voxt/internal/hotkey"
type overlayForm struct{}; type overlayAlias = overlayForm; type renamedForm struct{}; type receiverForm struct{}; type actionNamedForm struct{}; type searchLocalUI struct{}
func paint(c *ui.Context, local *overlayForm) { local.previewState(); local.layout(); local.overlay() }
func paintRenamed(context *ui.Context, form *renamedForm, a *probeActions, local *probeActions, controls probeViewActions, actions *actionNamedForm) {
	context.Theme(); form.overlay(); form.aliased(); form.slice()
	form.pointers(); a.overlay(); local.overlay(); controls.overlay(); actions.overlay()
}
func (renamed *receiverForm) paint() { renamed.previewState() }
func paintAlias(alias *overlayAlias) { alias.previewState() }
func paintGeneration(form *genForm) { form.settings() }
func paintDemo(local *meetingDemoHost) { local.modelsViewState() }
func paintSearch(local *searchLocalUI) { local.previewState(); local.overlay() }
func compose(c *ui.Context, local *overlayForm) { paint(c, local) }
func direct() { buildFeatureCustomKeysViewState() }
`)
	clean := parseRootFile(t, "clean_page.go", `package main
import "github.com/egoist/mygo/ui"
func cleanView(c *ui.Context, local *overlayForm) { compose(c, local) }
`)
	logic := parseRootFile(t, "settings_overlay_view_state.go", `package main
import "example.test/Voxt/internal/settings"
type settingsOverlayPreviewState struct{}; type overlayResult = settings.Overlay; type genForm struct{}; type meetingDemoHost struct{}; type modelsViewState struct{}; type probeActions struct{}; type probeViewActions interface { overlay() *settings.Overlay }
func (f *overlayForm) previewState() settingsOverlayPreviewState { return settingsOverlayPreviewState{} }; func (f *overlayForm) layout() {}; func (f *overlayForm) overlay() *settings.Overlay { return nil }
func (f *renamedForm) overlay() *settings.Overlay { return nil }; func (f *renamedForm) aliased() overlayResult { return overlayResult{} }; func (f *renamedForm) slice() []*settings.Overlay { return nil }; func (f *renamedForm) pointers() **settings.Overlay { return nil }
func (f *actionNamedForm) overlay() *settings.Overlay { return nil }; func (f *receiverForm) previewState() settingsOverlayPreviewState { return settingsOverlayPreviewState{} }; func (f *genForm) settings() (g settings.Generation, bad string) { return }; func (host *meetingDemoHost) modelsViewState() *modelsViewState { return nil }
func (s *searchLocalUI) previewState() {}; func (s *searchLocalUI) overlay() {}; func (a *probeActions) overlay() *settings.Overlay { return nil }; func buildFeatureCustomKeysViewState() {}
`)
	files := []rootFile{view, clean, logic}
	views := map[string]bool{"probe_page.go": true, "clean_page.go": true}
	aliases := appAliases(files)
	if localStateTypes(files, views, aliases)["probeActions"] || localStateTypes(files, views, aliases)["probeViewActions"] {
		t.Error("declared action types must not become local state through parameter names")
	}
	logicFunctions := logicFunctionFiles(files, views, module, aliases)
	got := viewViolations(view, module, logicFunctions, aliases, true)
	violations := strings.Join(got, "\n")
	for _, want := range []string{"internal/hotkey", "calls logic method overlayForm.previewState", "calls logic method overlayAlias.previewState", "calls logic method overlayForm.overlay", "calls logic method renamedForm.overlay", "calls logic method renamedForm.aliased", "calls logic method renamedForm.slice", "calls logic method renamedForm.pointers", "calls logic method actionNamedForm.overlay", "calls logic method receiverForm.previewState", "calls logic method genForm.settings", "calls logic function buildFeatureCustomKeysViewState"} {
		if !strings.Contains(violations, want) {
			t.Errorf("no violation containing %q:\n%s", want, violations)
		}
	}
	if len(got) != 12 || strings.Contains(violations, "calls logic function paint") {
		t.Errorf("got %q, want 12 genuine violations; actions, meetingDemoHost composition, ordinary local UI methods and same-view composition are allowed", got)
	}
	cleanViolations := viewViolations(clean, module, logicFunctions, aliases, true)
	if len(cleanViolations) != 1 || !strings.Contains(cleanViolations[0], "calls logic function compose (probe_page.go)") {
		t.Fatalf("clean view violations are %q, want logic propagated through the same-file wrapper in probe_page.go", cleanViolations)
	}
	ordered := parseRootFile(t, "ordered.go", `package main
import ("example.test/Voxt/internal/settings"; "example.test/Voxt/internal/i18n")
func ordered() { settings.Normalize(); i18n.T("ok") }
`)
	declaration := ordered.syntax.Decls[len(ordered.syntax.Decls)-1]
	if !functionUsesDomain(declaration, importNames(ordered.syntax), module) {
		t.Error("an allowed selector after a domain selector must not clear domain use")
	}
}
