// Package tsgen generates a typed TypeScript client for Go services bound
// with mygo.Bind and events declared with mygo.NewEvent.
//
// Types are derived with reflection following the encoding/json/v2 rules
// used on the wire. Parameter names, documentation and enum values are read
// from the Go sources when they are available.
package tsgen

import (
	"bytes"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
)

// Model describes the API to generate a client for.
type Model struct {
	Services []Service
	Events   []Event
}

// Service is a bound Go value whose methods are callable from the page.
type Service struct {
	Name    string
	Type    reflect.Type
	Methods []Method
}

// Method is a bound method. Params excludes the receiver and an optional
// leading context.Context (HasCtx).
type Method struct {
	Name   string
	Params []reflect.Type
	// Channels marks the parameters that are channels (mygo.Channel),
	// whose Params are the type of their values.
	Channels []bool
	Variadic bool
	Result   reflect.Type
	HasCtx   bool
	// PC is the entry of the method's code, used to find its source.
	PC uintptr
}

// Event is a typed event sent from Go to pages.
type Event struct {
	Name string
	Type reflect.Type
	// PC, File and Line locate the declaration, used to find its package
	// and documentation.
	PC   uintptr
	File string
	Line int
}

type generator struct {
	src   *sources
	names map[reflect.Type]string
	taken map[string]reflect.Type
	decls map[string]string
	busy  map[reflect.Type]bool
	// imported are the types of mygo-runtime the client uses.
	imported map[string]bool
}

// runtimeTypes are the types of package mygo whose JSON is a type of
// mygo-runtime, by the Go type's package path and name: the client
// imports that type instead of declaring one, which their JSON methods
// would leave unknown.
var runtimeTypes = map[string]string{
	"github.com/egoist/mygo.ThemePreferences": "ThemePreferences",
}

// runtimeSentinel holds the names of runtimeTypes in generator.taken, so
// that types of the app with those names get others.
type runtimeSentinel struct{}

// Generate renders the TypeScript client.
func Generate(m Model) ([]byte, error) {
	g := &generator{
		src:   newSources(),
		names: map[reflect.Type]string{},
		taken: map[string]reflect.Type{},
		decls: map[string]string{},
		busy:  map[reflect.Type]bool{},

		imported: map[string]bool{},
	}
	for _, name := range runtimeTypes {
		g.taken[name] = reflect.TypeFor[runtimeSentinel]()
	}

	// Learn package directories first so type docs can be found.
	type methodSrc struct {
		file string
		line int
	}
	locs := map[*Method]methodSrc{}
	for si := range m.Services {
		for mi := range m.Services[si].Methods {
			meth := &m.Services[si].Methods[mi]
			file, line := g.src.learnPC(meth.PC)
			locs[meth] = methodSrc{file, line}
		}
	}
	for _, e := range m.Events {
		g.src.learnPC(e.PC)
	}
	g.src.resolvePackages(collectPackages(m))

	var services bytes.Buffer
	channels := false
	for _, s := range m.Services {
		base := s.Type
		if base.Kind() == reflect.Pointer {
			base = base.Elem()
		}
		writeDoc(&services, "", g.src.typeDoc(base))
		fmt.Fprintf(&services, "export const %s = {\n", s.Name)
		for i := range s.Methods {
			meth := &s.Methods[i]
			loc := locs[meth]
			fd := g.src.funcDecl(loc.file, loc.line, meth.Name)
			var names []string
			doc := ""
			if fd != nil {
				names = paramNames(fd)
				if meth.HasCtx && len(names) > 0 {
					names = names[1:]
				}
				if fd.Doc != nil {
					doc = fd.Doc.Text()
				}
			}
			var params, args []string
			used := map[string]bool{}
			for pi, pt := range meth.Params {
				name := ""
				if pi < len(names) {
					name = names[pi]
				}
				name = safeName(or(name, fmt.Sprintf("arg%d", pi)))
				for used[name] {
					name += "_"
				}
				used[name] = true
				if meth.Variadic && pi == len(meth.Params)-1 {
					params = append(params, "..."+name+": "+arrayOf(g.ts(pt.Elem())))
					args = append(args, "..."+name)
					continue
				}
				typ := g.ts(pt)
				if pi < len(meth.Channels) && meth.Channels[pi] {
					typ = "Channel<" + typ + ">"
					channels = true
				}
				params = append(params, name+": "+typ)
				args = append(args, name)
			}
			result := "void"
			if meth.Result != nil {
				result = g.ts(meth.Result)
			}
			writeDoc(&services, "  ", doc)
			call := quote(s.Name + "." + meth.Name)
			if len(args) > 0 {
				call += ", " + strings.Join(args, ", ")
			}
			fmt.Fprintf(&services, "  %s(%s): Promise<%s> {\n    return call(%s);\n  },\n",
				camel(meth.Name), strings.Join(params, ", "), result, call)
		}
		services.WriteString("} as const;\n\n")
	}

	var events bytes.Buffer
	if len(m.Events) > 0 {
		events.WriteString("/** Events sent by the Go side. Subscribe with `events.name.on(listener)`. */\n")
		events.WriteString("export const events = {\n")
		for _, e := range m.Events {
			writeDoc(&events, "  ", g.src.valueDoc(e.File, e.Line))
			fmt.Fprintf(&events, "  %s: event<%s>(%s),\n", eventKey(e.Name), g.ts(e.Type), quote(e.Name))
		}
		events.WriteString("} as const;\n")
	}

	var out bytes.Buffer
	out.WriteString(header)
	// The runtime comes from the mygo-runtime package.
	var imports []string
	if services.Len() > 0 {
		imports = append(imports, "call")
	}
	if events.Len() > 0 {
		imports = append(imports, "event")
	}
	if channels {
		imports = append(imports, "type Channel")
	}
	for _, name := range slices.Sorted(maps.Keys(g.imported)) {
		imports = append(imports, "type "+name)
	}
	if len(imports) > 0 {
		fmt.Fprintf(&out, "import { %s } from %q;\n", strings.Join(imports, ", "), RuntimePackage)
	}
	if len(g.decls) > 0 {
		out.WriteString("\n// ---- Types ----\n\n")
		names := make([]string, 0, len(g.decls))
		for n := range g.decls {
			names = append(names, n)
		}
		slices.Sort(names)
		for _, n := range names {
			out.WriteString(g.decls[n])
			out.WriteByte('\n')
		}
	}
	if services.Len() > 0 {
		if len(g.decls) == 0 {
			out.WriteByte('\n')
		}
		out.WriteString("// ---- Services ----\n\n")
		out.Write(services.Bytes())
	}
	if events.Len() > 0 {
		out.WriteString("// ---- Events ----\n\n")
		out.Write(events.Bytes())
	}
	return bytes.TrimRight(out.Bytes(), "\n"), nil
}

func or(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}

// collectPackages lists the packages of all named types reachable from m.
func collectPackages(m Model) []string {
	seen := map[reflect.Type]bool{}
	pkgs := map[string]bool{}
	var visit func(t reflect.Type)
	visit = func(t reflect.Type) {
		if t == nil || seen[t] {
			return
		}
		seen[t] = true
		if t.PkgPath() != "" {
			pkgs[t.PkgPath()] = true
		}
		if _, ok := special(t); ok {
			return
		}
		switch t.Kind() {
		case reflect.Pointer, reflect.Slice, reflect.Array:
			visit(t.Elem())
		case reflect.Map:
			visit(t.Key())
			visit(t.Elem())
		case reflect.Struct:
			for _, f := range structFields(t) {
				visit(f.owner)
				visit(f.typ)
			}
		}
	}
	for _, s := range m.Services {
		visit(s.Type)
		for _, meth := range s.Methods {
			for _, p := range meth.Params {
				visit(p)
			}
			visit(meth.Result)
		}
	}
	for _, e := range m.Events {
		visit(e.Type)
	}
	out := make([]string, 0, len(pkgs))
	for p := range pkgs {
		out = append(out, p)
	}
	slices.Sort(out)
	return out
}

func arrayOf(t string) string {
	if strings.ContainsAny(t, "|&") || strings.HasPrefix(t, "(") {
		return "(" + t + ")[]"
	}
	return t + "[]"
}

// ts returns the TypeScript type for t, declaring named types as needed.
func (g *generator) ts(t reflect.Type) string {
	// Pointers first: *T has the JSON methods of T in its method set.
	if t.Kind() == reflect.Pointer {
		inner := g.ts(t.Elem())
		if strings.HasSuffix(inner, " | null") {
			return inner
		}
		return inner + " | null"
	}
	if name, ok := runtimeTypes[t.PkgPath()+"."+t.Name()]; ok && t.Name() != "" {
		g.imported[name] = true
		return name
	}
	if s, ok := special(t); ok {
		return s
	}
	switch t.Kind() {
	case reflect.Bool:
		return g.alias(t, "boolean")
	case reflect.String:
		return g.alias(t, "string")
	case reflect.Interface:
		return "unknown"
	case reflect.Slice, reflect.Array:
		return g.alias(t, "")
	case reflect.Map:
		return g.alias(t, "")
	case reflect.Struct:
		if t.Name() == "" {
			return g.inlineObject(t)
		}
		return g.declareStruct(t)
	}
	if isNumber(t.Kind()) {
		return g.alias(t, "number")
	}
	return "unknown"
}

// structural renders the underlying TypeScript type of t, ignoring its name.
func (g *generator) structural(t reflect.Type) string {
	switch t.Kind() {
	case reflect.Slice, reflect.Array:
		return arrayOf(g.ts(t.Elem()))
	case reflect.Map:
		key := "string"
		if isNumber(t.Key().Kind()) {
			if _, custom := special(t.Key()); !custom {
				key = "number"
			}
		}
		return "Record<" + key + ", " + g.ts(t.Elem()) + ">"
	}
	return "unknown"
}

// alias returns base for unnamed types and declares an exported alias for
// named ones. Named string and number types with constants become unions.
func (g *generator) alias(t reflect.Type, base string) string {
	if t.Name() == "" || t.PkgPath() == "" {
		if base == "" {
			return g.structural(t)
		}
		return base
	}
	if name, ok := g.names[t]; ok {
		return name
	}
	name := g.nameFor(t)
	g.busy[t] = true
	var body string
	if values := g.src.enumValues(t); len(values) > 0 {
		body = strings.Join(values, " | ")
	} else if base != "" {
		body = base
	} else {
		body = g.structural(t)
	}
	var b bytes.Buffer
	writeDoc(&b, "", g.src.typeDoc(t))
	fmt.Fprintf(&b, "export type %s = %s;\n", name, body)
	g.decls[name] = b.String()
	return name
}

func (g *generator) nameFor(t reflect.Type) string {
	if name, ok := g.names[t]; ok {
		return name
	}
	name := typeName(t.Name())
	if other, taken := g.taken[name]; taken && other != t {
		pkg := t.PkgPath()
		if i := strings.LastIndex(pkg, "/"); i >= 0 {
			pkg = pkg[i+1:]
		}
		base := typeName(pkg) + name
		name = base
		for n := 2; ; n++ {
			if _, taken := g.taken[name]; !taken {
				break
			}
			name = fmt.Sprintf("%s%d", base, n)
		}
	}
	g.taken[name] = t
	g.names[t] = name
	return name
}

func (g *generator) declareStruct(t reflect.Type) string {
	name := g.nameFor(t)
	if _, done := g.decls[name]; done || g.busy[t] {
		return name
	}
	g.busy[t] = true
	var b bytes.Buffer
	writeDoc(&b, "", g.src.typeDoc(t))
	fmt.Fprintf(&b, "export interface %s %s\n", name, g.object(t, ""))
	g.decls[name] = b.String()
	return name
}

// inlineObject renders an anonymous struct on one line.
func (g *generator) inlineObject(t reflect.Type) string {
	fields := structFields(t)
	if len(fields) == 0 {
		return "{}"
	}
	parts := make([]string, 0, len(fields))
	for _, f := range fields {
		typ, opt := g.fieldType(f)
		parts = append(parts, propertyName(f.name)+opt+": "+typ)
	}
	return "{ " + strings.Join(parts, "; ") + " }"
}

func (g *generator) fieldType(f field) (typ, opt string) {
	typ = g.ts(f.typ)
	if f.quoted && isNumber(f.typ.Kind()) {
		typ = "string"
	}
	if f.optional {
		// Omitted rather than null.
		return strings.TrimSuffix(typ, " | null"), "?"
	}
	return typ, ""
}

// object renders the members of a struct as an object type literal.
func (g *generator) object(t reflect.Type, indent string) string {
	fields := structFields(t)
	if len(fields) == 0 {
		return "{}"
	}
	var b bytes.Buffer
	b.WriteString("{\n")
	for _, f := range fields {
		typ, opt := g.fieldType(f)
		writeDoc(&b, indent+"  ", g.src.fieldDoc(f.owner, f.goName))
		fmt.Fprintf(&b, "%s  %s%s: %s;\n", indent, propertyName(f.name), opt, typ)
	}
	b.WriteString(indent + "}")
	return b.String()
}

// writeDoc writes a Go doc comment as JSDoc.
func writeDoc(b *bytes.Buffer, indent, doc string) {
	doc = strings.TrimSpace(doc)
	if doc == "" {
		return
	}
	doc = strings.ReplaceAll(doc, "*/", "*\\/")
	lines := strings.Split(doc, "\n")
	if len(lines) == 1 {
		fmt.Fprintf(b, "%s/** %s */\n", indent, lines[0])
		return
	}
	fmt.Fprintf(b, "%s/**\n", indent)
	for _, l := range lines {
		if l == "" {
			fmt.Fprintf(b, "%s *\n", indent)
		} else {
			fmt.Fprintf(b, "%s * %s\n", indent, l)
		}
	}
	fmt.Fprintf(b, "%s */\n", indent)
}

const header = "// Code generated by `mygo generate`. DO NOT EDIT.\n\n"

// RuntimePackage is the npm package of the page runtime, which generated
// clients import.
const RuntimePackage = "mygo-runtime"
