package glsl

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/egoist/mygo/internal/shaderir"
)

// Options are how Fragment writes a program for a renderer.
type Options struct {
	// FlipY is a boolean expression of the renderer's, true where GL's
	// window y counts up, as the position's does not, which negates dFdy;
	// "" where it never does.
	FlipY string
	// Block, unless empty, names a uniform block, laid out as std140
	// (UniformLanes says how), whose members the uniforms are.
	Block string
	// FlipRows, unless empty, names a uniform ivec4 that the renderer sets,
	// which gives, for each texture, the height of an image whose rows go
	// up, as GL's framebuffers draw them, or 0: texel reads flip those.
	FlipRows string
}

// Fragment returns p's fragment entry point as a function named name of
// GLSL 3.30 and GLSL ES 3.00, for a renderer's fragment shader to call,
// after the uniforms (U0, U1, ...), the textures (sampler2D T0, T1, ...)
// and the functions it reaches:
//
//	vec4 name(vec4 position, ...varyings)
//
// The position counts its y from the top of the target, as the renderer
// passes it.
func Fragment(p *shaderir.Program, name string, o Options) string {
	lib, entry, funcs := p.FragmentLibrary()
	if entry == nil {
		return ""
	}
	c := &compileContext{version: GLSLVersionES300, projectionMatrixUniformIndex: -1, structNames: map[string]string{}, library: true, flipY: o.FlipY, flipRows: o.FlipRows}

	lines := []string{"{{.Structs}}", ""}
	if o.Block != "" && len(lib.Uniforms) > 0 {
		lines = append(lines, fmt.Sprintf("layout(std140) uniform %s {", o.Block))
		for i, t := range lib.Uniforms {
			lines = append(lines, fmt.Sprintf("	%s;", c.varDecl(lib, &t, fmt.Sprintf("U%d", i))))
		}
		lines = append(lines, "};")
	} else {
		for i, t := range lib.Uniforms {
			lines = append(lines, fmt.Sprintf("uniform %s;", c.varDecl(lib, &t, fmt.Sprintf("U%d", i))))
		}
	}
	for i := 0; i < lib.TextureCount; i++ {
		lines = append(lines, fmt.Sprintf("uniform sampler2D T%d;", i))
	}
	if o.FlipRows != "" {
		lines = append(lines, fmt.Sprintf("uniform ivec4 %s;", o.FlipRows), "",
			"ivec2 flipRows(ivec2 p, int h) {",
			"\treturn h > 0 ? ivec2(p.x, h - 1 - p.y) : p;",
			"}")
	}
	lines = append(lines, "")
	for _, f := range funcs {
		lines = append(lines, c.function(lib, f, true)...)
	}
	for _, f := range funcs {
		lines = append(lines, "")
		lines = append(lines, c.function(lib, f, false)...)
	}

	var params, args []string
	for i, t := range entry.InParams {
		n := "position"
		if i > 0 {
			n = fmt.Sprintf("v%d", i-1)
		}
		params, args = append(params, c.varDecl(lib, &t, n)), append(args, n)
	}
	lines = append(lines, "",
		fmt.Sprintf("vec4 %s(%s) {", name, strings.Join(params, ", ")),
		fmt.Sprintf("\treturn F%d(%s);", entry.Index, strings.Join(args, ", ")),
		"}")

	ls := strings.Join(lines, "\n")
	var st []string
	for i, t := range c.structTypes {
		st = append(st, fmt.Sprintf("struct S%d {", i))
		for j, m := range t.Sub {
			st = append(st, fmt.Sprintf("\t%s;", c.varDecl(lib, &m, fmt.Sprintf("M%d", j))))
		}
		st = append(st, "};")
	}
	ls = strings.ReplaceAll(ls, "{{.Structs}}", strings.Join(st, "\n"))
	ls = regexp.MustCompile(`\n\n+`).ReplaceAllString(ls, "\n\n")
	return strings.TrimSpace(ls) + "\n"
}
