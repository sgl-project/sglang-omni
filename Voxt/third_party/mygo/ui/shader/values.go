package shader

import (
	"fmt"
	"math"

	"github.com/egoist/mygo/internal/shaderir"
	"github.com/egoist/mygo/ui"
)

// Uniform is a uniform variable of a shader, found by its name once
// (Shader.Uniform), which Values sets.
type Uniform struct {
	s    *Shader
	name string
	u    uniform
}

// Uniform returns the shader's uniform variable name, which Values sets
// by: look it up once, as a package variable or as the view starts. It
// panics when the shader has no such variable, as a name misspelled.
func (s *Shader) Uniform(name string) Uniform {
	u, ok := s.uniforms[name]
	if !ok || name == "" || name[0] == '_' {
		panic(fmt.Sprintf("shader: no uniform variable %s", name))
	}
	return Uniform{s: s, name: name, u: u}
}

// Values are the values of a shader's uniform variables, which
// Options.Values draws with, as Flutter's FragmentShader and Skia's
// SkRuntimeShaderBuilder hold them: set them once or each frame, by
// Uniform, and draw with them as often as needed; drawing copies them.
// Variables not set are zero. The setters panic for a variable of another
// type or another shader, and return v.
type Values struct {
	s      *Shader
	words  []uint32
	colors []colorValue
}

// colorValue is a color set to a vector (Values.Color), which a wide
// gamut draws in extended sRGB.
type colorValue struct {
	u uniform
	c ui.Color
}

// NewValues returns values for the shader's uniform variables, all zero.
func (s *Shader) NewValues() *Values {
	return &Values{s: s, words: make([]uint32, s.words)}
}

// check panics unless u is a variable of v's shader of a type ok accepts,
// named what.
func (v *Values) check(u Uniform, what string, ok func(t shaderir.Type) bool) {
	if u.s != v.s {
		panic(fmt.Sprintf("shader: %s is not a uniform variable of the values' shader", u.name))
	}
	if !ok(u.u.t) {
		panic(fmt.Sprintf("shader: %s is a %s, not %s", u.name, u.u.t.String(), what))
	}
}

func is(m shaderir.BasicType) func(shaderir.Type) bool {
	return func(t shaderir.Type) bool { return t.Main == m }
}

// lanesOf accepts the types whose lanes are numbers of kind k: scalars,
// vectors, matrices and arrays of them.
func lanesOf(k shaderir.BasicType) func(shaderir.Type) bool {
	return func(t shaderir.Type) bool {
		if t.Main == shaderir.Array {
			t = t.Sub[0]
		}
		return t.Scalar() == k
	}
}

// Float sets a float.
func (v *Values) Float(u Uniform, x float32) *Values {
	v.check(u, "a float", is(shaderir.Float))
	v.words[u.u.word] = math.Float32bits(x)
	return v
}

// Vec2 sets a vec2.
func (v *Values) Vec2(u Uniform, x, y float32) *Values {
	v.check(u, "a vec2", is(shaderir.Vec2))
	v.floats(u, x, y)
	return v
}

// Vec3 sets a vec3.
func (v *Values) Vec3(u Uniform, x, y, z float32) *Values {
	v.check(u, "a vec3", is(shaderir.Vec3))
	v.floats(u, x, y, z)
	return v
}

// Vec4 sets a vec4.
func (v *Values) Vec4(u Uniform, x, y, z, w float32) *Values {
	v.check(u, "a vec4", is(shaderir.Vec4))
	v.floats(u, x, y, z, w)
	return v
}

// Floats sets the components of a vector, a matrix (by columns) or an
// array of floats or of them (elements in order), from the first; it
// panics for more numbers than the variable holds.
func (v *Values) Floats(u Uniform, xs ...float32) *Values {
	v.check(u, "of floats", lanesOf(shaderir.Float))
	v.floats(u, xs...)
	return v
}

func (v *Values) floats(u Uniform, xs ...float32) {
	v.fits(u, len(xs))
	for i, x := range xs {
		v.words[u.u.word+i] = math.Float32bits(x)
	}
}

// Int sets an int.
func (v *Values) Int(u Uniform, x int32) *Values {
	v.check(u, "an int", is(shaderir.Int))
	v.words[u.u.word] = uint32(x)
	return v
}

// Ints sets the components of an ivec or an array of ints or of them, as
// Floats does.
func (v *Values) Ints(u Uniform, xs ...int32) *Values {
	v.check(u, "of ints", lanesOf(shaderir.Int))
	v.fits(u, len(xs))
	for i, x := range xs {
		v.words[u.u.word+i] = uint32(x)
	}
	return v
}

// Uint sets a uint.
func (v *Values) Uint(u Uniform, x uint32) *Values {
	v.check(u, "a uint", is(shaderir.Uint))
	v.words[u.u.word] = x
	return v
}

// Uints sets the components of a uvec or an array of uints or of them, as
// Floats does.
func (v *Values) Uints(u Uniform, xs ...uint32) *Values {
	v.check(u, "of uints", lanesOf(shaderir.Uint))
	v.fits(u, len(xs))
	copy(v.words[u.u.word:], xs)
	return v
}

// Bool sets a bool.
func (v *Values) Bool(u Uniform, b bool) *Values {
	v.check(u, "a bool", is(shaderir.Bool))
	v.words[u.u.word] = 0
	if b {
		v.words[u.u.word] = 1
	}
	return v
}

// Color sets a vec3 or vec4 to c's components, from 0 to 1, not
// premultiplied: its sRGB ones, or, where the window draws a wide gamut,
// those of a color outside sRGB in extended sRGB (below 0 or above 1).
func (v *Values) Color(u Uniform, c ui.Color) *Values {
	v.check(u, "a vec3 or vec4", func(t shaderir.Type) bool { return t.Main == shaderir.Vec3 || t.Main == shaderir.Vec4 })
	writeColor(v.words[u.u.word:u.u.word+u.u.n], straightSRGB(c))
	for i := range v.colors {
		if v.colors[i].u.word == u.u.word {
			v.colors[i].c = c
			return v
		}
	}
	v.colors = append(v.colors, colorValue{u.u, c})
	return v
}

// fits panics unless u holds n numbers.
func (v *Values) fits(u Uniform, n int) {
	if n > u.u.n {
		panic(fmt.Sprintf("shader: %d numbers for %s, a %s of %d", n, u.name, u.u.t.String(), u.u.n))
	}
}

// straightSRGB returns c's sRGB components, from 0 to 1.
func straightSRGB(c ui.Color) [4]float32 {
	return [4]float32{float32(c.R) / 255, float32(c.G) / 255, float32(c.B) / 255, float32(c.A) / 255}
}

// writeColor writes the components of a color into the lanes of a vec3
// or vec4.
func writeColor(lanes []uint32, c [4]float32) {
	for i := range lanes {
		lanes[i] = math.Float32bits(c[i])
	}
}
