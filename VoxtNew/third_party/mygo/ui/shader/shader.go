// Package shader draws with shaders an app writes: fragment shaders in
// Go's syntax, which run on the GPU, whichever renderer draws the window
// (Metal, Direct3D 11, OpenGL or WebGL 2), and on the CPU where none can.
//
// A shader is a Go file of package main whose function Fragment returns
// the color of a pixel, premultiplied, from its position:
//
//	package main
//
//	var Time float
//
//	func Fragment(dst vec4, pos vec2) vec4 {
//		v := 0.5 + 0.5*sin(pos.x/20+Time)
//		return vec4(v, v, 1, 1)
//	}
//
// dst is the pixel's center in pixels of the window (or the canvas drawn
// into), and pos the same point in DIPs from the top left corner of the
// rectangle the shader fills. The language is Ebitengine's Kage, whose
// compiler MyGo's is forked from: Go's syntax with the types bool, int,
// uint, float, vec2 to vec4, ivec2 to ivec4, uvec2 to uvec4, mat2 to mat4
// and arrays of them, GLSL's built-in functions (sin, mix, smoothstep,
// dot, ...), functions of its own, if, and for loops. Exported variables
// at the package level are uniforms, which Options.Uniforms sets by name,
// or Values by variables looked up once (Shader.Uniform).
//
// A shader reads up to four images, bitmaps or canvases (ui.Canvas),
// through these functions, in the images' pixels from their top left
// corners:
//
//	image0At(p vec2) vec4      // the pixel at p, transparent outside
//	image0Sample(p vec2) vec4  // blended between the pixels around p
//	image0Size() vec2          // the size in pixels
//
// and image1At to image3At and so on. rectSize returns the size of the
// rectangle the shader fills, in DIPs, and pixelScale the pixels a DIP.
//
// A shader that draws into a canvas makes the input of another, as the
// passes of an effect: one computes a field into a small canvas, the next
// reads it to draw the window.
package shader

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"reflect"
	"regexp"
	"sync"

	"github.com/egoist/mygo/internal/scene"
	"github.com/egoist/mygo/internal/shader"
	"github.com/egoist/mygo/internal/shaderir"
	"github.com/egoist/mygo/internal/shaderir/interp"
	"github.com/egoist/mygo/ui"
)

// Shader is a compiled shader, which draws in any window, from any
// goroutine's views.
type Shader struct {
	prog     *interp.Program
	fx       *scene.Effect
	uniforms map[string]uniform
	words    int
	kind     kind
	backdrop bool // it reads its backdrop
}

// kind is what a shader draws, by its entry point.
type kind uint8

const (
	fragment kind = iota // Fragment: a fill
	colorEffect
	distortionEffect
	layerEffect
)

// entries are the entry points of the kinds, and the Fragment that calls
// an effect's, from its content (layerAt, layerSample).
var entries = [...]struct{ name, wrap string }{
	fragment:         {"Fragment", ""},
	colorEffect:      {"ColorEffect", "return ColorEffect(q, layerAt(q))"},
	distortionEffect: {"DistortionEffect", "return layerSample(DistortionEffect(q))"},
	layerEffect:      {"LayerEffect", "return LayerEffect(q)"},
}

// uniform is a uniform variable of a shader: its type, and its lanes in
// the words of the uniforms.
type uniform struct {
	t       shaderir.Type
	word, n int
}

// The bridge's uniforms, which Draw sets: the images' sizes, the
// rectangle's size, the pixels a DIP and where an effect's box lies in
// its content, the first words.
var bridge = []string{shader.ImageSizesUniform, shader.RectSizeUniform, shader.ScaleUniform, shader.LayerInsetUniform,
	shader.BackdropRow0Uniform, shader.BackdropRow1Uniform, shader.BackdropAreaUniform}

// The words of the bridge's uniforms, which Compile checks.
const (
	scaleWord   = 2*shader.MaxImages + 2
	insetWord   = scaleWord + 1
	row0Word    = insetWord + 2
	row1Word    = row0Word + 3
	areaWord    = row1Word + 3
	bridgeWords = areaWord + 4
)

// bridgeWordsAt are where the bridge's uniforms start, in its order.
var bridgeWordsAt = []int{0, scaleWord - 2, scaleWord, insetWord, row0Word, row1Word, areaWord}

// readsBackdrop finds a shader reading its backdrop.
var readsBackdrop = regexp.MustCompile(`\bbackdrop(At|Sample)\b`)

// backdropImage reads a shader's backdrop, which its own images leave
// free.
var backdropImage = regexp.MustCompile(`\bimage` + fmt.Sprint(shader.BackdropImage) + `(At|Sample|Size)\b`)

// layerImage reads an effect's content, which its own images leave free.
var layerImage = regexp.MustCompile(`\bimage` + fmt.Sprint(shader.LayerImage) + `(At|Sample|Size)\b`)

// kindOf returns the kind of the shader of src by its entry point, of
// which it must have one.
func kindOf(src string) (kind, error) {
	f, err := parser.ParseFile(token.NewFileSet(), "", src, parser.SkipObjectResolution)
	if err != nil {
		return 0, nil // Compile reports it
	}
	var found []kind
	for _, d := range f.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok && fn.Recv == nil {
			for k, e := range entries {
				if fn.Name.Name == e.name {
					found = append(found, kind(k))
				}
			}
		}
	}
	switch {
	case len(found) == 0:
		return 0, nil // Compile reports the missing Fragment
	case len(found) > 1:
		return 0, fmt.Errorf("shader: %s and %s: a shader has one entry point", entries[found[0]].name, entries[found[1]].name)
	}
	k := found[0]
	if k != fragment && layerImage.MatchString(src) {
		return 0, fmt.Errorf("shader: an effect's shader reads image0 to image%d: image%d is its content (layerAt, layerSample)", shader.LayerImage-1, shader.LayerImage)
	}
	return k, nil
}

// Compile compiles a shader, src the source of a Go file as the package's
// documentation describes. It reports the errors of the source with their
// lines.
func Compile(src string) (*Shader, error) {
	k, err := kindOf(src)
	if err != nil {
		return nil, err
	}
	full := src
	if k != fragment {
		// An effect's entry point, called with the content.
		// An effect draws over all of its content, its positions from the
		// corner of the element's box within it.
		full += "\nfunc Fragment(dst vec4, pos vec2) vec4 {\n\tq := pos - " + shader.LayerInsetUniform + "/pixelScale()\n\t" + entries[k].wrap + "\n}\n"
	}
	p, err := shader.CompileFragment([]byte(full))
	if err != nil {
		return nil, err
	}
	for i, name := range bridge {
		if i >= len(p.UniformNames) || p.UniformNames[i] != name {
			return nil, fmt.Errorf("shader: the uniforms start with %v, not %v", p.UniformNames, bridge)
		}
	}
	backdrop := readsBackdrop.MatchString(src)
	if backdrop && backdropImage.MatchString(src) {
		return nil, fmt.Errorf("shader: a shader reading its backdrop reads image%d no more: it holds the backdrop (backdropAt, backdropSample)", shader.BackdropImage)
	}
	prog, err := interp.New(p)
	if err != nil {
		return nil, err
	}
	s := &Shader{prog: prog, uniforms: map[string]uniform{}, kind: k, backdrop: backdrop}
	for i, n := range interp.UniformWords(p) {
		if i < len(bridgeWordsAt) && s.words != bridgeWordsAt[i] {
			return nil, fmt.Errorf("shader: the bridge's %s at word %d, not %d", p.UniformNames[i], s.words, bridgeWordsAt[i])
		}
		s.uniforms[p.UniformNames[i]] = uniform{t: p.Uniforms[i], word: s.words, n: n}
		s.words += n
	}
	s.fx = &scene.Effect{Name: "shader", Backdrop: backdrop, Pixels: func() scene.EffectPixels { return &pixels{prog: prog, backdrop: backdrop} }}
	lanes, size := code(p, s.fx)
	l := &scene.ShaderLayout{Size: size}
	for _, at := range lanes {
		l.Lanes = append(l.Lanes, int32(at))
	}
	s.fx.Shader = l
	return s, nil
}

// MustCompile is Compile for a shader known to compile, as a package's
// variable: it panics when it does not.
func MustCompile(src string) *Shader {
	s, err := Compile(src)
	if err != nil {
		panic(err)
	}
	return s
}

// Options are what a shader draws with.
type Options struct {
	// Uniforms are the values of the shader's uniform variables, by name:
	// a number of any of Go's kinds (bool, an integer, a float) for one of
	// a scalar type, or a slice or an array of numbers, or a ui.Color (its
	// components from 0 to 1, not premultiplied), for a vector, a matrix
	// (by columns) or an array (elements in order); a variable missing is
	// zero, and a name the shader has not is left out. Integers set ints
	// and uints by their low 32 bits, as Go converts them.
	Uniforms map[string]any
	// Values, unless nil, are values of the shader's uniform variables set
	// by Uniform (Shader.NewValues), with no map and no conversions each
	// frame; Uniforms then sets the variables it names over them.
	Values *Values
	// Images are the images image0 to image3 read: bitmaps, or canvases'
	// (ui.Canvas.Bitmap); nil for none, which reads as transparent.
	Images [4]*ui.Bitmap
	// Radius rounds the corners of the rectangle, in DIPs.
	Radius float32
	// MaxSampleOffset is how far, at most, across and down in DIPs, an
	// effect (Effect) reads its content away from the pixel it draws, as
	// a distortion moves pixels: the content renders that much beyond the
	// element's box, as its shadows, the effect draws over all of it, and
	// it reads as transparent further away.
	MaxSampleOffset ui.Point
	// BackdropBlur, for a shader reading its backdrop (backdropAt,
	// backdropSample), is the standard deviation of the backdrop's blur,
	// in DIPs, 0 for none.
	BackdropBlur float32
}

// Draw fills r with the shader, in DIPs of p, as p's other drawings are:
// clipped, transformed and faded with the painter.
func (s *Shader) Draw(p *ui.Painter, r ui.Rect, o Options) {
	s.must(fragment)
	p.ShaderEffect(s.fx, r, o.Radius, s.input(p, r, r, o, ui.Point{}))
}

// Fill paints the text, paths and icons that fn paints with p with the
// shader in place of their colors, as a gradient fills them: each pixel
// the shader's color times the shape's coverage, the shader filling box,
// in DIPs, whose corner its positions start from and whose size rectSize
// returns. Emoji keep their colors, and the rest fn paints, as rectangles
// and images, paints as it would. Options.Radius does not apply.
func (s *Shader) Fill(p *ui.Painter, box ui.Rect, o Options, fn func()) {
	s.must(fragment)
	if s.backdrop {
		panic("shader: a shader reading its backdrop fills rectangles (Draw), not shapes (Fill)")
	}
	p.ShaderFill(s.fx, box, s.input(p, box, box, o, ui.Point{}), fn)
}

// FillPath fills path with the shader (non-zero winding), the shader
// filling the path's box (Path.Bounds).
func (s *Shader) FillPath(p *ui.Painter, path *ui.Path, o Options) {
	s.Fill(p, path.Bounds(), o, func() { p.FillPath(path, white) })
}

// StrokePath strokes the outline of path with the shader, width DIPs
// wide, with round joins and caps, the shader filling the path's box
// (Path.Bounds) grown by half the width.
func (s *Shader) StrokePath(p *ui.Painter, path *ui.Path, width float32, o Options) {
	b := path.Bounds()
	h := width / 2
	s.Fill(p, ui.Rect{X: b.X - h, Y: b.Y - h, W: b.W + width, H: b.H + width}, o, func() { p.StrokePath(path, width, white) })
}

// Text draws a line of text with its top-left corner at (x, y), size DIPs
// tall, filled with the shader, which fills the text's box.
func (s *Shader) Text(p *ui.Painter, x, y float32, text string, size float32, o Options) {
	w, h := p.MeasureText(0, ui.Span{Text: text, Size: size})
	s.Fill(p, ui.Rect{X: x, Y: y, W: w, H: h}, o, func() { p.Text(x, y, text, size, white) })
}

// Effect returns the shader drawing an element with o, for
// ui.Element.Effect: the shader of an effect, whose entry point reads the
// element's content, rendered with its children, in DIPs from the corner
// of its box, as SwiftUI's effects do:
//
//	// The content's colors, changed pixel by pixel (colorEffect).
//	func ColorEffect(pos vec2, color vec4) vec4
//
//	// Where to read the content for the pixel at pos (distortionEffect),
//	// transparent beyond it.
//	func DistortionEffect(pos vec2) vec2
//
//	// The pixel's color, from the content read anywhere with layerAt(p),
//	// its pixel at p, or layerSample(p), blended between pixels
//	// (layerEffect).
//	func LayerEffect(pos vec2) vec4
//
// rectSize returns the size of the element's box. An effect's shader
// reads images 0 to 2 of its own (image3 is the content), and moves
// pixels at most o.MaxSampleOffset away. It panics for a shader whose
// entry point is Fragment, which Draw and Fill draw.
func (s *Shader) Effect(o Options) Effect {
	if s.kind == fragment {
		panic("shader: Effect of a shader of Fragment: an effect's has ColorEffect, DistortionEffect or LayerEffect")
	}
	return Effect{s: s, o: o}
}

// Effect is a shader drawing an element in place of the element as it is
// (ui.Element.Effect), as Shader.Effect returns.
type Effect struct {
	s *Shader
	o Options
}

// EffectMargin returns how far the effect reads its content beyond the
// element's box, its MaxSampleOffset.
func (fx Effect) EffectMargin() ui.Point { return fx.o.MaxSampleOffset }

// PaintEffect paints the element from its content, the box's corner at
// inset in the content's pixels: the shader over all of the content, the
// box and the margin around it, reading the content as its last image.
func (fx Effect) PaintEffect(p *ui.Painter, box ui.Rect, content *ui.Bitmap, inset ui.Point) {
	if fx.s == nil || content == nil {
		return
	}
	o := fx.o
	o.Images[shader.LayerImage] = content
	s := p.Scale()
	w, h := content.Size()
	area := ui.Rect{X: box.X - inset.X/s, Y: box.Y - inset.Y/s, W: float32(w) / s, H: float32(h) / s}
	p.ShaderEffect(fx.s.fx, area, 0, fx.s.input(p, box, area, o, inset))
}

// must panics unless s is a shader of kind k.
func (s *Shader) must(k kind) {
	if s.kind != k {
		panic(fmt.Sprintf("shader: a shader of %s draws through Effect, not Draw or Fill", entries[s.kind].name))
	}
}

// wideColor is a color of a uniform in extended sRGB, outside sRGB.
type wideColor struct {
	u uniform
	c [4]float32
}

// input returns what the shader reads drawn with o in r, the box it
// fills, over area, the box's corner at inset in pixels from area's: the words of
// the uniforms' lanes, the bridge's first, in the frame's words, and those
// a wide gamut reads, when colors differ there.
func (s *Shader) input(p *ui.Painter, r, area ui.Rect, o Options, inset ui.Point) ui.ShaderInput {
	words := p.EffectWords(s.words)
	if v := o.Values; v != nil {
		if v.s != s {
			panic("shader: Options.Values of another shader")
		}
		copy(words[bridgeWords:], v.words[bridgeWords:])
	}
	for i, b := range o.Images {
		if b != nil {
			w, h := b.Size()
			words[2*i], words[2*i+1] = math.Float32bits(float32(w)), math.Float32bits(float32(h))
		}
	}
	words[scaleWord-2], words[scaleWord-1] = math.Float32bits(r.W), math.Float32bits(r.H)
	words[scaleWord] = math.Float32bits(p.Scale())
	words[insetWord], words[insetWord+1] = math.Float32bits(inset.X), math.Float32bits(inset.Y)
	var blur float32
	if s.backdrop {
		blur = max(o.BackdropBlur, 0) * p.Scale()
		bk := p.EffectBackdrop(area, blur, inset)
		i := shader.BackdropImage
		words[2*i], words[2*i+1] = math.Float32bits(float32(bk.W)), math.Float32bits(float32(bk.H))
		for k, x := range bk.ToFrame[0] {
			words[row0Word+k] = math.Float32bits(x)
		}
		for k, x := range bk.ToFrame[1] {
			words[row1Word+k] = math.Float32bits(x)
		}
		words[areaWord], words[areaWord+1], words[areaWord+2] = math.Float32bits(bk.Origin[0]), math.Float32bits(bk.Origin[1]), math.Float32bits(bk.Down)
	}
	var mapColors []colorValue
	for name, v := range o.Uniforms {
		u, ok := s.uniforms[name]
		if !ok || name[0] == '_' {
			continue
		}
		if c, ok := v.(ui.Color); ok && (u.t.Main == shaderir.Vec3 || u.t.Main == shaderir.Vec4) {
			mapColors = append(mapColors, colorValue{u, c})
			continue
		}
		kind := u.t.Scalar()
		if u.t.Main == shaderir.Array {
			kind = u.t.Sub[0].Scalar()
		}
		set(words[u.word:u.word+u.n], kind, name, v)
	}
	in := ui.ShaderInput{Uniforms: words, Images: o.Images, Blur: blur}
	// Colors, in sRGB, and outside it, in extended sRGB, for a wide gamut.
	var buf [8]wideColor
	wide := buf[:0]
	var set []colorValue
	if o.Values != nil {
		set = o.Values.colors
	}
	for _, list := range [2][]colorValue{set, mapColors} {
		for _, cv := range list {
			srgb, w := p.EffectColor(cv.c)
			writeColor(words[cv.u.word:cv.u.word+cv.u.n], srgb)
			if w != srgb {
				wide = append(wide, wideColor{cv.u, w})
			}
		}
	}
	if len(wide) > 0 {
		in.WideUniforms = p.EffectWords(s.words)
		copy(in.WideUniforms, words)
		for _, x := range wide {
			writeColor(in.WideUniforms[x.u.word:x.u.word+x.u.n], x.c)
		}
	}
	return in
}

// set writes value v of uniform name into its lanes, as numbers of kind:
// a number, a slice or an array of numbers of any of Go's kinds, or a
// ui.Color. Integers convert to int and uint keeping their low 32 bits, as
// Go converts them; floats drop their fraction.
func set(lanes []uint32, kind shaderir.BasicType, name string, v any) {
	if c, ok := v.(ui.Color); ok {
		for i, x := range [4]uint8{c.R, c.G, c.B, c.A} {
			if i < len(lanes) {
				lanes[i] = lane(kind, reflect.ValueOf(float64(x)/255))
			}
		}
		return
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Slice, reflect.Array:
		for i := 0; i < rv.Len() && i < len(lanes); i++ {
			if !number(rv.Index(i)) {
				panic(fmt.Sprintf("shader: the uniform %s is set to a %T, not numbers", name, v))
			}
			lanes[i] = lane(kind, rv.Index(i))
		}
	default:
		if !rv.IsValid() || !number(rv) {
			panic(fmt.Sprintf("shader: the uniform %s is set to a %T, not a number, numbers or a ui.Color", name, v))
		}
		if len(lanes) > 0 {
			lanes[0] = lane(kind, rv)
		}
	}
}

// number reports whether x is a bool, an integer or a float.
func number(x reflect.Value) bool {
	switch x.Kind() {
	case reflect.Bool, reflect.Float32, reflect.Float64,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return true
	}
	return false
}

// lane returns the number x as a lane of kind.
func lane(kind shaderir.BasicType, x reflect.Value) uint32 {
	var i int64
	var f float64
	switch x.Kind() {
	case reflect.Bool:
		if x.Bool() {
			i, f = 1, 1
		}
	case reflect.Float32, reflect.Float64:
		f = x.Float()
		i = int64(f)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		i, f = int64(x.Uint()), float64(x.Uint())
	default:
		i, f = x.Int(), float64(x.Int())
	}
	switch kind {
	case shaderir.Float:
		return math.Float32bits(float32(f))
	case shaderir.Bool:
		if f != 0 {
			return 1
		}
		return 0
	}
	return uint32(i)
}

// pixels draws a shader on the CPU, with the interpreter.
type pixels struct {
	prog  *interp.Program
	in    interp.Inputs
	rect  scene.Rect
	scale float32
	// backdrop tells that the shader reads its backdrop, as its image
	// shader.BackdropImage, the texels of bd.
	backdrop bool
	bd       sync.Once
}

func (x *pixels) Begin(op *scene.EffectOp, rect scene.Rect, radii [4]float32) {
	x.rect = rect
	x.in.Uniforms = op.Uniforms
	x.in.Images = x.in.Images[:0]
	for _, img := range op.Images {
		if img == nil || len(img.Pix) < 4*img.W*img.H {
			x.in.Images = append(x.in.Images, nil)
		} else {
			x.in.Images = append(x.in.Images, sceneImage{img})
		}
	}
	x.bd = sync.Once{}
	x.scale = 1
	if len(op.Uniforms) > scaleWord {
		x.scale = math.Float32frombits(op.Uniforms[scaleWord])
	}
}

func (x *pixels) Color(px, py float32, b *scene.BackdropImage) [4]float32 {
	if x.backdrop && b != nil {
		// The first color's backdrop, every color's, which cores ask for
		// at once.
		x.bd.Do(func() { x.in.Images[shader.BackdropImage] = backdropTexels{b} })
	}
	c, ok := x.prog.Color(&x.in, [4]float32{px, py, 0, 1}, (px-x.rect.X)/x.scale, (py-x.rect.Y)/x.scale)
	if !ok {
		return [4]float32{}
	}
	return c
}

// backdropTexels are the texels of an effect's backdrop as the
// interpreter reads them, as the GPU renderers' backdrop textures hold
// them.
type backdropTexels struct{ b *scene.BackdropImage }

func (t backdropTexels) Size() (int, int) { return t.b.W, t.b.H }

func (t backdropTexels) At(x, y int) [4]float32 { return [4]float32(t.b.Pix[4*(y*t.b.W+x):][:4]) }

// sceneImage is an image of a scene as the interpreter reads it.
type sceneImage struct{ img *scene.Image }

func (s sceneImage) Size() (int, int) { return s.img.W, s.img.H }

func (s sceneImage) At(x, y int) [4]float32 {
	p := s.img.Pix[4*(y*s.img.W+x):][:4]
	return [4]float32{float32(p[0]) / 255, float32(p[1]) / 255, float32(p[2]) / 255, float32(p[3]) / 255}
}

// white is the color the shapes a shader fills are painted with, whose
// coverage it multiplies.
var white = ui.Color{R: 255, G: 255, B: 255, A: 255}
