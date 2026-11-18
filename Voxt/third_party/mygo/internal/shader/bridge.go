package shader

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/egoist/mygo/internal/shaderir"
)

// MaxImages is how many images a shader reads, image0 to image3.
const MaxImages = 4

// Entry is the name of a shader's entry point, which returns the color
// of a pixel, premultiplied:
//
//	func Fragment(dst vec4, pos vec2) vec4
//
// dst is the pixel's center in the target, in its pixels from its top
// left corner, and pos the same point in DIPs from the top left corner of
// the rectangle the shader fills. It may take dst alone, or nothing.
const Entry = "Fragment"

// The uniform variables the bridge declares, which renderers set, before
// the shader's own: the images' sizes in pixels, the size of the
// rectangle the shader fills in DIPs, the pixels a DIP, and, for an
// effect's shader, where the corner of the element's box lies in its
// content (LayerImage), in its pixels.
const (
	ImageSizesUniform = "__imageSizes"
	RectSizeUniform   = "__rectSize"
	ScaleUniform      = "__scale"
	LayerInsetUniform = "__layerInset"
	// The backdrop of a shader reading it (BackdropImage): the rows of the
	// map from its positions to the frame's pixels, and where its texels
	// start in the frame, with how many pixels each covers across.
	BackdropRow0Uniform = "__backdropRow0"
	BackdropRow1Uniform = "__backdropRow1"
	BackdropAreaUniform = "__backdropArea"
)

// BackdropImage is the image a shader reads its backdrop from, through
// backdropAt and backdropSample: what the window drew behind it, which the
// renderers capture and blur as for effects reading their backdrop.
const BackdropImage = MaxImages - 2

// LayerImage is the image an effect's shader reads its element's content
// from, through layerAt and layerSample: the last, which its own images
// leave free.
const LayerImage = MaxImages - 1

// bridge is the source appended to a shader: the uniform variables that
// only renderers set, the functions that read them and the images, and
// the vertex entry point, which only gives the fragment entry point its
// arguments: renderers draw with vertex stages of their own.
var bridge = func() string {
	var b strings.Builder
	fmt.Fprintf(&b, `
%s

var %s [%d]vec2
var %s vec2
var %s float
var %s vec2
var %[8]s vec3
var %[9]s vec3
var %[10]s vec4

// rectSize returns the size of the rectangle the shader fills, in DIPs.
func rectSize() vec2 {
	return %[4]s
}

// pixelScale returns how many pixels of the target make a DIP.
func pixelScale() float {
	return %[5]s
}

// layerAt returns the pixel of the content of the element an effect
// draws at p, in DIPs from the corner of its box, premultiplied:
// transparent beyond what renders.
func layerAt(p vec2) vec4 {
	return image%[7]dAt(p*%[5]s + %[6]s)
}

// layerSample returns the content of the element an effect draws at p,
// in DIPs from the corner of its box, blended between the centers of the
// four pixels around p, premultiplied: transparent beyond what renders.
func layerSample(p vec2) vec4 {
	q := p*%[5]s + %[6]s
	in := step(vec2(0), q) - step(%[2]s[%[7]d], q)
	return image%[7]dSample(q) * in.x * in.y
}

// __backdropTexel returns where p, in DIPs from the corner of the box the
// shader fills, lies in the backdrop's texels.
func __backdropTexel(p vec2) vec2 {
	q := vec2(dot(%[8]s, vec3(p, 1)), dot(%[9]s, vec3(p, 1)))
	return (q - %[10]s.xy) / %[10]s.z
}

// backdropAt returns the backdrop's texel at p, in DIPs from the corner of
// the box the shader fills, premultiplied: what the window drew behind it,
// blurred as Options.BackdropBlur says.
func backdropAt(p vec2) vec4 {
	return image%[11]dAt(__backdropTexel(p))
}

// backdropSample returns the backdrop at p, in DIPs from the corner of the
// box the shader fills, blended between the texels around it.
func backdropSample(p vec2) vec4 {
	return image%[11]dSample(__backdropTexel(p))
}
`, InternalRegionBegin, ImageSizesUniform, MaxImages, RectSizeUniform, ScaleUniform, LayerInsetUniform, LayerImage,
		BackdropRow0Uniform, BackdropRow1Uniform, BackdropAreaUniform, BackdropImage)
	for i := range MaxImages {
		fmt.Fprintf(&b, `
// image%[1]dSize returns the size of image %[1]d, in its pixels.
func image%[1]dSize() vec2 {
	return %[2]s[%[1]d]
}

// image%[1]dAt returns the pixel of image %[1]d at p, in its pixels from its
// top left corner, premultiplied: transparent outside the image.
func image%[1]dAt(p vec2) vec4 {
	in := step(vec2(0), p) - step(%[2]s[%[1]d], p)
	return __texelAt(__t%[1]d, p) * in.x * in.y
}

// image%[1]dSample returns image %[1]d at p, in its pixels from its top left
// corner, blended between the centers of the four pixels around p, and
// clamped to its edges, as linear filtering samples a texture.
func image%[1]dSample(p vec2) vec4 {
	q := p - 0.5
	f := floor(q)
	w := q - f
	hi := %[2]s[%[1]d] - 1
	a := __texelAt(__t%[1]d, clamp(f, vec2(0), hi))
	b := __texelAt(__t%[1]d, clamp(f+vec2(1, 0), vec2(0), hi))
	c := __texelAt(__t%[1]d, clamp(f+vec2(0, 1), vec2(0), hi))
	d := __texelAt(__t%[1]d, clamp(f+vec2(1), vec2(0), hi))
	return mix(mix(a, b, w.x), mix(c, d, w.x), w.y)
}
`, i, ImageSizesUniform)
	}
	fmt.Fprintf(&b, `
func %s(dst vec2, pos vec2) (vec4, vec2) {
	return vec4(dst, 0, 1), pos
}

%s
`, vertexEntry, InternalRegionEnd)
	return b.String()
}()

const vertexEntry = "__vertex"

// CompileFragment compiles a MyGo shader: Go's syntax in package main,
// with uniform variables (exported, package level) that the app sets and
// the entry point Fragment, which may call functions of its own and read
// images through the functions image0At to image3At, image0Sample to
// image3Sample and image0Size to image3Size.
func CompileFragment(src []byte) (*shaderir.Program, error) {
	var b bytes.Buffer
	b.Write(src)
	b.WriteString(bridge)
	p, err := Compile(b.Bytes(), vertexEntry, Entry, MaxImages)
	if err != nil {
		return nil, err
	}
	if p.FragmentFunc.Block == nil {
		return nil, fmt.Errorf("shader: no entry point %s", Entry)
	}
	p.FragmentSource = bytes.Clone(src)
	return p, nil
}
