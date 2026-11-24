//go:build (linux && (amd64 || arm64)) || js

// Package gl draws scenes with OpenGL 3.3, OpenGL ES 3.0 or WebGL 2 into
// the framebuffer bound when Render is called: the one GTK's GtkGLArea
// binds while it emits its render signal, or a canvas's (package web). Every op of a scene is an instanced
// quad drawn by one shader (shader.glsl, compiled when the renderer
// starts), from the instances package gpu builds; clips are scissor
// rectangles, with the innermost rounded clip computed in the shader, as
// in packages d3d11 and metal. GL functions come from libepoxy on Linux,
// which GTK resolves its own through, called with purego (no cgo), and
// from the current WebGL 2 context in browsers (load_js.go), through
// syscall/js.
package gl

import (
	_ "embed"
	"fmt"
	"image"
	"os"
	"strings"
	"unsafe"
	"weak"

	"github.com/egoist/mygo/internal/gpu"
	"github.com/egoist/mygo/internal/scene"
)

//go:embed shader.glsl
var shaderSource string

// effectSource is the head and the tail of an effect's fragment shader,
// around the line "// effect" (see EffectSource).
//
//go:embed effect.glsl
var effectSource string

// EffectSource returns the source of an effect's fragment shader, after
// the version header: shader.glsl with EFFECT defined, the head of
// effect.glsl, the effect's shader (see scene.Effect), and the tail,
// which draws its instances with it.
func EffectSource(src string) string {
	head, tail, _ := strings.Cut(effectSource, "\n// effect\n")
	return "#define EFFECT\n" + shaderSource + "\n" + head + "\n" + src + "\n" + tail
}

// OpenGL enumerations.
const (
	glTriangleStrip      = 0x0005
	glBlend              = 0x0BE2
	glScissorTest        = 0x0C11
	glZero               = 0
	glOne                = 1
	glOneMinusSrcAlpha   = 0x0303
	glOneMinusSrc1Color  = 0x88FA
	glOneMinusSrc1Alpha  = 0x88FB
	glColorBufferBit     = 0x4000
	glArrayBuffer        = 0x8892
	glUniformBuffer      = 0x8A11
	glInvalidIndex       = 0xFFFFFFFF
	glStreamDraw         = 0x88E0
	glFloat              = 0x1406
	glRGBA32F            = 0x8814
	glUnsignedByte       = 0x1401
	glTexture2D          = 0x0DE1
	glTexture0           = 0x84C0
	glTextureMinFilter   = 0x2801
	glTextureMagFilter   = 0x2800
	glTextureWrapS       = 0x2802
	glTextureWrapT       = 0x2803
	glLinear             = 0x2601
	glClampToEdge        = 0x812F
	glR8                 = 0x8229
	glRed                = 0x1903
	glRGBA8              = 0x8058
	glRGBA               = 0x1908
	glUnpackAlignment    = 0x0CF5
	glUnpackRowLength    = 0x0CF2
	glVertexShader       = 0x8B31
	glFragmentShader     = 0x8B30
	glCompileStatus      = 0x8B81
	glLinkStatus         = 0x8B82
	glInfoLogLength      = 0x8B84
	glMaxTextureSize     = 0x0D33
	glVersion            = 0x1F02
	glRenderer           = 0x1F01
	glFramebuffer        = 0x8D40
	glColorAttachment0   = 0x8CE0
	glFramebufferDone    = 0x8CD5
	glFramebufferBinding = 0x8CA6
	glPackAlignment      = 0x0D05
	glReadFramebuffer    = 0x8CA8
	glDrawFramebuffer    = 0x8CA9
	glNearest            = 0x2600
	glNoError            = 0
)

type texture struct {
	tex       uint32
	w, h      int
	gen, ver  uint64
	lastFrame uint64
}

// canvasTexture is the texture a canvas (scene.NewCanvas) is drawn into,
// through its framebuffer, and what drawing it keeps, for as long as the
// canvas lives. Its rows go up, as GL's: images showing it flip it.
type canvasTexture struct {
	texture
	fb     uint32
	b      gpu.Builder
	canvas weak.Pointer[scene.Image]
}

// Renderer draws scenes into the framebuffer bound when Render is called,
// with the GL context that was current when New was called. All its
// methods must be called with that context current.
type Renderer struct {
	context uintptr // GDK's, to tell when the area's context changes
	es      bool
	// dual tells that the context blends with a second color from the
	// shader (OpenGL 3.3, or GLES with EXT_blend_func_extended).
	dual bool
	// software tells that the context draws on the CPU (Software).
	software bool
	maxSize  int
	program  uint32
	vao, buf uint32
	uSize    int32
	mask     texture
	color    texture
	empty    uint32 // a texture to bind where there is none
	images   map[uint64]*texture
	frame    uint64
	// canvases are the textures canvases are drawn into, and canvasTex
	// the names of those textures.
	canvases  map[uint64]*canvasTexture
	canvasTex map[uint32]bool

	// The passes computing the backdrops of effects: down and blur are
	// their programs, with their uniforms' locations, drawn with passVAO;
	// grab holds the area of the frame read, and backdrop the textures
	// they draw into, through backdropFB, the second only along rows.
	down, blur passProgram
	passVAO    uint32
	grab       texture
	backdrop   [2]texture
	backdropFB [2]uint32
	// header is what the shaders start with; effects are the programs of
	// the effects drawn, made as first drawn.
	header  string
	effects map[*scene.Effect]*effectProgram
	// ubo holds the uniforms of the shader being drawn (package ui/shader),
	// laid out in shaderBytes, which binding point 0 reads.
	ubo         uint32
	shaderBytes []byte
	// stops is the texture of the stops of gradients (uStops), stopsTex
	// the scene drawn's, 0 for none; blurred are the textures of the images
	// the frame draws blurred, the first used of them, and blurTmp the
	// textures their passes go through.
	stops    texture
	stopsTex uint32
	blurred  []target
	used     int
	blurTmp  [2]target

	b gpu.Builder
}

// effectProgram is the program of an effect and the location of its
// uSize, or why it has none.
type effectProgram struct {
	program uint32
	uSize   int32
	// flipRows is the location of a shader's uShaderFlipRows, which tells
	// it the canvases among its images, whose rows go up.
	flipRows int32
	err      error
}

// passProgram is the program of a pass computing backdrops and its
// uniforms' locations.
type passProgram struct {
	program                                           uint32
	origin, limit, shift, dir, down, rad, sigma, mode int32
}

// New returns a renderer for the current GL context.
func New() (*Renderer, error) {
	if err := load(); err != nil {
		return nil, err
	}
	r := &Renderer{images: map[uint64]*texture{}, context: currentContext()}
	if err := r.init(); err != nil {
		r.Release()
		return nil, err
	}
	return r, nil
}

// init creates the program, buffers and textures in the current context.
func (r *Renderer) init() error {
	version, es := contextVersion()
	r.es = es
	if r.es && version < 30 || !r.es && version < 33 {
		return fmt.Errorf("gl: OpenGL %d.%d is too old (%s)", version/10, version%10, versionString())
	}
	var size int32
	glGetIntegerv(glMaxTextureSize, &size)
	r.maxSize = int(size)
	r.dual = !r.es || hasExtension("GL_EXT_blend_func_extended")
	r.software = softwareRenderer(rendererString())
	header := "#version 330 core\n#define DUAL\n"
	if r.es {
		header = "#version 300 es\n"
		if r.dual {
			header += "#extension GL_EXT_blend_func_extended : require\n#define DUAL\n"
		}
		header += "precision highp float;\nprecision highp int;\n"
	}
	r.header = header
	vs, err := compile(glVertexShader, header+"#define VERTEX\n"+shaderSource)
	if err != nil {
		return err
	}
	defer glDeleteShader(vs)
	fs, err := compile(glFragmentShader, header+"#define FRAGMENT\n"+shaderSource)
	if err != nil {
		return err
	}
	defer glDeleteShader(fs)
	r.program = glCreateProgram()
	glAttachShader(r.program, vs)
	glAttachShader(r.program, fs)
	glLinkProgram(r.program)
	var ok int32
	if glGetProgramiv(r.program, glLinkStatus, &ok); ok == 0 {
		return fmt.Errorf("gl: cannot link the shader: %s", programLog(r.program))
	}
	glUseProgram(r.program)
	r.uSize = glGetUniformLocation(r.program, "uSize")
	glUniform1i(glGetUniformLocation(r.program, "uMask"), 0)
	glUniform1i(glGetUniformLocation(r.program, "uColor"), 1)
	glUniform1i(glGetUniformLocation(r.program, "uImage"), 2)
	glUniform1i(glGetUniformLocation(r.program, "uClipMask"), 4)
	glUniform1i(glGetUniformLocation(r.program, "uStops"), stopsUnit)
	glUseProgram(0)
	if r.down, err = passLink(header, "DOWN"); err != nil {
		return err
	}
	if r.blur, err = passLink(header, "BLUR"); err != nil {
		return err
	}
	glGenVertexArrays(1, &r.passVAO)

	// The instance buffer feeds fourteen float4 attributes per instance;
	// draws point them at their batch's instances.
	glGenVertexArrays(1, &r.vao)
	glGenBuffers(1, &r.buf)
	glBindVertexArray(r.vao)
	glBindBuffer(glArrayBuffer, r.buf)
	for i := range uint32(attributes) {
		glEnableVertexAttrib(i)
		glVertexAttribDivisor(i, 1)
	}
	glBindVertexArray(0)
	glBindBuffer(glArrayBuffer, 0)

	r.empty = newTexture(glRGBA8, glRGBA, 1, 1, []byte{0, 0, 0, 0}, 4)
	if e := glGetError(); e != glNoError {
		return fmt.Errorf("gl: setting up failed (error %#x)", e)
	}
	return nil
}

// effectProgram returns the program of e, compiled the first time from its
// GLSL, which drivers keep compiled; its program is 0 for an effect that
// does not compile, which draws nothing.
func (r *Renderer) effectProgram(e *scene.Effect) *effectProgram {
	if p := r.effects[e]; p != nil {
		return p
	}
	p := &effectProgram{}
	if r.effects == nil {
		r.effects = map[*scene.Effect]*effectProgram{}
	}
	r.effects[e] = p
	p.program, p.err = link(r.header+"#define VERTEX\n"+shaderSource, r.header+EffectSource(e.GLSL))
	if p.err != nil {
		p.err = fmt.Errorf("gl: the effect %s: %w", e.Name, p.err)
		fmt.Fprintln(os.Stderr, p.err)
		return p
	}
	glUseProgram(p.program)
	p.uSize = glGetUniformLocation(p.program, "uSize")
	glUniform1i(glGetUniformLocation(p.program, "uMask"), 0)
	glUniform1i(glGetUniformLocation(p.program, "uBackdrop"), 3)
	glUniform1i(glGetUniformLocation(p.program, "uClipMask"), 4)
	if e.Shader != nil {
		// A shader's uniforms come from binding point 0, its images from
		// units 5 to 8.
		if i := glGetUniformBlockIndex(p.program, "ShaderUniforms"); i != glInvalidIndex {
			glUniformBlockBinding(p.program, i, 0)
		}
		for i := range 4 {
			if l := glGetUniformLocation(p.program, fmt.Sprintf("T%d", i)); l >= 0 {
				glUniform1i(l, int32(5+i))
			}
		}
		p.flipRows = glGetUniformLocation(p.program, "uShaderFlipRows")
	}
	return p
}

// bindShader binds what the shader of batch b reads: its uniforms, laid
// out as its effect says, at binding point 0, and its images to units 5
// to 8. Unit 2 is active after.
func (r *Renderer) bindShader(b gpu.Batch) {
	r.shaderBytes = b.Effect.Shader.Pack(r.shaderBytes, b.Uniforms)
	if n := len(r.shaderBytes); n > 0 {
		if r.ubo == 0 {
			glGenBuffers(1, &r.ubo)
		}
		glBindBuffer(glUniformBuffer, r.ubo)
		glBufferData(glUniformBuffer, n, unsafe.Pointer(&r.shaderBytes[0]), glStreamDraw)
		glBindBufferBase(glUniformBuffer, 0, r.ubo)
	}
	// A canvas's rows go up, as GL's framebuffers draw them: the shader
	// flips the rows of those it reads.
	var flip [4]int32
	for i, t := range b.Textures {
		glActiveTexture(glTexture0 + 5 + uint32(i))
		if i == gpu.BackdropImage && b.Backdrop != 0 {
			// A shader reading its backdrop reads it as this image, its
			// rows going down.
			glBindTexture(glTexture2D, r.backdrop[0].tex)
			continue
		}
		glBindTexture(glTexture2D, or(uint32(t), r.empty))
		if img := b.Shader.Images[i]; t != 0 && img != nil && r.canvasTex[uint32(t)] {
			flip[i] = int32(img.H)
		}
	}
	glActiveTexture(glTexture0 + 2)
	if p := r.effects[b.Effect]; p != nil && p.flipRows >= 0 {
		glUniform4i(p.flipRows, flip[0], flip[1], flip[2], flip[3])
	}
}

// link compiles a vertex and a fragment shader and links them.
func link(vertex, fragment string) (uint32, error) {
	vs, err := compile(glVertexShader, vertex)
	if err != nil {
		return 0, err
	}
	defer glDeleteShader(vs)
	fs, err := compile(glFragmentShader, fragment)
	if err != nil {
		return 0, err
	}
	defer glDeleteShader(fs)
	program := glCreateProgram()
	glAttachShader(program, vs)
	glAttachShader(program, fs)
	glLinkProgram(program)
	var ok int32
	if glGetProgramiv(program, glLinkStatus, &ok); ok == 0 {
		err := fmt.Errorf("gl: cannot link the shader: %s", programLog(program))
		glDeleteProgram(program)
		return 0, err
	}
	return program, nil
}

// passLink compiles and links the program of a pass computing backdrops,
// the part of the shader defining name.
func passLink(header, name string) (passProgram, error) {
	vs, err := compile(glVertexShader, header+"#define PASS_VERTEX\n"+shaderSource)
	if err != nil {
		return passProgram{}, err
	}
	defer glDeleteShader(vs)
	fs, err := compile(glFragmentShader, header+"#define "+name+"\n"+shaderSource)
	if err != nil {
		return passProgram{}, err
	}
	defer glDeleteShader(fs)
	p := passProgram{program: glCreateProgram()}
	glAttachShader(p.program, vs)
	glAttachShader(p.program, fs)
	glLinkProgram(p.program)
	var ok int32
	if glGetProgramiv(p.program, glLinkStatus, &ok); ok == 0 {
		err := fmt.Errorf("gl: cannot link the %s pass: %s", name, programLog(p.program))
		glDeleteProgram(p.program)
		return passProgram{}, err
	}
	glUseProgram(p.program)
	glUniform1i(glGetUniformLocation(p.program, "uSrc"), 0)
	p.origin = glGetUniformLocation(p.program, "uOrigin")
	p.limit = glGetUniformLocation(p.program, "uLimit")
	p.shift = glGetUniformLocation(p.program, "uShift")
	p.dir = glGetUniformLocation(p.program, "uDir")
	p.down = glGetUniformLocation(p.program, "uDown")
	p.rad = glGetUniformLocation(p.program, "uRadius")
	p.sigma = glGetUniformLocation(p.program, "uSigma")
	p.mode = glGetUniformLocation(p.program, "uMode")
	glUseProgram(0)
	return p, nil
}

// attributes is the number of float4s in a gpu.Instance.
const attributes = gpu.InstanceSize / 16

func compile(kind uint32, src string) (uint32, error) {
	s := glCreateShader(kind)
	setShaderSource(s, src)
	glCompileShader(s)
	var ok int32
	if glGetShaderiv(s, glCompileStatus, &ok); ok == 0 {
		log := shaderLog(s)
		glDeleteShader(s)
		return 0, fmt.Errorf("gl: cannot compile the shader: %s", log)
	}
	return s, nil
}

func newTexture(internal int32, format uint32, w, h int, pix []byte, stride int) uint32 {
	var tex uint32
	glGenTextures(1, &tex)
	glBindTexture(glTexture2D, tex)
	glTexParameteri(glTexture2D, glTextureMinFilter, glLinear)
	glTexParameteri(glTexture2D, glTextureMagFilter, glLinear)
	glTexParameteri(glTexture2D, glTextureWrapS, glClampToEdge)
	glTexParameteri(glTexture2D, glTextureWrapT, glClampToEdge)
	var data unsafe.Pointer
	if len(pix) > 0 {
		data = unsafe.Pointer(&pix[0])
		unpack(stride, internal)
	}
	glTexImage2D(glTexture2D, 0, internal, int32(w), int32(h), 0, format, glUnsignedByte, data)
	unpackReset()
	return tex
}

// unpack sets the layout of the pixels uploads read: rows of stride bytes.
func unpack(stride int, internal int32) {
	bpp := 4
	if internal == glR8 {
		bpp = 1
	}
	glPixelStorei(glUnpackAlignment, 1)
	glPixelStorei(glUnpackRowLength, int32(stride/bpp))
}

func unpackReset() {
	glPixelStorei(glUnpackAlignment, 4)
	glPixelStorei(glUnpackRowLength, 0)
}

func (r *Renderer) syncAtlas(t *texture, a *scene.Atlas, internal int32, format uint32) error {
	if a == nil {
		return nil
	}
	if a.W > r.maxSize || a.H > r.maxSize {
		return fmt.Errorf("gl: a %d×%d atlas is larger than textures can be", a.W, a.H)
	}
	if t.tex == 0 || t.gen != a.Generation() || t.w != a.W || t.h != a.H {
		if t.tex != 0 {
			glDeleteTextures(1, &t.tex)
		}
		t.tex = newTexture(internal, format, a.W, a.H, a.Pix, a.W*a.BPP)
		t.w, t.h, t.gen, t.ver = a.W, a.H, a.Generation(), a.Version()
		return nil
	}
	rects, full := a.Changes(t.gen, t.ver)
	if full {
		rects = []image.Rectangle{image.Rect(0, 0, a.W, a.H)}
	}
	if len(rects) > 0 {
		glBindTexture(glTexture2D, t.tex)
		unpack(a.W*a.BPP, internal)
		for _, rc := range rects {
			src := a.Pix[(rc.Min.Y*a.W+rc.Min.X)*a.BPP:]
			glTexSubImage2D(glTexture2D, 0, int32(rc.Min.X), int32(rc.Min.Y), int32(rc.Dx()), int32(rc.Dy()), format, glUnsignedByte, unsafe.Pointer(&src[0]))
		}
		unpackReset()
	}
	t.ver = a.Version()
	return nil
}

func (r *Renderer) imageTexture(img *scene.Image) uintptr {
	if img.W > r.maxSize || img.H > r.maxSize {
		return 0
	}
	if img.Scene != nil {
		// A canvas, drawn before the scene showing it.
		if c := r.canvases[img.ID()]; c != nil {
			return uintptr(c.tex)
		}
		return 0
	}
	t := r.images[img.ID()]
	if t == nil {
		t = &texture{tex: newTexture(glRGBA8, glRGBA, img.W, img.H, img.Pix, img.W*4), w: img.W, h: img.H, ver: img.Version()}
		r.images[img.ID()] = t
	} else if t.w != img.W || t.h != img.H {
		glDeleteTextures(1, &t.tex)
		t.tex = newTexture(glRGBA8, glRGBA, img.W, img.H, img.Pix, img.W*4)
		t.w, t.h, t.ver = img.W, img.H, img.Version()
	} else if t.ver != img.Version() {
		glBindTexture(glTexture2D, t.tex)
		unpack(img.W*4, glRGBA8)
		glTexSubImage2D(glTexture2D, 0, 0, 0, int32(img.W), int32(img.H), glRGBA, glUnsignedByte, unsafe.Pointer(&img.Pix[0]))
		unpackReset()
		t.ver = img.Version()
	}
	t.lastFrame = r.frame
	return uintptr(t.tex)
}

// Render draws s into the bound framebuffer, which is s.Width×s.Height
// device pixels.
func (r *Renderer) Render(s *scene.Scene) error {
	if s.Width <= 0 || s.Height <= 0 {
		return nil
	}
	if c := currentContext(); c != r.context {
		// GTK realized the area again, with a new context: the old one
		// took the objects with it.
		r.forget()
		r.context = c
		if err := r.init(); err != nil {
			return err
		}
	}
	if err := r.draw(s); err != nil {
		return err
	}
	if e := glGetError(); e != glNoError {
		return fmt.Errorf("gl: drawing failed (error %#x)", e)
	}
	return nil
}

func (r *Renderer) draw(s *scene.Scene) error {
	r.frame++
	r.used = 0
	glActiveTexture(glTexture0)
	if err := r.syncAtlas(&r.mask, s.MaskAtlas, glR8, glRed); err != nil {
		return err
	}
	if err := r.syncAtlas(&r.color, s.ColorAtlas, glRGBA8, glRGBA); err != nil {
		return err
	}
	if err := r.drawCanvases(s); err != nil {
		return err
	}
	if err := r.drawScene(s, &r.b); err != nil {
		return err
	}
	if r.frame%120 == 0 {
		for key, t := range r.images {
			if r.frame-t.lastFrame > 240 {
				glDeleteTextures(1, &t.tex)
				delete(r.images, key)
			}
		}
		for key, t := range r.canvases {
			if t.canvas.Value() == nil {
				r.deleteCanvas(t)
				delete(r.canvases, key)
			}
		}
	}
	return nil
}

// drawCanvases draws the canvases s draws anew or shows into their
// textures, unless they hold their versions already, leaving the
// framebuffer bound as it was.
func (r *Renderer) drawCanvases(s *scene.Scene) (err error) {
	var outer int32
	glGetIntegerv(glFramebufferBinding, &outer)
	defer glBindFramebuffer(glFramebuffer, uint32(outer))
	s.EachCanvas(func(c *scene.Image) {
		if err == nil {
			err = r.drawCanvas(c)
		}
	})
	return err
}

func (r *Renderer) drawCanvas(c *scene.Image) error {
	if c.W > r.maxSize || c.H > r.maxSize {
		return nil
	}
	if r.canvases == nil {
		r.canvases, r.canvasTex = map[uint64]*canvasTexture{}, map[uint32]bool{}
	}
	t := r.canvases[c.ID()]
	if t == nil {
		t = &canvasTexture{canvas: weak.Make(c)}
		r.canvases[c.ID()] = t
	}
	if t.tex != 0 && t.w == c.W && t.h == c.H && t.ver == c.Version() {
		return nil
	}
	if t.tex == 0 || t.w != c.W || t.h != c.H {
		if t.tex != 0 {
			delete(r.canvasTex, t.tex)
			glDeleteTextures(1, &t.tex)
		}
		t.tex, t.w, t.h = newTexture(glRGBA8, glRGBA, c.W, c.H, nil, 0), c.W, c.H
		r.canvasTex[t.tex] = true
		if t.fb == 0 {
			glGenFramebuffers(1, &t.fb)
		}
		glBindFramebuffer(glFramebuffer, t.fb)
		glFramebufferTexture2D(glFramebuffer, glColorAttachment0, glTexture2D, t.tex, 0)
		if glCheckFramebufferState(glFramebuffer) != glFramebufferDone {
			return fmt.Errorf("gl: cannot draw into a %d×%d canvas", c.W, c.H)
		}
	}
	glBindFramebuffer(glFramebuffer, t.fb)
	cs := c.Scene
	cs.Width, cs.Height = c.W, c.H
	if err := r.drawScene(cs, &t.b); err != nil {
		return err
	}
	t.ver = c.Version()
	return nil
}

func (r *Renderer) deleteCanvas(t *canvasTexture) {
	delete(r.canvasTex, t.tex)
	if t.tex != 0 {
		glDeleteTextures(1, &t.tex)
	}
	if t.fb != 0 {
		glDeleteFramebuffers(1, &t.fb)
	}
	t.b.Release()
}

// drawScene draws s into the framebuffer bound, with b.
func (r *Renderer) drawScene(s *scene.Scene, b *gpu.Builder) error {
	b.BlurredImage = r.blurImage
	s = b.Build(s, r.imageTexture)
	b.BlurredImage = nil
	r.syncStops(b)
	// Canvases' rows go up, as GL's: images showing them sample them
	// flipped, which their source rectangles may be (scene.OpImage).
	if len(r.canvasTex) > 0 {
		for _, bt := range b.Batches {
			if bt.Image == 0 || !r.canvasTex[uint32(bt.Image)] {
				continue
			}
			for i := bt.Start; i < bt.Start+bt.Count; i++ {
				if in := &b.Instances[i]; in.Params[0] == 4 {
					in.UV[1], in.UV[3] = 1-in.UV[1], 1-in.UV[3]
				}
			}
		}
	}
	if err := r.fitBackdrop(b); err != nil {
		return err
	}
	// The framebuffer drawn into, GTK's, which the passes computing
	// backdrops leave for their own.
	var fb int32
	glGetIntegerv(glFramebufferBinding, &fb)

	glViewport(0, 0, int32(s.Width), int32(s.Height))
	glDisable(glScissorTest)
	c := s.Clear.Premul(1)
	glClearColor(c[0], c[1], c[2], c[3])
	glClear(glColorBufferBit)
	if len(b.Instances) == 0 {
		return nil
	}
	glBindBuffer(glArrayBuffer, r.buf)
	glBufferData(glArrayBuffer, len(b.Instances)*gpu.InstanceSize, unsafe.Pointer(&b.Instances[0]), glStreamDraw)
	r.bindState(s)
	bound, program, hole := uintptr(r.empty), r.program, false
	for _, bt := range b.Batches {
		sc := bt.Scissor
		sc.Left, sc.Top = max(sc.Left, 0), max(sc.Top, 0)
		sc.Right, sc.Bottom = min(sc.Right, int32(s.Width)), min(sc.Bottom, int32(s.Height))
		if bt.Count == 0 || sc.Empty() {
			continue
		}
		want := r.program
		if bt.Effect != nil {
			p := r.effectProgram(bt.Effect)
			if p.program == 0 {
				continue
			}
			want = p.program
		}
		if bt.Backdrop != 0 {
			// The effect shows what is drawn so far.
			r.readBackdrop(uint32(fb), s.Height, b.Backdrops[bt.Backdrop-1])
			glBindFramebuffer(glFramebuffer, uint32(fb))
			glViewport(0, 0, int32(s.Width), int32(s.Height))
			r.bindState(s)
			glActiveTexture(glTexture0 + 3)
			glBindTexture(glTexture2D, r.backdrop[0].tex)
			glActiveTexture(glTexture0 + 2)
			bound, program, hole = uintptr(r.empty), r.program, false
		}
		if bt.Hole != hole {
			hole = bt.Hole
			r.blendFunc(hole)
		}
		if want != program {
			program = want
			glUseProgram(program)
			if bt.Effect != nil {
				glUniform2f(r.effects[bt.Effect].uSize, float32(s.Width), float32(s.Height))
			}
		}
		if bt.Image != 0 && bt.Image != bound {
			bound = bt.Image
			glBindTexture(glTexture2D, uint32(bound)) // unit 2 is active
		}
		glActiveTexture(glTexture0 + 4)
		glBindTexture(glTexture2D, or(uint32(bt.ClipMask), r.empty))
		glActiveTexture(glTexture0 + 2)
		if bt.Shader != nil {
			r.bindShader(bt)
		}
		// Instances draw from the batch's first: without base instances in
		// OpenGL ES 3.0, the attributes start there.
		offset := uintptr(bt.Start * gpu.InstanceSize)
		for i := range uint32(attributes) {
			glVertexAttribPointer(i, 4, glFloat, false, int32(gpu.InstanceSize), offset+uintptr(i)*16)
		}
		// GL's scissor rectangles start at the bottom.
		glScissor(sc.Left, int32(s.Height)-sc.Bottom, sc.Right-sc.Left, sc.Bottom-sc.Top)
		glDrawArraysInstanced(glTriangleStrip, 0, 4, int32(bt.Count))
	}
	// Leave the state as GTK, which draws the area with the same context,
	// expects it.
	glDisable(glScissorTest)
	glDisable(glBlend)
	for unit := range stopsUnit + 1 {
		glActiveTexture(glTexture0 + uint32(unit))
		glBindTexture(glTexture2D, 0)
	}
	glActiveTexture(glTexture0)
	glBindBuffer(glArrayBuffer, 0)
	glBindBuffer(glUniformBuffer, 0)
	glBindVertexArray(0)
	glUseProgram(0)
	return nil
}

// stopsUnit is the texture unit of the stops of gradients (uStops).
const stopsUnit = 9

// syncStops uploads the stops of the gradients b built into r.stops,
// which r.stopsTex then is, or 0 for none.
func (r *Renderer) syncStops(b *gpu.Builder) {
	r.stopsTex = 0
	w, h := b.StopsSize()
	if w == 0 {
		return
	}
	data := b.Stops
	for len(data) < 4*w*h {
		data = append(data, 0)
	}
	b.Stops = data[:len(b.Stops)]
	glActiveTexture(glTexture0)
	if r.stops.tex == 0 || r.stops.w != w || r.stops.h < h {
		if r.stops.tex != 0 {
			glDeleteTextures(1, &r.stops.tex)
		}
		glGenTextures(1, &r.stops.tex)
		glBindTexture(glTexture2D, r.stops.tex)
		// Floats of 32 bits are not filterable: nearest, or the texture is
		// incomplete and reads as zero.
		glTexParameteri(glTexture2D, glTextureMinFilter, glNearest)
		glTexParameteri(glTexture2D, glTextureMagFilter, glNearest)
		glTexParameteri(glTexture2D, glTextureWrapS, glClampToEdge)
		glTexParameteri(glTexture2D, glTextureWrapT, glClampToEdge)
		glTexImage2D(glTexture2D, 0, glRGBA32F, int32(w), int32(h), 0, glRGBA, glFloat, unsafe.Pointer(&data[0]))
		r.stops.w, r.stops.h = w, h
	} else {
		glBindTexture(glTexture2D, r.stops.tex)
		glTexSubImage2D(glTexture2D, 0, 0, 0, int32(w), int32(h), glRGBA, glFloat, unsafe.Pointer(&data[0]))
	}
	glBindTexture(glTexture2D, 0)
	r.stopsTex = r.stops.tex
}

// target is a texture drawn into, with its framebuffer.
type target struct {
	tex, fb uint32
	w, h    int
}

// fit makes t a w×h texture of RGBA8 to draw into, unless it is.
func (t *target) fit(w, h int) error {
	if t.tex != 0 && t.w == w && t.h == h {
		return nil
	}
	if t.tex != 0 {
		glDeleteTextures(1, &t.tex)
	}
	t.tex = newTexture(glRGBA8, glRGBA, w, h, nil, 0)
	if t.fb == 0 {
		glGenFramebuffers(1, &t.fb)
	}
	glBindFramebuffer(glFramebuffer, t.fb)
	glFramebufferTexture2D(glFramebuffer, glColorAttachment0, glTexture2D, t.tex, 0)
	if glCheckFramebufferState(glFramebuffer) != glFramebufferDone {
		return fmt.Errorf("gl: cannot draw into a %d×%d texture", w, h)
	}
	t.w, t.h = w, h
	return nil
}

func (t *target) release() {
	if t.tex != 0 {
		glDeleteTextures(1, &t.tex)
	}
	if t.fb != 0 {
		glDeleteFramebuffers(1, &t.fb)
	}
	*t = target{}
}

// blurImage computes the area of img that bk says blurred
// (gpu.Builder.BlurredImage) into a texture of the frame's own, whose
// rows go down as an image's, as internal/raster computes it, and returns
// it, or 0. It leaves the framebuffer bound as it was.
func (r *Renderer) blurImage(img *scene.Image, bk scene.Backdrop) uintptr {
	src := uint32(r.imageTexture(img))
	w, h := bk.Size()
	if src == 0 || w <= 0 || h <= 0 || w > r.maxSize || h > r.maxSize {
		return 0
	}
	var fb int32
	glGetIntegerv(glFramebufferBinding, &fb)
	defer glBindFramebuffer(glFramebuffer, uint32(fb))
	for i := range r.blurTmp {
		if t := &r.blurTmp[i]; t.w < w || t.h < h {
			if t.fit(max(w, t.w), max(h, t.h)) != nil {
				return 0
			}
		}
	}
	if r.used == len(r.blurred) {
		r.blurred = append(r.blurred, target{})
	}
	out := &r.blurred[r.used]
	r.used++
	if out.fit(w, h) != nil {
		return 0
	}
	glDisable(glBlend)
	glEnable(glScissorTest)
	glBindVertexArray(r.passVAO)
	glActiveTexture(glTexture0)
	glBindTexture(glTexture2D, src)
	p := gpu.ImageDownPass(bk)
	if img.Scene != nil {
		// A canvas, whose rows go up: its row y is h-1-y.
		p.Shift = [2]int32{0, int32(img.H - 1)}
	} else {
		p.Mode |= gpu.PassUnflipped
	}
	r.down.set(p)
	if bk.Radius == 0 {
		r.passInto(out, w, h)
		glDisable(glScissorTest)
		return uintptr(out.tex)
	}
	r.passInto(&r.blurTmp[0], w, h)
	r.blur.set(gpu.ImageBlurPass(bk, [2]int32{1, 0}))
	glBindTexture(glTexture2D, r.blurTmp[0].tex)
	r.passInto(&r.blurTmp[1], w, h)
	r.blur.set(gpu.ImageBlurPass(bk, [2]int32{0, 1}))
	glBindTexture(glTexture2D, r.blurTmp[1].tex)
	r.passInto(out, w, h)
	glBindTexture(glTexture2D, 0)
	glDisable(glScissorTest)
	return uintptr(out.tex)
}

// passInto draws the program in use into the w×h texels at the start of
// t, reading the texture bound to unit 0.
func (r *Renderer) passInto(t *target, w, h int) {
	glBindFramebuffer(glFramebuffer, t.fb)
	glViewport(0, 0, int32(t.w), int32(t.h))
	glScissor(0, 0, int32(w), int32(h))
	glDrawArraysInstanced(glTriangleStrip, 0, 4, 1)
}

// bindState sets the state the instances of s draw with, leaving texture
// unit 2, the images', active.
func (r *Renderer) bindState(s *scene.Scene) {
	glUseProgram(r.program)
	glUniform2f(r.uSize, float32(s.Width), float32(s.Height))
	glBindVertexArray(r.vao)
	glBindBuffer(glArrayBuffer, r.buf)
	for unit, tex := range [5]uint32{or(r.mask.tex, r.empty), or(r.color.tex, r.empty), r.empty, r.empty, r.empty} {
		glActiveTexture(glTexture0 + uint32(unit))
		glBindTexture(glTexture2D, tex)
	}
	glActiveTexture(glTexture0 + stopsUnit)
	glBindTexture(glTexture2D, or(r.stopsTex, r.empty))
	glActiveTexture(glTexture0 + 2)
	glEnable(glBlend)
	r.blendFunc(false)
	glEnable(glScissorTest)
}

// blendFunc blends premultiplied colors over what is drawn, or for hole,
// takes their coverage away (scene.OpHole).
func (r *Renderer) blendFunc(hole bool) {
	src := uint32(glOne)
	if hole {
		src = glZero
	}
	if r.dual {
		glBlendFuncSeparate(src, glOneMinusSrc1Color, src, glOneMinusSrc1Alpha)
	} else {
		glBlendFunc(src, glOneMinusSrcAlpha)
	}
}

// readBackdrop computes the backdrop bk of what the framebuffer fb holds,
// a frame height pixels high, into r.backdrop[0]: it copies the area into
// grab, whose rows go up as GL's, averages its squares and blurs them.
func (r *Renderer) readBackdrop(fb uint32, height int, bk scene.Backdrop) {
	w, h := bk.Size()
	glDisable(glBlend)
	glBindFramebuffer(glFramebuffer, fb)
	glActiveTexture(glTexture0)
	glBindTexture(glTexture2D, r.grab.tex)
	glCopyTexSubImage2D(glTexture2D, 0, 0, 0, int32(bk.Area.Min.X), int32(height-bk.Area.Max.Y), int32(bk.Area.Dx()), int32(bk.Area.Dy()))
	glBindVertexArray(r.passVAO)
	// Rows go up in grab: the frame's row y is its row Max.Y-1-y.
	r.down.set(gpu.DownPass(bk, [2]int32{int32(bk.Area.Min.X), int32(bk.Area.Max.Y - 1)}))
	r.pass(0, w, h)
	if bk.Radius > 0 {
		r.blur.set(gpu.BlurPass(bk, [2]int32{1, 0}))
		glBindTexture(glTexture2D, r.backdrop[0].tex)
		r.pass(1, w, h)
		r.blur.set(gpu.BlurPass(bk, [2]int32{0, 1}))
		glBindTexture(glTexture2D, r.backdrop[1].tex)
		r.pass(0, w, h)
	}
}

// set uses the program with the uniforms of p.
func (pp *passProgram) set(p gpu.Pass) {
	glUseProgram(pp.program)
	glUniform2i(pp.origin, p.Origin[0], p.Origin[1])
	glUniform2i(pp.limit, p.Limit[0], p.Limit[1])
	glUniform2i(pp.shift, p.Shift[0], p.Shift[1])
	glUniform2i(pp.dir, p.Dir[0], p.Dir[1])
	glUniform1i(pp.down, p.Down)
	glUniform1i(pp.rad, p.Radius)
	glUniform1f(pp.sigma, p.Sigma)
	glUniform1i(pp.mode, p.Mode)
}

// pass draws the program in use into the w×h texels at the start of
// r.backdrop[i], reading the texture bound to unit 0.
func (r *Renderer) pass(i, w, h int) {
	glBindFramebuffer(glFramebuffer, r.backdropFB[i])
	glViewport(0, 0, int32(r.backdrop[i].w), int32(r.backdrop[i].h))
	glScissor(0, 0, int32(w), int32(h))
	glDrawArraysInstanced(glTriangleStrip, 0, 4, 1)
}

// fitBackdrop makes the textures of backdrops as large as the scene built
// needs, and the texture the area read is copied to.
func (r *Renderer) fitBackdrop(b *gpu.Builder) error {
	w, h := b.BackdropSize()
	aw, ah := 0, 0
	for _, bk := range b.Backdrops {
		aw, ah = max(aw, bk.Area.Dx()), max(ah, bk.Area.Dy())
	}
	if aw > r.grab.w || ah > r.grab.h {
		// Somewhat larger, so that a pane growing does not make it again
		// every frame.
		aw, ah = min(max(aw+aw/4, r.grab.w), r.maxSize), min(max(ah+ah/4, r.grab.h), r.maxSize)
		if r.grab.tex != 0 {
			glDeleteTextures(1, &r.grab.tex)
		}
		r.grab = texture{tex: newTexture(glRGBA8, glRGBA, aw, ah, nil, 0), w: aw, h: ah}
	}
	if w <= r.backdrop[0].w && h <= r.backdrop[0].h {
		return nil
	}
	w, h = min(max(w+w/4, r.backdrop[0].w), r.maxSize), min(max(h+h/4, r.backdrop[0].h), r.maxSize)
	for i := range r.backdrop {
		t := &r.backdrop[i]
		if t.tex != 0 {
			glDeleteTextures(1, &t.tex)
		}
		*t = texture{tex: newTexture(glRGBA8, glRGBA, w, h, nil, 0), w: w, h: h}
		if r.backdropFB[i] == 0 {
			glGenFramebuffers(1, &r.backdropFB[i])
		}
		var fb int32
		glGetIntegerv(glFramebufferBinding, &fb)
		glBindFramebuffer(glFramebuffer, r.backdropFB[i])
		glFramebufferTexture2D(glFramebuffer, glColorAttachment0, glTexture2D, t.tex, 0)
		done := glCheckFramebufferState(glFramebuffer) == glFramebufferDone
		glBindFramebuffer(glFramebuffer, uint32(fb))
		if !done {
			return fmt.Errorf("gl: cannot draw into a %d×%d backdrop texture", w, h)
		}
	}
	return nil
}

func or(a, b uint32) uint32 {
	if a != 0 {
		return a
	}
	return b
}

// forget drops the objects of a context that is gone.
func (r *Renderer) forget() {
	r.program, r.vao, r.buf, r.empty, r.ubo = 0, 0, 0, 0, 0
	r.down, r.blur, r.passVAO = passProgram{}, passProgram{}, 0
	clear(r.effects)
	r.grab, r.backdrop, r.backdropFB = texture{}, [2]texture{}, [2]uint32{}
	r.mask, r.color = texture{}, texture{}
	clear(r.images)
	for _, t := range r.canvases {
		t.b.Release()
	}
	clear(r.canvases)
	clear(r.canvasTex)
	r.stops, r.stopsTex, r.blurred, r.used, r.blurTmp = texture{}, 0, nil, 0, [2]target{}
}

// Software reports whether the context draws on the CPU, as Mesa's
// llvmpipe (where MYGO_GPU=1 has Linux draw with OpenGL anyway) and the
// SwiftShader of browsers without a GPU, for ui's Painter.Accelerated.
func (r *Renderer) Software() bool { return r.software }

// softwareRenderer reports whether a context's renderer string names a
// renderer drawing on the CPU (internal/linux's softwareGL, which picks
// whether surfaces draw with OpenGL, knows Mesa's).
func softwareRenderer(name string) bool {
	n := strings.ToLower(name)
	for _, s := range []string{"llvmpipe", "softpipe", "swrast", "software rasterizer", "swiftshader", "basic render driver"} {
		if strings.Contains(n, s) {
			return true
		}
	}
	return false
}

// Release deletes the renderer's objects when its context is current, and
// otherwise leaves them to the context, which deletes them with itself.
func (r *Renderer) Release() {
	r.b.Release()
	if errLoad != nil || currentContext() != r.context {
		r.forget()
		return
	}
	for _, t := range r.images {
		glDeleteTextures(1, &t.tex)
	}
	for _, t := range r.canvases {
		r.deleteCanvas(t)
	}
	for i := range r.blurred {
		r.blurred[i].release()
	}
	for i := range r.blurTmp {
		r.blurTmp[i].release()
	}
	r.blurred, r.used = nil, 0
	for _, tex := range []*uint32{&r.mask.tex, &r.color.tex, &r.empty, &r.grab.tex, &r.backdrop[0].tex, &r.backdrop[1].tex, &r.stops.tex} {
		if *tex != 0 {
			glDeleteTextures(1, tex)
		}
	}
	for i := range r.backdropFB {
		if r.backdropFB[i] != 0 {
			glDeleteFramebuffers(1, &r.backdropFB[i])
		}
	}
	for _, p := range r.effects {
		if p.program != 0 {
			glDeleteProgram(p.program)
		}
	}
	for _, p := range []uint32{r.down.program, r.blur.program} {
		if p != 0 {
			glDeleteProgram(p)
		}
	}
	if r.passVAO != 0 {
		glDeleteVertexArrays(1, &r.passVAO)
	}
	if r.ubo != 0 {
		glDeleteBuffers(1, &r.ubo)
	}
	if r.buf != 0 {
		glDeleteBuffers(1, &r.buf)
	}
	if r.vao != 0 {
		glDeleteVertexArrays(1, &r.vao)
	}
	if r.program != 0 {
		glDeleteProgram(r.program)
	}
	r.forget()
}

// FrameInfo describes the last frame for MYGO_FRAME_STATS.
func (r *Renderer) FrameInfo() string { return r.b.Summary() }
